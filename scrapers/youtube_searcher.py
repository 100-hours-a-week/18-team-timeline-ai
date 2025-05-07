import re, os
import pandas as pd
import dotenv
from typing import List
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from concurrent.futures import ThreadPoolExecutor, as_completed
from scrapers.base_searcher import BaseSearcher
from utils.exceptions import SearchRequestFailedError, InvalidQueryError
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api import TranscriptsDisabled, NoTranscriptFound
import asyncio
import aiohttp
import logging
import subprocess
import requests
import time
import numpy as np
import requests
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from sklearn.feature_extraction.text import TfidfVectorizer


class YouTubeCommentAsyncFetcher:
    """비동기 YouTube 댓글 수집기"""

    def __init__(
        self,
        api_key: str,
        model: str = "bge-m3:latest",
        max_comments: int = 2,
        timeout: int = 60,
    ):
        """

        Args:
            api_key (str): YouTube API Key
            max_comments (int, optional):  최대 댓글 수. Defaults to 100.
        """

        def _is_ollama_alive():
            try:
                response = requests.get("http://localhost:11434")
                return response.status_code == 200
            except requests.exceptions.RequestException:
                return False

        self.api_key = api_key
        self.max_comments = max_comments
        self.model = model
        if _is_ollama_alive():
            logging.info("Ollama server 실행 중!")
            return
        logging.warning("Ollama server is close. Do cmd Ollama serve....")
        proc = subprocess.Popen(
            ["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        for _ in range(timeout):
            if _is_ollama_alive():
                logging.info("Ollama server 실행 완료")
                return
            time.sleep(5)
        raise RuntimeError("Ollama 서버를 자동 실행시켰지만, 연결에 실패했어요.")

    @staticmethod
    def extract_video_id(url: str) -> str:
        match = re.search(r"v=([a-zA-Z0-9_-]+)", url)
        return match.group(1) if match else None

    def _get_embedding(self, text):
        try:
            response = requests.post(
                "http://localhost:11434/api/embeddings",
                json={"model": self.model, "prompt": text},
                timeout=30,
            )
            result = response.json()
            if "embedding" not in result:
                logging.error(f"응답 이상: {result.keys()}")
                return None
            # logging.info("반환 성공!")
            return result["embedding"]
        except requests.exceptions.RequestException as e:
            logging.warning(f"Ollama 임베딩 요청 실패: {e}")
            return None

    def get_embeddings(self, texts):
        embeddings = [None] * len(texts)

        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_idx = {
                executor.submit(self._get_embedding, text): idx
                for idx, text in enumerate(texts)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    embedding = future.result()
                    embeddings[idx] = embedding
                except Exception as e:
                    logging.warning(f"Ollama 임베딩 처리 실패: {e}")
                    embeddings[idx] = None

        # fallback: None을 0-vector로 대체
        fallback_dim = next((len(e) for e in embeddings if e is not None), 768)
        return [e if e is not None else [0.0] * fallback_dim for e in embeddings]

    @lru_cache(maxsize=128)
    def embed_captions(self, lines_tuple):
        return self.get_embeddings(list(lines_tuple))

    async def fetch_video_comments(
        self, session: aiohttp.ClientSession, video_id: str
    ) -> List[str]:
        url = "https://www.googleapis.com/youtube/v3/commentThreads"
        params = {
            "part": "snippet",
            "videoId": video_id,
            "key": self.api_key,
            "textFormat": "plainText",
            "maxResults": self.max_comments,
            "order": "relevance",
        }
        try:
            async with session.get(url, params=params) as response:
                data = await response.json()
                comments = []
                for item in data.get("items", []):
                    comment = item["snippet"]["topLevelComment"]["snippet"]
                    text = comment.get("textDisplay", "")
                    like_count = comment.get("likeCount", 0)
                    comments.append((like_count, text))
                logging.info(f"✅ ID {video_id} 댓글 {len(comments)}개 수집 완료")
                comments.sort(key=lambda x: x[0], reverse=True)
                sorted_comments = [
                    comment[1] for comment in comments[: self.max_comments]
                ]
                return sorted_comments
        except Exception as e:
            logging.warning(f"⚠️ {video_id} 수집 실패: {e}")
            return []

    @staticmethod
    def extract_top_caption_lines_by_keywords(
        caption_lines: List[str], top_k: int = 50
    ) -> List[str]:
        if not caption_lines:
            return []
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(caption_lines)
        scores = tfidf_matrix.sum(axis=1).A1
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [caption_lines[i] for i in sorted(top_indices)]

    @staticmethod
    async def get_youtube_captions(url: str) -> str:
        """
        주어진 YouTube URL에서 자막을 가져옵니다. 가능하면 한국어 자막을 우선합니다.

        Args:
            url (str): YouTube 동영상 URL

        Returns:
            str: 자막 텍스트 전체 (없으면 빈 문자열)
        """
        video_id = YouTubeCommentAsyncFetcher.extract_video_id(url)
        if not video_id:
            return ""

        def _get_transcript_sync(video_id: str) -> str:
            try:
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

                # Try Korean first
                if "ko" in [t.language_code for t in transcript_list]:
                    transcript = transcript_list.find_transcript(["ko"])
                else:
                    transcript = transcript_list.find_transcript(
                        ["en", "ja", "zh", "auto"]
                    )
                texts = "\n".join([entry.text for entry in transcript.fetch()])
                return texts
            except (TranscriptsDisabled, NoTranscriptFound) as e:
                logging.warning(f"자막 없음: {e}")
                return ""
            except Exception as e:
                logging.error(f"자막 수집 오류: {e}")
                return ""

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _get_transcript_sync, video_id)

    def get_relevant_captions_via_ollama_embedding(
        self,
        comment: str,
        caption_lines: List[str],
        caption_vecs: List[List[float]],
        top_k: int = 5,
    ) -> str:
        if not comment or not caption_lines or not caption_vecs:
            return ""

        comment_vec = self.get_embeddings([comment])[0]
        caption_vecs = np.array(caption_vecs)
        comment_vec = np.array(comment_vec)

        sims = (
            caption_vecs
            @ comment_vec
            / (
                np.linalg.norm(caption_vecs, axis=1) * np.linalg.norm(comment_vec)
                + 1e-8
            )
        )
        top_indices = np.argsort(-sims)[:top_k]
        selected_lines = [caption_lines[i] for i in sorted(top_indices)]
        return "\n".join(selected_lines)

    async def search(self, df: pd.DataFrame) -> List[dict]:
        """
        DataFrame에서 URL을 추출하고 댓글을 수집합니다.
        DataFrame은 'url' 컬럼을 포함해야 합니다.
        DataFrame이 비어있거나 'url' 컬럼이 없으면 ValueError를 발생시킵니다.
        댓글은 비동기적으로 수집되며, 최대 self.max_comments 개수로 제한됩니다.
        댓글 수집 중 오류가 발생하면 해당 비디오 ID에 대한 댓글은 빈 리스트로 반환됩니다.
        댓글 수집이 완료되면 모든 댓글을 리스트로 반환합니다.
        Args:
            df (pd.DataFrame): 댓글을 수집할 비디오 URL이 포함된 DataFrame

        Raises:
            ValueError: DataFrame이 비어있거나 'url' 컬럼이 없을 경우 발생
            SearchRequestFailedError: 댓글 수집 중 오류 발생

        Returns:
            List[dict]: {"url": ..., "comments": [...], "captions": "..."}
        """
        if df.empty or "url" not in df.columns:
            logging.error("DataFrame이 비어있거나 'url' 컬럼이 없습니다.")
            raise ValueError("DataFrame이 비어있거나 'url' 컬럼이 없습니다.")

        urls = df["url"].tolist()
        async with aiohttp.ClientSession() as session:
            comment_tasks = []
            caption_tasks = []
            for url in urls:
                video_id = self.extract_video_id(url)
                if video_id:
                    comment_tasks.append(self.fetch_video_comments(session, video_id))
                    caption_tasks.append(self.get_youtube_captions(url))
                else:
                    comment_tasks.append(asyncio.sleep(0, result=[]))  # 빈 댓글
                    caption_tasks.append(asyncio.sleep(0, result=""))  # 빈 자막
            try:
                all_comments = await asyncio.gather(*comment_tasks)
                all_captions = await asyncio.gather(*caption_tasks)
            except Exception as e:
                logging.error(f"댓글/자막 수집 중 오류 발생{e}")
                raise SearchRequestFailedError(e)
        results = []
        for url, comments, captions in zip(urls, all_comments, all_captions):
            raw_lines = [line.strip() for line in captions.splitlines() if line.strip()]
            caption_lines = self.extract_top_caption_lines_by_keywords(raw_lines)
            caption_embeddings = (
                self.get_embeddings(caption_lines) if caption_lines else []
            )
            for comment in comments:

                relevant_captions = self.get_relevant_captions_via_ollama_embedding(
                    comment, caption_lines, caption_embeddings, 10
                )

                results.append(
                    {
                        "url": url,
                        "comment": comment,
                        "captions": relevant_captions,
                    }
                )
        return results
