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
    """비동기 YouTube 댓글 수집기

    YouTube API를 사용하여 비디오의 댓글과 자막을 수집하고,
    Ollama를 사용하여 댓글과 자막 간의 관련성을 분석합니다.

    Attributes:
        api_key (str): YouTube API 키
        max_comments (int): 수집할 최대 댓글 수
        model (str): Ollama 모델 이름
    """

    def __init__(
        self,
        api_key: str,
        model: str = "bge-m3:latest",
        max_comments: int = 2,
        timeout: int = 60,
    ):
        """YouTubeCommentAsyncFetcher 초기화

        Args:
            api_key (str): YouTube API Key
            model (str, optional): Ollama 모델 이름. Defaults to "bge-m3:latest".
            max_comments (int, optional): 최대 댓글 수. Defaults to 2.
            timeout (int, optional): Ollama 서버 시작 대기 시간(초). Defaults to 60.

        Raises:
            RuntimeError: Ollama 서버 시작 실패 시
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
        """YouTube URL에서 비디오 ID를 추출

        Args:
            url (str): YouTube 비디오 URL

        Returns:
            str: 비디오 ID 또는 None (URL이 유효하지 않은 경우)
        """
        match = re.search(r"v=([a-zA-Z0-9_-]+)", url)
        return match.group(1) if match else None

    def _get_embedding(self, text):
        """텍스트의 임베딩 벡터를 Ollama를 통해 생성

        Args:
            text (str): 임베딩할 텍스트

        Returns:
            List[float]: 임베딩 벡터 또는 None (실패 시)
        """
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
            return result["embedding"]
        except requests.exceptions.RequestException as e:
            logging.warning(f"Ollama 임베딩 요청 실패: {e}")
            return None

    def get_embeddings(self, texts):
        """여러 텍스트의 임베딩 벡터를 병렬로 생성

        Args:
            texts (List[str]): 임베딩할 텍스트 리스트

        Returns:
            List[List[float]]: 각 텍스트의 임베딩 벡터 리스트
                실패한 경우 0-벡터로 대체
        """
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

        fallback_dim = next((len(e) for e in embeddings if e is not None), 768)
        return [e if e is not None else [0.0] * fallback_dim for e in embeddings]

    @lru_cache(maxsize=128)
    def embed_captions(self, lines_tuple):
        """자막 라인의 임베딩 벡터를 캐시하여 반환

        Args:
            lines_tuple (tuple): 자막 라인 튜플

        Returns:
            List[List[float]]: 임베딩 벡터 리스트
        """
        return self.get_embeddings(list(lines_tuple))

    async def fetch_video_comments(
        self, session: aiohttp.ClientSession, video_id: str
    ) -> List[str]:
        """비디오의 댓글을 비동기적으로 수집

        Args:
            session (aiohttp.ClientSession): HTTP 세션
            video_id (str): YouTube 비디오 ID

        Returns:
            List[str]: 수집된 댓글 리스트 (좋아요 순으로 정렬)
        """
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
        """TF-IDF를 사용하여 자막에서 중요한 라인 추출

        Args:
            caption_lines (List[str]): 자막 라인 리스트
            top_k (int, optional): 추출할 라인 수. Defaults to 50.

        Returns:
            List[str]: 중요도가 높은 자막 라인 리스트
        """
        if not caption_lines:
            return []
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(caption_lines)
        scores = tfidf_matrix.sum(axis=1).A1
        top_indices = np.argsort(scores)[-top_k:][::-1]
        return [caption_lines[i] for i in sorted(top_indices)]

    @staticmethod
    async def get_youtube_captions(url: str) -> str:
        """YouTube 비디오의 자막을 가져옴

        한국어 자막을 우선적으로 가져오며, 없는 경우 영어, 일본어, 중국어, 자동 생성 자막을 시도합니다.

        Args:
            url (str): YouTube 비디오 URL

        Returns:
            str: 자막 텍스트 (없으면 빈 문자열)
        """
        video_id = YouTubeCommentAsyncFetcher.extract_video_id(url)
        if not video_id:
            return ""

        def _get_transcript_sync(video_id: str) -> str:
            try:
                transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)

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
        """댓글과 관련된 자막 라인을 임베딩 기반으로 찾음

        Args:
            comment (str): 댓글 텍스트
            caption_lines (List[str]): 자막 라인 리스트
            caption_vecs (List[List[float]]): 자막 라인의 임베딩 벡터 리스트
            top_k (int, optional): 반환할 관련 자막 수. Defaults to 5.

        Returns:
            str: 관련 자막 라인들을 개행으로 구분한 문자열
        """
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
        """DataFrame의 URL에서 댓글과 자막을 수집하고 분석

        Args:
            df (pd.DataFrame): 'url' 컬럼을 포함한 DataFrame

        Raises:
            ValueError: DataFrame이 비어있거나 'url' 컬럼이 없는 경우
            SearchRequestFailedError: 댓글/자막 수집 중 오류 발생

        Returns:
            List[dict]: 각 URL에 대한 분석 결과 리스트
                각 결과는 다음 키를 포함:
                - url: 비디오 URL
                - comment: 댓글
                - captions: 댓글과 관련된 자막
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
                    comment_tasks.append(asyncio.sleep(0, result=[]))
                    caption_tasks.append(asyncio.sleep(0, result=""))
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
