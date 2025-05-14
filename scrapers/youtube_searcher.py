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
from utils.logger import Logger

logger = Logger.get_logger("youtube_searcher")


class YouTubeCommentAsyncFetcher:
    """비동기 YouTube 댓글 수집기

    YouTube API를 사용하여 비디오의 댓글과 자막을 수집하고,
    Ollama를 사용하여 댓글과 자막 간의 관련성을 분석합니다.

    Attributes:
        api_key (str): YouTube API 키
        max_comments (int): 수집할 최대 댓글 수
        model (str): Ollama 모델 이름
    """

    # URL 패턴 정의
    URL_PATTERN = re.compile(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    )

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
        self.api_key = api_key
        self.max_comments = max_comments

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

    def _is_valid_comment(self, comment: str) -> bool:
        """댓글이 유효한지 검사

        Args:
            comment (str): 검사할 댓글

        Returns:
            bool: URL이 포함되지 않은 유효한 댓글인 경우 True
        """
        # URL이 포함된 댓글 제외
        if self.URL_PATTERN.search(comment):
            return False
        return True

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
            "maxResults": self.max_comments * 2,  # URL 필터링을 위해 더 많은 댓글 수집
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

                    # URL이 포함되지 않은 댓글만 추가
                    if self._is_valid_comment(text):
                        comments.append((like_count, text))

                logger.info(f"✅ ID {video_id} 댓글 {len(comments)}개 수집 완료")
                comments.sort(key=lambda x: x[0], reverse=True)
                sorted_comments = [
                    comment[1] for comment in comments[: self.max_comments]
                ]
                return sorted_comments
        except Exception as e:
            logger.warning(f"⚠️ {video_id} 수집 실패: {e}")
            return []

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
            logger.error("DataFrame이 비어있거나 'url' 컬럼이 없습니다.")
            raise ValueError("DataFrame이 비어있거나 'url' 컬럼이 없습니다.")

        urls = df["url"].tolist()
        async with aiohttp.ClientSession() as session:
            comment_tasks = []
            for url in urls:
                video_id = self.extract_video_id(url)
                if video_id:
                    comment_tasks.append(self.fetch_video_comments(session, video_id))
                else:
                    comment_tasks.append(asyncio.sleep(0, result=[]))
            try:
                all_comments = await asyncio.gather(*comment_tasks)
            except Exception as e:
                logger.error(f"댓글/자막 수집 중 오류 발생: {e}")
                raise SearchRequestFailedError(e)
        results = []
        for url, comments in zip(urls, all_comments):
            for comment in comments:
                results.append(
                    {
                        "url": url,
                        "comment": comment,
                    }
                )
        return results
