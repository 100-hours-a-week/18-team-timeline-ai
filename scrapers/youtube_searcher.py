import re, os
import pandas as pd
import dotenv
from typing import List
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from concurrent.futures import ThreadPoolExecutor, as_completed
from scrapers.base_searcher import BaseSearcher
from utils.exceptions import SearchRequestFailedError, InvalidQueryError
from scrapers.daum_vclip_searcher import DaumVclipSearcher
import asyncio
import aiohttp
import logging


class YouTubeCommentAsyncFetcher:
    """비동기 YouTube 댓글 수집기"""

    def __init__(self, api_key: str, max_comments: int = 100):
        """

        Args:
            api_key (str): YouTube API Key
            max_comments (int, optional):  최대 댓글 수. Defaults to 100.
        """
        self.api_key = api_key
        self.max_comments = max_comments

    @staticmethod
    def extract_video_id(url: str) -> str:
        match = re.search(r"v=([a-zA-Z0-9_-]+)", url)
        return match.group(1) if match else None

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

    async def search(self, df: pd.DataFrame) -> List[str]:
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
            List[str]: 수집된 댓글 리스트
        """
        if df.empty or "url" not in df.columns:
            logging.error("DataFrame이 비어있거나 'url' 컬럼이 없습니다.")
            raise ValueError("DataFrame이 비어있거나 'url' 컬럼이 없습니다.")

        video_ids = [
            self.extract_video_id(url)
            for url in df["url"]
            if self.extract_video_id(url)
        ]
        try:
            async with aiohttp.ClientSession() as session:
                tasks = [self.fetch_video_comments(session, vid) for vid in video_ids]
                all_comments = await asyncio.gather(*tasks)
        except Exception as e:
            logging.error(f"댓글 수집 중 오류 발생: {e}")
            raise SearchRequestFailedError(f"댓글 수집 중 오류 발생: {e}")

        # flatten
        return [comment for comments in all_comments for comment in comments]
