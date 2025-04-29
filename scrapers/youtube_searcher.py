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


class YouTubeCommentAsyncFetcher:
    """비동기 YouTube 댓글 수집기"""

    def __init__(self, api_key: str, max_comments: int = 5):
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
        }
        try:
            async with session.get(url, params=params) as response:
                data = await response.json()
                comments = []
                for item in data.get("items", []):
                    comment = item["snippet"]["topLevelComment"]["snippet"]
                    text = comment.get("textDisplay", "")
                    comments.append(text)
                print(f"✅ ID {video_id} 댓글 {len(comments)}개 수집 완료")
                return comments
        except Exception as e:
            print(f"⚠️ {video_id} 수집 실패: {e}")
            return []

    async def search(self, df: pd.DataFrame) -> List[str]:
        if df.empty or "url" not in df.columns:
            raise ValueError("DataFrame이 비어있거나 'url' 컬럼이 없습니다.")

        video_ids = [
            self.extract_video_id(url)
            for url in df["url"]
            if self.extract_video_id(url)
        ]

        async with aiohttp.ClientSession() as session:
            tasks = [self.fetch_video_comments(session, vid) for vid in video_ids]
            all_comments = await asyncio.gather(*tasks)

        # flatten
        return [comment for comments in all_comments for comment in comments]
