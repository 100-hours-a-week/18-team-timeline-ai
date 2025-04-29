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

class YouTubeCommentSearcher(BaseSearcher):
    """YouTube 댓글 수집기 (병렬 지원)"""

    def __init__(self, api_key: str, max_comments: int = 5, max_workers: int = 6):
        self.api_key = api_key
        self.max_comments = max_comments
        self.max_workers = max_workers

    @staticmethod
    def extract_video_id(url: str) -> str:
        """유튜브 URL에서 영상 ID 추출"""
        match = re.search(r"v=([a-zA-Z0-9_-]+)", url)
        return match.group(1) if match else None

    def _fetch_comments(self, video_id: str) -> List[str]:
        """개별 video_id에 대해 댓글 가져오기"""
        youtube = build("youtube", "v3", developerKey=self.api_key)
        comments = []
        try:
            response = (
                youtube.commentThreads()
                .list(
                    part="snippet",
                    videoId=video_id,
                    textFormat="plainText",
                    maxResults=self.max_comments,
                )
                .execute()
            )

            for item in response["items"]:
                comment = item["snippet"]["topLevelComment"]["snippet"]
                author = comment.get("authorDisplayName", "알 수 없음")
                text = comment.get("textDisplay", "")
                comments.append(text)

        except HttpError as e:
            raise SearchRequestFailedError(f"ID: {video_id} 요청 실패: {e}")

        return comments

    def search(self, df: pd.DataFrame) -> List[str]:
        """DataFrame 기반 댓글 병렬 수집"""

        if df.empty or "url" not in df.columns:
            raise InvalidQueryError("DataFrame이 비어있거나 'url' 컬럼이 없습니다.")

        ripple = []
        video_ids = []

        # 유효한 유튜브 ID만 추출
        for idx, series in df.iterrows():
            video_url = series["url"]
            video_id = self.extract_video_id(video_url)
            if video_id:
                video_ids.append(video_id)

        errors = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._fetch_comments, vid): vid for vid in video_ids
            }

            for future in as_completed(futures):
                video_id = futures[future]
                try:
                    comments = future.result()
                    ripple.extend(comments)
                    print(f"✅ ID {video_id} 댓글 {len(comments)}개 수집 완료")
                except SearchRequestFailedError as e:
                    print(f"⚠️ 수집 실패 - {e}")
                    errors.append((video_id, str(e)))

        if errors:
            print("\n🚨 오류 발생 영상 리스트:")
            for video_id, error in errors:
                print(f"- {video_id}: {error}")

        return ripple
