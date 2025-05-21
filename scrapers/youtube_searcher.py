import re
from typing import List
from scrapers.base_searcher import BaseSearcher
import asyncio
import aiohttp
from utils.logger import Logger
import logging
from utils.error_utils import SearchRequestFailedError

logger = Logger.get_logger("youtube_searcher", log_level=logging.ERROR)


class YouTubeCommentAsyncFetcher(BaseSearcher):
    """YouTube 댓글 수집기

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

    # URL 패턴 정의
    URL_PATTERN = re.compile(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    )

    def __init__(
        self,
        api_key: str,
        max_comments: int = 10,
    ):
        """YouTubeCommentAsyncFetcher 초기화

        Args:
            api_key (str): YouTube API Key
            max_comments (int, optional): 최대 댓글 수. Defaults to 10.
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
        try:
            match = re.search(r"v=([a-zA-Z0-9_-]+)", url)
            return match.group(1) if match else None
        except Exception as e:
            logger.error(
                f"[YouTubeCommentAsyncFetcher] 비디오 ID 추출 중 오류 발생: {e}"
            )
            return None

    def _is_valid_comment(self, comment: str) -> bool:
        """댓글이 유효한지 검사

        Args:
            comment (str): 검사할 댓글

        Returns:
            bool: URL이 포함되지 않은 유효한 댓글인 경우 True
        """
        # URL이 포함된 댓글 제외
        try:
            if self.URL_PATTERN.search(comment):
                return False
            return True
        except Exception as e:
            logger.error(
                f"[YouTubeCommentAsyncFetcher] 댓글 유효성 검사 중 오류 발생: {e}"
            )
            return False

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

                logger.info(
                    f"[YouTubeCommentAsyncFetcher] ID {video_id} 댓글 {len(comments)}개 수집 완료"
                )
                comments.sort(key=lambda x: x[0], reverse=True)
                sorted_comments = [
                    comment[1] for comment in comments[: self.max_comments]
                ]
                return sorted_comments
        except Exception as e:
            logger.error(f"[YouTubeCommentAsyncFetcher] ID {video_id} 수집 실패: {e}")
            return []

    async def search(self, df: List[str]) -> List[dict]:
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
        if not df:
            logger.error(
                "[YouTubeCommentAsyncFetcher] DataFrame이 비어있거나 'url' 컬럼이 없습니다."
            )
            raise ValueError("DataFrame이 비어있거나 'url' 컬럼이 없습니다.")

        urls = df
        async with aiohttp.ClientSession() as session:
            comment_tasks = []
            for url in urls:
                video_id = self.extract_video_id(url)
                if video_id:
                    logger.info(
                        f"[YouTubeCommentAsyncFetcher] ID {video_id} 댓글 수집 시작"
                    )
                    comment_tasks.append(self.fetch_video_comments(session, video_id))
                else:
                    logger.warning(
                        f"[YouTubeCommentAsyncFetcher] ID {video_id} 비디오 ID 추출 실패"
                    )
                    comment_tasks.append(asyncio.sleep(0, result=[]))
            try:
                logger.info(
                    f"[YouTubeCommentAsyncFetcher] 모든 비디오에 대한 댓글 수집 시작"
                )
                all_comments = await asyncio.gather(*comment_tasks)
            except Exception as e:
                logger.error(
                    f"[YouTubeCommentAsyncFetcher] 댓글/자막 수집 중 오류 발생: {e}"
                )
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
        logger.info(f"[YouTubeCommentAsyncFetcher] 모든 비디오에 대한 댓글 수집 완료")
        return results
