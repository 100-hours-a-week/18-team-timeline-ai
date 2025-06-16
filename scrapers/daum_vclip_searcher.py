import re
from typing import Optional
from PyKakao import DaumSearch
from scrapers.base_searcher import BaseSearcher
from util.exceptions import (
    InvalidAPIKeyError,
    SearchRequestFailedError,
)
from util.logger import Logger

logger = Logger.get_logger("scrapers.daum_vclip")


class DaumVclipSearcher(BaseSearcher):
    """Daum Vclip 검색기"""

    def __init__(self, api_key: str, daum_service: Optional[DaumSearch] = None):
        """
        Args:
            service_key (str): Daum API Key
            daum_service (Optional[DaumSearch]): 주입할 DaumSearch 객체
                (테스트를 위해 필요 시 사용)
        Raises:
            InvalidAPIKeyError: 유효한 API Key가 아닐 경우 반환
        """
        self.service_key = api_key
        logger.info(
            f"[DaumVclipSearcher] 초기화 시작 - "
            f"API Key 설정: {api_key[:4]}...{api_key[-4:]}"
        )
        try:
            self.api = (
                daum_service
                if daum_service
                else DaumSearch(service_key=self.service_key)
            )
            logger.info("[DaumVclipSearcher] API 초기화 성공")
        except Exception as e:
            logger.error(
                f"[DaumVclipSearcher] API Key 초기화 실패 - " f"에러: {str(e)}"
            )
            raise InvalidAPIKeyError(f"API Key가 유효하지 않습니다: {e}")

    def search(self, query: str, page: int = 1) -> list[dict]:
        """비디오 클립 검색 후 목록을 반환

        Args:
            query (str): 검색 문자열
            page (int, optional): 검색하고 싶은 페이지. Defaults to 2.

        Raises:
            InvalidQueryError: 올바르지 않은 검색어일 때 반환
            ValueError: 검색어에 대한 영상을 찾을 수 없을 경우 반환

        Returns:
            list[dict]: {author, datetime, play_time, thumbnail, title, url}을
                포함하는 딕셔너리 리스트
        """
        if not query or query.strip() == "":
            logger.warning("[DaumVclipSearcher] 빈 검색어로 검색 시도")
            raise ValueError("검색어를 입력해야 합니다.")

        logger.info(
            f"[DaumVclipSearcher] 검색 시작 - " f"쿼리: '{query}', 페이지: {page}"
        )
        try:
            results = self.api.search_vclip(
                query=query, page=page, dataframe=False, data_type="json"
            )
            logger.debug(
                f"[DaumVclipSearcher] 검색 결과 수신 - "
                f"문서 수: {len(results.get('documents', []))}, "
                f"총 검색 결과 수: {results.get('meta', {}).get('total_count', 0)}"
            )
        except Exception as e:
            logger.error(
                f"[DaumVclipSearcher] 검색 요청 실패 - "
                f"쿼리: '{query}', 에러: {str(e)}"
            )
            raise SearchRequestFailedError(f"문제가 생겼습니다: {e}")

        if results is None:
            logger.error(f"[DaumVclipSearcher] 검색 결과 없음 - " f"쿼리: '{query}'")
            raise SearchRequestFailedError(
                f"'{query}'에 대한 검색 요청에 실패했습니다."
            )
        if not results["documents"]:
            logger.warning(f"[DaumVclipSearcher] 검색 결과 없음 - " f"쿼리: '{query}'")
            raise ValueError(f"'{query}'에 대한 검색 결과가 없습니다.")

        # YouTube URL만 필터링 (youtube.com 또는 youtu.be 도메인)
        youtube_pattern = r"(youtube\.com|youtu\.be)"
        filtered = [
            item["url"]
            for item in results["documents"]
            if "url" in item and re.search(youtube_pattern, item["url"], re.IGNORECASE)
        ]

        if not filtered:
            logger.warning(
                f"[DaumVclipSearcher] YouTube 검색 결과 없음 - " f"쿼리: '{query}'"
            )
            raise ValueError(f"'{query}'에 대한 YouTube 검색 결과가 없습니다.")

        logger.info(
            f"[DaumVclipSearcher] 검색 완료 - "
            f"쿼리: '{query}', "
            f"결과 수: {len(filtered)}개, "
            f"총 검색 결과: {len(results['documents'])}개"
        )
        return filtered
