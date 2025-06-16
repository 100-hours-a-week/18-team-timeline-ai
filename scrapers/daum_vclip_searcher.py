import re
from typing import Optional
from PyKakao import DaumSearch
from scrapers.base_searcher import BaseSearcher
from utils.exceptions import (
    InvalidAPIKeyError,
    SearchRequestFailedError,
)
from utils.logger import Logger

logger = Logger.get_logger("scrapers.daum_vclip")


class DaumVclipSearcher(BaseSearcher):
    """Daum Vclip 검색기"""

    def __init__(self, api_key: str, daum_service: Optional[DaumSearch] = None):
        """
        Args:
            service_key (str): Daum API Key
            daum_service (Optional[DaumSearch]): 주입할 DaumSearch 객체 (테스트를 위해 필요 시 사용)
        Raises:
            InvalidAPIKeyError: 유효한 API Key가 아닐 경우 반환
        """
        self.service_key = api_key
        logger.info(f"[DaumVclipSearcher] API Key 설정")
        try:
            self.api = (
                daum_service
                if daum_service
                else DaumSearch(service_key=self.service_key)
            )
        except Exception as e:
            logger.error(f"[DaumVclipSearcher] API Key 초기화 실패: {e}")
            raise InvalidAPIKeyError(f"API Key가 유효하지 않습니다.{e}")

    def search(self, query: str, page: int = 1) -> list[dict]:
        """비디오 클립 검색 후 목록을 반환

        Args:
            query (str): 검색 문자열
            page (int, optional): 검색하고 싶은 페이지. Defaults to 2.

        Raises:
            InvalidQueryError: 올바르지 않은 검색어일 때 반환
            ValueError: 검색어에 대한 영상을 찾을 수 없을 경우 반환
            ValueError:
        Returns:
            pd.DataFrame: {author, datetime, play_time, thumbnail, title, url}을 column으로 가지는 DataFrame 객체 반환
        """
        if not query or query.strip() == "":
            logger.warning("[DaumVclipSearcher] 빈 검색어로 검색 시도")
            raise ValueError("검색어를 입력해야 합니다.")

        logger.info(f"[DaumVclipSearcher] 검색 시작: query='{query}', page={page}")
        try:
            results = self.api.search_vclip(
                query=query, page=page, dataframe=False, data_type="json"
            )
            print(results)
        except Exception as e:
            logger.error(f"[DaumVclipSearcher]검색 요청 실패: {e}")
            raise SearchRequestFailedError(f"문제가 생겼습니다: {e}")

        if results is None:
            logger.error(f"[DaumVclipSearcher] 검색 결과 없음: '{query}'")
            raise SearchRequestFailedError(
                f"'{query}'에 대한 검색 요청에 실패했습니다."
            )
        if not results["documents"]:
            logger.warning(f"[DaumVclipSearcher] 검색 결과 없음: '{query}'")
            raise ValueError(f"'{query}'에 대한 검색 결과가 없습니다.")

        # YouTube URL만 필터링 (youtube.com 또는 youtu.be 도메인)
        youtube_pattern = r"(youtube\.com|youtu\.be)"
        filtered = [
            item["url"]
            for item in results["documents"]
            if "url" in item and re.search(youtube_pattern, item["url"], re.IGNORECASE)
        ]

        if not filtered:
            logger.warning(f"[DaumVclipSearcher] YouTube 검색 결과 없음: '{query}'")
            raise ValueError(f"'{query}'에 대한 YouTube 검색 결과가 없습니다.")

        logger.info(f"[DaumVclipSearcher] 검색 완료: {len(filtered)}개의 결과 발견")
        return filtered
