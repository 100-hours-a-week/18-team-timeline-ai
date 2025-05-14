from typing import Optional
from PyKakao import DaumSearch


class DaumKeywordMeaningChecker:
    """
    다음 검색을 통해 검색어가 유의미한지 판단
    - 검색 결과가 있으면 유의미(True)
    - 검색 결과가 없으면 무의미(False)
    """

    def __init__(self, api_key: str, daum_service: Optional[DaumSearch] = None):
        self.service_key = api_key
        try:
            self.api = (
                daum_service
                if daum_service
                else DaumSearch(service_key=self.service_key)
            )
        except Exception as e:
            raise ValueError(f"Daum API 연결 실패: {e}")

    def is_meaningful(self, keyword: str) -> bool:
        # 패턴 인식
        if not keyword or keyword.strip() == "":
            return False
        for c in keyword:
            if 'ㄱ' <= c <= 'ㅎ' or 'ㅏ' <= c <= 'ㅣ':
                return False

        # 웹 검색
        try:
            # 다음 Web 검색 → 결과 개수 확인
            result = self.api.search_web(query=keyword, data_type="json")
            documents = result.get("documents", [])
            return len(documents) > 3
        except Exception as e:
            print(f"[Daum API Error] 검색 실패: {e}")
            return False
