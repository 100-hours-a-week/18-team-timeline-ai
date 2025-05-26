from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional, Any, Union
import logging
from utils.logger import Logger
from ai_models.host import SystemRole

logger = Logger.get_logger("ai_models.store", log_level=logging.ERROR)


class ResultStore:
    """
    결과 저장소 클래스 - URL별 역할별 결과를 저장하고 관리
    """

    def __init__(self) -> None:
        """
        ResultStore 초기화
        """
        self._store: Dict[str, Dict[SystemRole, List[str]]] = OrderedDict()
        logger.info("ResultStore 인스턴스가 생성되었습니다.")

    def register(self, obj: Dict[str, Any]) -> None:
        """
        URL을 저장소에 등록

        Args:
            obj: URL 정보를 포함한 딕셔너리

        Raises:
            TypeError: 입력이 딕셔너리가 아닌 경우
            ValueError: URL 키가 없는 경우
        """
        if not isinstance(obj, dict):
            logger.error("register 실패: dict가 아님")
            raise TypeError("dict가 아닙니다.")

        url = obj.get("url")
        if not url:
            logger.error("register 실패: url 키가 없습니다.")
            raise ValueError("url 키가 없습니다.")

        if url not in self._store:
            self._store[url] = defaultdict(list)
            logger.info(f"등록 완료: {url}")
        else:
            logger.debug(f"이미 등록된 URL: {url}")

    def add_result(self, url: str, role: SystemRole, content: str) -> None:
        """
        결과 추가

        Args:
            url: 대상 URL
            role: 시스템 역할
            content: 결과 내용

        Raises:
            ValueError: URL이 비어 있는 경우
        """
        if not url:
            logger.error("add_result 실패: URL이 비어 있음")
            raise ValueError("URL은 비어 있을 수 없습니다.")

        try:
            if url not in self._store:
                logger.warning(f"URL이 등록되지 않아 자동 등록됨: {url}")
                self.register({"url": url})

            self._store[url][role].append(content)
            logger.info(f"결과 추가됨: [{url}][{role}] → {content[:30]}...")
        except Exception as e:
            logger.error(f"결과 추가 중 오류 발생: {e}")
            raise

    def get_results(self, url: str, role: SystemRole) -> List[str]:
        """
        특정 URL과 역할에 대한 결과 조회

        Args:
            url: 대상 URL
            role: 시스템 역할

        Returns:
            List[str]: 결과 목록
        """
        results = self._store.get(url, {}).get(role, [])
        logger.debug(f"get_results: {url} / {role} → {len(results)}개 반환")
        return results

    def display(self) -> None:
        """
        저장된 모든 결과를 콘솔에 출력
        """
        logger.info("저장된 결과를 출력합니다.")
        if not self._store:
            print("저장된 결과가 없습니다.")
            return

        for url in self._store:
            print(f"\nURL: {url}")
            for role in SystemRole:
                entries = self._store[url].get(role, [])
                if entries:
                    print(f"  {role}:")
                    for i, entry in enumerate(entries):
                        print(f"    {i+1}. {entry}")
        logger.info("출력 완료")

    def items(self):
        """
        저장소의 모든 항목 반환

        Returns:
            items: 저장소 항목 반복자
        """
        logger.debug("items() 호출됨")
        return self._store.items()

    def as_dict(self) -> Dict[str, Dict[str, List[str]]]:
        """
        저장소를 딕셔너리로 변환

        Returns:
            Dict: 변환된 딕셔너리
        """
        logger.debug("as_dict() 호출됨")
        return {
            url: {role.value: results for role, results in role_dict.items()}
            for url, role_dict in self._store.items()
        }

    def clear(self) -> None:
        """
        저장소 초기화
        """
        self._store.clear()
        logger.info("결과 저장소가 초기화되었습니다.")

    def get_urls(self) -> List[str]:
        """
        등록된 모든 URL 목록 반환

        Returns:
            List[str]: URL 목록
        """
        urls = list(self._store.keys())
        logger.debug(f"등록된 URL 목록 조회됨: {len(urls)}개")
        return urls


if __name__ == "__main__":
    # 테스트 코드
    store = ResultStore()

    # 정상 케이스 테스트
    store.register({"url": "https://example.com/b"})
    store.add_result(
        "https://example.com/a", role=SystemRole.TITLE, content="제목 결과"
    )
    store.add_result("https://example.com/b", SystemRole.SUMMARY, "요약 결과")
    store.add_result("https://example.com/b", SystemRole.TITLE, "제목 결과")
    store.add_result("https://example.com/b", SystemRole.TAG, "태그 결과")

    # 결과 출력
    store.display()

    # 딕셔너리 변환 테스트
    result_dict = store.as_dict()
    print("\n딕셔너리 변환 결과:")
    for url, data in result_dict.items():
        print(f"URL: {url}")
        for role, results in data.items():
            print(f"  {role}: {results}")
