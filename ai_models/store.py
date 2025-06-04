from collections import OrderedDict, defaultdict
import logging
from utils.logger import Logger
from ai_models.host import SystemRole

logger = Logger.get_logger("ai_models.store", log_level=logging.ERROR)


class ResultStore:
    def __init__(self) -> None:
        self._store = OrderedDict()
        logger.info("ResultStore 인스턴스가 생성되었습니다.")

    def register(self, obj: dict) -> None:
        """URL을 등록하는 메서드

        Args:
            obj (dict): URL 정보를 담은 딕셔너리
                {"url": str}

        Raises:
            ValueError: URL이 유효하지 않은 경우
            TypeError: 입력이 딕셔너리가 아닌 경우
        """
        try:
            if not isinstance(obj, dict):
                logger.error(f"[ResultStore] 잘못된 입력 타입: {type(obj)}")
                raise TypeError("입력은 딕셔너리여야 합니다")

            url = obj.get("url")
            if not url:
                logger.error("[ResultStore] URL이 제공되지 않음")
                raise ValueError("URL이 필요합니다")

            if not isinstance(url, str):
                logger.error(f"[ResultStore] 잘못된 URL 타입: {type(url)}")
                raise ValueError("URL은 문자열이어야 합니다")

            if not url.strip():
                logger.error("[ResultStore] 빈 URL")
                raise ValueError("URL은 비어있을 수 없습니다")

            # URL 형식 검사
            if not url.startswith(("http://", "https://")):
                logger.error(f"[ResultStore] 잘못된 URL 형식: {url}")
                raise ValueError("URL은 http:// 또는 https://로 시작해야 합니다")

            logger.info(f"[ResultStore] URL 등록 성공: {url}")
            self._store[url] = defaultdict(list)

        except Exception as e:
            logger.error(f"[ResultStore] URL 등록 실패: {str(e)}")
            raise

    def add_result(self, url: str, role: SystemRole, content: str):
        if not url:
            logger.error("add_result 실패: URL이 비어 있음")
            raise RuntimeError("URL은 비어 있을 수 없습니다.")

        if url not in self._store:
            logger.warning(f"URL이 등록되지 않아 자동 등록됨: {url}")
            self.register({"url": url})

        self._store[url][role].append(content)
        logger.info(f"결과 추가됨: [{url}][{role.name}] → {content[:30]}...")

    def get_results(self, url: str, role: SystemRole) -> list:
        results = self._store.get(url, {}).get(role, [])
        logger.debug(f"get_results: {url} / {role.name} → {len(results)}개 반환")
        return results

    def display(self):
        logger.info("저장된 결과를 출력합니다.")
        for url in self._store:
            print(f"\nURL: {url}")
            for role in SystemRole:
                entries = self._store[url].get(role, [])
                print(f"  {role.name}:")
                for i, entry in enumerate(entries):
                    print(f"    {i+1}. {entry}")
        logger.info("출력 완료")

    def items(self):
        logger.debug("items() 호출됨")
        return self._store.items()

    def as_dict(self):
        logger.debug("as_dict() 호출됨")
        return {
            url: {role.name: results for role, results in role_dict.items()}
            for url, role_dict in self._store.items()
        }

    def clear(self):
        self._store.clear()
        logger.info("결과 저장소가 초기화되었습니다.")

    def get_urls(self) -> list:
        urls = list(self._store.keys())
        logger.debug(f"등록된 URL 목록 조회됨: {len(urls)}개")
        return urls
