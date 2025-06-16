from collections import OrderedDict, defaultdict
import logging
from util.logger import Logger
from config.prompts import SystemRole

logger = Logger.get_logger("utils.store")


class ResultStore:
    def __init__(self) -> None:
        self._store = OrderedDict()
        logger.info("ResultStore 인스턴스가 생성되었습니다.")

    def register(self, obj: dict):
        if isinstance(obj, dict):
            logger.debug("register()에 dict가 전달됨")
            url = obj.get("url")
            if not url:
                logger.error("register 실패: url 키가 없습니다.")
                raise ValueError("url 키가 없습니다.")
        else:
            logger.error("register 실패: dict가 아님")
            raise TypeError("dict가 아닙니다.")

        if url not in self._store:
            self._store[url] = defaultdict(list)
            logger.info(f"등록 완료: {url}")
        else:
            logger.debug(f"이미 등록된 URL: {url}")

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
