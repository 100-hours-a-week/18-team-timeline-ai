import re
import logging

from utils.logger import Logger
from ai_models.host import SystemRole
from collections import OrderedDict, defaultdict

logger = Logger.get_logger("ai_models.store", log_level=logging.ERROR)


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

    def parse_all_in_one_content(self, content: str) -> dict:
        """
        all-in-one 응답 문자열을 파싱하여
        {SystemRole.summary: ..., SystemRole.title: ..., SystemRole.tag: ...} 딕셔너리로 반환

        유연한 포맷 대응을 위해 정규식을 사용
        """
        role_map = {
            "요약": SystemRole.summary,
            "제목": SystemRole.title,
            "태그": SystemRole.tag,
        }
        results = {}

        try:
            # 각 줄에서 역할별 값을 추출
            for line in content.strip().splitlines():
                match = re.match(r"^\s*[\d\-\.]*\s*(요약|제목|태그)\s*[:：]\s*(.+)$", line)
                if match:
                    key_kor = match.group(1).strip()
                    value = match.group(2).strip()
                    role = role_map.get(key_kor)
                    if role:
                        results[role] = value
        except Exception as e:
            logger.error(f"[파싱 오류] all-in-one content 파싱 실패: {repr(e)}")
            raise

        return results

    def add_result(self, url: str, role: SystemRole, content: str):
        if not url:
            logger.error("add_result 실패: URL이 비어 있음")
            raise RuntimeError("URL은 비어 있을 수 없습니다.")

        if url not in self._store:
            logger.warning(f"URL이 등록되지 않아 자동 등록됨: {url}")
            self.register({"url": url})

        if role == SystemRole.all_in_one:
            parsed_results = self.parse_all_in_one_content(content)
            for parsed_role, parsed_value in parsed_results.items():
                self._store[url][parsed_role].append(parsed_value)
                logger.info(f"[Parsed 저장] [{url}][{parsed_role.name}] → {parsed_value[:30]}...")
        else:
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


if __name__ == "__main__":
    store = ResultStore()

    store.register({"url": "https://example.com/b"})
    store.add_result("qsdqsdqdq", role=SystemRole.TITLE, content="제목 결과")
    store.add_result("https://example.com/a", SystemRole.SUMMARIZE, "요약 결과")
    store.add_result("https://example.com/b", SystemRole.TITLE, "제목 결과")
    store.add_result("https://example.com/b", SystemRole.TITLE, "제목 결과")
    store.add_result("https://example.com/b", SystemRole.TAG, "제목 결과")

    store.display()
