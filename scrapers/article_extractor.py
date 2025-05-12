from concurrent.futures import ThreadPoolExecutor, as_completed
from newspaper import Article
from typing import List, Dict, Optional
from scrapers.base_searcher import BaseSearcher
import re
from utils.logger import Logger

logger = Logger.get_logger("article_extractor")


class ArticleExtractor(BaseSearcher):
    """기사 URL로부터 본문을 추출하는 클래스"""

    def __init__(self, max_workers: int = 6):
        """초기화

        Args:
            max_workers (int, optional): 스레드 숫자. Defaults to 6.
        """
        self.max_workers = max_workers

    def _extract_single(self, url: dict) -> Optional[Dict[str, str]]:
        """기사 URL로부터 본문을 추출하는 메서드

        Args:
            url (dict): 기사 URL과 제목이 포함된 딕셔너리
                {"url": str, "title": str}

        Returns:
            Optional[Dict[str, str]]: 추출된 기사 정보
                {"url": str, "title": str, "input_text": str}
                실패 시 None
        """
        try:
            article = Article(url=url["url"], language="ko")
            article.download()
            article.parse()
            text = article.text.strip()
            title = url["title"]
            title = re.sub(r"^[\[\(【]{0,1}속보[\]\)】]{0,1}\s*", "", title)

            if not text:
                logger.warning(f"본문이 비어있음 - URL: {url['url']}")
                return None

            logger.info(
                f"기사 추출 성공 - URL: {url['url']}, "
                f"제목: {title}, 본문 길이: {len(text)}"
            )

            return {"url": url["url"], "title": title, "input_text": text}

        except Exception as e:
            logger.error(
                f"기사 추출 실패 - URL: {url['url']}, "
                f"에러: {type(e).__name__}: {str(e)}"
            )
            return None

    def search(self, urls: List[dict]) -> List[dict]:
        """여러 URL에서 기사를 병렬로 추출하는 메서드

        Args:
            urls (List[dict]): 기사의 URL과 제목이 포함된 딕셔너리 리스트
                [{"url": str, "title": str}, ...]

        Returns:
            List[dict]: 추출된 기사 정보 리스트
                [{"url": str, "title": str, "input_text": str}, ...]
        """
        if not urls:
            logger.warning("URL 리스트가 비어있습니다.")
            return []

        results = [None] * len(urls)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._extract_single, items): idx
                for idx, items in enumerate(urls)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    if result:
                        results[idx] = result
                except Exception as e:
                    url = urls[idx]
                    logger.error(
                        f"작업 처리 실패 - URL: {url['url']}, "
                        f"에러: {type(e).__name__}: {str(e)}"
                    )

        # None이 아닌 결과만 반환
        valid_results = [r for r in results if r is not None]
        logger.info(f"기사 추출 완료 - 전체: {len(urls)}, 성공: {len(valid_results)}")
        return valid_results


if __name__ == "__main__":
    URLS = [
        {
            "url": "https://www.hani.co.kr/arti/society/society_general/1192251.html",
            "title": "테스트 기사 1",
        },
        {
            "url": "https://www.hani.co.kr/arti/society/society_general/1192255.html",
            "title": "테스트 기사 2",
        },
        {
            "url": "https://www.hankyung.com/article/2025041493977",
            "title": "테스트 기사 3",
        },
    ]
    runner = ArticleExtractor()
    results = runner.search(urls=URLS)

    for result in results:
        print(f"\n제목: {result['title']}")
        print(f"URL: {result['url']}")
        print(f"본문 길이: {len(result['input_text'])}")
        print("-" * 50)
