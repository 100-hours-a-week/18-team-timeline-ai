from concurrent.futures import ThreadPoolExecutor, as_completed
from newspaper import Article
from typing import List, Tuple
from scrapers.base_searcher import BaseSearcher
import re
import logging


class ArticleExtractor(BaseSearcher):
    """기사 URL로부터 본문을 추출하는 클래스"""

    def __init__(self, max_workers: int = 6):
        """초기화

        Args:
            max_workers (int, optional): 스레드 숫자. Defaults to 6.
        """
        self.max_workers = max_workers

    def _extract_single(self, url: str) -> dict:
        """_기사 URL로부터 본문을 추출하는 메서드

        Args:
            url (str): 기사 URL

        Returns:
            Tuple[str, str, str]: 기사 URL, 제목, 본문
        """
        try:
            article = Article(url=url, language="ko")
            article.download()
            article.parse()
            text = article.text.strip()
            title = article.title.strip()
            title = re.sub(r"^[\[\(【]{0,1}속보[\]\)】]{0,1}\s*", "", title)
            if text:
                return {"url:": url, "title": title, "text": text}
        except Exception as e:
            print(f"Error exatracting {url}: {e}")

    def search(self, urls: List[str]) -> List[dict]:
        """_extract_single_ 메서드를 사용하여 URL 리스트를 스레드로 처리하는 메서드

        Args:
            urls (List[str]): 기사의 URL 리스트

        Returns:
            List[Tuple[str, str, str]]: 각 URL에 대한 (URL, 제목, 본문) 튜플 리스트
        """
        results = [None] * len(urls)
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._extract_single, url): idx
                for (idx, url) in enumerate(urls)
            }
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    if result:
                        results[idx] = result
                except Exception as e:
                    url = urls[idx]
                    logging.error(
                        f"오류 발생 - URL: {url} | 예외: {type(e).__name__}: {e}"
                    )
        return [r for r in results if r is not None]


if __name__ == "__main__":
    URLS = [
        "https://www.hani.co.kr/arti/society/society_general/1192251.html",
        "https://www.hani.co.kr/arti/society/society_general/1192255.html",
        "https://www.hankyung.com/article/2025041493977",
        "https://www.khan.co.kr/article/202504141136001",
        "https://www.mk.co.kr/news/politics/11290687",
        "https://www.chosun.com/politics/politics_general/2025/04/14/THWVKUHQG5CKFJF6CLZLP5PKM4",
    ]
    runner = ArticleExtractor()
    ret = runner.search(urls=URLS)

    print(*ret, sep="\n\n\n\n")
