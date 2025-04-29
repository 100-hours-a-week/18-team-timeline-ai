from concurrent.futures import ThreadPoolExecutor, as_completed
from newspaper import Article
from typing import List, Tuple
from base_searcher import BaseSearcher


class ArticleExtractor(BaseSearcher):
    """기사 URL로부터 본문을 추출하는 클래스"""

    def __init__(self, max_workers: int = 6):
        """초기화

        Args:
            max_workers (int, optional): 스레드 숫자. Defaults to 6.
        """
        self.max_worksers = max_workers

    def _extract_single(self, url: str) -> Tuple[str, str]:
        try:
            article = Article(url=url, language="ko")
            article.download()
            article.parse()
            text = article.text.strip()
            if text:
                return (url, text)
        except Exception as e:
            print(f"Error exatracting {url}: {e}")

    def search(self, urls: List[str]) -> List[Tuple[str, str]]:
        """URL의 리스트를 받아 (url, 본문) 리스트를 반환

        Args:
            urls (List[str]): url의 리스트

        Returns:
            List[Tuple[str, str]]: (url, 본문) 리스트
        """
        results = []
        with ThreadPoolExecutor(max_workers=self.max_worksers) as executor:
            futures = {executor.submit(self._extract_single, url): url for url in urls}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                except Exception as e:
                    url = futures[future]
                    print(f"오류 발생 - URL: {url} | 예외: {type(e).__name__}: {e}")
        return results


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
