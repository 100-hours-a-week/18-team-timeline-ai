from newspaper import Article
from typing import List, Dict, Optional, AsyncGenerator
from scrapers.base_searcher import BaseSearcher
import re
from utils.logger import Logger
import aiohttp
import asyncio
from config.settings import (
    OLLAMA_HOST,
    OLLAMA_MODEL,
    BATCH_SIZE,
)
import numpy as np
import logging

logger = Logger.get_logger("article_extractor", log_level=logging.ERROR)


class ArticleExtractor(BaseSearcher):
    """기사 URL로부터 본문을 추출하는 클래스"""

    def __init__(self, lang: str = "ko", max_workers: int = 6):
        """초기화

        Args:
            max_workers (int, optional): 스레드 숫자. Defaults to 6.
        """
        self.max_workers = max_workers
        self.lang = lang

    async def extract_single(self, url: dict) -> Optional[Dict[str, str]]:
        """기사 URL로부터 본문을 추출하는 메서드

        Args:
            url (dict): 기사 URL과 제목이 포함된 딕셔너리
                {"url": str, "title": str}
        """
        return await asyncio.to_thread(self._extract_single, url)

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
            article = Article(url=url["url"], language=self.lang)
            article.download()
            article.parse()
            text = article.text.strip()
            title = url["title"]
            title = re.sub(r"^[\[\(【]{0,1}속보[\]\)】]{0,1}\s*", "", title)
            if not text:
                logger.warning(
                    f"[ArticleExtractor] 본문이 비어있음 - URL: {url['url']}"
                )
                return None

            logger.info(
                f"[ArticleExtractor] 기사 추출 성공 - URL: {url['url']}, "
                f"제목: {title}, 본문 길이: {len(text)}"
            )

            return {
                "url": url["url"],
                "title": title,
                "input_text": text,
            }

        except Exception as e:
            logger.error(
                f"[ArticleExtractor] 기사 추출 실패 - URL: {url['url']}, "
                f"에러: {type(e).__name__}: {str(e)}"
            )
            return None

    async def search(self, urls: List[dict]) -> AsyncGenerator[dict, None]:
        """여러 URL에서 기사를 병렬로 추출하는 메서드

        Args:
            urls (List[dict]): 기사의 URL과 제목이 포함된 딕셔너리 리스트
                [{"url": str, "title": str}, ...]

        Returns:
            AsyncGenerator[dict, None]: 추출된 기사 정보 리스트
                [{"id": int, "url": str, "title": str, "input_text": str}, ...]

        Yields:
            Iterator[AsyncGenerator[dict, None]]: 추출된 기사 정보 리스트
        """

        task_list = []

        for idx, url in enumerate(urls):
            task = asyncio.create_task(self.extract_single(url))
            task_list.append(task)

        for task in asyncio.as_completed(task_list):
            try:
                result = await task
                if result:
                    yield result
            except Exception as e:
                logger.error(f"[ArticleExtractor] 작업 처리 실패 : {e}")


class ArticleParser:
    def __init__(
        self,
        session_size: int = 10,
        max_workers: int = 6,
    ):
        self.session_size = session_size
        self.max_workers = max_workers

    async def parse(self, text: dict) -> dict:
        """기사 본문을 문장으로 분리하는 메서드

        Args:
            text (dict): 기사 본문
                {"id": int, "title": str, "url": str, "input_text": str}
        Returns:
            dict: 기사 본문
                {"id": int, "title": str, "url": str, "sentences": List[str]}
        """
        logger.info(f"[ArticleParser] : 기사 본문 분리 시작 - 길이: {len(text)}")
        try:
            return {
                "title": text["title"],
                "url": text["url"],
                "sentences": [
                    line.strip()
                    for line in re.split(r"[.]", text["input_text"].strip())
                    if is_meaningful_sentence(line)
                ],
            }
        except Exception as e:
            logger.error(f"[ArticleParser] 기사 본문 분리 실패 : {e}")
            return None


class ArticleFilter:
    """
    기사 필터 클래스
    Ollama API를 사용해서 기사 본문에서 문장으로 분리하고,
    각 문장에 대해서 중요한 문장을 추출하는 클래스
    """

    def __init__(
        self,
        base_url: str = OLLAMA_HOST,
        model: str = OLLAMA_MODEL,
        batch_size: int = BATCH_SIZE,
        top_k: int = 2,
    ):
        """
        Args:
            base_url (str, optional): Ollama API 기본 URL.
                Defaults to "http://localhost:11434".
            model (str, optional): 사용할 모델 이름.
                Defaults to "bge-m3".
            batch_size (int, optional): 배치 처리 크기.
                Defaults to 64.
        """
        self.base_url = base_url
        self.model = model
        self.batch_size = batch_size
        self.session = None
        self.top_k = top_k

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self.session and not self.session.closed:
            await self.session.close()

    async def _make_embedding_request(self, text: str) -> List[float]:
        """Ollama API를 사용해서 문장을 임베딩하는 메서드

        Args:
            text (str): 문장

        Returns:
            List[float]: 임베딩 벡터
        """
        try:
            logger.info(f"[ArticleFilter] 임베딩 요청 시작 - {text}")
            async with self.session.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
            ) as response:
                logger.info(f"[ArticleFilter] 임베딩 요청 응답: {response.status}")
                if response.status == 200:
                    logger.info(f"[ArticleFilter] 임베딩 요청 완료 - {text}")
                    return await response.json()
                error_text = await response.text()
                raise Exception(f"{error_text}")
        except Exception as e:
            logger.error(f"[ArticleFilter] 임베딩 요청 실패 : {e}")
            return []

    async def extract_key_sentences(self, news: dict) -> List[str]:
        """기사 본문에서 키 문장을 추출하는 메서드

        Args:
            news (dict): 기사 정보

        Returns:
            List[str]: 키 문장 리스트
        """
        if not news["sentences"]:
            logger.warning(f"[ArticleFilter] 문장이 없는 기사 제거 - {news['title']}")
            return []
        try:
            # 제목 임베딩
            logger.info(f"[ArticleFilter] 제목 임베딩 시작 - {news['title']}")
            title_embed_response = await self._make_embedding_request(news["title"])
            logger.info(f"[ArticleFilter] 제목 임베딩 완료 - {news['title']}")
            title_embedding = title_embed_response["embedding"]

            # 문장별 임베딩 요청
            logger.info(f"[ArticleFilter] 문장별 임베딩 시작 - {news['title']}")
            sentence_embeddings = []
            for sentence in news["sentences"]:
                res = await self._make_embedding_request(sentence)
                sentence_embeddings.append((sentence, res["embedding"]))
            logger.info(f"[ArticleFilter] 문장별 임베딩 완료 - {news['title']}")
            # 유사도 계산 및 정렬
            logger.info(f"[ArticleFilter] 유사도 계산 시작 - {news['title']}")
            scored = [
                (sentence, cosine_similarity(embedding, title_embedding))
                for sentence, embedding in sentence_embeddings
            ]
            logger.info(f"[ArticleFilter] 유사도 계산 완료 - {news['title']}")
            scored.sort(key=lambda x: x[1], reverse=True)
            logger.info(f"[ArticleFilter] 유사도 정렬 완료 - {news['title']}")
            return [s for s, _ in scored[: self.top_k]]
        except Exception as e:
            logger.error(f"[ArticleFilter] 키 문장 추출 실패 : {e}")
            return []


def assign_id_from_URLS(URLS: List[dict]) -> List[dict]:
    """URLS에서 id를 생성하는 함수
    Args:
        URLS (List[dict]): URLS
    Returns:
        List[dict]: URLS에서 id를 생성한 딕셔너리 리스트
    """
    logger.info(f"[assign_id_from_URLS] URLS 길이: {len(URLS)}")
    try:
        return [{"id": i, **url} for i, url in enumerate(URLS)]
    except Exception as e:
        logger.error(f"[assign_id_from_URLS] URLS에서 id를 생성하는 중 오류 발생: {e}")
        return []


def is_meaningful_sentence(line: str) -> bool:
    """문장이 의미 있는 문장인지 확인하는 함수
    Args:
        line (str): 문장
    Returns:
        bool: 의미 있는 문장인지 여부
    """
    try:
        line = line.strip()
        if not line or line.lower() in {"광고", "com", "kr", "co"}:
            logger.warning(f"[is_meaningful_sentence] 의미 없는 문장 제거 - {line}")
            return False
        if "기자" in line and "@" in line:
            logger.warning(f"[is_meaningful_sentence] 의미 없는 문장 제거 - {line}")
            return False
        if "광고" in line or "사진=" in line or "Your browser" in line:
            logger.warning(f"[is_meaningful_sentence] 의미 없는 문장 제거 - {line}")
            return False
        if re.match(r"^\\d+\\s*[:\\-]*\\s*$", line):  # 숫자 단독 줄
            logger.warning(f"[is_meaningful_sentence] 의미 없는 문장 제거 - {line}")
            return False
        if line.strip() == "":
            logger.warning(f"[is_meaningful_sentence] 의미 없는 문장 제거 - {line}")
            return False
    except Exception as e:
        logger.error(f"[is_meaningful_sentence] 문장 확인 중 오류 발생: {e}")
        return False
    logger.info(f"[is_meaningful_sentence] 의미 있는 문장 확인 - {line}")
    return True


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """두 벡터의 코사인 유사도를 계산하는 함수"""
    try:
        a, b = np.array(a), np.array(b)
        logger.info(f"[cosine_similarity] 코사인 유사도 계산 - {a}, {b}")
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    except Exception as e:
        logger.error(f"[cosine_similarity] 코사인 유사도 계산 중 오류 발생: {e}")
        return 0.0


if __name__ == "__main__":

    async def main():
        URLS = [
            {
                "url": "https://www.hani.co.kr/arti/society/society_general/1192251.html",
                "title": "말 바꾼 윤석열 “계엄 길어야 하루”…헌재선 “며칠 예상”",
            },
            {
                "url": "https://www.hani.co.kr/arti/society/society_general/1192255.html",
                "title": "윤석열 40분간 “계엄은 평화적 메시지”…판사도 발언 ‘시간조절’ 당부",
            },
            {
                "url": "https://www.hankyung.com/article/2025041493977",
                "title": "'[속보] 韓대행 '국무위원들과 제게 부여된 마지막 소명 다할 것'",
            },
        ]
        URLS = assign_id_from_URLS(URLS)
        logger.info(f"[main] URLS 길이: {len(URLS)}")
        runner = ArticleExtractor()
        logger.info(f"[main] ArticleExtractor 초기화 완료")
        parser = ArticleParser()
        logger.info(f"[main] ArticleParser 초기화 완료")
        filter = ArticleFilter()
        logger.info(f"[main] ArticleFilter 초기화 완료")
        async with filter:
            async for result in runner.search(urls=URLS):
                logger.info(f"[main] 기사 추출 시작 - {result['title']}")
                print(f"ID: {result['id']}")
                print(f"제목: {result['title']}")
                print(f"URL: {result['url']}")
                print(f"본문 길이: {len(result['input_text'])}")
                print("-" * 100)
                parsed_result = await parser.parse(result)
                logger.info(f"[main] 기사 본문 분리 완료 - {result['title']}")
                for idx, line in enumerate(parsed_result["sentences"]):
                    print(f"{idx} : {line}")
                print("-" * 100)

                print(f"제목: {parsed_result['title']}")
                key_sentences = await filter.extract_key_sentences(parsed_result)
                logger.info(f"[main] 키 문장 추출 시작 - {parsed_result['title']}")
                for idx, line in enumerate(key_sentences):
                    print(f"{idx} : {line}")
                print("-" * 100)
                logger.info(f"[main] 키 문장 추출 완료 - {parsed_result['title']}")

    logger.info("[main] 프로그램 시작")
    asyncio.run(main())
    logger.info("[main] 프로그램 종료")
