import re
import aiohttp
import asyncio
import trafilatura
from utils.logger import Logger
import requests

from typing import List, Dict, Optional, AsyncGenerator
from scrapers.base_searcher import BaseSearcher

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
        # requests 세션 생성
        self.session = requests.Session()
        # 연결 풀 설정
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=100,  # 연결 풀 크기
            pool_maxsize=100,  # 최대 연결 수
            max_retries=3,  # 재시도 횟수
        )
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

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
            # requests를 사용하여 페이지 다운로드
            response = self.session.get(url["url"], timeout=10)
            response.raise_for_status()
            html_content = response.text

            # trafilatura로 본문 추출
            text = trafilatura.extract(
                html_content, include_comments=False, include_tables=False
            )
            if not text or not text.strip():
                logger.error(f"[ArticleExtractor] 본문이 비어있음 - URL: {url['url']}")
                return None

            # 리턴
            text = text.strip()
            title = url["title"]
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
        # 세마포어를 사용하여 동시 요청 수 제한
        semaphore = asyncio.Semaphore(self.max_workers)

        async def bounded_extract(url):
            async with semaphore:
                return await self.extract_single(url)

        task_list = []
        for url in urls:
            task = asyncio.create_task(bounded_extract(url))
            task_list.append(task)

        for task in asyncio.as_completed(task_list):
            try:
                result = await task
                if result:
                    yield result
            except Exception as e:
                logger.error(f"[ArticleExtractor] 작업 처리 실패 : {e}")

    def __del__(self):
        # 세션 정리
        if hasattr(self, "session"):
            self.session.close()


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
