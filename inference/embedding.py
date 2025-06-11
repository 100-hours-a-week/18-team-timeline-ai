from typing import List
from abc import ABC, abstractmethod
from utils.logger import Logger
import aiohttp
import asyncio
import logging
import orjson
from config.settings import (
    BATCH_SIZE,
    OLLAMA_HOST,
    OLLAMA_MODEL,
)

logger = Logger.get_logger("embedding", log_level=logging.ERROR)


class EmbeddingModel(ABC):
    """임베딩 모델 추상 클래스

    텍스트를 벡터로 변환하는 모델의 기본 인터페이스를 정의합니다.
    """

    @abstractmethod
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        pass

    @abstractmethod
    async def embed_query(self, text: str) -> List[float]:
        pass


class OllamaEmbeddingService(EmbeddingModel):
    """Ollama 임베딩 서비스 클래스

    Ollama API를 사용하여 텍스트를 벡터로 변환하는 서비스입니다.
    """

    def __init__(
        self,
        base_url: str = OLLAMA_HOST,
        model: str = OLLAMA_MODEL,
        batch_size: int = BATCH_SIZE,
    ):
        """OllamaEmbeddingService 초기화

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

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self.session and not self.session.closed:
            await self.session.close()

    async def _make_embedding_request(self, text: str) -> List[float]:
        """단일 텍스트에 대한 임베딩 요청을 수행합니다.

        Args:
            session (aiohttp.ClientSession): HTTP 세션
            text (str): 임베딩할 텍스트

        Returns:
            List[float]: 임베딩된 벡터

        Raises:
            Exception: 임베딩 생성 실패 시
        """
        logger.info(f"[OllamaEmbeddingService] 임베딩 생성 시작: {text}")
        payload = orjson.dumps(
            {"model": self.model, "prompt": text},
        )
        headers = {"Content-Type": "application/json"}
        logger.info(f"[OllamaEmbeddingService] 임베딩 생성 요청: {payload}")
        try:
            async with self.session.post(
                f"{self.base_url}/api/embeddings",
                data=payload,
                headers=headers,
            ) as response:
                logger.info(
                    f"[OllamaEmbeddingService] 임베딩 생성 응답: {response.status}"
                )
                response_text = await response.text()
                logger.info(
                    f"[OllamaEmbeddingService] 임베딩 생성 응답: {response_text}"
                )
                if response.status == 200:
                    logger.info(f"[OllamaEmbeddingService] 임베딩 생성 성공")
                    data = orjson.loads(response_text)
                    return data["embedding"]
                error_text = await response.text()
                raise Exception(f"{error_text}")
        except Exception as e:
            logger.error(f"[OllamaEmbeddingService] 임베딩 생성 실패: {e}")
            raise

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """문서 리스트를 배치 단위로 임베딩합니다.

        Args:
            texts (List[str]): 임베딩할 텍스트 리스트

        Returns:
            List[List[float]]: 임베딩된 벡터 리스트
        """

        all_embeddings = []
        try:
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                logger.info(
                    f"[OllamaEmbeddingService] 임베딩 생성 시작: {i} : {i + self.batch_size}"
                )
                tasks = [self._make_embedding_request(text) for text in batch]
                logger.info(f"[OllamaEmbeddingService] 임베딩 생성 대기: {tasks}")
                embeddings = await asyncio.gather(*tasks)
                all_embeddings.extend(embeddings)
                logger.info(f"배치 처리 완료: {i + len(batch)}/{len(texts)}")
            return all_embeddings
        except Exception as e:
            logger.error(f"[OllamaEmbeddingService] 임베딩 생성 실패: {e}")
            raise

    async def embed_query(self, text: str) -> List[float]:
        """단일 쿼리를 임베딩합니다.

        Args:
            text (str): 임베딩할 텍스트

        Returns:
            List[float]: 임베딩된 벡터
        """
        try:
            return await self._make_embedding_request(text)
        except Exception as e:
            logger.error(f"[OllamaEmbeddingService] 임베딩 생성 실패: {e}")
            raise
