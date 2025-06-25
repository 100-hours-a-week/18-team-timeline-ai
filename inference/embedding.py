from typing import List
from abc import ABC, abstractmethod
from utils.logger import Logger
import aiohttp
import asyncio
import orjson
from config.settings import BATCH_SIZE, OLLAMA_HOST, OLLAMA_MODELS, OLLAMA_PORT

logger = Logger.get_logger("embedding")


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
        base_url: str = f"{OLLAMA_HOST}:{OLLAMA_PORT}",
        model: str = OLLAMA_MODELS[0],
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
        logger.info(
            f"[OllamaEmbeddingService] 서버 {base_url} 초기화 완료 - "
            f"모델: {model}, 배치 크기: {batch_size}"
        )

    async def __aenter__(self):
        logger.info(f"[OllamaEmbeddingService] 서버 {self.base_url} 세션 생성 시작")
        try:
            timeout = aiohttp.ClientTimeout(total=300.0, connect=60.0, sock_read=60.0)
            self.session = aiohttp.ClientSession(timeout=timeout)
            logger.info(f"[OllamaEmbeddingService] 서버 {self.base_url} 세션 생성 완료")
            return self
        except Exception as e:
            logger.error(f"[OllamaEmbeddingService] 세션 생성 중 예외 발생: {str(e)}")
            if self.session and not self.session.closed:
                await self.session.close()
                logger.info(f"[OllamaEmbeddingService] 예외 발생으로 세션 강제 종료")
            raise

    async def __aexit__(self, exc_type, exc, tb):
        if self.session and not self.session.closed:
            logger.info(f"[OllamaEmbeddingService] 서버 {self.base_url} 세션 종료 시작")
            try:
                await self.session.close()
                # 세션 종료 후 잠깐 대기하여 완전한 정리 보장
                await asyncio.sleep(0.1)
                logger.info(
                    f"[OllamaEmbeddingService] 서버 {self.base_url} 세션 종료 완료"
                )
            except Exception as e:
                logger.warning(
                    f"[OllamaEmbeddingService] 세션 종료 중 예외 발생: {str(e)}"
                )
            finally:
                self.session = None
        else:
            logger.debug(f"[OllamaEmbeddingService] 세션이 이미 없거나 닫혀있음")
            self.session = None

    async def _make_embedding_request(self, text: str) -> List[float]:
        """단일 텍스트에 대한 임베딩 요청을 수행합니다.

        Args:
            text (str): 임베딩할 텍스트

        Returns:
            List[float]: 임베딩된 벡터

        Raises:
            Exception: 임베딩 생성 실패 시
        """
        logger.info(
            f"[OllamaEmbeddingService] 서버 {self.base_url} 임베딩 생성 시작 - "
            f"텍스트 길이: {len(text)}"
        )
        payload = orjson.dumps(
            {"model": self.model, "prompt": text},
        )
        headers = {"Content-Type": "application/json"}
        logger.debug(
            f"[OllamaEmbeddingService] 서버 {self.base_url} 요청 페이로드: {payload}"
        )

        try:
            # 세션 상태 확인 및 재생성
            if not self.session or self.session.closed:
                logger.warning(f"[OllamaEmbeddingService] 세션이 없거나 닫혀있음, 새 세션 생성")
                timeout = aiohttp.ClientTimeout(total=300.0, connect=60.0, sock_read=60.0)
                if self.session and not self.session.closed:
                    await self.session.close()
                self.session = aiohttp.ClientSession(timeout=timeout)
                logger.info(f"[OllamaEmbeddingService] 새 세션 생성 완료")

            async with self.session.post(
                f"{self.base_url}/api/embeddings",
                data=payload,
                headers=headers,
            ) as response:
                logger.debug(
                    f"[OllamaEmbeddingService] 서버 {self.base_url} 응답 상태 코드: {response.status}"
                )
                response_text = await response.text()

                if response.status == 200:
                    logger.info(
                        f"[OllamaEmbeddingService] 서버 {self.base_url} 임베딩 생성 성공"
                    )
                    data = orjson.loads(response_text)
                    embedding = data["embedding"]
                    logger.debug(
                        f"[OllamaEmbeddingService] 서버 {self.base_url} "
                        f"생성된 임베딩 크기: {len(embedding)}"
                    )
                    return embedding

                error_text = await response.text()
                logger.error(
                    f"[OllamaEmbeddingService] 서버 {self.base_url} 임베딩 생성 실패 - "
                    f"상태 코드: {response.status}, 에러: {error_text}"
                )
                raise Exception(f"임베딩 생성 실패: {error_text}")

        except aiohttp.ClientConnectorError as e:
            logger.error(
                f"[OllamaEmbeddingService] 서버 {self.base_url} 연결 오류: {str(e)}"
            )
            logger.error(f"[OllamaEmbeddingService] 연결 오류 타입: {type(e).__name__}")
            raise
        except asyncio.TimeoutError as e:
            logger.error(
                f"[OllamaEmbeddingService] 서버 {self.base_url} 타임아웃 오류: {str(e)}"
            )
            logger.error(
                f"[OllamaEmbeddingService] 타임아웃 오류 타입: {type(e).__name__}"
            )
            raise
        except aiohttp.ClientResponseError as e:
            logger.error(
                f"[OllamaEmbeddingService] 서버 {self.base_url} HTTP 응답 오류: {str(e)}"
            )
            logger.error(f"[OllamaEmbeddingService] HTTP 오류 타입: {type(e).__name__}")
            raise

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """문서 리스트를 배치 단위로 임베딩합니다.

        Args:
            texts (List[str]): 임베딩할 텍스트 리스트

        Returns:
            List[List[float]]: 임베딩된 벡터 리스트
        """
        logger.info(
            f"[OllamaEmbeddingService] 서버 {self.base_url} 문서 임베딩 시작 - "
            f"총 문서 수: {len(texts)}"
        )
        all_embeddings = []

        try:
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                logger.info(
                    f"[OllamaEmbeddingService] 서버 {self.base_url} 배치 처리 시작 - "
                    f"{i}~{i + len(batch)}/{len(texts)}"
                )
                tasks = [self._make_embedding_request(text) for text in batch]
                logger.debug(
                    f"[OllamaEmbeddingService] 서버 {self.base_url} "
                    f"배치 작업 생성 완료 - 작업 수: {len(tasks)}"
                )

                embeddings = await asyncio.gather(*tasks)
                all_embeddings.extend(embeddings)
                logger.info(
                    f"[OllamaEmbeddingService] 서버 {self.base_url} 배치 처리 완료 - "
                    f"{i + len(batch)}/{len(texts)}"
                )

            logger.info(
                f"[OllamaEmbeddingService] 서버 {self.base_url} 전체 문서 임베딩 완료 - "
                f"총 임베딩 수: {len(all_embeddings)}"
            )
            return all_embeddings

        except Exception as e:
            logger.error(
                f"[OllamaEmbeddingService] 서버 {self.base_url} "
                f"문서 임베딩 중 예외 발생: {str(e)}"
            )
            raise

    async def embed_query(self, text: str) -> List[float]:
        """단일 쿼리를 임베딩합니다.

        Args:
            text (str): 임베딩할 텍스트

        Returns:
            List[float]: 임베딩된 벡터
        """
        logger.info(
            f"[OllamaEmbeddingService] 서버 {self.base_url} 쿼리 임베딩 시작 - "
            f"텍스트 길이: {len(text)}"
        )
        try:
            embedding = await self._make_embedding_request(text)
            logger.info(
                f"[OllamaEmbeddingService] 서버 {self.base_url} 쿼리 임베딩 완료 - "
                f"임베딩 크기: {len(embedding)}"
            )
            return embedding
        except Exception as e:
            logger.error(
                f"[OllamaEmbeddingService] 서버 {self.base_url} "
                f"쿼리 임베딩 중 예외 발생: {str(e)}"
            )
            raise
