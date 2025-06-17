from typing import List, Dict, Any, Callable
from qdrant_client.async_qdrant_client import AsyncQdrantClient
from qdrant_client.http.models import Distance, VectorParams
from utils.logger import Logger
import asyncio
import socket
from config.settings import (
    QDRANT_PORT,
    BATCH_SIZE,
    VECTOR_SIZE,
    COLLECTION_NAME,
    QDRANT_HOST,
    QDRANT_API_KEY,
)
from contextlib import asynccontextmanager
from qdrant_client.http.exceptions import UnexpectedResponse

logger = Logger.get_logger("storage")


def is_qdrant_running(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        logger.debug(f"[QdrantStorage] Qdrant 서버 실행 확인: {host}:{port}")
        with socket.create_connection((host, port), timeout=timeout):
            logger.debug(f"[QdrantStorage] Qdrant 서버 실행 확인 완료: {host}:{port}")
            return True
    except (ConnectionRefusedError, TimeoutError, OSError) as e:
        logger.error(f"[QdrantStorage] Qdrant 서버 연결 실패: {str(e)}")
        return False


class QdrantStorage:
    def __init__(
        self,
        collection_name: str = COLLECTION_NAME,
        batch_size: int = BATCH_SIZE,
        vector_size: int = VECTOR_SIZE,
        host: str = QDRANT_HOST,
        port: int = QDRANT_PORT,
        api_key: str = QDRANT_API_KEY,
    ):
        self.collection_name = collection_name
        self.batch_size = batch_size
        self.vector_size = vector_size
        self._client: AsyncQdrantClient | None = None
        self._host = host
        self._port = port
        self._api_key = api_key
        self._closed = False

    async def __aenter__(self):
        if self._closed:
            raise RuntimeError("[QdrantStorage] 이미 종료된 클라이언트입니다.")

        try:
            url = f"{self._host}:{self._port}"
            self._client = AsyncQdrantClient(
                url=url,
                api_key=self._api_key,
                timeout=30.0,
                prefer_grpc=False,
            )
            logger.info(f"[QdrantStorage] Async 클라이언트 생성 완료: {url}")
            return self
        except Exception as e:
            logger.error(f"[QdrantStorage] 초기화 실패: {str(e)}")
            raise

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        if self._client is not None and not self._closed:
            try:
                await self._client.close()
                logger.info("[QdrantStorage] 클라이언트 연결 종료")
            except Exception as e:
                logger.error(f"[QdrantStorage] 클라이언트 연결 종료 실패: {str(e)}")
            finally:
                self._client = None
                self._closed = True

    @asynccontextmanager
    async def get_client(self):
        if self._client is None:
            raise RuntimeError("[QdrantStorage] 클라이언트가 초기화되지 않았습니다.")
        if self._closed:
            raise RuntimeError("[QdrantStorage] 이미 종료된 클라이언트입니다.")

        try:
            yield self._client
        except UnexpectedResponse as e:
            logger.error(f"[QdrantStorage] Qdrant 서버 응답 오류: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"[QdrantStorage] 클라이언트 작업 실패: {str(e)}")
            raise

    async def _init_collection(self):
        if self._client is None or self._closed:
            raise RuntimeError("[QdrantStorage] 클라이언트가 유효하지 않습니다.")

        try:
            logger.info(f"[QdrantStorage] 컬렉션 초기화 시작: {self.collection_name}")
            collections = await self._client.get_collections()
            exists = any(
                c.name == self.collection_name for c in collections.collections
            )

            if not exists:
                await self._client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size, distance=Distance.COSINE
                    ),
                )
                logger.info(f"[QdrantStorage] 컬렉션 생성 완료: {self.collection_name}")
            else:
                logger.info(f"[QdrantStorage] 컬렉션 이미 존재: {self.collection_name}")
        except Exception as e:
            logger.error(f"[QdrantStorage] 컬렉션 초기화 실패: {str(e)}")
            raise

    async def search(
        self,
        query: str,
        embedder: Any,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        if self._client is None or self._closed:
            raise RuntimeError("[QdrantStorage] 클라이언트가 유효하지 않습니다.")

        try:
            logger.info(f"[QdrantStorage] 임베딩 생성 시작: {query}")
            vector = await embedder.embed_query(query)
            logger.info(f"[QdrantStorage] 임베딩 생성 완료: {len(vector)}")

            async with self.get_client() as client:
                results = await client.search(
                    collection_name=self.collection_name,
                    query_vector=vector,
                    limit=limit,
                )
                logger.info(f"[QdrantStorage] 검색 결과: {len(results)}개")
                return [
                    {"id": hit.id, "score": hit.score, "payload": hit.payload}
                    for hit in results
                ]
        except Exception as e:
            logger.error(f"[QdrantStorage] 벡터 검색 실패: {str(e)}")
            raise

    async def store_batch(
        self,
        documents: List[dict[str, Any]],
        embeddings: List[List[float]],
        start_idx: int,
    ) -> None:
        if self._client is None or self._closed:
            raise RuntimeError("[QdrantStorage] 클라이언트가 유효하지 않습니다.")

        try:
            points = [
                {
                    "id": start_idx + i,
                    "vector": embedding,
                    "payload": {
                        "comment": doc["text"],
                        "labels": doc["metadata"]["labels"],
                        "ID": doc["metadata"]["ID"],
                    },
                }
                for i, (doc, embedding) in enumerate(zip(documents, embeddings))
            ]
            logger.info(f"[QdrantStorage] 저장 준비 완료: {len(points)}개 문서")

            async with self.get_client() as client:
                await client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                )
            logger.info(f"[QdrantStorage] 배치 저장 완료: {len(points)}개 문서")
        except Exception as e:
            logger.error(f"[QdrantStorage] 배치 저장 실패: {str(e)}")
            raise
