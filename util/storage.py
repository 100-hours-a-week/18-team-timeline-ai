from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from util.logger import Logger
import asyncio
from typing import Callable
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

logger = Logger.get_logger("utils.storage")


def is_qdrant_running(host: str, port: int, timeout: float = 1.0) -> bool:
    """Qdrant 서버가 실행 중인지 확인합니다.

    Args:
        host (str): Qdrant 서버 호스트
        port (int): Qdrant 서버 포트
        timeout (float, optional): 연결 시도 제한 시간. Defaults to 1.0.

    Returns:
        bool: 서버 실행 여부
    """
    try:
        logger.debug(f"[QdrantStorage] Qdrant 서버 실행 확인: {host}:{port}")
        with socket.create_connection((host, port), timeout=timeout):
            logger.debug(f"[QdrantStorage] Qdrant 서버 실행 확인 완료: {host}:{port}")
            return True
    except (ConnectionRefusedError, TimeoutError) as e:
        logger.error(f"[QdrantStorage] Qdrant 서버 연결 실패: {str(e)}")
        return False
    except OSError as e:
        logger.error(f"[QdrantStorage] Qdrant 서버 실행 확인 실패: {str(e)}")
        return False


class QdrantStorage:
    """Qdrant 벡터 저장소 클래스

    Qdrant 벡터 데이터베이스와의 상호작용을 담당하는 클래스입니다.
    """

    def __init__(
        self,
        collection_name: str = COLLECTION_NAME,
        batch_size: int = BATCH_SIZE,
        vector_size: int = VECTOR_SIZE,
        host: str = QDRANT_HOST,
        port: int = QDRANT_PORT,
        api_key: str = QDRANT_API_KEY,
    ):
        """QdrantStorage 초기화

        Args:
            collection_name (str): 컬렉션 이름
            batch_size (int, optional): 배치 처리 크기. Defaults to 64.
            vector_size (int, optional): 벡터 크기. Defaults to 1024.
            host (str, optional): Qdrant 서버 호스트. Defaults to "localhost".
            port (int, optional): Qdrant 서버 포트. Defaults to 6333.
            api_key (str, optional): Qdrant API 키. Defaults to None.
        """
        self.collection_name = collection_name
        self.batch_size = batch_size
        self.vector_size = vector_size
        self._client = None
        self._host = host
        self._port = port
        self._api_key = api_key
        self._closed = False

    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        if self._closed:
            raise RuntimeError("[QdrantStorage] 이미 종료된 클라이언트입니다.")

        try:
            url = f"{self._host}:{self._port}"
            self._client = QdrantClient(
                url=url,
                api_key=self._api_key,
                check_compatibility=False,
            )
            logger.info(f"[QdrantStorage] 클라이언트 생성 완료: {url}")
            return self
        except Exception as e:
            logger.error(f"[QdrantStorage] 초기화 실패: {str(e)}")
            raise

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        await self.close()

    async def close(self):
        """클라이언트 연결을 종료합니다."""
        if self._client is not None and not self._closed:
            try:
                self._client.close()
                logger.info("[QdrantStorage] 클라이언트 연결 종료")
            except Exception as e:
                logger.error(f"[QdrantStorage] 클라이언트 연결 종료 실패: {str(e)}")
            finally:
                self._client = None
                self._closed = True

    @asynccontextmanager
    async def get_client(self):
        """Qdrant 클라이언트를 컨텍스트 매니저로 제공합니다.

        Yields:
            QdrantClient: Qdrant 클라이언트 인스턴스
        """
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

    def _init_collection(self) -> None:
        """컬렉션을 초기화합니다."""
        if self._client is None:
            raise RuntimeError("[QdrantStorage] 클라이언트가 초기화되지 않았습니다.")
        if self._closed:
            raise RuntimeError("[QdrantStorage] 이미 종료된 클라이언트입니다.")

        try:
            logger.info(f"[QdrantStorage] 컬렉션 초기화 시작: {self.collection_name}")
            collections = self._client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            logger.info(f"[QdrantStorage] 컬렉션 존재 여부: {exists}")
            if not exists:
                self._client.create_collection(
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
        embedding_constructor: Callable,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """벡터 검색을 수행합니다.

        Args:
            query (str): 검색할 쿼리 텍스트
            embedding_constructor (Callable): 임베딩 생성자
            limit (int, optional): 반환할 결과 수. Defaults to 5.

        Returns:
            List[Dict[str, Any]]: 검색 결과

        Raises:
            RuntimeError: 클라이언트가 초기화되지 않은 경우
            Exception: 검색 실패 시
        """
        if self._client is None:
            raise RuntimeError("[QdrantStorage] 클라이언트가 초기화되지 않았습니다.")
        if self._closed:
            raise RuntimeError("[QdrantStorage] 이미 종료된 클라이언트입니다.")

        try:
            async with embedding_constructor() as embedder:
                logger.info(f"[QdrantStorage] 임베딩 생성 시작: {query}")
                vector = await embedder.embed_query(query)
                logger.info(f"[QdrantStorage] 임베딩 생성 완료: {len(vector)}")

                async with self.get_client() as client:
                    results = await asyncio.to_thread(
                        client.search,
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
        """배치 단위로 문서를 처리하고 Qdrant에 저장합니다.

        Args:
            documents (List[dict[str, Any]]): 처리할 문서 리스트
            embeddings (List[List[float]]): 각 문서의 임베딩 벡터 리스트
            start_idx (int): 현재 배치의 시작 인덱스

        Raises:
            Exception: 저장 실패 시
        """
        if self._client is None:
            raise RuntimeError("[QdrantStorage] 클라이언트가 초기화되지 않았습니다.")
        if self._closed:
            raise RuntimeError("[QdrantStorage] 이미 종료된 클라이언트입니다.")

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
                await asyncio.to_thread(
                    client.upsert, collection_name=self.collection_name, points=points
                )
            logger.info(f"[QdrantStorage] 배치 저장 완료: {len(points)}개 문서")
        except Exception as e:
            logger.error(f"[QdrantStorage] 배치 저장 실패: {str(e)}")
            raise
