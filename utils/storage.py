from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from utils.logger import Logger
import asyncio
from typing import Callable
import socket
from config.settings import (
    QDRANT_PORT,
    BATCH_SIZE,
    VECTOR_SIZE,
    COLLECTION_NAME,
)
from dotenv import load_dotenv
import os

load_dotenv(override=True)
QDRANT_HOST = os.getenv("QDRANT_HOST")

logger = Logger.get_logger("storage")


def is_qdrant_running(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        logger.debug(f"[QdrantStorage] Qdrant 서버 실행 확인: {host}:{port}")
        with socket.create_connection((host, port), timeout=timeout):
            logger.debug(f"[QdrantStorage] Qdrant 서버 실행 확인 완료: {host}:{port}")
            return True
    except OSError:
        logger.error(f"[QdrantStorage] Qdrant 서버 실행 확인 실패: {host}:{port}")
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
    ):
        """QdrantStorage 초기화

        Args:rr
            collection_name (str): 컬렉션 이름
            batch_size (int, optional): 배치 처리 크기. Defaults to 64.
            vector_size (int, optional): 벡터 크기. Defaults to 1024.
            host (str, optional): Qdrant 서버 호스트. Defaults to "localhost".
            port (int, optional): Qdrant 서버 포트. Defaults to 6333.
        """
        try:
            self.collection_name = collection_name
            self.batch_size = batch_size
            self.vector_size = vector_size

            if not is_qdrant_running(host, port):
                logger.error(
                    f"[QdrantStorage] 컨테이너 시작 실패: {host}:{port}. "
                    "Qdrant 서버가 실행 중이지 않습니다."
                )
                raise RuntimeError(
                    f"[QdrantStorage] 컨테이너 시작 실패: {host}:{port}. "
                    "Qdrant 서버가 실행 중이지 않습니다."
                )

            self.client = QdrantClient(host=host, port=port)
            logger.info(f"[QdrantStorage] 클라이언트 생성 완료: {host}/{port}")
            """
            self._init_collection()
            logger.info(f"[QdrantStorage] 컬렉션 초기화 완료: {self.collection_name}")
            logger.info(f"클라이언트: {host}/{port}")
            logger.info(f"컬렉션: {self.collection_name}")
            """
        except Exception as e:
            logger.error(f"[QdrantStorage] 초기화 실패: {str(e)}")
            raise OSError(f"[QdrantStorage] 초기화 실패: {str(e)}")

    def _init_collection(self) -> None:
        """컬렉션을 초기화합니다."""
        try:
            logger.info(f"[QdrantStorage] 컬렉션 초기화 시작: {self.collection_name}")
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)
            logger.info(f"[QdrantStorage] 컬렉션 존재 여부: {exists}")
            if not exists:
                self.client.create_collection(
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
            embedding (EmbeddingModel): 임베딩 모델
            limit (int, optional): 반환할 결과 수. Defaults to 5.

        Returns:
            List[Dict[str, Any]]: 검색 결과

        Raises:
            Exception: 검색 실패 시
        """
        try:
            async with embedding_constructor() as embedder:
                logger.info(f"[QdrantStorage] 임베딩 생성 시작: {query}")
                vector = await embedder.embed_query(query)
                logger.info(f"[QdrantStorage] 임베딩 생성 완료: {len(vector)}")
                results = await asyncio.to_thread(
                    self.client.search,
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
            await asyncio.to_thread(
                self.client.upsert, collection_name=self.collection_name, points=points
            )
            logger.info(f"[QdrantStorage] 배치 저장 완료: {len(points)}개 문서")
        except Exception as e:
            logger.error(f"[QdrantStorage] 배치 저장 실패: {str(e)}")
            raise
