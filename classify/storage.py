from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain.schema import Document
from utils.logger import Logger
import asyncio
from typing import Callable
import socket
import subprocess
import time
from config.settings import (
    QDRANT_HOST,
    QDRANT_PORT,
    DATASET_VOLUME,
    BATCH_SIZE,
    VECTOR_SIZE,
    COLLECTION_NAME,
)

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


def start_qdrant_container(
    name: str = "qdrant_autostart",
    host_port: int = QDRANT_PORT,
    grpc_port: int = QDRANT_PORT + 1,
    volume: str = DATASET_VOLUME,
):
    logger.info("Qdrant 서버가 꺼져 있어 컨테이너를 실행합니다.")
    subprocess.run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            name,
            "-p",
            f"{host_port}:{QDRANT_PORT}",
            "-p",
            f"{grpc_port}:{QDRANT_PORT + 1}",
            "-v",
            f"{volume}:{DATASET_VOLUME}",
            "qdrant/qdrant",
        ],
        check=True,
    )
    logger.info(f"[QdrantStorage] Qdrant 컨테이너 실행 완료: {name}")
    for _ in range(10):
        if is_qdrant_running("localhost", host_port):
            logger.info("[QdrantStorage] Qdrant 서버가 성공적으로 실행되었습니다.")
            return
        time.sleep(1)
    logger.error("[QdrantStorage] Qdrant 서버가 시작되지 않았습니다.")
    raise RuntimeError("[QdrantStorage] Qdrant 서버가 시작되지 않았습니다.")


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

        Args:
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
                logger.info(
                    f"[QdrantStorage] Qdrant 서버가 꺼져 있어 컨테이너를 실행합니다."
                )
                start_qdrant_container(host, port)

            self.client = QdrantClient(host=host, port=port)
            logger.info(f"[QdrantStorage] 클라이언트 생성 완료: {host}/{port}")
            self._init_collection()
            logger.info(f"[QdrantStorage] 컬렉션 초기화 완료: {self.collection_name}")
            logger.info(f"클라이언트: {host}/{port}")
            logger.info(f"컬렉션: {self.collection_name}")
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
                logger.info(f"[QdrantStorage] 임베딩 생성 완료: {vector}")
                results = await asyncio.to_thread(
                    self.client.search,
                    collection_name=self.collection_name,
                    query_vector=vector,
                    limit=limit,
                )
                logger.info(f"[QdrantStorage] 검색 결과: {results}")
                return [
                    {"id": hit.id, "score": hit.score, "payload": hit.payload}
                    for hit in results
                ]
        except Exception as e:
            logger.error(f"[QdrantStorage] 벡터 검색 실패: {str(e)}")
            raise

    async def store_batch(
        self,
        documents: List[Document],
        embeddings: List[List[float]],
        start_idx: int,
    ) -> None:
        """배치 단위로 문서를 처리하고 Qdrant에 저장합니다.

        Args:
            documents (List[Document]): 처리할 Document 객체 리스트
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
                        "comment": doc.page_content,
                        "labels": doc.metadata["labels"],
                        "ID": doc.metadata["ID"],
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


async def main():
    from classify.embedding import OllamaEmbeddingService
    from config.settings import LABELS, SENTIMENT_MAP, COLLECTION_NAME

    dict_labels = {i: label for i, label in enumerate(LABELS)}

    storage = QdrantStorage(collection_name=COLLECTION_NAME)
    ret = await storage.search(
        query="나가 죽어", embedding_constructor=OllamaEmbeddingService, limit=10
    )

    results = {"긍정": 0, "부정": 0, "중립": 0}

    print(ret[0].keys())
    for i, r in enumerate(ret):
        print(f"{i + 1}: {r['id']} : {r['payload']['comment']}, {r['score']}")
        for label in r["payload"]["labels"]:
            print(f"    {label}: {dict_labels[label]}")
            results[SENTIMENT_MAP[dict_labels[label]]] += 1

    print(results)
    total = sum(results.values())
    for key, value in results.items():
        print(f"{key}: {value / total * 100:.2f}% ({value}개)")

    print(f"감정 분류: {max(results, key=results.get)}")


if __name__ == "__main__":
    asyncio.run(main())
