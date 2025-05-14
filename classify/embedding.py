"""
임베딩 모듈

이 모듈은 텍스트를 벡터로 변환하는 기능을 제공합니다.
주요 기능:
- 텍스트 임베딩
- 배치 임베딩
"""

from typing import List
from abc import ABC, abstractmethod
from utils.logger import Logger
import aiohttp
import asyncio
from classify.storage import QdrantStorage
from langchain.schema import Document
from datasets import concatenate_datasets, load_dataset
from config.settings import (
    DATASET_NAME,
    DATASET_CACHE_DIR,
    COLLECTION_NAME,
    QDRANT_HOST,
    QDRANT_PORT,
    BATCH_SIZE,
    OLLAMA_HOST,
    OLLAMA_MODEL,
)

logger = Logger.get_logger("embedding")


class EmbeddingModel(ABC):
    """임베딩 모델 추상 클래스

    텍스트를 벡터로 변환하는 모델의 기본 인터페이스를 정의합니다.
    """

    @abstractmethod
    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """여러 문서를 임베딩합니다.

        Args:
            texts (List[str]): 임베딩할 텍스트 리스트

        Returns:
            List[List[float]]: 임베딩된 벡터 리스트
        """
        pass

    @abstractmethod
    async def embed_query(self, text: str) -> List[float]:
        """단일 쿼리를 임베딩합니다.

        Args:
            text (str): 임베딩할 텍스트

        Returns:
            List[float]: 임베딩된 벡터
        """
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
        async with self.session.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
        ) as response:
            if response.status == 200:
                data = await response.json()
                return data["embedding"]
            error_text = await response.text()
            raise Exception(f"임베딩 생성 실패: {error_text}")

    async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """문서 리스트를 배치 단위로 임베딩합니다.

        Args:
            texts (List[str]): 임베딩할 텍스트 리스트

        Returns:
            List[List[float]]: 임베딩된 벡터 리스트
        """

        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            tasks = [self._make_embedding_request(text) for text in batch]
            embeddings = await asyncio.gather(*tasks)
            all_embeddings.extend(embeddings)
            logger.info(f"배치 처리 완료: {i + len(batch)}/{len(texts)}")
        return all_embeddings

    async def embed_query(self, text: str) -> List[float]:
        """단일 쿼리를 임베딩합니다.

        Args:
            text (str): 임베딩할 텍스트

        Returns:
            List[float]: 임베딩된 벡터
        """
        return await self._make_embedding_request(text)


def create_documents(dataset) -> List[Document]:
    """데이터셋을 Document 객체 리스트로 변환합니다."""
    documents = []
    logger.info(f"dataset: {type(dataset)}")
    print(dataset[0])
    for iter in dataset:
        doc = Document(
            page_content=iter["text"],
            metadata={"labels": iter["labels"], "ID": iter["ID"]},
        )
        logger.debug(f"문서 생성 완료: {iter['ID']} - {iter['text']}")
        logger.debug(f"문서 레이블: {iter['labels']}")
        documents.append(doc)

    logger.info(f"총 {len(documents)}개의 문서 생성 완료")
    return documents


def load_kote_dataset(
    dataset_name: str = DATASET_NAME, cache_dir: str = DATASET_CACHE_DIR
):
    """KOTE 데이터셋을 로드합니다."""
    try:
        ds = load_dataset(dataset_name, cache_dir=cache_dir, trust_remote_code=True)
        # print(ds.keys())
        logger.info(
            f"데이터셋 로드 완료: {len(ds['train'])+ len(ds['test']) + len(ds['validation'])} 샘플"
        )
        return ds
    except Exception as e:
        logger.error(f"데이터셋 로드 실패: {str(e)}")
        raise


async def main(dataset):

    storage = QdrantStorage(
        collection_name=COLLECTION_NAME,
        host=QDRANT_HOST,
        port=QDRANT_PORT,
    )
    async with OllamaEmbeddingService() as embedder:
        dataset = concatenate_datasets(
            [dataset["train"], dataset["test"], dataset["validation"]]
        )
        documents = create_documents(dataset)
        for i in range(0, len(documents), BATCH_SIZE):
            # 현재 배치의 문서 추출
            batch_docs = documents[i : i + BATCH_SIZE]
            # 배치 내 문서들의 텍스트만 추출
            texts = [doc.page_content for doc in batch_docs]

            ret = await embedder.embed_documents(texts=texts)
            # 생성된 임베딩을 Qdrant에 즉시 저장 (비동기)
            await storage.store_batch(batch_docs, ret, i)

            # 진행 상황 로깅
            logger.info(
                f"배치 처리 완료: {i + len(batch_docs)}/{len(documents)} "
                f"({(i + len(batch_docs))/len(documents)*100:.1f}%)"
            )


if __name__ == "__main__":

    dataset = load_kote_dataset()
    asyncio.run(main(dataset))
