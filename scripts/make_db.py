import asyncio
from typing import List, Any
import logging
from utils.logger import Logger
from dotenv import load_dotenv
import os
from config.settings import (
    DATASET_CACHE_DIR,
    DATASET_NAME,
    BATCH_SIZE,
    COLLECTION_NAME,
    QDRANT_PORT,
)
from utils.storage import QdrantStorage
from inference.embedding import OllamaEmbeddingService

logger = Logger.get_logger("dattset", log_level=logging.ERROR)
load_dotenv(override=True)
QDRANT_HOST = os.getenv("QDRANT_HOST")


def create_documents(dataset) -> List[dict[str, Any]]:
    """데이터셋을 Document 객체 리스트로 변환합니다."""
    documents = []
    logger.info(f"dataset: {type(dataset)}")
    try:
        for iter in dataset:
            doc = {
                "text": iter["text"],
                "metadata": {
                    "labels": iter["labels"],
                    "ID": iter["ID"],
                },
            }
            logger.debug(f"문서 생성 완료: {iter['ID']} - {iter['text']}")
            logger.debug(f"문서 레이블: {iter['labels']}")
            documents.append(doc)
    except Exception as e:
        logger.error(f"[create_documents] 문서 생성 실패: {e}")
        raise
    logger.info(f"총 {len(documents)}개의 문서 생성 완료")
    return documents


def load_kote_dataset(
    dataset_name: str = DATASET_NAME, cache_dir: str = DATASET_CACHE_DIR
):
    from datasets import load_dataset

    """KOTE 데이터셋을 로드합니다."""
    try:
        ds = load_dataset(dataset_name, cache_dir=cache_dir, trust_remote_code=True)
        total_samples = len(ds["train"]) + len(ds["test"]) + len(ds["validation"])
        logger.info(f"[load_kote_dataset] 데이터셋 로드 완료: {total_samples} 샘플")
        return ds
    except Exception as e:
        logger.error(f"[load_kote_dataset] 데이터셋 로드 실패: {str(e)}")
        raise


async def main(dataset):

    storage = QdrantStorage(
        collection_name=COLLECTION_NAME,
        host=QDRANT_HOST,
        port=QDRANT_PORT,
    )
    async with OllamaEmbeddingService() as embedder:
        from datasets import concatenate_datasets

        dataset = concatenate_datasets(
            [dataset["train"], dataset["test"], dataset["validation"]]
        )
        documents = create_documents(dataset)
        for i in range(0, len(documents), BATCH_SIZE):
            # 현재 배치의 문서 추출
            batch_docs = documents[i : i + BATCH_SIZE]
            # 배치 내 문서들의 텍스트만 추출
            texts = [doc["text"] for doc in batch_docs]

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
