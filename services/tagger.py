import asyncio
from typing import Any, List, Dict
from utils.logger import Logger
from utils.storage import QdrantStorage
from contextlib import AsyncExitStack

logger = Logger.get_logger("tag_classifier")

TAG_LABELS = {
    1: "경제",
    2: "연예",
    3: "스포츠",
}
THRESHOLD = 0.75  # 코사인 유사도 기준 (0~1)

class TagClassifier:
    def __init__(
        self,
        embedder: Any,
        collection_name: str = "tag",  # Qdrant에 저장된 collection 이름
        label_map: Dict[int, str] = TAG_LABELS,
        threshold: float = THRESHOLD,
    ):
        self.embedder = embedder
        self.collection_name = collection_name
        self.label_map = label_map
        self.threshold = threshold
        self._storage = None

        logger.info(
            f"[TagClassifier] 초기화 완료 - 컬렉션: {collection_name}, 레이블: {label_map}"
        )

    async def __aenter__(self):
        self._stack = AsyncExitStack()
        self._storage = await self._stack.enter_async_context(
            QdrantStorage(collection_name=self.collection_name)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._stack.aclose()
        self._storage = None

    async def classify(self, title: str) -> int:
        """뉴스 제목을 태그 분류합니다.

        Returns:
            int: 0(기타), 1(경제), 2(연예), 3(스포츠)w
        """
        if not title:
            raise ValueError("title is required")

        if self._storage is None:
            raise RuntimeError("Storage not initialized")

        logger.info(f"[TagClassifier] 태그 분류 시작 - 입력 제목: {title}")

        try:
            # Qdrant 유사도 검색
            results = await self._storage.search(
                query=title,
                embedder=self.embedder,
                limit=10,
            )

            if not results:
                logger.info("[TagClassifier] 유사 결과 없음 - 기타(0) 반환")
                return 0

            label_scores = {label: 0.0 for label in self.label_map}
            for r in results:
                score = r.get("score", 0.0)
                if score < self.threshold:
                    continue
                for label in r["payload"]["labels"]:
                    if label in label_scores:
                        label_scores[label] += score

            if not any(score > 0 for score in label_scores.values()):
                logger.info("[TagClassifier] 임계값 이상 레이블 없음 - 기타(0) 반환")
                return 0

            best_label = max(label_scores, key=label_scores.get)
            logger.info(f"[TagClassifier] 최종 레이블: {self.label_map[best_label]}({best_label})")
            return best_label

        except Exception as e:
            logger.error(f"[TagClassifier] 분류 실패: {str(e)}")
            raise
