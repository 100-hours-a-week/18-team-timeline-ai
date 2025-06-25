from typing import Any, Dict
from utils.logger import Logger
from utils.storage import QdrantStorage
from contextlib import AsyncExitStack
from config.settings import TAG_LABELS, THRESHOLD, TAG_COLLECTION_NAME

logger = Logger.get_logger("tag_classifier")


class TagClassifier:
    def __init__(
        self,
        embedder: Any,
        collection_name: str = TAG_COLLECTION_NAME,
        threshold: float = THRESHOLD,
        label_map: Dict[int, str] = TAG_LABELS,
    ):
        self.embedder = embedder
        self.collection_name = collection_name
        self.threshold = threshold
        self.label_map = label_map
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
        """
        뉴스 제목을 태그 분류합니다.

        Returns:
            int: 0(기타), 1(경제), 2(연예), 3(스포츠)
        """
        if not title:
            raise ValueError("title is required")

        if self._storage is None:
            raise RuntimeError("Storage not initialized")

        logger.info(f"[TagClassifier] 태그 분류 시작 - 입력 제목: {title}")

        try:
            results = await self._storage.search(
                query=title,
                embedder=self.embedder,
                limit=10,
            )

            if not results:
                logger.info("[TagClassifier] 유사 결과 없음 - 기타(0) 반환")
                return 0

            label_scores = {label: 0.0 for label in self.label_map.keys()}

            for r in results:
                score = r.get("score", 0.0)
                if score < self.threshold:
                    continue

                label = r["payload"].get("labels")
                if isinstance(label, int) and label in label_scores:
                    label_scores[label] += score

            if not any(score > 0 for score in label_scores.values()):
                logger.info("[TagClassifier] 임계값 이상 레이블 없음 - 기타(0) 반환")
                return 0

            best_label = max(label_scores.keys(), key=lambda k: label_scores[k])
            logger.info(
                f"[TagClassifier] 최종 태그: {self.label_map.get(best_label, '기타')} ({best_label})"
            )
            return best_label

        except Exception as e:
            logger.error(f"[TagClassifier] 분류 실패: {str(e)}")
            raise
