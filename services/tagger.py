from typing import Any, Dict
from utils.logger import Logger
from utils.storage import QdrantStorage
from contextlib import AsyncExitStack
from config.settings import TAG_LABELS, THRESHOLD, TAG_COLLECTION_NAME
import logging

logger = Logger.get_logger("tag_classifier", log_level=logging.INFO)


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
        logger.info(
            f"[TagClassifier] __aenter__ 진입 (collection: {self.collection_name})"
        )
        self._stack = AsyncExitStack()
        try:
            self._storage = await self._stack.enter_async_context(
                QdrantStorage(collection_name=self.collection_name)
            )
            logger.info(
                f"[TagClassifier] QdrantStorage 진입 성공 (collection: {self.collection_name})"
            )
            return self
        except Exception as e:
            logger.error(f"[TagClassifier] QdrantStorage 진입 실패: {str(e)}")
            try:
                await self._stack.aclose()
                logger.info(
                    f"[TagClassifier] __aenter__ 예외 발생 시 스택 정리 완료 (collection: {self.collection_name})"
                )
            finally:
                self._storage = None
                self._stack = None
                logger.info(
                    f"[TagClassifier] __aenter__ finally: 내부 상태 None 처리 (collection: {self.collection_name})"
                )
            raise

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        logger.info(
            f"[TagClassifier] __aexit__ 진입 (collection: {self.collection_name})"
        )
        try:
            if hasattr(self, "_stack") and self._stack is not None:
                await self._stack.aclose()
                logger.info(
                    f"[TagClassifier] __aexit__ 스택 정리 완료 (collection: {self.collection_name})"
                )
        except Exception as e:
            logger.error(f"[TagClassifier] 세션 종료 중 예외 발생: {str(e)}")
        finally:
            self._storage = None
            self._stack = None
            logger.info(
                f"[TagClassifier] __aexit__ finally: 내부 상태 None 처리 (collection: {self.collection_name})"
            )

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
