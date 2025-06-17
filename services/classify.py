import asyncio
from typing import Any, List, Dict
from config.settings import LABELS, SENTIMENT_MAP, COLLECTION_NAME, DICT_LABELS
from utils.logger import Logger
from utils.storage import QdrantStorage  # AsyncQdrantClient 기반으로 구현된 storage
from utils.handling import handle_http_error

logger = Logger.get_logger("sentiment_aggregator")


class SentimentAggregator:
    """감정 분석 집계 클래스

    유사 댓글들의 감정을 집계하여 최종 감정을 결정합니다.
    """

    def __init__(
        self,
        embedder: Any,
        collection_name: str = COLLECTION_NAME,
        labels: List[str] = LABELS,
        sentiment_map: Dict[str, str] = SENTIMENT_MAP,
    ):
        """SentimentAggregator 초기화

        Args:
            embedder: 임베딩 생성기
            collection_name: 컬렉션 이름
            labels: 레이블 목록
            sentiment_map: 감정 맵
        """
        self.embedder = embedder
        self.collection_name = collection_name
        self.labels = labels
        self.sentiment_map = sentiment_map
        self._storage = None

        logger.info(
            f"[SentimentAggregator] 초기화 완료 - 컬렉션: {collection_name}, "
            f"레이블 수: {len(labels)}, 감정 유형: {list(sentiment_map.values())}"
        )

    async def __aenter__(self):
        """비동기 컨텍스트 매니저 진입"""
        self._storage = QdrantStorage(collection_name=self.collection_name)
        await self._storage.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 매니저 종료"""
        if self._storage is not None:
            await self._storage.__aexit__(exc_type, exc_val, exc_tb)
            self._storage = None

    async def aggregate_sentiment(self, query: str) -> Dict[str, float]:
        """유사 댓글들의 감정을 집계하여 최종 감정을 결정합니다.

        Args:
            query: 검색 쿼리

        Raises:
            ValueError: 쿼리가 없을 경우

        Returns:
            Dict[str, float]: 감정 비율
        """
        if not query:
            logger.error("[SentimentAggregator] 쿼리가 비어있습니다")
            raise ValueError("query is required")

        if self._storage is None:
            raise RuntimeError(
                "[SentimentAggregator] 스토리지가 초기화되지 않았습니다."
            )

        logger.info(
            f"[SentimentAggregator] 감정 집계 시작 - 쿼리: {query}, "
            f"컬렉션: {self.collection_name}"
        )

        try:
            logger.info(f"[SentimentAggregator] 검색 시작 - 쿼리: {query}")

            ret = await self._storage.search(
                query=query, embedder=self.embedder, limit=100
            )

            logger.info(
                f"[SentimentAggregator] 검색 완료 - 결과 수: {len(ret)}, "
                f"쿼리: {query}"
            )

            results = {"긍정": 0.0, "부정": 0.0, "중립": 0.0}
            for i, r in enumerate(ret):
                tmp = {"긍정": 0, "부정": 0, "중립": 0}
                for label in r["payload"]["labels"]:
                    sentiment = self.sentiment_map.get(label, "중립")
                    tmp[sentiment] += 1

                # 가중치 적용 (상위 결과에 더 높은 가중치)
                weight = 1.0 / (i + 1)
                for sentiment in results:
                    results[sentiment] += tmp[sentiment] * weight

            # 정규화
            total = sum(results.values())
            if total > 0:
                for sentiment in results:
                    results[sentiment] = (results[sentiment] / total) * 100

            logger.info(
                f"[SentimentAggregator] 감정 집계 완료 - 결과: {results}, "
                f"쿼리: {query}"
            )
            return results

        except Exception as e:
            logger.error(
                f"[SentimentAggregator] 감정 집계 실패 - 쿼리: {query}, 오류: {str(e)}"
            )
            raise

    async def aggregate_multiple_queries(self, queries: List[str]) -> Dict[str, float]:
        logger.info(
            f"[SentimentAggregator] 다중 쿼리 집계 시작 - "
            f"쿼리 수: {len(queries)}, 쿼리 목록: {queries}"
        )

        try:
            tasks = [self.aggregate_sentiment(query=q) for q in queries]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            logger.info(
                f"[SentimentAggregator] 다중 쿼리 집계 완료 - "
                f"결과 수: {len(results)}, 쿼리 수: {len(queries)}"
            )
        except Exception as e:
            logger.error(
                f"[SentimentAggregator] 다중 쿼리 집계 실패 - "
                f"쿼리 수: {len(queries)}, 오류: {str(e)}"
            )
            raise

        ret = {"긍정": 0.0, "부정": 0.0, "중립": 0.0}
        for i, result in enumerate(results):
            if not await handle_http_error(result, queries[i], logger):
                logger.warning(
                    f"[SentimentAggregator] 쿼리 처리 실패 - "
                    f"쿼리: {queries[i]}, 인덱스: {i}"
                )
                continue

            for sentiment in ret:
                ret[sentiment] += result[sentiment]

            logger.info(
                f"[SentimentAggregator] 쿼리 처리 완료 - "
                f"쿼리: {queries[i]}, 현재 감정: {result}, "
                f"누적 감정: {ret}"
            )

        logger.info(f"[SentimentAggregator] 최종 감정 집계 완료 - 최종 결과: {ret}")
        return ret
