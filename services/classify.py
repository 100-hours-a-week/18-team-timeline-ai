import asyncio
import aiohttp
from typing import Callable, List, Dict, Any
from util.logger import Logger
from config.settings import LABELS, SENTIMENT_MAP, COLLECTION_NAME, DICT_LABELS
from util.storage import QdrantStorage
from inference.embedding import OllamaEmbeddingService
from scrapers.daum_vclip_searcher import DaumVclipSearcher
from scrapers.youtube_searcher import YouTubeCommentAsyncFetcher
from config.prompts import SystemRole
from util.handling import handle_http_error
import time
import logging

logger = Logger.get_logger("sentiment_aggregator")


class SentimentAggregator:
    """감정 분석 집계 클래스

    유사 댓글들의 감정을 집계하여 최종 감정을 결정합니다.
    """

    def __init__(
        self,
        embedding_constructor: Callable = OllamaEmbeddingService,
        collection_name: str = COLLECTION_NAME,
        labels: List[str] = LABELS,
        sentiment_map: Dict[str, str] = SENTIMENT_MAP,
    ):
        """SentimentAggregator 초기화

        Args:
            embedding_constructor: 임베딩 생성 함수
            collection_name: 컬렉션 이름
            labels: 레이블 목록
            sentiment_map: 감정 맵
        """
        self.embedding_constructor = embedding_constructor
        self.collection_name = collection_name
        self.labels = labels
        self.sentiment_map = sentiment_map
        self.storage = QdrantStorage(collection_name=collection_name)

        logger.info(
            f"[SentimentAggregator] 초기화 완료 - 컬렉션: {collection_name}, "
            f"레이블 수: {len(labels)}, 감정 유형: {list(sentiment_map.values())}"
        )

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

        logger.info(
            f"[SentimentAggregator] 감정 집계 시작 - 쿼리: {query}, "
            f"컬렉션: {self.collection_name}"
        )

        try:
            logger.info(f"[SentimentAggregator] 검색 시작 - 쿼리: {query}")

            ret = await self.storage.search(
                query=query, embedding_constructor=self.embedding_constructor, limit=100
            )

            logger.info(
                f"[SentimentAggregator] 검색 완료 - 결과 수: {len(ret)}, "
                f"쿼리: {query}"
            )

            results = {"긍정": 0.0, "부정": 0.0, "중립": 0.0}
            for i, r in enumerate(ret):
                tmp = {"긍정": 0, "부정": 0, "중립": 0}

                logger.info(
                    f"[SentimentAggregator] 댓글 처리 중 - {i + 1}/{len(ret)}, "
                    f"ID: {r['id']}, 점수: {r['score']}, "
                    f"레이블 수: {len(r['payload']['labels'])}"
                )

                for label in r["payload"]["labels"]:
                    sentiment = self.sentiment_map[DICT_LABELS[label]]
                    tmp[sentiment] += 1
                    logger.debug(
                        f"[SentimentAggregator] 레이블 처리 - 레이블: {label}, "
                        f"감정: {sentiment}, 카운트: {tmp[sentiment]}"
                    )

                total = sum(tmp.values())
                if total > 0:
                    for key, value in tmp.items():
                        results[key] += value / total * r["score"]

                logger.info(
                    f"[SentimentAggregator] 현재까지 감정 집계 - "
                    f"현재: {tmp}, 누적: {results}"
                )

        except Exception as e:
            logger.error(
                f"[SentimentAggregator] 감정 집계 실패 - 쿼리: {query}, "
                f"오류: {str(e)}"
            )
            raise

        logger.info(
            f"[SentimentAggregator] 감정 집계 완료 - 최종 결과: {results}, "
            f"쿼리: {query}"
        )
        return results

    async def aggregate_multiple_queries(self, queries: List[str]) -> Dict[str, float]:
        """여러 쿼리의 감정을 집계하여 최종 감정을 결정합니다.

        Args:
            queries: 검색 쿼리 목록

        Returns:
            Dict[str, float]: 감정 비율
        """
        logger.info(
            f"[SentimentAggregator] 다중 쿼리 집계 시작 - "
            f"쿼리 수: {len(queries)}, 쿼리 목록: {queries}"
        )

        tasks = [self.aggregate_sentiment(query=q) for q in queries]
        try:
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

        logger.info(f"[SentimentAggregator] 최종 감정 집계 완료 - " f"최종 결과: {ret}")
        return ret
