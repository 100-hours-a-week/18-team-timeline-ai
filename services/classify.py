import os
import dotenv
import asyncio
from typing import Callable, List, Dict
from utils.logger import Logger
from config.settings import LABELS, SENTIMENT_MAP, COLLECTION_NAME, DICT_LABELS
from utils.storage import QdrantStorage
from inference.embedding import OllamaEmbeddingService
from scrapers.daum_vclip_searcher import DaumVclipSearcher
from scrapers.youtube_searcher import YouTubeCommentAsyncFetcher
from utils.handling import handle_http_error
import time
import logging

logger = Logger.get_logger("sentiment_aggregator", log_level=logging.ERROR)


class SentimentAggregator:
    """감정 분석 집계 클래스

    유사 댓글들의 감정을 집계하여 최종 감정을 결정합니다.
    """

    def __init__(self):
        pass

    async def aggregate_sentiment(
        self,
        query: str,
        embedding_constructor: Callable = OllamaEmbeddingService,
        collection_name: str = COLLECTION_NAME,
        labels: List[str] = LABELS,
        sentiment_map: Dict[str, str] = SENTIMENT_MAP,
    ):
        """유사 댓글들의 감정을 집계하여 최종 감정을 결정합니다.

        Args:
            query (str): 검색 쿼리
            embedding_constructor (Callable, optional): 임베딩 생성 함수. Defaults to OllamaEmbeddingService.
            collection_name (str, optional): 컬렉션 이름. Defaults to COLLECTION_NAME.
            labels (List[str], optional): 레이블 목록. Defaults to LABELS.
            sentiment_map (Dict[str, str], optional): 감정 맵. Defaults to SENTIMENT_MAP.

        Raises:
            ValueError: 쿼리가 없을 경우

        Returns:
            Dict[str, float]: 감정 비율
        """
        if not query:
            logger.error("[SentimentAggregator] query is required")
            raise ValueError("query is required")
        logger.info(f"[SentimentAggregator] 감정 집계 시작: {query}")
        storage = QdrantStorage(collection_name=collection_name)

        try:
            logger.info(f"[SentimentAggregator] 검색 시작: {query}")
            ret = await storage.search(
                query=query, embedding_constructor=embedding_constructor, limit=100
            )
            logger.info(f"[SentimentAggregator] 검색 완료: {len(ret)}개")
            dict_labels = DICT_LABELS
            results = {"긍정": 0.0, "부정": 0.0, "중립": 0.0}
            for i, r in enumerate(ret):
                tmp = {"긍정": 0, "부정": 0, "중립": 0}
                logger.info(
                    f"[SentimentAggregator] {i + 1}] {r['id']} : {r['payload']['comment']}, {r['score']}, {len(r['payload']['labels'])}"
                )
                for label in r["payload"]["labels"]:
                    tmp[SENTIMENT_MAP[dict_labels[label]]] += 1
                logger.info(f"[SentimentAggregator] 현재 감정: {tmp}")
                total = sum(tmp.values())
                for key, value in tmp.items():
                    results[key] = value / total * r["score"]
                logger.info(f"[SentimentAggregator] 현재 감정: {results}")
        except Exception as e:
            logger.error(f"[SentimentAggregator] 감정 집계 실패: {str(e)}")
            raise
        logger.info(f"[SentimentAggregator] 최종 감정: {results}")
        return results

    async def aggregate_multiple_queries(
        self,
        queries: List[str],
        embedding_constructor: Callable = OllamaEmbeddingService,
        collection_name: str = COLLECTION_NAME,
        labels: List[str] = LABELS,
        sentiment_map: Dict[str, str] = SENTIMENT_MAP,
    ):
        """여러 쿼리의 감정을 집계하여 최종 감정을 결정합니다.

        Args:
            queries (List[str]): 검색 쿼리 목록
            embedding_constructor (Callable, optional): 임베딩 생성 함수. Defaults to OllamaEmbeddingService.
            collection_name (str, optional): 컬렉션 이름. Defaults to COLLECTION_NAME.
            labels (List[str], optional): 레이블 목록. Defaults to LABELS.
            sentiment_map (Dict[str, str], optional): 감정 맵. Defaults to SENTIMENT_MAP.

        Returns:
            Dict[str, float]: 감정 비율
        """

        aggregator = SentimentAggregator()
        logger.info(f"[SentimentAggregator] 감정 분류 객체 생성: {queries}")
        tasks = [
            aggregator.aggregate_sentiment(
                query=q,
                embedding_constructor=embedding_constructor,
                collection_name=collection_name,
                labels=labels,
                sentiment_map=sentiment_map,
            )
            for q in queries
        ]
        # 비동기 실행
        try:
            logger.info(f"[SentimentAggregator] 다중 쿼리 집계 시작: {queries}")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            logger.info(f"[SentimentAggregator] 다중 쿼리 집계 완료: {len(results)}개")
        except Exception as e:
            logger.error(f"[SentimentAggregator] 다중 쿼리 집계 실패: {str(e)}")
            raise
        # 실패한 작업이 있다면 로그
        ret = {"긍정": 0, "부정": 0, "중립": 0}
        for i, result in enumerate(results):
            if not await handle_http_error(result, queries[i], logger):
                continue
            ret["긍정"] += result["긍정"]
            ret["부정"] += result["부정"]
            ret["중립"] += result["중립"]
        logger.info(f"[SentimentAggregator] 최종 감정: {ret}")
        return ret
