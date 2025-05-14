"""
감정 분석 집계 모듈

이 모듈은 유사 댓글들의 감정을 집계하여 최종 감정을 결정합니다.
주요 기능:
- 유사 댓글들의 감정 집계
- 감정 비율 계산
- 주요 감정 결정
"""

import os
import dotenv
import asyncio
from typing import Callable, List, Dict
from utils.logger import Logger
from config.settings import LABELS, SENTIMENT_MAP, COLLECTION_NAME, DICT_LABELS
from classify.storage import QdrantStorage
from classify.embedding import OllamaEmbeddingService
from scrapers.daum_vclip_searcher import DaumVclipSearcher
from scrapers.youtube_searcher import YouTubeCommentAsyncFetcher

logger = Logger.get_logger("sentiment_aggregator")


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
        if not query:
            raise ValueError("query is required")
        storage = QdrantStorage(collection_name=collection_name)
        ret = await storage.search(
            query=query, embedding_constructor=embedding_constructor, limit=10
        )
        try:
            dict_labels = DICT_LABELS
            results = {"긍정": 0.0, "부정": 0.0, "중립": 0.0}
            for i, r in enumerate(ret):
                tmp = {"긍정": 0, "부정": 0, "중립": 0}
                logger.info(
                    f"{i + 1}: {r['id']} : {r['payload']['comment']}, {r['score']}"
                )
                for label in r["payload"]["labels"]:
                    logger.info(f"    {label}: {dict_labels[label]}")
                    tmp[SENTIMENT_MAP[dict_labels[label]]] += 1
                for key, value in tmp.items():
                    results[key] += value / len(ret)
        except Exception as e:
            logger.error(f"감정 집계 실패: {str(e)}")
            raise
        logger.info(f"최종 감정: {ret}")
        return results

    async def aggregate_multiple_queries(
        self,
        queries: List[str],
        embedding_constructor: Callable = OllamaEmbeddingService,
        collection_name: str = COLLECTION_NAME,
        labels: List[str] = LABELS,
        sentiment_map: Dict[str, str] = SENTIMENT_MAP,
    ):
        aggregator = SentimentAggregator()

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
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 실패한 작업이 있다면 로그
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"쿼리 '{queries[i]}' 실패: {str(result)}")
        ret = {"긍정": 0, "부정": 0, "중립": 0}
        for r in results:
            ret["긍정"] += r["긍정"]
            ret["부정"] += r["부정"]
            ret["중립"] += r["중립"]
        return ret


async def main():
    dotenv.load_dotenv(override=True)
    YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
    REST_API_KEY = os.getenv("REST_API_KEY")
    daum_vclip_searcher = DaumVclipSearcher(api_key=REST_API_KEY)
    youtube_searcher = YouTubeCommentAsyncFetcher(
        api_key=YOUTUBE_API_KEY, max_comments=10
    )
    df = daum_vclip_searcher.search(query="손흥민 유튜브")
    ripple = await youtube_searcher.search(df=df)
    ripple = [r["comment"] for r in ripple]
    aggregator = SentimentAggregator()
    ret = await aggregator.aggregate_multiple_queries(
        queries=ripple,
        embedding_constructor=OllamaEmbeddingService,
    )
    total = sum(ret.values())
    print(ret)
    for key, value in ret.items():
        print(f"{key}: {value / total * 100:.2f}% (점수: {value:.2f} )")


if __name__ == "__main__":
    asyncio.run(main())
