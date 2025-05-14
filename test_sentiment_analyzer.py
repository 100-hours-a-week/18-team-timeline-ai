"""
감정 분석 테스트 모듈

이 모듈은 Daum Vclip 검색과 YouTube 댓글 수집을 통한 감정 분석 테스트를 수행합니다.
주요 기능:
- Daum Vclip 검색
- YouTube 댓글 수집
- 감정 분석 및 통계
"""

import asyncio
import os
from dotenv import load_dotenv
from utils.logger import Logger
from vector_db.embeddings.embeddings import Embeddings
from vector_db.storage.vector_store import QdrantVectorStore
from vector_db.sentiment.sentiment_analyzer import SentimentAnalyzer
from scrapers.daum_vclip_searcher import DaumVclipSearcher
from scrapers.youtube_searcher import YouTubeCommentAsyncFetcher
from datasets import load_dataset

# 로거 설정
logger = Logger.get_logger("test_sentiment_analyzer")


def load_kote_dataset():
    """KOTE 데이터셋을 로드합니다."""
    try:
        ds = load_dataset("searle-j/kote", cache_dir=".dataset", trust_remote_code=True)
        logger.info(
            f"데이터셋 로드 완료: {len(ds['train'])} 학습 샘플, "
            f"{len(ds['test'])} 테스트 샘플"
        )
        return ds
    except Exception as e:
        logger.error(f"데이터셋 로드 실패: {str(e)}")
        raise


async def test_sentiment_analysis():
    """
    감정 분석 테스트 실행

    테스트 항목:
    1. Daum Vclip 검색
    2. YouTube 댓글 수집
    3. 감정 분석 및 통계
    """
    try:
        # API 키 설정
        load_dotenv()
        daum_api_key = os.getenv("REST_API_KEY")
        youtube_api_key = os.getenv("YOUTUBE_API_KEY")

        if not daum_api_key or not youtube_api_key:
            raise ValueError("API 키가 설정되지 않았습니다.")

        # 임베딩과 벡터 스토어 초기화
        embeddings = Embeddings()
        vector_store = QdrantVectorStore(
            collection_name="kote_comments", vector_size=1024
        )
        await vector_store.create_collection()
        dataset = load_kote_dataset()
        embeddings.store_dataset(dataset)
        # 감정 분석기 초기화
        analyzer = SentimentAnalyzer(vector_store, embeddings)

        # Daum Vclip 검색
        vclip_searcher = DaumVclipSearcher(daum_api_key)
        search_query = "윤석열 유튜브"
        logger.info(f"Daum Vclip 검색 시작: '{search_query}'")

        try:
            search_results = vclip_searcher.search(search_query)
            logger.info(f"검색 결과: {len(search_results)}개 영상 발견")
        except Exception as e:
            logger.error(f"Daum Vclip 검색 실패: {str(e)}")
            return

        # YouTube 댓글 수집
        comment_fetcher = YouTubeCommentAsyncFetcher(
            api_key=youtube_api_key, max_comments=10
        )

        try:
            comments = await comment_fetcher.search(search_results)
            logger.info(f"댓글 수집 완료: {len(comments)}개 댓글")
        except Exception as e:
            logger.error(f"YouTube 댓글 수집 실패: {str(e)}")
            return

        # 댓글과 감정 저장
        texts = [comment["comment"] for comment in comments]
        # 임시로 모든 댓글을 "중립"으로 설정
        sentiments = ["중립"] * len(texts)

        # 수집된 댓글들로 감정 분석 테스트
        logger.info("\n감정 분석 결과:")
        for comment in comments:
            try:
                main_sentiment, sentiment_stats = await analyzer.analyze_sentiment(
                    comment["comment"]
                )
                logger.info(f"\n댓글: {comment['comment'][:50]}...")
                logger.info(f"주요 감정: {main_sentiment}")
                logger.info("감정 통계:")
                for sentiment, ratio in sentiment_stats.items():
                    logger.info(f"{sentiment}: {ratio:.2%}")
            except Exception as e:
                logger.error(f"감정 분석 실패 ({comment['comment'][:30]}...): {str(e)}")

        logger.info("\n모든 테스트가 성공적으로 완료되었습니다.")

    except Exception as e:
        logger.error(f"테스트 중 오류 발생: {str(e)}")
        raise


if __name__ == "__main__":
    # 테스트 실행
    asyncio.run(test_sentiment_analysis())
