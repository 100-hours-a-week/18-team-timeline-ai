import dotenv
import os
import asyncio
from scrapers.daum_vclip_searcher import DaumVclipSearcher
from scrapers.youtube_searcher import YouTubeCommentAsyncFetcher
from ai_models.graph.classify import ClassifyGraph
from ai_models.runner import Runner
from pprint import pprint
import logging
import logging

"""
logging.basicConfig(
    level=logging.INFO,  # ← 이 부분이 핵심
    format="%(asctime)s - %(levelname)s - %(message)s",
)
"""
QUERY = "오타니 유튜브"


async def main():
    MODEL = "bge-m3:latest"
    dotenv.load_dotenv(override=True)
    YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
    REST_API_KEY = os.getenv("REST_API_KEY")
    video_searcher = DaumVclipSearcher(api_key=REST_API_KEY)
    youtube_searcher = YouTubeCommentAsyncFetcher(
        api_key=YOUTUBE_API_KEY, max_comments=100
    )
    df = video_searcher.search(QUERY)
    ripple = await youtube_searcher.search(df=df)
    return ripple


def test_agentic_comment_graph():
    # ✅ 서버 및 모델 정보
    SERVER = "http://35.216.120.155:8001"
    MODEL = "models/HyperCLOVAX-SEED-Text-Instruct-1.5B"
    loop = asyncio.get_event_loop()
    data = loop.run_until_complete(main())
    pprint(data)
    # ✅ 그래프 빌드 및 실행
    graph = ClassifyGraph(server=SERVER, model=MODEL).build()
    runner = Runner(graph=graph)
    texts = [
        {"input_text": d["comment"], "transcript": d["captions"], "query": QUERY[:-4]}
        for d in data
    ]
    result = runner.run(texts=texts)

    # ✅ 결과 출력
    print("=== 최종 결과 ===")
    pprint(result)

    # 감정 분석 결과 개수 세기
    sentiment_counts = {"긍정": 0, "부정": 0, "중립": 0}

    # 영문 키를 한글 키로 매핑

    for sentiment_dict in result:
        sentiment = sentiment_dict.get("emotion", "중립")  # 기본값은 중립
        sentiment_counts[sentiment] += 1

    total = len(result)
    print("\n감정 분석 결과:")
    for sentiment, count in sentiment_counts.items():
        percentage = (count / total) * 100
        print(f"{sentiment}: {count}개 ({percentage:.1f}%)")


if __name__ == "__main__":
    test_agentic_comment_graph()
