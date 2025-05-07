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

logging.basicConfig(
    level=logging.INFO,  # ← 이 부분이 핵심
    format="%(asctime)s - %(levelname)s - %(message)s",
)


async def main():
    MODEL = "bge-m3:latest"
    dotenv.load_dotenv(override=True)
    YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
    REST_API_KEY = os.getenv("REST_API_KEY")
    video_searcher = DaumVclipSearcher(api_key=REST_API_KEY)
    youtube_searcher = YouTubeCommentAsyncFetcher(api_key=YOUTUBE_API_KEY)
    df = video_searcher.search("첼시 vs 리버풀")
    ripple = await youtube_searcher.search(df=df)
    return ripple


def test_agentic_comment_graph():
    # ✅ 서버 및 모델 정보
    SERVER = "https://c654-34-143-254-151.ngrok-free.app"
    MODEL = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"
    loop = asyncio.get_event_loop()
    data = loop.run_until_complete(main())
    pprint(data)
    # ✅ 그래프 빌드 및 실행
    graph = ClassifyGraph(server=SERVER, model=MODEL).build()
    runner = Runner(graph=graph)
    texts = [{"input_text": d["comment"], "transcript": d["captions"]} for d in data]
    result = runner.run(texts=texts)

    # ✅ 결과 출력
    print("=== 최종 결과 ===")
    pprint(result)


if __name__ == "__main__":
    test_agentic_comment_graph()
