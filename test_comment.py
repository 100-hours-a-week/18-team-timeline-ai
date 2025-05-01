import dotenv
import os
import asyncio
from scrapers.daum_vclip_searcher import DaumVclipSearcher
from scrapers.youtube_searcher import YouTubeCommentAsyncFetcher


async def main():
    dotenv.load_dotenv(override=True)
    YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
    REST_API_KEY = os.getenv("REST_API_KEY")
    video_searcher = DaumVclipSearcher(api_key=REST_API_KEY)
    youtube_searcher = YouTubeCommentAsyncFetcher(api_key=YOUTUBE_API_KEY)
    df = video_searcher.search("손흥민")
    ripple = await youtube_searcher.search(df=df)
    print(*ripple)


from ai_models.graph.classify import AgenticCommentGraph


def test_agentic_comment_graph():
    # ✅ 서버 및 모델 정보
    SERVER = "https://01ed-35-197-152-206.ngrok-free.app"
    MODEL = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"

    # ✅ 테스트용 입력값
    input_state = {
        "input_text": "윤석열 대통령은 G7 정상회의에서 어떤 발언을 했나요?",
        "worker_id": 1001,
        "status": "init",
        "summary": "",
        "score": 0,
        "context_docs": [],
    }

    # ✅ 그래프 빌드 및 실행
    graph = AgenticCommentGraph(server=SERVER, model=MODEL).build()
    result = graph.invoke(input_state)

    # ✅ 결과 출력
    print("=== 최종 결과 ===")
    for k, v in result.items():
        print(f"{k}: {v}")

    # ✅ 간단한 검증
    assert "score" in result
    assert result["status"] == "done"
    assert isinstance(result["context_docs"], list)
    print("✅ 테스트 통과")


if __name__ == "__main__":
    # asyncio.run(main=main())
    test_agentic_comment_graph()
