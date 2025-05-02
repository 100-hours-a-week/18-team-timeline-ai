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
    SERVER = "https://8acc-34-125-119-95.ngrok-free.app"
    MODEL = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"

    inputs = [
        "흥민 우승 쌉가능! 다들 인정하십니까? ㅋㅋ 우리의캡틴 우리의아들 사랑합니다",
        "웃는모습보는 순간넘행복했습니다 💘 제발하루빨리 발부상회복되길 기도드립니다 🙏",
        "앤지 축구로는 결승못간다. 오늘 전반처럼 롱볼축구로 결승가야한다.  😢 제자르 설치면.. 꼭 지거나 우승 못함.. 후반에 먹은 1골이 복선 같은데.. 우리 흥민이 납치설 일축하네 손흥민 트로피 올리면서 울고 웃는 모습 보면 진짜 죽어도 여한이 없다. "
        "겉옷 깔@쌈하네 조 하트 ♡ 쏘니  반갑다❤ 어째 옷이 꺼정하냐 표정하고  몸짓이 꺼벙이같네 좀 더 이쪽으로 와보세요❤ 회복잘해요 기다리고있어요❤ 당수가 아니라 태권도로 똥이냐 설사냐 진짜 너무 맞말이네 신영씨 너무 좋아해서 미안하기도 한데 어딘지 살짝 리춘수 느낌이... ㅋㅋㅋ ㅋ ㅋㅋ ㅋㅋㅋ ㅋㅋㅋ ㅋㅋ 😂😂😂😂 김신영최고ㅋㅋㅋ 과연........ㅋㅋ 쏘니 웃는모습 너무 예쁘다."
        "손흥민이 웃는 모습은 언제나 좋다.",
        "나도 웃는다"
        "쏘니 화이팅❤❤❤ ❤❤❤❤❤❤❤❤❤❤❤ 예~👍🏻쏘니 결승전에서 꼭 보자^^"
        "손흥민 너무 못한다.",
    ]

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
