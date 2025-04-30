# test_solid_summary_graph.py

from solid_summary_graph import SummarizationGraph, SummaryState


def test_summary_graph():
    SERVER = "http://localhost:8000"  # 예시 서버 주소 (ngrok 등 사용 가능)
    MODEL = "test-model"  # 예시 모델 이름 (실제 모델로 교체 필요)

    graph = SummarizationGraph(server=SERVER, model=MODEL).build()

    init_state = SummaryState(
        input_text="윤석열 대통령이 반도체 산업 지원 방안을 발표했습니다.",
        summary="",
        score=0,
        worker_id=1,
        retry_count=0,
        status="",
        title="",
        title_score=0,
        tag="",
    )

    result = graph.invoke(init_state)

    print("\n✅ 최종 결과:")
    for k, v in result.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    test_summary_graph()
