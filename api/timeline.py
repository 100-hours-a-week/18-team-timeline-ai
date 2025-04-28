import re
import json
import time
from typing import TypedDict, List

from langchain.chat_models import ChatOpenAI
from langchain.schema import BaseOutputParser
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.runnables import RunnableMap

# 🔧 vLLM 서버 2개 정의
llm_servers = [
    ChatOpenAI(
        base_url="http://localhost:8000/v1",
        api_key="not-needed",
        model="naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B",
        temperature=0.7,
    ),
    ChatOpenAI(
        base_url="http://localhost:8001/v1",
        api_key="not-needed",
        model="naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B",
        temperature=0.7,
    ),
]


def get_llm(idx: int) -> ChatOpenAI:
    return llm_servers[idx % len(llm_servers)]


# 🔡 입력/출력 상태 정의
class GraphState(TypedDict):
    input_text: str
    summary: str
    score: int
    worker_id: int


class SummaryScoreParser(BaseOutputParser):
    def parse(self, text: str):
        cleaned = re.sub(r"^```json\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
        return json.loads(cleaned)


# 📝 요약 노드
def summarize_node(state: GraphState) -> GraphState:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a Korean news summarization assistant. Respond ONLY with a 3-line summary.",
            ),
            ("human", "다음 뉴스 기사를 요약해줘:\n\n{input_text}"),
        ]
    )
    llm = get_llm(state["worker_id"])
    runnable = prompt | llm
    result = runnable.invoke({"input_text": state["input_text"]})
    return {**state, "summary": result.content}


# 📊 평가 노드
def evaluate_score_node(state: GraphState) -> GraphState:
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a news summary evaluator.\n"
                    "Evaluate how well the summary covers the original article.\n"
                    'Respond ONLY with JSON: {{ "summary": "...", "score": 0~100 }}'
                ),
            ),
            (
                "system",
                (
                    "- 90~100: Very accurate and concise with complete key info\n"
                    "- 70~89: Mostly good but missing minor info or slightly vague\n"
                    "- 50~69: Incomplete or unclear summary\n"
                    "- 0~49: Failed or misleading summary\n\n"
                    "Respond ONLY with a JSON object like:\n"
                    "No explanation or markdown code block allowed."
                ),
            ),
            ("human", "원문:\n{input_text}\n\n요약:\n{summary}"),
        ]
    )
    llm = get_llm(state["worker_id"])
    runnable = prompt | llm | SummaryScoreParser()
    result = runnable.invoke(
        {"input_text": state["input_text"], "summary": state["summary"]}
    )
    return {**state, "score": result["score"]}


# 🔁 LangGraph 생성
def build_graph():
    builder = StateGraph(GraphState)
    builder.add_node("summarize", summarize_node)
    builder.add_node("evaluate_score", evaluate_score_node)
    builder.set_entry_point("summarize")
    builder.add_edge("summarize", "evaluate_score")
    builder.add_edge("evaluate_score", END)
    return builder.compile()


# 🚀 실행 함수
def run_graph(graph, text: str, worker_id: int) -> GraphState:
    return graph.invoke({"input_text": text, "worker_id": worker_id})


# 🔧 메인 함수
def main():
    num = 100  # 실행할 총 작업 수
    graph = build_graph()
    results: List[GraphState] = []
    start = time.time()

    print(f"🚀 LangGraph 병렬 실행 시작 (총 {num}개)\n")

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = [
            executor.submit(run_graph, graph, f"{text}. {i}", i % len(llm_servers))
            for i in range(num)
        ]
        for i, future in enumerate(as_completed(futures), 1):
            try:
                result = future.result()
                results.append(result)
                print(f"✅ {i}/{num} 완료 | 점수: {result['score']}")
            except Exception as e:
                print(f"❌ {i}/{num} 실패: {e}")

    print(f"\n🎯 전체 완료: {len(results)}개")
    for i, res in enumerate(results, 1):
        print(f"\n📝 요약 {i}")
        print(json.dumps(res, indent=2, ensure_ascii=False))

    print(f"\n⏱️ 총 소요 시간: {time.time() - start:.2f}s")


if __name__ == "__main__":
    text = """
    대통령 권한대행인 국무총리가 헌법재판관을 지명한 이른바 ‘한덕수의 난’이 헌법재판소의 결정으로 8일 만에 진압됐다. 한덕수 대통령 권한대행은 윤석열 전 대통령의 최측근인 이완규 법제처장을 재판관으로 지명·임명하려 했지만 헌재는 한 권한대행의 모순과 궤변을 모두 배척했다.

    한 권한대행은 지난 14일 헌재에 낸 의견서에서 “(재판관 후보자와 관련해) 장차 공직에 임명하겠다는 의사를 표시한 ‘발표’일 뿐 ‘지명’에 해당하지 않는다”며 “(헌법소원의 대상인) 공권력 행사가 아니다”라고 주장했다. 헌법소원 대상이 아니므로 각하돼야 한다는 궤변이었다.

    헌재는 그러나 16일 결정문에서 “한 권한대행이 가까운 장래에 국회에 인사청문 요청안을 제출하는 등 후속 절차를 진행해 후보자를 재판관으로 임명할 것임이 확실히 예측된다고 볼 수 있고, 이 사건 헌법소원 심판의 종국결정 선고 전에 이 사건 후보자가 재판관으로 임명될 가능성도 존재한다”고 짚었다. ‘지명이 아닌 발표에 불과했다’는 황당한 주장을 물리친 셈이다.

    한 권한대행은 또 ‘헌재가 이미 대통령 탄핵 사건에서 포고령에 관한 판단을 했으므로 신청인이 주장하는 자기관련성은 거의 없다’고도 했다. 계엄 포고령 1호에 대한 헌법소원을 낸 김정환 변호사가 효력정지 가처분을 신청할 당사자 적격이 없다는 주장이었다. 그러나 헌재는 김 변호사의 당사자 적격을 인정하며 “(한 권한대행의 재판관 지명·임명으로) 신청인만이 아니라 계속 중인 헌법재판 사건의 모든 당사자들의 재판을 받을 권리가 침해될 수 있다”고 밝혔다. 한 권한대행의 재판관 지명으로 헌법재판 과정에서 피해를 볼 국민이 많다는 점을 강조한 것이다.

    헌재의 신뢰가 훼손될 수 있다는 문제의식도 이번 결정에 반영됐다. 헌재는 향후 헌법소원 본안에서 한 권한대행의 재판관 지명이 위헌으로 판단될 경우, 이미 임명된 재판관들의 결정에 대한 재심이 허용되지 않는다면 “헌법과 법률이 정한 자격과 절차에 의해 임명된 재판관이 아닌 사람에 의한 결정이 헌재의 결정으로서 효력을 가지게 돼 헌법재판의 규범력이 현저히 약화되고 헌법재판에 대한 신뢰가 크게 훼손될 것”이라고 밝혔다.

    전문가들은 헌재가 법적인 혼란을 줄이기 위한 결정을 내렸다고 짚었다. 헌법연구관 출신인 이황희 성균관대 법학전문대학원 교수는 “권한대행의 행위 자체가 위헌적이라고 평가받는 상황에서 이를 헌재가 저지했다는 측면에서 의미가 있다”고 말했다. 헌재 헌법연구부장 출신인 김승대 변호사는 “헌재는 이미 권한대행의 권한 행사에 대해 의구심을 갖고 있음을 표출한 셈”이라며 “본안에서도 한 대행의 주장이 받아들여질 가능성은 희박하다”고 말했다.

    헌재의 이번 결정으로 사실상 이완규·함상훈 후보자 지명은 철회될 가능성이 높다는 게 법조계의 중론이다. 이번 가처분 신청 사건의 당사자인 김정환 변호사는 “본안 판단 이전에 대통령선거가 치러질 가능성이 크며, 새로운 대통령이 지명을 철회하고 새로 재판관 2명을 임명할 수 있다”며 “사실상 (한 대행 지명 후보자들의 임명은) 끝났다고 보면 된다”고 말했다.
    """
    main()
