from typing import List
from langgraph.graph import StateGraph, END, START, Graph
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

from textwrap import dedent
from langchain.schema import BaseOutputParser
import json
import re


class SummaryState(dict):
    input_text: str
    summary: str
    score: int
    worker_id: int
    retry_count: int
    status: str


class SummaryScoreParser(BaseOutputParser):
    def parse(self, text: str):
        cleaned = re.sub(r"^```json\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            cleaned = cleaned.replace('}"', '", "').replace("}{", "},{")
            print(f"⚠️ JSON 파싱 에러 발생: {e}")
            print(f"⚠️ 문제의 원본 텍스트:\n{text}\n")
            return json.loads(cleaned)


examples = [
    HumanMessage(
        """
    도널드 트럼프 미국 대통령은 반도체를 비롯한 전자제품에도 관세를 부과하겠다는 입장을 재확인하며 관세 정책에 후퇴가 없음을 시사했다.
    트럼프 대통령은 13일(현지시간) 자신의 사회관계망서비스(SNS)에 올린 글에서 "지난 금요일(4월 11일)에 발표한 것은 관세 예외(exception)가 아니다. 이들 제품은 기존 20% 펜타닐 관세를 적용받고 있으며 단지 다른 관세 범주(bucket)로 옮기는 것"이라고 밝혔다.
    이어 "우리는 다가오는 국가 안보 관세 조사에서 반도체와 전자제품 공급망 전체를 들여다볼 것"이라고 말했다.
    앞서 트럼프 대통령은 지난 11일 대통령 각서에서 상호관세에서 제외되는 반도체 등 전자제품 품목을 구체적으로 명시했고, 관세 징수를 담당하는 세관국경보호국(CBP)이 같은 날 이를 공지했다.
    이에 따라 반도체 등 전자제품은 미국이 중국에 부과한 125% 상호관세, 그리고 한국을 비롯한 나머지 국가에 부과한 상호관세(트럼프 대통령의 유예 조치로 7월 8일까지는 10% 기본관세만 적용)를 내지 않아도 된다.
    다만 미국이 마약성 진통제인 펜타닐의 미국 유입 차단에 협조하지 않는다는 이유로 중국에 별도 행정명령을 통해 부과한 20% 관세는 여전히 적용받는다.
    이를 두고 미국 언론과 업계에서는 트럼프 대통령이 강경 기조에서 한발 물러나 전자제품은 아예 관세에서 면제하는 게 아니냐는 관측이 제기됐으며, 민주당 등에서는 정책에 일관성이 없다고 비판했다.
    그러자 관세를 담당하는 트럼프 행정부 당국자들은 이날 방송에 출연해 반도체 등 전자제품은 지난 2일 발표한 국가별 상호관세에서 제외될 뿐 앞으로 진행할 '무역확장법 232조' 조사를 통해 관세를 부과할 방침이라고 설명했다.
    반도체 등 국가 안보에 중요한 품목은 앞서 25% 관세를 부과한 철강이나 자동차와 마찬가지로 상호관세와 중첩되지 않는 품목별 관세를 부과하겠다는 것이다.
    트럼프 대통령도 이날 관세 강행 의지를 피력했다.
    그는 "다른 나라들이 우리를 상대로 이용한 비(非)금전적 관세 장벽 및 불공정한 무역수지와 관련해 누구도 봐주지 않겠다(Nobody is getting off the hook). 
    특히 우리를 최악으로 대우하는 중국은 봐주지 않겠다"고 밝혔다.
    트럼프 대통령은 "우리는 제품을 미국에서 만들어야 하며 우리는 다른 나라에 인질로 잡히지 않을 것이다. 특히 중국같이 미국민을 무시하기 위해 가진 모든 권력을 이용할 적대적인 교역국에 대해 그렇다"라고 강조했다.
    """
    ),
    AIMessage(
        dedent(
            """

            트럼프 대통령은 관세 정책에 변화가 없음을 재확인하며, 반도체 및 전자제품에 대한 기존 20% 펜타닐 관세를 유지하고 국가 안보 조사 시 반도체와 전자제품 공급망을 검토할 것임을 발표하였습니다. 
            이는 중국에 부과된 상호관세의 일부 면제에도 불구한 조치입니다. 
            트럼프 대통령은 모든 국가, 특히 미국을 최악으로 대우하는 중국에 대해 관세 정책을 계속 추진할 것임을 강조했습니다.
            """
        )
    ),
]


class SummarizationGraph:
    def __init__(
        self, server: str, model: str, examples: List = examples, max_retries: int = 3
    ):
        self.server = server
        self.model = model
        self.examples = examples
        self.max_retries = max_retries

    def _make_llm(self):
        return ChatOpenAI(
            base_url=f"{self.server}/v1",
            api_key="not-needed",
            model=self.model,
            temperature=0.3,
        )

    def _make_summarize_node(self, llm):
        def summarize(state: SummaryState) -> SummaryState:
            system_prompt = """
            당신은 뉴스 요약 전문가입니다.
            - 3줄 이내, 완결된 문장, 핵심 사실만 예시를 바탕으로 요약문만을 제시하세요.
            - 예측, 해석, 사견은 금지합니다.
            """
            prompt = ChatPromptTemplate.from_messages(
                [("system", system_prompt), *self.examples, ("human", "{input_text}")]
            )
            runnable = prompt | llm
            result = runnable.invoke({"input_text": state["input_text"]})
            state["summary"] = result.content
            return state

        return summarize

    def _make_evaluate_node(self, llm):
        response_schemas = [
            ResponseSchema(name="summary", description="요약된 문장. 완전한 문장이어야 함."),
            ResponseSchema(name="score", description="요약 품질 점수 (0~100 사이의 정수)"),
        ]
        parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = parser.get_format_instructions()
        def evaluate(state: SummaryState) -> SummaryState:
            eval_prompt = ChatPromptTemplate(
                messages=[
                    SystemMessagePromptTemplate.from_template(
                        f"""
                        당신은 뉴스 요약 평가자입니다.
                        다음 기준에 따라 채점하고, 반드시 다음 포맷에 맞춰 답변하세요.
                        
                        {format_instructions}
                        
                        - 90~100: 문장에 의견이 들어가지 않고 문법 상 어색함이 없으며 문장이 3줄 이하이며 핵심 사실을 정확히 요약함.
                        - 70~89: 3줄 이내이고 대체로 좋음 (약간의 어색함이나 불명확한 부분이 있을 수 있음)
                        - 50~69: 3줄 이상이며 불완전 (핵심 누락 또는 문법적 문제가 존재함)
                        - 0~49: 실패 (요약이 원문과 거의 무관하거나 문법이 심각하게 어색함)
                    """
                    ),
                    HumanMessagePromptTemplate.from_template(
                        "원문:\n{input_text}\n\n요약:\n{summary}"
                    ),
                ],
                input_variables=["input_text", "summary"],
            )
            runnable = eval_prompt | llm | parser()
            try:
                result = runnable.invoke(
                    {
                        "input_text": state["input_text"],
                        "summary": state["summary"],
                    }
                )
                state["score"] = result["score"]
            except Exception as e:
                print(e)
                state["score"] = 0
            return state

        return evaluate

    def _make_retry_node(self):
        def retry(state: SummaryState) -> SummaryState:
            state["retry_count"] = state.get("retry_count", 0) + 1
            return state

        return retry

    def _make_log_fail_node(self):
        def log_fail(state: SummaryState) -> SummaryState:
            score = state.get("score", 0)
            state["fail_reason"] = "품질 문제" if score < 50 else "시스템 오류"
            return state

        return log_fail

    def _make_save_node(self):
        def save(state: SummaryState) -> SummaryState:
            state["status"] = "saved"
            return state

        return save

    def _make_check_score(self):
        def check(state: SummaryState) -> str:
            score = state.get("score", 0)
            retries = state.get("retry_count", 0)
            if score >= 85:
                return "save"
            elif retries < self.max_retries:
                return "retry"
            else:
                return "log_fail"

        return check

    def build(self):
        llm = self._make_llm()
        graph = StateGraph(SummaryState)

        graph.add_node("summarize", self._make_summarize_node(llm))
        graph.add_node("evaluate", self._make_evaluate_node(llm))
        graph.add_node("retry", self._make_retry_node())
        graph.add_node("log_fail", self._make_log_fail_node())
        graph.add_node("save", self._make_save_node())

        graph.add_edge(START, "summarize")
        graph.add_edge("summarize", "evaluate")
        graph.add_conditional_edges(
            "evaluate",
            self._make_check_score(),
            {
                "save": "save",
                "retry": "retry",
                "log_fail": "log_fail",
            },
        )
        graph.add_edge("retry", "summarize")
        graph.add_edge("save", END)
        graph.add_edge("log_fail", END)
        return graph.compile()
