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
from pprint import pprint
from textwrap import dedent
from langchain.schema import BaseOutputParser
import json
import re
import logging

# 로그 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
            temperature=0.1,
        )

    def _make_summarize_node(self, llm):
        summary_schema = [
            ResponseSchema(
                name="summary",
                description="요약된 3줄 이내의 예측, 해석, 사견이 없는 완전한 문장",
            )
        ]
        parser = StructuredOutputParser.from_response_schemas(summary_schema)

        # \'{{\"summary\": "...",}}\'
        def summarize(state: SummaryState) -> SummaryState:
            system_prompt = """
            당신은 뉴스 요약 전문가입니다. 뉴스를 요약해주세요.
            - 3줄 이내, 완결된 문장, 핵심 사실만 요약만을 제시하세요.
            - 예시의 형식을 따라서 반드시 JSON으로 제공하세요.
            \'{{\'summary\': \'요약\'}}\'
            - 예측, 해석, 사견은 금지합니다.
            """
            prompt = ChatPromptTemplate.from_messages(
                [("system", system_prompt), ("human", "{input_text}")]
            )
            runnable = prompt | llm | parser
            try:
                """
                TODO
                파싱 성능 올리기
                """
                result = runnable.invoke({"input_text": state["input_text"]})
                state["summary"] = result["summary"]
                logger.info(f"✅ 요약 생성 완료: {result['summary']}")
            except Exception as e:
                logger.exception(f"❌ 요약 생성 실패: {e}")
                state["summary"] = ""
            return state

        return summarize

    def _make_evaluate_node(self, llm):
        response_schemas = [
            ResponseSchema(
                name="score", description="요약 품질 점수 (0~100 사이의 정수)"
            ),
        ]
        parser = StructuredOutputParser.from_response_schemas(response_schemas)

        def evaluate(state: SummaryState) -> SummaryState:
            eval_prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessagePromptTemplate.from_template(
                        """
                        당신은 뉴스 요약 평가자입니다.
                        다음 기준에 따라 채점하세요.
                        예시의 형식을 참고하여 반드시 JSON으로 작성하세요.
                        예시: \'{{\'score\': 75}}\'
                        - 90~100: 문장에 의견이 들어가지 않고 문법 상 어색함이 없으며 문장이 3줄 이하이며 핵심 사실을 정확히 요약함.
                        - 70~89: 3줄 이내이고 대체로 좋음 (약간의 어색함이나 불명확한 부분이 있을 수 있음)
                        - 50~69: 3줄 이상이며 불완전 (핵심 누락 또는 문법적 문제가 존재함)
                        - 0~49: 실패 (요약이 원문과 거의 무관하거나 문법이 심각하게 어색함)
                        """
                    ),
                    HumanMessagePromptTemplate.from_template(
                        "원문:\n{input_text}\n\n요약:\n{summary}"
                    ),
                ]
            )
            runnable = eval_prompt | llm | parser
            try:
                result = runnable.invoke(
                    {
                        "input_text": state["input_text"],
                        "summary": state["summary"],
                    }
                )
                state["score"] = result["score"]
                logger.info(f"평가 완료: {result['score']}")
            except Exception as e:
                logger.exception(f"평가 실패: {e}")
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
        app = graph.compile()
        # mermaid_code = app.get_graph().draw_mermaid()
        # print(mermaid_code)
        return app
