from typing import List
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import BaseOutputParser
import json
import re


class GraphState(dict):
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
        except json.JSONDecodeError:
            cleaned = cleaned.replace('}"', '", "').replace("}{", "},{")
            return json.loads(cleaned)


class SummarizationGraph:
    def __init__(self, server: str, model: str, examples: List, max_retries: int = 3):
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
        def summarize(state: GraphState) -> GraphState:
            system_prompt = """
            당신은 뉴스 요약 전문가입니다.
            - 3줄 이내, 완결된 문장, 핵심 사실만 요약하세요.
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
        def evaluate(state: GraphState) -> GraphState:
            eval_prompt = ChatPromptTemplate(
                messages=[
                    SystemMessagePromptTemplate.from_template(
                        """
                        You are a strict JSON evaluator for news summaries.
                        Respond ONLY with JSON: { "summary": "...", "score": 숫자 }
                        - 90~100: 완벽
                        - 70~89: 대체로 좋음
                        - 50~69: 불완전
                        - 0~49: 실패
                    """
                    ),
                    HumanMessagePromptTemplate.from_template(
                        "원문:\n{input_text}\n\n요약:\n{summary}"
                    ),
                ],
                input_variables=["input_text", "summary"],
            )
            runnable = eval_prompt | llm | SummaryScoreParser()
            try:
                result = runnable.invoke(
                    {
                        "input_text": state["input_text"],
                        "summary": state["summary"],
                    }
                )
                state["score"] = result["score"]
            except Exception:
                state["score"] = 0
            return state

        return evaluate

    def _make_retry_node(self):
        def retry(state: GraphState) -> GraphState:
            state["retry_count"] = state.get("retry_count", 0) + 1
            return state

        return retry

    def _make_log_fail_node(self):
        def log_fail(state: GraphState) -> GraphState:
            score = state.get("score", 0)
            state["fail_reason"] = "품질 문제" if score < 50 else "시스템 오류"
            return state

        return log_fail

    def _make_save_node(self):
        def save(state: GraphState) -> GraphState:
            state["status"] = "saved"
            return state

        return save

    def _make_check_score(self):
        def check(state: GraphState) -> str:
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
        graph = StateGraph(GraphState)

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
