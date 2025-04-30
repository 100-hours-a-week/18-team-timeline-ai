# solid_summary_graph.py
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from textwrap import dedent
from langchain.schema import BaseOutputParser
import json
import re
from typing import List


class SummaryState(dict):
    input_text: str
    summary: str
    score: int
    worker_id: int
    retry_count: int
    status: str
    title: str
    title_score: int
    tag: str


class SummaryScoreParser(BaseOutputParser):
    def parse(self, text: str):
        cleaned = re.sub(r"^```json\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            matched = re.search(r"\{.*\}", text, re.DOTALL)
            if matched:
                return json.loads(matched.group())
            raise ValueError("Invalid JSON format")


examples = [
    HumanMessage("도널드 트럼프..."),
    AIMessage("트럼프 대통령은 관세 정책에 변화가 없음을 재확인..."),
]

TAG_CANDIDATES = ["정치", "경제", "사회", "국제", "과학", "스포츠"]


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

    def _make_prompt_node(self, llm, system_prompt, input_keys, output_key):
        def node(state: SummaryState) -> SummaryState:
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    *self.examples,
                    ("human", "{" + input_keys[0] + "}"),
                ]
            )
            runnable = prompt | llm
            result = runnable.invoke({input_keys[0]: state[input_keys[0]]})
            state[output_key] = result.content
            return state

        return node

    def _make_summary_eval_node(self, llm):
        def node(state: SummaryState) -> SummaryState:
            prompt = ChatPromptTemplate(
                messages=[
                    SystemMessagePromptTemplate.from_template(
                        '요약 평가 기준 JSON {"summary": "...", "score": 숫자}'
                    ),
                    HumanMessagePromptTemplate.from_template(
                        "원문:\n{input_text}\n\n요약:\n{summary}"
                    ),
                ],
                input_variables=["input_text", "summary"],
            )
            runnable = prompt | llm | SummaryScoreParser()
            try:
                result = runnable.invoke(
                    {"input_text": state["input_text"], "summary": state["summary"]}
                )
                state["score"] = result["score"]
            except:
                state["score"] = 0
            return state

        return node

    def _make_title_eval_node(self, llm):
        def node(state: SummaryState) -> SummaryState:
            prompt = ChatPromptTemplate(
                messages=[
                    SystemMessagePromptTemplate.from_template(
                        '제목 평가 기준 JSON {"title": "...", "score": 숫자}'
                    ),
                    HumanMessagePromptTemplate.from_template("제목:\n{title}"),
                ],
                input_variables=["title"],
            )
            runnable = prompt | llm | SummaryScoreParser()
            try:
                result = runnable.invoke({"title": state["title"]})
                state["title_score"] = result["score"]
            except:
                state["title_score"] = 0
            return state

        return node

    def _make_check_score(self, key, threshold, retry_label, fail_label, success_label):
        def check(state: SummaryState) -> str:
            score = state.get(key, 0)
            retries = state.get("retry_count", 0)
            if score >= threshold:
                return success_label
            elif retries < self.max_retries:
                return retry_label
            else:
                return fail_label

        return check

    def build(self):
        llm = self._make_llm()
        graph = StateGraph(SummaryState)

        # 요약 및 평가 흐름
        graph.add_node(
            "summarize",
            self._make_prompt_node(
                llm, "뉴스 요약 전문가. 3줄 이내. 사견 금지.", ["input_text"], "summary"
            ),
        )
        graph.add_node("evaluate_summary", self._make_summary_eval_node(llm))
        graph.add_node(
            "retry",
            lambda state: {**state, "retry_count": state.get("retry_count", 0) + 1},
        )
        graph.add_node(
            "fallback_summary",
            lambda state: {
                **state,
                "summary": "요약 실패. 원문 참고.",
                "status": "fallback_summary",
            },
        )
        graph.add_node(
            "save_summary", lambda state: {**state, "status": "saved_summary"}
        )

        graph.add_edge(START, "summarize")
        graph.add_edge("summarize", "evaluate_summary")
        graph.add_conditional_edges(
            "evaluate_summary",
            self._make_check_score(
                "score", 85, "retry", "fallback_summary", "save_summary"
            ),
            {
                "retry": "retry",
                "fallback_summary": "fallback_summary",
                "save_summary": "save_summary",
            },
        )
        graph.add_edge("retry", "summarize")

        # 제목 생성 및 평가 흐름
        graph.add_node(
            "generate_title",
            self._make_prompt_node(
                llm, "뉴스 제목 생성. 명확하고 요약된 문구.", ["summary"], "title"
            ),
        )
        graph.add_node("evaluate_title", self._make_title_eval_node(llm))
        graph.add_node(
            "retry_title",
            lambda state: {**state, "retry_count": state.get("retry_count", 0) + 1},
        )
        graph.add_node("fallback_title", lambda state: {**state, "title": "대체 제목"})

        graph.add_edge("save_summary", "generate_title")
        graph.add_edge("fallback_summary", "generate_title")
        graph.add_edge("generate_title", "evaluate_title")
        graph.add_conditional_edges(
            "evaluate_title",
            self._make_check_score(
                "title_score", 80, "retry_title", "fallback_title", "tagging"
            ),
            {
                "retry_title": "generate_title",
                "fallback_title": "fallback_title",
                "tagging": "tagging",
            },
        )
        graph.add_edge("fallback_title", "tagging")

        # 태깅 및 메타 저장 흐름
        graph.add_node("tagging", lambda state: {**state, "tag": TAG_CANDIDATES[0]})
        graph.add_node("save_meta", lambda state: {**state, "status": "meta_saved"})
        graph.add_node(
            "feedback_ready", lambda state: {**state, "status": "feedback_ready"}
        )
        graph.add_node("log_success", lambda state: {**state, "status": "success"})

        graph.add_edge("tagging", "save_meta")
        graph.add_edge("save_meta", "feedback_ready")
        graph.add_edge("feedback_ready", "log_success")
        graph.add_edge("log_success", END)

        return graph.compile()
