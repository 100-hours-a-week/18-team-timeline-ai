from langgraph.graph import StateGraph, END, START
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.schema import BaseOutputParser
import json
import re
from typing import List, TypedDict, Optional
import logging

# 로그 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SummaryState(TypedDict):
    input_text: str
    summary: str
    score: int
    worker_id: int
    retry_count_summary: int
    retry_count_title: int
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


class TotalSummarizationGraph:
    def __init__(self, server: str, model: str, max_retries: int = 3):
        self.server = server
        self.model = model
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
                description="요약된 36자 이내의 예측, 해석, 사견이 없는 완전한 문장",
            )
        ]
        parser = StructuredOutputParser.from_response_schemas(summary_schema)

        def summarize(state: SummaryState) -> SummaryState:
            system_prompt = """
            - 반드시 36자 이내의 2줄 이내의 문장을 제시하세요.
            - 주어진 글에 대해 간략한 요약을 제시하세요.
            - 예시의 형식을 참고하여 반드시 JSON으로 작성하세요.
            \'{{\'summary\': \'요약\'}}\'
            """
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    ("human", "{input_text}"),
                ]
            )
            runnable = prompt | llm | parser
            result = runnable.invoke({"input_text": state["input_text"]})
            print(f"요약 생성 결과:\n {result}")
            try:
                print(f"요약 생성 시작:\n {state['input_text']}")
                result = runnable.invoke({"input_text": state["input_text"]})
                print(f"요약 생성 결과:\n {result}")
                state["summary"] = result["summary"]
                logger.info(f"✅요약 생성 완료: {result['summary']}")
                if not state["summary"]:
                    raise ValueError("요약이 비어있습니다.")
            except Exception as e:
                logger.exception(f"❌ 요약 생성 실패: {e}, {state['summary']}")
                state["summary"] = None

            return state

        return summarize

    def _make_title_node(self, llm):
        title_schema = [
            ResponseSchema(
                name="title",
                description="현재 글의 제목. 18자 이내의 완전한 문장",
            )
        ]
        parser = StructuredOutputParser.from_response_schemas(title_schema)

        def makeTitle(state: SummaryState) -> SummaryState:
            system_prompt = """
            당신은 뉴스 제목 생성 전문가입니다. 뉴스의 제목을 지어주세요.
            - 반드시 18자 이내, 완결된 문장, 핵심 사실만 요약만을 제시하세요.
            - 예시의 형식을 참고하여 반드시 JSON으로 작성하세요.
            \'{{\'title\': \'제목\'}}\'
            """
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    ("human", "{input_text}"),
                ]
            )
            runnable = prompt | llm | parser
            try:
                result = runnable.invoke({"input_text": state["summary"]})
                state["title"] = result["title"]
                logger.info(f"✅요약 생성 완료: {result['title']}")
            except Exception as e:
                logger.exception(f"❌ 요약 생성 실패: {e}")
                state["title"] = state["summary"]
            return state

        return makeTitle

    def _make_title_eval_node(self, llm):
        response_schemas = [
            ResponseSchema(
                name="score", description="요약 품질 점수 (0~100 사이의 정수)"
            ),
        ]
        parser = StructuredOutputParser.from_response_schemas(response_schemas)

        def node(state: SummaryState) -> SummaryState:
            eval_prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessagePromptTemplate.from_template(
                        """
                        당신은 뉴스 제목 평가자입니다. 
                        예시의 형식을 참고하여 반드시 JSON으로 작성하세요.
                        예시: \'{{\'score\': 75}}\'
                        - 90~100: 문장에 의견이 들어가지 않고 문법 상 어색함이 없으며 문장이 18자 이하이며 핵심 사실을 정확히 요약함.
                        - 70~89: 1줄 이내이고 대체로 좋음 (약간의 어색함이나 불명확한 부분이 있을 수 있음)
                        - 50~69: 1줄 이상이며 불완전 (핵심 누락 또는 문법적 문제가 존재함)
                        - 0~49: 실패 (요약이 원문과 거의 무관하거나 문법이 심각하게 어색함)
                        """
                    ),
                    HumanMessagePromptTemplate.from_template(
                        "원문:\n{summary}\n\n제목:\n{title}"
                    ),
                ]
            )
            runnable = eval_prompt | llm | parser
            if not state.get("summary"):
                state["title_score"] = 0

                logger.info("요약이 비어있어 평가를 건너뜁니다.")
                return state
            try:
                logger.info(f"제목 평가 시작: {state['title']}")
                logger.info(f"원문: {state['summary']}")
                result = runnable.invoke(
                    {
                        "summary": state["summary"],
                        "title": state["title"],
                    }
                )
                state["title_score"] = int(result["score"])
                logger.info(f"평가 완료: {result['score']}")
            except Exception as e:
                state["title_score"] = 0
                logger.exception(f"평가 실패: {e}")
            return state

        return node

    def _make_classify_tag_node(self, llm):
        response_schemas = [
            ResponseSchema(
                name="tag",
                description="뉴스의 카테고리 태그. 경제, 연예, 스포츠, 과학, 기타 중 하나",
            ),
        ]
        parser = StructuredOutputParser.from_response_schemas(response_schemas)

        def classify_tag(state: SummaryState) -> SummaryState:
            system_prompt = """
            당신은 뉴스 카테고리 분류 전문가입니다. 아래 카테고리 중 하나만 골라서 응답하세요.
            [경제, 연예, 스포츠, 과학, 기타]
            예시의 형식을 참고하여 반드시 JSON으로 작성하세요.
            \'{{\'tag\': \'카테고리\'}}\'
            """
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    ("human", "{input_text}"),
                ]
            )
            runnable = prompt | llm | parser
            try:
                print(f"카테고리 분류 시작\n: {state['summary']}")
                logger.info(f"카테고리 분류 시작: {state['summary']}")
                result = runnable.invoke({"input_text": state["summary"]})
                state["tag"] = result["tag"].strip()
                logger.info(f"✅카테고리 분류 완료: {result['tag']}")

            except Exception as e:
                logger.exception(f"❌ 카테고리 분류 실패: {e}")
                state["tag"] = ""
            return state

        return classify_tag

    def _make_eval_node(self, llm):
        response_schemas = [
            ResponseSchema(
                name="score", description="요약 품질 점수 (0~100 사이의 정수)"
            ),
        ]
        parser = StructuredOutputParser.from_response_schemas(response_schemas)

        def node(state: SummaryState) -> SummaryState:
            eval_prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessagePromptTemplate.from_template(
                        """
                        당신은 뉴스 요약 평가자입니다.
                        다음 기준에 따라 채점하세요.
                        예시의 형식을 참고하여 반드시 JSON으로 작성하세요.
                        \'{{\'score\': 75}}\'
                        - 90~100: 문장에 의견이 들어가지 않고 문법 상 어색함이 없으며 문장이 2줄 이하이며 핵심 사실을 정확히 요약함.
                        - 70~89: 2줄 이내이고 대체로 좋음 (약간의 어색함이나 불명확한 부분이 있을 수 있음)
                        - 50~69: 2줄 이상이며 불완전 (핵심 누락 또는 문법적 문제가 존재함)
                        - 0~49: 실패 (요약이 원문과 거의 무관하거나 문법이 심각하게 어색함)
                    """
                    ),
                    HumanMessagePromptTemplate.from_template(
                        "원문:\n{input_text}\n\n요약:\n{summary}"
                    ),
                ]
            )
            runnable = eval_prompt | llm | parser
            if state["summary"] is None:
                state["score"] = 0
                logger.info("요약이 비어있어 평가를 건너뜁니다.")
                return state

            try:
                result = runnable.invoke(
                    {
                        "input_text": state["input_text"],
                        "summary": state["summary"],
                    }
                )
                state["score"] = int(result["score"])
                logger.info(f"평가 완료: {result['score']}")
            except Exception as e:
                state["score"] = 0
                logger.exception(f"평가 실패: {e}")
            return state

        return node

    def _make_retry_node(self, cnt):
        def retry(state: SummaryState) -> SummaryState:
            state[cnt] = state.get(cnt, 0) + 1
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

    def _make_check_score(self, cur_score, retry_count):
        def check(state: SummaryState) -> str:
            score = state.get(cur_score, 0)
            retries = state.get(retry_count, 0)
            if score >= 85:
                return "save"
            elif retries < self.max_retries:
                return "retry"
            else:
                state["summary"] = state["input_text"]
                return "log_fail"

        return check

    def build(self):
        llm = self._make_llm()

        graph = StateGraph(SummaryState)
        graph.add_node("summarize", self._make_summarize_node(llm))
        graph.add_node("make_title", self._make_title_node(llm))
        graph.add_node("eval_title", self._make_title_eval_node(llm))
        graph.add_node("classify_tag", self._make_classify_tag_node(llm))
        graph.add_node("eval_summary", self._make_eval_node(llm))
        graph.add_node("retry_summary", self._make_retry_node("retry_count_summary"))
        graph.add_node("retry_title", self._make_retry_node("retry_count_title"))

        graph.add_node("log_fail_summary", self._make_log_fail_node())
        graph.add_node("log_fail_title", self._make_log_fail_node())
        graph.add_node("save_summary", self._make_save_node())
        graph.add_node("save_title", self._make_save_node())

        graph.add_edge(START, "summarize")
        graph.add_edge("summarize", "eval_summary")
        graph.add_conditional_edges(
            "eval_summary",
            self._make_check_score("score", "retry_count_summary"),
            {
                "save": "save_summary",
                "retry": "retry_summary",
                "log_fail": "log_fail_summary",
            },
        )
        graph.add_edge("save_summary", "make_title")
        graph.add_edge("log_fail_summary", "make_title")
        graph.add_edge("retry_summary", "summarize")
        graph.add_edge("make_title", "eval_title")
        graph.add_conditional_edges(
            "eval_title",
            self._make_check_score("title_score", "retry_count_title"),
            {
                "save": "save_title",
                "retry": "retry_title",
                "log_fail": "log_fail_title",
            },
        )
        graph.add_edge("save_title", "classify_tag")
        graph.add_edge("log_fail_title", "classify_tag")
        graph.add_edge("retry_title", "make_title")
        graph.add_edge("classify_tag", END)
        app = graph.compile()
        # mermaid_code = app.get_graph().draw_mermaid()
        # print(mermaid_code)
        logger.info("에이전트 구축 완료")
        return app
