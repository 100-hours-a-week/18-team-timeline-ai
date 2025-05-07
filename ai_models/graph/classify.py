from typing import List, Literal
from langchain_openai import ChatOpenAI
from textwrap import dedent
from pydantic import BaseModel, Field
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, END, StateGraph


import logging
from dotenv import load_dotenv


class ClassifyState(dict):
    input_text: str
    transcript: str
    timeline: str
    emotion: str
    score: int
    retry_count: int


class ClassifyGraph:
    def __init__(self, server: str, model: str, max_retries: int = 3):
        self.max_retries = max_retries
        self.server = server
        self.model = model
        logging.info("초기화 완료")

    def _make_llm(self):
        return ChatOpenAI(
            base_url=f"{self.server}/v1",
            api_key="not-needed",
            model=self.model,
            temperature=0.1,
        )

    def _make_classify_node(self, llm):
        classify_schema = [
            ResponseSchema(
                name="emotion",
                description="댓글의 감정을 분류합니다. 반드시 다음 중 하나: 긍정, 부정, 관련없음",
            )
        ]

        parser = StructuredOutputParser.from_response_schemas(classify_schema)

        def classify(state: ClassifyState) -> ClassifyState:
            system = dedent(
                f"""
                당신은 감정 분류 전문가입니다. 댓글을 보고 분류하세요.
                가능한 분류값은 다음 중 하나입니다: 긍정, 부정, 관련없음
                - 예시의 형식을 따라서 반드시 JSON으로 제공하세요.
                \'{{{{\'emotion\': \'긍정\'}}}}\'
                다음의 문맥을 참조하세요:
                {state.get("transcript", "")}
                \n\n
                {state.get("timeline", "")}
                댓글:
                """
            )

            prompt = ChatPromptTemplate.from_messages(
                [("system", system), ("human", "{input_text}")]
            )

            try:
                runnbale = prompt | llm | parser
                logging.info(f"댓글 감정 분류 시작: {state['input_text']}")
                result = runnbale.invoke({"input_text": state["input_text"]})
                state["emotion"] = result["emotion"]
                logging.info(f"댓글 감정 분류 완료: {state['emotion']}")
                if not state["emotion"]:
                    raise ValueError("감정 분류 실패")
            except Exception as e:
                logging.error(f"댓글 분류 실패: {e}, {state['input_text']}")
                state["emotion"] = None

            return state

        return classify

    def _make_retry_node(self):
        def retry(state: ClassifyState) -> ClassifyState:
            state["retry_count"] = state.get("retry_count", 0) + 1
            return state

        return retry

    def _make_check_emotion(self):
        def check(state: ClassifyState) -> str:
            if state.get("emotion") is not None:
                return "end"
            elif state.get("retry_count", 0) < self.max_retries:
                return "retry"
            else:
                return "end"

        return check

    def build(self):
        llm = self._make_llm()
        graph = StateGraph(ClassifyState)

        graph.add_node("classify", self._make_classify_node(llm))
        graph.add_node("retry", self._make_retry_node())

        graph.add_edge(START, "classify")
        graph.add_conditional_edges(
            "classify", self._make_check_emotion(), {"end": END, "retry": "retry"}
        )
        graph.add_edge("retry", "classify")

        return graph.compile()


if __name__ == "__main__":
    SERVER = "https://b79f-34-125-17-94.ngrok-free.app"
    MODEL = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"
