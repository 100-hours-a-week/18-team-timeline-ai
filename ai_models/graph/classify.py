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
    """감정 분류 그래프의 상태를 관리하는 클래스

    텍스트의 감정을 분류하는 과정에서 필요한 상태 정보를 저장합니다.

    Attributes:
        input_text (str): 분류할 원본 텍스트
        emotion (str): 분류된 감정 (긍정/부정/중립)
        score (int): 분류 신뢰도 점수
        worker_id (int): 작업자 ID
        retry_count (int): 재시도 횟수
        status (str): 현재 상태
    """

    input_text: str
    emotion: str
    score: int
    worker_id: int
    retry_count: int
    status: str


class ClassifyGraph:
    """감정 분류 그래프

    텍스트의 감정을 분류하는 그래프를 구성하고 실행합니다.
    LangGraph를 사용하여 감정 분류 작업을 순차적으로 처리합니다.

    Attributes:
        server (str): LLM 서버 URL
        model (str): 사용할 LLM 모델 이름
        max_retries (int): 최대 재시도 횟수
    """

    def __init__(self, server: str, model: str, max_retries: int = 3):
        """ClassifyGraph 초기화

        Args:
            server (str): LLM 서버 URL
            model (str): 사용할 LLM 모델 이름
            max_retries (int, optional): 최대 재시도 횟수. Defaults to 3.
        """
        self.server = server
        self.model = model
        self.max_retries = max_retries

    def _make_llm(self):
        """LLM 인스턴스 생성

        Returns:
            ChatOpenAI: 설정된 LLM 인스턴스
        """
        return ChatOpenAI(
            base_url=f"{self.server}/v1",
            api_key="not-needed",
            model=self.model,
            temperature=0.1,
        )

    def _make_classify_node(self, llm):
        """분류 노드 생성

        Args:
            llm: LLM 인스턴스

        Returns:
            function: 분류 함수
                입력된 텍스트의 감정을 분류하여 상태를 업데이트합니다.
        """
        response_schemas = [
            ResponseSchema(
                name="emotion",
                description="텍스트의 감정 (긍정/부정/중립)",
            ),
            ResponseSchema(
                name="score", description="분류 신뢰도 점수 (0~100 사이의 정수)"
            ),
        ]
        parser = StructuredOutputParser.from_response_schemas(response_schemas)

        def classify(state: ClassifyState) -> ClassifyState:
            system_prompt = """
            당신은 텍스트 감정 분석 전문가입니다.
            주어진 텍스트의 감정을 분석하여 긍정/부정/중립 중 하나로 분류하고,
            분류에 대한 신뢰도를 0~100 사이의 점수로 평가하세요.

            예시의 형식을 참고하여 반드시 JSON으로 작성하세요.
            \'{{\'emotion\': \'긍정\', \'score\': 85}}\'
            """
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    ("human", "{input_text}"),
                ]
            )
            runnable = prompt | llm | parser
            try:
                result = runnable.invoke({"input_text": state["input_text"]})
                state["emotion"] = result["emotion"]
                state["score"] = int(result["score"])
                logging.info(
                    f"✅감정 분류 완료: {result['emotion']} (점수: {result['score']})"
                )
            except Exception as e:
                logging.exception(f"❌ 감정 분류 실패: {e}")
                state["emotion"] = "중립"
                state["score"] = 0
            return state

        return classify

    def _make_retry_node(self):
        """재시도 노드 생성

        Returns:
            function: 재시도 함수
                재시도 횟수를 증가시키고 상태를 반환합니다.
        """

        def retry(state: ClassifyState) -> ClassifyState:
            state["retry_count"] = state.get("retry_count", 0) + 1
            return state

        return retry

    def _make_check_emotion(self):
        """감정 확인 노드 생성

        Returns:
            function: 확인 함수
                분류된 감정과 점수를 확인하여 다음 단계를 결정합니다.
        """

        def check(state: ClassifyState) -> str:
            score = state.get("score", 0)
            retries = state.get("retry_count", 0)
            if score >= 85:
                return "end"
            elif retries < self.max_retries:
                return "retry"
            else:
                return "end"

        return check

    def build(self):
        """감정 분류 그래프 구축

        Returns:
            StateGraph: 구성된 감정 분류 그래프
        """
        llm = self._make_llm()
        graph = StateGraph(ClassifyState)

        graph.add_node("classify", self._make_classify_node(llm))
        graph.add_node("retry", self._make_retry_node())

        graph.add_edge(START, "classify")
        graph.add_conditional_edges(
            "classify", self._make_check_emotion(), {"end": END, "retry": "retry"}
        )
        graph.add_edge("retry", "classify")

        app = graph.compile()
        return app


if __name__ == "__main__":
    SERVER = "https://b79f-34-125-17-94.ngrok-free.app"
    MODEL = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"
