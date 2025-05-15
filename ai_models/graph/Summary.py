from langgraph.graph import StateGraph, END, START
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from typing import TypedDict
from utils.logger import Logger

logger = Logger.get_logger("summary")


class SummaryState(TypedDict):
    """요약 그래프의 상태를 관리하는 클래스

    텍스트 요약 과정에서 필요한 상태 정보를 저장합니다.

    Attributes:
        input_text (str): 요약할 원본 텍스트
        text (str): 생성된 요약 텍스트
        score (int): 요약 품질 점수
        worker_id (int): 작업자 ID
        retry_count (int): 재시도 횟수
        status (str): 현재 상태
    """

    url: str
    title: str
    input_text: str
    text: str
    score: int
    worker_id: int
    retry_count: int
    status: str


'''
class SummaryScoreParser(BaseOutputParser):
    """요약 점수 파싱기

    LLM의 출력을 파싱하여 요약 점수를 추출합니다.
    JSON 형식의 출력을 처리하며, 파싱 실패 시 정규식을 사용하여 복구를 시도합니다.
    """

    def parse(self, text: str):
        """텍스트에서 JSON을 파싱하여 점수를 추출

        Args:
            text (str): 파싱할 텍스트

        Returns:
            dict: 파싱된 JSON 객체

        Raises:
            ValueError: JSON 파싱 실패 시
        """
        cleaned = re.sub(r"^```json\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            matched = re.search(r"\{.*\}", text, re.DOTALL)
            if matched:
                return json.loads(matched.group())
            raise ValueError("Invalid JSON format")
'''


class SummarizationGraph:
    """요약 그래프

    텍스트를 요약하는 그래프를 구성하고 실행합니다.
    LangGraph를 사용하여 요약 작업을 순차적으로 처리합니다.

    Attributes:
        server (str): LLM 서버 URL
        model (str): 사용할 LLM 모델 이름
        max_retries (int): 최대 재시도 횟수
    """

    def __init__(self, server: str, model: str, max_retries: int = 3):
        """SummarizationGraph 초기화

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
            temperature=0.2,
        )

    def _make_summarize_node(self, llm):
        """요약 노드 생성

        Args:
            llm: LLM 인스턴스

        Returns:
            function: 요약 함수
                입력된 텍스트를 요약하여 상태를 업데이트합니다.
        """
        summary_schema = [
            ResponseSchema(
                name="summary",
                description="요약된 24자 이내의 예측, 해석, 사견이 없는 완전한 문장",
            )
        ]
        parser = StructuredOutputParser.from_response_schemas(summary_schema)

        def summarize(state: SummaryState) -> SummaryState:
            system_prompt = """
            - 반드시 1줄 요약을 제시하세요.
            - 예시의 형식을 참고하여 반드시 JSON으로 작성하세요.
            \'{{\'summary\': \'...\'}}\'
            """
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    ("human", "제목: \n{title}\n\n본문: \n{input_text}"),
                ]
            )
            try:
                runnable = prompt | llm | parser
                logger.info(f"요약 생성 시작:\n {state['input_text']}")
                result = runnable.invoke(
                    {"title": state["title"], "input_text": state["input_text"]}
                )
                state["text"] = result["summary"]
                logger.info(f"✅요약 생성 완료: {result['summary']}")
                if not state["text"]:
                    raise ValueError("요약이 비어있습니다.")
            except Exception as e:
                logger.error(f"❌ 요약 생성 실패: {e}")
                state["text"] = None

            return state

        return summarize

    def _make_eval_node(self, llm):
        """요약 평가 노드 생성

        Args:
            llm: LLM 인스턴스

        Returns:
            function: 요약 평가 함수
                생성된 요약의 품질을 평가하여 점수를 부여합니다.
        """
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
                        예시: \'{{\'score\': 75}}\'
                        - 90~100: 문장에 의견이 들어가지 않고 문법 상 어색함이 없으며
                          문장이 1줄이며 핵심 사실을 정확히 요약함.
                        - 70~89: 1줄이고 대체로 좋음
                          (약간의 어색함이나 불명확한 부분이 있을 수 있음)
                        - 50~69: 1줄 초과이며 불완전 (핵심 누락 또는 문법적 문제가 존재함)
                        - 0~49: 실패 (요약이 원문과 거의 무관하거나 문법이 심각하게 어색함)
                        """
                    ),
                    HumanMessagePromptTemplate.from_template(
                        "원문:\n{input_text}\n\n요약:\n{summary}"
                    ),
                ]
            )
            runnable = eval_prompt | llm | parser
            if state["text"] is None:
                state["score"] = 0
                logger.info("요약이 비어있어 평가를 건너뜁니다.")
                return state

            try:
                result = runnable.invoke(
                    {
                        "input_text": state["input_text"],
                        "summary": state["text"],
                    }
                )
                state["score"] = int(result["score"])
                logger.info(f"평가 완료: {result['score']}")
            except Exception as e:
                state["score"] = 0
                logger.error(f"평가 실패: {e}")
            return state

        return node

    def _make_retry_node(self):
        """재시도 노드 생성

        Returns:
            function: 재시도 함수
                재시도 횟수를 증가시키고 상태를 반환합니다.
        """

        def retry(state: SummaryState) -> SummaryState:
            state["retry_count"] = state.get("retry_count", 0) + 1
            return state

        return retry

    def _make_log_fail_node(self):
        """실패 로깅 노드 생성

        Returns:
            function: 로깅 함수
                실패 원인을 기록하고 상태를 반환합니다.
        """

        def log_fail(state: SummaryState) -> SummaryState:
            score = state.get("score", 0)
            state["fail_reason"] = "품질 문제" if score < 50 else "시스템 오류"
            return state

        return log_fail

    def _make_save_node(self):
        """저장 노드 생성

        Returns:
            function: 저장 함수
                상태를 저장 완료로 표시하고 반환합니다.
        """

        def save(state: SummaryState) -> SummaryState:
            state["status"] = "saved"
            return state

        return save

    def _make_check_score(self):
        """점수 확인 노드 생성

        Returns:
            function: 확인 함수
                점수와 재시도 횟수를 확인하여 다음 단계를 결정합니다.
        """

        def check(state: SummaryState) -> str:
            score = state.get("score", 0)
            retries = state.get("retry_count", 0)
            if score >= 85:
                return "save"
            elif retries < self.max_retries:
                return "retry"
            else:
                state["text"] = state["input_text"]
                return "log_fail"

        return check

    def build(self):
        """요약 그래프 구축

        Returns:
            StateGraph: 구성된 요약 그래프
        """
        llm = self._make_llm()

        graph = StateGraph(SummaryState)
        graph.add_node("summarize", self._make_summarize_node(llm))
        graph.add_node("eval_summary", self._make_eval_node(llm))
        graph.add_node("retry", self._make_retry_node())
        graph.add_node("log_fail", self._make_log_fail_node())
        graph.add_node("save", self._make_save_node())

        graph.add_edge(START, "summarize")
        graph.add_edge("summarize", "eval_summary")
        graph.add_conditional_edges(
            "eval_summary",
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
        return app
