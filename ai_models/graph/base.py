from abc import ABC, abstractmethod
from typing import TypedDict, Any, Dict
from langchain.chat_models import ChatOpenAI
from langgraph.graph import StateGraph, START, END


class BaseState(TypedDict):
    """기본 상태 클래스"""

    input_text: str
    status: str
    worker_id: int
    retry_count: int


class BaseNode(ABC):
    """기본 노드 추상 클래스"""

    @abstractmethod
    def process(self, state: BaseState) -> BaseState:
        """상태를 처리하는 메서드"""
        pass


class BaseGraph(ABC):
    """기본 그래프 추상 클래스"""

    def __init__(self, server: str, model: str, max_retries: int = 3):
        """그래프 초기화

        Args:
            server (str): LLM 서버 URL
            model (str): 사용할 LLM 모델 이름
            max_retries (int, optional): 최대 재시도 횟수. Defaults to 3.
        """
        self.server = server
        self.model = model
        self.max_retries = max_retries

    def _make_llm(self) -> ChatOpenAI:
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

    @abstractmethod
    def build(self) -> StateGraph:
        """그래프 구축

        Returns:
            StateGraph: 구성된 그래프
        """
        pass


class RetryNode(BaseNode):
    """재시도 노드"""

    def __init__(self, retry_key: str):
        """재시도 노드 초기화

        Args:
            retry_key (str): 재시도 횟수를 저장할 상태 키
        """
        self.retry_key = retry_key

    def process(self, state: BaseState) -> BaseState:
        """재시도 횟수를 증가시키고 상태를 반환

        Args:
            state (BaseState): 현재 상태

        Returns:
            BaseState: 업데이트된 상태
        """
        state[self.retry_key] = state.get(self.retry_key, 0) + 1
        return state


class LogFailNode(BaseNode):
    """실패 로깅 노드"""

    def process(self, state: BaseState) -> BaseState:
        """실패 원인을 기록하고 상태를 반환

        Args:
            state (BaseState): 현재 상태

        Returns:
            BaseState: 업데이트된 상태
        """
        score = state.get("score", 0)
        state["fail_reason"] = "품질 문제" if score < 50 else "시스템 오류"
        return state


class SaveNode(BaseNode):
    """저장 노드"""

    def process(self, state: BaseState) -> BaseState:
        """상태를 저장 완료로 표시하고 반환

        Args:
            state (BaseState): 현재 상태

        Returns:
            BaseState: 업데이트된 상태
        """
        state["status"] = "saved"
        return state
