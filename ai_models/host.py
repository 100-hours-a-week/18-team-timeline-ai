import aiohttp
from utils.logger import Logger
from utils.handling import handle_http_error
from enum import Enum
import logging
import orjson
from typing import Dict, Any, Optional

logger = Logger.get_logger("ai_models.host", log_level=logging.ERROR)


class SystemRole(Enum):
    summary = "summary"
    title = "title"


SYSTEM_PROMPT = {
    SystemRole.summary: (
        "당신은 한국어 문장 요약 전문가입니다. 주어진 문장을 정확히 파악한 뒤, 그 의미를 유지하면서도 "
        "가장 핵심적인 내용을 한 문장으로 요약합니다. 모든 응답은 반드시 **32자 이내의 반말**로 작성해야 하며, "
        "요약 이외의 설명, 사족, 감정 표현은 절대 포함하지 마세요. 예를 들어, '다음은 요약입니다:' 또는 "
        "'~입니다' 같은 표현도 사용하지 마세요. 이 작업은 민감한 사안에 대한 의견을 요구하는 것이 아닌, "
        "기계적인 요약 작업임을 유의하세요."
        "모든 제목은 평서문(~다) 형식으로 끝나야 한다."
    ),
    SystemRole.title: (
        "당신은 한국어 뉴스 제목을 가장 효과적으로 짓는 전문가입니다. 주어진 문장의 핵심을 파악해 가장 간결하고 "
        "주목을 끌 수 있는 제목을 지으세요. 모든 응답은 반드시 **18자 이내**여야 하며, 제목 이외의 말은 절대 하지 마세요. "
        "‘다음은 제목입니다:’ 같은 서술도 금지됩니다. 제목은 반드시 간결하고 사실 기반으로 작성하세요. "
        "비문, 감정 표현, 수식어 과잉은 지양하고, 독자의 관심을 끌 수 있도록 단호하고 분명하게 작성하세요."
    ),
}


class Host:
    # 초기화
    def __init__(
        self,
        host: str,
        model: str,
        timeout: int = 60,
        temperature: float = 0.5,
        max_tokens: int = 48,
        verbose: bool = False,
        concurrency: int = 32,
        retry_attempts: int = 3,
        retry_delay: float = 1.0,
    ):
        self.host = host
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.session: Optional[aiohttp.ClientSession] = None
        self.verbose = verbose
        self.concurrency = concurrency

    async def __aenter__(self):
        """
        초기화

        Raises:
            RuntimeError: 호스트 연결 실패

        Returns:
            Host: 호스트 객체
        """
        try:
            self.session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(
                    limit=self.concurrency * 2, limit_per_host=20
                )
            )
            if not await self.check_connection():
                logger.error(f"[Host] Failed to connect to the host: {self.host}")
                raise RuntimeError(f"Failed to connect to the host: {self.host}")

            self._is_connected = True
            logger.info(f"[Host] Connected to the host: {self.host}")
            return self
        except Exception as e:
            # 세션 생성 후 연결 실패 시 세션 정리
            if self.session is not None:
                await self.session.close()
                self.session = None
            raise RuntimeError(f"[Host] Connection initialization failed: {str(e)}")

    async def __aexit__(self, exc_type, exc_value, traceback):
        """
        종료

        Args:
            exc_type (_type_): 예외 타입
            exc_value (_type_): 예외 값
            traceback (_type_): 예외 추적 정보
        """
        logger.info(f"[Host] Closing the host: {self.host}")
        await self.session.close()

    async def close(self):
        """
        종료
        """
        if self.session:
            await self.session.close()
            self.session = None

    async def check_connection(self):
        """
        연결 확인

        Returns:
            bool: 연결 여부
        """
        url = f"{self.host}/v1/models"
        logger.info(f"[Host] Checking connection to the host: {url}")
        try:
            async with self.session.get(url, timeout=self.timeout) as response:
                logger.info(f"[Host] {response.status}")
                response.raise_for_status()
                json_response = await response.json()
                logger.info(f"[Host] {json_response}")
                return True
        except (
            aiohttp.ClientError,
            aiohttp.ClientConnectorError,
            aiohttp.ClientResponseError,
            aiohttp.ServerDisconnectedError,
        ) as e:
            logger.error(f"[Host] {e}")
            return False

    async def query(self, task: SystemRole, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        AI 모델에 쿼리 요청

        Args:
            task: 요청 타입
            payload: 요청 데이터

        Raises:
            Exception: 요청 실패
            e: 예외 정보

        Returns:
            Dict[str, Any]: 응답 데이터
        """
        if not self.session or not self._is_connected:
            raise RuntimeError("Host session is not initialized or connected")

        if not payload or "text" not in payload:
            raise ValueError("Invalid payload: must contain 'text' field")

        headers = {"Content-Type": "application/json"}

        body = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT[task],
                },
                {"role": "user", "content": payload["text"]},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        logger.info(f"[Host] {body}")
        url = f"{self.host}/v1/chat/completions"
        logger.info(f"[Host] {url}")
        try:
            async with self.session.post(
                url,
                data=orjson.dumps(body),
                timeout=self.timeout,
                headers=headers,
            ) as response:
                response.raise_for_status()
                response_text = await response.text()
                logger.info(f"[Host] {response_text}")
                result = orjson.loads(response_text)
                success = await handle_http_error(result, body, logger)
                if success:
                    logger.info(f"[Host] {result}")
                    return result
                else:
                    logger.error(f"[Host] {result}")
                    raise Exception(result)
        except (
            aiohttp.ClientError,
            aiohttp.ClientConnectorError,
            aiohttp.ClientResponseError,
            aiohttp.ServerDisconnectedError,
        ) as e:
            logger.error(f"[Host] {e}")
            raise e
