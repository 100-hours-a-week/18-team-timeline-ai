import aiohttp
from utils.logger import Logger
from config.prompts import SYSTEM_PROMPT, SystemRole
import orjson
from typing import Dict, Any, Optional
import asyncio

logger = Logger.get_logger("ai_models.host")


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
        concurrency: int = 128,
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
        self._is_connected = False

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
        url = f"{self.host}/health"
        logger.info(f"[Host] Checking connection to the host: {url}")
        try:
            async with self.session.get(url, timeout=self.timeout) as response:
                logger.info(f"[Host] Health check status: {response.status}")
                response.raise_for_status()
                return True
        except (
            aiohttp.ClientError,
            aiohttp.ClientConnectorError,
            aiohttp.ClientResponseError,
            aiohttp.ServerDisconnectedError,
        ) as e:
            logger.error(f"[Host] Health check failed: {e}")
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

        # 텍스트 유효성 검사
        text = payload["text"]
        if not text or not text.strip():
            logger.warning(f"[Host] 빈 텍스트 감지: {text}")
            return {"choices": [{"message": {"content": "빈 텍스트입니다."}}]}

        if len(text) > 8000:  # 토큰 제한 고려
            logger.warning(f"[Host] 텍스트가 너무 김: {len(text)} 문자")
            text = text[:8000]

        # 시스템 프롬프트 길이 제한
        system_prompt = SYSTEM_PROMPT[task]
        if len(system_prompt) > 1000:  # 시스템 프롬프트 길이 제한
            logger.warning(
                f"[Host] 시스템 프롬프트가 너무 김: {len(system_prompt)} 문자"
            )
            system_prompt = system_prompt[:1000]

        headers = {"Content-Type": "application/json"}

        body = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {"role": "user", "content": text},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

        logger.debug(f"[Host] Request body: {body}")
        logger.debug(f"[Host] 텍스트 길이: {len(text)}")
        logger.debug(f"[Host] 시스템 프롬프트 길이: {len(system_prompt)}")
        url = f"{self.host}/v1/chat/completions"

        try:
            async with self.session.post(
                url,
                data=orjson.dumps(body),
                timeout=self.timeout,
                headers=headers,
            ) as response:
                logger.debug(f"[Host] HTTP 응답 수신 - 상태 코드: {response.status}")

                # 400 오류 시 상세 정보 로깅
                if response.status == 400:
                    response_text = await response.text()
                    logger.error(f"[Host] Bad Request 응답 내용: {response_text}")
                    logger.error(f"[Host] Bad Request 요청 본문: {body}")
                    logger.error(f"[Host] Bad Request 요청 URL: {url}")
                    logger.error(f"[Host] Bad Request 요청 헤더: {headers}")

                response.raise_for_status()
                response_text = await response.text()
                logger.debug(f"[Host] Response text: {response_text}")

                try:
                    result = orjson.loads(response_text)
                except orjson.JSONDecodeError as e:
                    logger.error(
                        f"[Host] Failed to parse response as JSON: {response_text}"
                    )
                    raise ValueError(f"Invalid JSON response: {str(e)}")

                if response.status == 400:
                    logger.error(f"[Host] Bad Request: {result}")
                    raise ValueError(
                        f"Bad Request: {result.get('error', 'Unknown error')}"
                    )

                return result

        except aiohttp.ClientResponseError as e:
            logger.error(f"[Host] HTTP {e.status} error: {e.message}")
            raise
        except aiohttp.ClientConnectorError as e:
            logger.error(f"[Host] 연결 오류: {e}")
            logger.error(f"[Host] 연결 오류 타입: {type(e).__name__}")
            import traceback

            logger.error(f"[Host] 연결 오류 스택 트레이스: {traceback.format_exc()}")
            raise
        except asyncio.TimeoutError as e:
            logger.error(f"[Host] 타임아웃 오류: {e}")
            logger.error(f"[Host] 타임아웃 오류 타입: {type(e).__name__}")
            import traceback

            logger.error(
                f"[Host] 타임아웃 오류 스택 트레이스: {traceback.format_exc()}"
            )
            raise
        except Exception as e:
            logger.error(f"[Host] Unexpected error: {str(e)}")
            raise
