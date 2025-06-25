import logging
import aiohttp
from config.prompts import SYSTEM_PROMPT, SystemRole
import orjson
from typing import Dict, Any, Optional
import asyncio


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
        logging.info(f"[Host] __aenter__ 진입 (host: {self.host}, model: {self.model})")
        try:
            self.session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(
                    limit=self.concurrency * 2, limit_per_host=20
                )
            )
            logging.info(f"[Host] aiohttp.ClientSession 생성 완료 (host: {self.host})")
            if not await self.check_connection():
                logging.error(f"[Host] Failed to connect to the host: {self.host}")
                await self.session.close()
                self.session = None
                self._is_connected = False
                raise RuntimeError(f"Failed to connect to the host: {self.host}")
            self._is_connected = True
            logging.info(f"[Host] Connected to the host: {self.host}")
            return self
        except Exception as e:
            logging.error(f"[Host] __aenter__ 예외 발생: {str(e)} (host: {self.host})")
            if self.session is not None:
                try:
                    await self.session.close()
                    logging.info(
                        f"[Host] 예외 발생으로 세션 강제 종료 (host: {self.host})"
                    )
                except Exception as ce:
                    logging.warning(
                        f"[Host] 세션 강제 종료 중 예외: {str(ce)} (host: {self.host})"
                    )
                finally:
                    self.session = None
            self._is_connected = False
            raise RuntimeError(f"[Host] Connection initialization failed: {str(e)}")

    async def __aexit__(self, exc_type, exc_value, traceback):
        logging.info(f"[Host] __aexit__ 진입 (host: {self.host})")
        try:
            if self.session:
                await self.session.close()
                logging.info(f"[Host] 세션 정상 종료 (host: {self.host})")
        except Exception as e:
            logging.error(
                f"[Host] 세션 종료 중 예외 발생: {str(e)} (host: {self.host})"
            )
        finally:
            self.session = None
            self._is_connected = False
            logging.info(
                f"[Host] __aexit__ finally: 세션 None 처리 (host: {self.host})"
            )

    async def close(self):
        logging.info(f"[Host] close() 호출 (host: {self.host})")
        if self.session:
            try:
                await self.session.close()
                logging.info(f"[Host] 세션 close() 정상 종료 (host: {self.host})")
            except Exception as e:
                logging.error(
                    f"[Host] 세션 close() 중 예외 발생: {str(e)} (host: {self.host})"
                )
            finally:
                self.session = None
                self._is_connected = False
                logging.info(
                    f"[Host] close() finally: 세션 None 처리 (host: {self.host})"
                )
        else:
            logging.info(
                f"[Host] close() - 이미 닫혀있거나 세션 없음 (host: {self.host})"
            )

    async def check_connection(self):
        url = f"{self.host}/health"
        if not self.session:
            logging.error(f"[Host] check_connection: 세션이 None임 (host: {self.host})")
            return False
        if self.session.closed:
            logging.error(
                f"[Host] check_connection: 세션이 이미 닫힘 (host: {self.host})"
            )
            return False
        logging.info(f"[Host] Checking connection to the host: {url}")
        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with self.session.get(url, timeout=timeout) as response:
                logging.info(f"[Host] Health check status: {response.status}")
                response.raise_for_status()
                return True
        except (
            aiohttp.ClientError,
            aiohttp.ClientConnectorError,
            aiohttp.ClientResponseError,
            aiohttp.ServerDisconnectedError,
        ) as e:
            logging.error(f"[Host] Health check failed: {e} (host: {self.host})")
            return False
        except Exception as e:
            logging.error(f"[Host] Health check 예외: {str(e)} (host: {self.host})")
            return False

    async def query(self, task: SystemRole, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.session or not self._is_connected:
            logging.error(
                f"[Host] query: 세션이 초기화되지 않았거나 연결 안됨 (host: {self.host})"
            )
            raise RuntimeError("Host session is not initialized or connected")
        if self.session.closed:
            logging.error(f"[Host] query: 세션이 이미 닫힘 (host: {self.host})")
            raise RuntimeError("Host session is closed")
        if not payload or "text" not in payload:
            logging.error(f"[Host] query: payload가 잘못됨 (host: {self.host})")
            raise ValueError("Invalid payload: must contain 'text' field")
        text = payload["text"]
        if not text or not text.strip():
            logging.warning(f"[Host] 빈 텍스트 감지: {text}")
            return {"choices": [{"message": {"content": "빈 텍스트입니다."}}]}
        if len(text) > 1000:
            logging.warning(f"[Host] 텍스트가 너무 김: {len(text)} 문자")
            text = text[:1000]
        system_prompt = SYSTEM_PROMPT[task]
        if len(system_prompt) > 1000:
            logging.warning(
                f"[Host] 시스템 프롬프트가 너무 김: {len(system_prompt)} 문자"
            )
            system_prompt = system_prompt[:1000]
        headers = {"Content-Type": "application/json"}
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        logging.debug(f"[Host] Request body: {body}")
        logging.debug(f"[Host] 텍스트 길이: {len(text)}")
        logging.debug(f"[Host] 시스템 프롬프트 길이: {len(system_prompt)}")
        url = f"{self.host}/v1/chat/completions"
        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            async with self.session.post(
                url,
                data=orjson.dumps(body),
                timeout=timeout,
                headers=headers,
            ) as response:
                logging.debug(f"[Host] HTTP 응답 수신 - 상태 코드: {response.status}")
                if response.status == 400:
                    response_text = await response.text()
                    logging.error(f"[Host] Bad Request 응답 내용: {response_text}")
                    logging.error(f"[Host] Bad Request 요청 본문: {body}")
                    logging.error(f"[Host] Bad Request 요청 URL: {url}")
                    logging.error(f"[Host] Bad Request 요청 헤더: {headers}")
                response.raise_for_status()
                response_text = await response.text()
                logging.debug(f"[Host] Response text: {response_text}")
                try:
                    result = orjson.loads(response_text)
                except orjson.JSONDecodeError as e:
                    logging.error(
                        f"[Host] Failed to parse response as JSON: {response_text}"
                    )
                    raise ValueError(f"Invalid JSON response: {str(e)}")
                if response.status == 400:
                    logging.error(f"[Host] Bad Request: {result}")
                    raise ValueError(
                        f"Bad Request: {result.get('error', 'Unknown error')}"
                    )
                return result
        except aiohttp.ClientResponseError as e:
            logging.error(f"[Host] HTTP {e.status} error: {e.message}")
            raise
        except aiohttp.ClientConnectorError as e:
            logging.error(f"[Host] 연결 오류: {e}")
            logging.error(f"[Host] 연결 오류 타입: {type(e).__name__}")
            import traceback

            logging.error(f"[Host] 연결 오류 스택 트레이스: {traceback.format_exc()}")
            raise
        except asyncio.TimeoutError as e:
            logging.error(f"[Host] 타임아웃 오류: {e}")
            logging.error(f"[Host] 타임아웃 오류 타입: {type(e).__name__}")
            import traceback

            logging.error(
                f"[Host] 타임아웃 오류 스택 트레이스: {traceback.format_exc()}"
            )
            raise
        except Exception as e:
            logging.error(f"[Host] Unexpected error: {str(e)}")
            raise
