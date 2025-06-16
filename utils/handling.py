import asyncio
from aiohttp import ClientConnectorError, ServerDisconnectedError, ClientResponseError
from utils.logger import Logger
import time
from datetime import datetime
import json

logger = Logger.get_logger("utils.handling")


def format_log_message(
    level: str,
    message: str,
    query: str,
    status: int = None,
    response_time: float = None,
    error: str = None,
    additional_info: dict = None,
) -> str:
    """표준화된 로그 메시지 포맷을 생성합니다.

    Args:
        level: 로그 레벨 (INFO, ERROR, WARNING 등)
        message: 기본 메시지
        query: 실행된 쿼리
        status: HTTP 상태 코드 (선택)
        response_time: 응답 시간 (선택)
        error: 에러 메시지 (선택)
        additional_info: 추가 정보 (선택)

    Returns:
        str: 포맷된 로그 메시지
    """
    timestamp = datetime.utcnow().isoformat()
    log_data = {
        "timestamp": timestamp,
        "level": level,
        "message": message,
        "query": query,
    }

    if status is not None:
        log_data["status"] = status
    if response_time is not None:
        log_data["response_time"] = f"{response_time:.6f}"
    if error is not None:
        log_data["error"] = error
    if additional_info is not None:
        log_data.update(additional_info)

    return json.dumps(log_data, ensure_ascii=False)


async def handle_http_error(result, query, logger):
    start_time = time.time()

    if isinstance(result, ClientConnectorError):
        response_time = time.time() - start_time
        log_msg = format_log_message(
            level="ERROR",
            message="Connection failed",
            query=query,
            response_time=response_time,
            error=str(result),
        )
        logger.error(log_msg)
        await asyncio.sleep(3)
        return False

    elif isinstance(result, ServerDisconnectedError):
        response_time = time.time() - start_time
        log_msg = format_log_message(
            level="ERROR",
            message="Server disconnected",
            query=query,
            response_time=response_time,
            error=str(result),
        )
        logger.error(log_msg)
        await asyncio.sleep(1.5)
        return False

    elif isinstance(result, ClientResponseError):
        response_time = time.time() - start_time
        if 500 <= result.status < 600:
            log_msg = format_log_message(
                level="WARNING",
                message="Server internal error",
                query=query,
                status=result.status,
                response_time=response_time,
                error=str(result),
            )
            logger.warning(log_msg)
            await asyncio.sleep(1.5)
        elif 400 <= result.status < 500:
            log_msg = format_log_message(
                level="WARNING",
                message="Client error",
                query=query,
                status=result.status,
                response_time=response_time,
                error=str(result),
            )
            logger.warning(log_msg)
            await asyncio.sleep(1.5)
        else:
            log_msg = format_log_message(
                level="WARNING",
                message="Unexpected HTTP error",
                query=query,
                status=result.status,
                response_time=response_time,
                error=str(result),
            )
            logger.warning(log_msg)
        return False

    elif isinstance(result, dict) and "error" in result:
        response_time = time.time() - start_time
        log_msg = format_log_message(
            level="ERROR",
            message="API error",
            query=query,
            response_time=response_time,
            error=str(result),
        )
        logger.error(log_msg)
        await asyncio.sleep(2)
        return False

    elif isinstance(result, Exception):
        response_time = time.time() - start_time
        log_msg = format_log_message(
            level="ERROR",
            message="Unexpected error",
            query=query,
            response_time=response_time,
            error=str(result),
        )
        logger.error(log_msg)
        return False
    else:
        response_time = time.time() - start_time
        log_msg = format_log_message(
            level="INFO",
            message="Success",
            query=query,
            status=200,
            response_time=response_time,
            additional_info={"result": str(result)[:200]},
        )
        logger.info(log_msg)
        return True
