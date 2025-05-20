import asyncio
from aiohttp import ClientConnectorError, ServerDisconnectedError, ClientResponseError
from utils.logger import Logger

logger = Logger.get_logger("utils.handling")


async def handle_http_error(result, query, logger):
    if isinstance(result, ClientConnectorError):
        logger.error(f"[RESPONSE ERROR] 쿼리 '{query}' 실패: {str(result)}")
        await asyncio.sleep(3)
        return False

    elif isinstance(result, ServerDisconnectedError):
        logger.error(f"[RESPONSE ERROR] 쿼리 '{query}' 실패: {str(result)}")
        await asyncio.sleep(1.5)
        return False

    elif isinstance(result, ClientResponseError):
        logger.error(f"[RESPONSE ERROR] 쿼리 '{query}' 실패: {str(result)}")
        if 500 <= result.status < 600:
            logger.warning(
                f"[RESPONSE WARNING] 서버 내부 오류 {result.status}: '{query}'"
            )
            await asyncio.sleep(1.5)  # 재시도 여지 있음
        elif 400 <= result.status < 500:
            logger.warning(f"[RESPONSE WARNING] 잘못된 요청 {result.status}: '{query}'")
            await asyncio.sleep(1.5)  # 재시도 여지 있음
        else:
            logger.warning(
                f"[RESPONSE WARNING] 기타 HTTP 오류 {result.status}: '{query}'"
            )
        return False
        # 응답 내 error 메시지 처리 (성공처럼 생긴 실패)
    elif isinstance(result, dict) and "error" in result:
        logger.error(f"[RESPONSE ERROR] 쿼리 '{query}' 실패: {result}")
        await asyncio.sleep(2)
        return False

    elif isinstance(result, Exception):
        logger.error(f"[RESPONSE ERROR] 쿼리 '{query}' 실패: {str(result)}")
        return False
    else:
        logger.info(f"[RESPONSE SUCCESS] 쿼리 '{query}' 성공: {str(result)}")
        return True
