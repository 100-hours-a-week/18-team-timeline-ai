from fastapi.responses import JSONResponse
from schemas.response_schema import ErrorResponse
import asyncio


async def error_response(code, msg: str):
    await asyncio.sleep(0.1)  # 100ms 지연
    return JSONResponse(
        status_code=code, content=ErrorResponse(success=False, message=msg).model_dump()
    )
