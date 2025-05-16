from fastapi.responses import JSONResponse
from models.response_schema import ErrorResponse


def error_response(code, msg: str):
    return JSONResponse(
        status_code=code,
        content=ErrorResponse(
            success=False,
            message=msg
        ).model_dump()
    )
