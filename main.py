from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from api.router import router as api_router

from limiter import limiter
from slowapi.errors import RateLimitExceeded
from starlette.status import HTTP_429_TOO_MANY_REQUESTS

app = FastAPI(title="AI News Timeline API", version="1.0.0")
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=HTTP_429_TOO_MANY_REQUESTS,
        content={"detail": "현재 요청이 너무 많습니다. 잠시 후 다시 시도해 주세요."},
    )


# Router
app.include_router(api_router, prefix="/api")


# 테스트용 엔드포인트
@app.get("/ping")
def ping():
    return {"status": "ok"}
