from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from api.router import router as api_router

from config.limiter import limiter
from slowapi.errors import RateLimitExceeded
from starlette.status import HTTP_429_TOO_MANY_REQUESTS

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

import gc
from contextlib import asynccontextmanager
from services.category import NewsClassifier

# ---------------------------------------------------------
# Monitoring
if not isinstance(trace.get_tracer_provider(), TracerProvider):
    trace.set_tracer_provider(TracerProvider())

otlp_exporter = OTLPSpanExporter()
span_processor = BatchSpanProcessor(otlp_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)
print("Trace insert checking")

# ---------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[Startup] Loading NewsClassifier...")
    app.state.classifier = NewsClassifier(threshold=0.2)
    yield
    print("[Shutdown] Releasing NewsClassifier...")
    del app.state.classifier
    gc.collect()


app = FastAPI(title="AI News Timeline API", version="1.0.0", lifespan=lifespan)
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
