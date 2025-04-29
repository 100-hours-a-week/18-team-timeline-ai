from fastapi import FastAPI
from api.router import router as api_router

app = FastAPI(title="AI News Timeline API", version="1.0.0")
app.include_router(api_router, prefix="/api")

# 테스트용 엔드포인트
@app.get("/ping")
def ping():
    return {"status": "ok"}
