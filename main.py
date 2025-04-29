from fastapi import FastAPI
from api import router as api_router
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="AI News Timeline API", version="1.0.0")

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow_origins=["https://backend.com"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 라우터 등록
app.include_router(api_router, prefix="/api")

# 테스트용 엔드포인트
@app.get("/ping")
def ping():
    return {"status": "ok"}


'''
from summarization import summarize
URL = "https://www.yna.co.kr/view/AKR20250414002900071?site=popularnews_view"
ret = summarize(URL=URL)
print(ret)
'''