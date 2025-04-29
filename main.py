from fastapi import FastAPI
from api import router as api_router

app = FastAPI(title="AI News Timeline API", version="1.0.0")


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