import os
import dotenv
import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from utils.env_utils import get_model, get_server
from models.response_schema import CommonResponse, ErrorResponse
from models.response_schema import CommentRequest, CommentData

from scrapers.daum_vclip_searcher import DaumVclipSearcher
from scrapers.youtube_searcher import YouTubeCommentAsyncFetcher

from ai_models.runner import Runner
from ai_models.graph.classify import ClassifyGraph

# -------------------------------------------------------------------

router = APIRouter()
SERVER = get_server()
MODEL = get_model()

logging.basicConfig(
    level=logging.INFO,  # ← 이 부분이 핵심
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# -------------------------------------------------------------------


async def main(query: str, num: int):
    dotenv.load_dotenv(override=True)
    YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
    REST_API_KEY = os.getenv("REST_API_KEY")
    video_searcher = DaumVclipSearcher(api_key=REST_API_KEY)
    youtube_searcher = YouTubeCommentAsyncFetcher(api_key=YOUTUBE_API_KEY, max_comments=num)
    df = video_searcher.search(query=query)
    print("df")
    print(df)

    ripple = await youtube_searcher.search(df=df)
    return ripple


# -------------------------------------------------------------------


@router.post(
    "",
    response_model=CommonResponse[CommentData],
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def classify_comments(request: CommentRequest):
    # Request parsing
    num = request.num
    if not num:
        num = 3
    query_str = " ".join(request.query)
    query_str += " youtube"

    # 기사 수집
    data = await main(query_str, num)
    if not data:
        return JSONResponse(
            status_code=404,
            content=ErrorResponse(
                success=False,
                message="기사를 찾을 수 없습니다."
            ).model_dump()
        )

    # 그래프 빌드 및 실행
    graph = ClassifyGraph(server=SERVER, model=MODEL).build()
    runner = Runner(graph=graph)
    texts = [{"input_text": d["comment"], "transcript": d["captions"]} for d in data]
    result = runner.run(texts=texts)
    if not result:
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                success=False,
                message="댓글 분류기 내부 에러 발생"
            ).model_dump()
        )

    # 통계
    summary = CommentData(
        positive=0,
        neutral=0,
        negative=0,
    )
    for comment in result:
        emo = comment["emotion"]
        if "긍정" in emo:
            summary.positive += 1
        elif "부정" in emo:
            summary.negative += 1
        else:
            summary.neutral += 1

    # 백분율
    total = summary.positive + summary.neutral + summary.negative
    summary.positive = int(summary.positive * 100 / total)
    summary.negative = int(summary.negative * 100 / total)
    summary.neutral = 100 - summary.positive - summary.negative

    # ----------------------------------------------------

    return CommonResponse(
        success=True, message="데이터가 성공적으로 생성되었습니다.", data=summary
    )
