import os
import dotenv
import logging

from fastapi import APIRouter
from utils.error_utils import error_response
from models.response_schema import CommonResponse, ErrorResponse
from models.response_schema import CommentRequest, CommentData

from scrapers.daum_vclip_searcher import DaumVclipSearcher
from scrapers.youtube_searcher import YouTubeCommentAsyncFetcher

from classify.embedding import OllamaEmbeddingService
from classify.sentiment import SentimentAggregator

# -------------------------------------------------------------------

router = APIRouter()

dotenv.load_dotenv(override=True)
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
REST_API_KEY = os.getenv("REST_API_KEY")

daum_vclip_searcher = DaumVclipSearcher(api_key=REST_API_KEY)
youtube_searcher = YouTubeCommentAsyncFetcher(api_key=YOUTUBE_API_KEY, max_comments=10)

logging.basicConfig(
    level=logging.INFO,  # ← 이 부분이 핵심
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# -------------------------------------------------------------------


async def main(query: str):
    # 댓글 데이터 수집
    df = daum_vclip_searcher.search(query=query)
    if not df:
        return error_response(404, "DaumVclip 검색 결과가 없습니다!")

    ripple = await youtube_searcher.search(df=df)
    if not ripple:
        return error_response(500, "Youtube 데이터를 불러오는 데 실패했습니다")
    
    ripple = [r["comment"] for r in ripple]
    if not ripple:
        return error_response(404, "Youtube 댓글이 없습니다")

    aggregator = SentimentAggregator()
    ret = await aggregator.aggregate_multiple_queries(
        queries=ripple,
        embedding_constructor=OllamaEmbeddingService,
    )

    # 댓글 데이터 분류
    total = sum(ret.values())
    if total == 0:
        ret["긍정"] = 0
        ret["부정"] = 0
        ret["중립"] = 0
    else:
        ret["긍정"] = int(ret["긍정"] * 100 / total)
        ret["부정"] = int(ret["부정"] * 100 / total)
        ret["중립"] = 100 - ret["긍정"] - ret["부정"]

    return ret


# -------------------------------------------------------------------


@router.post(
    "",
    response_model=CommonResponse[CommentData],
    responses={
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def classify_comments(request: CommentRequest):
    # Request parsing
    num = request.num
    if not num or not isinstance(num, int) or num > 50:
        num = 10
    query_str = " ".join(request.query)

    # 통계
    res = await main(query=query_str)
    if not res:
        return error_response(500, "댓글 분류 실패!")
    summary = CommentData(
        positive=res["긍정"],
        neutral=res["중립"],
        negative=res["부정"],
    )

    # ----------------------------------------------------

    return CommonResponse(
        success=True, message="데이터가 성공적으로 생성되었습니다.", data=summary
    )
