import os
import dotenv

from fastapi import APIRouter
from util.error_utils import error_response
from schemas.response_schema import (
    CommonResponse,
    ErrorResponse,
    CommentRequest,
    CommentData,
)

from scrapers.daum_vclip_searcher import DaumVclipSearcher
from scrapers.youtube_searcher import YouTubeCommentAsyncFetcher

from services.classify import SentimentAggregator
from util.logger import Logger

# -------------------------------------------------------------------

router = APIRouter()
logger = Logger.get_logger("api_comment")

dotenv.load_dotenv(override=True)
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
REST_API_KEY = os.getenv("REST_API_KEY")

daum_vclip_searcher = DaumVclipSearcher(api_key=REST_API_KEY)
youtube_searcher = YouTubeCommentAsyncFetcher(api_key=YOUTUBE_API_KEY, max_comments=10)

# -------------------------------------------------------------------


async def main(query: str):
    # 댓글 데이터 수집
    df = daum_vclip_searcher.search(query=query)
    if not df:
        logger.warning("DaumVclip 검색 결과가 없습니다!")
        return error_response(404, "DaumVclip 검색 결과가 없습니다!")

    ripple = await youtube_searcher.search(df=df)
    if not ripple:
        logger.error("Youtube 데이터를 불러오는 데 실패했습니다")
        return error_response(500, "Youtube 데이터를 불러오는 데 실패했습니다")

    ripple = [r["comment"] for r in ripple]
    if not ripple:
        logger.warning("Youtube 댓글이 없습니다")
        return error_response(404, "Youtube 댓글이 없습니다")

    aggregator = SentimentAggregator()
    ret = await aggregator.aggregate_multiple_queries(
        queries=ripple,
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
        logger.error("댓글 분류 실패!")
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
