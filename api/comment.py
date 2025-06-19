import os
import dotenv

from fastapi import APIRouter
from utils.error_utils import error_response
from schemas.response_schema import (
    CommonResponse,
    ErrorResponse,
    CommentRequest,
    CommentData,
)

from scrapers.daum_vclip_searcher import DaumVclipSearcher
from scrapers.youtube_searcher import YouTubeCommentAsyncFetcher

from services.classify import SentimentAggregator
from utils.logger import Logger
from inference.embedding import OllamaEmbeddingService

router = APIRouter()
logger = Logger.get_logger("api_comment")

dotenv.load_dotenv(override=True)
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
REST_API_KEY = os.getenv("REST_API_KEY")

daum_vclip_searcher = DaumVclipSearcher(api_key=REST_API_KEY)
youtube_searcher = YouTubeCommentAsyncFetcher(api_key=YOUTUBE_API_KEY, max_comments=10)


async def main(query_str: str):
    # 유튜브 영상 링크 찾기
    df = daum_vclip_searcher.search(query=query_str+" 유튜브")
    if not df:
        logger.warning(f"DaumVclip: {query_str+" 유튜브"} 검색 결과 없음, 다른 방법으로 재시도")
        df = daum_vclip_searcher.search(query=query_str)
    if not df:
        logger.warning(f"DaumVclip: {query_str} 검색 결과가 없습니다!")
        return error_response(404, "DaumVclip 검색 결과가 없습니다!")

    # 유튜브 댓글 추출하기
    ripple = await youtube_searcher.search(df=df)
    if not ripple:
        logger.error("Youtube 데이터를 불러오는 데 실패했습니다")
        return error_response(500, "Youtube 데이터를 불러오는 데 실패했습니다")

    ripple = [r["comment"] for r in ripple]
    if not ripple:
        logger.warning("Youtube 댓글이 없습니다")
        return error_response(404, "Youtube 댓글이 없습니다")

    # 댓글 분류하기
    async with OllamaEmbeddingService() as embedder:
        async with SentimentAggregator(embedder=embedder) as aggregator:
            ret = await aggregator.aggregate_multiple_queries(queries=ripple)

            total = sum(ret.values())
            if total == 0:
                result = {"긍정": 0, "부정": 0, "중립": 0}
            else:
                result = {
                    "긍정": int(ret["긍정"] * 100 / total),
                    "부정": int(ret["부정"] * 100 / total),
                }
                result["중립"] = 100 - result["긍정"] - result["부정"]

            return result


@router.post(
    "",
    response_model=CommonResponse[CommentData],
    responses={
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def classify_comments(request: CommentRequest):
    num = request.num
    if not num or not isinstance(num, int) or num > 50:
        num = 10
    query_str = " ".join(request.query)

    res = await main(query=query_str)
    if not res:
        logger.error("댓글 분류 실패!")
        return error_response(500, "댓글 분류 실패!")

    summary = CommentData(
        positive=res["긍정"],
        neutral=res["중립"],
        negative=res["부정"],
    )

    return CommonResponse(
        success=True,
        message="데이터가 성공적으로 생성되었습니다.",
        data=summary,
    )
