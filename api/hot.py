from fastapi import APIRouter
from dotenv import load_dotenv

from util.env_utils import get_serp_key
from util.error_utils import error_response
from util.logger import Logger

from scrapers.serpapi import get_trending_keywords
from schemas.response_schema import CommonResponse, ErrorResponse, HotRequest, HotData

router = APIRouter()
logger = Logger.get_logger("api.hot")


@router.post(
    "",
    response_model=CommonResponse[HotData],
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
def get_hot_topics(request: HotRequest):
    # SerpAPI
    load_dotenv()
    SERP_API_KEY = get_serp_key(0)
    if not SERP_API_KEY:
        return error_response(500, "SERP_API_KEY를 찾을 수 없습니다.")

    # 핫이슈 수집
    keywords = get_trending_keywords(SERP_API_KEY)
    if not keywords:
        return error_response(404, "핫이슈가 없습니다.")

    # -------------------------------------------

    count = min(request.num, len(keywords))
    return CommonResponse(
        success=True,
        message="데이터가 성공적으로 생성되었습니다.",
        data=HotData(keywords=keywords[:count]),
    )
