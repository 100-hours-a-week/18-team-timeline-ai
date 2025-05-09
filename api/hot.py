from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from dotenv import load_dotenv
from utils.env_utils import get_serp_key
from scrapers.serpapi import get_trending_keywords
from models.response_schema import CommonResponse, ErrorResponse
from models.response_schema import HotRequest, HotData

router = APIRouter()


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
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                success=False,
                message="SERP_API_KEY 없음"
            ).model_dump()
        )

    # 핫이슈 수집
    keywords = get_trending_keywords(SERP_API_KEY)
    if not keywords:
        return JSONResponse(
            status_code=404,
            content=ErrorResponse(
                success=False,
                message="핫이슈가 없습니다"
            ).model_dump()
        )

    # -------------------------------------------

    count = min(request.num, len(keywords))
    return CommonResponse(
        success=True,
        message="데이터가 성공적으로 생성되었습니다.",
        data=HotData(keywords=keywords[:count]),
    )
