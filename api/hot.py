import os
from fastapi import APIRouter, HTTPException
from dotenv import load_dotenv
from scrapers.serpapi import get_trending_keywords
from models.response_schema import CommonResponse, ErrorResponse, HotRequest, HotData

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
    load_dotenv()
    SERP_API_KEYS = os.getenv("SERP_API_KEYS")
    if not SERP_API_KEYS:
        raise HTTPException(
            status_code=500, detail="SERP_API_KEYS not found in .env file."
        )

    SERP_API_KEYS = SERP_API_KEYS.split(",")
    SERP_API_KEY = SERP_API_KEYS[0].strip()

    keywords = get_trending_keywords(SERP_API_KEY)
    if not keywords:
        raise HTTPException(status_code=404, detail="핫이슈가 없습니다.")

    count = min(request.num, len(keywords))

    return CommonResponse(
        success=True,
        message="데이터가 성공적으로 생성되었습니다.",
        data=HotData(keywords=keywords[:count]),
    )
