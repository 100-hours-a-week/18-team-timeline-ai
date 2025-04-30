from fastapi import APIRouter, HTTPException
from models.response_schema import CommonResponse, ErrorResponse
from models.response_schema import MergeRequest
from models.timeline_card import TimelineCard

router = APIRouter()

@router.post(
    "",
    response_model=CommonResponse[TimelineCard],
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    }
)
def merge_timeline(request: MergeRequest):
    # 실제 AI 요약 호출은 생략 (테스트용)
    if not request.timeline:
        raise HTTPException(status_code=400, detail="timeline 데이터가 비어 있습니다.")

    # 실제 병합 대신 첫 번째 카드만 응답
    first_card = request.timeline[0]

    return CommonResponse(
        success=True,
        message="Merge 엔드포인트 테스트용 응답",
        data=first_card
    )
