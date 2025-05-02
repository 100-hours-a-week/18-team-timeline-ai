from fastapi import APIRouter
from models.timeline_card import TimelineCard
from models.response_schema import CommonResponse, ErrorResponse
from models.response_schema import TimelineRequest, TimelineData

router = APIRouter()


@router.post(
    "",
    response_model=CommonResponse[TimelineData],
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    }
)
def get_timeline(request: TimelineRequest):
    # 실제 AI 요약 호출은 생략 (테스트용)
    query_str = " ".join(request.query)
    start_date = request.startAt.strftime("%Y-%m-%d")
    end_date = request.endAt.strftime("%Y-%m-%d")

    # 응답용 더미 타임라인 카드 2개 생성
    dummy_cards = [
        TimelineCard(
            title=f"'{query_str}' 관련 주요 사건 1",
            content="첫 번째 하드코딩된 타임라인 내용입니다.",
            duration="DAY",
            startAt=request.startAt,
            endAt=request.endAt,
            source=["https://example.com/article1"]
        ),
        TimelineCard(
            title=f"'{query_str}' 관련 주요 사건 2",
            content="두 번째 하드코딩된 타임라인입니다.",
            duration="DAY",
            startAt=request.startAt,
            endAt=request.endAt,
            source=["https://example.com/article2"]
        ),
    ]

    # 응답용 더미 타임라인 생성
    dummy_data = TimelineData(
        title=f"{query_str} ({start_date}~{end_date}) 요약",
        summary=f"'{query_str}' 키워드에 대한 {start_date}부터 {end_date}까지의 요약입니다.",
        image="https://example.com/image.jpg",
        category="KTB",  # enum 적용 예정
        timeline=dummy_cards
    )

    return CommonResponse(
        success=True,
        message="Timeline 엔드포인트 테스트용 응답",
        data=dummy_data
    )
