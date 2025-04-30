from fastapi import APIRouter
from models.response_schema import CommonResponse, ErrorResponse
from models.response_schema import CommentRequest, CommentData

router = APIRouter()

@router.post(
    "",
    response_model=CommonResponse[CommentData],
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    }
)
def classify_comments(request: CommentRequest):
    # 실제 감성 분석 모델은 생략하고 테스트용 하드코딩

    return CommonResponse(
        success=True,
        message="Comment 엔드포인트 테스트용 응답",
        data=CommentData(
            positive=request.num,
            neutral=0,
            negative=0
        )
    )
