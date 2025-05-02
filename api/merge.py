from fastapi import APIRouter, HTTPException
from utils.timeline_utils import next_duration
from models.response_schema import CommonResponse, ErrorResponse
from models.response_schema import MergeRequest
from models.timeline_card import TimelineCard

from ai_models.runner import Runner
from ai_models.graph.total_summary import TotalSummarizationGraph

# -------------------------------------------------------------------

router = APIRouter()

SERVER = "https://5a09-34-125-119-95.ngrok-free.app"
MODEL = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"
graph_total = TotalSummarizationGraph(SERVER, MODEL).build()
final_runner = Runner(graph=graph_total)

# -------------------------------------------------------------------


@router.post(
    "",
    response_model=CommonResponse[TimelineCard],
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    }
)
def merge_timeline(request: MergeRequest):
    # Request exception
    if not request.timeline:
        raise HTTPException(status_code=400, detail="timeline 데이터가 비어 있습니다.")

    # Request parsing
    imgs = []
    contents = []
    cards = request.timeline

    for card in cards:
        imgs.extend(card.source)
        contents.append(card.content)
    concat_content = {"text": "\n\n".join(contents)}
    final_res = final_runner.run(texts=[concat_content])

    # Merged card
    merged_card = TimelineCard(
        title=final_res['title'],
        content=final_res['summary'],
        duration=next_duration(cards[0].duration),
        startAt=cards[0].startAt,
        endAt=cards[0].endAt,
        source=imgs
    )

    # ----------------------------------------------------

    return CommonResponse(
        success=True,
        message="Merge 엔드포인트 테스트용 응답",
        data=merged_card
    )
