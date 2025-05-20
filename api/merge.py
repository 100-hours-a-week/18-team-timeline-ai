import os
import dotenv

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from utils.timeline_utils import next_timeline_type
from utils.timeline_utils import shrink_if_needed

from models.response_schema import CommonResponse, ErrorResponse
from models.response_schema import MergeRequest
from models.timeline_card import TimelineCard

from ai_models.runner import Runner
from ai_models.graph.total_summary import TotalSummarizationGraph

# -------------------------------------------------------------------

router = APIRouter()

dotenv.load_dotenv(override=True)
SERVER = os.getenv("SERVER")
MODEL = os.getenv("MODEL")

graph_total = TotalSummarizationGraph(SERVER, MODEL).build()
final_runner = Runner(graph=graph_total)

# -------------------------------------------------------------------


@router.post(
    "",
    response_model=CommonResponse[TimelineCard],
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
def merge_timeline(request: MergeRequest):
    # Request exception
    if not request.timeline:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                success=False, message="Timeline이 비어 있습니다."
            ).model_dump(),
        )

    # Request parsing
    imgs = []
    contents = []
    cards = request.timeline

    for card in cards:
        imgs.extend(card.source)
        contents.append(card.content)
    contents = shrink_if_needed(contents)
    concat_content = {"input_text": "\n\n".join(contents)}
    final_res = final_runner.run(texts=[concat_content])[0]

    # Merged card
    merged_card = TimelineCard(
        title=final_res["title"],
        content=final_res["summary"],
        duration=next_timeline_type(cards[0].duration),
        startAt=cards[0].startAt,
        endAt=cards[len(cards) - 1].endAt,
        source=imgs,
    )
    merged_card.startAt = merged_card.startAt.date().isoformat()
    merged_card.endAt = merged_card.endAt.date().isoformat()

    # ----------------------------------------------------

    return CommonResponse(
        success=True, message="데이터가 성공적으로 생성되었습니다.", data=merged_card
    )
