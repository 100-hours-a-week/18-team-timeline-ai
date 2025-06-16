import os
import dotenv
from config.limiter import limiter

from util.env_utils import get_serper_key
from util.error_utils import error_response
from util.timeline_utils import (
    convert_tag,
    short_sentence,
    compress_sentence,
    shrink_if_needed,
)

from fastapi import APIRouter, Request
from pipelines.total_pipeline import TotalPipeline
from pipelines.summary_pipeline import Pipeline
from schemas.timeline_card import TimelineCard
from schemas.response_schema import (
    CommonResponse,
    ErrorResponse,
    TimelineData,
    TimelineRequest,
)

from scrapers.url_to_img import get_img_link
from scrapers.serper import distribute_news_serper
from scrapers.filter import DaumKeywordMeaningChecker
from util.logger import Logger

# -------------------------------------------------------------------

router = APIRouter()
logger = Logger.get_logger("api_timeline")

dotenv.load_dotenv(override=True)
SERVER = os.getenv("SERVER")
MODEL = os.getenv("MODEL")
REST_API_KEY = os.getenv("REST_API_KEY")
API_KEY = os.getenv("OPENAI_API_KEY")

checker = DaumKeywordMeaningChecker(REST_API_KEY)
tag_names = ["", "ECONOMY", "ENTERTAINMENT", "SPORTS", "SCIENCE"]
base_img_url = "https://github.com/user-attachments/assets/"
img_links = [
    "1eeef1f6-3e0a-416a-bc4d-4922b27db855",
    "6cf88794-2743-4dd1-858c-4fcd76f8f107",
    "35ee8d58-b5d8-47c0-82e8-38073f4193eb",
    "3f4248cb-7d8d-4532-a71a-2346e8a82957",
    "e3b550d9-1d62-4940-b942-5b431ba6674e",
]

# -------------------------------------------------------------------


@router.post(
    "",
    response_model=CommonResponse[TimelineData],
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
@limiter.limit("30/minute;1000/day")
async def get_timeline(request: Request, payload: TimelineRequest):
    """타임라인을 생성하는 엔드포인트

    Args:
        request (Request): FastAPI 요청 객체
        payload (TimelineRequest): 타임라인 요청 데이터

    Returns:
        Dict: 타임라인 데이터
    """

    # Request parsing
    query_str = " ".join(payload.query)

    # Meaningful checking
    if not checker.is_meaningful(query_str):
        return await error_response(404, "기사가 나오지 않는 검색어입니다.")

    # Scraping
    SERPER_API_KEY = get_serper_key(0)
    if not SERPER_API_KEY:
        return await error_response(500, "SERPER_API_KEY를 찾을 수 없습니다.")
    scraping_res = distribute_news_serper(
        query=query_str,
        startAt=payload.startAt,
        endAt=payload.endAt,
        api_key=SERPER_API_KEY,
    )

    # Scraping result parsing
    if scraping_res:
        urls, titles, dates = zip(*scraping_res)
        urls = list(urls)
        titles = list(titles)
        dates = list(dates)
        scraping_list = [{"url": u, "title": t} for u, t in zip(urls, titles)]
    else:
        return await error_response(404, "스크래핑에 실패했습니다.")

    # 1st Summarization
    summary = []
    first_res = await Pipeline(scraping_list, SERVER, MODEL)
    if not first_res:
        return await error_response(500, "인공지능 1차 요약 실패!")
    for i, url in enumerate(urls):
        data = first_res.get(url)
        if not data or not data.get("summary"):
            logger.error(f'기사 내용이 없습니다! "{titles[i][:15]}..."')
            return await error_response(500, "인공지능 1차 요약 도중 빈 요약 반환")
        else:
            summary.append(data["summary"][0])

    # Timeline cards
    card_list = []
    for i, title in enumerate(titles):
        news_title = short_sentence(title)
        logger.info(f"[제목 {i+1}] {news_title}")

        card = TimelineCard(
            title=news_title,
            content=compress_sentence(summary[i]),
            duration="DAY",
            startAt=dates[i],
            endAt=dates[i],
            source=[urls[i]],
        )
        card_list.append(card)

    # 2nd Summarization
    total_texts = [card.content for card in card_list]
    total_texts = shrink_if_needed(total_texts)

    final_res = await TotalPipeline(total_texts, API_KEY)
    if not final_res or not final_res["total_summary"]:
        return await error_response(500, "인공지능 2차 요약 실패!")

    # Tag extraction
    final_res = final_res["total_summary"]
    print(f"원본 제목: {final_res["title"][0]}")
    total_title = short_sentence(final_res["title"][0])
    tag_id = convert_tag(request.app.state.classifier.classify(total_title))

    # Image Extraction
    img_link = get_img_link(urls[0])
    if not img_link:
        img_link = base_img_url + img_links[tag_id]

    # Timeline
    timeline = TimelineData(
        title=total_title,
        summary=short_sentence(final_res["summary"][0]),
        image=img_link,
        category=tag_names[tag_id],
        timeline=card_list,
    )

    # ----------------------------------------------------

    return CommonResponse(
        success=True, message="데이터가 성공적으로 생성되었습니다.", data=timeline
    )
