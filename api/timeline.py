import os
import dotenv
import logging
from limiter import limiter

from utils.env_utils import get_serper_key
from utils.error_utils import error_response
from utils.timeline_utils import convert_tag, short_sentence
from utils.timeline_utils import compress_sentence, shrink_if_needed

from fastapi import APIRouter, Request

from models.timeline_card import TimelineCard
from models.response_schema import CommonResponse, ErrorResponse
from models.response_schema import TimelineRequest, TimelineData

from scrapers.url_to_img import get_img_link
from scrapers.serper import distribute_news_serper
from scrapers.filter import DaumKeywordMeaningChecker
from ai_models.pipeline import Pipeline, TotalPipeline


# -------------------------------------------------------------------

router = APIRouter()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

dotenv.load_dotenv(override=True)
SERVER = os.getenv("SERVER")
MODEL = os.getenv("MODEL")
REST_API_KEY = os.getenv("REST_API_KEY")

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
    # Request parsing
    query_str = " ".join(payload.query)

    # Meaningful checking
    if not checker.is_meaningful(query_str):
        return error_response(404, "기사가 나오지 않는 검색어입니다.")

    # Scraping
    SERPER_API_KEY = get_serper_key(0)
    if not SERPER_API_KEY:
        return error_response(500, "SERPER_API_KEY를 찾을 수 없습니다.")
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
        return error_response(404, "스크래핑에 실패했습니다.")

    # 1st Summarization
    summary = []
    first_res = await Pipeline(scraping_list, SERVER, MODEL, repeat=1)
    if not first_res:
        return error_response(500, "인공지능 1차 요약 실패!")
    for i, url in enumerate(urls):
        data = first_res[url]
        if not data or not data["summary"]:
            logging.error(f'기사 내용이 없습니다! "{titles[i][:15]}..."')
            return error_response(500, "인공지능 1차 요약 도중 빈 요약 반환")
        else:
            summary.append(data["summary"][0])

    # Timeline cards
    card_list = []
    for i, title in enumerate(titles):
        news_title = short_sentence(title)
        logging.info(f"[제목 {i+1}] {news_title}")

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
    final_res = await TotalPipeline(total_texts, SERVER, MODEL, repeat=1)
    if not final_res or not final_res["total_summary"]:
        error_response(500, "인공지능 2차 요약 실패!")

    # Tag extraction
    final_res = final_res["total_summary"]
    tag_id = convert_tag(final_res["tag"][0])

    # Image Extraction
    img_link = get_img_link(urls[0])
    if not img_link:
        img_link = base_img_url + img_links[tag_id]

    # Timeline
    timeline = TimelineData(
        title=short_sentence(final_res["title"][0]),
        summary=short_sentence(final_res["summary"][0]),
        image=img_link,
        category=tag_names[tag_id],
        timeline=card_list,
    )

    # ----------------------------------------------------

    return CommonResponse(
        success=True, message="데이터가 성공적으로 생성되었습니다.", data=timeline
    )
