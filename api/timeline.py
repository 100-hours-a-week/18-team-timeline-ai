import os
import dotenv
import logging

from utils.env_utils import get_serper_key
from utils.timeline_utils import convert_tag, short_sentence, compress_sentence

from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded

from models.timeline_card import TimelineCard
from models.response_schema import CommonResponse, ErrorResponse
from models.response_schema import TimelineRequest, TimelineData

from scrapers.url_to_img import get_img_link
from scrapers.serper import distribute_news_serper
from scrapers.filter import DaumKeywordMeaningChecker
from scrapers.article_extractor import ArticleExtractor

from ai_models.runner import Runner
from ai_models.graph.total_summary import TotalSummarizationGraph
from ai_models.graph.Summary import SummarizationGraph

# -------------------------------------------------------------------

router = APIRouter()
extractor = ArticleExtractor()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)

dotenv.load_dotenv(override=True)
SERVER = os.getenv("SERVER")
MODEL = os.getenv("MODEL")
REST_API_KEY = os.getenv("REST_API_KEY")

graph = SummarizationGraph(SERVER, MODEL).build()
graph_total = TotalSummarizationGraph(SERVER, MODEL).build()

runner = Runner(graph=graph)
final_runner = Runner(graph=graph_total)
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
def get_timeline(request: TimelineRequest):
    # Rate limit
    limiter = request.app.state.limiter
    limiter.limit("30/minute;1000/day")(request)

    # Request parsing
    query_str = " ".join(request.query)

    # Meaningful checking
    if not checker.is_meaningful(query_str):
        return JSONResponse(
            status_code=404,
            content=ErrorResponse(
                success=False,
                message="기사가 나오지 않는 검색어입니다."
            ).model_dump()
        )

    # Scraping
    SERPER_API_KEY = get_serper_key(0)
    if not SERPER_API_KEY:
        raise HTTPException(status_code=500, detail="SERPER_API_KEY not found")
    scraping_res = distribute_news_serper(
        query=query_str,
        startAt=request.startAt,
        endAt=request.endAt,
        api_key=SERPER_API_KEY,
    )

    if scraping_res:
        urls, titles, dates = zip(*scraping_res)
        urls = list(urls)
        titles = list(titles)
        dates = list(dates)
        scraping_list = [{"url": u, "title": t} for u, t in zip(urls, titles)]
    else:
        return JSONResponse(
            status_code=404,
            content=ErrorResponse(
                success=False,
                message="스크래핑에 실패했습니다"
            ).model_dump()
        )

    # Extract Article
    try:
        articles = extractor.search(urls=scraping_list)
    except Exception as e:
        logging.exception("기사 추출 실패")
        return e
    logging.info(f"{len(articles)}개 기사 추출 완료")

    # Naver Clova - Maximum 4096 Tokens
    for i, article in enumerate(articles):
        articles[i]["input_text"] = articles[i]["input_text"][:3000]

    # 1st Summarization
    first_res = runner.run(texts=articles)
    if not first_res:
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                success=False, message="인공지능 1차 요약 실패"
            ).model_dump(),
        )
    for i, res in enumerate(first_res):
        if not res["text"]:
            print(f"기사 내용이 없습니다! \"{titles[i][:15]}...\"")
            return JSONResponse(
                status_code=404,
                content=ErrorResponse(
                    success=False, message="인공지능 1차 요약 도중 빈 요약 반환"
                ).model_dump(),
            )

    # Timeline cards
    card_list = []
    for i, res in enumerate(first_res):
        news_title = short_sentence(titles[i])
        logging.info(f"[제목 {i+1}] {news_title}")

        card = TimelineCard(
            title=news_title,
            content=compress_sentence(res["text"]),
            duration="DAY",
            startAt=dates[i],
            endAt=dates[i],
            source=[urls[i]],
        )
        card_list.append(card)

    # 2nd Summarization
    summarized_texts = [r["text"] for r in first_res]
    summarized_texts = {"input_text": "\n\n".join(summarized_texts)}
    final_res = final_runner.run(texts=[summarized_texts])
    if not final_res:
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                success=False, message="인공지능 2차 요약 실패"
            ).model_dump(),
        )

    # Tag extraction
    final_res = final_res[0]
    tag_id = convert_tag(final_res["tag"])

    # Image Extraction
    img_link = get_img_link(urls[0])
    if not img_link:
        img_link = base_img_url + img_links[tag_id]

    # Timeline
    timeline = TimelineData(
        title=short_sentence(final_res["title"]),
        summary=short_sentence(final_res["summary"]),
        image=img_link,
        category=tag_names[tag_id],
        timeline=card_list,
    )

    # ----------------------------------------------------

    return CommonResponse(
        success=True, message="데이터가 성공적으로 생성되었습니다.", data=timeline
    )
