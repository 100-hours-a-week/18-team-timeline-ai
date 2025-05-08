import logging
from utils.env_utils import get_server, get_model, get_serper_key
from utils.timeline_utils import convert_tag, extract_first_sentence

from fastapi import APIRouter, HTTPException
from models.timeline_card import TimelineCard
from models.response_schema import CommonResponse, ErrorResponse
from models.response_schema import TimelineRequest, TimelineData

from scrapers.url_to_img import get_img_link
from scrapers.serper import distribute_news_serper
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

SERVER = get_server()
MODEL = get_model()
graph = SummarizationGraph(SERVER, MODEL).build()
graph_total = TotalSummarizationGraph(SERVER, MODEL).build()

runner = Runner(graph=graph)
final_runner = Runner(graph=graph_total)

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
        500: {"model": ErrorResponse},
    },
)
def get_timeline(request: TimelineRequest):
    # Request parsing
    query_str = " ".join(request.query)

    # Scraping
    SERPER_API_KEY = get_serper_key(0)
    if not SERPER_API_KEY:
        raise HTTPException(status_code=500, detail="SERPER_API_KEY not found")
    scraping_res = distribute_news_serper(query=query_str,
                                          startAt=request.startAt,
                                          endAt=request.endAt,
                                          api_key=SERPER_API_KEY)

    if scraping_res:
        urls, dates = zip(*scraping_res)
        urls = list(urls)
        dates = list(dates)
    else:
        urls, dates = [], []

    if not urls:
        return ErrorResponse(
            success=False, message="기사를 찾을 수 없습니다."
        )

    # Extract Article
    try:
        articles = extractor.search(urls=urls)
    except Exception as e:
        logging.exception("기사 추출 실패")
        return e
    logging.info(f"{len(articles)}개 기사 추출 완료")

    print("기사 추출본입니다.")
    print(articles)

    # 1st Summarization
    first_res = runner.run(texts=articles)
    if not first_res:
        print("1차 요약 실패!")
        return ErrorResponse(
            success=False, message="인공지능 1차 요약 실패"
        )

    # Timeline cards
    card_list = []
    for i, res in enumerate(first_res):
        logging.info(f"[제목 {i+1}] {articles[i]['title']}")
        logging.info(f"[결과 {i+1}] {res['text'][:30]}...")

        card = TimelineCard(
            title=articles[i]["title"],
            content=res["text"],
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
        print("2차 요약 실패!")
        return ErrorResponse(
            success=False, message="인공지능 2차 요약 실패"
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
        title=final_res["title"],
        summary=extract_first_sentence(final_res["summary"]),
        image=img_link,
        category=tag_names[tag_id],
        timeline=card_list,
    )

    # ----------------------------------------------------

    return CommonResponse(
        success=True, message="데이터가 성공적으로 생성되었습니다.", data=timeline
    )
