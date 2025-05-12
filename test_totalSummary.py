import logging
import time
from pprint import pprint

from ai_models.runner import Runner
from scrapers.article_extractor import ArticleExtractor
from ai_models.graph.total_summary import TotalSummarizationGraph
from ai_models.graph.Summary import SummarizationGraph
import dotenv

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(name)s: %(message)s")


def test_summary_graph():
    URLS = [
        {
            "url": "https://www.hani.co.kr/arti/society/society_general/1192251.html",
            "title": "조선일보",
        },
        {
            "url": "https://www.hani.co.kr/arti/society/society_general/1192255.html",
            "title": "조선일보",
        },
        {"url": "https://www.hankyung.com/article/2025041493977", "title": "한겨레"},
        {"url": "https://www.khan.co.kr/article/202504141136001", "title": "경향신문"},
        {"url": "https://www.mk.co.kr/news/politics/11290687", "title": "매일경제"},
        {
            "url": "https://www.chosun.com/politics/politics_general/2025/04/14/THWVKUHQG5CKFJF6CLZLP5PKM4",
            "title": "조선일보",
        },
    ]
    SERVER = "http://35.216.120.155:8001"
    MODEL = "models/HyperCLOVAX-SEED-Text-Instruct-1.5B"

    logging.info("📦 모델 및 그래프 초기화 중...")
    graph = SummarizationGraph(SERVER, MODEL).build()
    graph_total = TotalSummarizationGraph(SERVER, MODEL).build()

    logging.info("📰 기사 본문 추출 시작...")
    extractor = ArticleExtractor()
    start_time = time.time()
    len_title = 0
    try:
        articles = extractor.search(urls=URLS)
        logging.info(articles)
        for article in articles:
            len_title += len(article["title"])
        len_title /= len(articles)
        logging.info(f"평균 제목 길이: {len_title}")
    except Exception as e:
        logging.exception("❌ 기사 추출 실패:")
        return
    logging.info(
        f"✅ {len(articles)}개 기사 본문 추출 완료 (소요 시간: {time.time() - start_time:.2f}s)"
    )
    exit(0)
    len_text = 0
    logging.info("📄 1차 요약(개별 기사) 시작...")
    runner = Runner(graph=graph)
    first_results = runner.run(texts=articles)
    for i, res in enumerate(first_results):
        len_text += len(res["text"])
        logging.info(f"📝 [1차 제목 {i+1}] {articles[i]['title']}")
        logging.info(f"📝 [1차 결과 {i+1}] {res['text'][:60]}...")
        logging.info(f"📝 [1차 본문 {i+1}] {res['input_text'][:60]}...")
    len_text /= len(first_results)
    logging.info(f"평균 본문 길이: {len_text}")
    summarized_texts = [r["text"] for r in first_results]
    summarized_texts = {"input_text": "\n\n".join(summarized_texts)}

    logging.info("📚 2차 요약(통합 요약) 시작...")
    final_runner = Runner(graph=graph_total)
    final_results = final_runner.run(texts=[summarized_texts])
    len_text_final = len(final_results[0]["summary"])
    len_title_final = len(final_results[0]["title"])
    logging.info(f"평균 최종 요약 길이: {len_text_final}")
    logging.info(f"📝 [2차 결과] {len_title_final}...")
    logging.info("✅ 최종 통합 요약 결과:")
    logging.info(final_results)
    pprint(final_results)


if __name__ == "__main__":
    dotenv.load_dotenv(override=True)
    test_summary_graph()
