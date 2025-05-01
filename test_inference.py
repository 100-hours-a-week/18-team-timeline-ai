from ai_models.runner import Runner
from scrapers.article_extractor import ArticleExtractor
from pprint import pprint
from ai_models.graph.total_summary import TotalSummarizationGraph
from ai_models.graph.Summary import SummarizationGraph


def test_summary_graph():
    URLS = [
        "https://www.hani.co.kr/arti/society/society_general/1192251.html",
        "https://www.hani.co.kr/arti/society/society_general/1192255.html",
        "https://www.hankyung.com/article/2025041493977",
        "https://www.khan.co.kr/article/202504141136001",
        "https://www.mk.co.kr/news/politics/11290687",
        "https://www.chosun.com/politics/politics_general/2025/04/14/THWVKUHQG5CKFJF6CLZLP5PKM4",
    ]
    SERVER = "https://0e71-34-139-129-239.ngrok-free.app"
    MODEL = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"

    # 그래프 구성
    graph = SummarizationGraph(SERVER, MODEL).build()
    graph_total = TotalSummarizationGraph(SERVER, MODEL).build()

    # 기사 추출
    extractor = ArticleExtractor()
    texts = extractor.search(urls=URLS)

    # 1차 요약: 기사별 요약
    runner = Runner(graph=graph)
    first_stage_results = runner.run(texts=texts)

    # 2차 요약: 요약문들을 통합 요약
    summarized_texts = [r["summary"] for r in first_stage_results]
    final_runner = Runner(graph=graph_total)
    final_results = final_runner.run(texts=summarized_texts)

    pprint(final_results)


if __name__ == "__main__":
    test_summary_graph()
