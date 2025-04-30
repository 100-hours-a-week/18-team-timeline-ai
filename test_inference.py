from ai_models.graph.Summary import SummarizationGraph
from ai_models.runner import Runner
from scrapers.article_extractor import ArticleExtractor
from pprint import pprint


def main():
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
    graph = SummarizationGraph(SERVER, MODEL).build()
    runner = Runner(graph=graph)
    extractor = ArticleExtractor()
    texts = extractor.search(urls=URLS)
    results = runner.run(texts=texts)
    
    pprint(results)


if __name__ == "__main__":
    main()
