import time
from pprint import pprint
import asyncio
from ai_models.runner import Runner
from scrapers.article_extractor import (
    ArticleExtractor,
    assign_id_from_URLS,
    ArticleParser,
    ArticleFilter,
)
from ai_models.graph.total_summary import TotalSummarizationGraph
from ai_models.graph.Summary import SummarizationGraph
from line_profiler import profile

if __name__ == "__main__":

    @profile
    async def main():
        URLS = [
            {
                "url": "https://www.hani.co.kr/arti/society/society_general/1192251.html",
                "title": "말 바꾼 윤석열 “계엄 길어야 하루”…헌재선 “며칠 예상”",
            },
            {
                "url": "https://www.hani.co.kr/arti/society/society_general/1192255.html",
                "title": "윤석열 40분간 “계엄은 평화적 메시지”…판사도 발언 ‘시간조절’ 당부",
            },
            {
                "url": "https://www.hankyung.com/article/2025041493977",
                "title": "'[속보] 韓대행 '국무위원들과 제게 부여된 마지막 소명 다할 것'",
            },
        ]
        URLS = assign_id_from_URLS(URLS)
        start_time = time.time()
        runner = ArticleExtractor()
        parser = ArticleParser()
        filter = ArticleFilter(top_k=2)
        results = [None] * len(URLS)
        async with filter:
            async for result in runner.search(urls=URLS):
                parsed_result = await parser.parse(result)
                key_sentences = await filter.extract_key_sentences(parsed_result)
                results[result["id"]] = {
                    "title": result["title"],
                    "input_text": ". ".join(key_sentences),
                    "url": result["url"],
                }
        results = [r for r in results if r["input_text"] is not None]
        # SERVER = "http://35.216.120.155:8001"
        # MODEL = "models/HyperCLOVAX-SEED-Text-Instruct-1.5B"
        SERVER = "https://b530-34-34-56-0.ngrok-free.app"
        MODEL = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"
        graph = SummarizationGraph(SERVER, MODEL).build()
        graph_total = TotalSummarizationGraph(SERVER, MODEL).build()
        runner = Runner(graph=graph)
        results = await runner.run(results)
        pprint(results)
        texts = [r["text"] for r in results]
        texts = "\n".join(texts)
        runner = Runner(graph=graph_total)
        results = await runner.run([{"input_text": texts}])
        pprint(results)
        end_time = time.time()
        print(f"총 소요 시간: {end_time - start_time}초")

    asyncio.run(main())
