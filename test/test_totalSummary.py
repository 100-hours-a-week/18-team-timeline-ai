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
                "url": "https://people.com/donald-trump-reacts-joe-biden-aggressive-prostate-cancer-diagnosis-11737239",
                "title": "Former president Joe Biden diagnosed with aggressive prostate cancer",
            },
            {
                "url": "https://www.washingtonpost.com/politics/2025/05/18/biden-prostate-cancer",
                "title": "Joe Biden diagnosed with 'aggressive' prostate cancer",
            },
            {
                "url": "https://theconversation.com/joe-biden-has-advanced-prostate-cancer-with-a-gleason-score-of-9-what-does-this-mean-256998",
                "title": "Joe Biden has advanced prostate cancer with a Gleason score of 9. What does this mean?",
            },
        ]
        URLS = assign_id_from_URLS(URLS)
        start_time = time.time()
        runner = ArticleExtractor(lang="en")
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
        SERVER = "https://80b6-34-125-151-224.ngrok-free.app"
        MODEL = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
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
