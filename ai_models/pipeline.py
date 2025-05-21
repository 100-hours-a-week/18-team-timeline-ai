import asyncio
from ai_models.host import Host, SystemRole
from scrapers.article_extractor import ArticleExtractor, ArticleParser, ArticleFilter
from ai_models.manager import BatchManager, wrapper
from ai_models.store import ResultStore


async def Pipeline(
    urls: list[dict[str, str]],
    server,
    model,
    repeat: int = 5,
    roles: list[SystemRole] = [SystemRole.SUMMARIZE],
    batch_size: int = 5,
    max_wait_time: float = 0.5,
) -> ResultStore:

    async with (
        Host(
            server,
            model,
        ) as host,
    ):

        extractor = ArticleExtractor()
        parser = ArticleParser()
        filter = ArticleFilter(top_k=4)
        results_dict = ResultStore()
        manager = BatchManager(host, batch_size=batch_size, max_wait_time=max_wait_time)
        runner = asyncio.create_task(manager.run())
        for item in urls:
            url = item["url"]
            if url:
                results_dict.register({"url": url})
        async with filter:
            async for result in extractor.search(urls):
                parsed = await parser.parse(result)
                key_sentences = await filter.extract_key_sentences(parsed)
                sentence = ". ".join(key_sentences)

                tasks = []
                for _ in range(repeat):
                    for role in roles:
                        tasks.append(
                            asyncio.create_task(
                                wrapper(result["url"], role, sentence, manager)
                            )
                        )

                for task in asyncio.as_completed(tasks):
                    try:
                        url, role, response = await task
                        content = (
                            response["choices"][0]["message"]["content"]
                            if response
                            else "실패"
                        )
                        results_dict.add_result(url=url, role=role, content=content)
                    except Exception as e:
                        import traceback

                        traceback.print_exc()
                        await asyncio.sleep(1)
                        raise e
            manager.running = False
            runner.cancel()
            try:
                await runner
            except asyncio.CancelledError:
                pass

    return results_dict.as_dict()


async def TotalPipeline(
    text: list[str],
    server,
    model,
    repeat: int = 5,
    roles: list[SystemRole] = [SystemRole.SUMMARIZE, SystemRole.TITLE, SystemRole.TAG],
    batch_size: int = 5,
    max_wait_time: float = 0.5,
):
    async with (
        Host(
            server,
            model,
        ) as host,
    ):
        results_dict = ResultStore()
        manager = BatchManager(
            host=host, batch_size=batch_size, max_wait_time=max_wait_time
        )
        runner = asyncio.create_task(manager.run())
        tasks = []
        sentence = ". ".join(text)
        for _ in range(repeat):
            for role in roles:
                tasks.append(
                    asyncio.create_task(
                        wrapper("total_summary", role, sentence, manager)
                    )
                )

        for task in asyncio.as_completed(tasks):
            try:
                url, role, response = await task
                content = (
                    response["choices"][0]["message"]["content"] if response else "실패"
                )
                results_dict.add_result(url=url, role=role, content=content)
            except Exception as e:
                import traceback

                traceback.print_exc()
                await asyncio.sleep(1)
                raise e
        manager.running = False
        runner.cancel()
        try:
            await runner
        except asyncio.CancelledError:
            pass

    return results_dict.as_dict()


if __name__ == "__main__":
    import time

    SERVER = "http://fcab-34-118-242-65.ngrok-free.app"
    MODEL = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
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
    start = time.perf_counter()
    roles = [SystemRole.SUMMARIZE, SystemRole.TAG, SystemRole.TITLE]

    result1 = asyncio.run(Pipeline(URLS, SERVER, MODEL, repeat=1))

    print(result1)
    texts = [s for r in result1.values() for s in r.get("SUMMARIZE", [])]
    print(texts)
    result2 = asyncio.run(TotalPipeline(texts, SERVER, MODEL, repeat=1))
    print(result2)
    end = time.perf_counter()
    print(f"총 실행 시간: {end - start:.2f}s")
