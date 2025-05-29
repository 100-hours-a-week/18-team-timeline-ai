import asyncio
import logging
from utils.logger import Logger

from ai_models.host import Host, SystemRole
from scrapers.article_extractor import ArticleExtractor
from ai_models.manager import BatchManager, wrapper
from ai_models.store import ResultStore

logger = Logger.get_logger("ai_models.pipeline", log_level=logging.ERROR)


async def Pipeline(
    urls: list[dict[str, str]],
    server,
    model,
    repeat: int = 5,
    roles: list[SystemRole] = [SystemRole.summary],
    batch_size: int = 64,
    max_wait_time: float = 2,
) -> ResultStore:

    async with (
        Host(
            server,
            model,
        ) as host,
    ):

        extractor = ArticleExtractor()
        results_dict = ResultStore()
        manager = BatchManager(host, batch_size=batch_size, max_wait_time=max_wait_time)
        runner = asyncio.create_task(manager.run())
        for item in urls:
            url = item["url"]
            if url:
                results_dict.register({"url": url})
        url_sentences = {}
        """
        async with filter:
            async for result in extractor.search(urls):
                parsed = await parser.parse(result)
                key_sentences = await filter.extract_key_sentences(parsed)

                sentence = ". ".join(key_sentences)
                url_sentences[result["url"]] = sentence
        """

        async for result in extractor.search(urls):
            url_sentences[result["url"]] = result["input_text"]

        tasks = []
        for url, input_text in url_sentences.items():
            # logging.info("URL %s | Sentence %s", url, input_text[:10])
            for _ in range(repeat):
                for role in roles:
                    logger.info(f"[PIPELINE] URL :{url}, ROLE : {role}, text : {input_text[:10]}")
                    tasks.append(asyncio.create_task(wrapper(url, role, input_text, manager)))

        try:
            tasks = await asyncio.gather(*tasks, return_exceptions=True)
            manager.running = False
            await asyncio.sleep(1.0)
            for task in tasks:
                url, role, response = task
                logger.info(f"[PIPELINE] URL : {url}")
                if isinstance(response, dict) and "error" in response:
                    logger.error(repr(task))
                    continue

                content = (
                    response["choices"][0]["message"]["content"] if response else "실패"
                )
                results_dict.add_result(url=url, role=role, content=content)
            # logger.info(results_dict.display())

        except Exception as e:
            import traceback

            traceback.print_exc()
            await asyncio.sleep(1)
            raise e
        finally:
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
    roles: list[SystemRole] = [SystemRole.summary, SystemRole.title],
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
        await asyncio.sleep(1.0)
        try:
            await runner
        except asyncio.CancelledError:
            pass

    return results_dict.as_dict()


if __name__ == "__main__":
    import time

    SERVER = "http://1d5d-34-124-161-59.ngrok-free.app"
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
        {
            "url": "https://www.ytn.co.kr/_ln/0134_202505210904491383",
            "title": "다 꺼낸 구글의 '승부수'...삼성 이어 젠틀몬스터와 손 잡는다 [지금이뉴스]",
        },
        {
            "url": "https://m.news.zum.com/articles/98407004/%EC%9D%B4%EB%B2%88%EC%97%94-%EC%93%B8%EB%A7%8C%ED%95%A0%EA%B9%8C-%EA%B5%AC%EA%B8%80-%EC%82%BC%EC%84%B1-%EC%8A%A4%EB%A7%88%ED%8A%B8-%EC%95%88%EA%B2%BD-%ED%98%91%EC%97%85",
            "title": "이번엔 쓸만할까?…구글-삼성 스마트 안경 협업",
        },
        {
            "url": "https://www.news1.kr/world/usa-canada/5789241",
            "title": "챗봇에 흔들린 구글, AI 모드로 검색 기능 강화 나섰다",
        },
        {
            "url": "https://www.newstong.co.kr/view3.aspx?seq=13665995&allSeq=27&txtSearch=&cate=0&cnt=-5&subCate=2&order=default&newsNo=0",
            "title": "구글, 음성·영상으로도 검색 이용…예약 등 에이전트 기능도",
        },
    ]
    start = time.perf_counter()
    roles = [SystemRole.summary, SystemRole.tag, SystemRole.title]

    result1 = asyncio.run(Pipeline(URLS, SERVER, MODEL, repeat=1))
    from pprint import pprint

    print(result1)
    texts = [s for r in result1.values() for s in r.get("summary", [])]
    print(texts)
    result2 = asyncio.run(TotalPipeline(texts, SERVER, MODEL, repeat=1))
    print(result2)
    end = time.perf_counter()
    print(f"총 실행 시간: {end - start:.2f}s")
