import asyncio
from typing import List, Dict
from collections import defaultdict, OrderedDict
from ai_models.host import Host, SystemRole
from utils.logger import Logger
import logging
import time
import uuid

logger = Logger.get_logger("ai_models.manager", log_level=logging.ERROR)


class BatchManager:
    def __init__(
        self,
        host: Host,
        batch_size: int = 4,
        max_wait_time: float = 1.0,
    ):
        self.host = host
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.input_queue = asyncio.Queue()
        self.output_queue = asyncio.Queue()
        self.pending_tasks = {}
        self.running = False

    async def submit(self, role: SystemRole, payload: dict):
        task_id = uuid.uuid4().hex
        queue = asyncio.Queue(maxsize=1)
        logger.info(f"[BatchManager] 요청 제출: {task_id}")
        self.pending_tasks[task_id] = queue
        logger.info(f"[BatchManager] 요청 제출 완료: {task_id}")
        await self.input_queue.put((task_id, role, payload))
        return await queue.get()

    async def _gather_batch(self) -> List[dict]:
        """
        요청 모음

        Returns:
            List[dict]: 요청 모음
        """
        batch = []
        start = asyncio.get_event_loop().time()
        logger.info(f"[BatchManager] 배치 모음 시작: {batch}")
        while len(batch) < self.batch_size:
            timeout = self.max_wait_time - (asyncio.get_event_loop().time() - start)
            if timeout <= 0:
                logger.info(f"[BatchManager] 배치 모음 완료: {batch}")
                break
            try:
                logger.info(f"[BatchManager] 배치 모음 대기: {batch}")
                item = await asyncio.wait_for(self.input_queue.get(), timeout)
                logger.info(f"[BatchManager] 배치 모음 대기 완료: {batch}")
                batch.append(item)
            except asyncio.TimeoutError:
                logger.info(f"[BatchManager] 배치 모음 완료: {batch}")
                break
        logger.info(f"[BatchManager] 배치 모음 완료: {batch}")
        return batch

    async def run(self):
        self.running = True
        while self.running or not self.input_queue.empty():
            batch = await self._gather_batch()
            if batch:
                await self.process_batch(batch)

    async def process_batch(self, batch: List[tuple]):
        """
        요청 처리

        Args:
            batch (List[dict]): 요청 모음
        """
        results = await asyncio.gather(
            *[self.host.query(role, payload) for task_id, role, payload in batch]
        )

        # 결과 전달
        for (task_id, _, _), result in zip(batch, results):
            queue = self.pending_tasks.get(task_id)
            if queue:
                await queue.put(result)
                del self.pending_tasks[task_id]


async def wrapper(url, role, content, manager):
    try:
        result = await manager.submit(role, {"text": content})
        return url, role, result
    except Exception as e:
        import traceback

        traceback.print_exc()
        return url, role, {"error": str(e)}


async def main():
    from scrapers.article_extractor import (
        ArticleExtractor,
        ArticleParser,
        ArticleFilter,
    )

    async with (
        Host(
            "http://fcab-34-118-242-65.ngrok-free.app",
            "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct",
        ) as host,
    ):
        extractor = ArticleExtractor()
        parser = ArticleParser()
        filter = ArticleFilter(top_k=4)
        manager = BatchManager(host, batch_size=5, max_wait_time=0.5)
        runner = asyncio.create_task(manager.run())
        results_dict = OrderedDict()

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
        for item in URLS:
            url = item["url"]
            if url not in results_dict:
                results_dict[url] = defaultdict(list)

        async with filter:
            async for result in extractor.search(URLS):

                parsed_result = await parser.parse(result)
                key_sentences = await filter.extract_key_sentences(parsed_result)
                sentence = ". ".join(key_sentences)

                tasks = []
                for _ in range(10):
                    for role in [
                        SystemRole.SUMMARIZE,
                        SystemRole.TITLE,
                        SystemRole.TAG,
                    ]:

                        task = asyncio.create_task(
                            wrapper(result["url"], role, sentence, manager)
                        )  # 👈 wrapper는 반드시 Task로 만들어야 함
                        tasks.append(task)

                # 결과가 오는 순서대로 스트리밍 출력
                for task in asyncio.as_completed(tasks):
                    try:
                        url, role, result = await task
                        content = (
                            result["choices"][0]["message"]["content"]
                            if result
                            else "실패"
                        )
                        print(f"[{url}][{str(role)}] → {content}")
                        results_dict[url][role].append(content)
                        print("-" * 100)
                    except Exception as e:
                        import traceback

                        traceback.print_exc()
                        print(f"❌ 요청 실패: {repr(e)}")

                        await asyncio.sleep(1)
                        manager.running = False
                        runner.cancel()
                        try:
                            await runner
                        except asyncio.CancelledError:
                            pass

        for url in results_dict:
            print(f"\n📌 URL: {url}")
            for role in [
                SystemRole.SUMMARIZE,
                SystemRole.TITLE,
                SystemRole.TAG,
            ]:
                entries = results_dict[url].get(role, [])
                print(f"  {role.name}:")
                for i, entry in enumerate(entries):
                    print(f"    {i+1}. {entry}")


if __name__ == "__main__":
    start = time.perf_counter()
    asyncio.run(main())
    end = time.perf_counter()
    print(f"총 실행 시간: {end - start:.2f}s")
