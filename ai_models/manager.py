import asyncio
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from collections import defaultdict, OrderedDict
from ai_models.host import Host, SystemRole
from utils.logger import Logger
import logging
import time
import uuid
from contextlib import asynccontextmanager

logger = Logger.get_logger("ai_models.manager", log_level=logging.ERROR)


class BatchManager:
    """
    AI 모델 요청을 배치로 처리하는 관리자 클래스
    """

    def __init__(
        self,
        host: Host,
        batch_size: int = 256,
        max_wait_time: float = 1.0,
        max_retries: int = 3,
    ):
        """
        초기화

        Args:
            host: AI 모델 호스트
            batch_size: 배치 크기
            max_wait_time: 최대 대기 시간(초)
            max_retries: 최대 재시도 횟수
        """
        self.host = host
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.max_retries = max_retries
        self.input_queue = asyncio.Queue()
        self.output_queue = asyncio.Queue()
        self.pending_tasks = {}
        self.running = False
        self._cleanup_task = None

    async def submit(self, role: SystemRole, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        요청 제출

        Args:
            role: 시스템 역할
            payload: 요청 데이터

        Returns:
            Dict: 응답 데이터
        """
        if not self.running:
            logger.warning("BatchManager가 실행 중이 아닙니다. 자동으로 시작합니다.")
            self.running = True
            self._cleanup_task = asyncio.create_task(self._cleanup_pending_tasks())

        task_id = uuid.uuid4().hex
        queue = asyncio.Queue(maxsize=1)

        if logger.log_level <= logging.DEBUG:
            logger.debug(f"[BatchManager] 요청 제출: {task_id}")

        self.pending_tasks[task_id] = (queue, time.time())
        await self.input_queue.put((task_id, role, payload))

        try:
            # 타임아웃 설정으로 무한정 대기하지 않도록 함
            result = await asyncio.wait_for(queue.get(), timeout=60.0)
            return result
        except asyncio.TimeoutError:
            logger.error(f"[BatchManager] 요청 {task_id}에 대한 응답 대기 시간 초과")
            del self.pending_tasks[task_id]
            return {"error": "응답 대기 시간 초과"}

    async def _cleanup_pending_tasks(self) -> None:
        """
        오래된 대기 태스크 정리 (백그라운드 작업)
        """
        while self.running:
            try:
                current_time = time.time()
                expired_tasks = []

                for task_id, (queue, submit_time) in list(self.pending_tasks.items()):
                    # 5분 이상 대기 중인 태스크는 만료로 처리
                    if current_time - submit_time > 300:
                        expired_tasks.append(task_id)

                for task_id in expired_tasks:
                    if task_id in self.pending_tasks:
                        queue, _ = self.pending_tasks.pop(task_id)
                        await queue.put({"error": "태스크 만료"})
                        logger.warning(f"[BatchManager] 만료된 태스크 정리: {task_id}")

                await asyncio.sleep(60)  # 1분마다 확인
            except Exception as e:
                logger.error(f"[BatchManager] 태스크 정리 중 오류: {e}")
                await asyncio.sleep(60)

    async def _gather_batch(self) -> List[Tuple[str, SystemRole, Dict[str, Any]]]:
        """
        요청 모음

        Returns:
            List[Tuple]: 요청 배치
        """
        batch = []
        start_time = asyncio.get_event_loop().time()

        if logger.log_level <= logging.DEBUG:
            logger.debug(f"[BatchManager] 배치 모음 시작")

        while len(batch) < self.batch_size:
            # 남은 대기 시간 계산
            elapsed = asyncio.get_event_loop().time() - start_time
            timeout = max(0.1, self.max_wait_time - elapsed)

            if timeout <= 0.1:  # 최소 타임아웃 설정
                break

            try:
                item = await asyncio.wait_for(self.input_queue.get(), timeout=timeout)
                batch.append(item)
            except asyncio.TimeoutError:
                break

        if logger.log_level <= logging.DEBUG:
            logger.debug(f"[BatchManager] 배치 모음 완료: {len(batch)}개 항목")

        return batch

    async def run(self) -> None:
        """
        배치 처리 실행
        """
        self.running = True

        try:
            while self.running or not self.input_queue.empty():
                batch = await self._gather_batch()

                if batch:
                    await self.process_batch(batch)
                else:
                    # 배치가 비어있으면 잠시 대기
                    await asyncio.sleep(0.1)
        except Exception as e:
            logger.error(f"[BatchManager] 실행 중 오류: {e}")
            raise
        finally:
            logger.info("[BatchManager] 실행 종료")
            self.running = False

    async def process_batch(
        self, batch: List[Tuple[str, SystemRole, Dict[str, Any]]]
    ) -> None:
        """
        요청 배치 처리

        Args:
            batch: 요청 배치
        """
        if not batch:
            return

        # 요청 처리
        results = await asyncio.gather(
            *[self._process_request(role, payload) for task_id, role, payload in batch],
            return_exceptions=True,
        )

        # 결과 전달
        for (task_id, _, _), result in zip(batch, results):
            queue, _ = self.pending_tasks.get(task_id, (None, None))

            if queue:
                if isinstance(result, Exception):
                    logger.error(
                        f"[BatchManager] 요청 {task_id} 처리 중 오류: {result}"
                    )
                    await queue.put({"error": str(result)})
                else:
                    await queue.put(result)

                # 처리 완료된 태스크 정리
                del self.pending_tasks[task_id]
            else:
                logger.warning(
                    f"[BatchManager] 태스크 {task_id}에 대한 큐를 찾을 수 없음"
                )

    async def _process_request(
        self, role: SystemRole, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        단일 요청 처리 (재시도 로직 포함)

        Args:
            role: 시스템 역할
            payload: 요청 데이터

        Returns:
            Dict: 응답 데이터
        """
        for attempt in range(self.max_retries):
            try:
                return await self.host.query(role, payload)
            except Exception as e:
                logger.warning(
                    f"[BatchManager] 요청 처리 실패 (시도 {attempt+1}/{self.max_retries}): {e}"
                )

                if attempt < self.max_retries - 1:
                    # 지수 백오프 (0.5초, 1초, 2초, ...)
                    await asyncio.sleep(0.5 * (2**attempt))
                else:
                    raise


async def wrapper(
    url: str, role: SystemRole, content: str, manager: BatchManager
) -> Tuple[str, SystemRole, Dict[str, Any]]:
    """
    BatchManager 요청 래퍼 함수

    Args:
        url: 대상 URL
        role: 시스템 역할
        content: 요청 내용
        manager: 배치 관리자

    Returns:
        Tuple: (URL, 역할, 응답)
    """
    try:
        if not content or not content.strip():
            logger.warning(f"[wrapper] 빈 내용 무시: URL={url}, ROLE={role}")
            return url, role, {"error": "빈 내용"}

        result = await manager.submit(role, {"text": content})
        return url, role, result
    except Exception as e:
        logger.error(f"[wrapper] 요청 실패: URL={url}, ROLE={role}, ERROR={e}")
        return url, role, {"error": str(e)}


@asynccontextmanager
async def create_batch_manager(
    host: Host, batch_size: int = 4, max_wait_time: float = 1.0
) -> BatchManager:
    """
    BatchManager 생성 및 관리를 위한 컨텍스트 관리자

    Args:
        host: AI 모델 호스트
        batch_size: 배치 크기
        max_wait_time: 최대 대기 시간(초)

    Yields:
        BatchManager: 배치 관리자 인스턴스
    """
    manager = BatchManager(host, batch_size=batch_size, max_wait_time=max_wait_time)
    runner = asyncio.create_task(manager.run())

    try:
        yield manager
    finally:
        manager.running = False
        await asyncio.sleep(0.5)

        try:
            await asyncio.wait_for(runner, timeout=1.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            logger.warning(
                "[create_batch_manager] Runner 태스크 정리 중 타임아웃 또는 취소됨"
            )


if __name__ == "__main__":
    # 테스트 코드
    from scrapers.article_extractor import (
        ArticleExtractor,
        ArticleParser,
        ArticleFilter,
    )

    async def main():
        # 테스트 설정
        SERVER = "http://example-server.com"  # 실제 서버 URL로 변경 필요
        MODEL = "example-model"  # 실제 모델 이름으로 변경 필요

        # 테스트 URL
        URLS = [
            {
                "url": "https://www.example.com/article1",
                "title": "테스트 기사 1",
            },
            {
                "url": "https://www.example.com/article2",
                "title": "테스트 기사 2",
            },
        ]

        try:
            async with Host(SERVER, MODEL) as host:
                extractor = ArticleExtractor()
                parser = ArticleParser()
                filter = ArticleFilter(top_k=4)

                # 컨텍스트 관리자를 사용한 BatchManager 생성
                async with create_batch_manager(
                    host, batch_size=5, max_wait_time=0.5
                ) as manager:
                    results_dict = OrderedDict()

                    # URL 등록
                    for item in URLS:
                        url = item["url"]
                        if url not in results_dict:
                            results_dict[url] = defaultdict(list)

                    # 텍스트 추출 및 처리
                    async with filter:
                        async for result in extractor.search(URLS):
                            parsed_result = await parser.parse(result)
                            key_sentences = await filter.extract_key_sentences(
                                parsed_result
                            )
                            sentence = ". ".join(key_sentences)

                            # 태스크 생성
                            tasks = []
                            for _ in range(2):  # 테스트를 위해 반복 횟수 감소
                                for role in [
                                    SystemRole.SUMMARY,
                                    SystemRole.TITLE,
                                    SystemRole.TAG,
                                ]:
                                    task = asyncio.create_task(
                                        wrapper(result["url"], role, sentence, manager)
                                    )
                                    tasks.append(task)

                            # 결과 수집 (완료되는 순서대로)
                            for task in asyncio.as_completed(tasks):
                                try:
                                    url, role, result = await task

                                    if isinstance(result, dict) and "error" in result:
                                        print(
                                            f"❌ 요청 실패: URL={url}, ROLE={role}, ERROR={result['error']}"
                                        )
                                        continue

                                    content = (
                                        result["choices"][0]["message"]["content"]
                                        if result
                                        else "실패"
                                    )
                                    print(f"[{url}][{role}] → {content}")
                                    results_dict[url][role].append(content)
                                    print("-" * 50)
                                except Exception as e:
                                    print(f"❌ 태스크 처리 중 오류: {e}")

                    # 결과 출력
                    for url in results_dict:
                        print(f"\n📌 URL: {url}")
                        for role in [
                            SystemRole.SUMMARY,
                            SystemRole.TITLE,
                            SystemRole.TAG,
                        ]:
                            entries = results_dict[url].get(role, [])
                            print(f"  {role}:")
                            for i, entry in enumerate(entries):
                                print(f"    {i+1}. {entry}")
        except Exception as e:
            print(f"테스트 실행 중 오류 발생: {e}")

    # 테스트 실행
    start = time.perf_counter()
    asyncio.run(main())
    end = time.perf_counter()
    print(f"총 실행 시간: {end - start:.2f}s")
