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
        self._cleanup_task = None
        self._lock = asyncio.Lock()  # 동시성 제어를 위한 락 추가

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
            logger.warning("[BatchManager] 실행 중이 아닙니다. 자동으로 시작합니다.")
            self.running = True
            # 이미 실행 중인 cleanup 태스크가 없을 때만 새로 생성
            if self._cleanup_task is None or self._cleanup_task.done():
                self._cleanup_task = asyncio.create_task(self._cleanup_pending_tasks())

    async def submit(self, role: SystemRole, payload: dict):
        task_id = uuid.uuid4().hex
        queue = asyncio.Queue(maxsize=1)

        if logger.log_level <= logging.DEBUG:
            logger.debug(f"[BatchManager] 요청 제출: {task_id}")

        # 락을 사용하여 pending_tasks 딕셔너리 접근 동기화
        async with self._lock:
            self.pending_tasks[task_id] = (queue, time.time())

        try:
            await self.input_queue.put((task_id, role, payload))
            # 타임아웃 설정으로 무한정 대기하지 않도록 함
            result = await asyncio.wait_for(queue.get(), timeout=60.0)
            return result
        except asyncio.TimeoutError:
            logger.error(f"[BatchManager] 요청 {task_id}에 대한 응답 대기 시간 초과")
            # 락을 사용하여 pending_tasks 딕셔너리 접근 동기화
            async with self._lock:
                if task_id in self.pending_tasks:
                    del self.pending_tasks[task_id]
            return {"error": "[BatchManager] 응답 대기 시간 초과"}
        except Exception as e:
            logger.error(f"[BatchManager] 요청 {task_id} 처리 중 예외 발생: {e}")
            # 락을 사용하여 pending_tasks 딕셔너리 접근 동기화
            async with self._lock:
                if task_id in self.pending_tasks:
                    del self.pending_tasks[task_id]
            return {"error": f"[BatchManager] 예외 발생: {str(e)}"}

    async def _cleanup_pending_tasks(self) -> None:
        """
        오래된 대기 태스크 정리 (백그라운드 작업)
        30초마다 실행되어 5분 이상 대기 중인 태스크를 정리합니다.
        """
        while self.running:
            try:
                current_time = time.time()
                expired_tasks = []

                # 락을 사용하여 pending_tasks 딕셔너리 접근 동기화
                async with self._lock:
                    for task_id, (queue, submit_time) in list(
                        self.pending_tasks.items()
                    ):
                        # 5분 이상 대기 중인 태스크는 만료로 처리
                        if current_time - submit_time > 300:
                            expired_tasks.append((task_id, queue))

                # 락 밖에서 만료된 태스크 처리
                for task_id, queue in expired_tasks:
                    try:
                        await queue.put({"error": "태스크 만료"})
                        logger.warning(f"[BatchManager] 만료된 태스크 정리: {task_id}")

                        # 락을 사용하여 pending_tasks 딕셔너리 접근 동기화
                        async with self._lock:
                            if task_id in self.pending_tasks:
                                del self.pending_tasks[task_id]
                    except Exception as e:
                        logger.error(
                            f"[BatchManager] 만료된 태스크 {task_id} 정리 중 오류: {e}"
                        )

                await asyncio.sleep(30)  # 30초마다 확인
            except Exception as e:
                logger.error(f"[BatchManager] 태스크 정리 중 오류: {e}")
                await asyncio.sleep(30)  # 오류 발생해도 계속 실행

    async def _gather_batch(self) -> List[Tuple[str, SystemRole, Dict[str, Any]]]:
        """
        요청 모음

        Returns:
            List[dict]: 요청 모음
        """
        batch = []
        start_time = time.time()
        if logger.log_level <= logging.DEBUG:
            logger.debug("[BatchManager] 배치 모음 시작")

        try:
            # 첫 번째 항목은 최대 30초 대기
            try:
                item = await asyncio.wait_for(self.input_queue.get(), timeout=30)
                batch.append(item)
            except asyncio.TimeoutError:
                # 타임아웃 발생 시 빈 배치 반환
                return []

            # 배치 크기에 도달하거나 최대 대기 시간을 초과할 때까지 항목 수집
            while len(batch) < self.batch_size:
                elapsed = time.time() - start_time
                if elapsed >= self.max_wait_time:
                    break

                try:
                    # get_nowait() 대신 짧은 타임아웃으로 get() 사용
                    # QueueEmpty 예외 방지
                    remaining_time = max(0.01, self.max_wait_time - elapsed)
                    item = await asyncio.wait_for(
                        self.input_queue.get(), timeout=remaining_time
                    )
                    batch.append(item)
                except asyncio.TimeoutError:
                    # 더 이상 항목이 없거나 시간 초과
                    break
                except Exception as e:
                    logger.error(f"[BatchManager] 배치 모음 중 오류: {e}")
                    break
        except Exception as e:
            logger.error(f"[BatchManager] 배치 모음 중 예외 발생: {e}")
            # 예외가 발생해도 수집된 항목은 처리

        if logger.log_level <= logging.DEBUG:
            logger.debug(f"[BatchManager] 배치 모음 완료: {len(batch)}개 항목")

        return batch

    async def run(self):
        self.running = True
        idle_count = 0
        try:
            while self.running or not self.input_queue.empty():
                batch = await self._gather_batch()

                if batch:
                    idle_count = 0
                    await self.process_batch(batch)
                else:
                    idle_count += 1
                    # 지수 백오프로 대기 시간 조정 (최대 5초)
                    wait_time = min(5.0, 0.1 * (2 ** min(idle_count, 6)))
                    await asyncio.sleep(wait_time)
        except Exception as e:
            logger.error(f"[BatchManager] 실행 중 오류: {str(e)}")
            # 실행 중 오류가 발생해도 정상 종료 처리
            self.running = False
            raise
        finally:
            logger.info("[BatchManager] 실행 종료")
            self.running = False
            # cleanup 태스크 정리
            if self._cleanup_task and not self._cleanup_task.done():
                self._cleanup_task.cancel()
                try:
                    await self._cleanup_task
                except asyncio.CancelledError:
                    pass

    async def process_batch(
        self, batch: List[Tuple[str, SystemRole, Dict[str, Any]]]
    ) -> None:
        """
        요청 처리

        Args:
            batch (List[dict]): 요청 모음
        """
        results = await asyncio.gather(
            *[
                self._process_request(task_id, role, payload)
                for task_id, role, payload in batch
            ],
            return_exceptions=True,
        )

        # 결과 전달
        for (task_id, _, _), result in zip(batch, results):
            try:
                # 락을 사용하여 pending_tasks 딕셔너리 접근 동기화
                async with self._lock:
                    queue_info = self.pending_tasks.pop(task_id, None)

                if queue_info:
                    queue, _ = queue_info
                    if isinstance(result, Exception):
                        logger.error(
                            f"[BatchManager] 요청 {task_id} 처리 중 오류: {result}"
                        )
                        await queue.put({"error": str(result)})
                    else:
                        await queue.put(result)
                else:
                    logger.warning(
                        f"[BatchManager] 태스크 {task_id}에 대한 큐를 찾을 수 없음"
                    )
            except Exception as e:
                logger.error(
                    f"[BatchManager] 결과 전달 중 오류 (태스크 {task_id}): {e}"
                )

    async def _process_request(
        self, task_id: str, role: SystemRole, payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        단일 요청 처리 (재시도 로직 포함)

        Args:
            task_id: 태스크 ID
            role: 시스템 역할
            payload: 요청 데이터

        Returns:
            Dict: 응답 데이터
        """
        last_error = None
        for attempt in range(self.max_retries):
            try:
                # 빈 페이로드 검사 추가
                if not payload or (
                    isinstance(payload, dict) and not payload.get("text")
                ):
                    logger.warning(f"[BatchManager] 빈 페이로드 감지: {task_id}")
                    return {"error": "빈 페이로드"}

                # 비동기 함수로 변환하여 호출
                return await self.host.query(role, payload)
            except Exception as e:
                last_error = e
                logger.warning(
                    f"[BatchManager] 요청 처리 실패 (시도 {attempt+1}/{self.max_retries}): {e}"
                )

                if attempt < self.max_retries - 1:
                    # 지수 백오프 (0.5초, 1초, 2초, ...)
                    await asyncio.sleep(0.5 * (2**attempt))

        # 모든 재시도 실패 시
        error_msg = f"최대 재시도 횟수({self.max_retries}) 초과: {last_error}"
        logger.error(f"[BatchManager] {task_id} - {error_msg}")
        return {"error": error_msg}


async def wrapper(url, role, content, manager):
    try:
        result = await manager.submit(role, {"text": content})
        return url, role, result
    except Exception as e:
        import traceback

        traceback.print_exc()
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
        # 정상적인 종료 처리
        manager.running = False
        await asyncio.sleep(0.5)

        try:
            # 최대 5초 대기로 변경하여 정상 종료 기회 제공
            await asyncio.wait_for(runner, timeout=5.0)
        except (asyncio.TimeoutError, asyncio.CancelledError):
            logger.warning("[BatchManager] Runner 태스크 정리 중 타임아웃 또는 취소됨")
            # 태스크 강제 취소
            if not runner.done():
                runner.cancel()
                try:
                    await runner
                except asyncio.CancelledError:
                    pass


if __name__ == "__main__":
    # 테스트 코드
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

                    # 텍스트 추출 및 처리
                    async with filter:
                        async for result in extractor.search(URLS):
                            try:
                                parsed_result = await parser.parse(result)
                                key_sentences = await filter.extract_key_sentences(
                                    parsed_result
                                )

                                # 빈 결과 처리
                                if not key_sentences:
                                    print(f"⚠️ 키 문장 추출 실패: URL={result['url']}")
                                    continue

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
                                            wrapper(
                                                result["url"], role, sentence, manager
                                            )
                                        )
                                        tasks.append(task)

                                # 결과 수집 (완료되는 순서대로)
                                for task in asyncio.as_completed(tasks):
                                    try:
                                        url, role, result = await task

                                        if (
                                            isinstance(result, dict)
                                            and "error" in result
                                        ):
                                            print(
                                                f"❌ 요청 실패: URL={url}, ROLE={role}, ERROR={result['error']}"
                                            )
                                            continue

                                        content = (
                                            result["choices"][0]["message"]["content"]
                                            if result
                                            and "choices" in result
                                            and result["choices"]
                                            else "실패"
                                        )
                                        print(f"[{url}][{role}] → {content}")
                                        results_dict[url][role].append(content)
                                        print("-" * 50)
                                    except Exception as e:
                                        print(f"❌ 태스크 처리 중 오류: {e}")
                            except Exception as e:
                                print(f"❌ 결과 처리 중 오류: {e}")

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

            print("✅ 테스트 완료")
        except Exception as e:
            print(f"❌ 테스트 실행 중 오류 발생: {e}")

    # 테스트 실행
    start = time.perf_counter()
    asyncio.run(main())
    end = time.perf_counter()
    print(f"총 실행 시간: {end - start:.2f}s")
