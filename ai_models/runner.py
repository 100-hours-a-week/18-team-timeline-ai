import time
import asyncio
from typing import List, Any, Generator
from utils.logger import Logger
from tqdm import tqdm

logger = Logger.get_logger("ai_runner")


class Runner:
    """AI 모델 실행기

    AI 모델의 그래프를 병렬로 실행하고 결과를 수집하는 클래스입니다.
    ThreadPoolExecutor를 사용하여 여러 입력을 동시에 처리합니다.

    Attributes:
        graph (Any): 실행할 AI 모델 그래프
        max_workers (int): 동시에 처리할 최대 스레드 수
        config (dict): 그래프 실행에 필요한 설정값
    """

    def __init__(
        self,
        graph: Any,
        max_workers: int = 5,
        batch_size: int = 32,
        config: dict = None,
    ):
        """Runner 초기화

        Args:
            graph (Any): 실행할 AI 모델 그래프
            max_workers (int, optional): 동시에 처리할 최대 스레드 수. Defaults to 6.
            config (dict, optional): 그래프 실행에 필요한 설정값. Defaults to None.
                설정값이 제공되지 않으면 기본값 {"recursion_limit": 100}이 사용됩니다.
        """
        self.graph = graph
        self.max_workers = max_workers
        self.batch_size = batch_size
        self.config = {"recursion_limit": 100}
        if config:
            self.config.update(config)

    def _create_batches(self, items: List[dict]) -> Generator[List[dict], None, None]:
        """입력 데이터를 배치로 나누는 제너레이터

        Args:
            items (List[dict]): 배치로 나눌 입력 데이터 리스트

        Yields:
            List[dict]: 배치 단위의 입력 데이터
        """
        for i in range(0, len(items), self.batch_size):
            logger.info(f"[Runner] 배치 생성 - {i + 1}/{len(items)}")
            yield items[i : i + self.batch_size]

    async def run_graph(self, item: dict) -> dict:
        """단일 입력에 대해 그래프를 실행

                Args:
                    item (dict): 그래프에 입력할 데이터
                        - {"url": str, "title": str, "text": str} 형식
                        - 또는 {"input_text": str} 형식
        ß
                Returns:
                    dict: 그래프 실행 결과
                        결과는 graph.invoke 메서드의 StateGraph 객체입니다.
        """
        try:
            logger.info(f"[Runner] 그래프 실행 시작 - 입력: {item.get('url', 'N/A')}")
            result = await self.graph.ainvoke(item, self.config)
            logger.debug(f"[Runner] 그래프 실행 성공 - 입력: {item.get('url', 'N/A')}")
            return result
        except Exception as e:
            logger.error(
                f"[Runner] 그래프 실행 실패 - 입력: {item.get('url', 'N/A')}, "
                f"[Runner] 에러: {type(e).__name__}: {str(e)}"
            )
            raise

    def run(self, texts: List[dict]) -> List[dict]:
        """여러 입력을 병렬로 처리하는 메서드

        Args:
            batch (List[dict]): 처리할 배치 데이터

        Returns:
            List[dict]: 배치 처리 결과
        """
        results = [None] * len(batch)
        try:
            logger.info(f"[Runner] 세마포어 생성 - {self.max_workers}")
            semaphore = asyncio.Semaphore(self.max_workers)
            logger.info(f"[Runner] 세마포어 생성 완료 - {self.max_workers}")
        except Exception as e:
            logger.error(f"[Runner] 세마포어 생성 실패: {e}")
            raise

        async def process_with_semaphore(idx: int, item: dict):
            async with semaphore:
                try:
                    logger.info(f"[Runner] 그래프 실행 시작 - {idx + 1}/{len(batch)}")
                    result = await self.run_graph(item)
                    logger.info(f"[Runner] 그래프 실행 완료 - {idx + 1}/{len(batch)}")
                    if result:
                        results[idx] = result
                        logger.info(f"[Runner] 작업 완료 - {idx + 1}/{len(batch)}")
                except Exception as e:
                    logger.error(
                        f"[Runner] 작업 실패 - {idx + 1}/{len(batch)}, "
                        f"[Runner] 에러: {type(e).__name__}: {str(e)}"
                    )

        try:
            logger.info(f"[Runner] 그래프 실행 시작 - {len(batch)}")
            tasks = [
                process_with_semaphore(idx, item) for idx, item in enumerate(batch)
            ]
            await asyncio.gather(*tasks)
            logger.info(f"[Runner] 그래프 실행 완료 - {len(batch)}")
            return [r for r in results if r is not None]
        except Exception as e:
            logger.error(f"[Runner] 그래프 실행 실패: {e}")
            return []

    async def run(self, texts: List[dict]) -> List[dict]:
        """여러 입력을 배치 단위로 비동기 처리하는 메서드

        asyncio를 사용하여 여러 입력을 배치 단위로 동시에 처리합니다.
        각 배치는 run_batch 메서드를 통해 처리되며, 결과는 입력 순서대로 반환됩니다.

        Args:
            texts (List[dict]): 처리할 입력 데이터 리스트
            반드시 사용하는 graph 객체와 키를 맞춰서 입력해주세요.

        Returns:
            List[dict]: 각 입력에 대한 처리 결과 리스트
                실패한 입력의 결과는 None으로 처리되며, 최종 결과에서 제외됩니다.

        Note:
            - 처리 시간과 진행 상황이 로그로 기록됩니다.
            - 실패한 입력에 대한 오류도 로그로 기록됩니다.
        """
        if not texts:
            logger.warning("[Runner] 입력 데이터가 비어있습니다.")
            return []

        start = time.time()
        logger.info(f"[Runner] 입력 데이터 개수: {len(texts)}")
        all_results = []
        batches = list(self._create_batches(texts))
        # 배치 단위로 처리
        for batch_idx, batch in enumerate(tqdm(batches, desc="배치"), 1):
            logger.info(
                f"[Runner] 배치 처리 시작 - {batch_idx}번째 배치 ({len(batch)}개)"
            )
            batch_start = time.time()

            # 배치 처리
            try:
                batch_results = await self.run_batch(batch)
                all_results.extend(batch_results)
            except Exception as e:
                logger.error(f"[Runner] 배치 처리 실패: {e}")
                raise

            # 배치 처리 시간 기록
            batch_time = time.time() - batch_start
            logger.info(
                f"[Runner] 배치 처리 완료 - {batch_idx}번째 배치, "
                f"[Runner] 성공: {len(batch_results)}/{len(batch)}, "
                f"[Runner] 소요시간: {batch_time:.2f}초"
            )

        # 전체 처리 시간 기록
        elapsed_time = time.time() - start
        success_count = len(all_results)
        logger.info(
            f"[Runner] 작업 완료 - 전체: {len(texts)}, 성공: {success_count}, "
            f"[Runner] 실패: {len(texts) - success_count}, "
            f"[Runner] 소요시간: {elapsed_time:.2f}초"
        )
        return all_results
