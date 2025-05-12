import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Any
from utils.logger import Logger

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
        max_workers: int = 6,
        config: dict = None,
    ):
        """Runner 초기화

        Args:
            graph (Any): 실행할 AI 모델 그래프
            max_workers (int, optional): 동시에 처리할 최대 스레드 수. Defaults to 6.
            config (dict, optional): 그래프 실행에 필요한 설정값. Defaults to None.
                설정값이 제공되지 않으면 기본값 {"recursion_limit": 1000}이 사용됩니다.
        """
        self.graph = graph
        self.max_workers = max_workers
        self.config = {"recursion_limit": 1000}
        if config:
            self.config.update(config)
            logger.info(f"설정 업데이트: {config}")

    def run_graph(self, item: dict) -> dict:
        """단일 입력에 대해 그래프를 실행

        Args:
            item (dict): 그래프에 입력할 데이터
                - {"url": str, "title": str, "text": str} 형식
                - 또는 {"input_text": str} 형식

        Returns:
            dict: 그래프 실행 결과
                결과는 graph.invoke 메서드의 StateGraph 객체입니다.
        """
        try:
            result = self.graph.invoke(item, self.config)
            logger.debug(f"그래프 실행 성공 - 입력: {item.get('url', 'N/A')}")
            return result
        except Exception as e:
            logger.error(
                f"그래프 실행 실패 - 입력: {item.get('url', 'N/A')}, "
                f"에러: {type(e).__name__}: {str(e)}"
            )
            raise

    def run(self, texts: List[dict]) -> List[dict]:
        """여러 입력을 병렬로 처리하는 메서드

        ThreadPoolExecutor를 사용하여 여러 입력을 동시에 처리합니다.
        각 입력은 run_graph 메서드를 통해 처리되며, 결과는 입력 순서대로 반환됩니다.

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
            logger.warning("입력 데이터가 비어있습니다.")
            return []

        results = [None] * len(texts)
        start = time.time()

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 각 입력에 대한 작업 제출
            futures = {
                executor.submit(self.run_graph, items): idx
                for idx, items in enumerate(texts)
            }
            logger.info(f"작업 시작 - 총 {len(texts)}개")

            # 완료된 작업의 결과 수집
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    if result:
                        results[idx] = result
                        logger.info(f"작업 완료 - {idx + 1}/{len(texts)}")
                except Exception as e:
                    logger.error(
                        f"작업 실패 - {idx + 1}/{len(texts)}, "
                        f"에러: {type(e).__name__}: {str(e)}"
                    )

        # 전체 처리 시간 기록
        elapsed_time = time.time() - start
        success_count = len([r for r in results if r is not None])
        logger.info(
            f"작업 완료 - 전체: {len(texts)}, 성공: {success_count}, "
            f"실패: {len(texts) - success_count}, "
            f"소요시간: {elapsed_time:.2f}초"
        )
        return [r for r in results if r is not None]
