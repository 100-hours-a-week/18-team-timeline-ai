import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Any
import logging


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

    def run_graph(self, item: dict) -> dict:
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
        return self.graph.invoke(
            item,
            self.config,
        )

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
        results = [None] * len(texts)
        start = time.time()

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 각 입력에 대한 작업 제출
            futures = {
                executor.submit(self.run_graph, items): idx
                for idx, items in enumerate(texts)
            }
            logging.info(f"{len(texts)}개 작업 시작")

            # 완료된 작업의 결과 수집
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    if result:
                        logging.info(result)
                        results[idx] = result
                    logging.info(f"{idx}/{len(texts)} 완료")
                except Exception as e:
                    logging.error(f"{idx}/{len(texts)} 실패: {e}")

        # 전체 처리 시간 기록
        logging.info(f"{len(texts)}개 작업 완료")
        logging.info(f"\n총 소요 시간: {time.time() - start:.2f}s")
        return [r for r in results if r is not None]