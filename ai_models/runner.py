import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Any
import logging


class Runner:
    def __init__(
        self,
        graph: Any,
        max_workers: int = 6,
    ):
        self.graph = graph
        self.max_workers = max_workers

    def run_graph(self, text: str, worker_id: int) -> dict:
        """_summary_

        Args:
            text (str): _description_
            worker_id (int): _description_

        Returns:
            dict: _description_
        """
        return self.graph.invoke(
            {
                "input_text": text,
                "worker_id": worker_id,
            },
            {"recursion_limit": 1000},
        )

    def run(self, texts: List[dict]) -> List[dict]:
        """
        texts를 스레드로 처리하는 메서드
        run_graph 메서드를 사용하여 각 텍스트를 처리합니다.
        Args:
            texts (List[dict]): 기사 URL, 제목, 본문을 포함하는 딕셔너리 리스트
        - 각 딕셔너리는 {"url": str, "title": str, "text": str} 형식입니다.
        - 혹은 {"text": str} 형식으로만 제공되어도 괜찮습니다.
        Returns:
            List[dict]: 각 text에 대한 처리 결과를 제공하는 딕셔너리 리스트
            딕셔너리의 결과는 graph.invoke 메서드의 StateGraph 객체입니다.
        """
        results = [None] * len(texts)
        start = time.time()

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:

            futures = {
                executor.submit(self.run_graph, items["text"], idx): idx
                for idx, items in enumerate(texts)
            }
            logging.info(f"⏳ {len(texts)}개 작업 시작")
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result = future.result()
                    if result:
                        result["worker_id"] = idx
                        result["source_id"] = 0
                        result["input_text"] = texts[idx]["text"]
                        results[idx] = result
                    logging.info(f"✅ {idx}/{len(texts)} 완료")
                except Exception as e:
                    logging.error(f"❌ {idx}/{len(texts)} 실패: {e}")

        logging.info(f"✅ {len(texts)}개 작업 완료")
        logging.info(f"\n⏱️ 총 소요 시간: {time.time() - start:.2f}s")
        return [r for r in results if r is not None]
