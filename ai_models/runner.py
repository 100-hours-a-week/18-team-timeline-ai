import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Any


class Runner:
    def __init__(
        self,
        graph: Any,
        max_workers: int = 6,
    ):
        self.graph = graph
        self.max_workers = max_workers

    def run_graph(self, text: str, worker_id: int) -> dict:
        """단일 입력을 graph에 넣고 결과 반환"""
        return self.graph.invoke({
            "input_text": text,
            "worker_id": worker_id,
        })

    def run(self, texts: List) -> List[dict]:
        """모든 입력에 대해 병렬로 graph 실행"""
        results = []
        start = time.time()

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self.run_graph, text, idx)
                for idx, (source_id, text) in enumerate(texts)
            ]
            for idx, future in enumerate(as_completed(futures), 1):
                try:
                    result = future.result()
                    results.append(result)
                    print(f"✅ {idx}/{len(texts)} 완료")
                except Exception as e:
                    print(f"❌ {idx}/{len(texts)} 실패: {e}")

        print(f"\n⏱️ 총 소요 시간: {time.time() - start:.2f}s")
        return results
