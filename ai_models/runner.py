import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple
from scrapers.article_extractor import ArticleExtractor
from pprint import pprint


class SummarizationRunner:
    def __init__(
        self, urls: List[str], extractor: ArticleExtractor, graph, max_workers: int = 6
    ):
        self.urls = urls
        self.extractor = extractor
        self.graph = graph
        self.max_workers = max_workers

    def prepare_texts(self) -> List[Tuple[str, str]]:
        return self.extractor.search(self.urls)

    def run_graph(self, text: str, worker_id: int) -> dict:
        return self.graph.invoke({"input_text": text, "worker_id": worker_id})

    def parallel_run(self, pairs: List[Tuple[str, str]]):
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(self.run_graph, text, idx)
                for idx, (url, text) in enumerate(pairs)
            ]
            for idx, future in enumerate(as_completed(futures), 1):
                try:
                    result = future.result()
                    results.append(result)
                    print(
                        f"✅ {idx}/{len(pairs)} 완료 | 점수: {result.get('score', '-')}"
                    )
                except Exception as e:
                    print(f"❌ {idx}/{len(pairs)} 실패: {e}")
        return results

    def run(self):
        start = time.time()
        pairs = self.prepare_texts()
        print(f"🔍 {len(pairs)}개 기사 추출 완료. 요약 시작!")
        results = self.parallel_run(pairs)
        print("\n🎯 전체 완료!")
        for i, res in enumerate(results, 1):
            pprint(res)
        print(f"\n⏱️ 총 소요 시간: {time.time() - start:.2f}s")
