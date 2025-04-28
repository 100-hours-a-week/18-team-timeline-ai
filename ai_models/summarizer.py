import re
import json
import time
from typing import TypedDict, List, Tuple
from textwrap import dedent
from langchain.chat_models import ChatOpenAI
from langchain.schema import BaseOutputParser
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END, START
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.messages import AIMessage, HumanMessage
from newspaper import Article
from langchain.globals import set_debug, get_debug
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate


class ArticleExtractor:
    """기사 URL으로부터 본문을 추출하는 클래스"""

    @staticmethod
    def extract(urls: list[str], max_workers: int = 6) -> list[Tuple[str, str]]:
        """기사의 URL 리스트를 받아 본문을 추출하는 함수
        Args:
            URLs (list[str]): 기사 링크들의 리스트
            max_workers (int): 동시에 작업할 스레드 수 (기본값 6)

        Returns:
            list[str]: 각 기사 본문 텍스트 리스트 ("NOT"은 제외)
        """

        def _extract_single(url: str) -> Tuple[str, str]:
            try:
                article = Article(url=url, language="ko")
                article.download()
                article.parse()
                text = article.text.strip()
                if text and text != "NOT":
                    return text
                else:
                    return None
            except Exception as e:
                print(f"⚠️ Error while extracting {url}: {e}")
                return None

        texts = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_extract_single, url): url for url in urls}
            for future in as_completed(futures):
                url = futures[future]  # ✅ 어떤 url에 대한 future였는지 꺼냄
                text = future.result()
                if text:
                    texts.append((url, text))  # ✅ (url, text) 형태로 추가

        return texts


class GraphState(TypedDict):
    input_text: str
    summary: str
    score: int
    worker_id: int
    retry_count: int
    status: str


class SummaryScoreParser(BaseOutputParser):
    def parse(self, text: str):
        cleaned = re.sub(r"^```json\s*|\s*```$", "", text.strip(), flags=re.MULTILINE)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            print(f"⚠️ JSONDecodeError: {e} → fallback 시도")
            # 콤마 누락이나 사소한 오류를 간단히 복구 시도
            cleaned = cleaned.replace('}"', '", "').replace("}{", "},{")
            return json.loads(cleaned)


class SummarizationGraph:
    """Summarization + Evaluation + Retry Graph 생성기"""

    def __init__(self, server, model, examples: list, max_retries: int = 3):
        self.llm = ChatOpenAI(
            base_url=f"{server}/v1",
            api_key="not-needed",
            model=model,
            temperature=0.3,
        )
        self.examples = examples
        self.max_retries = max_retries

    def summarize_node(self, state: GraphState) -> GraphState:
        """📝 뉴스 요약 노드"""
        system_prompt = """
        당신은 뉴스 요약 전문가입니다.
        - 3줄 이내, 완결된 문장, 핵심 사실만 요약하세요.
        - 예측, 해석, 사견은 금지합니다.
        """
        prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), *self.examples, ("human", "{input_text}")]
        )
        runnable = prompt | self.llm
        result = runnable.invoke({"input_text": state["input_text"]})

        retry_count = state.get("retry_count", 0)
        return {"summary": result.content, "retry_count": retry_count}

    def evaluate_node(self, state: GraphState) -> GraphState:
        """📊 요약 평가 노드"""
        eval_prompt = ChatPromptTemplate(
                messages=[
                    SystemMessagePromptTemplate.from_template(
                        """
                        You are a strict JSON evaluator for news summaries.
                        Respond ONLY with JSON object: \'{{\"summary\": "...", \"score\": 숫자}}\'.
                        
                        - 90~100: 문장에 의견이 들어가지 않고 문법 상 어색함이 없으며 핵심 사실을 정확히 요약함.
                        - 70~89: 대체로 좋음 (약간의 어색함이나 불명확한 부분이 있을 수 있음)
                        - 50~69: 불완전 (핵심 누락 또는 문법적 문제가 존재함)
                        - 0~49: 실패 (요약이 원문과 거의 무관하거나 문법이 심각하게 어색함)
                        """
                    ),
                    HumanMessagePromptTemplate.from_template(
                        "원문:\n{input_text}\n\n요약:\n{summary}"
                    ),
                ],
                input_variables=["input_text", "summary"],
            )

        runnable = eval_prompt | self.llm | SummaryScoreParser()
        try:
            result = runnable.invoke({
                "input_text": state["input_text"],
                "summary": state["summary"],
            })
            score = result["score"]
        except Exception as e:
            print(f"평가 실패 발생: {e} → score = 0 으로 처리하고 재시도 예정")
            score = 0  # 에러 났으면 실패로 간주

        state["score"] = score
        return state

    def check_score(self, state: GraphState) -> str:
        """Score에 따라 성공/재시도/실패 분기"""
        score = state["score"]
        retries = state.get("retry_count", 0)

        if int(score) >= 85:
            return "save"
        elif retries < self.max_retries:
            return "retry"
        else:
            return "log_fail"

    def retry_node(self, state: GraphState) -> GraphState:
        """⚙️ 재시도 시 파라미터 조정"""
        retry_count = state.get("retry_count", 0) + 1
        state["retry_count"] = retry_count
        return state

    def log_fail_node(self, state: GraphState) -> GraphState:
        """🪵 실패 기록 (품질 문제 / 시스템 오류 판단)"""
        score = state.get("score", 0)
        if score < 50:
            state["fail_reason"] = "품질 문제"
        else:
            state["fail_reason"] = "시스템 오류"
        return state

    def save_node(self, state: GraphState) -> GraphState:
        """성공한 요약 저장"""
        state["status"] = "saved"
        return state

    def build(self):
        """LangGraph 전체 컴파일"""
        graph = StateGraph(GraphState)

        # 노드 추가

        graph.add_node("summarize", self.summarize_node)
        graph.add_node("evaluate", self.evaluate_node)
        graph.add_node("retry", self.retry_node)
        graph.add_node("log_fail", self.log_fail_node)
        graph.add_node("save", self.save_node)

        # 흐름 연결
        graph.add_edge(START, "summarize")
        graph.add_edge("summarize", "evaluate")
        graph.add_conditional_edges(
            "evaluate",
            self.check_score,
            {
                "save": "save",
                "retry": "retry",
                "log_fail": "log_fail",
            },
        )

        graph.add_edge("retry", "summarize")
        graph.add_edge("save", END)
        graph.add_edge("log_fail", END)

        return graph.compile()


examples = [
    HumanMessage(
        """
    도널드 트럼프 미국 대통령은 반도체를 비롯한 전자제품에도 관세를 부과하겠다는 입장을 재확인하며 관세 정책에 후퇴가 없음을 시사했다.
    트럼프 대통령은 13일(현지시간) 자신의 사회관계망서비스(SNS)에 올린 글에서 "지난 금요일(4월 11일)에 발표한 것은 관세 예외(exception)가 아니다. 이들 제품은 기존 20% 펜타닐 관세를 적용받고 있으며 단지 다른 관세 범주(bucket)로 옮기는 것"이라고 밝혔다.
    이어 "우리는 다가오는 국가 안보 관세 조사에서 반도체와 전자제품 공급망 전체를 들여다볼 것"이라고 말했다.
    앞서 트럼프 대통령은 지난 11일 대통령 각서에서 상호관세에서 제외되는 반도체 등 전자제품 품목을 구체적으로 명시했고, 관세 징수를 담당하는 세관국경보호국(CBP)이 같은 날 이를 공지했다.
    이에 따라 반도체 등 전자제품은 미국이 중국에 부과한 125% 상호관세, 그리고 한국을 비롯한 나머지 국가에 부과한 상호관세(트럼프 대통령의 유예 조치로 7월 8일까지는 10% 기본관세만 적용)를 내지 않아도 된다.
    다만 미국이 마약성 진통제인 펜타닐의 미국 유입 차단에 협조하지 않는다는 이유로 중국에 별도 행정명령을 통해 부과한 20% 관세는 여전히 적용받는다.
    이를 두고 미국 언론과 업계에서는 트럼프 대통령이 강경 기조에서 한발 물러나 전자제품은 아예 관세에서 면제하는 게 아니냐는 관측이 제기됐으며, 민주당 등에서는 정책에 일관성이 없다고 비판했다.
    그러자 관세를 담당하는 트럼프 행정부 당국자들은 이날 방송에 출연해 반도체 등 전자제품은 지난 2일 발표한 국가별 상호관세에서 제외될 뿐 앞으로 진행할 '무역확장법 232조' 조사를 통해 관세를 부과할 방침이라고 설명했다.
    반도체 등 국가 안보에 중요한 품목은 앞서 25% 관세를 부과한 철강이나 자동차와 마찬가지로 상호관세와 중첩되지 않는 품목별 관세를 부과하겠다는 것이다.
    트럼프 대통령도 이날 관세 강행 의지를 피력했다.
    그는 "다른 나라들이 우리를 상대로 이용한 비(非)금전적 관세 장벽 및 불공정한 무역수지와 관련해 누구도 봐주지 않겠다(Nobody is getting off the hook). 
    특히 우리를 최악으로 대우하는 중국은 봐주지 않겠다"고 밝혔다.
    트럼프 대통령은 "우리는 제품을 미국에서 만들어야 하며 우리는 다른 나라에 인질로 잡히지 않을 것이다. 특히 중국같이 미국민을 무시하기 위해 가진 모든 권력을 이용할 적대적인 교역국에 대해 그렇다"라고 강조했다.
    """
    ),
    AIMessage(
        dedent(
            """

            트럼프 대통령은 관세 정책에 변화가 없음을 재확인하며, 반도체 및 전자제품에 대한 기존 20% 펜타닐 관세를 유지하고 국가 안보 조사 시 반도체와 전자제품 공급망을 검토할 것임을 발표하였습니다. 
            이는 중국에 부과된 상호관세의 일부 면제에도 불구한 조치입니다. 
            트럼프 대통령은 모든 국가, 특히 미국을 최악으로 대우하는 중국에 대해 관세 정책을 계속 추진할 것임을 강조했습니다.
            """
        )
    ),
]


class SummarizationRunner:
    """URL 리스트를 입력 받아 병렬로 요약/평가하는 실행 클래스"""

    def __init__(self, urls: List[str], server: str, model: str, max_workers: int = 6):
        self.urls = urls
        self.max_workers = max_workers
        self.graph = SummarizationGraph(server, model, examples=examples).build()

    def prepare_texts(self) -> List[Tuple[str, str]]:
        return ArticleExtractor.extract(self.urls, max_workers=self.max_workers)

    def run_graph(self, text: str, worker_id: int) -> GraphState:
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
                    print(f"✅ {idx}/{len(pairs)} 완료 | 점수: {result['score']}")
                except Exception as e:
                    print(f"❌ {idx}/{len(pairs)} 실패: {e}")
        return results

    def run(self):
        start = time.time()
        pairs = self.prepare_texts()
        print(f"🔍 추출 완료. {len(pairs)}개 기사 요약 시작!")
        results = self.parallel_run(pairs)
        print("\n🎯 전체 완료!")
        for i, res in enumerate(results, 1):
            print(f"\n📝 요약 {i}")
            print(json.dumps(res, indent=2, ensure_ascii=False))
        print(f"\n⏱️ 총 소요 시간: {time.time() - start:.2f}s")


# 🔧 메인 함수
def main():
    URLS = [
        "https://www.hani.co.kr/arti/society/society_general/1192251.html",
        "https://www.hani.co.kr/arti/society/society_general/1192255.html",
        "https://www.hankyung.com/article/2025041493977",
        "https://www.khan.co.kr/article/202504141136001",
        "https://www.mk.co.kr/news/politics/11290687",
        "https://www.chosun.com/politics/politics_general/2025/04/14/THWVKUHQG5CKFJF6CLZLP5PKM4",
    ]
    SERVER = "https://4e45-34-87-129-244.ngrok-free.app"
    MODEL = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B"

    runner = SummarizationRunner(URLS, SERVER, MODEL)
    runner.run()


if __name__ == "__main__":
    main()
