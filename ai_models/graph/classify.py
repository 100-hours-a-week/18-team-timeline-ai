from typing import List
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_community.utilities import WikipediaAPIWrapper
from textwrap import dedent
from pydantic import BaseModel
from pprint import pprint
import logging

# 로그 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 시스템 프롬프트 정의
system_prompt = dedent("""
You are an AI assistant designed to support Korean timeline-based question answering and comment classification.
You have access to the following tools:

1. `search_wiki`: Use this when a person's name or major entity is mentioned. It searches Wikipedia in Korean.
2. `search_web`: Use this if Wikipedia does not provide sufficient information. It represents a general web search.
3. `refine_timeline_card`: Use this to process a list of timeline events into a summarized document for further reasoning.

Follow these exact steps:

Step-by-step:
1. Read and understand the user comment or question.
2. If you need information not in the timeline, use one of the tools.
    - Use this format for tool calls:
        Action: <tool_name>
        Action Input: <input>
3. Wait for the tool output ("Observation") and cite the tool source:
        [Source: <tool_name> | <input summary> | <source url or internal>]
4. Repeat tool usage if necessary until you gather sufficient context.
5. Provide a final answer based on reasoning and tool results.

Constraints:
- Only use tools when necessary.
- Do not hallucinate or make claims without citation.
- Final answers must be concise, accurate, and cite all factual sources.

Example:
User: 김범수는 누구야?

Action: search_wiki
Action Input: 김범수

Observation: 김범수는 카카오 창업자이며 서울대 출신의 기업가입니다...
[Source: search_wiki | 김범수 | https://ko.wikipedia.org/wiki/김범수]

Final Answer: 김범수는 서울대 출신의 기업가로, 카카오를 창업한 인물입니다.
""")

# 상태 정의
class TimelineState(BaseModel):
    timeline: List[str]

class TimelineResult(BaseModel):
    context_docs: List[str]

class AgenticCommentGraph:
    def __init__(self, server: str, model: str, max_retries: int = 3):
        self.max_retries = max_retries
        self.server = server
        self.model = model
        self.tools = [self.search_wiki, self.search_web, self.refine_timeline_card]

    @tool
    def search_wiki(self, query: str, k: int = 3) -> str:
        """인물이 글에 포함이 되어있다면 위키피디아에서 검색하세요."""
        wiki_search = WikipediaAPIWrapper(num_results=k, lang="ko")
        try:
            result = wiki_search.run(query)
        except Exception as e:
            raise RuntimeError(f"Error in search_wiki: {e}")
        return result

    @tool
    def search_web(self, query: str, k: int = 3) -> str:
        """TODO: 웹 검색 도구를 구현하세요."""
        
        return "웹 검색 결과"

    @tool
    def refine_timeline_card(self, state: TimelineState) -> TimelineResult:
        """타임라인을 받아서 정제해서 context 문서로 반환"""
        document = "\n\n".join(state.timeline)
        logger.info(f"refine_timeline_card: {document}\n\n")
        return TimelineResult(context_docs=[document])

    def _make_llm(self):
        llm = ChatOpenAI(
            base_url=f"{self.server}/v1",
            api_key="not-needed",
            model=self.model,
            temperature=0.1,
            tool_choice="auto",
        )
        return llm

    def build(self):
        llm = self._make_llm()
        graph = create_react_agent(
            model=llm,
            tools=self.tools,
            prompt=system_prompt,
        )
        return graph

if __name__ == "__main__":
    SERVER = "https://8acc-34-125-119-95.ngrok-free.app"
    MODEL = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"
    tmp = AgenticCommentGraph(server=SERVER, model=MODEL).build()
    print(tmp)

    inputs = {
        "input_text": "카카오 김범수 의장의 창업 배경과 사회 기여에 대해 알려줘.",
        "timeline": [
            "김범수는 서울대학교 산업공학과를 졸업했다.",
            "카카오를 창업하여 한국 모바일 메신저 시장을 선도했다.",
            "카카오 사회공헌재단을 설립해 다양한 사회적 활동을 지원하고 있다.",
        ],
        "score": 0,
        "worker_id": 42,
        "context_docs": [],
    }

    for step in tmp.stream(inputs):
        step_name = step.get('__step__') or next(iter(step.keys()), 'unknown')
        print(f"\n🧩 step: {step_name}")
        pprint(step.get('agent', step))
