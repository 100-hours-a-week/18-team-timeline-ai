from typing import List
from langchain_community.tools import TavilySearchResults
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import create_react_agent, ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_community.utilities import WikipediaAPIWrapper
from textwrap import dedent
from pydantic import BaseModel
from pprint import pprint
import logging

# 로그 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 시스템 프롬프트 정의
system_prompt = dedent(
    """
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
"""
)


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
        """현재 모르는 정보 또는 최신 정보를 인터넷에서 검색합니다."""
        tavily_search = TavilySearchResults(max_results=k)
        try:
            result = tavily_search.run(query)
        except Exception as e:
            raise RuntimeError(f"Error in search_web: {e}")
        formatted_docs = "\n\n---\n\n".join(
            [
                f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
                for doc in result
            ]
        )
        if len(result) > 0:
            return formatted_docs
        return "웹 검색 결과를 찾을 수 없습니다."

    @tool
    def refine_timeline_card(self, state: TimelineState) -> TimelineResult:
        """사건에 대한 전체적인 문맥이 필요할 때 활용하세요."""
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
        llm_with_tools = llm.bind_tools(self.tools)
        pprint(llm_with_tools)
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

    inputs = {"messages": [HumanMessage(content="고윤정이 누구야?")]}
    results = tmp.invoke(inputs)
    for step in results["messages"]:
        print(step.content)
        print()
