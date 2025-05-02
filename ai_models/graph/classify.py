from typing import List
from langchain_community.tools import TavilySearchResults
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.utilities import WikipediaAPIWrapper
from textwrap import dedent
from pydantic import BaseModel
from pprint import pprint
from langchain.agents import AgentExecutor, create_react_agent
from langchain.agents.format_scratchpad import format_log_to_str
import logging
from dotenv import load_dotenv
# 로그 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
load_dotenv()

# 시스템 프롬프트 정의
system_prompt = PromptTemplate(
    input_variables=["input", "agent_scratchpad"],
    partial_variables={
        "tools": "\n".join(
            [
                "search_wiki(query: str, k: int = 3) -> str - 인물이 글에 포함이 되어있다면 위키피디아에서 검색하세요.",
                "search_web(query: str, k: int = 3) -> str - 현재 모르는 정보 또는 최신 정보를 인터넷에서 검색합니다."
            ]
        ),
        "tool_names": ", ".join(["search_wiki", "search_web"])
    },
    template=dedent(
        """
You are an AI assistant designed to support Korean timeline-based question answering and comment classification.
You have access to the following tools:
{tools}

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
- You MUST choose only one Action from at a time.
- Use the tools only when necessary.
- Do not hallucinate or make claims without citation.
- Final answers must be concise, accurate, and cite all factual sources.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, must be one of
Action Input: the input to the action
Observation: the result of the action
Final Answer: the final answer to the question

Questions: {input}
Thought: {agent_scratchpad}
Action: {tool_names}


"""
    )
)

@tool
def search_wiki(query: str, k: int = 3) -> str:
    """인물이 글에 포함이 되어있다면 위키피디아에서 검색하세요."""
    wiki_search = WikipediaAPIWrapper(num_results=k, lang="ko")
    try:
        result = wiki_search.run(query)
    except Exception as e:
        raise RuntimeError(f"Error in search_wiki: {e}")
    return result

@tool
def search_web(query: str, k: int = 3) -> str:
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

tools = [search_wiki, search_web]

class AgenticCommentGraph:
    def __init__(self, server: str, model: str, max_retries: int = 3):
        self.max_retries = max_retries
        self.server = server
        self.model = model

    def _make_llm(self):
        llm = ChatOpenAI(
            base_url=f"{self.server}/v1",
            api_key="not-needed",
            model=self.model,
            temperature=0.1,
        )
        return llm.bind_tools(tools)

    def build(self):
        llm = self._make_llm()
        agent = create_react_agent(
            llm=llm,
            tools=tools,
            prompt=system_prompt,   
        )
        return AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

if __name__ == "__main__":
    SERVER = "https://81fe-34-125-119-95.ngrok-free.app"
    MODEL = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"
    tmp = AgenticCommentGraph(server=SERVER, model=MODEL).build()
    print(tmp)

    inputs = {
        "input": "폭삭 속았수다가 뭐야?",
    }
    results = tmp.invoke(inputs)
    print("\n\n=== 결과 ===")
    pprint(results)
