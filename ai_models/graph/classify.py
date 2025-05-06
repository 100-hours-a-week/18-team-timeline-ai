from typing import List
from langchain_community.tools import TavilySearchResults, WikipediaQueryRun
from langchain_community.tools.wikidata.tool import WikidataQueryRun, WikidataAPIWrapper
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.utilities import WikipediaAPIWrapper
from textwrap import dedent
from pydantic import BaseModel
from pprint import pprint
from langchain.agents import AgentExecutor, create_react_agent

import logging
from dotenv import load_dotenv

class ClassifyGraph:
    def __init__(self, server: str, model: str, max_retries: int = 3):
        self.max_retries = max_retries
        self.server = server
        self.model = model


if __name__ == "__main__":
    SERVER = "https://b79f-34-125-17-94.ngrok-free.app"
    MODEL = "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"
