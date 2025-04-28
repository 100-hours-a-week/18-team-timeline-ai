from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from newspaper import Article
from datetime import datetime
from textwrap import dedent
import asyncio
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

load_dotenv()
model = ChatOllama(model="phi4:latest", temperature=0)
"""
model_name = "gogamza/kobart-summarization"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

summarization_pipe = pipeline(
    "summarization", model=model, tokenizer=tokenizer, max_length=32, min_length=20, do_sample=False
)

model = HuggingFacePipeline(pipeline=summarization_pipe)
"""
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
today = datetime.today().strftime("%Y-%m-%d")
system = f"""
당신은 뉴스 요약을 전문으로 하는 AI 어시스턴트입니다.
- 주어진 뉴스 기사 본문을 3줄 이내의 짧은 글로 요약하세요.
- 각 문장은 완결된 형태로 작성하세요.
- 예측, 사견, 해석은 금지합니다.
- 텍스트가 없다면 "NOT TEXT"를 출력하세요.
오늘 날짜는 {today}입니다.
"""
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        *examples,
        ("human", "{text}"),
    ]
)
chain = prompt | model | StrOutputParser()


async def summarize(URL: str) -> str:
    """기사의 URL을 받아 본문을 요약해주는 함수.
    Args:
        URL (str): 기사의 URL

    Returns:
        str: 기사의 본문을 요약한 글로 3줄 이내의 짧은 글 형식을 지킨다.
    """
    article = Article(url=URL, language="ko")
    article.download()
    article.parse()
    text = article.text.strip()
    if not text:
        return "NOT TEXT"
    print(text)

    return await chain.ainvoke(text)


async def main():
    import time

    URLS = [
        "https://www.hani.co.kr/arti/society/society_general/1192251.html",
        "https://www.hani.co.kr/arti/society/society_general/1192255.html",
        "https://www.hankyung.com/article/2025041493977",
        "https://www.khan.co.kr/article/202504141136001",
        "https://www.mk.co.kr/news/politics/11290687",
        "https://www.chosun.com/politics/politics_general/2025/04/14/THWVKUHQG5CKFJF6CLZLP5PKM4",
    ]
    """
    start = time.time()
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(summarize, URLS))

    for idx, (url, summary) in enumerate(zip(URLS, results)):
        if summary == "NOT TEXT":
            continue
        print(f"{url}\n\\n{summary}\n\n")
    print(f"{time.time() - start:2f}")
    """
    from langchain_community.utilities import GoogleSerperAPIWrapper
    import pprint

    search = GoogleSerperAPIWrapper(type="news", tbs="qdr:m6", k=5)
    results = search.results("윤석열")
    pprint.pp(results)

    news = results["news"]
    links = [result["link"] for result in news]
    titles = [result["title"] for result in news]
    dates = [result["date"] for result in news]

    start = time.time()

    results = await asyncio.gather(*[summarize(url) for url in links])

    for idx, (url, summary, title, date) in enumerate(
        zip(links, results, titles, dates)
    ):
        if summary == "NOT TEXT":
            continue
        print(f"{date} : {title}\n\n")
        print(f"{url}\n\n{summary}\n\n")
    print(f"{time.time() - start:2f}")


if __name__ == "__main__":
    asyncio.run(main())
