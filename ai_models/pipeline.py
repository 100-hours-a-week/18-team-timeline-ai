import asyncio
from typing import List, Dict, Tuple
from ai_models.host import Host, SystemRole
from scrapers.article_extractor import ArticleExtractor
from ai_models.manager import BatchManager, wrapper
from ai_models.store import ResultStore
import logging
from utils.logger import Logger
from openai import AsyncOpenAI

logger = Logger.get_logger("ai_models.pipeline", log_level=logging.ERROR)


async def Pipeline(
    urls: List[Dict[str, str]],
    server: str,
    model: str,
    repeat: int = 1,
    roles: List[SystemRole] = [SystemRole.summary],
    batch_size: int = 256,
    max_wait_time: float = 2.0,
) -> Dict[str, Dict[str, List[str]]]:
    """
    URL 목록에 대한 AI 처리 파이프라인

    Args:
        urls: 처리할 URL 목록
        server: AI 서버 URL
        model: 사용할 모델 이름
        repeat: 각 URL당 반복 처리 횟수
        roles: 처리할 시스템 역할 목록
        batch_size: 배치 크기
        max_wait_time: 최대 대기 시간(초)

    Returns:
        Dict: 처리 결과
    """
    if not urls:
        logger.warning("[PIPELINE]: 입력 URL이 없습니다.")
        return {}

    logger.info(f"[PIPELINE]: {len(urls)}개 URL, {len(roles)}개 역할, 반복 {repeat}")

    results_dict = ResultStore()
    manager = None
    runner = None

    try:
        async with Host(server, model) as host:
            extractor = ArticleExtractor()

            manager = BatchManager(
                host, batch_size=batch_size, max_wait_time=max_wait_time
            )
            runner = asyncio.create_task(manager.run())
            # URL 등록
            for item in urls:
                url = item.get("url")
                if url:
                    results_dict.register({"url": url})
                else:
                    logger.warning(f"[PIPELINE] URL이 없는 항목 무시: {item}")

            # 텍스트 추출
            url_sentences = {}
            async for result in extractor.search(urls):
                if not result or not result.get("input_text"):
                    print(f"❌ 문장 분리 실패: URL={result['url']}")
                    continue
                url_sentences[result["url"]] = result["input_text"]
            # 태스크 생성
            tasks = []
            for url, input_text in url_sentences.items():
                for _ in range(repeat):
                    for role in roles:
                        logger.info(
                            f"[PIPELINE] URL: {url}, ROLE: {role}, TEXT: {input_text[:10]}..."
                        )
                        tasks.append(
                            asyncio.create_task(wrapper(url, role, input_text, manager))
                        )

            results = await asyncio.gather(*tasks, return_exceptions=True)
            manager.running = False
            await asyncio.sleep(1.0)

            for task in results:
                if isinstance(task, Exception):
                    logger.error(f"[PIPELINE] 태스크 예외 발생: {repr(task)}")
                    continue
                url, role, response = task
                logger.info(f"[PIPELINE] URL : {url}")
                if isinstance(response, dict) and "error" in response:
                    logger.error(
                        f"응답 오류: URL={url}, ROLE={role}, ERROR={response['error']}"
                    )
                    continue

                try:
                    content = (
                        response["choices"][0]["message"]["content"]
                        if response
                        else "실패"
                    )
                    results_dict.add_result(url=url, role=role, content=content)
                except (KeyError, TypeError, IndexError) as e:
                    logger.error(f"응답 파싱 오류: {e}, 응답: {response}")

            logger.info("[PIPELINE] 모든 태스크 처리 완료")

    except Exception as e:
        import traceback

        logger.error(f"[PIPELINE] 실행 중 예외 발생: {e}")
        logger.error(traceback.format_exc())
        raise
    finally:
        # 리소스 정리
        if manager:
            manager.running = False
            await asyncio.sleep(0.5)

        if runner:
            try:
                await asyncio.wait_for(runner, timeout=1.0)
            except asyncio.TimeoutError as e:
                logger.warning((f"[PIPELINE] {e}"))
            except asyncio.CancelledError as e:
                logger.warning(f"[PIPELINE] {e}")

    return results_dict.as_dict()


async def TotalPipeline(
    text: list[str],
    api_key: str,
    model: str = "gpt-4o",
    repeat: int = 5,
    roles: list[SystemRole] = [SystemRole.summary, SystemRole.title],
    batch_size: int = 5,
    max_wait_time: float = 0.5,
) -> Dict[str, Dict[str, List[str]]]:
    """
    텍스트 목록에 대한 통합 AI 처리 파이프라인

    Args:
        text: 처리할 텍스트 목록
        server: AI 서버 URL
        model: 사용할 모델 이름
        repeat: 각 텍스트당 반복 처리 횟수
        roles: 처리할 시스템 역할 목록
        batch_size: 배치 크기
        max_wait_time: 최대 대기 시간(초)

    Returns:
        Dict: 처리 결과
    """
    if not text:
        logger.warning("TotalPipeline: 입력 텍스트가 없습니다.")
        return {}

    logger.info(
        f"TotalPipeline 시작: {len(text)}개 텍스트, {len(roles)}개 역할, 반복 {repeat}회"
    )

    results_dict = ResultStore()
    try:

        client = AsyncOpenAI(api_key=api_key)
        sentence = "\n".join(text)
        logger.debug(f"결합된 텍스트의 길이:{len(text)}")
        tasks = []
        tasks = []
        for _ in range(repeat):
            for role in roles:
                tasks.append(
                    asyncio.create_task(
                        process_with_openai(client, sentence, role, model)
                    )
                )

        # 태스크 결과 처리
        for task in asyncio.as_completed(tasks):
            try:
                role, response = await task
                content = response if response else "실패"
                results_dict.add_result(url="total_summary", role=role, content=content)
            except Exception as e:
                logger.error(f"[TotalPipeline] 태스크 처리 중 오류: {e}")
                continue

    except Exception as e:
        logger.error(f"[TotalPipeline] 실행 중 예외 발생: {e}")
        import traceback

        logger.error(traceback.format_exc())
        raise

    return results_dict.as_dict()


async def process_with_openai(
    client: AsyncOpenAI, text: str, role: SystemRole, model: str
) -> Tuple[SystemRole, str]:
    """
    OpenAI API를 사용하여 텍스트 처리
    """
    from ai_models.host import SYSTEM_PROMPT

    max_retries = 3
    retry_delay = 1

    system_prompt = SYSTEM_PROMPT[role]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]

    # 프롬프트 로깅
    logger.debug(f"[OpenAI] Role: {role.value}")
    logger.debug(f"[OpenAI] System Prompt: {role.value}")
    logger.debug(f"[OpenAI] User Prompt: {text}")

    for attempt in range(max_retries):
        try:
            response = await client.chat.completions.create(
                model=model, messages=messages, temperature=0.5, max_tokens=64
            )

            result = response.choices[0].message.content
            logger.debug(f"[OpenAI] Response: {result}")
            return role, result

        except Exception as e:
            error_message = str(e)

            if "insufficient_quota" in error_message or "429" in error_message:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2**attempt)
                    logger.warning(
                        f"[OpenAI] 할당량 초과. {wait_time}초 후 재시도... (시도 {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(
                        f"[OpenAI] 최대 재시도 횟수 초과. 할당량 초과 오류가 지속됩니다."
                    )
                    return (
                        role,
                        "OpenAI API 할당량이 초과되었습니다. 잠시 후 다시 시도해주세요.",
                    )

            logger.error(f"[OpenAI] 처리 중 오류: {e}")
            return role, f"오류 발생: {str(e)}"


if __name__ == "__main__":
    import time
    import os
    from dotenv import load_dotenv

    load_dotenv(override=True)
    SERVER = os.getenv("SERVER")
    MODEL = os.getenv("MODEL")
    # SERVER = "http://1d5d-34-124-161-59.ngrok-free.app"
    # MODEL = "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
    URLS = [
        {
            "url": "https://www.hani.co.kr/arti/society/society_general/1192251.html",
            "title": '말 바꾼 윤석열 "계엄 길어야 하루"…헌재선 "며칠 예상"',
        },
        {
            "url": "https://www.hani.co.kr/arti/society/society_general/1192255.html",
            "title": "윤석열 40분간 \"계엄은 평화적 메시지\"…판사도 발언 '시간조절' 당부",
        },
        {
            "url": "https://www.hankyung.com/article/2025041493977",
            "title": "'[속보] 韓대행 '국무위원들과 제게 부여된 마지막 소명 다할 것'",
        },
        {
            "url": "https://www.ytn.co.kr/_ln/0134_202505210904491383",
            "title": "다 꺼낸 구글의 '승부수'...삼성 이어 젠틀몬스터와 손 잡는다 [지금이뉴스]",
        },
        {
            "url": "https://m.news.zum.com/articles/98407004/%EC%9D%B4%EB%B2%88%EC%97%94-%EC%93%B8%EB%A7%8C%ED%95%A0%EA%B9%8C-%EA%B5%AC%EA%B8%80-%EC%82%BC%EC%84%B1-%EC%8A%A4%EB%A7%88%ED%8A%B8-%EC%95%88%EA%B2%BD-%ED%98%91%EC%97%85",
            "title": "이번엔 쓸만할까?…구글-삼성 스마트 안경 협업",
        },
        {
            "url": "https://www.news1.kr/world/usa-canada/5789241",
            "title": "챗봇에 흔들린 구글, AI 모드로 검색 기능 강화 나섰다",
        },
        {
            "url": "https://www.newstong.co.kr/view3.aspx?seq=13665995&allSeq=27&txtSearch=&cate=0&cnt=-5&subCate=2&order=default&newsNo=0",
            "title": "구글, 음성·영상으로도 검색 이용…예약 등 에이전트 기능도",
        },
    ]
    start = time.perf_counter()

    result1 = asyncio.run(Pipeline(URLS, SERVER, MODEL, repeat=1))

    texts = [s for r in result1.values() for s in r.get("summary", [])]
    print(texts)
    API_KEY = os.getenv("OPENAI_API_KEY")
    result2 = asyncio.run(TotalPipeline(texts, API_KEY, repeat=1))
    print(result2)
    end = time.perf_counter()
    print(f"총 실행 시간: {end - start:.2f}s")
