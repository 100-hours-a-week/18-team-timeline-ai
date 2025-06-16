from typing import List, Dict, Tuple
from config.prompts import SystemRole

from util.store import ResultStore
from util.logger import Logger
from openai import AsyncOpenAI
import asyncio

logger = Logger.get_logger("ai_models.pipeline")


async def TotalPipeline(
    text: list[str],
    api_key: str,
    model: str = "gpt-4",
    roles: list[SystemRole] = [SystemRole.summary, SystemRole.title],
    batch_size: int = 5,
    max_wait_time: float = 0.5,
) -> Dict[str, Dict[str, List[str]]]:
    """
    텍스트 목록에 대한 통합 AI 처리 파이프라인

    Args:
        text: 처리할 텍스트 목록
        api_key: OpenAI API 키
        model: 사용할 모델 이름
        roles: 처리할 시스템 역할 목록
        batch_size: 배치 크기
        max_wait_time: 최대 대기 시간(초)

    Returns:
        Dict: 처리 결과
    """
    if not text:
        logger.warning("TotalPipeline: 입력 텍스트가 없습니다.")
        return {}

    logger.info(f"TotalPipeline 시작: {len(text)}개 텍스트, {len(roles)}개 역할")

    results_dict = ResultStore()
    try:
        client = AsyncOpenAI(api_key=api_key)
        sentence = "\n".join(text)
        logger.debug(f"결합된 텍스트의 길이:{len(text)}")
        tasks = []

        for role in roles:
            tasks.append(
                asyncio.create_task(process_with_openai(client, sentence, role, model))
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
    """OpenAI API를 사용하여 단일 텍스트 처리"""
    from config.prompts import SYSTEM_PROMPT

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
                        f"[OpenAI] 할당량 초과. {wait_time}초 후 재시도... "
                        f"(시도 {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(
                        "[OpenAI] 최대 재시도 횟수 초과. "
                        "할당량 초과 오류가 지속됩니다."
                    )
                    return (
                        role,
                        "OpenAI API 할당량이 초과되었습니다. "
                        "잠시 후 다시 시도해주세요.",
                    )

            logger.error(f"[OpenAI] 처리 중 오류: {e}")
            return role, f"오류 발생: {str(e)}"
