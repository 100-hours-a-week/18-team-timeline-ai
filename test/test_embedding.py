"""
임베딩 모듈 테스트

이 모듈은 임베딩 관련 기능을 테스트합니다.
주요 테스트:
- 단일 텍스트 임베딩
- 배치 텍스트 임베딩
- 에러 처리
"""

import pytest
import asyncio
from typing import List
from classify.embedding import OllamaEmbeddingService


@pytest.fixture
def embedding_service():
    """테스트용 임베딩 서비스 인스턴스를 생성합니다."""
    return OllamaEmbeddingService(
        base_url="http://localhost:11434", model="bge-m3", batch_size=2
    )


@pytest.mark.asyncio
async def test_embed_query(embedding_service):
    """단일 쿼리 임베딩 테스트"""
    # 테스트 데이터
    text = "안녕하세요, 반갑습니다."

    # 임베딩 수행
    vector = await embedding_service.embed_query(text)

    # 검증
    assert isinstance(vector, list)
    assert len(vector) > 0
    assert all(isinstance(x, float) for x in vector)


@pytest.mark.asyncio
async def test_embed_documents(embedding_service):
    """배치 문서 임베딩 테스트"""
    # 테스트 데이터
    texts = [
        "첫 번째 테스트 문장입니다.",
        "두 번째 테스트 문장입니다.",
        "세 번째 테스트 문장입니다.",
    ]

    # 임베딩 수행
    vectors = await embedding_service.embed_documents(texts)

    # 검증
    assert isinstance(vectors, list)
    assert len(vectors) == len(texts)
    assert all(isinstance(v, list) for v in vectors)
    assert all(len(v) > 0 for v in vectors)
    assert all(all(isinstance(x, float) for x in v) for v in vectors)


@pytest.mark.asyncio
async def test_embed_documents_empty(embedding_service):
    """빈 문서 리스트 임베딩 테스트"""
    # 빈 리스트로 테스트
    vectors = await embedding_service.embed_documents([])

    # 검증
    assert isinstance(vectors, list)
    assert len(vectors) == 0


@pytest.mark.asyncio
async def test_embed_query_error(embedding_service):
    """임베딩 에러 처리 테스트"""
    # 잘못된 URL로 서비스 생성
    error_service = OllamaEmbeddingService(
        base_url="http://invalid-url:1234", model="bge-m3"
    )

    # 에러 발생 확인
    with pytest.raises(Exception) as exc_info:
        await error_service.embed_query("테스트 문장")

    error_message = str(exc_info.value)
    assert any(
        [
            "Cannot connect to host" in error_message,
            "Connection refused" in error_message,
            "Connection error" in error_message,
            "Failed to connect" in error_message,
        ]
    )


@pytest.mark.asyncio
async def test_embed_documents_batch_size(embedding_service):
    """배치 크기 테스트"""
    # 배치 크기보다 큰 데이터
    texts = ["테스트 문장 " + str(i) for i in range(5)]
    embedding_service.batch_size = 2

    # 임베딩 수행
    vectors = await embedding_service.embed_documents(texts)

    # 검증
    assert len(vectors) == len(texts)
    assert all(len(v) > 0 for v in vectors)


@pytest.mark.asyncio
async def test_embed_documents_unicode(embedding_service):
    """유니코드 텍스트 임베딩 테스트"""
    # 다양한 유니코드 문자 포함
    texts = [
        "안녕하세요! Hello! こんにちは!",
        "테스트 문장입니다. Test sentence. テスト文です。",
        "한글 English 日本語混合",
    ]

    # 임베딩 수행
    vectors = await embedding_service.embed_documents(texts)

    # 검증
    assert len(vectors) == len(texts)
    assert all(len(v) > 0 for v in vectors)


@pytest.mark.asyncio
async def test_embed_query_special_chars(embedding_service):
    """특수 문자 임베딩 테스트"""
    # 특수 문자 포함
    text = "!@#$%^&*()_+{}|:<>?[]\\;',./"

    # 임베딩 수행
    vector = await embedding_service.embed_query(text)

    # 검증
    assert len(vector) > 0
    assert all(isinstance(x, float) for x in vector)


@pytest.mark.asyncio
async def test_embed_documents_concurrent(embedding_service):
    """동시성 테스트"""
    # 여러 요청 동시 실행
    texts = ["동시성 테스트 " + str(i) for i in range(3)]
    tasks = [
        embedding_service.embed_documents(texts),
        embedding_service.embed_documents(texts),
        embedding_service.embed_documents(texts),
    ]

    # 동시 실행
    results = await asyncio.gather(*tasks)

    # 검증
    assert len(results) == 3
    for vectors in results:
        assert len(vectors) == len(texts)
        assert all(len(v) > 0 for v in vectors)
