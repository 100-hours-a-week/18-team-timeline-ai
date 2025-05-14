from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain.schema import Document
from datasets import load_dataset
import logging
import asyncio
import aiohttp
from typing import List
import time
import dotenv
import os
from scrapers.daum_vclip_searcher import DaumVclipSearcher
from scrapers.youtube_searcher import YouTubeCommentAsyncFetcher
from ai_models.graph.classify import ClassifyGraph
from ai_models.runner import Runner

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
LABELS = [
    "불평/불만",
    "환영/호의",
    "감동/감탄",
    "지긋지긋",
    "고마움",
    "슬픔",
    "화남/분노",
    "존경",
    "기대감",
    "우쭐댐/무시함",
    "안타까움/실망",
    "비장함",
    "의심/불신",
    "뿌듯함",
    "편안/쾌적",
    "신기함/관심",
    "아껴주는",
    "부끄러움",
    "공포/무서움",
    "절망",
    "한심함",
    "역겨움/징그러움",
    "짜증",
    "어이없음",
    "없음",
    "패배/자기혐오",
    "귀찮음",
    "힘듦/지침",
    "즐거움/신남",
    "깨달음",
    "죄책감",
    "증오/혐오",
    "흐뭇함(귀여움/예쁨)",
    "당황/난처",
    "경악",
    "부담/안_내킴",
    "서러움",
    "재미없음",
    "불쌍함/연민",
    "놀람",
    "행복",
    "불안/걱정",
    "기쁨",
    "안심/신뢰",
]

# 감정 레이블 매핑 (KOTE 데이터셋의 실제 레이블 기반)
SENTIMENT_MAP = {
    # 긍정
    "환영/호의": "긍정",
    "감동/감탄": "긍정",
    "고마움": "긍정",
    "존경": "긍정",
    "기대감": "긍정",
    "뿌듯함": "긍정",
    "편안/쾌적": "긍정",
    "신기함/관심": "긍정",
    "즐거움/신남": "긍정",
    "깨달음": "긍정",
    "흐뭇함(귀여움/예쁨)": "긍정",
    "행복": "긍정",
    "기쁨": "긍정",
    "안심/신뢰": "긍정",
    "아껴주는": "긍정",
    # 부정
    "불평/불만": "부정",
    "지긋지긋": "부정",
    "슬픔": "부정",
    "화남/분노": "부정",
    "우쭐댐/무시함": "부정",
    "안타까움/실망": "부정",
    "의심/불신": "부정",
    "부끄러움": "부정",
    "공포/무서움": "부정",
    "절망": "부정",
    "한심함": "부정",
    "역겨움/징그러움": "부정",
    "짜증": "부정",
    "어이없음": "부정",
    "패배/자기혐오": "부정",
    "귀찮음": "부정",
    "힘듦/지침": "부정",
    "죄책감": "부정",
    "증오/혐오": "부정",
    "당황/난처": "부정",
    "경악": "부정",
    "부담/안_내킴": "부정",
    "서러움": "부정",
    "재미없음": "부정",
    "불안/걱정": "부정",
    # 중립
    "없음": "중립",
    "불쌍함/연민": "중립",
    "놀람": "중립",
    "비장함": "중립",
}


class OllamaEmbeddings:
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url
        self.model = "bge-m3"  # Ollama에서 사용할 모델
        self.batch_size = 128  # 배치 크기 설정

async def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """문서 리스트를 배치 단위로 비동기 임베딩합니다."""
        async with aiohttp.ClientSession() as session:
            all_embeddings = []
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i : i + self.batch_size]
                embeddings = await self.embed_batch(session, batch)
                all_embeddings.extend(embeddings)
                logger.info(f"배치 처리 완료: {i + len(batch)}/{len(texts)}")
            return all_embeddings

    async def embed_query(self, text: str) -> List[float]:
        """단일 쿼리를 임베딩합니다."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/embeddings",
                json={"model": self.model, "prompt": text},
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data["embedding"]
                else:
                    error_text = await response.text()
                    raise Exception(f"임베딩 생성 실패: {error_text}")


def load_kote_dataset():
    """KOTE 데이터셋을 로드합니다."""
    try:
        ds = load_dataset("searle-j/kote", cache_dir=".dataset", trust_remote_code=True)
        logger.info(
            f"데이터셋 로드 완료: {len(ds['train'])} 학습 샘플, "
            f"{len(ds['test'])} 테스트 샘플"
        )
        return ds
    except Exception as e:
        logger.error(f"데이터셋 로드 실패: {str(e)}")
        raise


def create_documents(dataset):
    """데이터셋을 Document 객체 리스트로 변환합니다."""
    documents = []
    unique_labels = set()  # 고유한 레이블 수집

    for split in ["test"]:
        for item in dataset[split]:
            # labels가 리스트인 경우 첫 번째 감정 사용
            label_indices = (
                item["labels"][0]
                if isinstance(item["labels"], list)
                else item["labels"]
            )
            # 숫자 인덱스를 실제 레이블로 변환
            if isinstance(label_indices, list):
                labels = [LABELS[idx] for idx in label_indices]
                label = labels[0]  # 첫 번째 레이블 사용
            else:
                label = LABELS[label_indices]

            unique_labels.add(label)  # 고유 레이블 수집
            doc = Document(
                page_content=item["text"],
                metadata={"label": label, "split": split, "id": item["ID"]},
            )
            documents.append(doc)

    logger.info(f"총 {len(documents)}개의 문서 생성 완료")
    logger.info(f"고유한 감정 레이블: {sorted(unique_labels)}")
    return documents


async def process_and_store_batch(
    client: QdrantClient,
    collection_name: str,
    documents: List[Document],
    embeddings: List[List[float]],
    start_idx: int,
):
    """배치 단위로 문서를 처리하고 Qdrant에 저장합니다.

    Args:
        client: Qdrant 클라이언트 인스턴스
        collection_name: 저장할 컬렉션 이름
        documents: 처리할 Document 객체 리스트
        embeddings: 각 문서의 임베딩 벡터 리스트
        start_idx: 현재 배치의 시작 인덱스
    """
    # Qdrant에 저장할 포인트 리스트 생성
    points = []
    for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
        # 각 문서를 Qdrant 포인트 형식으로 변환
        points.append(
            {
                "id": start_idx + i,  # 고유 ID (전체 문서 중 순서)
                "vector": embedding,  # 임베딩 벡터
                "payload": {  # 메타데이터
                    "text": doc.page_content,  # 원본 텍스트
                    "label": doc.metadata["label"],  # 감정 레이블
                    "split": doc.metadata["split"],  # 데이터셋 분할(train/val/test)
                    "id": doc.metadata["id"],  # 원본 데이터셋 ID
                },
            }
        )

    # 비동기로 Qdrant에 배치 저장
    # asyncio.to_thread를 사용하여 동기 함수를 비동기로 실행
    await asyncio.to_thread(
        client.upsert, collection_name=collection_name, points=points
    )


async def update_comment_sentiment(
    client: QdrantClient, collection_name: str, comment_id: str, new_sentiment: str
):
    """댓글의 감정 분류를 수정합니다.

    Args:
        client: Qdrant 클라이언트 인스턴스
        collection_name: 컬렉션 이름
        comment_id: 수정할 댓글의 ID
        new_sentiment: 새로운 감정 분류
    """
    try:
        # ID로 댓글 검색
        results = await asyncio.to_thread(
            client.scroll,
            collection_name=collection_name,
            scroll_filter={"must": [{"key": "id", "match": {"value": comment_id}}]},
        )

        if not results[0]:  # 검색 결과가 없는 경우
            logger.error(f"ID가 {comment_id}인 댓글을 찾을 수 없습니다.")
            return False

        point = results[0][0]
        point_id = point.id

        # 감정 분류 업데이트
        await asyncio.to_thread(
            client.set_payload,
            collection_name=collection_name,
            payload={"label": new_sentiment},
            points=[point_id],
        )

        logger.info(
            f"댓글 {comment_id}의 감정 분류가 '{new_sentiment}'로 수정되었습니다."
        )
        return True

    except Exception as e:
        logger.error(f"감정 분류 수정 중 오류 발생: {str(e)}")
        return False


def get_sentiment_from_similar_comments(results):
    """유사한 댓글들의 감정을 기반으로 감정을 분류합니다."""
    sentiment_counts = {"긍정": 0, "부정": 0, "중립": 0}
    total_comments = len(results)

    for result in results:
        original_sentiments = result.payload["label"]
        # 레이블이 리스트가 아닌 경우 리스트로 변환
        if not isinstance(original_sentiments, list):
            original_sentiments = [original_sentiments]

        # 각 댓글의 감정을 카운트
        comment_sentiments = {"긍정": 0, "부정": 0, "중립": 0}
        for sentiment in original_sentiments:
            # 숫자 인덱스를 실제 레이블로 변환
            if isinstance(sentiment, (int, float)):
                try:
                    sentiment = LABELS[int(sentiment)]
                except (IndexError, ValueError):
                    logger.warning(f"잘못된 감정 인덱스: {sentiment}")
                    continue

            # 레이블이 매핑에 없는 경우 로깅
            if sentiment not in SENTIMENT_MAP:
                logger.warning(f"알 수 없는 감정 레이블: {sentiment}")
                continue

            mapped_sentiment = SENTIMENT_MAP[sentiment]
            comment_sentiments[mapped_sentiment] += 1

        # 해당 댓글의 가장 많은 감정을 선택
        max_sentiment = max(comment_sentiments.items(), key=lambda x: x[1])[0]
        sentiment_counts[max_sentiment] += 1

    # 감정 비율 계산
    sentiment_percentages = {
        sentiment: (count / total_comments) * 100 if total_comments > 0 else 0
        for sentiment, count in sentiment_counts.items()
    }

    # 가장 많은 감정과 비율 반환
    max_sentiment = max(sentiment_counts.items(), key=lambda x: x[1])[0]
    return max_sentiment, sentiment_percentages, sentiment_counts


async def main():
    # 1. 임베딩 모델 로드
    embedding_model = OllamaEmbeddings()

    # 2. Qdrant 클라이언트 설정
    client = QdrantClient(host="localhost", port=6333)

    # 3. 데이터셋 로드
    dataset = load_kote_dataset()
    documents = create_documents(dataset)

    # 4. Qdrant 컬렉션 생성
    collection_name = "kote_comments"
    vector_size = 1024

    try:
        # 컬렉션 존재 여부 확인
        collections = client.get_collections()
        collection_exists = any(
            col.name == collection_name for col in collections.collections
        )

        if collection_exists:
            logger.info(f"기존 컬렉션 '{collection_name}' 사용")

            # YouTube 댓글 검색
            dotenv.load_dotenv(override=True)
            YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
            REST_API_KEY = os.getenv("REST_API_KEY")
            video_searcher = DaumVclipSearcher(api_key=REST_API_KEY)
            youtube_searcher = YouTubeCommentAsyncFetcher(
                api_key=YOUTUBE_API_KEY, max_comments=20
            )
            start_time = time.time()
            df = video_searcher.search("윤석열 유튜브")
            test_queries = await youtube_searcher.search(df=df)

            # 전체 댓글의 감정 통계
            total_sentiment_counts = {"긍정": 0, "부정": 0, "중립": 0}
            total_comments = 0

            for query in test_queries:
                print(f"\n🔍 검색어: {query['comment']}")
                embedding = await embedding_model.embed_query(query["comment"])
                results = await asyncio.to_thread(
                    client.search,
                    collection_name=collection_name,
                    query_vector=embedding,
                    limit=5,  # 상위 5개 유사 댓글
                )

                print(
                    f"원본 댓글: {query['comment']} | "
                    f"채널: {query.get('channel', 'N/A')} | "
                    f"좋아요: {query.get('like_count', 0)}"
                )

                # 유사 댓글들의 감정을 기반으로 감정 분류
                sentiment, percentages, counts = get_sentiment_from_similar_comments(
                    results
                )
                print(f"분류된 감정: {sentiment}")
                print("감정 분포:")
                for s, p in percentages.items():
                    print(f"  {s}: {p:.1f}%")
                    total_sentiment_counts[s] += counts[s]
                total_comments += len(results)

                print("\n유사 댓글:")
                for result in results:
                    print(
                        f"유사도: {result.score:.4f} | "
                        f"감정: {result.payload['label']} | "
                        f"내용: {result.payload['text']}"
                    )

            # 전체 통계 출력
            print("\n📊 전체 댓글 감정 통계:")
            for sentiment, count in total_sentiment_counts.items():
                percentage = (count / total_comments) * 100 if total_comments > 0 else 0
                print(f"{sentiment}: {count}개 ({percentage:.1f}%)")
            end_time = time.time()
            logger.info(f"테스트 검색 완료! 소요 시간: {end_time - start_time:.1f}초")
            return
        else:
            # 컬렉션이 없을 경우에만 생성
            client.create_collection(
                collection_name=collection_name,
                vectors_config={"size": vector_size, "distance": "Cosine"},
            )
            logger.info(f"새 컬렉션 '{collection_name}' 생성 완료")
    except Exception as e:
        logger.error(f"컬렉션 확인/생성 중 오류 발생: {str(e)}")
        raise

    # 5. 배치 단위로 문서 처리 및 저장
    batch_size = 128  # 한 번에 처리할 문서 수
    start_time = time.time()

    # 전체 문서를 배치 크기만큼 나누어 처리
    for i in range(0, len(documents), batch_size):
        # 현재 배치의 문서 추출
        batch_docs = documents[i : i + batch_size]
        # 배치 내 문서들의 텍스트만 추출
        texts = [doc.page_content for doc in batch_docs]

        # 현재 배치의 임베딩 생성 (비동기)
        embeddings = await embedding_model.embed_documents(texts)

        # 생성된 임베딩을 Qdrant에 즉시 저장 (비동기)
        await process_and_store_batch(
            client, collection_name, batch_docs, embeddings, i
        )

        # 진행 상황 로깅
        logger.info(
            f"배치 처리 완료: {i + len(batch_docs)}/{len(documents)} "
            f"({(i + len(batch_docs))/len(documents)*100:.1f}%)"
        )

    # 전체 처리 시간 계산 및 로깅
    end_time = time.time()
    logger.info(f"전체 처리 완료! 소요 시간: {end_time - start_time:.1f}초")


if __name__ == "__main__":
    asyncio.run(main())
