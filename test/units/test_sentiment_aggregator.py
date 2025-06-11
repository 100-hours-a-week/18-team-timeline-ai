import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from services.classify import SentimentAggregator


@pytest.fixture
def sentiment_aggregator():
    return SentimentAggregator()


@pytest.fixture
def mock_qdrant_storage():
    with patch("utils.storage.QdrantStorage", autospec=True) as MockStorage:
        instance = MockStorage.return_value
        instance.search_similar.return_value = AsyncMock(
            return_value=[
                MagicMock(payload={"sentiment": "positive"}, score=0.9),
                MagicMock(payload={"sentiment": "negative"}, score=0.8),
                MagicMock(payload={"sentiment": "neutral"}, score=0.7),
            ]
        )
        yield instance


@pytest.fixture
def mock_embedding_service():
    with patch(
        "inference.embedding.OllamaEmbeddingService", autospec=True
    ) as MockEmbeddingService:
        instance = MockEmbeddingService.return_value
        instance.get_embedding.return_value = AsyncMock(
            return_value=[0.1, 0.2, 0.3]  # Dummy embedding
        )
        yield instance


@pytest.mark.asyncio
async def test_aggregate_sentiment_success(
    sentiment_aggregator, mock_qdrant_storage, mock_embedding_service
):
    query = "test query"
    result = await sentiment_aggregator.aggregate_sentiment(
        query=query,
        embedding_constructor=mock_embedding_service,
        collection_name="test_collection",
        labels=["positive", "negative", "neutral"],
    )

    assert "긍정" in result
    assert "부정" in result
    assert "중립" in result
    assert result["긍정"] > 0
    assert result["부정"] > 0
    assert result["중립"] > 0
    mock_embedding_service.return_value.get_embedding.assert_called_once_with(query)
    mock_qdrant_storage.return_value.search_similar.assert_called_once()


@pytest.mark.asyncio
async def test_aggregate_sentiment_no_query(
    sentiment_aggregator, mock_qdrant_storage, mock_embedding_service
):
    with pytest.raises(ValueError, match="query is required"):
        await sentiment_aggregator.aggregate_sentiment(query=None)


@pytest.mark.asyncio
async def test_aggregate_sentiment_empty_search_results(
    sentiment_aggregator, mock_qdrant_storage, mock_embedding_service
):
    mock_qdrant_storage.return_value.search_similar.return_value = AsyncMock(
        return_value=[]
    )

    query = "no results"
    result = await sentiment_aggregator.aggregate_sentiment(
        query=query,
        embedding_constructor=mock_embedding_service,
        collection_name="test_collection",
        labels=["positive", "negative", "neutral"],
    )

    assert result == {"긍정": 0, "부정": 0, "중립": 0}


@pytest.mark.asyncio
async def test_aggregate_multiple_queries(
    sentiment_aggregator, mock_qdrant_storage, mock_embedding_service
):
    queries = ["query1", "query2"]

    # Mock the aggregate_sentiment for multiple queries
    with patch.object(
        sentiment_aggregator, "aggregate_sentiment", new_callable=AsyncMock
    ) as mock_aggregate_sentiment:
        mock_aggregate_sentiment.side_effect = [
            {"긍정": 50, "부정": 30, "중립": 20},
            {"긍정": 10, "부정": 70, "중립": 20},
        ]

        result = await sentiment_aggregator.aggregate_multiple_queries(
            queries=queries,
            embedding_constructor=mock_embedding_service,
            collection_name="test_collection",
            labels=["positive", "negative", "neutral"],
        )

        assert result["긍정"] == 60  # Sum of positive from both queries
        assert result["부정"] == 100  # Sum of negative from both queries
        assert result["중립"] == 40  # Sum of neutral from both queries
        assert mock_aggregate_sentiment.call_count == 2
