import pytest
from fastapi.testclient import TestClient
from main import app  # Assuming your FastAPI app instance is in main.py
from unittest.mock import patch, AsyncMock, MagicMock
import pandas as pd

client = TestClient(app)


@pytest.fixture
def mock_daum_vclip_searcher():
    with patch(
        "scrapers.daum_vclip_searcher.DaumVclipSearcher", autospec=True
    ) as MockDaumSearcher:
        instance = MockDaumSearcher.return_value
        instance.search.return_value = pd.DataFrame(
            {"url": ["http://daum.net/video1"], "title": ["Daum Video"]}
        )
        yield instance


@pytest.fixture
def mock_youtube_searcher():
    with patch(
        "scrapers.youtube_searcher.YouTubeCommentAsyncFetcher", autospec=True
    ) as MockYoutubeSearcher:
        instance = MockYoutubeSearcher.return_value
        instance.search.return_value = [
            {
                "url": "http://youtube.com/video1",
                "comment": "Good video",
                "captions": "",
            },
            {
                "url": "http://youtube.com/video1",
                "comment": "Bad video",
                "captions": "",
            },
        ]
        yield instance


@pytest.fixture
def mock_sentiment_aggregator():
    with patch(
        "services.classify.SentimentAggregator", autospec=True
    ) as MockAggregator:
        instance = MockAggregator.return_value
        instance.aggregate_multiple_queries.return_value = AsyncMock(
            return_value={"긍정": 70, "부정": 20, "중립": 10}
        )
        yield instance


def test_classify_comments_success(
    mock_daum_vclip_searcher, mock_youtube_searcher, mock_sentiment_aggregator
):
    response = client.post("/api/comment", json={"num": 10, "query": ["테스트 댓글"]})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "data" in data
    assert data["data"]["positive"] == 70
    assert data["data"]["negative"] == 20
    assert data["data"]["neutral"] == 10


def test_classify_comments_empty_query():
    response = client.post("/api/comment", json={"num": 10, "query": []})
    assert (
        response.status_code == 404
    )  # As per current code, empty query means no meaningful content
    assert response.json()["detail"] == "DaumVclip 검색 결과가 없습니다!"


def test_classify_comments_no_daum_results(
    mock_daum_vclip_searcher, mock_youtube_searcher, mock_sentiment_aggregator
):
    mock_daum_vclip_searcher.search.return_value = pd.DataFrame()
    response = client.post("/api/comment", json={"num": 10, "query": ["테스트 댓글"]})
    assert response.status_code == 404
    assert response.json()["detail"] == "DaumVclip 검색 결과가 없습니다!"


def test_classify_comments_no_youtube_results(
    mock_daum_vclip_searcher, mock_youtube_searcher, mock_sentiment_aggregator
):
    mock_youtube_searcher.search.return_value = []
    response = client.post("/api/comment", json={"num": 10, "query": ["테스트 댓글"]})
    assert response.status_code == 500
    assert response.json()["detail"] == "Youtube 데이터를 불러오는 데 실패했습니다"


def test_classify_comments_no_comments_extracted(
    mock_daum_vclip_searcher, mock_youtube_searcher, mock_sentiment_aggregator
):
    mock_youtube_searcher.search.return_value = [
        {"url": "http://youtube.com/video1", "comment": "", "captions": ""}
    ]  # Simulate empty comments
    response = client.post("/api/comment", json={"num": 10, "query": ["테스트 댓글"]})
    assert response.status_code == 404
    assert response.json()["detail"] == "Youtube 댓글이 없습니다"


def test_classify_comments_sentiment_aggregator_failure(
    mock_daum_vclip_searcher, mock_youtube_searcher, mock_sentiment_aggregator
):
    mock_sentiment_aggregator.aggregate_multiple_queries.return_value = AsyncMock(
        return_value=None
    )
    response = client.post("/api/comment", json={"num": 10, "query": ["테스트 댓글"]})
    assert response.status_code == 500
    assert response.json()["detail"] == "댓글 분류 실패!"
