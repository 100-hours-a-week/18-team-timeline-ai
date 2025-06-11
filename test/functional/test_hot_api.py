import pytest
from fastapi.testclient import TestClient
from main import app  # Assuming your FastAPI app instance is in main.py
from unittest.mock import patch, MagicMock
import os

client = TestClient(app)


@pytest.fixture
def mock_get_trending_keywords():
    with patch("scrapers.serpapi.get_trending_keywords") as MockGetTrendingKeywords:
        MockGetTrendingKeywords.return_value = [
            "keyword1",
            "keyword2",
            "keyword3",
            "keyword4",
            "keyword5",
        ]
        yield MockGetTrendingKeywords


def test_get_hot_topics_success(mock_get_trending_keywords):
    # Temporarily set SERP_API_KEY for the test
    original_serp_key = os.getenv("SERP_API_KEY")
    os.environ["SERP_API_KEY"] = "fake_serp_key"

    response = client.post("/api/hot", json={"num": 3})
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "data" in data
    assert data["data"]["keywords"] == ["keyword1", "keyword2", "keyword3"]
    mock_get_trending_keywords.assert_called_once_with("fake_serp_key")

    # Restore original SERP_API_KEY
    if original_serp_key:
        os.environ["SERP_API_KEY"] = original_serp_key
    else:
        del os.environ["SERP_API_KEY"]


def test_get_hot_topics_no_serp_key():
    # Ensure SERP_API_KEY is not set
    original_serp_key = os.getenv("SERP_API_KEY")
    if "SERP_API_KEY" in os.environ:
        del os.environ["SERP_API_KEY"]

    response = client.post("/api/hot", json={"num": 3})
    assert response.status_code == 500
    assert response.json()["detail"] == "SERP_API_KEY를 찾을 수 없습니다."

    # Restore original SERP_API_KEY
    if original_serp_key:
        os.environ["SERP_API_KEY"] = original_serp_key


def test_get_hot_topics_no_keywords_returned(mock_get_trending_keywords):
    mock_get_trending_keywords.return_value = []  # Simulate no keywords returned

    original_serp_key = os.getenv("SERP_API_KEY")
    os.environ["SERP_API_KEY"] = "fake_serp_key"

    response = client.post("/api/hot", json={"num": 3})
    assert response.status_code == 404
    assert response.json()["detail"] == "핫이슈가 없습니다."

    # Restore original SERP_API_KEY
    if original_serp_key:
        os.environ["SERP_API_KEY"] = original_serp_key
    else:
        del os.environ["SERP_API_KEY"]
