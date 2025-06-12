import pytest
from fastapi.testclient import TestClient
from main import app  # Assuming your FastAPI app instance is in main.py
import os

client = TestClient(app)


def test_get_timeline_success():
    response = client.post(
        "/api/timeline",
        json={"query": ["최신 뉴스"], "startAt": "2023-01-01", "endAt": "2023-01-07"},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "timeline" in data["data"]
    assert isinstance(data["data"]["timeline"], list)
    assert len(data["data"]["timeline"]) > 0
    assert "failed_urls" in data["data"]


def test_get_timeline_empty_query():
    response = client.post(
        "/api/timeline",
        json={"query": [], "startAt": "2023-01-01", "endAt": "2023-01-07"},
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "URL이 비어있습니다."


def test_get_timeline_invalid_url():
    response = client.post(
        "/api/timeline",
        json={"query": ["invalid-url"], "startAt": "2023-01-01", "endAt": "2023-01-07"},
    )
    assert response.status_code == 400
    assert "유효하지 않은 URL입니다" in response.json()["detail"]


def test_get_timeline_no_serper_key():
    # Temporarily remove SERPER_API_KEY environment variable
    # (This requires careful handling of env vars in tests)
    # For simplicity, this test might need a more sophisticated setup
    # or be skipped in automated environments where env vars are always set.
    # For now, we'll assume a missing key leads to a 500 error as per current code.
    original_serper_key = os.getenv("SERPER_API_KEY")
    os.environ["SERPER_API_KEY"] = ""

    response = client.post(
        "/api/timeline",
        json={"query": ["test news"], "startAt": "2023-01-01", "endAt": "2023-01-07"},
    )
    assert response.status_code == 500
    assert response.json()["detail"] == "SERPER_API_KEY를 찾을 수 없습니다."

    # Restore original SERPER_API_KEY
    if original_serper_key:
        os.environ["SERPER_API_KEY"] = original_serper_key
    else:
        del os.environ["SERPER_API_KEY"]


# You would add more tests here for various scenarios, e.g.,
# - no scraping results
# - AI summarization failure
# - specific URL timeout scenarios (if mockable in functional test context)
# - valid date ranges that return no news
