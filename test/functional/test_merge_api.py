import pytest
from fastapi.testclient import TestClient
from main import app  # Assuming your FastAPI app instance is in main.py
from unittest.mock import patch, AsyncMock
from domain.timeline_card import TimelineCard
from datetime import date

client = TestClient(app)


@pytest.fixture
def mock_total_pipeline():
    with patch(
        "pipelines.total_pipeline.TotalPipeline", new_callable=AsyncMock
    ) as MockTotalPipeline:
        instance = MockTotalPipeline.return_value
        instance.return_value = {
            "total_summary": {"title": ["Merged Title"], "summary": ["Merged content."]}
        }
        yield instance


def test_merge_timeline_success(mock_total_pipeline):
    # Create dummy TimelineCard objects for the request
    card1 = TimelineCard(
        title="Title 1",
        content="Content 1",
        duration="DAY",
        startAt=date(2023, 1, 1),
        endAt=date(2023, 1, 1),
        source=["http://example.com/source1"],
    )
    card2 = TimelineCard(
        title="Title 2",
        content="Content 2",
        duration="DAY",
        startAt=date(2023, 1, 2),
        endAt=date(2023, 1, 2),
        source=["http://example.com/source2"],
    )

    response = client.post(
        "/api/merge",
        json={
            "timeline": [
                card1.model_dump_json(),  # Convert Pydantic model to JSON string
                card2.model_dump_json(),
            ]
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "data" in data
    assert data["data"]["title"] == "Merged Title"
    assert data["data"]["content"] == "Merged content."
    assert data["data"]["source"] == [
        "http://example.com/source1",
        "http://example.com/source2",
    ]
    mock_total_pipeline.assert_called_once()


def test_merge_timeline_empty_timeline():
    response = client.post("/api/merge", json={"timeline": []})
    assert response.status_code == 400
    assert response.json()["detail"] == "Timeline이 비어 있습니다."


def test_merge_timeline_pipeline_failure():
    with patch(
        "pipelines.total_pipeline.TotalPipeline", new_callable=AsyncMock
    ) as MockTotalPipeline:
        MockTotalPipeline.return_value = AsyncMock(
            return_value=None
        )  # Simulate pipeline failure

        card = TimelineCard(
            title="Title 1",
            content="Content 1",
            duration="DAY",
            startAt=date(2023, 1, 1),
            endAt=date(2023, 1, 1),
            source=["http://example.com/source1"],
        )

        response = client.post(
            "/api/merge", json={"timeline": [card.model_dump_json()]}
        )
        assert response.status_code == 500
        assert response.json()["detail"] == "인공지능이 병합 요약에 실패했습니다."
