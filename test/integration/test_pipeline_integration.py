import pytest
from unittest.mock import AsyncMock, patch
from pipelines.summary_pipeline import Pipeline
from pipelines.total_pipeline import TotalPipeline


@pytest.fixture
def mock_article_extractor():
    with patch(
        "scrapers.article_extractor.ArticleExtractor", autospec=True
    ) as MockExtractor:
        instance = MockExtractor.return_value
        instance.search.return_value = AsyncMock(
            return_value=[
                {
                    "url": "http://example.com/article1",
                    "title": "Test Article 1",
                    "input_text": "Content of article 1.",
                },
                {
                    "url": "http://example.com/article2",
                    "title": "Test Article 2",
                    "input_text": "Content of article 2.",
                },
            ]
        ).__aiter__()
        yield instance


@pytest.fixture
def mock_host():
    with patch("inference.host.Host", autospec=True) as MockHost:
        instance = MockHost.return_value
        instance.__aenter__.return_value = instance
        instance.process_request.return_value = AsyncMock(
            side_effect=[
                {"choices": [{"message": {"content": "Summary of article 1."}}]},
                {"choices": [{"message": {"content": "Summary of article 2."}}]},
            ]
        )
        yield instance


@pytest.fixture
def mock_openai_client():
    with patch("openai.AsyncOpenAI", autospec=True) as MockOpenAI:
        instance = MockOpenAI.return_value
        instance.chat.completions.create.return_value = AsyncMock()
        instance.chat.completions.create.return_value.choices = [
            AsyncMock(message=AsyncMock(content="Total summary title"))
        ]
        instance.chat.completions.create.return_value.choices = [
            AsyncMock(message=AsyncMock(content="Total summary content"))
        ]
        yield instance


@pytest.mark.asyncio
async def test_pipeline_integration(
    mock_article_extractor, mock_host, mock_openai_client
):
    urls_data = [
        {"url": "http://example.com/article1", "title": "Test Article 1"},
        {"url": "http://example.com/article2", "title": "Test Article 2"},
    ]
    server_url = "http://mock-server.com"
    model_name = "mock-model"
    openai_api_key = "mock-api-key"

    # Test Summary Pipeline
    summary_results = await Pipeline(urls_data, server_url, model_name)

    assert "http://example.com/article1" in summary_results
    assert "http://example.com/article2" in summary_results
    assert (
        summary_results["http://example.com/article1"]["summary"][0]
        == "Summary of article 1."
    )
    assert (
        summary_results["http://example.com/article2"]["summary"][0]
        == "Summary of article 2."
    )
    mock_article_extractor.search.assert_called_once_with(urls_data)
    assert mock_host.__aenter__.called
    assert mock_host.process_request.call_count == 2  # One for each article

    # Prepare data for Total Pipeline
    total_texts = [
        summary_results["http://example.com/article1"]["summary"][0],
        summary_results["http://example.com/article2"]["summary"][0],
    ]

    # Test Total Pipeline
    total_summary_results = await TotalPipeline(total_texts, openai_api_key, model_name)

    assert "total_summary" in total_summary_results
    assert total_summary_results["total_summary"]["title"][0] == "Total summary title"
    assert (
        total_summary_results["total_summary"]["summary"][0] == "Total summary content"
    )
    mock_openai_client.chat.completions.create.assert_called_once()


# You would add more integration tests here, e.g.,
# - Error handling scenarios in pipelines
# - Edge cases like empty content from extractor
