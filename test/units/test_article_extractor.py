import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from scrapers.article_extractor import ArticleExtractor
import requests


@pytest.fixture
def article_extractor():
    return ArticleExtractor()


@pytest.mark.asyncio
async def test_extract_single_success(article_extractor):
    # Mock requests.Session.get for successful response
    with patch("requests.Session.get") as mock_get:
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.text = (
            "<html><body><h1>Test Title</h1><p>Test content.</p></body></html>"
        )
        mock_response.raise_for_status.return_value = None
        mock_get.return_value.__enter__.return_value = mock_response

        # Mock trafilatura.extract
        with patch("trafilatura.extract", return_value="Test content.") as mock_extract:
            url_data = {"url": "http://example.com/article1", "title": "Test Title"}
            result = await article_extractor.extract_single(url_data)

            assert result is not None
            assert result["url"] == "http://example.com/article1"
            assert result["title"] == "Test Title"
            assert result["input_text"] == "Test content."
            mock_get.assert_called_once()
            mock_extract.assert_called_once()


@pytest.mark.asyncio
async def test_extract_single_timeout_with_retry(article_extractor):
    # Mock requests.Session.get to raise Timeout on first two attempts, then succeed
    with patch("requests.Session.get") as mock_get:
        mock_response_success = AsyncMock()
        mock_response_success.status_code = 200
        mock_response_success.text = (
            "<html><body><h1>Test Title</h1><p>Test content.</p></body></html>"
        )
        mock_response_success.raise_for_status.return_value = None

        mock_get.side_effect = [
            requests.exceptions.Timeout("Read timed out."),
            requests.exceptions.Timeout("Read timed out."),
            mock_response_success,
        ]

        with patch("trafilatura.extract", return_value="Test content."):
            url_data = {
                "url": "http://example.com/timeout_article",
                "title": "Timeout Test",
            }
            result = await article_extractor.extract_single(url_data)

            assert result is not None
            assert mock_get.call_count == 3  # Should be called 3 times due to 2 retries


@pytest.mark.asyncio
async def test_extract_single_max_retries_exceeded(article_extractor):
    # Mock requests.Session.get to always raise Timeout
    with patch(
        "requests.Session.get",
        side_effect=requests.exceptions.Timeout("Read timed out."),
    ) as mock_get:
        url_data = {"url": "http://example.com/fail_article", "title": "Fail Test"}
        result = await article_extractor.extract_single(url_data)

        assert result is None
        assert (
            mock_get.call_count == 3
        )  # Should be called 3 times (initial + 2 retries)


@pytest.mark.asyncio
async def test_extract_single_empty_content(article_extractor):
    # Mock requests.Session.get for successful response but empty content
    with patch("requests.Session.get") as mock_get:
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.text = "<html><body></body></html>"
        mock_response.raise_for_status.return_value = None
        mock_get.return_value.__enter__.return_value = mock_response

        with patch("trafilatura.extract", return_value=""):
            url_data = {"url": "http://example.com/empty", "title": "Empty Content"}
            result = await article_extractor.extract_single(url_data)

            assert result is None


@pytest.mark.asyncio
async def test_search_multiple_urls(article_extractor):
    # Mock extract_single to simulate results
    with patch.object(
        article_extractor, "extract_single", new_callable=AsyncMock
    ) as mock_extract_single:
        mock_extract_single.side_effect = [
            {"url": "url1", "title": "title1", "input_text": "content1"},
            None,  # Simulate a failed extraction
            {"url": "url3", "title": "title3", "input_text": "content3"},
        ]

        urls_data = [
            {"url": "url1", "title": "title1"},
            {"url": "url2", "title": "title2"},
            {"url": "url3", "title": "title3"},
        ]

        results = [r async for r in article_extractor.search(urls_data)]

        assert len(results) == 2  # Only successful extractions should be yielded
        assert results[0]["url"] == "url1"
        assert results[1]["url"] == "url3"
        assert mock_extract_single.call_count == 3
