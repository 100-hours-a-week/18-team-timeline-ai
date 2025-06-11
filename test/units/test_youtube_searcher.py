import pytest
import asyncio
from unittest.mock import AsyncMock, patch
from scrapers.youtube_searcher import YouTubeCommentAsyncFetcher


@pytest.fixture
def youtube_fetcher():
    return YouTubeCommentAsyncFetcher(api_key="test_api_key")


@pytest.mark.asyncio
async def test_extract_video_id_success():
    url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    video_id = YouTubeCommentAsyncFetcher.extract_video_id(url)
    assert video_id == "dQw4w9WgXcQ"


@pytest.mark.asyncio
async def test_extract_video_id_short_url():
    url = "https://youtu.be/dQw4w9WgXcQ"
    video_id = YouTubeCommentAsyncFetcher.extract_video_id(url)
    assert video_id == "dQw4w9WgXcQ"


@pytest.mark.asyncio
async def test_extract_video_id_invalid_url():
    url = "https://notyoutube.com/video"
    video_id = YouTubeCommentAsyncFetcher.extract_video_id(url)
    assert video_id is None


@pytest.mark.asyncio
async def test_search_success(youtube_fetcher):
    mock_video_id = "test_video_id"
    mock_df_data = [
        {
            "url": f"https://www.youtube.com/watch?v={mock_video_id}",
            "title": "Test Video",
        }
    ]

    # Mock the API calls for comments and captions
    with patch("aiohttp.ClientSession.get", new_callable=AsyncMock) as mock_get_request:
        # Mock for comments API call
        mock_comments_response = AsyncMock()
        mock_comments_response.status = 200
        mock_comments_response.json.return_value = {
            "items": [
                {
                    "snippet": {
                        "topLevelComment": {"snippet": {"textOriginal": "Comment 1"}}
                    }
                },
                {
                    "snippet": {
                        "topLevelComment": {"snippet": {"textOriginal": "Comment 2"}}
                    }
                },
            ]
        }

        # Mock for captions API call (assuming successful for a video ID)
        mock_captions_response = AsyncMock()
        mock_captions_response.status = 200
        mock_captions_response.text.return_value = '<transcript><text start="0">Caption 1</text><text start="1">Caption 2</text></transcript>'

        # Define side_effect for mock_get_request based on the URL
        def get_side_effect(url, params=None):
            if "commentThreads" in url:
                return mock_comments_response
            elif "api.video.google.com" in url:
                return mock_captions_response
            else:
                return AsyncMock(status=404)

        mock_get_request.side_effect = get_side_effect

        # Mock extract_video_id to return a fixed ID for the test
        with patch.object(
            youtube_fetcher, "extract_video_id", return_value=mock_video_id
        ):
            results = await youtube_fetcher.search(df=mock_df_data)

            assert len(results) == 1
            assert (
                results[0]["url"] == f"https://www.youtube.com/watch?v={mock_video_id}"
            )
            assert results[0]["comment"] == "Comment 1 Comment 2"
            assert results[0]["captions"] == "Caption 1 Caption 2"
            assert (
                mock_get_request.call_count == 2
            )  # One for comments, one for captions


@pytest.mark.asyncio
async def test_search_no_video_id(youtube_fetcher):
    mock_df_data = [{"url": "https://notyoutube.com/invalid", "title": "Invalid Video"}]
    results = await youtube_fetcher.search(df=mock_df_data)
    assert len(results) == 0


@pytest.mark.asyncio
async def test_search_api_error_comments(youtube_fetcher):
    mock_video_id = "test_video_id"
    mock_df_data = [
        {
            "url": f"https://www.youtube.com/watch?v={mock_video_id}",
            "title": "Test Video",
        }
    ]

    with patch("aiohttp.ClientSession.get", new_callable=AsyncMock) as mock_get_request:
        mock_comments_response_error = AsyncMock()
        mock_comments_response_error.status = 500
        mock_comments_response_error.text.return_value = "API Error"

        mock_captions_response = AsyncMock()
        mock_captions_response.status = 200
        mock_captions_response.text.return_value = "<transcript></transcript>"

        def get_side_effect(url, params=None):
            if "commentThreads" in url:
                return mock_comments_response_error
            elif "api.video.google.com" in url:
                return mock_captions_response
            else:
                return AsyncMock(status=404)

        mock_get_request.side_effect = get_side_effect

        with patch.object(
            youtube_fetcher, "extract_video_id", return_value=mock_video_id
        ):
            results = await youtube_fetcher.search(df=mock_df_data)

            assert len(results) == 0  # Should return empty if comments fail
            assert mock_get_request.call_count == 2  # Still tries both
