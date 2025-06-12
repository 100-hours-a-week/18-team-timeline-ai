import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
import pandas as pd

from scrapers.daum_vclip_searcher import DaumVclipSearcher
from scrapers.youtube_searcher import YouTubeCommentAsyncFetcher


@pytest.fixture
def mock_daum_vclip_searcher():
    with patch(
        "scrapers.daum_vclip_searcher.DaumVclipSearcher", autospec=True
    ) as MockDaumSearcher:
        instance = MockDaumSearcher.return_value
        # Simulate a DataFrame returned by DaumVclipSearcher
        instance.search.return_value = pd.DataFrame(
            {
                "url": [
                    "https://www.youtube.com/watch?v=video1",
                    "https://www.youtube.com/watch?v=video2",
                ],
                "title": ["Video Title 1", "Video Title 2"],
            }
        )
        yield instance


@pytest.fixture
def mock_youtube_fetcher():
    with patch(
        "scrapers.youtube_searcher.YouTubeCommentAsyncFetcher", autospec=True
    ) as MockYoutubeFetcher:
        instance = MockYoutubeFetcher.return_value
        # Simulate comments and captions returned by YouTubeCommentAsyncFetcher
        instance.search.side_effect = [
            AsyncMock(
                return_value=[
                    {
                        "url": "https://www.youtube.com/watch?v=video1",
                        "comment": "Comment A",
                        "captions": "Caption X",
                    },
                    {
                        "url": "https://www.youtube.com/watch?v=video1",
                        "comment": "Comment B",
                        "captions": "Caption Y",
                    },
                ]
            ),
            AsyncMock(
                return_value=[
                    {
                        "url": "https://www.youtube.com/watch?v=video2",
                        "comment": "Comment C",
                        "captions": "Caption Z",
                    }
                ]
            ),
        ]
        yield instance


@pytest.mark.asyncio
async def test_comment_pipeline_integration(
    mock_daum_vclip_searcher, mock_youtube_fetcher
):
    # Instantiate the actual scrapers, but their 'search' methods are mocked by fixtures
    daum_searcher = DaumVclipSearcher(api_key="dummy_daum_key")
    youtube_fetcher_instance = YouTubeCommentAsyncFetcher(api_key="dummy_youtube_key")

    # Simulate the flow from main function in api/comment.py
    df_result = daum_searcher.search(query="test query")
    assert not df_result.empty
    mock_daum_vclip_searcher.search.assert_called_once_with(query="test query")

    ripple_results = await youtube_fetcher_instance.search(df=df_result)
    assert len(ripple_results) == 3  # Total comments and captions
    assert ripple_results[0]["comment"] == "Comment A"
    assert ripple_results[1]["comment"] == "Comment B"
    assert ripple_results[2]["comment"] == "Comment C"
    assert mock_youtube_fetcher.search.call_count == 1  # Only called once with the df

    # Check that the data structure is as expected for further processing (e.g., sentiment analysis)
    extracted_comments = [r["comment"] for r in ripple_results if r.get("comment")]
    assert len(extracted_comments) == 3
    assert "Comment A" in extracted_comments
    assert "Comment B" in extracted_comments
    assert "Comment C" in extracted_comments


# You can add more integration tests here for edge cases, e.g.:
# - When Daum search returns empty DataFrame
# - When YouTube search returns no comments/captions for a video
