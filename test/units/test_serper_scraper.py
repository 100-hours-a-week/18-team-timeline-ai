import pytest
from unittest.mock import patch, MagicMock
from scrapers.serper import distribute_news_serper


def test_distribute_news_serper_success():
    mock_response = {
        "news": [
            {
                "title": "Title 1",
                "link": "http://example.com/news1",
                "date": "2 days ago",
            },
            {
                "title": "Title 2",
                "link": "http://example.com/news2",
                "date": "1 day ago",
            },
        ]
    }

    with patch("requests.get") as mock_get:
        mock_get.return_value = MagicMock(json=lambda: mock_response, status_code=200)

        result = distribute_news_serper(
            query="test query",
            startAt="2023-01-01",
            endAt="2023-01-07",
            api_key="fake_key",
        )

        assert len(result) == 2
        assert result[0] == ("http://example.com/news1", "Title 1", "2023-01-05")
        assert result[1] == ("http://example.com/news2", "Title 2", "2023-01-06")
        mock_get.assert_called_once()


def test_distribute_news_serper_empty_news():
    mock_response = {"news": []}

    with patch("requests.get") as mock_get:
        mock_get.return_value = MagicMock(json=lambda: mock_response, status_code=200)

        result = distribute_news_serper(
            query="empty query",
            startAt="2023-01-01",
            endAt="2023-01-07",
            api_key="fake_key",
        )

        assert result is None
        mock_get.assert_called_once()


def test_distribute_news_serper_api_error():
    with patch("requests.get") as mock_get:
        mock_get.return_value = MagicMock(status_code=500, text="Internal Server Error")

        result = distribute_news_serper(
            query="error query",
            startAt="2023-01-01",
            endAt="2023-01-07",
            api_key="fake_key",
        )

        assert result is None
        mock_get.assert_called_once()


def test_distribute_news_serper_no_news_key():
    mock_response = {"some_other_key": []}

    with patch("requests.get") as mock_get:
        mock_get.return_value = MagicMock(json=lambda: mock_response, status_code=200)

        result = distribute_news_serper(
            query="no news key",
            startAt="2023-01-01",
            endAt="2023-01-07",
            api_key="fake_key",
        )

        assert result is None
        mock_get.assert_called_once()
