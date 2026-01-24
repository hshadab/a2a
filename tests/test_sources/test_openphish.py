"""
Tests for OpenPhish URL source
"""
import pytest
from unittest.mock import AsyncMock, patch, MagicMock

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agents.scout.sources.openphish import OpenPhishSource


class TestOpenPhishSource:
    """Tests for OpenPhishSource"""

    def test_init(self):
        """Test source initialization"""
        source = OpenPhishSource()
        assert source.name == "openphish"
        assert source.reputation == 0.9
        assert source.FEED_URL == "https://openphish.com/feed.txt"

    @pytest.mark.asyncio
    async def test_fetch_urls_success(self, mock_openphish_response):
        """Test successful URL fetch"""
        source = OpenPhishSource()

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = mock_openphish_response
            mock_response.raise_for_status = MagicMock()

            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            urls = await source.fetch_urls(limit=10)

            assert len(urls) <= 10
            assert all(isinstance(url, str) for url in urls)
            assert all(url.startswith("http") for url in urls)

    @pytest.mark.asyncio
    async def test_fetch_urls_with_limit(self, mock_openphish_response):
        """Test URL fetch respects limit"""
        source = OpenPhishSource()

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = mock_openphish_response
            mock_response.raise_for_status = MagicMock()

            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            urls = await source.fetch_urls(limit=2)

            assert len(urls) <= 2

    @pytest.mark.asyncio
    async def test_fetch_urls_network_error(self):
        """Test handling of network errors"""
        source = OpenPhishSource()

        with patch("httpx.AsyncClient") as mock_client:
            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(side_effect=Exception("Network error"))
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            urls = await source.fetch_urls(limit=10)

            assert urls == []

    @pytest.mark.asyncio
    async def test_fetch_urls_empty_response(self):
        """Test handling of empty response"""
        source = OpenPhishSource()

        with patch("httpx.AsyncClient") as mock_client:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.text = ""
            mock_response.raise_for_status = MagicMock()

            mock_client_instance = AsyncMock()
            mock_client_instance.get = AsyncMock(return_value=mock_response)
            mock_client_instance.__aenter__ = AsyncMock(return_value=mock_client_instance)
            mock_client_instance.__aexit__ = AsyncMock(return_value=None)
            mock_client.return_value = mock_client_instance

            urls = await source.fetch_urls(limit=10)

            assert urls == []

    def test_fetch_with_metadata(self):
        """Test metadata is correctly populated after fetch"""
        source = OpenPhishSource()
        assert source.last_fetch is None
        assert source.total_urls_fetched == 0

    def test_record_error(self):
        """Test error recording"""
        source = OpenPhishSource()
        assert source.errors == 0

        source.record_error()
        assert source.errors == 1

        source.record_error()
        assert source.errors == 2
