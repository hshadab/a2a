"""
PhishTank URL Source

Fetches URLs from PhishTank's free API.
https://phishtank.org/
"""
import httpx
from typing import List
import json
import gzip
import io

from .base import URLSource


class PhishTankSource(URLSource):
    """
    Fetch phishing URLs from PhishTank.

    PhishTank provides a free database of verified phishing URLs.
    Requires registration for API key, but has a public data dump.
    """

    # PhishTank data dump URL (updated hourly)
    DATA_URL = "http://data.phishtank.com/data/online-valid.json.gz"

    # Alternative: PhishTank API (requires key)
    API_URL = "https://checkurl.phishtank.com/checkurl/"

    def __init__(self, api_key: str = None):
        super().__init__(name="phishtank", reputation=0.95)
        self.api_key = api_key
        self._cache = []
        self._cache_index = 0

    async def fetch_urls(self, limit: int = 100) -> List[str]:
        """
        Fetch phishing URLs from PhishTank.

        Uses the public data dump for bulk fetching.
        """
        # If we have cached data, return from cache
        if self._cache and self._cache_index < len(self._cache):
            urls = self._cache[self._cache_index:self._cache_index + limit]
            self._cache_index += limit
            return urls

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                # Fetch the gzipped JSON dump
                response = await client.get(self.DATA_URL)
                response.raise_for_status()

                # Decompress and parse
                gzip_bytes = io.BytesIO(response.content)
                with gzip.GzipFile(fileobj=gzip_bytes) as f:
                    data = json.load(f)

                # Extract URLs
                urls = []
                for entry in data:
                    if entry.get("verified") == "yes":
                        url = entry.get("url")
                        if url:
                            urls.append(url)

                # Cache for subsequent calls
                self._cache = urls
                self._cache_index = limit

                return urls[:limit]

        except Exception as e:
            self.record_error()
            print(f"PhishTank fetch error: {e}")
            return []

    async def check_url(self, url: str) -> dict:
        """
        Check a specific URL against PhishTank API.

        Requires API key.
        """
        if not self.api_key:
            return {"error": "No API key configured"}

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.API_URL,
                    data={
                        "url": url,
                        "format": "json",
                        "app_key": self.api_key
                    }
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            self.record_error()
            return {"error": str(e)}
