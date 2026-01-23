"""
OpenPhish URL Source

Fetches URLs from OpenPhish's free feed.
https://openphish.com/
"""
import httpx
from typing import List

from .base import URLSource


class OpenPhishSource(URLSource):
    """
    Fetch phishing URLs from OpenPhish.

    OpenPhish provides a free feed of phishing URLs updated every 12 hours.
    """

    FEED_URL = "https://openphish.com/feed.txt"

    def __init__(self):
        super().__init__(name="openphish", reputation=0.90)
        self._last_urls = set()

    async def fetch_urls(self, limit: int = 100) -> List[str]:
        """
        Fetch phishing URLs from OpenPhish feed.

        Returns only new URLs (not seen in last fetch).
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(self.FEED_URL)
                response.raise_for_status()

                # Parse text feed (one URL per line)
                all_urls = [
                    line.strip()
                    for line in response.text.split('\n')
                    if line.strip() and line.startswith('http')
                ]

                # Filter to new URLs only
                current_set = set(all_urls)
                new_urls = list(current_set - self._last_urls)

                # Update cache
                self._last_urls = current_set

                return new_urls[:limit]

        except Exception as e:
            self.record_error()
            print(f"OpenPhish fetch error: {e}")
            return []
