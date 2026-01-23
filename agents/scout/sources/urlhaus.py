"""
URLhaus URL Source

Fetches malware URLs from URLhaus (abuse.ch).
https://urlhaus.abuse.ch/
"""
import httpx
from typing import List
import csv
import io

from .base import URLSource


class URLhausSource(URLSource):
    """
    Fetch malware URLs from URLhaus.

    URLhaus is a project from abuse.ch collecting malware distribution URLs.
    """

    # CSV feed of online URLs
    FEED_URL = "https://urlhaus.abuse.ch/downloads/csv_online/"

    def __init__(self):
        super().__init__(name="urlhaus", reputation=0.92)
        self._seen_ids = set()

    async def fetch_urls(self, limit: int = 100) -> List[str]:
        """
        Fetch malware URLs from URLhaus CSV feed.
        """
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.get(self.FEED_URL)
                response.raise_for_status()

                # Parse CSV (skip comment lines starting with #)
                lines = [
                    line for line in response.text.split('\n')
                    if line and not line.startswith('#')
                ]

                urls = []
                reader = csv.reader(io.StringIO('\n'.join(lines)))

                for row in reader:
                    if len(row) >= 3:
                        url_id = row[0]
                        url = row[2]

                        # Skip if we've seen this ID
                        if url_id in self._seen_ids:
                            continue

                        self._seen_ids.add(url_id)
                        if url.startswith('http'):
                            urls.append(url)

                        if len(urls) >= limit:
                            break

                return urls

        except Exception as e:
            self.record_error()
            print(f"URLhaus fetch error: {e}")
            return []
