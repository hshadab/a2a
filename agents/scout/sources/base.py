"""
Base class for URL sources
"""
from abc import ABC, abstractmethod
from typing import List, Tuple
from datetime import datetime


class URLSource(ABC):
    """Base class for URL discovery sources"""

    def __init__(self, name: str, reputation: float = 0.5):
        """
        Args:
            name: Source identifier
            reputation: Trust score 0-1 (higher = more trusted)
        """
        self.name = name
        self.reputation = reputation
        self.last_fetch: datetime = None
        self.total_urls_fetched: int = 0
        self.errors: int = 0

    @abstractmethod
    async def fetch_urls(self, limit: int = 100) -> List[str]:
        """
        Fetch URLs from this source.

        Returns list of URLs to analyze.
        """
        pass

    async def fetch_with_metadata(self, limit: int = 100) -> Tuple[List[str], dict]:
        """Fetch URLs with source metadata"""
        urls = await self.fetch_urls(limit)
        self.last_fetch = datetime.utcnow()
        self.total_urls_fetched += len(urls)

        metadata = {
            "source": self.name,
            "reputation": self.reputation,
            "count": len(urls),
            "fetched_at": self.last_fetch.isoformat()
        }

        return urls, metadata

    def record_error(self):
        """Record an error from this source"""
        self.errors += 1
