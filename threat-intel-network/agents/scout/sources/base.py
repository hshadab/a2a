"""
Base class for URL sources
"""
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any, Optional
from datetime import datetime


class URLSource(ABC):
    """Base class for URL discovery sources"""

    name: str
    reputation: float
    last_fetch: Optional[datetime]
    total_urls_fetched: int
    errors: int

    def __init__(self, name: str, reputation: float = 0.5) -> None:
        """
        Args:
            name: Source identifier
            reputation: Trust score 0-1 (higher = more trusted)
        """
        self.name = name
        self.reputation = reputation
        self.last_fetch: Optional[datetime] = None
        self.total_urls_fetched: int = 0
        self.errors: int = 0

    @abstractmethod
    async def fetch_urls(self, limit: int = 100) -> List[str]:
        """
        Fetch URLs from this source.

        Returns list of URLs to analyze.
        """
        pass

    async def fetch_with_metadata(self, limit: int = 100) -> Tuple[List[str], Dict[str, Any]]:
        """Fetch URLs with source metadata"""
        urls = await self.fetch_urls(limit)
        self.last_fetch = datetime.utcnow()
        self.total_urls_fetched += len(urls)

        metadata: Dict[str, Any] = {
            "source": self.name,
            "reputation": self.reputation,
            "count": len(urls),
            "fetched_at": self.last_fetch.isoformat()
        }

        return urls, metadata

    def record_error(self) -> None:
        """Record an error from this source"""
        self.errors += 1
