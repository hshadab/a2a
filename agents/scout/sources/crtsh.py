"""
Certificate Transparency Source

Monitors crt.sh for newly issued certificates that may indicate phishing domains.
Looks for certificates issued to domains that impersonate known brands.
"""
import asyncio
import re
from datetime import datetime, timedelta
from typing import List, Set, Optional
from urllib.parse import urlparse

import httpx

from .base import URLSource


class CertTransparencySource(URLSource):
    """
    Source that queries crt.sh for recent certificates matching brand patterns.

    Certificate Transparency logs provide a way to detect newly registered
    domains that may be used for phishing by looking for certificates
    issued to suspicious domain names.
    """

    API_URL = "https://crt.sh/"

    # Brand patterns to search for in CT logs
    BRAND_PATTERNS = [
        "%paypal%",
        "%amazon%",
        "%microsoft%",
        "%google%",
        "%apple%",
        "%facebook%",
        "%instagram%",
        "%netflix%",
        "%chase%",
        "%wellsfargo%",
        "%bankofamerica%",
        "%coinbase%",
        "%binance%",
        "%metamask%",
    ]

    # Legitimate brand domains to exclude
    LEGITIMATE_DOMAINS = {
        "paypal.com", "paypal.me", "paypalobjects.com",
        "amazon.com", "amazon.co.uk", "amazonaws.com", "amazonses.com",
        "microsoft.com", "microsoftonline.com", "azure.com", "office.com",
        "google.com", "google.co.uk", "googleapis.com", "gstatic.com",
        "apple.com", "icloud.com", "apple-dns.net",
        "facebook.com", "fb.com", "fbcdn.net",
        "instagram.com", "cdninstagram.com",
        "netflix.com", "nflximg.net", "nflxso.net",
        "chase.com", "jpmorgan.com",
        "wellsfargo.com",
        "bankofamerica.com",
        "coinbase.com",
        "binance.com", "binance.us",
        "metamask.io",
    }

    def __init__(self, lookback_days: int = 1):
        """
        Args:
            lookback_days: How many days back to search for certificates
        """
        super().__init__(name="crtsh", reputation=0.85)
        self.lookback_days = lookback_days
        self._seen_domains: Set[str] = set()
        self._last_query_time: Optional[datetime] = None

    async def fetch_urls(self, limit: int = 100) -> List[str]:
        """
        Fetch suspicious domains from Certificate Transparency logs.

        Returns list of URLs constructed from suspicious newly-registered domains.
        """
        suspicious_urls: List[str] = []

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Query CT logs for each brand pattern
                for pattern in self.BRAND_PATTERNS:
                    if len(suspicious_urls) >= limit:
                        break

                    domains = await self._query_crtsh(client, pattern)

                    for domain in domains:
                        if len(suspicious_urls) >= limit:
                            break

                        # Skip if already seen
                        if domain in self._seen_domains:
                            continue

                        # Skip legitimate domains
                        if self._is_legitimate(domain):
                            continue

                        # Add to seen set
                        self._seen_domains.add(domain)

                        # Convert domain to URL (assume HTTPS)
                        url = f"https://{domain}/"
                        suspicious_urls.append(url)

                    # Small delay between API calls to be respectful
                    await asyncio.sleep(0.5)

            self._last_query_time = datetime.utcnow()

        except httpx.TimeoutException:
            self.record_error()
        except httpx.HTTPError as e:
            self.record_error()
        except Exception as e:
            self.record_error()

        return suspicious_urls

    async def _query_crtsh(self, client: httpx.AsyncClient, pattern: str) -> List[str]:
        """
        Query crt.sh for certificates matching a pattern.

        Args:
            client: HTTP client
            pattern: SQL LIKE pattern to search for

        Returns:
            List of domain names from matching certificates
        """
        domains: List[str] = []

        try:
            # Use JSON output for easier parsing
            params = {
                "q": pattern,
                "output": "json",
            }

            response = await client.get(self.API_URL, params=params)
            response.raise_for_status()

            data = response.json()

            if not data:
                return []

            # Extract unique domains from results
            seen_in_query: Set[str] = set()

            for entry in data:
                # Get the domain name value
                name_value = entry.get("name_value", "")

                # Handle wildcard certs and multi-domain certs
                for name in name_value.split("\n"):
                    name = name.strip().lower()

                    # Remove wildcard prefix
                    if name.startswith("*."):
                        name = name[2:]

                    # Skip if already seen in this query
                    if name in seen_in_query:
                        continue

                    # Basic validation
                    if not name or "." not in name:
                        continue

                    # Filter by certificate age if we have the data
                    not_before = entry.get("not_before")
                    if not_before:
                        try:
                            cert_date = datetime.fromisoformat(not_before.replace("Z", "+00:00"))
                            cutoff = datetime.utcnow().replace(tzinfo=cert_date.tzinfo) - timedelta(days=self.lookback_days)
                            if cert_date < cutoff:
                                continue
                        except (ValueError, TypeError):
                            pass

                    seen_in_query.add(name)
                    domains.append(name)

        except (httpx.HTTPError, ValueError, KeyError):
            pass

        return domains

    def _is_legitimate(self, domain: str) -> bool:
        """
        Check if a domain is a legitimate brand domain.

        Args:
            domain: Domain to check

        Returns:
            True if domain appears to be legitimate
        """
        domain = domain.lower()

        # Check exact match with legitimate domains
        if domain in self.LEGITIMATE_DOMAINS:
            return True

        # Check if it's a subdomain of a legitimate domain
        for legit in self.LEGITIMATE_DOMAINS:
            if domain.endswith("." + legit):
                return True

        return False

    def clear_seen_cache(self):
        """Clear the cache of seen domains"""
        self._seen_domains.clear()
