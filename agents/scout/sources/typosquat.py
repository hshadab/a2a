"""
Typosquatting Domain Generator

Proactively generates typosquatting variations of known brands
and checks if they resolve (are registered and active).
This is ORIGINAL discovery - finding threats before anyone reports them.
"""
import asyncio
import socket
from typing import List, Set
from datetime import datetime, timedelta

import httpx

from .base import URLSource


class TyposquatSource(URLSource):
    """
    Proactively discovers phishing domains by:
    1. Generating typosquatting variations of popular brands
    2. Checking if those domains are registered and active
    3. Returning active suspicious domains for classification

    This is truly original threat discovery.
    """

    # Brands to protect
    BRANDS = [
        "paypal", "amazon", "apple", "microsoft", "google",
        "netflix", "facebook", "instagram", "chase", "coinbase",
        "binance", "metamask", "wellsfargo", "bankofamerica"
    ]

    # Suspicious TLDs often used for phishing
    SUSPICIOUS_TLDS = [
        ".xyz", ".top", ".tk", ".ml", ".ga", ".cf",
        ".pw", ".cc", ".ws", ".site", ".online", ".live"
    ]

    # Typosquatting transformations
    TRANSFORMS = [
        lambda s: s.replace('a', '4'),      # payp4l
        lambda s: s.replace('e', '3'),      # n3tflix
        lambda s: s.replace('i', '1'),      # m1crosoft
        lambda s: s.replace('o', '0'),      # amaz0n
        lambda s: s.replace('l', '1'),      # paypa1
        lambda s: s + s[-1],                # amazzonn
        lambda s: s[:-1],                   # amazo
        lambda s: s[0] + s[0] + s[1:],      # aamazon
        lambda s: s.replace('ll', 'l'),     # paypa
        lambda s: s + '-secure',            # amazon-secure
        lambda s: s + '-login',             # paypal-login
        lambda s: s + '-verify',            # chase-verify
        lambda s: 'secure-' + s,            # secure-amazon
        lambda s: 'login-' + s,             # login-paypal
        lambda s: s + 'security',           # amazonsecurity
        lambda s: s + 'support',            # applesupport
    ]

    def __init__(self, check_interval_hours: int = 6):
        super().__init__(name="typosquat", reputation=0.80)
        self._seen_domains: Set[str] = set()
        self._last_full_scan: datetime = datetime.min
        self._check_interval = timedelta(hours=check_interval_hours)

    async def fetch_urls(self, limit: int = 50) -> List[str]:
        """
        Generate typosquat variations and check which ones are active.
        """
        # Only do full scan periodically
        now = datetime.utcnow()
        if now - self._last_full_scan < self._check_interval:
            return []

        active_urls: List[str] = []
        candidates = self._generate_candidates()

        async with httpx.AsyncClient(timeout=5.0) as client:
            # Check candidates in batches
            for domain in candidates:
                if len(active_urls) >= limit:
                    break

                if domain in self._seen_domains:
                    continue

                # Check if domain resolves
                if await self._is_domain_active(domain, client):
                    url = f"https://{domain}/"
                    active_urls.append(url)
                    self._seen_domains.add(domain)

                # Rate limit
                await asyncio.sleep(0.1)

        self._last_full_scan = now
        return active_urls

    def _generate_candidates(self) -> List[str]:
        """Generate typosquatting domain candidates."""
        candidates = []

        for brand in self.BRANDS:
            for transform in self.TRANSFORMS:
                try:
                    typo = transform(brand)
                    if typo != brand:  # Only if actually different
                        for tld in self.SUSPICIOUS_TLDS:
                            candidates.append(typo + tld)
                except Exception:
                    continue

        return candidates

    async def _is_domain_active(self, domain: str, client: httpx.AsyncClient) -> bool:
        """
        Check if a domain is registered and has an active web server.
        """
        try:
            # First check DNS resolution
            socket.setdefaulttimeout(2)
            socket.gethostbyname(domain)

            # Then check if web server responds
            response = await client.head(f"https://{domain}/", follow_redirects=True)
            return response.status_code < 500

        except (socket.gaierror, socket.timeout):
            # Domain doesn't resolve
            return False
        except httpx.HTTPError:
            # Domain resolves but HTTP failed - still suspicious
            return True
        except Exception:
            return False
