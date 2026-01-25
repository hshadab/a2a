"""
Twitter/X Source

Monitors security researcher accounts on Twitter/X for shared phishing URLs.
Requires a Twitter API bearer token for access.
"""
import logging
import re
from datetime import datetime, timedelta
from typing import List, Set, Optional
from urllib.parse import urlparse

import httpx

from .base import URLSource

logger = logging.getLogger("scout.twitter")


class TwitterSource(URLSource):
    """
    Source that monitors Twitter/X for phishing URLs shared by security researchers.

    Follows known threat intelligence accounts and extracts URLs from their tweets.
    """

    API_URL = "https://api.twitter.com/2"

    # Security researcher accounts to monitor
    MONITORED_ACCOUNTS = [
        "MalwareHunterTeam",
        "abuse_ch",
        "PhishingAI",
        "URLhaus",
        "malaboratorium",
        "JAMESWT_MHT",
        "executemalware",
        "ViriBack",
        "pr0xylife",
        "Cryptolaemus1",
    ]

    # Keywords that indicate a tweet contains phishing/malware URLs
    INDICATOR_KEYWORDS = [
        "phishing",
        "phish",
        "credential",
        "scam",
        "fake login",
        "malware",
        "dropper",
        "c2",
        "ioc",
        "indicator",
    ]

    # Domains to exclude from results (common URL shorteners, etc.)
    EXCLUDED_DOMAINS = {
        "twitter.com",
        "t.co",
        "bit.ly",
        "goo.gl",
        "tinyurl.com",
        "youtu.be",
        "youtube.com",
        "github.com",
        "pastebin.com",
        "virustotal.com",
        "urlscan.io",
        "any.run",
        "app.any.run",
        "bazaar.abuse.ch",
    }

    def __init__(self, bearer_token: Optional[str] = None):
        """
        Args:
            bearer_token: Twitter API v2 bearer token
        """
        super().__init__(name="twitter", reputation=0.88)
        self.bearer_token = bearer_token
        self._seen_tweet_ids: Set[str] = set()
        self._user_id_cache: dict = {}

    async def fetch_urls(self, limit: int = 100) -> List[str]:
        """
        Fetch URLs from recent tweets by monitored security researcher accounts.

        Returns list of URLs that appear to be phishing/malware related.
        """
        if not self.bearer_token:
            # No API key, return empty
            return []

        suspicious_urls: List[str] = []

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                headers = {
                    "Authorization": f"Bearer {self.bearer_token}",
                }

                for account in self.MONITORED_ACCOUNTS:
                    if len(suspicious_urls) >= limit:
                        break

                    urls = await self._fetch_account_urls(client, headers, account)

                    for url in urls:
                        if len(suspicious_urls) >= limit:
                            break
                        if url not in suspicious_urls:
                            suspicious_urls.append(url)

        except httpx.TimeoutException as e:
            self.record_error()
            logger.warning(f"Twitter API timeout: {e}")
        except httpx.HTTPError as e:
            self.record_error()
            logger.warning(f"Twitter API HTTP error: {e}")
        except Exception as e:
            self.record_error()
            logger.error(f"Twitter API fetch error: {e}", exc_info=True)

        return suspicious_urls

    async def _fetch_account_urls(
        self,
        client: httpx.AsyncClient,
        headers: dict,
        username: str
    ) -> List[str]:
        """
        Fetch URLs from a specific account's recent tweets.

        Args:
            client: HTTP client
            headers: Request headers with auth
            username: Twitter username to fetch

        Returns:
            List of URLs from tweets
        """
        urls: List[str] = []

        try:
            # Get user ID if not cached
            if username not in self._user_id_cache:
                user_response = await client.get(
                    f"{self.API_URL}/users/by/username/{username}",
                    headers=headers,
                )
                user_response.raise_for_status()
                user_data = user_response.json()
                self._user_id_cache[username] = user_data["data"]["id"]

            user_id = self._user_id_cache[username]

            # Fetch recent tweets
            params = {
                "max_results": 20,
                "tweet.fields": "created_at,entities",
                "expansions": "entities.urls",
            }

            response = await client.get(
                f"{self.API_URL}/users/{user_id}/tweets",
                headers=headers,
                params=params,
            )
            response.raise_for_status()

            data = response.json()
            tweets = data.get("data", [])

            for tweet in tweets:
                tweet_id = tweet.get("id")

                # Skip if we've seen this tweet
                if tweet_id in self._seen_tweet_ids:
                    continue

                self._seen_tweet_ids.add(tweet_id)

                # Check if tweet text contains indicator keywords
                text = tweet.get("text", "").lower()
                has_indicator = any(kw in text for kw in self.INDICATOR_KEYWORDS)

                if not has_indicator:
                    continue

                # Extract URLs from entities
                entities = tweet.get("entities", {})
                tweet_urls = entities.get("urls", [])

                for url_entity in tweet_urls:
                    expanded_url = url_entity.get("expanded_url", "")

                    if not expanded_url:
                        continue

                    # Filter out excluded domains
                    if self._is_excluded(expanded_url):
                        continue

                    urls.append(expanded_url)

                # Also extract URLs from text using regex
                text_urls = self._extract_urls_from_text(tweet.get("text", ""))
                for url in text_urls:
                    if not self._is_excluded(url):
                        urls.append(url)

        except (httpx.HTTPError, KeyError, ValueError):
            pass

        return urls

    def _extract_urls_from_text(self, text: str) -> List[str]:
        """
        Extract URLs from tweet text using regex.

        Args:
            text: Tweet text

        Returns:
            List of URLs found
        """
        # URL pattern that handles defanged URLs
        # Handles hxxp://, [.], etc.
        url_pattern = r'(?:hxxps?|https?):\/\/[^\s<>"{}|\\^`\[\]]+'
        defanged_pattern = r'(?:hxxps?|https?):\/\/[^\s<>"{}|\\^`]+'

        urls = []

        # Find standard URLs
        for match in re.finditer(url_pattern, text, re.IGNORECASE):
            url = match.group(0)
            # Refang if needed
            url = self._refang_url(url)
            if url:
                urls.append(url)

        # Look for defanged URLs like example[.]com
        defanged_domain = r'\b[\w.-]+\[\.\][\w.-]+(?:/[^\s]*)?'
        for match in re.finditer(defanged_domain, text):
            domain = match.group(0)
            # Refang
            domain = domain.replace("[.]", ".")
            domain = domain.replace("[", "").replace("]", "")
            if "." in domain:
                url = f"https://{domain}"
                urls.append(url)

        return urls

    def _refang_url(self, url: str) -> Optional[str]:
        """
        Convert defanged URL back to normal format.

        Args:
            url: Potentially defanged URL

        Returns:
            Normal URL or None if invalid
        """
        url = url.replace("hxxp://", "http://")
        url = url.replace("hxxps://", "https://")
        url = url.replace("[.]", ".")
        url = url.replace("[:]", ":")
        url = url.replace("[", "").replace("]", "")

        # Validate
        try:
            parsed = urlparse(url)
            if parsed.scheme and parsed.netloc:
                return url
        except (ValueError, AttributeError):
            pass

        return None

    def _is_excluded(self, url: str) -> bool:
        """
        Check if URL should be excluded.

        Args:
            url: URL to check

        Returns:
            True if URL should be excluded
        """
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()

            # Remove port if present
            if ":" in domain:
                domain = domain.split(":")[0]

            # Check against exclusion list
            for excluded in self.EXCLUDED_DOMAINS:
                if domain == excluded or domain.endswith("." + excluded):
                    return True

        except (ValueError, AttributeError):
            return True

        return False

    def clear_seen_cache(self):
        """Clear the cache of seen tweet IDs"""
        self._seen_tweet_ids.clear()
