"""
Paste Site Source

Monitors paste sites for URLs shared in pastes containing threat-related keywords.
Uses psbdmp.ws API for paste searching.
"""
import re
from datetime import datetime
from typing import List, Set, Optional
from urllib.parse import urlparse

import httpx

from .base import URLSource


class PasteSiteSource(URLSource):
    """
    Source that searches paste sites for phishing/malware URLs.

    Uses the psbdmp.ws API to search for pastes containing threat-related
    keywords and extracts URLs from those pastes.
    """

    PSBDMP_API = "https://psbdmp.ws/api/v3/search"

    # Keywords to search for in pastes
    KEYWORDS = [
        "phishing",
        "credential",
        "password",
        "login page",
        "fake site",
        "scam url",
        "malware url",
        "c2 server",
        "dropper",
        "stealer",
        "panel",
    ]

    # Domains to exclude (common legitimate sites that appear in pastes)
    EXCLUDED_DOMAINS = {
        "github.com",
        "google.com",
        "microsoft.com",
        "amazon.com",
        "stackoverflow.com",
        "pastebin.com",
        "paste.ee",
        "hastebin.com",
        "ghostbin.com",
        "dpaste.org",
        "ideone.com",
        "jsfiddle.net",
        "codepen.io",
        "replit.com",
        "virustotal.com",
        "urlscan.io",
        "any.run",
        "hybrid-analysis.com",
    }

    # File extensions that likely indicate malware rather than phishing
    MALWARE_EXTENSIONS = {
        ".exe", ".dll", ".scr", ".bat", ".cmd", ".ps1",
        ".vbs", ".js", ".jar", ".msi", ".hta",
    }

    def __init__(self):
        super().__init__(name="pastebin", reputation=0.70)
        self._seen_paste_ids: Set[str] = set()
        self._seen_urls: Set[str] = set()

    async def fetch_urls(self, limit: int = 100) -> List[str]:
        """
        Fetch URLs from paste sites by searching for threat-related keywords.

        Returns list of URLs found in pastes.
        """
        suspicious_urls: List[str] = []

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                for keyword in self.KEYWORDS:
                    if len(suspicious_urls) >= limit:
                        break

                    urls = await self._search_pastes(client, keyword)

                    for url in urls:
                        if len(suspicious_urls) >= limit:
                            break
                        if url not in suspicious_urls and url not in self._seen_urls:
                            self._seen_urls.add(url)
                            suspicious_urls.append(url)

        except httpx.TimeoutException:
            self.record_error()
        except httpx.HTTPError as e:
            self.record_error()
        except Exception as e:
            self.record_error()

        return suspicious_urls

    async def _search_pastes(self, client: httpx.AsyncClient, keyword: str) -> List[str]:
        """
        Search for pastes containing a keyword and extract URLs.

        Args:
            client: HTTP client
            keyword: Search keyword

        Returns:
            List of URLs from matching pastes
        """
        urls: List[str] = []

        try:
            # Search for pastes
            response = await client.get(
                self.PSBDMP_API,
                params={"q": keyword},
            )

            # API might return 404 for no results
            if response.status_code == 404:
                return []

            response.raise_for_status()
            data = response.json()

            pastes = data.get("data", [])

            for paste in pastes[:10]:  # Limit pastes per keyword
                paste_id = paste.get("id")

                if not paste_id or paste_id in self._seen_paste_ids:
                    continue

                self._seen_paste_ids.add(paste_id)

                # Get paste content
                paste_urls = await self._extract_urls_from_paste(client, paste_id)
                urls.extend(paste_urls)

        except (httpx.HTTPError, ValueError, KeyError):
            pass

        return urls

    async def _extract_urls_from_paste(
        self,
        client: httpx.AsyncClient,
        paste_id: str
    ) -> List[str]:
        """
        Fetch paste content and extract URLs.

        Args:
            client: HTTP client
            paste_id: Paste ID to fetch

        Returns:
            List of URLs from paste
        """
        urls: List[str] = []

        try:
            # Fetch paste content
            response = await client.get(
                f"https://psbdmp.ws/api/v3/dump/{paste_id}",
            )

            if response.status_code != 200:
                return []

            data = response.json()
            content = data.get("content", "")

            if not content:
                return []

            # Extract URLs from content
            extracted = self._extract_urls_from_text(content)

            for url in extracted:
                # Filter out excluded domains
                if self._is_excluded(url):
                    continue

                # Filter out obvious malware download URLs
                if self._is_malware_url(url):
                    continue

                urls.append(url)

        except (httpx.HTTPError, ValueError, KeyError):
            pass

        return urls

    def _extract_urls_from_text(self, text: str) -> List[str]:
        """
        Extract URLs from text, handling defanged formats.

        Args:
            text: Text to extract URLs from

        Returns:
            List of URLs found
        """
        urls: Set[str] = set()

        # Standard URL pattern
        url_pattern = r'(?:https?|hxxps?|ftp):\/\/[^\s<>"\'{}|\\^`\[\]]*'

        for match in re.finditer(url_pattern, text, re.IGNORECASE):
            url = match.group(0)
            # Clean up trailing punctuation
            url = url.rstrip('.,;:!?\'"])}>')
            # Refang
            url = self._refang_url(url)
            if url:
                urls.add(url)

        # Defanged domain pattern (example[.]com)
        defanged_pattern = r'\b([\w][\w.-]*)\[\.\]([\w.-]+)(?:\/[^\s]*)?'

        for match in re.finditer(defanged_pattern, text):
            domain = match.group(0)
            domain = domain.replace("[.]", ".")
            domain = domain.replace("[", "").replace("]", "")
            url = f"https://{domain}"
            urls.add(url)

        # IP address URLs (common in malware configs)
        ip_pattern = r'\b(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})(?::(\d+))?(\/[^\s]*)?'

        for match in re.finditer(ip_pattern, text):
            ip = match.group(1)
            port = match.group(2)
            path = match.group(3) or ""

            # Only include if it looks like a URL context
            if port:
                url = f"http://{ip}:{port}{path}"
            else:
                url = f"http://{ip}{path}"
            urls.add(url)

        return list(urls)

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
            if parsed.scheme in ("http", "https", "ftp") and parsed.netloc:
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

    def _is_malware_url(self, url: str) -> bool:
        """
        Check if URL appears to be a malware download rather than phishing.

        We're focused on phishing URLs, so we filter out obvious malware downloads.

        Args:
            url: URL to check

        Returns:
            True if URL appears to be malware download
        """
        url_lower = url.lower()

        for ext in self.MALWARE_EXTENSIONS:
            if url_lower.endswith(ext):
                return True

        return False

    def clear_seen_cache(self):
        """Clear the cache of seen pastes and URLs"""
        self._seen_paste_ids.clear()
        self._seen_urls.clear()
