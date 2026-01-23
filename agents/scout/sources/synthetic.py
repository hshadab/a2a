"""
Synthetic URL Source

Generates synthetic URLs for demo and testing purposes.
Produces a mix of obviously phishing and legitimate-looking URLs.
"""
import random
import string
import hashlib
from typing import List
from datetime import datetime

from .base import URLSource


# Brand names commonly targeted by phishing
BRANDS = [
    "paypal", "amazon", "apple", "microsoft", "google", "netflix",
    "facebook", "instagram", "twitter", "linkedin", "dropbox", "adobe",
    "chase", "wellsfargo", "bankofamerica", "citibank", "usbank",
    "coinbase", "binance", "metamask", "phantom", "opensea"
]

# Typosquat variations
TYPO_PATTERNS = [
    lambda s: s.replace('a', '4'),
    lambda s: s.replace('e', '3'),
    lambda s: s.replace('i', '1'),
    lambda s: s.replace('o', '0'),
    lambda s: s.replace('l', '1'),
    lambda s: s + s[-1],  # Double last letter
    lambda s: s[:-1],     # Remove last letter
    lambda s: s[0] + s[0] + s[1:],  # Double first letter
    lambda s: s.replace('ll', 'l'),
    lambda s: s.replace('ss', 's'),
]

# Suspicious TLDs
SUSPICIOUS_TLDS = [".xyz", ".top", ".tk", ".ml", ".ga", ".cf", ".gq", ".pw", ".cc", ".ws"]

# Legitimate TLDs
LEGIT_TLDS = [".com", ".org", ".net", ".io", ".co"]

# Phishing path patterns
PHISHING_PATHS = [
    "/login", "/signin", "/account/verify", "/secure/login",
    "/update-payment", "/verify-identity", "/unlock-account",
    "/password-reset", "/confirm-details", "/billing/update",
    "/security-check", "/validate", "/auth", "/oauth/authorize"
]

# Legitimate paths
LEGIT_PATHS = [
    "/", "/about", "/contact", "/products", "/services",
    "/blog", "/news", "/help", "/support", "/pricing"
]


class SyntheticSource(URLSource):
    """
    Generates synthetic URLs for demo and testing.

    Produces a configurable mix of:
    - Obviously phishing URLs (typosquats, suspicious TLDs)
    - Legitimate-looking URLs
    - Ambiguous URLs

    Useful for:
    - Demo without relying on external APIs
    - Testing classifier accuracy
    - Generating training data
    """

    def __init__(self, phishing_ratio: float = 0.3, seed: int = None):
        """
        Args:
            phishing_ratio: Proportion of URLs that should be phishing (0-1)
            seed: Random seed for reproducibility
        """
        super().__init__(name="synthetic", reputation=0.5)
        self.phishing_ratio = phishing_ratio
        self.rng = random.Random(seed)
        self._generated_count = 0

    def _generate_phishing_url(self) -> str:
        """Generate a phishing-looking URL"""
        brand = self.rng.choice(BRANDS)

        # Apply typosquat
        typo_func = self.rng.choice(TYPO_PATTERNS)
        typo_brand = typo_func(brand)

        # Add suspicious elements
        elements = []

        # Sometimes add "secure", "login", etc. prefix
        if self.rng.random() < 0.4:
            prefix = self.rng.choice(["secure-", "login-", "account-", "verify-", ""])
            elements.append(prefix)

        elements.append(typo_brand)

        # Sometimes add suffix
        if self.rng.random() < 0.5:
            suffix = self.rng.choice([
                "-secure", "-login", "-verify", "-account",
                "-support", "-help", "-update", ""
            ])
            elements.append(suffix)

        domain = "".join(elements)

        # Choose TLD (mostly suspicious)
        if self.rng.random() < 0.7:
            tld = self.rng.choice(SUSPICIOUS_TLDS)
        else:
            tld = self.rng.choice(LEGIT_TLDS)

        # Add path
        path = self.rng.choice(PHISHING_PATHS)

        # Sometimes add query params
        query = ""
        if self.rng.random() < 0.3:
            query = f"?id={self._random_string(8)}&token={self._random_string(16)}"

        return f"http://{domain}{tld}{path}{query}"

    def _generate_legit_url(self) -> str:
        """Generate a legitimate-looking URL"""
        # Use real brand name (correctly spelled)
        brand = self.rng.choice(BRANDS)

        # Legitimate TLD
        tld = self.rng.choice(LEGIT_TLDS)

        # Normal path
        path = self.rng.choice(LEGIT_PATHS)

        # HTTPS
        return f"https://{brand}{tld}{path}"

    def _generate_random_url(self) -> str:
        """Generate a random URL (could be anything)"""
        # Random domain
        domain_length = self.rng.randint(5, 15)
        domain = self._random_string(domain_length).lower()

        # Random TLD
        tld = self.rng.choice(SUSPICIOUS_TLDS + LEGIT_TLDS)

        # Random path
        path_segments = self.rng.randint(0, 3)
        path = "/" + "/".join(
            self._random_string(self.rng.randint(3, 10)).lower()
            for _ in range(path_segments)
        )

        protocol = self.rng.choice(["http", "https"])

        return f"{protocol}://{domain}{tld}{path}"

    def _random_string(self, length: int) -> str:
        """Generate random alphanumeric string"""
        chars = string.ascii_lowercase + string.digits
        return ''.join(self.rng.choice(chars) for _ in range(length))

    async def fetch_urls(self, limit: int = 100) -> List[str]:
        """
        Generate synthetic URLs.

        Distribution:
        - phishing_ratio% obvious phishing
        - (1-phishing_ratio)/2% legitimate
        - (1-phishing_ratio)/2% random/ambiguous
        """
        urls = []

        for _ in range(limit):
            r = self.rng.random()

            if r < self.phishing_ratio:
                url = self._generate_phishing_url()
            elif r < self.phishing_ratio + (1 - self.phishing_ratio) / 2:
                url = self._generate_legit_url()
            else:
                url = self._generate_random_url()

            # Make each URL unique by adding a timestamp-based hash
            self._generated_count += 1
            unique_suffix = hashlib.md5(
                f"{self._generated_count}{datetime.utcnow().isoformat()}".encode()
            ).hexdigest()[:6]

            # Add uniqueness to path
            if '?' in url:
                url = url.replace('?', f'-{unique_suffix}?')
            else:
                url = url.rstrip('/') + f'-{unique_suffix}'

            urls.append(url)

        return urls


class AlexaTopSource(URLSource):
    """
    Provides known-safe URLs from Alexa Top Sites.

    Used for generating legitimate training data.
    """

    # Hardcoded sample of top domains (would use API in production)
    TOP_DOMAINS = [
        "google.com", "youtube.com", "facebook.com", "amazon.com",
        "wikipedia.org", "twitter.com", "instagram.com", "linkedin.com",
        "reddit.com", "netflix.com", "microsoft.com", "apple.com",
        "github.com", "stackoverflow.com", "medium.com", "nytimes.com",
        "bbc.com", "cnn.com", "walmart.com", "ebay.com"
    ]

    def __init__(self):
        super().__init__(name="alexa_top", reputation=0.99)

    async def fetch_urls(self, limit: int = 100) -> List[str]:
        """Return known-safe URLs"""
        urls = []
        for domain in self.TOP_DOMAINS[:limit]:
            urls.append(f"https://{domain}/")
        return urls
