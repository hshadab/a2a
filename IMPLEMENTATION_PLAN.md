# Implementation Plan: Complete the Threat Intelligence Network

This plan outlines how to make all placeholder/stub items fully functional.

---

## Overview

| Item | Effort | Priority | Dependencies |
|------|--------|----------|--------------|
| Certificate Transparency Source | 2-3 hours | HIGH | None |
| Twitter/X Source | 3-4 hours | MEDIUM | X API key |
| Paste Site Source | 2-3 hours | MEDIUM | None |
| Campaign Clustering | 4-6 hours | LOW | More data needed |
| Dynamic Reputation System | 3-4 hours | MEDIUM | None |
| Test Suite | 4-6 hours | HIGH | pytest |

---

## 1. Certificate Transparency Source

**Purpose**: Discover newly registered domains by monitoring CT logs (certificates issued in real-time).

**File**: `agents/scout/sources/cert_transparency.py`

```python
"""
Certificate Transparency Log Source

Monitors CT logs via crt.sh to discover newly issued certificates.
New domains often indicate phishing infrastructure being set up.
"""
import httpx
import re
from typing import List
from datetime import datetime, timedelta

from .base import URLSource


class CertTransparencySource(URLSource):
    """
    Fetch newly registered domains from Certificate Transparency logs.

    Uses crt.sh API to query recent certificates.
    High-value source for catching phishing sites before they're widely reported.
    """

    # crt.sh API endpoint
    API_URL = "https://crt.sh/"

    # Suspicious keywords in domain names
    SUSPICIOUS_KEYWORDS = [
        'login', 'signin', 'account', 'verify', 'secure', 'update',
        'banking', 'paypal', 'amazon', 'microsoft', 'apple', 'google',
        'facebook', 'netflix', 'support', 'helpdesk', 'password'
    ]

    # High-risk TLDs often used for phishing
    HIGH_RISK_TLDS = ['.xyz', '.top', '.tk', '.ml', '.ga', '.cf', '.gq', '.work', '.click']

    def __init__(self):
        super().__init__(name="certificate_transparency", reputation=0.65)
        self._seen_domains = set()
        self._last_check = datetime.utcnow() - timedelta(hours=1)

    async def fetch_urls(self, limit: int = 100) -> List[str]:
        """
        Fetch suspicious domains from recent CT log entries.

        Strategy:
        1. Query crt.sh for certificates issued in last hour
        2. Filter for suspicious domain patterns
        3. Convert to URLs for analysis
        """
        try:
            urls = []

            # Query for each suspicious keyword
            for keyword in self.SUSPICIOUS_KEYWORDS[:5]:  # Limit to avoid rate limits
                domain_urls = await self._search_keyword(keyword, limit // 5)
                urls.extend(domain_urls)

                if len(urls) >= limit:
                    break

            # Deduplicate and filter already-seen
            unique_urls = []
            for url in urls:
                domain = self._extract_domain(url)
                if domain and domain not in self._seen_domains:
                    self._seen_domains.add(domain)
                    unique_urls.append(url)

            self._last_check = datetime.utcnow()
            return unique_urls[:limit]

        except Exception as e:
            self.record_error()
            print(f"CT log fetch error: {e}")
            return []

    async def _search_keyword(self, keyword: str, limit: int) -> List[str]:
        """Search CT logs for domains containing keyword"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # crt.sh JSON API
                response = await client.get(
                    self.API_URL,
                    params={
                        'q': f'%.{keyword}%',
                        'output': 'json'
                    }
                )

                if response.status_code != 200:
                    return []

                certs = response.json()
                urls = []

                for cert in certs[:limit]:
                    domain = cert.get('common_name', '')
                    if domain and self._is_suspicious(domain):
                        # Convert domain to URL
                        url = f"https://{domain.lstrip('*.')}"
                        urls.append(url)

                return urls

        except Exception:
            return []

    def _is_suspicious(self, domain: str) -> bool:
        """Check if domain has suspicious characteristics"""
        domain_lower = domain.lower()

        # Check for high-risk TLDs
        for tld in self.HIGH_RISK_TLDS:
            if domain_lower.endswith(tld):
                return True

        # Check for brand names with typos/additions
        brands = ['paypal', 'amazon', 'microsoft', 'apple', 'google', 'facebook', 'netflix']
        for brand in brands:
            if brand in domain_lower and domain_lower != f"{brand}.com":
                return True

        return False

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        match = re.search(r'https?://([^/]+)', url)
        return match.group(1) if match else None
```

**Registration**: Add to `agents/scout/sources/__init__.py`:
```python
from .cert_transparency import CertTransparencySource
```

---

## 2. Twitter/X Source

**Purpose**: Monitor threat intelligence accounts on X for reported phishing URLs.

**File**: `agents/scout/sources/twitter.py`

**Requirements**: X API Bearer Token (set in environment)

```python
"""
Twitter/X Threat Intelligence Source

Monitors X for phishing URLs shared by threat intelligence accounts.
"""
import httpx
import re
import os
from typing import List

from .base import URLSource


class TwitterSource(URLSource):
    """
    Fetch phishing URLs from Twitter/X threat intelligence feeds.

    Monitors specific accounts and hashtags for malicious URL reports.
    """

    API_URL = "https://api.twitter.com/2"

    # Trusted threat intel accounts to monitor
    TRUSTED_ACCOUNTS = [
        'PhishTank',
        'urlaboratories',
        'malaboratories',
        'abuse_ch',
        'JPCERT',
        'MalwareTechBlog'
    ]

    # Search keywords
    SEARCH_KEYWORDS = [
        '#phishing',
        '#malware',
        '#infosec threat',
        'malicious URL',
        'phishing campaign'
    ]

    # URL pattern to extract
    URL_PATTERN = re.compile(r'https?://[^\s<>"\']+')

    def __init__(self):
        super().__init__(name="twitter", reputation=0.70)
        self._bearer_token = os.environ.get('TWITTER_BEARER_TOKEN')
        self._seen_tweet_ids = set()

        if not self._bearer_token:
            print("WARNING: TWITTER_BEARER_TOKEN not set - Twitter source disabled")

    async def fetch_urls(self, limit: int = 100) -> List[str]:
        """Fetch URLs from recent threat intel tweets"""
        if not self._bearer_token:
            return []

        try:
            urls = []

            # Search for phishing-related tweets
            for keyword in self.SEARCH_KEYWORDS[:3]:
                tweet_urls = await self._search_tweets(keyword, limit // 3)
                urls.extend(tweet_urls)

            # Deduplicate
            unique_urls = list(set(urls))
            return unique_urls[:limit]

        except Exception as e:
            self.record_error()
            print(f"Twitter fetch error: {e}")
            return []

    async def _search_tweets(self, query: str, limit: int) -> List[str]:
        """Search Twitter for tweets containing query"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.API_URL}/tweets/search/recent",
                    params={
                        'query': f'{query} has:links -is:retweet',
                        'max_results': min(limit, 100),
                        'tweet.fields': 'entities,author_id'
                    },
                    headers={
                        'Authorization': f'Bearer {self._bearer_token}'
                    }
                )

                if response.status_code != 200:
                    return []

                data = response.json()
                tweets = data.get('data', [])

                urls = []
                for tweet in tweets:
                    tweet_id = tweet.get('id')
                    if tweet_id in self._seen_tweet_ids:
                        continue

                    self._seen_tweet_ids.add(tweet_id)

                    # Extract URLs from entities
                    entities = tweet.get('entities', {})
                    for url_obj in entities.get('urls', []):
                        expanded_url = url_obj.get('expanded_url', '')
                        # Skip twitter.com and t.co links
                        if expanded_url and 'twitter.com' not in expanded_url and 't.co' not in expanded_url:
                            urls.append(expanded_url)

                return urls

        except Exception:
            return []
```

---

## 3. Paste Site Source

**Purpose**: Monitor paste sites for leaked credentials and phishing URLs.

**File**: `agents/scout/sources/pastebin.py`

```python
"""
Paste Site Scraping Source

Monitors paste sites for leaked URLs and credentials.
"""
import httpx
import re
from typing import List

from .base import URLSource


class PasteSiteSource(URLSource):
    """
    Scrape paste sites for malicious URLs.

    Monitors public paste sites where attackers often dump
    phishing URLs, credential harvests, and malware links.
    """

    # Paste sites with accessible APIs
    PASTE_SOURCES = [
        {
            'name': 'pastebin_scrape',
            'url': 'https://psbdmp.ws/api/v3/getbystring',
            'method': 'keyword_search'
        }
    ]

    # Keywords to search for
    KEYWORDS = [
        'phishing',
        'credentials',
        'login',
        'leaked',
        'password dump'
    ]

    URL_PATTERN = re.compile(r'https?://[^\s<>"\'\\]+')

    def __init__(self):
        super().__init__(name="paste_sites", reputation=0.45)  # Lower trust - user submitted
        self._seen_paste_ids = set()

    async def fetch_urls(self, limit: int = 100) -> List[str]:
        """Fetch URLs from recent pastes"""
        try:
            urls = []

            # Search each source
            for keyword in self.KEYWORDS[:3]:
                paste_urls = await self._search_pastes(keyword)
                urls.extend(paste_urls)

            # Deduplicate and limit
            unique_urls = list(set(urls))
            return unique_urls[:limit]

        except Exception as e:
            self.record_error()
            print(f"Paste site fetch error: {e}")
            return []

    async def _search_pastes(self, keyword: str) -> List[str]:
        """Search paste aggregator for keyword"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                # Use paste aggregator API
                response = await client.get(
                    'https://psbdmp.ws/api/v3/search',
                    params={'q': keyword}
                )

                if response.status_code != 200:
                    return []

                pastes = response.json()
                urls = []

                for paste in pastes.get('data', [])[:10]:
                    paste_id = paste.get('id')
                    if paste_id in self._seen_paste_ids:
                        continue

                    self._seen_paste_ids.add(paste_id)

                    # Fetch paste content
                    content = await self._get_paste_content(client, paste_id)
                    if content:
                        # Extract URLs from paste
                        found_urls = self.URL_PATTERN.findall(content)
                        # Filter to likely malicious
                        urls.extend([u for u in found_urls if self._looks_suspicious(u)])

                return urls

        except Exception:
            return []

    async def _get_paste_content(self, client: httpx.AsyncClient, paste_id: str) -> str:
        """Fetch individual paste content"""
        try:
            response = await client.get(f'https://psbdmp.ws/api/v3/get/{paste_id}')
            if response.status_code == 200:
                return response.json().get('content', '')
        except Exception:
            pass
        return ''

    def _looks_suspicious(self, url: str) -> bool:
        """Quick check if URL looks like phishing"""
        url_lower = url.lower()

        # Skip common safe domains
        safe_domains = ['github.com', 'google.com', 'microsoft.com', 'apple.com']
        for safe in safe_domains:
            if safe in url_lower:
                return False

        # Check for suspicious patterns
        suspicious_patterns = ['login', 'verify', 'account', 'secure', 'update', 'password']
        for pattern in suspicious_patterns:
            if pattern in url_lower:
                return True

        return False
```

---

## 4. Campaign Clustering

**Purpose**: Group related phishing domains into campaigns to identify threat actors.

**File**: `shared/clustering.py`

```python
"""
Campaign Clustering Module

Groups related phishing URLs into campaigns based on:
- Infrastructure (shared IPs, registrars)
- Domain patterns (typosquatting, similar TLDs)
- Temporal proximity (registered/discovered together)
"""
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional
from datetime import datetime
import hashlib
from collections import defaultdict

from .types import URLRecord


@dataclass
class Campaign:
    """A cluster of related phishing domains"""
    id: str
    name: str
    domains: List[str] = field(default_factory=list)
    ips: Set[str] = field(default_factory=set)
    registrars: Set[str] = field(default_factory=set)
    brands_targeted: Set[str] = field(default_factory=set)
    first_seen: datetime = None
    last_seen: datetime = None
    confidence: float = 0.0
    threat_actor: Optional[str] = None

    @property
    def size(self) -> int:
        return len(self.domains)

    def add_domain(self, domain: str, ip: str = None, registrar: str = None, brand: str = None):
        if domain not in self.domains:
            self.domains.append(domain)
        if ip:
            self.ips.add(ip)
        if registrar:
            self.registrars.add(registrar)
        if brand:
            self.brands_targeted.add(brand)


class CampaignClusterer:
    """
    Clusters phishing URLs into campaigns.

    Uses multiple signals:
    1. Shared infrastructure (IP, registrar, ASN)
    2. Domain similarity (typosquatting patterns)
    3. Temporal clustering (discovered together)
    """

    # Similarity thresholds
    DOMAIN_SIMILARITY_THRESHOLD = 0.7
    TEMPORAL_WINDOW_HOURS = 24
    MIN_CLUSTER_SIZE = 3

    def __init__(self):
        self.campaigns: Dict[str, Campaign] = {}
        self._domain_to_campaign: Dict[str, str] = {}

    def cluster(self, records: List[URLRecord]) -> List[Campaign]:
        """
        Cluster URL records into campaigns.

        Returns list of identified campaigns.
        """
        # Group by infrastructure
        ip_groups = self._group_by_ip(records)
        registrar_groups = self._group_by_registrar(records)

        # Group by brand targeting
        brand_groups = self._group_by_brand(records)

        # Merge overlapping groups
        campaigns = self._merge_groups(ip_groups, registrar_groups, brand_groups)

        # Filter small clusters
        campaigns = [c for c in campaigns if c.size >= self.MIN_CLUSTER_SIZE]

        # Calculate confidence scores
        for campaign in campaigns:
            campaign.confidence = self._calculate_confidence(campaign)

        return sorted(campaigns, key=lambda c: c.size, reverse=True)

    def _group_by_ip(self, records: List[URLRecord]) -> Dict[str, List[URLRecord]]:
        """Group records by IP address"""
        groups = defaultdict(list)
        for record in records:
            if hasattr(record, 'ip') and record.ip:
                groups[record.ip].append(record)
        return dict(groups)

    def _group_by_registrar(self, records: List[URLRecord]) -> Dict[str, List[URLRecord]]:
        """Group records by registrar"""
        groups = defaultdict(list)
        for record in records:
            if hasattr(record, 'registrar') and record.registrar:
                groups[record.registrar].append(record)
        return dict(groups)

    def _group_by_brand(self, records: List[URLRecord]) -> Dict[str, List[URLRecord]]:
        """Group records by targeted brand"""
        groups = defaultdict(list)
        for record in records:
            if hasattr(record, 'brand_targeted') and record.brand_targeted:
                groups[record.brand_targeted].append(record)
        return dict(groups)

    def _merge_groups(self, *group_dicts) -> List[Campaign]:
        """Merge overlapping groups into campaigns"""
        all_domains = set()
        domain_features = {}

        for groups in group_dicts:
            for key, records in groups.items():
                for record in records:
                    domain = record.url
                    all_domains.add(domain)
                    if domain not in domain_features:
                        domain_features[domain] = {'ips': set(), 'registrars': set(), 'brands': set()}
                    # Add features
                    if hasattr(record, 'ip') and record.ip:
                        domain_features[domain]['ips'].add(record.ip)
                    if hasattr(record, 'registrar') and record.registrar:
                        domain_features[domain]['registrars'].add(record.registrar)
                    if hasattr(record, 'brand_targeted') and record.brand_targeted:
                        domain_features[domain]['brands'].add(record.brand_targeted)

        # Simple clustering: group domains sharing any infrastructure
        visited = set()
        campaigns = []

        for domain in all_domains:
            if domain in visited:
                continue

            # BFS to find connected domains
            cluster = []
            queue = [domain]

            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                visited.add(current)
                cluster.append(current)

                # Find domains with shared features
                current_features = domain_features.get(current, {})
                for other_domain in all_domains:
                    if other_domain in visited:
                        continue
                    other_features = domain_features.get(other_domain, {})

                    # Check for shared infrastructure
                    if (current_features['ips'] & other_features['ips'] or
                        current_features['registrars'] & other_features['registrars']):
                        queue.append(other_domain)

            if cluster:
                campaign_id = hashlib.md5(''.join(sorted(cluster)).encode()).hexdigest()[:12]
                campaign = Campaign(
                    id=campaign_id,
                    name=f"Campaign-{campaign_id[:8]}",
                    domains=cluster,
                    first_seen=datetime.utcnow()
                )

                # Add metadata
                for d in cluster:
                    features = domain_features.get(d, {})
                    campaign.ips.update(features.get('ips', []))
                    campaign.registrars.update(features.get('registrars', []))
                    campaign.brands_targeted.update(features.get('brands', []))

                campaigns.append(campaign)

        return campaigns

    def _calculate_confidence(self, campaign: Campaign) -> float:
        """Calculate confidence score for campaign"""
        score = 0.0

        # More domains = higher confidence
        if campaign.size >= 10:
            score += 0.3
        elif campaign.size >= 5:
            score += 0.2
        else:
            score += 0.1

        # Shared infrastructure increases confidence
        if len(campaign.ips) == 1:  # All on same IP
            score += 0.3
        if len(campaign.registrars) == 1:  # Same registrar
            score += 0.2

        # Targeting same brand
        if len(campaign.brands_targeted) == 1:
            score += 0.2

        return min(score, 1.0)
```

---

## 5. Dynamic Reputation System

**Purpose**: Track and update source reputation based on accuracy over time.

**File**: `shared/reputation.py`

```python
"""
Dynamic Reputation System

Tracks source accuracy and adjusts trust scores over time.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional
from datetime import datetime, timedelta
import math


@dataclass
class SourceMetrics:
    """Metrics for a single source"""
    source_name: str
    initial_reputation: float
    current_reputation: float = None

    # Accuracy tracking
    total_urls: int = 0
    confirmed_correct: int = 0
    confirmed_wrong: int = 0

    # Detection metrics
    phishing_detected: int = 0
    safe_detected: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    # Timing
    last_update: datetime = None

    def __post_init__(self):
        if self.current_reputation is None:
            self.current_reputation = self.initial_reputation
        if self.last_update is None:
            self.last_update = datetime.utcnow()

    @property
    def accuracy(self) -> float:
        """Calculate overall accuracy"""
        total = self.confirmed_correct + self.confirmed_wrong
        if total == 0:
            return self.initial_reputation
        return self.confirmed_correct / total

    @property
    def false_positive_rate(self) -> float:
        """Calculate false positive rate"""
        total_safe = self.safe_detected + self.false_negatives
        if total_safe == 0:
            return 0.0
        return self.false_positives / total_safe


class ReputationManager:
    """
    Manages dynamic reputation scores for all sources.

    Features:
    - Tracks accuracy over time
    - Applies reputation decay for stale sources
    - Provides confidence intervals
    """

    # Decay configuration
    DECAY_HALF_LIFE_DAYS = 30  # Reputation decays towards initial over 30 days
    MIN_SAMPLES_FOR_UPDATE = 10  # Need at least 10 samples to adjust reputation

    # Bounds
    MIN_REPUTATION = 0.1
    MAX_REPUTATION = 0.99

    def __init__(self):
        self.sources: Dict[str, SourceMetrics] = {}

    def register_source(self, name: str, initial_reputation: float):
        """Register a new source"""
        if name not in self.sources:
            self.sources[name] = SourceMetrics(
                source_name=name,
                initial_reputation=initial_reputation
            )

    def record_classification(
        self,
        source_name: str,
        url: str,
        predicted: str,
        actual: Optional[str] = None
    ):
        """
        Record a classification result.

        Args:
            source_name: Source that provided the URL
            url: The classified URL
            predicted: Our classification (PHISHING, SAFE, SUSPICIOUS)
            actual: Ground truth if known (for feedback)
        """
        if source_name not in self.sources:
            return

        metrics = self.sources[source_name]
        metrics.total_urls += 1

        if predicted == "PHISHING":
            metrics.phishing_detected += 1
        elif predicted == "SAFE":
            metrics.safe_detected += 1

        # If we have ground truth, update accuracy
        if actual:
            if predicted == actual:
                metrics.confirmed_correct += 1
            else:
                metrics.confirmed_wrong += 1
                if predicted == "PHISHING" and actual == "SAFE":
                    metrics.false_positives += 1
                elif predicted == "SAFE" and actual == "PHISHING":
                    metrics.false_negatives += 1

        # Update reputation if we have enough samples
        if metrics.total_urls >= self.MIN_SAMPLES_FOR_UPDATE:
            self._update_reputation(metrics)

    def _update_reputation(self, metrics: SourceMetrics):
        """Update reputation based on accuracy"""
        if metrics.confirmed_correct + metrics.confirmed_wrong < self.MIN_SAMPLES_FOR_UPDATE:
            return

        # Calculate new reputation based on accuracy
        accuracy = metrics.accuracy

        # Blend with initial reputation (more samples = trust accuracy more)
        sample_weight = min(1.0, metrics.total_urls / 100)
        new_rep = (sample_weight * accuracy) + ((1 - sample_weight) * metrics.initial_reputation)

        # Apply decay towards initial reputation
        days_since_update = (datetime.utcnow() - metrics.last_update).days
        decay_factor = math.exp(-days_since_update * math.log(2) / self.DECAY_HALF_LIFE_DAYS)

        new_rep = (decay_factor * new_rep) + ((1 - decay_factor) * metrics.initial_reputation)

        # Clamp to bounds
        metrics.current_reputation = max(self.MIN_REPUTATION, min(self.MAX_REPUTATION, new_rep))
        metrics.last_update = datetime.utcnow()

    def get_reputation(self, source_name: str) -> float:
        """Get current reputation for a source"""
        if source_name not in self.sources:
            return 0.5  # Default for unknown sources
        return self.sources[source_name].current_reputation

    def get_metrics(self, source_name: str) -> Optional[SourceMetrics]:
        """Get full metrics for a source"""
        return self.sources.get(source_name)

    def get_all_metrics(self) -> Dict[str, SourceMetrics]:
        """Get metrics for all sources"""
        return self.sources.copy()


# Global instance
reputation_manager = ReputationManager()
```

---

## 6. Test Suite

**Purpose**: Comprehensive testing for all components.

### Setup

**Add to `requirements.txt`**:
```
pytest>=7.0.0
pytest-asyncio>=0.21.0
pytest-cov>=4.0.0
respx>=0.20.0
```

**Create `pytest.ini`**:
```ini
[pytest]
asyncio_mode = auto
testpaths = tests
python_files = test_*.py
python_functions = test_*
addopts = -v --cov=shared --cov=agents --cov-report=html
```

### Test Structure

```
tests/
├── __init__.py
├── conftest.py
├── test_sources/
│   ├── __init__.py
│   ├── test_openphish.py
│   ├── test_phishtank.py
│   ├── test_synthetic.py
│   └── test_cert_transparency.py
├── test_features/
│   ├── __init__.py
│   └── test_extraction.py
├── test_prover/
│   ├── __init__.py
│   └── test_prover.py
├── test_clustering/
│   ├── __init__.py
│   └── test_campaigns.py
└── test_reputation/
    ├── __init__.py
    └── test_reputation.py
```

**File**: `tests/conftest.py`

```python
"""
Pytest fixtures for threat-intel-network tests
"""
import pytest
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_urls():
    """Sample URLs for testing"""
    return [
        "https://legitimate-bank.com/login",
        "https://paypa1-secure.xyz/verify",
        "https://amaz0n-update.tk/account",
        "https://google.com",
        "https://microsoft-support-help.click/password"
    ]


@pytest.fixture
def sample_phishing_features():
    """Sample feature vectors for phishing URLs"""
    return [
        0.8,  # url_length_norm
        0.7,  # domain_length_norm
        0.5,  # path_depth
        0.3,  # has_ip
        0.9,  # suspicious_tld
        0.6,  # entropy
        # ... more features
    ]
```

**File**: `tests/test_sources/test_openphish.py`

```python
"""
Tests for OpenPhish source
"""
import pytest
import respx
from httpx import Response

from agents.scout.sources.openphish import OpenPhishSource


@pytest.fixture
def openphish_source():
    return OpenPhishSource()


@pytest.fixture
def mock_feed():
    return """https://phishing1.example.com/login
https://phishing2.example.net/verify
https://phishing3.example.org/account
"""


@respx.mock
@pytest.mark.asyncio
async def test_fetch_urls_success(openphish_source, mock_feed):
    """Test successful URL fetch"""
    respx.get("https://openphish.com/feed.txt").mock(
        return_value=Response(200, text=mock_feed)
    )

    urls = await openphish_source.fetch_urls(limit=10)

    assert len(urls) == 3
    assert all(url.startswith('https://') for url in urls)


@respx.mock
@pytest.mark.asyncio
async def test_fetch_urls_filters_duplicates(openphish_source, mock_feed):
    """Test that duplicate URLs are filtered"""
    respx.get("https://openphish.com/feed.txt").mock(
        return_value=Response(200, text=mock_feed)
    )

    # First fetch
    urls1 = await openphish_source.fetch_urls(limit=10)
    # Second fetch should return empty (all seen)
    urls2 = await openphish_source.fetch_urls(limit=10)

    assert len(urls1) == 3
    assert len(urls2) == 0


@respx.mock
@pytest.mark.asyncio
async def test_fetch_urls_handles_error(openphish_source):
    """Test error handling"""
    respx.get("https://openphish.com/feed.txt").mock(
        return_value=Response(500)
    )

    urls = await openphish_source.fetch_urls(limit=10)

    assert urls == []
    assert openphish_source.errors == 1


def test_source_metadata(openphish_source):
    """Test source metadata"""
    assert openphish_source.name == "openphish"
    assert openphish_source.reputation == 0.90
```

**File**: `tests/test_features/test_extraction.py`

```python
"""
Tests for URL feature extraction
"""
import pytest

from agents.analyst.features import extract_features


def test_extract_features_phishing_url():
    """Test feature extraction for obvious phishing URL"""
    url = "https://paypa1-secure-login.xyz/verify/account"
    features = extract_features(url)

    assert features['suspicious_tld'] == 1.0  # .xyz is high risk
    assert features['has_brand_name'] == 1.0  # paypal-like
    assert features['entropy'] > 0.5  # Mixed characters


def test_extract_features_safe_url():
    """Test feature extraction for safe URL"""
    url = "https://www.google.com/search"
    features = extract_features(url)

    assert features['suspicious_tld'] == 0.0
    assert features['has_ip'] == 0.0


def test_extract_features_ip_url():
    """Test feature extraction for IP-based URL"""
    url = "http://192.168.1.1/admin/login"
    features = extract_features(url)

    assert features['has_ip'] == 1.0
    assert features['uses_https'] == 0.0
```

---

## Implementation Order

### Phase 1: Foundation (Day 1-2)
1. Set up pytest infrastructure
2. Write tests for existing sources
3. Implement dynamic reputation system

### Phase 2: New Sources (Day 3-5)
4. Certificate Transparency source + tests
5. Twitter/X source + tests
6. Paste site source + tests

### Phase 3: Intelligence (Day 6-8)
7. Campaign clustering module + tests
8. Integrate clustering into Scout agent
9. Add clustering visualization to UI

### Phase 4: Polish (Day 9-10)
10. Integration tests for full pipeline
11. Documentation updates
12. Deploy to Render

---

## Environment Variables Needed

```bash
# For Twitter/X source
TWITTER_BEARER_TOKEN=your_bearer_token

# Existing
DATABASE_URL=postgresql://...
PRIVATE_KEY=0x...
```

---

## Commands

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/test_sources/test_openphish.py

# Run only async tests
pytest -m asyncio
```
