"""
Campaign Clustering Module

Groups related phishing domains into campaigns based on:
- Domain similarity
- Temporal proximity
- Target brand matching
- Infrastructure overlap
"""
import asyncio
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple
from difflib import SequenceMatcher

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.types import Classification, ClassificationRecord


@dataclass
class Campaign:
    """Represents a phishing campaign"""
    id: str
    name: str
    domains: Set[str] = field(default_factory=set)
    target_brand: Optional[str] = None
    first_seen: datetime = field(default_factory=datetime.utcnow)
    last_seen: datetime = field(default_factory=datetime.utcnow)
    confidence: float = 0.0
    url_count: int = 0
    infrastructure: Dict[str, Set[str]] = field(default_factory=dict)

    def __post_init__(self):
        if not self.infrastructure:
            self.infrastructure = {
                "ips": set(),
                "registrars": set(),
                "nameservers": set(),
            }

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "name": self.name,
            "domains": list(self.domains),
            "target_brand": self.target_brand,
            "first_seen": self.first_seen.isoformat(),
            "last_seen": self.last_seen.isoformat(),
            "confidence": self.confidence,
            "url_count": self.url_count,
            "infrastructure": {
                k: list(v) for k, v in self.infrastructure.items()
            },
        }


class CampaignClusterer:
    """
    Clusters phishing domains into campaigns based on various similarity metrics.
    """

    # Minimum similarity score to consider domains as related
    SIMILARITY_THRESHOLD = 0.6

    # Time window for temporal clustering (domains seen within this window may be related)
    TIME_WINDOW_HOURS = 24

    # Minimum number of domains to form a campaign
    MIN_CAMPAIGN_SIZE = 2

    # Brand patterns for campaign naming
    BRAND_PATTERNS = {
        "paypal": r"paypa[l1]|pypal",
        "amazon": r"amaz[o0]n|amazn",
        "microsoft": r"micr[o0]s[o0]ft|msft|m1crosoft",
        "google": r"g[o0][o0]gle|googl",
        "apple": r"app[l1]e|appl",
        "facebook": r"faceb[o0][o0]k|fb",
        "netflix": r"netf[l1]ix|netf1x",
        "chase": r"chase|jp\s*morgan",
        "wellsfargo": r"wel[l1]s?\s*farg[o0]",
        "coinbase": r"c[o0]inbase|coinbas",
        "binance": r"b[i1]nance|bnb",
    }

    def __init__(self):
        self.campaigns: Dict[str, Campaign] = {}
        self._domain_to_campaign: Dict[str, str] = {}
        self._lock = asyncio.Lock()

    async def cluster_domain(
        self,
        record: ClassificationRecord,
        ip: Optional[str] = None,
        registrar: Optional[str] = None,
    ) -> Optional[Campaign]:
        """
        Attempt to cluster a newly classified domain into an existing campaign
        or create a new campaign.

        Args:
            record: Classification record for the domain
            ip: IP address of the domain (if known)
            registrar: Domain registrar (if known)

        Returns:
            Campaign if domain was clustered, None if not
        """
        # Only cluster phishing domains
        if record.classification != Classification.PHISHING:
            return None

        domain = record.domain

        async with self._lock:
            # Check if already clustered
            if domain in self._domain_to_campaign:
                campaign_id = self._domain_to_campaign[domain]
                return self.campaigns.get(campaign_id)

            # Try to find matching campaign
            best_campaign = None
            best_score = 0.0

            for campaign in self.campaigns.values():
                score = await self._calculate_campaign_match(
                    domain=domain,
                    features=record.features,
                    campaign=campaign,
                    ip=ip,
                    registrar=registrar,
                )

                if score > best_score and score >= self.SIMILARITY_THRESHOLD:
                    best_score = score
                    best_campaign = campaign

            if best_campaign:
                # Add to existing campaign
                await self._add_to_campaign(best_campaign, domain, ip, registrar)
                return best_campaign
            else:
                # Check if we should create a new campaign
                # Look for other recent unclustered domains that might match
                new_campaign = await self._try_create_campaign(
                    domain=domain,
                    features=record.features,
                    ip=ip,
                    registrar=registrar,
                )
                return new_campaign

    async def _calculate_campaign_match(
        self,
        domain: str,
        features: Dict,
        campaign: Campaign,
        ip: Optional[str] = None,
        registrar: Optional[str] = None,
    ) -> float:
        """
        Calculate how well a domain matches an existing campaign.

        Args:
            domain: Domain to check
            features: Features from the domain
            campaign: Campaign to match against
            ip: IP address
            registrar: Domain registrar

        Returns:
            Match score between 0 and 1
        """
        scores = []
        weights = []

        # 1. Domain similarity with existing campaign domains
        max_domain_sim = 0.0
        for camp_domain in campaign.domains:
            sim = self._domain_similarity(domain, camp_domain)
            max_domain_sim = max(max_domain_sim, sim)

        if max_domain_sim > 0:
            scores.append(max_domain_sim)
            weights.append(0.4)

        # 2. Target brand match
        domain_brand = self._detect_brand(domain)
        if domain_brand and domain_brand == campaign.target_brand:
            scores.append(1.0)
            weights.append(0.3)
        elif domain_brand and campaign.target_brand:
            scores.append(0.0)
            weights.append(0.3)

        # 3. Infrastructure overlap
        if ip and ip in campaign.infrastructure.get("ips", set()):
            scores.append(1.0)
            weights.append(0.2)

        if registrar and registrar in campaign.infrastructure.get("registrars", set()):
            scores.append(0.8)
            weights.append(0.1)

        # 4. Temporal proximity
        time_diff = abs((datetime.utcnow() - campaign.last_seen).total_seconds())
        if time_diff <= self.TIME_WINDOW_HOURS * 3600:
            time_score = 1.0 - (time_diff / (self.TIME_WINDOW_HOURS * 3600))
            scores.append(time_score)
            weights.append(0.1)

        if not scores:
            return 0.0

        # Weighted average
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0

        return sum(s * w for s, w in zip(scores, weights)) / total_weight

    def _domain_similarity(self, domain1: str, domain2: str) -> float:
        """
        Calculate similarity between two domains.

        Uses a combination of:
        - Sequence matching
        - Common substring detection
        - TLD normalization
        """
        # Normalize domains
        d1 = self._normalize_domain(domain1)
        d2 = self._normalize_domain(domain2)

        # Sequence matcher ratio
        seq_ratio = SequenceMatcher(None, d1, d2).ratio()

        # Check for common patterns (numbers substituted for letters, etc.)
        d1_normalized = self._normalize_leetspeak(d1)
        d2_normalized = self._normalize_leetspeak(d2)

        if d1_normalized == d2_normalized and d1 != d2:
            return 0.9

        normalized_ratio = SequenceMatcher(None, d1_normalized, d2_normalized).ratio()

        return max(seq_ratio, normalized_ratio)

    def _normalize_domain(self, domain: str) -> str:
        """Remove TLD and common prefixes from domain"""
        domain = domain.lower()

        # Remove common prefixes
        for prefix in ["www.", "secure.", "login.", "account.", "verify."]:
            if domain.startswith(prefix):
                domain = domain[len(prefix):]

        # Remove TLD
        parts = domain.split(".")
        if len(parts) > 1:
            domain = ".".join(parts[:-1])

        return domain

    def _normalize_leetspeak(self, text: str) -> str:
        """Convert leetspeak to normal text"""
        replacements = {
            "0": "o",
            "1": "l",
            "3": "e",
            "4": "a",
            "5": "s",
            "7": "t",
            "8": "b",
            "@": "a",
            "$": "s",
        }

        result = text.lower()
        for old, new in replacements.items():
            result = result.replace(old, new)

        return result

    def _detect_brand(self, domain: str) -> Optional[str]:
        """Detect which brand a domain is targeting"""
        domain_lower = domain.lower()

        for brand, pattern in self.BRAND_PATTERNS.items():
            if re.search(pattern, domain_lower, re.IGNORECASE):
                return brand

        return None

    async def _add_to_campaign(
        self,
        campaign: Campaign,
        domain: str,
        ip: Optional[str] = None,
        registrar: Optional[str] = None,
    ):
        """Add a domain to an existing campaign"""
        campaign.domains.add(domain)
        campaign.url_count += 1
        campaign.last_seen = datetime.utcnow()

        # Update infrastructure
        if ip:
            campaign.infrastructure["ips"].add(ip)
        if registrar:
            campaign.infrastructure["registrars"].add(registrar)

        # Update confidence based on size
        campaign.confidence = min(0.95, 0.5 + 0.1 * len(campaign.domains))

        # Map domain to campaign
        self._domain_to_campaign[domain] = campaign.id

    async def _try_create_campaign(
        self,
        domain: str,
        features: Dict,
        ip: Optional[str] = None,
        registrar: Optional[str] = None,
    ) -> Optional[Campaign]:
        """
        Try to create a new campaign if this domain represents
        a new clustering opportunity.
        """
        # Detect brand
        brand = self._detect_brand(domain)

        # Create campaign
        campaign_id = str(uuid.uuid4())
        name = f"Campaign-{brand.upper() if brand else 'UNKNOWN'}-{campaign_id[:8]}"

        campaign = Campaign(
            id=campaign_id,
            name=name,
            domains={domain},
            target_brand=brand,
            first_seen=datetime.utcnow(),
            last_seen=datetime.utcnow(),
            confidence=0.5,
            url_count=1,
        )

        if ip:
            campaign.infrastructure["ips"].add(ip)
        if registrar:
            campaign.infrastructure["registrars"].add(registrar)

        # Store campaign
        self.campaigns[campaign_id] = campaign
        self._domain_to_campaign[domain] = campaign_id

        return campaign

    def get_campaign(self, campaign_id: str) -> Optional[Campaign]:
        """Get a campaign by ID"""
        return self.campaigns.get(campaign_id)

    def get_campaign_for_domain(self, domain: str) -> Optional[Campaign]:
        """Get the campaign a domain belongs to"""
        campaign_id = self._domain_to_campaign.get(domain)
        if campaign_id:
            return self.campaigns.get(campaign_id)
        return None

    def get_all_campaigns(self) -> List[Campaign]:
        """Get all campaigns"""
        return list(self.campaigns.values())

    def get_active_campaigns(self, hours: int = 24) -> List[Campaign]:
        """Get campaigns active within the specified time window"""
        cutoff = datetime.utcnow() - timedelta(hours=hours)
        return [
            c for c in self.campaigns.values()
            if c.last_seen >= cutoff
        ]

    def get_campaigns_by_brand(self, brand: str) -> List[Campaign]:
        """Get all campaigns targeting a specific brand"""
        return [
            c for c in self.campaigns.values()
            if c.target_brand and c.target_brand.lower() == brand.lower()
        ]

    def get_statistics(self) -> Dict:
        """Get clustering statistics"""
        total_campaigns = len(self.campaigns)
        total_domains = len(self._domain_to_campaign)

        brands = {}
        for campaign in self.campaigns.values():
            if campaign.target_brand:
                brands[campaign.target_brand] = brands.get(campaign.target_brand, 0) + 1

        active_campaigns = self.get_active_campaigns(hours=24)

        return {
            "total_campaigns": total_campaigns,
            "total_clustered_domains": total_domains,
            "active_campaigns_24h": len(active_campaigns),
            "campaigns_by_brand": brands,
            "avg_domains_per_campaign": total_domains / total_campaigns if total_campaigns else 0,
        }


# Global clusterer instance
campaign_clusterer = CampaignClusterer()
