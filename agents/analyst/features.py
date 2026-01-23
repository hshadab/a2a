"""
URL Feature Extraction

Extracts features from URLs for classification.
These features are used as input to the zkML classifier.
"""
import re
import math
import string
from urllib.parse import urlparse, parse_qs
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


# ============ Constants ============

# TLDs commonly associated with phishing
HIGH_RISK_TLDS = {
    '.xyz', '.top', '.tk', '.ml', '.ga', '.cf', '.gq', '.pw',
    '.cc', '.ws', '.info', '.biz', '.club', '.online', '.site',
    '.website', '.space', '.tech', '.click', '.link', '.work'
}

MEDIUM_RISK_TLDS = {
    '.co', '.io', '.me', '.tv', '.us', '.in', '.ru', '.cn'
}

# Brand names commonly targeted
BRAND_NAMES = [
    'paypal', 'amazon', 'apple', 'microsoft', 'google', 'netflix',
    'facebook', 'instagram', 'twitter', 'linkedin', 'dropbox', 'adobe',
    'chase', 'wellsfargo', 'bankofamerica', 'citibank', 'usbank',
    'coinbase', 'binance', 'metamask', 'phantom', 'opensea',
    'ebay', 'walmart', 'target', 'bestbuy', 'costco',
    'spotify', 'disney', 'hbo', 'hulu', 'steam'
]

# Suspicious keywords in paths
SUSPICIOUS_KEYWORDS = [
    'login', 'signin', 'sign-in', 'account', 'verify', 'secure',
    'update', 'confirm', 'password', 'credential', 'auth', 'oauth',
    'validate', 'unlock', 'suspend', 'billing', 'payment', 'invoice'
]


@dataclass
class URLFeatures:
    """Features extracted from a URL"""
    # Basic URL structure
    url_length: int
    domain_length: int
    path_length: int
    query_length: int

    # Domain characteristics
    subdomain_count: int
    has_ip_address: bool
    has_port: bool
    uses_https: bool

    # Character analysis
    digit_count: int
    special_char_count: int
    digit_ratio: float
    entropy: float

    # TLD risk
    tld: str
    tld_risk_score: float

    # Typosquatting detection
    typosquat_score: float
    brand_match: Optional[str]
    levenshtein_distance: Optional[int]

    # Path analysis
    path_depth: int
    query_param_count: int
    has_suspicious_path: bool
    suspicious_keyword_count: int

    # Raw domain for reference
    domain: str

    def to_vector(self) -> List[float]:
        """Convert features to numeric vector for model input"""
        return [
            self.url_length / 200,  # Normalize
            self.domain_length / 50,
            self.path_length / 100,
            self.query_length / 100,
            self.subdomain_count / 5,
            1.0 if self.has_ip_address else 0.0,
            1.0 if self.has_port else 0.0,
            1.0 if self.uses_https else 0.0,
            self.digit_count / 20,
            self.special_char_count / 20,
            self.digit_ratio,
            self.entropy / 5,
            self.tld_risk_score,
            self.typosquat_score,
            1.0 if self.brand_match else 0.0,
            (self.levenshtein_distance or 10) / 10,
            self.path_depth / 5,
            self.query_param_count / 10,
            1.0 if self.has_suspicious_path else 0.0,
            self.suspicious_keyword_count / 5,
            # Padding to get to expected size
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        ]

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'url_length': self.url_length,
            'domain_length': self.domain_length,
            'path_length': self.path_length,
            'query_length': self.query_length,
            'subdomain_count': self.subdomain_count,
            'has_ip_address': self.has_ip_address,
            'has_port': self.has_port,
            'uses_https': self.uses_https,
            'digit_count': self.digit_count,
            'special_char_count': self.special_char_count,
            'digit_ratio': self.digit_ratio,
            'entropy': self.entropy,
            'tld': self.tld,
            'tld_risk_score': self.tld_risk_score,
            'typosquat_score': self.typosquat_score,
            'brand_match': self.brand_match,
            'levenshtein_distance': self.levenshtein_distance,
            'path_depth': self.path_depth,
            'query_param_count': self.query_param_count,
            'has_suspicious_path': self.has_suspicious_path,
            'suspicious_keyword_count': self.suspicious_keyword_count,
            'domain': self.domain
        }


def extract_features(url: str) -> URLFeatures:
    """
    Extract all features from a URL.

    Args:
        url: The URL to analyze

    Returns:
        URLFeatures object with all extracted features
    """
    try:
        parsed = urlparse(url)
    except (ValueError, AttributeError):
        parsed = urlparse(f"http://{url}")

    domain = parsed.netloc.lower()
    path = parsed.path
    query = parsed.query

    # Remove port from domain if present
    if ':' in domain:
        domain, _ = domain.rsplit(':', 1)
        has_port = True
    else:
        has_port = False

    # Extract TLD
    tld = get_tld(domain)

    # Check for IP address
    has_ip = is_ip_address(domain)

    # Count subdomains
    subdomain_count = domain.count('.') if not has_ip else 0

    # Character analysis
    digit_count = sum(c.isdigit() for c in url)
    special_chars = set(url) - set(string.ascii_letters + string.digits + ':/.-_?=&')
    special_char_count = len(special_chars)

    total_chars = len(url)
    digit_ratio = digit_count / total_chars if total_chars > 0 else 0

    # Entropy
    entropy = calculate_entropy(domain)

    # TLD risk score
    tld_risk = get_tld_risk_score(tld)

    # Typosquatting detection
    typo_score, brand_match, lev_distance = detect_typosquatting(domain)

    # Path analysis
    path_depth = path.count('/') if path else 0
    query_params = parse_qs(query)
    query_param_count = len(query_params)

    # Suspicious keywords
    path_lower = path.lower() if path else ""
    suspicious_count = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in path_lower)
    has_suspicious_path = suspicious_count > 0

    return URLFeatures(
        url_length=len(url),
        domain_length=len(domain),
        path_length=len(path) if path else 0,
        query_length=len(query) if query else 0,
        subdomain_count=subdomain_count,
        has_ip_address=has_ip,
        has_port=has_port,
        uses_https=parsed.scheme == 'https',
        digit_count=digit_count,
        special_char_count=special_char_count,
        digit_ratio=digit_ratio,
        entropy=entropy,
        tld=tld,
        tld_risk_score=tld_risk,
        typosquat_score=typo_score,
        brand_match=brand_match,
        levenshtein_distance=lev_distance,
        path_depth=path_depth,
        query_param_count=query_param_count,
        has_suspicious_path=has_suspicious_path,
        suspicious_keyword_count=suspicious_count,
        domain=domain
    )


def get_tld(domain: str) -> str:
    """Extract TLD from domain"""
    parts = domain.split('.')
    if len(parts) >= 2:
        return '.' + parts[-1]
    return ''


def is_ip_address(domain: str) -> bool:
    """Check if domain is an IP address"""
    # IPv4
    ipv4_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
    if re.match(ipv4_pattern, domain):
        return True

    # IPv6 (simplified check)
    if ':' in domain and all(c in '0123456789abcdefABCDEF:' for c in domain):
        return True

    return False


def get_tld_risk_score(tld: str) -> float:
    """Get risk score for a TLD (0-1, higher = more risky)"""
    tld_lower = tld.lower()

    if tld_lower in HIGH_RISK_TLDS:
        return 0.9
    elif tld_lower in MEDIUM_RISK_TLDS:
        return 0.5
    elif tld_lower in {'.com', '.org', '.net', '.edu', '.gov'}:
        return 0.1
    else:
        return 0.3


def calculate_entropy(text: str) -> float:
    """Calculate Shannon entropy of text"""
    if not text:
        return 0.0

    # Count character frequencies
    freq = {}
    for char in text:
        freq[char] = freq.get(char, 0) + 1

    # Calculate entropy
    length = len(text)
    entropy = 0.0
    for count in freq.values():
        prob = count / length
        if prob > 0:
            entropy -= prob * math.log2(prob)

    return entropy


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings"""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def detect_typosquatting(domain: str) -> Tuple[float, Optional[str], Optional[int]]:
    """
    Detect if domain is typosquatting a known brand.

    Returns:
        (typosquat_score, matched_brand, levenshtein_distance)
    """
    # Remove TLD
    domain_parts = domain.split('.')
    if len(domain_parts) > 1:
        main_domain = domain_parts[-2]  # Get domain without TLD
    else:
        main_domain = domain

    # Clean domain (remove common prefixes/suffixes)
    clean_domain = main_domain.lower()
    for prefix in ['secure-', 'login-', 'account-', 'verify-', 'www.']:
        if clean_domain.startswith(prefix):
            clean_domain = clean_domain[len(prefix):]
    for suffix in ['-secure', '-login', '-verify', '-account', '-support']:
        if clean_domain.endswith(suffix):
            clean_domain = clean_domain[:-len(suffix)]

    best_score = 0.0
    best_brand = None
    best_distance = None

    for brand in BRAND_NAMES:
        # Exact match (but different domain, so it's impersonation)
        if clean_domain == brand:
            # Check if it's the real domain
            real_domains = {f"{brand}.com", f"{brand}.org", f"{brand}.net"}
            if domain not in real_domains:
                return 1.0, brand, 0

        # Calculate similarity
        distance = levenshtein_distance(clean_domain, brand)
        max_len = max(len(clean_domain), len(brand))

        if max_len > 0:
            similarity = 1 - (distance / max_len)

            # High similarity (typosquat) but not exact
            if similarity > 0.7 and distance > 0 and distance <= 3:
                if similarity > best_score:
                    best_score = similarity
                    best_brand = brand
                    best_distance = distance

        # Check for brand name contained in domain
        if brand in clean_domain and clean_domain != brand:
            score = 0.8 * (len(brand) / len(clean_domain))
            if score > best_score:
                best_score = score
                best_brand = brand
                best_distance = 0

    return best_score, best_brand, best_distance


def extract_batch_features(urls: List[str]) -> List[URLFeatures]:
    """Extract features for a batch of URLs"""
    return [extract_features(url) for url in urls]
