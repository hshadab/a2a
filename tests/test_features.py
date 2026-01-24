"""
Tests for URL feature extraction
"""
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.analyst.features import (
    extract_features,
    extract_batch_features,
    get_tld,
    is_ip_address,
    get_tld_risk_score,
    calculate_entropy,
    levenshtein_distance,
    detect_typosquatting,
    URLFeatures,
)


class TestFeatureExtraction:
    """Tests for URL feature extraction"""

    def test_extract_features_basic(self):
        """Test basic feature extraction"""
        url = "https://example.com/path/to/page"
        features = extract_features(url)

        assert isinstance(features, URLFeatures)
        assert features.domain == "example.com"
        assert features.uses_https is True
        assert features.url_length == len(url)

    def test_extract_features_phishing_url(self):
        """Test feature extraction for phishing-like URL"""
        url = "https://secure-paypal-login.xyz/verify/account"
        features = extract_features(url)

        assert features.tld == ".xyz"
        assert features.tld_risk_score == 0.9  # HIGH_RISK_TLD
        assert features.has_suspicious_path is True
        assert features.suspicious_keyword_count > 0

    def test_extract_features_ip_address(self):
        """Test feature extraction for IP address URL"""
        url = "http://192.168.1.1/admin"
        features = extract_features(url)

        assert features.has_ip_address is True
        assert features.subdomain_count == 0

    def test_extract_features_with_port(self):
        """Test feature extraction for URL with port"""
        url = "http://example.com:8080/api"
        features = extract_features(url)

        assert features.has_port is True

    def test_extract_features_typosquatting(self):
        """Test typosquatting detection"""
        url = "https://paypa1.com/login"
        features = extract_features(url)

        assert features.typosquat_score > 0
        assert features.brand_match == "paypal"

    def test_to_vector(self):
        """Test feature vector conversion"""
        url = "https://example.com/path"
        features = extract_features(url)
        vector = features.to_vector()

        assert isinstance(vector, list)
        assert len(vector) == 32  # Expected vector size
        assert all(isinstance(v, float) for v in vector)

    def test_to_dict(self):
        """Test feature dictionary conversion"""
        url = "https://example.com/path"
        features = extract_features(url)
        feature_dict = features.to_dict()

        assert isinstance(feature_dict, dict)
        assert "url_length" in feature_dict
        assert "domain" in feature_dict
        assert "tld_risk_score" in feature_dict


class TestGetTld:
    """Tests for TLD extraction"""

    def test_get_tld_com(self):
        assert get_tld("example.com") == ".com"

    def test_get_tld_xyz(self):
        assert get_tld("example.xyz") == ".xyz"

    def test_get_tld_subdomain(self):
        assert get_tld("sub.example.org") == ".org"

    def test_get_tld_no_tld(self):
        assert get_tld("localhost") == ""


class TestIsIpAddress:
    """Tests for IP address detection"""

    def test_ipv4_valid(self):
        assert is_ip_address("192.168.1.1") is True
        assert is_ip_address("10.0.0.1") is True
        assert is_ip_address("255.255.255.255") is True

    def test_ipv4_invalid(self):
        assert is_ip_address("example.com") is False
        assert is_ip_address("192.168.1") is False

    def test_not_ip(self):
        assert is_ip_address("www.google.com") is False


class TestTldRiskScore:
    """Tests for TLD risk scoring"""

    def test_high_risk_tlds(self):
        assert get_tld_risk_score(".xyz") == 0.9
        assert get_tld_risk_score(".top") == 0.9
        assert get_tld_risk_score(".tk") == 0.9
        assert get_tld_risk_score(".ml") == 0.9

    def test_medium_risk_tlds(self):
        assert get_tld_risk_score(".co") == 0.5
        assert get_tld_risk_score(".io") == 0.5

    def test_low_risk_tlds(self):
        assert get_tld_risk_score(".com") == 0.1
        assert get_tld_risk_score(".org") == 0.1
        assert get_tld_risk_score(".gov") == 0.1

    def test_unknown_tld(self):
        assert get_tld_risk_score(".unknown") == 0.3


class TestCalculateEntropy:
    """Tests for entropy calculation"""

    def test_entropy_empty(self):
        assert calculate_entropy("") == 0.0

    def test_entropy_single_char(self):
        assert calculate_entropy("aaaa") == 0.0

    def test_entropy_varied(self):
        entropy = calculate_entropy("abcdefgh")
        assert entropy > 0
        assert entropy <= 3.0  # log2(8) = 3

    def test_entropy_random_like(self):
        # Higher entropy for more random-looking strings
        low_entropy = calculate_entropy("aaaaaaa")
        high_entropy = calculate_entropy("a1b2c3d4")
        assert high_entropy > low_entropy


class TestLevenshteinDistance:
    """Tests for Levenshtein distance calculation"""

    def test_identical_strings(self):
        assert levenshtein_distance("test", "test") == 0

    def test_one_char_difference(self):
        assert levenshtein_distance("test", "tast") == 1
        assert levenshtein_distance("test", "tests") == 1

    def test_different_strings(self):
        assert levenshtein_distance("abc", "xyz") == 3

    def test_empty_string(self):
        assert levenshtein_distance("", "test") == 4
        assert levenshtein_distance("test", "") == 4

    def test_symmetry(self):
        assert levenshtein_distance("abc", "abcd") == levenshtein_distance("abcd", "abc")


class TestDetectTyposquatting:
    """Tests for typosquatting detection"""

    def test_exact_brand_different_tld(self):
        score, brand, dist = detect_typosquatting("paypal.xyz")
        assert score == 1.0
        assert brand == "paypal"
        assert dist == 0

    def test_typosquat_one_char(self):
        score, brand, dist = detect_typosquatting("paypa1.com")
        assert score > 0.7
        assert brand == "paypal"
        assert dist <= 2

    def test_brand_in_domain(self):
        score, brand, dist = detect_typosquatting("paypal-secure.xyz")
        assert score > 0
        assert brand == "paypal"

    def test_legitimate_domain(self):
        score, brand, dist = detect_typosquatting("randomsite.com")
        assert score == 0.0
        assert brand is None

    def test_real_brand_domain(self):
        # Real brand domains shouldn't trigger typosquatting
        score, brand, dist = detect_typosquatting("paypal.com")
        assert score == 0.0 or brand is None


class TestBatchFeatures:
    """Tests for batch feature extraction"""

    def test_extract_batch_features(self):
        urls = [
            "https://example.com",
            "https://test.org/path",
            "http://192.168.1.1/admin",
        ]
        features = extract_batch_features(urls)

        assert len(features) == 3
        assert all(isinstance(f, URLFeatures) for f in features)

    def test_extract_batch_features_empty(self):
        features = extract_batch_features([])
        assert features == []
