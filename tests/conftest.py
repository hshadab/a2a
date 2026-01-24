"""
Pytest fixtures for Threat Intelligence Network tests
"""
import asyncio
import pytest
import sys
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from shared.database import InMemoryDatabase
from shared.types import Classification, ClassificationRecord, BatchRecord, PolicyDecision


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def mock_db():
    """Provide an in-memory database for tests"""
    db = InMemoryDatabase()
    await db.connect()
    yield db


@pytest.fixture
def sample_urls() -> List[str]:
    """Sample URLs for testing"""
    return [
        "https://secure-paypal-login.xyz/verify",
        "https://google.com",
        "https://amaz0n-account.top/signin",
        "https://192.168.1.1/admin",
        "https://microsoft.com/login",
        "https://my-legitimate-bank.com/account",
        "https://faceb00k-login.tk/auth",
        "https://github.com",
    ]


@pytest.fixture
def sample_phishing_urls() -> List[str]:
    """URLs that should be classified as phishing"""
    return [
        "https://secure-paypal-login.xyz/verify",
        "https://amaz0n-account.top/signin",
        "https://faceb00k-login.tk/auth",
        "https://app1e-id-verify.ml/secure",
        "https://netfl1x-account.gq/billing",
    ]


@pytest.fixture
def sample_safe_urls() -> List[str]:
    """URLs that should be classified as safe"""
    return [
        "https://google.com",
        "https://microsoft.com/login",
        "https://github.com",
        "https://apple.com",
        "https://amazon.com",
    ]


@pytest.fixture
def sample_classification_record() -> ClassificationRecord:
    """Sample classification record for testing"""
    return ClassificationRecord(
        url="https://test-phishing.xyz/login",
        domain="test-phishing.xyz",
        classification=Classification.PHISHING,
        confidence=0.95,
        proof_hash="abc123def456",
        model_commitment="model_commit_123",
        input_commitment="input_commit_123",
        output_commitment="output_commit_123",
        features={
            "url_length": 35,
            "domain_length": 16,
            "tld_risk_score": 0.9,
            "typosquat_score": 0.0,
        },
        context_used={
            "domain_phish_rate": None,
            "registrar_phish_rate": None,
        },
        source="test_source",
        batch_id="batch-123",
        analyst_paid_usdc=0.0005,
        policy_proof_hash="policy_proof_123",
    )


@pytest.fixture
def sample_batch_record() -> BatchRecord:
    """Sample batch record for testing"""
    return BatchRecord(
        id="batch-123",
        url_count=10,
        source="test_source",
        policy_decision=PolicyDecision.AUTHORIZED,
        policy_proof_hash="policy_proof_123",
        policy_paid_usdc=0.001,
        total_analyst_paid_usdc=0.005,
    )


@pytest.fixture
def mock_http_response():
    """Factory fixture for creating mock HTTP responses"""
    def _create_response(content: str, status_code: int = 200, headers: Dict[str, str] = None):
        class MockResponse:
            def __init__(self):
                self.status_code = status_code
                self.text = content
                self.headers = headers or {}

            def json(self):
                import json
                return json.loads(content)

            def raise_for_status(self):
                if self.status_code >= 400:
                    raise Exception(f"HTTP {self.status_code}")

        return MockResponse()

    return _create_response


@pytest.fixture
def mock_openphish_response() -> str:
    """Mock OpenPhish feed response"""
    return """https://phishing-site1.xyz/login
https://fake-bank.top/verify
https://scam-paypal.ml/account
https://malicious-amazon.tk/signin
https://bad-microsoft.gq/oauth"""


@pytest.fixture
def mock_urlhaus_response() -> Dict[str, Any]:
    """Mock URLhaus API response"""
    return {
        "query_status": "ok",
        "urls": [
            {
                "url": "https://malware-host1.xyz/payload",
                "url_status": "online",
                "threat": "malware_download",
                "tags": ["exe", "trojan"],
            },
            {
                "url": "https://malware-host2.top/dropper",
                "url_status": "online",
                "threat": "malware_download",
                "tags": ["dll", "ransomware"],
            },
        ]
    }


@pytest.fixture
def mock_crtsh_response() -> List[Dict[str, Any]]:
    """Mock crt.sh API response"""
    return [
        {
            "id": 12345,
            "name_value": "*.paypa1-secure.xyz",
            "not_before": "2025-01-20T00:00:00",
            "not_after": "2025-04-20T00:00:00",
            "issuer_name": "Let's Encrypt Authority X3",
        },
        {
            "id": 12346,
            "name_value": "login.amaz0n-verify.top",
            "not_before": "2025-01-19T00:00:00",
            "not_after": "2025-04-19T00:00:00",
            "issuer_name": "Let's Encrypt Authority X3",
        },
    ]
