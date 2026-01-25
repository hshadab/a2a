"""
Shared types for the Threat Intelligence Network
"""
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
import uuid


class Classification(str, Enum):
    PHISHING = "PHISHING"
    SAFE = "SAFE"
    SUSPICIOUS = "SUSPICIOUS"


class PolicyDecision(str, Enum):
    AUTHORIZED = "AUTHORIZED"
    DENIED = "DENIED"


class AgentStatus(str, Enum):
    IDLE = "idle"
    ACTIVE = "active"
    WORKING = "working"
    ERROR = "error"


# ============ URL Batch Types ============

class URLBatch(BaseModel):
    batch_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    urls: List[str]
    source: str
    created_at: datetime = Field(default_factory=datetime.utcnow)


class URLFeatures(BaseModel):
    url: str
    domain: str
    tld: str
    domain_length: int
    path_length: int
    path_entropy: float
    query_param_count: int
    has_ip_address: bool
    has_port: bool
    subdomain_count: int
    special_char_count: int
    digit_ratio: float
    tld_risk_score: float
    typosquat_score: float
    brand_similarity: Optional[str] = None

    # Context from database (filled at classification time)
    domain_phish_rate: Optional[float] = None
    registrar_phish_rate: Optional[float] = None
    ip_phish_rate: Optional[float] = None
    similar_domains_phish_rate: Optional[float] = None


# ============ Authorization Types ============

class AuthorizationRequest(BaseModel):
    batch_id: str
    url_count: int
    estimated_cost_usdc: float
    budget_remaining_usdc: float
    source_reputation: float  # 0-1
    novelty_score: float      # 0-1, how new/unique are these URLs
    time_since_last_batch_seconds: int
    threat_level: float       # 0-1, current activity level


class AuthorizationResponse(BaseModel):
    batch_id: str
    decision: PolicyDecision
    confidence: float
    proof: str
    proof_hash: str
    model_commitment: str
    input_commitment: str
    output_commitment: str
    prove_time_ms: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ============ Classification Types ============

class ClassificationRequest(BaseModel):
    batch_id: str
    urls: List[str]
    policy_proof_hash: str  # Link to authorization


class ClassificationResult(BaseModel):
    url: str
    classification: Classification
    confidence: float
    features: URLFeatures


class ClassificationResponse(BaseModel):
    batch_id: str
    results: List[ClassificationResult]
    proof: str
    proof_hash: str
    model_commitment: str
    input_commitment: str
    output_commitment: str
    prove_time_ms: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ============ Payment Types ============

class PaymentRequest(BaseModel):
    amount_usdc: float
    recipient: str
    memo: str
    chain_id: int = 8453  # Base mainnet
    token_address: str = "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913"  # USDC


class PaymentReceipt(BaseModel):
    tx_hash: str
    amount_usdc: float
    sender: str
    recipient: str
    timestamp: datetime
    block_number: int
    chain_id: int


class X402PaymentChallenge(BaseModel):
    """x402 v1 Payment Required response (legacy)"""
    version: str = "1"
    amount: str
    currency: str = "USDC"
    recipient: str
    chain_id: int = 8453
    chain: str = "eip155:8453"  # CAIP-2 format for interoperability
    token_address: str
    expires: int
    nonce: str


class X402PaymentRequirement(BaseModel):
    """x402 v2 payment requirement (single payment option)"""
    scheme: str = "exact"  # "exact" or "up-to"
    network: str = "base-mainnet"  # Network identifier
    maxAmountRequired: str  # Amount in base units (e.g., "1000" = 0.001 USDC)
    resource: str  # Resource path being paid for
    description: Optional[str] = None  # Human-readable description
    payTo: str  # Recipient address
    asset: str  # Token contract address
    maxTimeoutSeconds: int = 300  # Payment deadline


class X402PaymentRequired(BaseModel):
    """x402 v2 Payment Required response structure"""
    x402Version: int = 2
    accepts: List[X402PaymentRequirement]
    error: Optional[str] = None


# ============ A2A Types ============

class SkillDefinition(BaseModel):
    id: str
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    price: Dict[str, Any]
    proof_policy: Optional[Dict[str, Any]] = None


class AgentCard(BaseModel):
    schema_version: str = "1.0"
    name: str
    description: str
    url: str
    capabilities: Dict[str, Any]


# ============ A2A v0.3 Types ============

class AgentCapabilitiesV3(BaseModel):
    """A2A v0.3 capabilities object"""
    streaming: bool = False
    pushNotifications: bool = False
    stateTransitionHistory: bool = True


class AgentSkillV3(BaseModel):
    """A2A v0.3 skill definition"""
    id: str
    name: str
    description: str
    tags: List[str] = []
    inputModes: List[str] = ["application/json"]
    outputModes: List[str] = ["application/json"]
    # x402 extension for pricing
    price: Optional[Dict[str, Any]] = None


class AgentCardV3(BaseModel):
    """A2A v0.3 compliant agent card"""
    name: str
    description: str
    url: str
    version: str = "1.0.0"
    protocolVersion: str = "0.3"
    capabilities: AgentCapabilitiesV3 = Field(default_factory=AgentCapabilitiesV3)
    authentication: Dict[str, Any] = Field(default_factory=lambda: {"schemes": ["none"]})
    skills: List[AgentSkillV3] = []
    provider: str = "ThreatProof Network"
    documentationUrl: Optional[str] = None
    # x402 extensions
    defaultPaymentAddress: Optional[str] = None
    supportedPaymentMethods: List[str] = ["x402"]


# ============ Event Types (for WebSocket) ============

class EventType(str, Enum):
    SCOUT_FOUND_URLS = "SCOUT_FOUND_URLS"
    POLICY_REQUESTING = "POLICY_REQUESTING"
    POLICY_PROVING = "POLICY_PROVING"
    POLICY_RESPONSE = "POLICY_RESPONSE"
    POLICY_VERIFIED = "POLICY_VERIFIED"
    PAYMENT_SENDING = "PAYMENT_SENDING"
    PAYMENT_SENT = "PAYMENT_SENT"
    ANALYST_PROCESSING = "ANALYST_PROCESSING"
    ANALYST_PROVING = "ANALYST_PROVING"
    ANALYST_RESPONSE = "ANALYST_RESPONSE"
    WORK_VERIFIED = "WORK_VERIFIED"
    DATABASE_UPDATED = "DATABASE_UPDATED"
    ERROR = "ERROR"


class Event(BaseModel):
    type: EventType
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    data: Dict[str, Any]


# ============ Database Types ============

class ClassificationRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    url: str
    domain: str
    classification: Classification
    confidence: float
    proof_hash: str
    model_commitment: str
    input_commitment: str
    output_commitment: str
    features: Dict[str, Any]
    context_used: Dict[str, Any]
    source: str
    batch_id: str
    analyst_paid_usdc: float
    policy_proof_hash: str
    classified_at: datetime = Field(default_factory=datetime.utcnow)


class BatchRecord(BaseModel):
    id: str
    url_count: int
    source: str
    policy_decision: PolicyDecision
    policy_proof_hash: str
    policy_paid_usdc: float
    total_analyst_paid_usdc: float
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None


class DomainStats(BaseModel):
    domain: str
    times_seen: int = 0
    phishing_count: int = 0
    safe_count: int = 0
    suspicious_count: int = 0
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None

    @property
    def phishing_rate(self) -> float:
        if self.times_seen == 0:
            return 0.0
        return self.phishing_count / self.times_seen


class RegistrarStats(BaseModel):
    registrar: str
    domains_seen: int = 0
    phishing_count: int = 0
    last_updated: Optional[datetime] = None

    @property
    def phishing_rate(self) -> float:
        if self.domains_seen == 0:
            return 0.0
        return self.phishing_count / self.domains_seen


class IPStats(BaseModel):
    ip: str
    domains_hosted: int = 0
    phishing_count: int = 0
    last_updated: Optional[datetime] = None

    @property
    def phishing_rate(self) -> float:
        if self.domains_hosted == 0:
            return 0.0
        return self.phishing_count / self.domains_hosted


# ============ Stats Types ============

class NetworkStats(BaseModel):
    total_urls: int = 0
    phishing_count: int = 0
    safe_count: int = 0
    suspicious_count: int = 0
    total_batches: int = 0
    total_proofs: int = 0
    treasury_balance_usdc: float = 0.0
    total_spent_usdc: float = 0.0
    policy_paid_usdc: float = 0.0
    analyst_paid_usdc: float = 0.0
    running_since: Optional[datetime] = None


class AgentStats(BaseModel):
    name: str
    status: AgentStatus
    urls_processed_today: int = 0
    proofs_generated_today: int = 0
    earnings_today_usdc: float = 0.0
    last_active: Optional[datetime] = None
