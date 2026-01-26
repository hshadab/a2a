"""
Scout Agent (2-Agent Model with Mutual Work Verification)

Discovers URLs one at a time with zkML proofs.
Part of the circular economy: Scout ←→ Analyst

Per-URL Payment Flow:
- Analyst pays Scout $0.001 per URL discovery
- Scout pays Analyst $0.001 per URL classification feedback
- Net change per agent: $0.00 (only gas consumed)

Self-Authorization (prevents rogue spending):
- Each agent generates its own zkML spending proof
- Each agent self-verifies its proof BEFORE spending

Mutual Work Verification (per URL):
- Scout discovers 1 URL → generates quality work proof
- Analyst verifies Scout's work proof → pays Scout
- Analyst classifies URL → generates classification work proof
- Scout verifies Analyst's work proof → pays feedback
"""
import asyncio
import uuid
import time
from datetime import datetime
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import uvicorn

import sys
sys.path.append('../..')

from shared.config import config
from shared.types import (
    URLBatch, Classification,
    BatchRecord, ClassificationRecord, NetworkStats
)
from shared.database import db
from shared.events import (
    broadcaster,
    emit_scout_found_urls, emit_scout_authorizing, emit_scout_authorized,
    emit_spending_proof_verified, emit_payment_sending, emit_payment_sent,
    emit_work_verified, emit_database_updated, emit_error
)
from shared.a2a import (
    A2AClient,
    invoke_analyst_classification,
    build_agent_card, build_skill,
    build_agent_card_v3, build_skill_v3
)
from shared.x402 import X402Client
from shared.logging_config import scout_logger as logger
from shared.prover import authorization_prover, classifier_prover, quality_scorer_prover

from sources import (
    PhishTankSource, OpenPhishSource, URLhausSource,
    SyntheticSource, AlexaTopSource,
    CertTransparencySource, TwitterSource, PasteSiteSource,
    TyposquatSource
)
from shared.reputation import reputation_manager
import httpx


# ============ Scout Agent (2-Agent Model) ============

class ScoutAgent:
    """
    Scout Agent discovers URLs one at a time with zkML work proofs.

    Per-URL Circular Economy:
    - Analyst pays Scout $0.001 per URL discovery
    - Scout pays Analyst $0.001 per URL classification feedback
    - Net change: $0.00 (only gas consumed)

    Per-URL Flow:
    1. Discover 1 URL from sources
    2. Generate quality work proof (proves Scout analyzed the URL)
    3. Analyst verifies work proof → pays Scout
    4. Analyst classifies → Scout verifies classification proof → pays feedback
    """

    def __init__(self):
        # Build sources list based on configuration
        # Priority order: fast feeds first for reliability, then original discovery
        self.sources = [
            # FAST AGGREGATION FEEDS - quick HTTP fetches, reliable
            OpenPhishSource(),    # Real phishing URLs (free, single HTTP fetch)
            URLhausSource(),      # Real malware URLs (free, single HTTP fetch)

            # ORIGINAL DISCOVERY - slower but finds threats before anyone else
            CertTransparencySource(lookback_days=1),  # New certs (queries 14 patterns - slow)
            TyposquatSource(check_interval_hours=6),  # Proactive typosquat scanning
        ]

        # Add PhishTank source if enabled (uses public data dump, API key optional)
        if config.enable_phishtank_source:
            self.sources.append(PhishTankSource(api_key=config.phishtank_api_key))

        # Add Twitter source if enabled and configured
        if config.enable_twitter_source and config.twitter_bearer_token:
            self.sources.append(TwitterSource(bearer_token=config.twitter_bearer_token))

        # Add paste site source if enabled
        if config.enable_paste_source:
            self.sources.append(PasteSiteSource())

        # FALLBACK - synthetic for demo if all else empty (always last)
        self.sources.append(SyntheticSource(phishing_ratio=0.35))

        self.a2a_client = A2AClient()
        self.x402_client = X402Client()

        self.running = False
        self.last_batch_time: Optional[datetime] = None
        self.batches_processed = 0
        self.urls_processed = 0

        # Track total paid to Analyst for feedback
        self.total_paid_analyst_feedback = 0.0

        # Model commitment for spending authorization
        self.spending_model_commitment = None
        # Model commitment for quality work proof
        self.quality_model_commitment = None

    async def initialize(self):
        """Initialize model commitments for spending and work proofs"""
        self.spending_model_commitment = authorization_prover.prover.get_model_commitment(
            config.authorization_model_path
        )
        self.quality_model_commitment = quality_scorer_prover.prover.get_model_commitment(
            config.quality_scorer_model_path
        )
        logger.info(f"Scout Agent initialized. Spending model commitment: {self.spending_model_commitment[:16]}...")
        logger.info(f"Quality scorer model commitment: {self.quality_model_commitment[:16]}...")

    async def start(self):
        """Start the continuous discovery loop"""
        self.running = True
        await self.initialize()
        logger.info(f"Scout Agent starting... Interval: {config.scout_interval_seconds}s")

        while self.running:
            try:
                # In 2-agent model, Scout waits for Analyst to call discover-urls
                # This loop is for monitoring/maintenance only
                await asyncio.sleep(config.scout_interval_seconds)
            except Exception as e:
                logger.error(f"Error in Scout loop: {e}", exc_info=True)
                await emit_error(None, str(e))

    async def stop(self):
        """Stop the discovery loop"""
        self.running = False

    async def _generate_spending_proof(
        self,
        batch_id: str,
        url_count: int,
        estimated_cost: float,
        budget_remaining: float,
        source_reputation: float
    ) -> dict:
        """
        Generate a self-authorization spending proof.

        This proves Scout has evaluated the spending decision correctly
        using the zkML authorization model.
        """
        time_since_last = 0
        if self.last_batch_time:
            time_since_last = int((datetime.utcnow() - self.last_batch_time).total_seconds())

        await emit_scout_authorizing(batch_id, url_count, estimated_cost)

        # Generate zkML proof for spending authorization
        proof_result = await authorization_prover.prove_authorization(
            batch_size=url_count,
            budget_remaining=budget_remaining,
            estimated_cost=estimated_cost,
            source_reputation=source_reputation,
            novelty_score=0.9,  # High novelty for discovered URLs
            time_since_last=time_since_last,
            threat_level=0.5   # Moderate threat level
        )

        # Extract decision from proof output
        output = proof_result.output
        decision = output.get("decision", "AUTHORIZED")
        confidence = output.get("confidence", 0.9)

        logger.info(f"Scout self-authorization: {decision} (confidence: {confidence:.2f})")

        await emit_scout_authorized(
            batch_id=batch_id,
            decision=decision,
            confidence=confidence,
            proof_hash=proof_result.proof_hash,
            prove_time_ms=proof_result.prove_time_ms
        )

        return {
            "decision": decision,
            "confidence": confidence,
            "proof": proof_result.proof_hex,
            "proof_hash": proof_result.proof_hash,
            "model_commitment": proof_result.model_commitment,
            "input_commitment": proof_result.input_commitment,
            "output_commitment": proof_result.output_commitment,
            "prove_time_ms": proof_result.prove_time_ms
        }

    async def _verify_own_spending_proof(self, spending_proof: dict) -> tuple[bool, int]:
        """
        Self-verify own spending authorization proof BEFORE spending.

        This ensures Scout is authorized to spend and prevents rogue spending.
        Each agent must verify its own proof before making any payment.
        """
        start_time = time.time()

        try:
            proof_hex = spending_proof.get("proof", "")
            if not proof_hex:
                logger.warning("No spending proof to verify")
                return False, 0

            proof_bytes = bytes.fromhex(proof_hex)

            result = await authorization_prover.verify_authorization(
                proof=proof_bytes,
                model_commitment=spending_proof.get("model_commitment", ""),
                input_commitment=spending_proof.get("input_commitment", ""),
                output_commitment=spending_proof.get("output_commitment", "")
            )

            verify_time = int((time.time() - start_time) * 1000)

            if not result.valid:
                logger.error(f"Self spending proof verification failed: {result.error}")
            else:
                logger.info(f"Self spending proof verified in {verify_time}ms - authorized to spend")

            return result.valid, verify_time

        except (ValueError, TypeError) as e:
            logger.error(f"Error verifying own spending proof: {e}")
            verify_time = int((time.time() - start_time) * 1000)
            return False, verify_time

    async def _generate_quality_work_proof(
        self,
        request_id: str,
        url: str,
        source_reputation: float
    ) -> dict:
        """
        Generate a quality work proof for a single discovered URL.

        This proves Scout actually analyzed the URL and scored its quality.
        Analyst verifies this proof BEFORE paying Scout.

        Returns work proof with quality tier assessment.
        """
        from agents.analyst.features import extract_features

        logger.info(f"Generating quality work proof for URL: {url[:50]}...")

        # Extract features for quality scoring
        try:
            features = extract_features(url)
            url_features = features.to_vector()[:16]  # First 16 features
        except Exception as e:
            logger.warning(f"Error extracting features for {url}: {e}")
            url_features = [0.5] * 16  # Default features

        # Generate single URL quality proof
        proof_result = await quality_scorer_prover.prove_quality_score(
            url_features=url_features,
            source_reputation=source_reputation,
            is_novel=True,
            age_hours=0.5,
            threat_feed_count=0
        )

        # Extract quality tier from proof output
        output = proof_result.output
        quality_tier = output.get("quality_tier", output.get("classification", "MEDIUM"))
        confidence = output.get("confidence", 0.8)

        logger.info(f"Quality work proof: {quality_tier} (confidence: {confidence:.2f})")

        return {
            "quality_tier": quality_tier,
            "confidence": confidence,
            "proof": proof_result.proof_hex,
            "proof_hash": proof_result.proof_hash,
            "model_commitment": proof_result.model_commitment,
            "input_commitment": proof_result.input_commitment,
            "output_commitment": proof_result.output_commitment,
            "prove_time_ms": proof_result.prove_time_ms
        }

    async def _discover_url(self) -> tuple[Optional[str], str, float]:
        """Discover a single URL from sources (filters already-classified)"""
        for source in self.sources:
            try:
                urls = await source.fetch_urls(limit=10)  # Fetch small batch, return 1
                if urls:
                    # Filter out already classified
                    novel_urls = await db.filter_novel_urls(urls)
                    if novel_urls:
                        # Use dynamic reputation if enabled
                        if config.enable_dynamic_reputation:
                            reputation = reputation_manager.get_reputation(source.name)
                        else:
                            reputation = source.reputation
                        # Return just the first novel URL
                        return novel_urls[0], source.name, reputation
            except (ConnectionError, TimeoutError) as e:
                logger.warning(f"Network error fetching from {source.name}: {e}")
                source.record_error()
            except Exception as e:
                logger.error(f"Error fetching from {source.name}: {e}", exc_info=True)
                source.record_error()

        return None, "", 0.0

    async def _get_budget(self) -> float:
        """Get current treasury balance"""
        try:
            return self.x402_client.get_balance()
        except (ConnectionError, ValueError) as e:
            logger.warning(f"Could not get real balance: {e}, using simulated")
            return 1000.0

    async def _pay_agent(self, agent_url: str, amount: float, memo: str):
        """Make payment to an agent"""
        try:
            # Get agent's wallet address from their card
            card = await self.a2a_client.get_agent_card(agent_url)
            wallet = card.capabilities.get("wallet", config.treasury_address)

            return await self.x402_client.make_payment(
                recipient=wallet,
                amount_usdc=amount,
                memo=memo
            )
        except (ConnectionError, TimeoutError) as e:
            logger.warning(f"Payment network error (continuing with simulated): {e}")
            return None
        except Exception as e:
            logger.warning(f"Payment error (continuing with simulated): {e}")
            return None

    async def _pay_analyst_feedback(self, request_id: str):
        """
        Pay Analyst for classification feedback (per URL).

        Self-Authorization Flow:
        1. Generate spending proof
        2. Self-verify the proof (prevents rogue spending)
        3. Only pay if proof is valid

        This completes the circular economy:
        - Analyst pays Scout $0.001 per URL discovery
        - Scout pays Analyst $0.001 per URL classification feedback
        """
        feedback_cost = config.feedback_price_per_url

        # 1. Generate spending proof
        budget = await self._get_budget()
        spending_proof = await self._generate_spending_proof(
            batch_id=request_id,
            url_count=1,
            estimated_cost=feedback_cost,
            budget_remaining=budget,
            source_reputation=1.0  # High trust for feedback
        )

        if spending_proof.get("decision") != "AUTHORIZED":
            logger.warning(f"Feedback payment not authorized: {spending_proof.get('decision')}")
            return None

        # 2. Self-verify the spending proof BEFORE spending
        proof_valid, verify_time = await self._verify_own_spending_proof(spending_proof)
        await emit_spending_proof_verified(request_id, "scout", proof_valid, verify_time)

        if not proof_valid:
            logger.error(f"Self spending proof verification failed - aborting payment")
            return None

        # 3. Now we're authorized to spend - proceed with payment
        await emit_payment_sending(request_id, feedback_cost, config.analyst_url)

        receipt = await self._pay_agent(
            config.analyst_url,
            feedback_cost,
            f"feedback-{request_id}"
        )

        tx_hash = receipt.tx_hash if receipt else "simulated"
        self.total_paid_analyst_feedback += feedback_cost
        await emit_payment_sent(request_id, tx_hash, feedback_cost)

        # Confirm payment with Analyst
        await self._confirm_analyst_feedback_payment(request_id, tx_hash, feedback_cost)

        logger.info(f"Paid Analyst ${feedback_cost} for feedback (tx: {tx_hash})")
        return receipt

    async def _confirm_analyst_feedback_payment(
        self,
        request_id: str,
        tx_hash: str,
        amount: float
    ):
        """
        Confirm feedback payment to Analyst for single URL.

        Note: We already self-verified our spending proof before paying.
        Analyst does NOT verify our spending proof - each agent self-verifies.
        """
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    f"{config.analyst_url}/confirm-feedback-payment",
                    json={
                        "request_id": request_id,
                        "tx_hash": tx_hash,
                        "amount": amount
                    }
                )
                if response.status_code == 200:
                    logger.info(f"Feedback payment confirmed with Analyst for {request_id}")
                else:
                    logger.warning(f"Feedback payment confirmation response: {response.status_code}")
        except Exception as e:
            # Non-critical - payment was already made
            logger.warning(f"Could not confirm feedback payment with Analyst: {e}")

    async def _verify_work_proof(self, class_response: dict) -> tuple[bool, int]:
        """Verify the analyst's work proof"""
        start_time = time.time()

        try:
            # Get proof bytes from hex
            proof_hex = class_response.get("proof", "")
            if not proof_hex:
                logger.warning("No proof provided in classification response")
                return False, 0

            proof_bytes = bytes.fromhex(proof_hex)

            # Verify using the classifier prover
            result = await classifier_prover.verify_classification(
                proof=proof_bytes,
                model_commitment=class_response.get("model_commitment", ""),
                input_commitment=class_response.get("input_commitment", ""),
                output_commitment=class_response.get("output_commitment", "")
            )

            verify_time = int((time.time() - start_time) * 1000)

            if not result.valid:
                logger.error(f"Work proof invalid: {result.error}")
            else:
                logger.info(f"Work proof verified in {verify_time}ms")

            return result.valid, verify_time

        except (ValueError, TypeError) as e:
            logger.error(f"Error verifying work proof: {e}")
            verify_time = int((time.time() - start_time) * 1000)
            return False, verify_time

    async def _store_classifications(
        self,
        batch_id: str,
        results: List[dict],
        proof_hash: str,
        model_commitment: str,
        source: str,
        scout_spending_proof_hash: str,
        analyst_paid: float
    ):
        """Store classification results in database"""
        # Import clustering lazily to avoid circular imports
        if config.enable_clustering:
            try:
                from agents.analyst.clustering import campaign_clusterer
            except ImportError:
                campaign_clusterer = None
        else:
            campaign_clusterer = None

        records = []
        per_url_payment = analyst_paid / len(results) if results else 0

        for result in results:
            record = ClassificationRecord(
                url=result["url"],
                domain=result.get("domain", self._extract_domain(result["url"])),
                classification=Classification(result["classification"]),
                confidence=result["confidence"],
                proof_hash=proof_hash,
                model_commitment=model_commitment,
                input_commitment=result.get("input_commitment", ""),
                output_commitment=result.get("output_commitment", ""),
                features=result.get("features", {}),
                context_used=result.get("context_used", {}),
                source=source,
                batch_id=batch_id,
                analyst_paid_usdc=per_url_payment,
                policy_proof_hash=scout_spending_proof_hash  # Now stores Scout's spending proof
            )
            records.append(record)

        await db.insert_classifications_batch(records)

        # Campaign clustering for phishing domains
        if campaign_clusterer:
            for record in records:
                try:
                    campaign = await campaign_clusterer.cluster_domain(record)
                    if campaign:
                        logger.debug(f"Domain {record.domain} clustered into campaign {campaign.name}")
                except Exception as e:
                    logger.warning(f"Error clustering domain {record.domain}: {e}")

        # Update reputation tracking
        if config.enable_dynamic_reputation:
            for record in records:
                try:
                    await reputation_manager.record_classification(
                        source_name=source,
                        url=record.url,
                        predicted=record.classification.value,
                        confidence=record.confidence,
                    )
                except Exception as e:
                    logger.warning(f"Error recording reputation for {record.url}: {e}")

    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.netloc
        except (ValueError, AttributeError):
            return url


# ============ FastAPI App ============

scout = ScoutAgent()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup validation
    if config.production_mode:
        config.assert_production_ready()
        logger.info("Production mode enabled - all validations passed")
    else:
        logger.warning("Running in DEMO mode - simulated proofs/payments allowed")

    # Initialize database and scout
    await db.connect()
    await db.init_schema()
    await scout.initialize()
    asyncio.create_task(scout.start())
    yield
    # Shutdown
    await scout.stop()
    await db.close()


app = FastAPI(
    title="Scout Agent (2-Agent Model)",
    description="Discovers URLs with self-authorized spending proofs. Part of Scout ←→ Analyst circular economy.",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/.well-known/agent.json")
async def agent_card():
    """A2A v0.3 Agent Card (2-Agent Model with Mutual Work Verification)"""
    return build_agent_card_v3(
        name="Scout Agent",
        description="Discovers suspicious URLs with zkML quality work proof. Analyst verifies work proof before paying. Self-authorizes spending proofs.",
        url=config.scout_url,
        version="2.1.0",
        streaming=False,
        push_notifications=False,
        state_transition_history=True,
        provider="ThreatProof Network",
        documentation_url=f"{config.scout_url}/docs",
        default_payment_address=config.treasury_address,
        supported_payment_methods=["x402"],
        skills=[
            build_skill_v3(
                skill_id="discover-url",
                name="URL Discovery",
                description="Discover a suspicious URL from Certificate Transparency, typosquatting detection, and threat feeds. Returns 1 URL + quality work proof (for buyer verification) + spending proof (self-verified).",
                tags=["threat-intel", "url-discovery", "phishing", "security", "zkml", "work-proof"],
                input_modes=["application/json"],
                output_modes=["application/json"],
                price_amount=config.discovery_price_per_url,
                price_currency="USDC",
                price_per="url",
                chain=config.base_chain_caip2,
                proof_required=True,  # Quality work proof included for buyer to verify
                model_commitment=scout.quality_model_commitment  # Work proof model
            )
        ]
    )


@app.get("/")
async def root():
    """ThreatProof - Verifiable Threat Intelligence Network (2-Agent Model)"""
    stats = await db.get_network_stats()
    return {
        "name": "ThreatProof",
        "description": "Autonomous threat intelligence with zkML-verified classifications (2-Agent Model)",
        "status": "running" if scout.running else "stopped",
        "architecture": "2-Agent Circular Economy: Scout ←→ Analyst",
        "stats": {
            "batches_processed": scout.batches_processed,
            "urls_classified": scout.urls_processed,
            "total_in_database": stats.total_urls if stats else 0,
            "phishing_detected": stats.phishing_count if stats else 0,
            "feedback_paid_usdc": scout.total_paid_analyst_feedback,
        },
        "agents": {
            "scout": config.scout_url,
            "analyst": config.analyst_url
        },
        "payment_flow": {
            "analyst_to_scout": f"${config.discovery_price_per_url} per URL (discovery)",
            "scout_to_analyst": f"${config.feedback_price_per_url} per URL (feedback)",
            "net_change": "$0.00 (only gas consumed)"
        },
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "stats": "/stats",
            "websocket": "/ws",
            "discover_url": "/skills/discover-url"
        },
        "technology": ["A2A Protocol", "x402 Payments", "zkML Self-Authorization", "Base/USDC"]
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "running": scout.running,
        "batches_processed": scout.batches_processed,
        "urls_processed": scout.urls_processed,
        "architecture": "2-agent",
        "self_authorization": True,
        "mutual_work_verification": True,
        "spending_model_commitment": scout.spending_model_commitment,
        "quality_model_commitment": scout.quality_model_commitment,
        "zkml_available": authorization_prover.prover.zkml_available
    }


@app.get("/stats")
async def stats():
    """Get network statistics"""
    network_stats = await db.get_network_stats()
    return {
        "network": network_stats,
        "scout": {
            "batches_processed": scout.batches_processed,
            "urls_processed": scout.urls_processed,
            "feedback_paid_usdc": scout.total_paid_analyst_feedback
        }
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time events"""
    await broadcaster.connect(websocket)
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        broadcaster.disconnect(websocket)


# ============ URL Discovery Service (2-Agent Model with Self-Authorization) ============

from pydantic import BaseModel

class DiscoverRequest(BaseModel):
    """Request for single URL discovery"""
    requester_id: str = "analyst"


class PaymentDue(BaseModel):
    """Payment information for proof-gated payment flow"""
    amount: float
    currency: str = "USDC"
    recipient: str
    chain: str
    memo: str


class SpendingProof(BaseModel):
    """Self-authorization spending proof"""
    decision: str
    confidence: float
    proof: str
    proof_hash: str
    model_commitment: str
    input_commitment: str
    output_commitment: str
    prove_time_ms: int


class WorkProof(BaseModel):
    """Quality work proof - proves Scout analyzed the URL"""
    quality_tier: str  # HIGH, MEDIUM, LOW, NOISE
    confidence: float
    proof: str
    proof_hash: str
    model_commitment: str
    input_commitment: str
    output_commitment: str
    prove_time_ms: int


class DiscoverResponse(BaseModel):
    """Response from single URL discovery (2-Agent Model)"""
    request_id: str
    url: str  # Single URL
    source: str
    source_reputation: float
    # Self-authorization spending proof (Scout self-verifies before spending)
    spending_proof: SpendingProof
    # Work proof - Analyst verifies before paying Scout
    work_proof: WorkProof
    timestamp: str
    payment_due: PaymentDue


@app.post("/skills/discover-url")
async def discover_url(request: DiscoverRequest) -> DiscoverResponse:
    """
    Single URL Discovery Service (2-Agent Model with Mutual Work Verification).

    Per-URL Flow:
    1. Scout discovers 1 URL from threat sources
    2. Scout generates quality work proof (proves analysis)
    3. Scout generates spending proof (self-verified)
    4. Returns URL + work_proof + spending_proof + payment_due
    5. Analyst verifies Scout's work proof
    6. If valid, Analyst pays Scout $0.001
    7. If invalid, no payment
    """
    request_id = str(uuid.uuid4())
    logger.info(f"URL Discovery request: {request_id}")

    await emit_scout_found_urls(request_id, 0, "discovering", [])

    # 1. Discover single URL from sources
    url, source_name, source_reputation = await scout._discover_url()

    if not url:
        raise HTTPException(
            status_code=404,
            detail="No new URL discovered from sources"
        )

    logger.info(f"Discovered URL from {source_name}: {url[:50]}...")

    await emit_scout_found_urls(request_id, 1, source_name, [url])

    # 2. Generate quality work proof (Scout's work proof for Analyst to verify)
    work_proof_data = await scout._generate_quality_work_proof(
        request_id=request_id,
        url=url,
        source_reputation=source_reputation
    )

    # 3. Get current budget
    budget = await scout._get_budget()

    # Calculate discovery cost (per URL)
    discovery_cost = config.discovery_price_per_url

    # 4. Generate self-authorization spending proof (Scout self-verifies)
    spending_proof_data = await scout._generate_spending_proof(
        batch_id=request_id,
        url_count=1,
        estimated_cost=discovery_cost,
        budget_remaining=budget,
        source_reputation=source_reputation
    )

    if spending_proof_data.get("decision") != "AUTHORIZED":
        raise HTTPException(
            status_code=403,
            detail=f"Self-authorization denied: {spending_proof_data.get('decision')}"
        )

    # 5. Build response with work proof and spending proof
    spending_proof = SpendingProof(
        decision=spending_proof_data["decision"],
        confidence=spending_proof_data["confidence"],
        proof=spending_proof_data["proof"],
        proof_hash=spending_proof_data["proof_hash"],
        model_commitment=spending_proof_data["model_commitment"],
        input_commitment=spending_proof_data["input_commitment"],
        output_commitment=spending_proof_data["output_commitment"],
        prove_time_ms=spending_proof_data["prove_time_ms"]
    )

    work_proof = WorkProof(
        quality_tier=work_proof_data["quality_tier"],
        confidence=work_proof_data["confidence"],
        proof=work_proof_data["proof"],
        proof_hash=work_proof_data["proof_hash"],
        model_commitment=work_proof_data["model_commitment"],
        input_commitment=work_proof_data["input_commitment"],
        output_commitment=work_proof_data["output_commitment"],
        prove_time_ms=work_proof_data["prove_time_ms"]
    )

    payment_due = PaymentDue(
        amount=discovery_cost,
        currency="USDC",
        recipient=config.treasury_address,
        chain=config.base_chain_caip2,
        memo=f"discover-{request_id}"
    )

    return DiscoverResponse(
        request_id=request_id,
        url=url,
        source=source_name,
        source_reputation=source_reputation,
        spending_proof=spending_proof,
        work_proof=work_proof,
        timestamp=datetime.utcnow().isoformat(),
        payment_due=payment_due
    )


class ConfirmDiscoveryPaymentRequest(BaseModel):
    """Request to confirm payment for URL discovery"""
    request_id: str
    tx_hash: str
    amount: float


@app.post("/confirm-discovery-payment")
async def confirm_discovery_payment(request: ConfirmDiscoveryPaymentRequest):
    """
    Confirm payment received for single URL discovery (2-Agent Model).

    Called by Analyst after:
    1. Verifying Scout's work proof (quality proof)
    2. Generating its own spending proof
    3. Self-verifying its spending proof (prevents rogue spending)
    4. Making the USDC payment to Scout

    Note: Analyst self-verifies its own spending proof before paying.
    Scout does NOT verify Analyst's spending proof - each agent verifies their own.
    """
    # Verify payment if in production mode
    if config.production_mode and request.tx_hash != "simulated":
        try:
            tolerance = 1 - config.payment_tolerance
            is_valid, error = scout.x402_client.verify_payment(
                tx_hash=request.tx_hash,
                expected_recipient=config.treasury_address,
                expected_amount_usdc=request.amount * tolerance
            )
            if not is_valid:
                raise HTTPException(
                    status_code=400,
                    detail=f"Payment verification failed: {error}"
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Payment verification error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    logger.info(f"Discovery payment confirmed for {request.request_id}: ${request.amount} USDC (tx: {request.tx_hash})")

    # Pay Analyst feedback (completes the circular economy)
    try:
        await scout._pay_analyst_feedback(request.request_id)
    except Exception as e:
        logger.warning(f"Could not pay Analyst feedback: {e}")

    return {
        "status": "confirmed",
        "request_id": request.request_id,
        "amount": request.amount,
        "feedback_payment": "sent"
    }


if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
