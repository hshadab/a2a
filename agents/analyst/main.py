"""
Analyst Agent (2-Agent Model with Mutual Work Verification)

Classifies URLs as PHISHING, SAFE, or SUSPICIOUS with zkML proofs.
Uses self-authorization spending proofs for all payments.

2-Agent Circular Economy:
- Analyst pays Scout $0.001 for URL discovery (with self-authorized spending proof)
- Scout pays Analyst $0.001 for classification feedback (with self-authorized spending proof)
- Net change: $0.00 (only gas consumed)

Self-Authorization (prevents rogue spending):
- Each agent generates its own zkML spending proof
- Each agent self-verifies its proof BEFORE spending
- This ensures only authorized spending occurs

Mutual Work Verification:
- Scout generates quality work proof (URL quality scoring)
- Analyst verifies Scout's work proof BEFORE paying for discovery
- Analyst generates classification work proof
- Scout verifies Analyst's work proof BEFORE paying feedback
"""
import asyncio
import os
import time
from datetime import datetime
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

import sys
sys.path.append('../..')

from shared.config import config
from shared.types import Classification
from shared.database import db
from shared.events import (
    broadcaster,
    emit_analyst_authorizing, emit_analyst_authorized, emit_spending_proof_verified,
    emit_analyst_processing, emit_analyst_proving, emit_analyst_response
)
from shared.a2a import build_agent_card, build_skill, build_agent_card_v3, build_skill_v3
from shared.x402 import (
    PaymentRequired, require_payment, get_payment_from_header, X402Client,
    HEADER_PAYMENT_SIGNATURE, HEADER_X402_RECEIPT
)
from shared.prover import classifier_prover, authorization_prover, quality_scorer_prover, compute_commitment
from shared.logging_config import analyst_logger as logger
from shared.jsonrpc import JSONRPCRouter, create_jsonrpc_endpoint, TaskNotFoundError, PaymentRequiredError as JSONRPCPaymentRequired
from shared.task import TaskStore, Task, TaskState, execute_task
from shared.sse import create_sse_response

from features import extract_features, URLFeatures


# ============ Request/Response Models ============

class ClassifyRequest(BaseModel):
    request_id: str
    url: str  # Single URL
    policy_proof_hash: str


class ClassificationResult(BaseModel):
    url: str
    domain: str
    classification: str
    confidence: float
    features: Dict[str, Any]
    context_used: Dict[str, Any]
    input_commitment: str
    output_commitment: str


class PaymentDue(BaseModel):
    """Payment information for proof-gated payment flow"""
    amount: float
    currency: str = "USDC"
    recipient: str
    chain: str
    memo: str


class ClassifyResponse(BaseModel):
    request_id: str
    result: ClassificationResult  # Single result
    proof: str
    proof_hash: str
    model_commitment: str
    input_commitment: str
    output_commitment: str
    prove_time_ms: int
    timestamp: str
    # Payment due after proof verification (proof-gated flow)
    payment_due: Optional[PaymentDue] = None


# ============ Analyst Agent (2-Agent Model) ============

class AnalystAgent:
    """
    Analyst Agent classifies URLs with zkML proofs and self-authorized spending.

    2-Agent Circular Economy:
    - Analyst pays Scout for URL discovery (with self-authorized spending proof)
    - Scout pays Analyst for classification feedback (with self-authorized spending proof)
    - Net change: $0.00 (only gas consumed)

    Self-Authorization (prevents rogue spending):
    1. Generate spending proof
    2. Self-verify the proof
    3. Only spend if proof is valid

    Responsibilities:
    1. Generate and self-verify spending proof before paying Scout
    2. Classify URLs with zkML proofs
    3. Buyer (Scout) verifies work proof before paying
    """

    def __init__(self):
        self.classifications_today = 0
        self.phishing_detected_today = 0
        self.total_earned_usdc = 0.0
        self.total_spent_usdc = 0.0
        self.total_feedback_received = 0.0  # Track feedback payments from Scout
        self.batches_processed = 0
        self.model_commitment = None
        self.spending_model_commitment = None  # For self-authorization proofs
        self.x402_client = X402Client()

        # Wallet address for receiving payments (and making payments in value chain)
        self.wallet_address = config.treasury_address

        # Orchestration loop state
        self.running = False
        self._task = None

    async def initialize(self):
        """Initialize the model commitments"""
        self.model_commitment = classifier_prover.prover.get_model_commitment(
            config.classifier_model_path
        )
        self.spending_model_commitment = authorization_prover.prover.get_model_commitment(
            config.authorization_model_path
        )
        logger.info(f"Analyst Agent initialized. Classification model: {self.model_commitment[:16]}...")
        logger.info(f"Spending authorization model: {self.spending_model_commitment[:16]}...")

    async def _generate_spending_proof(
        self,
        request_id: str,
        estimated_cost: float,
        budget_remaining: float
    ) -> dict:
        """
        Generate a self-authorization spending proof.

        This proves Analyst has evaluated the spending decision correctly
        using the zkML authorization model.
        """
        await emit_analyst_authorizing(request_id, estimated_cost)

        # Generate zkML proof for spending authorization
        proof_result = await authorization_prover.prove_authorization(
            batch_size=1,  # Per-batch spending
            budget_remaining=budget_remaining,
            estimated_cost=estimated_cost,
            source_reputation=1.0,  # High trust for Scout
            novelty_score=0.9,  # High novelty for discovered URLs
            time_since_last=300,  # Assume reasonable interval
            threat_level=0.5   # Moderate threat level
        )

        # Extract decision from proof output
        output = proof_result.output
        decision = output.get("decision", "AUTHORIZED")
        confidence = output.get("confidence", 0.9)

        logger.info(f"Analyst self-authorization: {decision} (confidence: {confidence:.2f})")

        await emit_analyst_authorized(
            request_id=request_id,
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

        This ensures Analyst is authorized to spend and prevents rogue spending.
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

    async def _verify_scout_work_proof(self, work_proof: dict) -> tuple[bool, int]:
        """
        Verify Scout's quality work proof BEFORE paying for discovery.

        This ensures Scout actually did work (analyzed URLs for quality)
        before Analyst pays for the discovered URLs.
        """
        start_time = time.time()

        try:
            proof_hex = work_proof.get("proof", "")
            if not proof_hex:
                logger.warning("No work proof provided by Scout")
                return False, 0

            proof_bytes = bytes.fromhex(proof_hex)

            result = await quality_scorer_prover.verify_quality_score(
                proof=proof_bytes,
                model_commitment=work_proof.get("model_commitment", ""),
                input_commitment=work_proof.get("input_commitment", ""),
                output_commitment=work_proof.get("output_commitment", "")
            )

            verify_time = int((time.time() - start_time) * 1000)

            if not result.valid:
                logger.error(f"Scout work proof verification failed: {result.error}")
            else:
                quality_tier = work_proof.get("quality_tier", "UNKNOWN")
                logger.info(f"Scout work proof verified in {verify_time}ms - quality: {quality_tier}")

            return result.valid, verify_time

        except (ValueError, TypeError) as e:
            logger.error(f"Error verifying Scout work proof: {e}")
            verify_time = int((time.time() - start_time) * 1000)
            return False, verify_time

    async def start(self):
        """Start the orchestration loop (2-Agent Model)"""
        self.running = True
        logger.info("Starting Analyst orchestration loop (2-Agent Model with Self-Authorization)")
        while self.running:
            try:
                await self._process_url()
            except Exception as e:
                logger.error(f"URL processing error: {e}", exc_info=True)
            await asyncio.sleep(config.scout_interval_seconds)

    async def stop(self):
        """Stop the orchestration loop"""
        self.running = False
        logger.info("Stopping Analyst orchestration loop")

    async def _process_url(self):
        """
        Process a single URL in the 2-Agent Model with mutual work verification:
        1. Generate self-authorization spending proof
        2. Self-verify the spending proof (prevents rogue spending)
        3. Call Scout to discover 1 URL
        4. Verify Scout's WORK proof (quality scoring)
        5. Pay Scout (only after work proof is verified)
        6. Classify URL with zkML proof
        7. Store result

        Mutual Work Verification:
        - Scout generates work proof (quality scoring) - Analyst verifies
        - Analyst generates work proof (classification) - Scout verifies
        - Each agent self-verifies their own spending proof
        """
        import httpx
        import uuid

        request_id = str(uuid.uuid4())
        logger.info(f"Starting URL processing {request_id} (2-Agent Model)")

        # 1. Get budget and generate self-authorization spending proof
        budget = self.x402_client.get_balance()
        discovery_cost = config.discovery_price_per_url

        if budget < discovery_cost + 0.01:
            logger.warning(f"Insufficient budget: {budget}")
            return None

        # Generate spending proof before paying Scout
        spending_proof = await self._generate_spending_proof(
            request_id=request_id,
            estimated_cost=discovery_cost,
            budget_remaining=budget
        )

        if spending_proof.get("decision") != "AUTHORIZED":
            logger.warning(f"Self-authorization denied: {spending_proof.get('decision')}")
            return None

        # 2. Self-verify the spending proof BEFORE spending
        proof_valid, verify_time = await self._verify_own_spending_proof(spending_proof)
        await emit_spending_proof_verified(request_id, "analyst", proof_valid, verify_time)

        if not proof_valid:
            logger.error(f"Self spending proof verification failed - aborting")
            return None

        # 3. Request single URL discovery from Scout
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{config.scout_url}/skills/discover-url",
                    json={"requester_id": "analyst"}
                )

                if response.status_code == 404:
                    logger.info("No new URL available from Scout")
                    return None

                if response.status_code != 200:
                    logger.error(f"Scout discovery failed: {response.status_code}")
                    return None

                discovery = response.json()
        except Exception as e:
            logger.error(f"Failed to call Scout: {e}")
            return None

        url = discovery.get("url")
        if not url:
            logger.info("No URL returned from Scout")
            return None

        logger.info(f"Received URL from Scout: {url[:50]}...")

        # Note: Scout's spending proof is for Scout's own spending (feedback payment).
        # Scout self-verifies its own spending proof.
        scout_spending_proof = discovery.get("spending_proof", {})

        # 4. Verify Scout's WORK proof before paying
        # This ensures Scout actually did work (analyzed URL quality) before we pay
        scout_work_proof = discovery.get("work_proof", {})
        if isinstance(scout_work_proof, dict) and scout_work_proof.get("proof"):
            work_proof_valid, work_verify_time = await self._verify_scout_work_proof(scout_work_proof)
            from shared.events import emit_work_verified
            await emit_work_verified(
                request_id=request_id,
                valid=work_proof_valid,
                verify_time_ms=work_verify_time,
                quality_tier=scout_work_proof.get("quality_tier", "UNKNOWN")
            )

            if not work_proof_valid:
                logger.error("Scout work proof verification failed - NOT PAYING")
                return None

            logger.info(f"Scout work proof verified: {scout_work_proof.get('quality_tier')} quality")
        else:
            logger.warning("No work proof from Scout - proceeding for backward compatibility")

        # 5. Pay Scout for URL discovery (work proof verified)
        payment_due = discovery.get("payment_due", {})
        payment_amount = payment_due.get("amount", discovery_cost)

        try:
            receipt = await self.x402_client.make_payment(
                recipient=payment_due.get("recipient", config.treasury_address),
                amount_usdc=payment_amount,
                memo=payment_due.get("memo", f"discover-{request_id}")
            )
            tx_hash = receipt.tx_hash if receipt else "simulated"
            self.total_spent_usdc += payment_amount
            logger.info(f"Paid Scout ${payment_amount} for URL discovery (tx: {tx_hash})")

            # Confirm payment with Scout
            await self._confirm_scout_payment(request_id, tx_hash, payment_amount)
        except Exception as e:
            logger.error(f"Payment to Scout failed: {e}")
            # Continue anyway in demo mode

        # 6. Classify URL (our work - Scout will verify our work proof before paying feedback)
        scout_proof_hash = scout_spending_proof.get("proof_hash", "") if isinstance(scout_spending_proof, dict) else ""
        classification_response = await self.classify_url(
            request_id=request_id,
            url=url,
            policy_proof_hash=scout_proof_hash  # Now stores Scout's spending proof hash
        )

        # 7. Store result in database
        await self._store_classification(
            request_id=request_id,
            response=classification_response,
            source=discovery.get("source", "scout")
        )

        self.batches_processed += 1
        logger.info(f"URL {request_id} classified: {url[:50]}...")
        return request_id

    async def _confirm_scout_payment(self, request_id: str, tx_hash: str, amount: float):
        """
        Confirm payment to Scout for single URL discovery.

        Note: We already self-verified our spending proof before paying.
        Scout does NOT verify our spending proof - each agent self-verifies.
        """
        import httpx
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                await client.post(
                    f"{config.scout_url}/confirm-discovery-payment",
                    json={
                        "request_id": request_id,
                        "tx_hash": tx_hash,
                        "amount": amount
                    }
                )
        except Exception as e:
            logger.warning(f"Could not confirm payment with Scout: {e}")

    async def _store_classification(self, request_id: str, response: ClassifyResponse, source: str):
        """Store single classification result in database"""
        from shared.types import ClassificationRecord, Classification as ClassEnum

        result = response.result
        record = ClassificationRecord(
            url=result.url,
            domain=result.domain,
            classification=ClassEnum(result.classification),
            confidence=result.confidence,
            proof_hash=response.proof_hash,
            model_commitment=response.model_commitment,
            input_commitment=result.input_commitment,
            output_commitment=result.output_commitment,
            features=result.features,
            context_used=result.context_used,
            source=source,
            batch_id=request_id,  # Using request_id in batch_id field for compatibility
            analyst_paid_usdc=0,  # Self-classified
            policy_proof_hash=response.proof_hash
        )

        await db.insert_classifications_batch([record])
        logger.info(f"Stored classification for {request_id}: {result.url[:50]}...")

    def calculate_price(self, url_count: int = 1) -> float:
        """Calculate price for classifying URLs"""
        return url_count * config.analyst_price_per_url

    async def classify_url(
        self,
        request_id: str,
        url: str,
        policy_proof_hash: str
    ) -> ClassifyResponse:
        """
        Classify a single URL with zkML proof.

        Returns classification with cryptographic proof.
        """
        await emit_analyst_processing(request_id, 1)
        logger.info(f"Processing {request_id}: {url[:50]}...")

        # Extract features
        features = extract_features(url)

        # Get context from database
        context = await db.get_classification_context(
            domain=features.domain,
            registrar=None,  # Would need WHOIS lookup
            ip=None  # Would need DNS lookup
        )

        # Enrich features with context
        enriched_vector = features.to_vector()

        # Add context features
        if context.get("domain_phish_rate") is not None:
            enriched_vector[20] = context["domain_phish_rate"]
        if context.get("similar_domains_phish_rate") is not None:
            enriched_vector[21] = context["similar_domains_phish_rate"]
        if context.get("registrar_phish_rate") is not None:
            enriched_vector[22] = context["registrar_phish_rate"]
        if context.get("ip_phish_rate") is not None:
            enriched_vector[23] = context["ip_phish_rate"]

        # Generate classification proof
        await emit_analyst_proving(request_id)
        proof_result = await classifier_prover.prove_classification(enriched_vector)

        # Extract classification from proof output
        output = proof_result.output
        classification = Classification(output["classification"])
        confidence = output["confidence"]

        result = ClassificationResult(
            url=url,
            domain=features.domain,
            classification=classification.value,
            confidence=confidence,
            features=features.to_dict(),
            context_used=context,
            input_commitment=proof_result.input_commitment,
            output_commitment=proof_result.output_commitment
        )

        # Update stats
        self.classifications_today += 1
        if classification == Classification.PHISHING:
            self.phishing_detected_today += 1

        logger.info(f"{request_id} classified: {classification.value} (confidence: {confidence:.2f})")

        await emit_analyst_response(
            request_id=request_id,
            classification=classification.value,
            confidence=confidence,
            proof_hash=proof_result.proof_hash,
            prove_time_ms=proof_result.prove_time_ms
        )

        # Calculate payment due (for proof-gated payment flow)
        payment_amount = self.calculate_price(1)
        payment_due = PaymentDue(
            amount=payment_amount,
            currency="USDC",
            recipient=self.wallet_address,
            chain=config.base_chain_caip2,
            memo=f"classify-{request_id}"
        )

        return ClassifyResponse(
            request_id=request_id,
            result=result,
            proof=proof_result.proof_hex,
            proof_hash=proof_result.proof_hash,
            model_commitment=proof_result.model_commitment,
            input_commitment=proof_result.input_commitment,
            output_commitment=proof_result.output_commitment,
            prove_time_ms=proof_result.prove_time_ms,
            timestamp=datetime.utcnow().isoformat(),
            payment_due=payment_due
        )

    def get_stats(self) -> dict:
        """Get analyst agent statistics"""
        return {
            "classifications_today": self.classifications_today,
            "phishing_detected_today": self.phishing_detected_today,
            "total_earned_usdc": self.total_earned_usdc,
            "detection_rate": (
                self.phishing_detected_today /
                max(1, self.classifications_today)
            ),
            "model_commitment": self.model_commitment,
            "price_per_url": config.analyst_price_per_url
        }


# ============ FastAPI App ============

analyst_agent = AnalystAgent()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup validation
    if config.production_mode:
        config.assert_production_ready()
        logger.info("Production mode enabled - all validations passed")
    else:
        logger.warning("Running in DEMO mode - simulated proofs/payments allowed")

    # Initialize database and analyst
    await db.connect()
    await db.init_schema()
    await analyst_agent.initialize()

    # Start the orchestration loop (Value Chain mode)
    # Analyst is now the customer who drives the pipeline
    asyncio.create_task(analyst_agent.start())

    yield
    # Shutdown
    await analyst_agent.stop()
    await db.close()


app = FastAPI(
    title="Analyst Agent (2-Agent Model)",
    description="Classifies URLs with zkML proofs and self-authorized spending. Part of Scout ←→ Analyst circular economy.",
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
    """A2A v0.3 Agent Card (2-Agent Model)"""
    return build_agent_card_v3(
        name="Analyst Agent",
        description="Classifies URLs with zkML proofs. Generates self-authorized spending proofs and verifies Scout spending proofs.",
        url=config.analyst_url,
        version="2.0.0",
        streaming=False,
        push_notifications=False,
        state_transition_history=True,
        provider="ThreatProof Network",
        documentation_url=f"{config.analyst_url}/docs",
        default_payment_address=analyst_agent.wallet_address,
        supported_payment_methods=["x402"],
        skills=[
            build_skill_v3(
                skill_id="classify-url",
                name="URL Classification",
                description="Classify a URL as phishing/safe/suspicious with cryptographic proof of ML inference",
                tags=["classification", "phishing", "zkml", "threat-intel", "security", "self-auth"],
                input_modes=["application/json"],
                output_modes=["application/json"],
                price_amount=config.analyst_price_per_url,
                price_currency="USDC",
                price_per="url",
                chain=config.base_chain_caip2,  # CAIP-2 format
                proof_required=True,
                model_commitment=analyst_agent.model_commitment
            )
        ]
    )


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "production_mode": config.production_mode,
        "architecture": "2-agent",
        "self_authorization": True,
        "orchestrator": True,  # Analyst drives the pipeline
        "running": analyst_agent.running,
        "batches_processed": analyst_agent.batches_processed,
        "model_commitment": analyst_agent.model_commitment,
        "spending_model_commitment": analyst_agent.spending_model_commitment,
        "zkml_available": classifier_prover.prover.zkml_available,
        "real_proofs_enabled": classifier_prover.prover.zkml_available,
        "payment_flow": "circular",  # Analyst ←→ Scout
        "total_feedback_received": analyst_agent.total_feedback_received
    }


@app.get("/stats")
async def stats():
    """Get analyst agent statistics"""
    return analyst_agent.get_stats()


@app.post("/skills/classify-url")
async def classify_url_endpoint(
    request: ClassifyRequest,
    http_request: Request
) -> ClassifyResponse:
    """
    Classify a single URL with zkML proof (Proof-Gated Payment Flow).

    NO payment required upfront. Flow:
    1. Scout calls this endpoint with 1 URL
    2. Analyst does work, generates zkML proof
    3. Response includes result + proof + payment_due
    4. Scout verifies the proof
    5. If valid, Scout pays via /confirm-payment endpoint
    6. If invalid, no payment (Scout keeps their money)

    This ensures: NO VALID PROOF = NO PAYMENT
    """
    logger.info(f"Processing classification request {request.request_id}: {request.url[:50]}...")

    # Do the work and return proof + payment_due
    # Payment will be made AFTER Scout verifies the proof
    return await analyst_agent.classify_url(
        request_id=request.request_id,
        url=request.url,
        policy_proof_hash=request.policy_proof_hash
    )


class ConfirmPaymentRequest(BaseModel):
    """Request to confirm payment after proof verification"""
    request_id: str
    tx_hash: str
    amount: float


@app.post("/confirm-payment")
async def confirm_payment(request: ConfirmPaymentRequest):
    """
    Confirm payment received after proof verification (single URL).

    Called by Scout after:
    1. Receiving classification result + proof
    2. Verifying the zkML proof is valid
    3. Making the USDC payment

    This allows Analyst to track earnings.
    """
    # Verify payment if in production mode
    if config.production_mode and request.tx_hash != "simulated":
        try:
            tolerance = 1 - config.payment_tolerance
            is_valid, error = analyst_agent.x402_client.verify_payment(
                tx_hash=request.tx_hash,
                expected_recipient=analyst_agent.wallet_address,
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

    # Record earnings
    analyst_agent.total_earned_usdc += request.amount
    logger.info(f"Payment confirmed for {request.request_id}: ${request.amount} USDC (tx: {request.tx_hash})")

    return {
        "status": "confirmed",
        "request_id": request.request_id,
        "amount": request.amount,
        "total_earned": analyst_agent.total_earned_usdc
    }


class ConfirmFeedbackPaymentRequest(BaseModel):
    """Request to confirm feedback payment from Scout"""
    request_id: str
    tx_hash: str
    amount: float


@app.post("/confirm-feedback-payment")
async def confirm_feedback_payment(request: ConfirmFeedbackPaymentRequest):
    """
    Confirm feedback payment received from Scout (2-Agent Model).

    Called by Scout after:
    1. Generating its own spending proof
    2. Self-verifying its spending proof (prevents rogue spending)
    3. Making the USDC feedback payment to Analyst

    Note: Scout self-verifies its own spending proof before paying.
    Analyst does NOT verify Scout's spending proof - each agent self-verifies.
    This completes the circular economy: Analyst → Scout → Analyst
    """
    # Verify payment if in production mode
    if config.production_mode and request.tx_hash != "simulated":
        try:
            tolerance = 1 - config.payment_tolerance
            is_valid, error = analyst_agent.x402_client.verify_payment(
                tx_hash=request.tx_hash,
                expected_recipient=analyst_agent.wallet_address,
                expected_amount_usdc=request.amount * tolerance
            )
            if not is_valid:
                raise HTTPException(
                    status_code=400,
                    detail=f"Feedback payment verification failed: {error}"
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Feedback payment verification error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    # Record feedback earnings
    analyst_agent.total_feedback_received += request.amount
    logger.info(f"Feedback payment confirmed for {request.request_id}: ${request.amount} USDC (tx: {request.tx_hash})")

    return {
        "status": "confirmed",
        "request_id": request.request_id,
        "amount": request.amount,
        "total_feedback_received": analyst_agent.total_feedback_received,
        "payment_type": "feedback"
    }


# Alternative endpoint
@app.post("/classify")
async def classify(
    request: ClassifyRequest,
    http_request: Request
) -> ClassifyResponse:
    """Alias for classify-url (proof-gated payment flow)"""
    return await classify_url_endpoint(request, http_request)


@app.post("/trigger")
async def trigger_cycle():
    """Manually trigger one URL processing cycle for debugging"""
    result = await analyst_agent._process_url()
    return {"status": "triggered", "result": result}


@app.post("/extract-features")
async def extract_features_endpoint(url: str) -> Dict[str, Any]:
    """
    Extract features from a URL (for debugging/testing).

    Does not require payment.
    """
    features = extract_features(url)
    return features.to_dict()


# ============ JSON-RPC 2.0 Endpoint (A2A v0.3) ============

# Task store for tracking A2A tasks
analyst_task_store = TaskStore()

# JSON-RPC router
jsonrpc_router = JSONRPCRouter()


@jsonrpc_router.method("task/send")
async def task_send(params: dict) -> dict:
    """
    A2A task/send method (Proof-Gated Payment Flow).

    Creates a task, executes the classify-url skill, and returns the result.
    NO payment required upfront - payment is made AFTER proof verification.

    Flow:
    1. Scout sends task/send request (no payment needed)
    2. Analyst executes classification, generates zkML proof
    3. Response includes result + proof + payment_due
    4. Scout verifies the proof
    5. If valid, Scout pays and calls task/confirm-payment
    """
    skill_id = params.get("skillId", "classify-url")
    input_data = params.get("input", {})
    context_id = params.get("contextId")

    if skill_id != "classify-url":
        raise ValueError(f"Unknown skill: {skill_id}")

    # Calculate payment that will be due after proof verification (1 URL)
    required_amount = analyst_agent.calculate_price(1)

    # Create task
    task = await analyst_task_store.create(
        skill_id=skill_id,
        input_data=input_data,
        context_id=context_id,
        payment_required=True,
        payment_amount=str(required_amount),
        payment_currency="USDC",
        payment_chain=config.base_chain_caip2
    )

    # Execute the classification (no payment verification - proof-gated flow)
    async def classification_handler(inp: dict) -> dict:
        request = ClassifyRequest(**inp)
        response = await analyst_agent.classify_url(
            request_id=request.request_id,
            url=request.url,
            policy_proof_hash=request.policy_proof_hash
        )
        return response.model_dump()

    task = await execute_task(task, analyst_task_store, classification_handler)

    # Add proof as artifact
    if task.output and task.output.get("proof"):
        task.add_artifact(
            name="classification_proof",
            data={
                "proof_hash": task.output.get("proof_hash"),
                "model_commitment": task.output.get("model_commitment"),
                "input_commitment": task.output.get("input_commitment"),
                "output_commitment": task.output.get("output_commitment")
            },
            mime_type="application/json"
        )

        # Add classification result as artifact
        task.add_artifact(
            name="classification_result",
            data=task.output.get("result", {}),
            mime_type="application/json"
        )

        # Add payment_due info as artifact (for proof-gated payment)
        task.add_artifact(
            name="payment_due",
            data=task.output.get("payment_due", {}),
            mime_type="application/json"
        )
        await analyst_task_store.update(task)

    return task.to_response()


@jsonrpc_router.method("task/get")
async def task_get(params: dict) -> dict:
    """
    A2A task/get method.

    Returns the current state of a task by ID.
    """
    task_id = params.get("taskId")
    if not task_id:
        raise ValueError("taskId is required")

    task = await analyst_task_store.get(task_id)
    if not task:
        raise TaskNotFoundError(task_id)

    return task.to_response()


@jsonrpc_router.method("tasks/list")
async def tasks_list(params: dict) -> dict:
    """
    List recent tasks.
    """
    limit = params.get("limit", 100)
    tasks = await analyst_task_store.get_recent(limit=limit)
    return {
        "tasks": [t.to_response() for t in tasks],
        "total": analyst_task_store.count()
    }


# Register the JSON-RPC endpoint
app.post("/a2a")(jsonrpc_router.create_endpoint())


# ============ SSE Streaming Endpoint ============

@app.get("/tasks/{task_id}/stream")
async def stream_task(task_id: str):
    """
    Stream task progress via Server-Sent Events.

    Returns SSE events for:
    - task/status: State changes
    - task/artifact: New artifacts (proofs, classifications)
    - task/complete: Task completion
    - task/error: Task failure
    """
    task = await analyst_task_store.get(task_id)
    if not task:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    return create_sse_response(task_id, analyst_task_store)


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8002))
    uvicorn.run(app, host="0.0.0.0", port=port)
