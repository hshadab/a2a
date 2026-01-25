"""
Analyst Agent

Classifies URLs as PHISHING, SAFE, or SUSPICIOUS with zkML proofs.
Uses Jolt Atlas to generate cryptographic proofs that the classification
was computed correctly.

Payment Flow (Proof-Gated):
1. Scout requests classification (no payment required upfront)
2. Analyst does work, generates zkML proof
3. Analyst returns results + proof + payment_due
4. Scout verifies the proof
5. If valid, Scout pays Analyst
6. If invalid, no payment

This ensures: NO VALID PROOF = NO PAYMENT
"""
import asyncio
import os
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
    emit_analyst_processing, emit_analyst_proving, emit_analyst_response
)
from shared.a2a import build_agent_card, build_skill, build_agent_card_v3, build_skill_v3
from shared.x402 import (
    PaymentRequired, require_payment, get_payment_from_header, X402Client,
    HEADER_PAYMENT_SIGNATURE, HEADER_X402_RECEIPT
)
from shared.prover import classifier_prover, compute_commitment
from shared.logging_config import analyst_logger as logger
from shared.jsonrpc import JSONRPCRouter, create_jsonrpc_endpoint, TaskNotFoundError, PaymentRequiredError as JSONRPCPaymentRequired
from shared.task import TaskStore, Task, TaskState, execute_task
from shared.sse import create_sse_response

from features import extract_features, URLFeatures


# ============ Request/Response Models ============

class ClassifyRequest(BaseModel):
    batch_id: str
    urls: List[str]
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
    batch_id: str
    results: List[ClassificationResult]
    proof: str
    proof_hash: str
    model_commitment: str
    input_commitment: str
    output_commitment: str
    prove_time_ms: int
    timestamp: str
    # Payment due after proof verification (proof-gated flow)
    payment_due: Optional[PaymentDue] = None


# ============ Analyst Agent ============

class AnalystAgent:
    """
    Analyst Agent classifies URLs with zkML proofs.

    Uses:
    - URL features (domain, path, TLD, etc.)
    - Database context (historical phishing rates)
    - zkML proof generation via Jolt Atlas
    """

    def __init__(self):
        self.classifications_today = 0
        self.phishing_detected_today = 0
        self.total_earned_usdc = 0.0
        self.model_commitment = None
        self.x402_client = X402Client()

        # Wallet address for receiving payments
        self.wallet_address = config.treasury_address

    async def initialize(self):
        """Initialize the model commitment"""
        self.model_commitment = classifier_prover.prover.get_model_commitment(
            config.classifier_model_path
        )
        logger.info(f"Analyst Agent initialized. Model commitment: {self.model_commitment[:16]}...")

    def calculate_price(self, url_count: int) -> float:
        """Calculate price for classifying URLs"""
        return url_count * config.analyst_price_per_url

    async def classify_batch(
        self,
        batch_id: str,
        urls: List[str],
        policy_proof_hash: str
    ) -> ClassifyResponse:
        """
        Classify a batch of URLs with zkML proof.

        Returns classifications with cryptographic proof.
        """
        await emit_analyst_processing(batch_id, len(urls))
        logger.info(f"Processing batch {batch_id}: {len(urls)} URLs")

        results = []
        all_features = []

        # Process each URL
        for i, url in enumerate(urls):
            # Extract features
            features = extract_features(url)
            all_features.append(features)

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

            # Generate individual classification proof
            proof_result = await classifier_prover.prove_classification(enriched_vector)

            # Extract classification from proof output
            output = proof_result.output
            classification = Classification(output["classification"])
            confidence = output["confidence"]

            results.append(ClassificationResult(
                url=url,
                domain=features.domain,
                classification=classification.value,
                confidence=confidence,
                features=features.to_dict(),
                context_used=context,
                input_commitment=proof_result.input_commitment,
                output_commitment=proof_result.output_commitment
            ))

            # Update progress
            if (i + 1) % 10 == 0:
                await emit_analyst_processing(batch_id, len(urls), i + 1)
                logger.debug(f"Batch {batch_id}: processed {i + 1}/{len(urls)} URLs")

        # Generate batch proof
        await emit_analyst_proving(batch_id)

        batch_features = [f.to_vector() for f in all_features]
        _, batch_proof = await classifier_prover.prove_batch_classification(batch_features)

        # Update stats
        self.classifications_today += len(urls)
        self.phishing_detected_today += sum(
            1 for r in results if r.classification == Classification.PHISHING.value
        )

        # Count results
        phishing_count = sum(1 for r in results if r.classification == "PHISHING")
        safe_count = sum(1 for r in results if r.classification == "SAFE")
        suspicious_count = sum(1 for r in results if r.classification == "SUSPICIOUS")

        logger.info(f"Batch {batch_id} classified: {phishing_count} phishing, {safe_count} safe, {suspicious_count} suspicious")

        await emit_analyst_response(
            batch_id=batch_id,
            phishing_count=phishing_count,
            safe_count=safe_count,
            suspicious_count=suspicious_count,
            proof_hash=batch_proof.proof_hash,
            prove_time_ms=batch_proof.prove_time_ms
        )

        # Calculate payment due (for proof-gated payment flow)
        payment_amount = self.calculate_price(len(urls))
        payment_due = PaymentDue(
            amount=payment_amount,
            currency="USDC",
            recipient=self.wallet_address,
            chain=config.base_chain_caip2,
            memo=f"classify-{batch_id}"
        )

        return ClassifyResponse(
            batch_id=batch_id,
            results=results,
            proof=batch_proof.proof_hex,
            proof_hash=batch_proof.proof_hash,
            model_commitment=batch_proof.model_commitment,
            input_commitment=batch_proof.input_commitment,
            output_commitment=batch_proof.output_commitment,
            prove_time_ms=batch_proof.prove_time_ms,
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
    await analyst_agent.initialize()
    yield
    # Shutdown
    await db.close()


app = FastAPI(
    title="Analyst Agent",
    description="Classifies URLs with zkML proofs",
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
    """A2A v0.3 Agent Card"""
    return build_agent_card_v3(
        name="Analyst Agent",
        description="Classifies URLs as phishing/safe/suspicious with zkML proofs",
        url=config.analyst_url,
        version="1.0.0",
        streaming=False,
        push_notifications=False,
        state_transition_history=True,
        provider="ThreatProof Network",
        documentation_url=f"{config.analyst_url}/docs",
        default_payment_address=analyst_agent.wallet_address,
        supported_payment_methods=["x402"],
        skills=[
            build_skill_v3(
                skill_id="classify-urls",
                name="URL Classification",
                description="Classify URLs as phishing/safe/suspicious with cryptographic proof of ML inference",
                tags=["classification", "phishing", "zkml", "threat-intel", "security"],
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
        "model_commitment": analyst_agent.model_commitment,
        "zkml_available": classifier_prover.prover.zkml_available,
        "real_proofs_enabled": classifier_prover.prover.zkml_available,
        "payment_flow": "proof-gated"
    }


@app.get("/stats")
async def stats():
    """Get analyst agent statistics"""
    return analyst_agent.get_stats()


@app.post("/skills/classify-urls")
async def classify_urls(
    request: ClassifyRequest,
    http_request: Request
) -> ClassifyResponse:
    """
    Classify URLs with zkML proof (Proof-Gated Payment Flow).

    NO payment required upfront. Flow:
    1. Scout calls this endpoint
    2. Analyst does work, generates zkML proof
    3. Response includes results + proof + payment_due
    4. Scout verifies the proof
    5. If valid, Scout pays via /confirm-payment endpoint
    6. If invalid, no payment (Scout keeps their money)

    This ensures: NO VALID PROOF = NO PAYMENT
    """
    logger.info(f"Processing classification request for batch {request.batch_id} ({len(request.urls)} URLs)")

    # Do the work and return proof + payment_due
    # Payment will be made AFTER Scout verifies the proof
    return await analyst_agent.classify_batch(
        batch_id=request.batch_id,
        urls=request.urls,
        policy_proof_hash=request.policy_proof_hash
    )


class ConfirmPaymentRequest(BaseModel):
    """Request to confirm payment after proof verification"""
    batch_id: str
    tx_hash: str
    amount: float


@app.post("/confirm-payment")
async def confirm_payment(request: ConfirmPaymentRequest):
    """
    Confirm payment received after proof verification.

    Called by Scout after:
    1. Receiving classification results + proof
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
    logger.info(f"Payment confirmed for batch {request.batch_id}: ${request.amount} USDC (tx: {request.tx_hash})")

    return {
        "status": "confirmed",
        "batch_id": request.batch_id,
        "amount": request.amount,
        "total_earned": analyst_agent.total_earned_usdc
    }


# Alternative endpoint
@app.post("/classify")
async def classify(
    request: ClassifyRequest,
    http_request: Request
) -> ClassifyResponse:
    """Alias for classify-urls (proof-gated payment flow)"""
    return await classify_urls(request, http_request)


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

    Creates a task, executes the classify-urls skill, and returns the result.
    NO payment required upfront - payment is made AFTER proof verification.

    Flow:
    1. Scout sends task/send request (no payment needed)
    2. Analyst executes classification, generates zkML proof
    3. Response includes results + proof + payment_due
    4. Scout verifies the proof
    5. If valid, Scout pays and calls task/confirm-payment
    """
    skill_id = params.get("skillId", "classify-urls")
    input_data = params.get("input", {})
    context_id = params.get("contextId")

    if skill_id != "classify-urls":
        raise ValueError(f"Unknown skill: {skill_id}")

    # Calculate payment that will be due after proof verification
    urls = input_data.get("urls", [])
    required_amount = analyst_agent.calculate_price(len(urls))

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
        response = await analyst_agent.classify_batch(
            batch_id=request.batch_id,
            urls=request.urls,
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

        # Add individual classification results as artifact
        task.add_artifact(
            name="classification_results",
            data=task.output.get("results", []),
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
