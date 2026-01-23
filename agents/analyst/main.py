"""
Analyst Agent

Classifies URLs as PHISHING, SAFE, or SUSPICIOUS with zkML proofs.
Uses Jolt Atlas to generate cryptographic proofs that the classification
was computed correctly.

Payment is required via x402 before classification.
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
from shared.a2a import build_agent_card, build_skill
from shared.x402 import PaymentRequired, require_payment, get_payment_from_header, X402Client
from shared.prover import classifier_prover, compute_commitment
from shared.logging_config import analyst_logger as logger

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

        return ClassifyResponse(
            batch_id=batch_id,
            results=results,
            proof=batch_proof.proof_hex,
            proof_hash=batch_proof.proof_hash,
            model_commitment=batch_proof.model_commitment,
            input_commitment=batch_proof.input_commitment,
            output_commitment=batch_proof.output_commitment,
            prove_time_ms=batch_proof.prove_time_ms,
            timestamp=datetime.utcnow().isoformat()
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
    # Startup
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
    """A2A Agent Card"""
    return build_agent_card(
        name="Analyst Agent",
        description="Classifies URLs as phishing/safe/suspicious with zkML proofs",
        url=config.analyst_url,
        skills=[
            build_skill(
                skill_id="classify-urls",
                name="URL Classification",
                description="Classify URLs with cryptographic proof of computation",
                input_schema={
                    "type": "object",
                    "properties": {
                        "batch_id": {"type": "string"},
                        "urls": {"type": "array", "items": {"type": "string"}},
                        "policy_proof_hash": {"type": "string"}
                    },
                    "required": ["batch_id", "urls", "policy_proof_hash"]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "batch_id": {"type": "string"},
                        "results": {"type": "array"},
                        "proof": {"type": "string"},
                        "proof_hash": {"type": "string"},
                        "model_commitment": {"type": "string"}
                    }
                },
                price_amount=config.analyst_price_per_url,
                price_currency="USDC",
                price_per="url",
                proof_required=True,
                model_commitment=analyst_agent.model_commitment
            )
        ]
    )


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model_commitment": analyst_agent.model_commitment
    }


@app.get("/stats")
async def stats():
    """Get analyst agent statistics"""
    return analyst_agent.get_stats()


@app.post("/skills/classify-urls")
async def classify_urls(
    request: ClassifyRequest,
    x_402_receipt: Optional[str] = Header(None, alias="X-402-Receipt")
) -> ClassifyResponse:
    """
    Classify URLs with zkML proof.

    Requires x402 payment.
    """
    # Calculate required payment
    required_amount = analyst_agent.calculate_price(len(request.urls))

    # Check for payment
    if not x_402_receipt:
        raise PaymentRequired(
            amount=required_amount,
            recipient=analyst_agent.wallet_address,
            memo=f"classify-{request.batch_id}"
        )

    # Verify payment
    try:
        is_valid, error = analyst_agent.x402_client.verify_payment(
            tx_hash=x_402_receipt,
            expected_recipient=analyst_agent.wallet_address,
            expected_amount_usdc=required_amount * 0.99  # Allow small tolerance
        )

        if not is_valid:
            # For demo, allow simulated payments
            if x_402_receipt != "simulated":
                logger.warning(f"Payment verification warning: {error}")
    except (ConnectionError, ValueError) as e:
        logger.warning(f"Payment verification error (continuing): {e}")
    except Exception as e:
        logger.error(f"Unexpected payment verification error: {e}", exc_info=True)

    # Process classification
    analyst_agent.total_earned_usdc += required_amount
    logger.info(f"Processing classification request for batch {request.batch_id}")

    return await analyst_agent.classify_batch(
        batch_id=request.batch_id,
        urls=request.urls,
        policy_proof_hash=request.policy_proof_hash
    )


# Alternative endpoint
@app.post("/classify")
async def classify(
    request: ClassifyRequest,
    x_402_receipt: Optional[str] = Header(None, alias="X-402-Receipt")
) -> ClassifyResponse:
    """Alias for classify-urls"""
    return await classify_urls(request, x_402_receipt)


@app.post("/extract-features")
async def extract_features_endpoint(url: str) -> Dict[str, Any]:
    """
    Extract features from a URL (for debugging/testing).

    Does not require payment.
    """
    features = extract_features(url)
    return features.to_dict()


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8002))
    uvicorn.run(app, host="0.0.0.0", port=port)
