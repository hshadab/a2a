"""
Policy Agent

Authorizes spending on URL classification with zkML proofs.
Uses Jolt Atlas to generate cryptographic proofs that the authorization
decision was computed correctly.
"""
import asyncio
import os
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

import sys
sys.path.append('../..')

from shared.config import config
from shared.types import (
    AuthorizationRequest, AuthorizationResponse, PolicyDecision
)
from shared.events import (
    broadcaster,
    emit_policy_proving, emit_policy_response
)
from shared.a2a import build_agent_card, build_skill
from shared.prover import authorization_prover, compute_commitment
from shared.logging_config import policy_logger as logger


# ============ Request/Response Models ============

class AuthorizeRequest(BaseModel):
    batch_id: str
    url_count: int
    estimated_cost_usdc: float
    budget_remaining_usdc: float
    source_reputation: float
    novelty_score: float
    time_since_last_batch_seconds: int
    threat_level: float


class AuthorizeResponse(BaseModel):
    batch_id: str
    decision: str
    confidence: float
    proof: str
    proof_hash: str
    model_commitment: str
    input_commitment: str
    output_commitment: str
    prove_time_ms: int
    timestamp: str


# ============ Policy Agent ============

class PolicyAgent:
    """
    Policy Agent authorizes spending with zkML proofs.

    Decision factors:
    - Budget remaining vs estimated cost
    - Source reputation
    - URL novelty (prioritize new/unique URLs)
    - Time since last batch (prevent spam)
    - Current threat level
    """

    def __init__(self):
        self.authorizations_today = 0
        self.denials_today = 0
        self.total_authorized_usdc = 0.0
        self.model_commitment = None

        # Configurable thresholds
        self.min_budget_ratio = 0.1  # Deny if cost > 10% of remaining budget
        self.min_source_reputation = 0.3
        self.min_novelty_score = 0.1
        self.min_batch_interval = 60  # Minimum seconds between batches

    async def initialize(self):
        """Initialize the model commitment"""
        self.model_commitment = authorization_prover.prover.get_model_commitment(
            config.authorization_model_path
        )
        logger.info(f"Policy Agent initialized. Model commitment: {self.model_commitment[:16]}...")

    async def authorize(self, request: AuthorizeRequest) -> AuthorizeResponse:
        """
        Make an authorization decision with zkML proof.

        Returns decision (AUTHORIZED/DENIED) with proof.
        """
        await emit_policy_proving(request.batch_id)

        # Generate proof using Jolt Atlas
        proof_result = await authorization_prover.prove_authorization(
            batch_size=request.url_count,
            budget_remaining=request.budget_remaining_usdc,
            estimated_cost=request.estimated_cost_usdc,
            source_reputation=request.source_reputation,
            novelty_score=request.novelty_score,
            time_since_last=request.time_since_last_batch_seconds,
            threat_level=request.threat_level
        )

        # Extract decision from proof output
        output = proof_result.output
        decision = PolicyDecision(output["decision"])
        confidence = output["confidence"]

        # Update stats
        if decision == PolicyDecision.AUTHORIZED:
            self.authorizations_today += 1
            self.total_authorized_usdc += request.estimated_cost_usdc
            logger.info(f"Batch {request.batch_id} AUTHORIZED (confidence: {confidence:.2f})")
        else:
            self.denials_today += 1
            logger.info(f"Batch {request.batch_id} DENIED (confidence: {confidence:.2f})")

        response = AuthorizeResponse(
            batch_id=request.batch_id,
            decision=decision.value,
            confidence=confidence,
            proof=proof_result.proof_hex,
            proof_hash=proof_result.proof_hash,
            model_commitment=proof_result.model_commitment,
            input_commitment=proof_result.input_commitment,
            output_commitment=proof_result.output_commitment,
            prove_time_ms=proof_result.prove_time_ms,
            timestamp=datetime.utcnow().isoformat()
        )

        await emit_policy_response(
            batch_id=request.batch_id,
            decision=decision.value,
            confidence=confidence,
            proof_hash=proof_result.proof_hash,
            prove_time_ms=proof_result.prove_time_ms
        )

        return response

    def get_stats(self) -> dict:
        """Get policy agent statistics"""
        return {
            "authorizations_today": self.authorizations_today,
            "denials_today": self.denials_today,
            "total_authorized_usdc": self.total_authorized_usdc,
            "approval_rate": (
                self.authorizations_today /
                max(1, self.authorizations_today + self.denials_today)
            ),
            "model_commitment": self.model_commitment
        }


# ============ FastAPI App ============

policy_agent = PolicyAgent()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await policy_agent.initialize()
    yield
    # Shutdown


app = FastAPI(
    title="Policy Agent",
    description="Authorizes spending on URL classification with zkML proofs",
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
        name="Policy Agent",
        description="Authorizes spending on URL classification with zkML proofs",
        url=config.policy_url,
        skills=[
            build_skill(
                skill_id="authorize-batch",
                name="Batch Authorization",
                description="Authorize spending for URL classification batch",
                input_schema={
                    "type": "object",
                    "properties": {
                        "batch_id": {"type": "string"},
                        "url_count": {"type": "integer"},
                        "estimated_cost_usdc": {"type": "number"},
                        "budget_remaining_usdc": {"type": "number"},
                        "source_reputation": {"type": "number"},
                        "novelty_score": {"type": "number"},
                        "time_since_last_batch_seconds": {"type": "integer"},
                        "threat_level": {"type": "number"}
                    },
                    "required": [
                        "batch_id", "url_count", "estimated_cost_usdc",
                        "budget_remaining_usdc"
                    ]
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "batch_id": {"type": "string"},
                        "decision": {"type": "string", "enum": ["AUTHORIZED", "DENIED"]},
                        "confidence": {"type": "number"},
                        "proof": {"type": "string"},
                        "proof_hash": {"type": "string"},
                        "model_commitment": {"type": "string"},
                        "input_commitment": {"type": "string"},
                        "output_commitment": {"type": "string"},
                        "prove_time_ms": {"type": "integer"}
                    }
                },
                price_amount=config.policy_price_per_decision,
                price_currency="USDC",
                price_per="decision",
                proof_required=True,
                model_commitment=policy_agent.model_commitment
            )
        ]
    )


@app.get("/health")
async def health():
    import os
    zkml_path = os.environ.get('ZKML_CLI_PATH', 'zkml-cli')
    zkml_exists = os.path.isfile(zkml_path) if zkml_path.startswith('/') else False
    return {
        "status": "healthy",
        "model_commitment": policy_agent.model_commitment,
        "zkml_cli_path": zkml_path,
        "zkml_binary_exists": zkml_exists,
        "zkml_available": authorization_prover.prover.zkml_available,
        "jolt_model_dir": os.environ.get('JOLT_MODEL_DIR', 'not set')
    }


@app.get("/stats")
async def stats():
    """Get policy agent statistics"""
    return policy_agent.get_stats()


@app.get("/debug-prover")
async def debug_prover():
    """Debug endpoint to test prover directly"""
    import subprocess
    import os

    zkml_path = os.environ.get('ZKML_CLI_PATH', 'zkml-cli')
    jolt_model_dir = os.environ.get('JOLT_MODEL_DIR', '')
    model_path = os.path.join(jolt_model_dir, 'network.onnx') if jolt_model_dir else ''

    result = {
        "zkml_path": zkml_path,
        "zkml_exists": os.path.isfile(zkml_path),
        "zkml_executable": os.access(zkml_path, os.X_OK) if os.path.isfile(zkml_path) else False,
        "jolt_model_dir": jolt_model_dir,
        "model_path": model_path,
        "model_exists": os.path.isfile(model_path) if model_path else False,
    }

    # Try to run the binary with proper one-hot encoded inputs
    if result["zkml_exists"] and result["model_exists"]:
        try:
            # Build one-hot vector for: budget=15, trust=7, amount=8, category=0, velocity=2, day=1, time=1
            # From vocab.json: budget_15->15, trust_7->23, amount_8->32, category_0->40, velocity_2->46, day_1->53, time_1->61
            one_hot = [0] * 64
            one_hot[15] = 1   # budget_15
            one_hot[23] = 1   # trust_7
            one_hot[32] = 1   # amount_8
            one_hot[40] = 1   # category_0
            one_hot[46] = 1   # velocity_2
            one_hot[53] = 1   # day_1
            one_hot[61] = 1   # time_1
            test_inputs = [str(v) for v in one_hot]
            result["test_one_hot_indices"] = [i for i, v in enumerate(one_hot) if v == 1]

            cmd = [zkml_path, model_path] + test_inputs
            proc = subprocess.run(cmd, capture_output=True, timeout=180)
            result["test_returncode"] = proc.returncode
            result["test_stdout_full"] = proc.stdout.decode()
            result["test_stderr_last"] = proc.stderr.decode()[-4000:]  # Last 4000 chars of stderr
            result["test_stdout_len"] = len(proc.stdout)
            result["test_stderr_len"] = len(proc.stderr)
        except Exception as e:
            result["test_error"] = str(e)

    return result


@app.post("/skills/authorize-batch")
async def authorize_batch(request: AuthorizeRequest) -> AuthorizeResponse:
    """
    Authorize a batch of URLs for classification.

    Returns authorization decision with zkML proof.
    """
    return await policy_agent.authorize(request)


# Alternative endpoint for A2A compatibility
@app.post("/authorize")
async def authorize(request: AuthorizeRequest) -> AuthorizeResponse:
    """Alias for authorize-batch"""
    return await policy_agent.authorize(request)


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
