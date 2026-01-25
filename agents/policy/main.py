"""
Policy Agent

Authorizes spending on URL classification with zkML proofs.
Uses Jolt Atlas to generate cryptographic proofs that the authorization
decision was computed correctly.

Model version: 2026-01-24 (64->16->4 architecture for MAX_TENSOR_SIZE=1024)
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
from shared.a2a import build_agent_card, build_skill, build_agent_card_v3, build_skill_v3
from shared.prover import authorization_prover, compute_commitment
from shared.logging_config import policy_logger as logger
from shared.jsonrpc import JSONRPCRouter, create_jsonrpc_endpoint, TaskNotFoundError
from shared.task import TaskStore, Task, TaskState, execute_task
from shared.sse import create_sse_response


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
    """A2A v0.3 Agent Card"""
    return build_agent_card_v3(
        name="Policy Agent",
        description="Authorizes spending on URL classification with zkML proofs",
        url=config.policy_url,
        version="1.0.0",
        streaming=False,
        push_notifications=False,
        state_transition_history=True,
        provider="ThreatProof Network",
        documentation_url=f"{config.policy_url}/docs",
        default_payment_address=config.treasury_address,
        supported_payment_methods=["x402"],
        skills=[
            build_skill_v3(
                skill_id="authorize-batch",
                name="Batch Authorization",
                description="Authorize spending for URL classification batch with zkML proof of correct policy evaluation",
                tags=["authorization", "policy", "zkml", "budget"],
                input_modes=["application/json"],
                output_modes=["application/json"],
                price_amount=config.policy_price_per_decision,
                price_currency="USDC",
                price_per="decision",
                chain=config.base_chain_caip2,  # CAIP-2 format
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
    import hashlib

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

    # Get model hash to verify it's the new small model
    if result["model_exists"]:
        with open(model_path, 'rb') as f:
            result["model_hash"] = hashlib.sha256(f.read()).hexdigest()
        result["expected_hash"] = "15cc010f93ca0733ab186593161c350a020bc9d49125e065d3a8632d95efeade"
        result["is_new_model"] = result["model_hash"] == result["expected_hash"]

    # Test the new smaller authorization model with one-hot inputs
    if result["zkml_exists"] and result.get("is_new_model"):
        try:
            # Build one-hot vector (should work with new 64->16->4 model)
            one_hot = [0] * 64
            one_hot[15] = 1   # budget_15
            one_hot[23] = 1   # trust_7
            one_hot[30] = 1   # amount_6
            one_hot[41] = 1   # category_1
            one_hot[46] = 1   # velocity_2
            one_hot[54] = 1   # day_2
            one_hot[62] = 1   # time_2
            test_inputs = [str(v) for v in one_hot]

            cmd = [zkml_path, model_path] + test_inputs
            result["command"] = f"{zkml_path} {model_path} " + " ".join(test_inputs[:10]) + " ..."

            proc = subprocess.run(cmd, capture_output=True, timeout=120)
            result["auth_model_test"] = {
                "returncode": proc.returncode,
                "success": proc.returncode == 0,
                "stdout_preview": proc.stdout.decode()[:500] if proc.stdout else "",
                "stderr_last": proc.stderr.decode()[-500:] if proc.stderr else ""
            }
        except subprocess.TimeoutExpired:
            result["auth_model_test_error"] = "Timeout after 120s"
        except Exception as e:
            result["auth_model_test_error"] = str(e)
    elif result["zkml_exists"] and not result.get("is_new_model"):
        result["auth_model_test_skipped"] = "Model is not the new smaller version"

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


# ============ JSON-RPC 2.0 Endpoint (A2A v0.3) ============

# Task store for tracking A2A tasks
policy_task_store = TaskStore()

# JSON-RPC router
jsonrpc_router = JSONRPCRouter()


@jsonrpc_router.method("task/send")
async def task_send(params: dict) -> dict:
    """
    A2A task/send method.

    Creates a task, executes the authorize-batch skill, and returns the result.
    """
    skill_id = params.get("skillId", "authorize-batch")
    input_data = params.get("input", {})
    context_id = params.get("contextId")

    if skill_id != "authorize-batch":
        raise ValueError(f"Unknown skill: {skill_id}")

    # Create task
    task = await policy_task_store.create(
        skill_id=skill_id,
        input_data=input_data,
        context_id=context_id,
        payment_required=True,
        payment_amount=str(config.policy_price_per_decision),
        payment_currency="USDC",
        payment_chain=config.base_chain_caip2
    )

    # Execute the authorization
    async def authorization_handler(inp: dict) -> dict:
        request = AuthorizeRequest(**inp)
        response = await policy_agent.authorize(request)
        return response.model_dump()

    task = await execute_task(task, policy_task_store, authorization_handler)

    # Add proof as artifact
    if task.output and task.output.get("proof"):
        task.add_artifact(
            name="authorization_proof",
            data={
                "proof_hash": task.output.get("proof_hash"),
                "model_commitment": task.output.get("model_commitment"),
                "input_commitment": task.output.get("input_commitment"),
                "output_commitment": task.output.get("output_commitment")
            },
            mime_type="application/json"
        )
        await policy_task_store.update(task)

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

    task = await policy_task_store.get(task_id)
    if not task:
        raise TaskNotFoundError(task_id)

    return task.to_response()


@jsonrpc_router.method("tasks/list")
async def tasks_list(params: dict) -> dict:
    """
    List recent tasks.
    """
    limit = params.get("limit", 100)
    tasks = await policy_task_store.get_recent(limit=limit)
    return {
        "tasks": [t.to_response() for t in tasks],
        "total": policy_task_store.count()
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
    - task/artifact: New artifacts
    - task/complete: Task completion
    - task/error: Task failure
    """
    task = await policy_task_store.get(task_id)
    if not task:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")

    return create_sse_response(task_id, policy_task_store)


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8001))
    uvicorn.run(app, host="0.0.0.0", port=port)
