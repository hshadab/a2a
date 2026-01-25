"""
A2A (Agent-to-Agent) Protocol Implementation

Based on Google's A2A specification for agent discovery and communication.
"""
import httpx
from typing import Optional, Dict, Any, List
from datetime import datetime
import json

from .types import AgentCard, SkillDefinition, AgentCardV3, AgentSkillV3, AgentCapabilitiesV3
from .config import config


class A2AClient:
    """
    Client for A2A agent discovery and skill invocation.
    """

    def __init__(self, timeout: float = 60.0):
        self.timeout = timeout
        self.known_agents: Dict[str, AgentCard] = {}
        self._agent_cache: Dict[str, datetime] = {}
        self.cache_ttl_seconds = 300  # 5 minutes

    async def get_agent_card(self, agent_url: str) -> AgentCard:
        """
        Fetch an agent's card from its well-known endpoint.
        """
        cache_key = agent_url
        now = datetime.utcnow()

        # Check cache
        if cache_key in self.known_agents:
            cached_time = self._agent_cache.get(cache_key)
            if cached_time and (now - cached_time).total_seconds() < self.cache_ttl_seconds:
                return self.known_agents[cache_key]

        # Fetch fresh
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{agent_url}/.well-known/agent.json")
            response.raise_for_status()
            data = response.json()

        card = AgentCard(**data)
        self.known_agents[cache_key] = card
        self._agent_cache[cache_key] = now

        return card

    async def discover_by_skill(self, skill_id: str, registry_url: Optional[str] = None) -> Optional[AgentCard]:
        """
        Discover an agent that provides a specific skill.

        If registry_url is provided, queries the registry.
        Otherwise, searches known agents.
        """
        if registry_url:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(f"{registry_url}/discover/{skill_id}")
                if response.status_code == 200:
                    data = response.json()
                    if data:
                        return AgentCard(**data)
                return None

        # Search known agents
        for agent in self.known_agents.values():
            skills = agent.capabilities.get("skills", [])
            for skill in skills:
                if skill.get("id") == skill_id:
                    return agent

        return None

    async def invoke_skill(
        self,
        agent_url: str,
        skill_id: str,
        input_data: Dict[str, Any],
        payment_receipt: Optional[str] = None,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Invoke a skill on an agent.

        Args:
            agent_url: Base URL of the agent
            skill_id: ID of the skill to invoke
            input_data: Input data for the skill
            payment_receipt: Optional x402 payment tx hash
            timeout: Optional timeout override

        Returns:
            Skill output data

        Raises:
            httpx.HTTPStatusError: If the request fails
            PaymentRequiredError: If payment is required (402)
        """
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json"
        }

        if payment_receipt:
            headers["X-402-Receipt"] = payment_receipt

        async with httpx.AsyncClient(timeout=timeout or self.timeout) as client:
            response = await client.post(
                f"{agent_url}/skills/{skill_id}",
                json=input_data,
                headers=headers
            )

            if response.status_code == 402:
                # Payment required
                payment_info = response.headers.get("X-402-Payment")
                raise PaymentRequiredError(
                    payment_info=json.loads(payment_info) if payment_info else None,
                    response=response
                )

            response.raise_for_status()
            return response.json()

    async def health_check(self, agent_url: str) -> bool:
        """Check if an agent is healthy"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{agent_url}/health")
                return response.status_code == 200
        except Exception:
            return False

    def register_agent(self, agent_url: str, card: AgentCard):
        """Manually register an agent"""
        self.known_agents[agent_url] = card
        self._agent_cache[agent_url] = datetime.utcnow()


class PaymentRequiredError(Exception):
    """Raised when a skill invocation requires payment"""

    def __init__(self, payment_info: Optional[Dict[str, Any]], response: httpx.Response):
        self.payment_info = payment_info
        self.response = response
        super().__init__(f"Payment required: {payment_info}")


# ============ Agent Card Builder ============

def build_agent_card(
    name: str,
    description: str,
    url: str,
    skills: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Build an A2A agent card (legacy v1.0).

    Returns a dict suitable for JSON serialization.
    """
    return {
        "schema_version": "1.0",
        "name": name,
        "description": description,
        "url": url,
        "capabilities": {
            "skills": skills
        }
    }


def build_agent_card_v3(
    name: str,
    description: str,
    url: str,
    skills: List[Dict[str, Any]],
    version: str = "1.0.0",
    streaming: bool = False,
    push_notifications: bool = False,
    state_transition_history: bool = True,
    provider: str = "ThreatProof Network",
    documentation_url: Optional[str] = None,
    default_payment_address: Optional[str] = None,
    supported_payment_methods: List[str] = None
) -> Dict[str, Any]:
    """
    Build an A2A v0.3 compliant agent card.

    Returns a dict suitable for JSON serialization with:
    - protocolVersion: "0.3"
    - capabilities object with streaming flags
    - skills with inputModes, outputModes, and x402 pricing
    """
    if supported_payment_methods is None:
        supported_payment_methods = ["x402"]

    return {
        "name": name,
        "description": description,
        "url": url,
        "version": version,
        "protocolVersion": "0.3",
        "capabilities": {
            "streaming": streaming,
            "pushNotifications": push_notifications,
            "stateTransitionHistory": state_transition_history
        },
        "authentication": {
            "schemes": ["none"]
        },
        "skills": skills,
        "provider": provider,
        "documentationUrl": documentation_url,
        "defaultPaymentAddress": default_payment_address,
        "supportedPaymentMethods": supported_payment_methods
    }


def build_skill(
    skill_id: str,
    name: str,
    description: str,
    input_schema: Dict[str, Any],
    output_schema: Dict[str, Any],
    price_amount: float,
    price_currency: str = "USDC",
    price_per: str = "call",
    proof_required: bool = False,
    model_commitment: Optional[str] = None
) -> Dict[str, Any]:
    """Build a skill definition for an agent card (legacy format)"""
    skill = {
        "id": skill_id,
        "name": name,
        "description": description,
        "input_schema": input_schema,
        "output_schema": output_schema,
        "price": {
            "amount": str(price_amount),
            "currency": price_currency,
            "per": price_per
        }
    }

    if proof_required:
        skill["proof_policy"] = {
            "proof_required": True,
            "proof_type": "zkml",
            "model_commitment": model_commitment
        }

    return skill


def build_skill_v3(
    skill_id: str,
    name: str,
    description: str,
    tags: List[str] = None,
    input_modes: List[str] = None,
    output_modes: List[str] = None,
    price_amount: float = 0,
    price_currency: str = "USDC",
    price_per: str = "call",
    chain: str = "eip155:8453",
    proof_required: bool = False,
    model_commitment: Optional[str] = None
) -> Dict[str, Any]:
    """
    Build an A2A v0.3 skill definition.

    Features:
    - inputModes/outputModes for content negotiation
    - CAIP-2 chain identifier in price
    - Tags for skill discovery
    """
    if tags is None:
        tags = []
    if input_modes is None:
        input_modes = ["application/json"]
    if output_modes is None:
        output_modes = ["application/json"]

    skill = {
        "id": skill_id,
        "name": name,
        "description": description,
        "tags": tags,
        "inputModes": input_modes,
        "outputModes": output_modes,
    }

    # x402 price extension with CAIP-2 chain
    if price_amount > 0:
        skill["price"] = {
            "amount": str(price_amount),
            "currency": price_currency,
            "per": price_per,
            "chain": chain  # CAIP-2 format
        }

    # zkML proof extension
    if proof_required:
        skill["proofPolicy"] = {
            "required": True,
            "type": "zkml",
            "modelCommitment": model_commitment
        }

    return skill


# ============ Pre-configured clients for our agents ============

async def get_policy_agent() -> AgentCard:
    """Get the Policy Agent card"""
    client = A2AClient()
    return await client.get_agent_card(config.policy_url)


async def get_analyst_agent() -> AgentCard:
    """Get the Analyst Agent card"""
    client = A2AClient()
    return await client.get_agent_card(config.analyst_url)


async def invoke_policy_authorization(
    batch_id: str,
    url_count: int,
    estimated_cost: float,
    budget_remaining: float,
    source_reputation: float,
    novelty_score: float,
    time_since_last: int,
    threat_level: float,
    payment_receipt: Optional[str] = None
) -> Dict[str, Any]:
    """Convenience function to invoke Policy Agent authorization"""
    client = A2AClient()
    return await client.invoke_skill(
        agent_url=config.policy_url,
        skill_id="authorize-batch",
        input_data={
            "batch_id": batch_id,
            "url_count": url_count,
            "estimated_cost_usdc": estimated_cost,
            "budget_remaining_usdc": budget_remaining,
            "source_reputation": source_reputation,
            "novelty_score": novelty_score,
            "time_since_last_batch_seconds": time_since_last,
            "threat_level": threat_level
        },
        payment_receipt=payment_receipt,
        timeout=120  # Proof generation can take time
    )


async def invoke_analyst_classification(
    batch_id: str,
    urls: List[str],
    policy_proof_hash: str,
    payment_receipt: Optional[str] = None
) -> Dict[str, Any]:
    """Convenience function to invoke Analyst Agent classification"""
    client = A2AClient()
    return await client.invoke_skill(
        agent_url=config.analyst_url,
        skill_id="classify-urls",
        input_data={
            "batch_id": batch_id,
            "urls": urls,
            "policy_proof_hash": policy_proof_hash
        },
        payment_receipt=payment_receipt,
        timeout=300  # Classification + proof generation for many URLs
    )
