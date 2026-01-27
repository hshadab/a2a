"""
Activity Model for comprehensive pipeline tracking.

Captures ALL pipeline events (URL discovery, authorization, proofs, verification,
classification, payments) with blockchain explorer links.
"""
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
import uuid


class ActivityCategory(str, Enum):
    """Categories for activity tracking"""
    DISCOVERY = "discovery"        # Scout found URLs
    AUTHORIZATION = "authorization" # Spending proofs
    CLASSIFICATION = "classification"
    PAYMENT = "payment"
    VERIFICATION = "verification"
    ERROR = "error"


class Activity(BaseModel):
    """
    Unified activity record for all pipeline events.

    Stored in Redis for persistence and served via /activities API.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    request_id: str
    event_type: str
    category: ActivityCategory
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    agent: str  # "scout" or "analyst"
    title: str
    description: str
    data: Dict[str, Any] = Field(default_factory=dict)

    # Payment fields
    tx_hash: Optional[str] = None
    amount_usdc: Optional[float] = None
    explorer_url: Optional[str] = None  # https://basescan.org/tx/{tx_hash}

    # Proof fields
    proof_hash: Optional[str] = None
    prove_time_ms: Optional[int] = None

    # Classification fields
    url: Optional[str] = None
    classification: Optional[str] = None
    confidence: Optional[float] = None


def get_explorer_url(tx_hash: str, testnet: bool = False) -> Optional[str]:
    """
    Get blockchain explorer URL for a transaction.

    Args:
        tx_hash: Transaction hash
        testnet: If True, use Sepolia testnet explorer

    Returns:
        Explorer URL or None if tx_hash is simulated/invalid
    """
    if not tx_hash or tx_hash == "simulated" or tx_hash.startswith("sim_"):
        return None

    base_url = "https://sepolia.basescan.org" if testnet else "https://basescan.org"
    prefixed = tx_hash if tx_hash.startswith("0x") else f"0x{tx_hash}"
    return f"{base_url}/tx/{prefixed}"


def event_to_activity(event_type: str, data: Dict[str, Any]) -> Activity:
    """
    Convert a Redis pub/sub event to an Activity record.

    Args:
        event_type: The EventType string (e.g., "SCOUT_FOUND_URLS")
        data: Event data dictionary

    Returns:
        Activity record ready for persistence
    """
    request_id = data.get("request_id", str(uuid.uuid4()))

    # Determine agent from event type
    if event_type.startswith("SCOUT"):
        agent = "scout"
    elif event_type.startswith("ANALYST"):
        agent = "analyst"
    else:
        agent = data.get("agent", "system")

    # Map event type to category, title, description
    category, title, description = _map_event(event_type, data, agent)

    # Build activity
    activity = Activity(
        request_id=request_id,
        event_type=event_type,
        category=category,
        agent=agent,
        title=title,
        description=description,
        data=data,
    )

    # Extract specific fields based on event type
    _extract_fields(activity, event_type, data)

    return activity


def _map_event(event_type: str, data: Dict[str, Any], agent: str) -> tuple:
    """Map event type to category, title, and description."""

    # Discovery events
    if event_type == "SCOUT_FOUND_URLS":
        url_count = data.get("url_count", 1)
        source = data.get("source", "unknown")
        title = f"Found {url_count} URL{'s' if url_count != 1 else ''}"
        description = f"Scout discovered {url_count} URL{'s' if url_count != 1 else ''} from {source}"
        return ActivityCategory.DISCOVERY, title, description

    # Authorization events
    if event_type == "SCOUT_AUTHORIZING":
        cost = data.get("estimated_cost", 0)
        title = "Authorizing spending"
        description = f"Scout generating spending proof for ${cost:.4f}"
        return ActivityCategory.AUTHORIZATION, title, description

    if event_type == "SCOUT_AUTHORIZED":
        confidence = data.get("confidence", 0) * 100
        title = "Spending authorized"
        description = f"Scout spending proof generated ({confidence:.0f}% confidence)"
        return ActivityCategory.AUTHORIZATION, title, description

    if event_type == "ANALYST_AUTHORIZING":
        cost = data.get("estimated_cost", 0)
        title = "Authorizing spending"
        description = f"Analyst generating spending proof for ${cost:.4f}"
        return ActivityCategory.AUTHORIZATION, title, description

    if event_type == "ANALYST_AUTHORIZED":
        confidence = data.get("confidence", 0) * 100
        title = "Spending authorized"
        description = f"Analyst spending proof generated ({confidence:.0f}% confidence)"
        return ActivityCategory.AUTHORIZATION, title, description

    # Verification events
    if event_type == "SPENDING_PROOF_VERIFIED":
        agent_name = data.get("agent", "Agent")
        valid = data.get("valid", True)
        verify_time = data.get("verify_time_ms", 0)
        if valid:
            title = "Spending proof verified"
            description = f"{agent_name.capitalize()} spending proof verified in {verify_time}ms"
        else:
            title = "Spending proof failed"
            description = f"{agent_name.capitalize()} spending proof verification failed"
        return ActivityCategory.VERIFICATION, title, description

    if event_type == "WORK_VERIFIED":
        valid = data.get("valid", True)
        verify_time = data.get("verify_time_ms", 0)
        quality_tier = data.get("quality_tier", "UNKNOWN")
        if valid:
            title = f"{quality_tier} quality verified"
            description = f"Work proof verified in {verify_time}ms"
        else:
            title = "Work verification failed"
            description = f"Work proof failed verification ({verify_time}ms)"
        return ActivityCategory.VERIFICATION, title, description

    # Payment events
    if event_type == "PAYMENT_SENDING":
        amount = data.get("amount_usdc", 0)
        recipient = data.get("recipient", "")[:10] + "..."
        title = f"Sending ${amount:.4f}"
        description = f"Payment of ${amount:.4f} USDC to {recipient}"
        return ActivityCategory.PAYMENT, title, description

    if event_type == "PAYMENT_SENT":
        amount = data.get("amount_usdc", 0)
        tx_hash = data.get("tx_hash", "")
        tx_short = tx_hash[:10] + "..." if tx_hash and tx_hash != "simulated" else "simulated"
        title = f"Paid ${amount:.4f}"
        description = f"Payment of ${amount:.4f} USDC confirmed (tx: {tx_short})"
        return ActivityCategory.PAYMENT, title, description

    # Classification events
    if event_type == "ANALYST_PROCESSING":
        url_count = data.get("url_count", 1)
        title = "Processing URL"
        description = f"Analyst processing {url_count} URL{'s' if url_count != 1 else ''}"
        return ActivityCategory.CLASSIFICATION, title, description

    if event_type == "ANALYST_PROVING":
        title = "Generating proof"
        description = "Analyst generating classification zkML proof"
        return ActivityCategory.CLASSIFICATION, title, description

    if event_type == "ANALYST_RESPONSE":
        classification = data.get("classification", "UNKNOWN")
        confidence = data.get("confidence", 0) * 100
        title = f"Classified: {classification}"
        description = f"URL classified as {classification} ({confidence:.0f}% confidence)"
        return ActivityCategory.CLASSIFICATION, title, description

    # Database events
    if event_type == "DATABASE_UPDATED":
        urls_added = data.get("urls_added", 0)
        total = data.get("total_urls", 0)
        title = f"Added {urls_added} URL{'s' if urls_added != 1 else ''}"
        description = f"Database updated: {urls_added} new, {total} total"
        return ActivityCategory.CLASSIFICATION, title, description

    # Error events
    if event_type == "ERROR":
        error = data.get("error", "Unknown error")
        title = "Error"
        description = error[:100]
        return ActivityCategory.ERROR, title, description

    # Default fallback
    title = event_type.replace("_", " ").title()
    description = f"Event: {event_type}"
    return ActivityCategory.CLASSIFICATION, title, description


def _extract_fields(activity: Activity, event_type: str, data: Dict[str, Any]):
    """Extract specific fields from event data into activity."""

    # Payment fields
    if event_type in ("PAYMENT_SENT", "PAYMENT_SENDING"):
        activity.tx_hash = data.get("tx_hash")
        activity.amount_usdc = data.get("amount_usdc")
        if activity.tx_hash:
            activity.explorer_url = get_explorer_url(activity.tx_hash)

    # Proof fields
    if "proof_hash" in data:
        activity.proof_hash = data.get("proof_hash")
    if "prove_time_ms" in data:
        activity.prove_time_ms = data.get("prove_time_ms")

    # Classification fields
    if event_type == "ANALYST_RESPONSE":
        activity.classification = data.get("classification")
        activity.confidence = data.get("confidence")
        activity.proof_hash = data.get("proof_hash")
        activity.prove_time_ms = data.get("prove_time_ms")

    # URL from various events
    if "url" in data:
        activity.url = data.get("url")
    elif "sample_urls" in data and data.get("sample_urls"):
        activity.url = data["sample_urls"][0]
