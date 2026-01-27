"""
Event Broadcasting System

Broadcasts events to connected WebSocket clients for real-time UI updates.
"""
import asyncio
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Set
from fastapi import WebSocket, WebSocketDisconnect
import redis.asyncio as redis

from .types import Event, EventType
from .config import config


class EventBroadcaster:
    """
    Broadcasts events to connected WebSocket clients.

    Supports both direct WebSocket connections and Redis pub/sub
    for distributed deployments.
    """

    def __init__(self, use_redis: bool = True):
        self.connections: Set[WebSocket] = set()
        self.use_redis = use_redis
        self._redis: Optional[redis.Redis] = None
        self._pubsub = None
        self._listener_task: Optional[asyncio.Task] = None

    async def connect(self, websocket: WebSocket):
        """Accept a new WebSocket connection"""
        await websocket.accept()
        self.connections.add(websocket)

    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        self.connections.discard(websocket)

    async def _get_redis(self) -> redis.Redis:
        """Get or create Redis connection"""
        if self._redis is None:
            self._redis = await redis.from_url(config.redis_url)
        return self._redis

    async def broadcast(self, event: Event):
        """
        Broadcast an event to all connected clients.

        If Redis is enabled, also publishes to Redis for distributed deployments.
        """
        message = event.model_dump_json()

        # Broadcast to local connections
        disconnected = set()
        for connection in self.connections:
            try:
                await connection.send_text(message)
            except Exception:
                disconnected.add(connection)

        # Clean up disconnected
        for conn in disconnected:
            self.connections.discard(conn)

        # Publish to Redis if enabled
        if self.use_redis:
            try:
                r = await self._get_redis()
                await r.publish("threat_intel_events", message)
            except Exception as e:
                print(f"Redis publish error: {e}")

    async def broadcast_dict(self, event_type: EventType, data: Dict[str, Any]):
        """Convenience method to broadcast from dict"""
        event = Event(type=event_type, data=data)
        await self.broadcast(event)

    async def start_redis_listener(self):
        """Start listening to Redis pub/sub for distributed events"""
        if not self.use_redis:
            return

        r = await self._get_redis()
        self._pubsub = r.pubsub()
        await self._pubsub.subscribe("threat_intel_events")

        async def listener():
            async for message in self._pubsub.listen():
                if message["type"] == "message":
                    data = message["data"]
                    if isinstance(data, bytes):
                        data = data.decode("utf-8")

                    # Broadcast to local connections (but don't re-publish to Redis)
                    disconnected = set()
                    for connection in self.connections:
                        try:
                            await connection.send_text(data)
                        except Exception:
                            disconnected.add(connection)

                    for conn in disconnected:
                        self.connections.discard(conn)

        self._listener_task = asyncio.create_task(listener())

    async def stop(self):
        """Clean up connections"""
        if self._listener_task:
            self._listener_task.cancel()
            try:
                await self._listener_task
            except asyncio.CancelledError:
                pass

        if self._pubsub:
            await self._pubsub.unsubscribe("threat_intel_events")

        if self._redis:
            await self._redis.close()

        for conn in self.connections:
            try:
                await conn.close()
            except Exception:
                pass


# Global broadcaster instance
broadcaster = EventBroadcaster()


# ============ Convenience functions for common events ============
# 2-Agent Model: Scout + Analyst with self-authorization (Per-URL)

async def emit_scout_found_urls(request_id: str, url_count: int, source: str, sample_urls: list = None):
    """Scout found URL(s) from source"""
    await broadcaster.broadcast_dict(
        EventType.SCOUT_FOUND_URLS,
        {
            "request_id": request_id,
            "url_count": url_count,
            "source": source,
            "sample_urls": sample_urls[:5] if sample_urls else []
        }
    )


async def emit_scout_authorizing(request_id: str, url_count: int, estimated_cost: float):
    """Scout is generating a spending authorization proof"""
    await broadcaster.broadcast_dict(
        EventType.SCOUT_AUTHORIZING,
        {
            "request_id": request_id,
            "url_count": url_count,
            "estimated_cost": estimated_cost
        }
    )


async def emit_scout_authorized(
    request_id: str,
    decision: str,
    confidence: float,
    proof_hash: str,
    prove_time_ms: int
):
    """Scout spending authorization proof generated"""
    await broadcaster.broadcast_dict(
        EventType.SCOUT_AUTHORIZED,
        {
            "request_id": request_id,
            "decision": decision,
            "confidence": confidence,
            "proof_hash": proof_hash,
            "prove_time_ms": prove_time_ms
        }
    )


async def emit_analyst_authorizing(request_id: str, estimated_cost: float):
    """Analyst is generating a spending authorization proof"""
    await broadcaster.broadcast_dict(
        EventType.ANALYST_AUTHORIZING,
        {
            "request_id": request_id,
            "estimated_cost": estimated_cost
        }
    )


async def emit_analyst_authorized(
    request_id: str,
    decision: str,
    confidence: float,
    proof_hash: str,
    prove_time_ms: int
):
    """Analyst spending authorization proof generated"""
    await broadcaster.broadcast_dict(
        EventType.ANALYST_AUTHORIZED,
        {
            "request_id": request_id,
            "decision": decision,
            "confidence": confidence,
            "proof_hash": proof_hash,
            "prove_time_ms": prove_time_ms
        }
    )


async def emit_spending_proof_verified(request_id: str, agent: str, valid: bool, verify_time_ms: int):
    """A spending proof was self-verified by the agent before spending"""
    await broadcaster.broadcast_dict(
        EventType.SPENDING_PROOF_VERIFIED,
        {
            "request_id": request_id,
            "agent": agent,  # "scout" or "analyst"
            "valid": valid,
            "verify_time_ms": verify_time_ms
        }
    )


async def emit_payment_sending(request_id: str, amount: float, recipient: str):
    """Payment is being sent"""
    await broadcaster.broadcast_dict(
        EventType.PAYMENT_SENDING,
        {
            "request_id": request_id,
            "amount_usdc": amount,
            "recipient": recipient
        }
    )


async def emit_payment_sent(request_id: str, tx_hash: str, amount: float):
    """Payment was sent successfully"""
    await broadcaster.broadcast_dict(
        EventType.PAYMENT_SENT,
        {
            "request_id": request_id,
            "tx_hash": tx_hash,
            "amount_usdc": amount
        }
    )


async def emit_analyst_processing(request_id: str, url_count: int, progress: Optional[int] = None):
    """Analyst is processing URL(s)"""
    await broadcaster.broadcast_dict(
        EventType.ANALYST_PROCESSING,
        {
            "request_id": request_id,
            "url_count": url_count,
            "progress": progress
        }
    )


async def emit_analyst_proving(request_id: str, progress: Optional[float] = None):
    """Analyst is generating classification proof"""
    await broadcaster.broadcast_dict(
        EventType.ANALYST_PROVING,
        {
            "request_id": request_id,
            "progress": progress
        }
    )


async def emit_analyst_response(
    request_id: str,
    classification: str = None,
    confidence: float = None,
    proof_hash: str = None,
    prove_time_ms: int = None,
    # Legacy batch fields (for backward compatibility)
    phishing_count: int = None,
    safe_count: int = None,
    suspicious_count: int = None,
    sample_results: list = None
):
    """Analyst classification complete"""
    data = {"request_id": request_id}

    # Single URL response
    if classification is not None:
        data["classification"] = classification
        data["confidence"] = confidence

    # Proof info
    if proof_hash is not None:
        data["proof_hash"] = proof_hash
    if prove_time_ms is not None:
        data["prove_time_ms"] = prove_time_ms

    # Legacy batch fields
    if phishing_count is not None:
        data["phishing_count"] = phishing_count
    if safe_count is not None:
        data["safe_count"] = safe_count
    if suspicious_count is not None:
        data["suspicious_count"] = suspicious_count
    if sample_results:
        data["sample_results"] = sample_results[:5]

    await broadcaster.broadcast_dict(EventType.ANALYST_RESPONSE, data)


async def emit_work_verified(
    request_id: str,
    valid: bool,
    verify_time_ms: int,
    quality_tier: str = None
):
    """Work proof was verified by the buyer"""
    data = {
        "request_id": request_id,
        "valid": valid,
        "verify_time_ms": verify_time_ms
    }
    if quality_tier:
        data["quality_tier"] = quality_tier

    await broadcaster.broadcast_dict(EventType.WORK_VERIFIED, data)


async def emit_database_updated(
    request_id: str,
    urls_added: int,
    total_urls: int,
    total_phishing: int,
    total_safe: int,
    total_suspicious: int
):
    """Database was updated with new classifications"""
    await broadcaster.broadcast_dict(
        EventType.DATABASE_UPDATED,
        {
            "request_id": request_id,
            "urls_added": urls_added,
            "total_urls": total_urls,
            "total_phishing": total_phishing,
            "total_safe": total_safe,
            "total_suspicious": total_suspicious
        }
    )


async def emit_error(request_id: Optional[str], error: str, details: Optional[str] = None):
    """Error occurred during processing"""
    await broadcaster.broadcast_dict(
        EventType.ERROR,
        {
            "request_id": request_id,
            "error": error,
            "details": details
        }
    )
