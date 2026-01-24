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

async def emit_scout_found_urls(batch_id: str, url_count: int, source: str, sample_urls: list = None):
    await broadcaster.broadcast_dict(
        EventType.SCOUT_FOUND_URLS,
        {
            "batch_id": batch_id,
            "url_count": url_count,
            "source": source,
            "sample_urls": sample_urls[:5] if sample_urls else []  # Show first 5 URLs
        }
    )


async def emit_policy_requesting(batch_id: str, url_count: int, estimated_cost: float):
    await broadcaster.broadcast_dict(
        EventType.POLICY_REQUESTING,
        {
            "batch_id": batch_id,
            "url_count": url_count,
            "estimated_cost_usdc": estimated_cost
        }
    )


async def emit_policy_proving(batch_id: str, progress: Optional[float] = None):
    await broadcaster.broadcast_dict(
        EventType.POLICY_PROVING,
        {
            "batch_id": batch_id,
            "progress": progress
        }
    )


async def emit_policy_response(
    batch_id: str,
    decision: str,
    confidence: float,
    proof_hash: str,
    prove_time_ms: int
):
    await broadcaster.broadcast_dict(
        EventType.POLICY_RESPONSE,
        {
            "batch_id": batch_id,
            "decision": decision,
            "confidence": confidence,
            "proof_hash": proof_hash,
            "prove_time_ms": prove_time_ms
        }
    )


async def emit_policy_verified(batch_id: str, valid: bool, verify_time_ms: int):
    await broadcaster.broadcast_dict(
        EventType.POLICY_VERIFIED,
        {
            "batch_id": batch_id,
            "valid": valid,
            "verify_time_ms": verify_time_ms
        }
    )


async def emit_payment_sending(batch_id: str, amount: float, recipient: str):
    await broadcaster.broadcast_dict(
        EventType.PAYMENT_SENDING,
        {
            "batch_id": batch_id,
            "amount_usdc": amount,
            "recipient": recipient
        }
    )


async def emit_payment_sent(batch_id: str, tx_hash: str, amount: float):
    await broadcaster.broadcast_dict(
        EventType.PAYMENT_SENT,
        {
            "batch_id": batch_id,
            "tx_hash": tx_hash,
            "amount_usdc": amount
        }
    )


async def emit_analyst_processing(batch_id: str, url_count: int, progress: Optional[int] = None):
    await broadcaster.broadcast_dict(
        EventType.ANALYST_PROCESSING,
        {
            "batch_id": batch_id,
            "url_count": url_count,
            "progress": progress
        }
    )


async def emit_analyst_proving(batch_id: str, progress: Optional[float] = None):
    await broadcaster.broadcast_dict(
        EventType.ANALYST_PROVING,
        {
            "batch_id": batch_id,
            "progress": progress
        }
    )


async def emit_analyst_response(
    batch_id: str,
    phishing_count: int,
    safe_count: int,
    suspicious_count: int,
    proof_hash: str,
    prove_time_ms: int,
    sample_results: list = None
):
    await broadcaster.broadcast_dict(
        EventType.ANALYST_RESPONSE,
        {
            "batch_id": batch_id,
            "phishing_count": phishing_count,
            "safe_count": safe_count,
            "suspicious_count": suspicious_count,
            "proof_hash": proof_hash,
            "prove_time_ms": prove_time_ms,
            "sample_results": sample_results[:5] if sample_results else []  # Show first 5 classified URLs
        }
    )


async def emit_work_verified(batch_id: str, valid: bool, verify_time_ms: int):
    await broadcaster.broadcast_dict(
        EventType.WORK_VERIFIED,
        {
            "batch_id": batch_id,
            "valid": valid,
            "verify_time_ms": verify_time_ms
        }
    )


async def emit_database_updated(
    batch_id: str,
    urls_added: int,
    total_urls: int,
    total_phishing: int,
    total_safe: int,
    total_suspicious: int
):
    await broadcaster.broadcast_dict(
        EventType.DATABASE_UPDATED,
        {
            "batch_id": batch_id,
            "urls_added": urls_added,
            "total_urls": total_urls,
            "total_phishing": total_phishing,
            "total_safe": total_safe,
            "total_suspicious": total_suspicious
        }
    )


async def emit_error(batch_id: Optional[str], error: str, details: Optional[str] = None):
    await broadcaster.broadcast_dict(
        EventType.ERROR,
        {
            "batch_id": batch_id,
            "error": error,
            "details": details
        }
    )
