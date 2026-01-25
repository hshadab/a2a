"""
Scout Agent

Discovers URLs to analyze and orchestrates the classification pipeline.
Runs continuously, fetching URLs, requesting authorization, paying for classification,
and storing results.
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
    URLBatch, PolicyDecision, Classification,
    BatchRecord, ClassificationRecord, NetworkStats
)
from shared.database import db
from shared.events import (
    broadcaster,
    emit_scout_found_urls, emit_policy_requesting, emit_policy_verified,
    emit_payment_sending, emit_payment_sent, emit_work_verified,
    emit_database_updated, emit_error
)
from shared.a2a import (
    A2AClient, PaymentRequiredError,
    invoke_policy_authorization, invoke_analyst_classification,
    build_agent_card, build_skill,
    build_agent_card_v3, build_skill_v3
)
from shared.x402 import X402Client
from shared.logging_config import scout_logger as logger
from shared.prover import authorization_prover, classifier_prover

from sources import (
    PhishTankSource, OpenPhishSource, URLhausSource,
    SyntheticSource, AlexaTopSource,
    CertTransparencySource, TwitterSource, PasteSiteSource,
    TyposquatSource
)
from shared.reputation import reputation_manager


# ============ Scout Agent ============

class ScoutAgent:
    """
    Scout Agent orchestrates the threat intelligence pipeline.

    Responsibilities:
    1. Discover URLs from multiple sources
    2. Filter out already-classified URLs
    3. Request authorization from Policy Agent
    4. Pay Analyst Agent for classification
    5. Verify proofs and store results
    """

    def __init__(self):
        # Build sources list based on configuration
        # Priority order: original discovery first, then aggregation, then synthetic fallback
        self.sources = [
            # ORIGINAL DISCOVERY - finds threats before anyone else
            CertTransparencySource(lookback_days=1),  # New certs impersonating brands
            TyposquatSource(check_interval_hours=6),  # Proactive typosquat scanning

            # AGGREGATION - known threat feeds
            OpenPhishSource(),    # Real phishing URLs (free, no API key)
            URLhausSource(),      # Real malware URLs (free, no API key)
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

    async def start(self):
        """Start the continuous discovery loop"""
        self.running = True
        logger.info(f"Scout Agent starting... Interval: {config.scout_interval_seconds}s")

        while self.running:
            try:
                await self.process_batch()
            except Exception as e:
                logger.error(f"Error in batch processing: {e}", exc_info=True)
                await emit_error(None, str(e))

            # Wait for next interval
            await asyncio.sleep(config.scout_interval_seconds)

    async def stop(self):
        """Stop the discovery loop"""
        self.running = False

    async def process_batch(self) -> Optional[str]:
        """
        Process one batch of URLs through the pipeline.

        Returns batch_id if successful, None if no URLs to process.
        """
        # 1. Discover URLs
        urls, source_name, source_reputation = await self._discover_urls()
        if not urls:
            logger.debug("No new URLs found")
            return None

        batch_id = str(uuid.uuid4())
        logger.info(f"Batch {batch_id}: Found {len(urls)} URLs from {source_name}")

        await emit_scout_found_urls(batch_id, len(urls), source_name, sample_urls=urls)

        # 2. Calculate costs
        policy_cost = config.policy_price_per_decision
        analyst_cost = len(urls) * config.analyst_price_per_url
        total_cost = policy_cost + analyst_cost

        # 3. Get budget (from treasury)
        budget = await self._get_budget()
        if budget < total_cost:
            logger.warning(f"Insufficient budget: {budget} < {total_cost}")
            await emit_error(batch_id, "Insufficient budget")
            return None

        # 4. Request authorization from Policy Agent
        await emit_policy_requesting(batch_id, len(urls), total_cost)

        try:
            auth_response = await self._request_authorization(
                batch_id=batch_id,
                url_count=len(urls),
                estimated_cost=total_cost,
                budget_remaining=budget,
                source_reputation=source_reputation
            )
        except (ConnectionError, TimeoutError, ValueError) as e:
            logger.error(f"Authorization failed: {e}")
            await emit_error(batch_id, f"Authorization failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Authorization failed with unexpected error: {e}", exc_info=True)
            await emit_error(batch_id, f"Authorization failed: {e}")
            return None

        if auth_response["decision"] != PolicyDecision.AUTHORIZED.value:
            logger.info(f"Batch denied: {auth_response['decision']}")
            await emit_error(batch_id, f"Batch denied: {auth_response['decision']}")
            return None

        # 5. Verify policy proof
        policy_valid, verify_time = await self._verify_policy_proof(auth_response)
        await emit_policy_verified(batch_id, policy_valid, verify_time)

        if not policy_valid:
            logger.error("Policy proof verification failed!")
            await emit_error(batch_id, "Policy proof verification failed")
            return None

        # 6. Pay Policy Agent
        await emit_payment_sending(batch_id, policy_cost, config.policy_url)
        policy_receipt = await self._pay_agent(config.policy_url, policy_cost, f"policy-{batch_id}")
        await emit_payment_sent(batch_id, policy_receipt.tx_hash if policy_receipt else "simulated", policy_cost)

        # 7. Record batch in database
        batch_record = BatchRecord(
            id=batch_id,
            url_count=len(urls),
            source=source_name,
            policy_decision=PolicyDecision.AUTHORIZED,
            policy_proof_hash=auth_response["proof_hash"],
            policy_paid_usdc=policy_cost,
            total_analyst_paid_usdc=0  # Updated after classification
        )
        await db.insert_batch(batch_record)

        # 8. Request classification from Analyst Agent (with payment)
        await emit_payment_sending(batch_id, analyst_cost, config.analyst_url)

        try:
            class_response = await self._request_classification(
                batch_id=batch_id,
                urls=urls,
                policy_proof_hash=auth_response["proof_hash"],
                payment_amount=analyst_cost
            )
            await emit_payment_sent(batch_id, class_response.get("payment_tx", "simulated"), analyst_cost)
        except (ConnectionError, TimeoutError, ValueError) as e:
            logger.error(f"Classification failed: {e}")
            await emit_error(batch_id, f"Classification failed: {e}")
            return None
        except Exception as e:
            logger.error(f"Classification failed with unexpected error: {e}", exc_info=True)
            await emit_error(batch_id, f"Classification failed: {e}")
            return None

        # 9. Verify work proof
        work_valid, work_verify_time = await self._verify_work_proof(class_response)
        await emit_work_verified(batch_id, work_valid, work_verify_time)

        if not work_valid:
            logger.error("Work proof verification failed!")
            await emit_error(batch_id, "Work proof verification failed")
            return None

        # 10. Store classifications
        await self._store_classifications(
            batch_id=batch_id,
            results=class_response["results"],
            proof_hash=class_response["proof_hash"],
            model_commitment=class_response["model_commitment"],
            source=source_name,
            policy_proof_hash=auth_response["proof_hash"],
            analyst_paid=analyst_cost
        )

        # 11. Update batch as complete
        await db.complete_batch(batch_id, analyst_cost)

        # 12. Emit database update event
        stats = await db.get_network_stats()
        await emit_database_updated(
            batch_id=batch_id,
            urls_added=len(urls),
            total_urls=stats.total_urls,
            total_phishing=stats.phishing_count,
            total_safe=stats.safe_count,
            total_suspicious=stats.suspicious_count
        )

        self.batches_processed += 1
        self.urls_processed += len(urls)
        self.last_batch_time = datetime.utcnow()

        logger.info(f"Batch {batch_id} complete: {len(urls)} URLs classified")
        return batch_id

    async def _discover_urls(self) -> tuple[List[str], str, float]:
        """Discover URLs from sources and filter already-classified ones"""
        for source in self.sources:
            try:
                urls = await source.fetch_urls(limit=config.scout_batch_size)
                if urls:
                    # Filter out already classified
                    novel_urls = await db.filter_novel_urls(urls)
                    if novel_urls:
                        # Use dynamic reputation if enabled
                        if config.enable_dynamic_reputation:
                            reputation = reputation_manager.get_reputation(source.name)
                        else:
                            reputation = source.reputation
                        return novel_urls, source.name, reputation
            except (ConnectionError, TimeoutError) as e:
                logger.warning(f"Network error fetching from {source.name}: {e}")
                source.record_error()
            except Exception as e:
                logger.error(f"Error fetching from {source.name}: {e}", exc_info=True)
                source.record_error()

        return [], "", 0.0

    async def _get_budget(self) -> float:
        """Get current treasury balance"""
        try:
            return self.x402_client.get_balance()
        except (ConnectionError, ValueError) as e:
            logger.warning(f"Could not get real balance: {e}, using simulated")
            return 1000.0

    async def _request_authorization(
        self,
        batch_id: str,
        url_count: int,
        estimated_cost: float,
        budget_remaining: float,
        source_reputation: float
    ) -> dict:
        """Request authorization from Policy Agent"""
        time_since_last = 0
        if self.last_batch_time:
            time_since_last = int((datetime.utcnow() - self.last_batch_time).total_seconds())

        return await invoke_policy_authorization(
            batch_id=batch_id,
            url_count=url_count,
            estimated_cost=estimated_cost,
            budget_remaining=budget_remaining,
            source_reputation=source_reputation,
            novelty_score=0.9,  # Assume high novelty for now
            time_since_last=time_since_last,
            threat_level=0.5   # Moderate threat level
        )

    async def _verify_policy_proof(self, auth_response: dict) -> tuple[bool, int]:
        """Verify the policy agent's proof"""
        start_time = time.time()

        try:
            # Get proof bytes from hex
            proof_hex = auth_response.get("proof", "")
            if not proof_hex:
                logger.warning("No proof provided in auth response")
                return False, 0

            proof_bytes = bytes.fromhex(proof_hex)

            # Verify using the prover
            result = await authorization_prover.verify_authorization(
                proof=proof_bytes,
                model_commitment=auth_response.get("model_commitment", ""),
                input_commitment=auth_response.get("input_commitment", ""),
                output_commitment=auth_response.get("output_commitment", "")
            )

            verify_time = int((time.time() - start_time) * 1000)

            if not result.valid:
                logger.error(f"Policy proof invalid: {result.error}")
            else:
                logger.info(f"Policy proof verified in {verify_time}ms")

            return result.valid, verify_time

        except (ValueError, TypeError) as e:
            logger.error(f"Error verifying policy proof: {e}")
            verify_time = int((time.time() - start_time) * 1000)
            return False, verify_time

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

    async def _request_classification(
        self,
        batch_id: str,
        urls: List[str],
        policy_proof_hash: str,
        payment_amount: float
    ) -> dict:
        """Request classification from Analyst Agent with payment"""
        try:
            # First attempt without payment to get 402
            return await invoke_analyst_classification(
                batch_id=batch_id,
                urls=urls,
                policy_proof_hash=policy_proof_hash
            )
        except PaymentRequiredError as e:
            # Make payment
            payment_info = e.payment_info
            if payment_info:
                receipt = await self.x402_client.make_payment(
                    recipient=payment_info["recipient"],
                    amount_usdc=float(payment_info["amount"]),
                    memo=f"classify-{batch_id}"
                )
                tx_hash = receipt.tx_hash if receipt else "simulated"
            else:
                tx_hash = "simulated"

            # Retry with payment
            response = await invoke_analyst_classification(
                batch_id=batch_id,
                urls=urls,
                policy_proof_hash=policy_proof_hash,
                payment_receipt=tx_hash
            )
            response["payment_tx"] = tx_hash
            return response

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
        policy_proof_hash: str,
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
                policy_proof_hash=policy_proof_hash
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

    # Initialize database and start scout loop
    await db.connect()
    await db.init_schema()
    asyncio.create_task(scout.start())
    yield
    # Shutdown
    await scout.stop()
    await db.close()


app = FastAPI(
    title="Scout Agent",
    description="Discovers URLs and orchestrates the threat intelligence pipeline",
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
        name="Scout Agent",
        description="Discovers suspicious URLs and orchestrates threat intelligence gathering",
        url=config.scout_url,
        version="1.0.0",
        streaming=False,
        push_notifications=False,
        state_transition_history=True,
        provider="ThreatProof Network",
        documentation_url=f"{config.scout_url}/docs",
        skills=[
            build_skill_v3(
                skill_id="discover-urls",
                name="URL Discovery",
                description="Discover suspicious URLs from multiple sources including Certificate Transparency, typosquatting detection, and threat feeds",
                tags=["threat-intel", "url-discovery", "phishing", "security"],
                input_modes=["application/json"],
                output_modes=["application/json"],
                price_amount=0,  # Scout is the initiator, no payment required
                proof_required=False
            )
        ]
    )


@app.get("/")
async def root():
    """ThreatProof - Verifiable Threat Intelligence Network"""
    stats = await db.get_network_stats()
    return {
        "name": "ThreatProof",
        "description": "Autonomous threat intelligence with zkML-verified classifications",
        "status": "running" if scout.running else "stopped",
        "stats": {
            "batches_processed": scout.batches_processed,
            "urls_classified": scout.urls_processed,
            "total_in_database": stats.total_urls if stats else 0,
            "phishing_detected": stats.phishing_count if stats else 0,
        },
        "agents": {
            "scout": config.scout_url,
            "policy": config.policy_url,
            "analyst": config.analyst_url
        },
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "stats": "/stats",
            "websocket": "/ws",
            "trigger_batch": "/trigger"
        },
        "technology": ["A2A Protocol", "x402 Payments", "zkML Proofs", "Base/USDC"]
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "running": scout.running,
        "batches_processed": scout.batches_processed,
        "urls_processed": scout.urls_processed
    }


@app.get("/stats")
async def stats():
    """Get network statistics"""
    return await db.get_network_stats()


@app.post("/trigger")
async def trigger_batch(x_api_key: Optional[str] = Header(None, alias="X-API-Key")):
    """
    Manually trigger a batch (for testing).

    Requires API key authentication if SCOUT_API_KEY is configured.
    """
    # Check API key if configured
    if config.scout_api_key:
        if not x_api_key:
            raise HTTPException(
                status_code=401,
                detail="API key required. Set X-API-Key header."
            )
        if x_api_key != config.scout_api_key:
            raise HTTPException(
                status_code=401,
                detail="Invalid API key."
            )

    batch_id = await scout.process_batch()
    return {"batch_id": batch_id}


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


if __name__ == "__main__":
    import os
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
