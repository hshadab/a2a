"""
Database Client for Threat Intelligence Network

PostgreSQL-based storage for classifications, stats, and proofs.
Falls back to in-memory storage for demo mode.
"""
import asyncio
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any, Tuple
from contextlib import asynccontextmanager
import json

from .types import (
    Classification, ClassificationRecord, BatchRecord,
    DomainStats, RegistrarStats, IPStats, NetworkStats
)
from .config import config
from .logging_config import database_logger as logger

# Try to import asyncpg, fall back to in-memory mode
try:
    import asyncpg
    from asyncpg import Pool
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False
    Pool = None


class InMemoryDatabase:
    """
    In-memory database for demo mode when PostgreSQL is not available.
    """

    def __init__(self):
        self.classifications: Dict[str, ClassificationRecord] = {}
        self.batches: Dict[str, BatchRecord] = {}
        self.domain_stats: Dict[str, DomainStats] = {}
        self.registrar_stats: Dict[str, RegistrarStats] = {}
        self.ip_stats: Dict[str, IPStats] = {}
        self.first_batch_time: Optional[datetime] = None

    async def connect(self):
        logger.info("Running in DEMO MODE (in-memory storage)")

    async def close(self):
        pass

    async def init_schema(self):
        pass

    async def insert_classification(self, record: ClassificationRecord):
        self.classifications[record.url] = record
        await self._update_domain_stats(record.domain, record.classification)

    async def insert_classifications_batch(self, records: List[ClassificationRecord]):
        for record in records:
            await self.insert_classification(record)

    async def url_exists(self, url: str) -> bool:
        return url in self.classifications

    async def filter_novel_urls(self, urls: List[str]) -> List[str]:
        return [url for url in urls if url not in self.classifications]

    async def get_classification(self, url: str) -> Optional[ClassificationRecord]:
        return self.classifications.get(url)

    async def insert_batch(self, record: BatchRecord):
        self.batches[record.id] = record
        if self.first_batch_time is None:
            self.first_batch_time = record.created_at

    async def complete_batch(self, batch_id: str, total_analyst_paid: float):
        if batch_id in self.batches:
            self.batches[batch_id].completed_at = datetime.utcnow()
            self.batches[batch_id].total_analyst_paid_usdc = total_analyst_paid

    async def get_last_batch_time(self) -> Optional[datetime]:
        if not self.batches:
            return None
        return max(b.created_at for b in self.batches.values())

    async def _update_domain_stats(self, domain: str, classification: Classification):
        if domain not in self.domain_stats:
            self.domain_stats[domain] = DomainStats(
                domain=domain,
                times_seen=0,
                phishing_count=0,
                safe_count=0,
                suspicious_count=0,
                first_seen=datetime.utcnow(),
                last_seen=datetime.utcnow()
            )

        stats = self.domain_stats[domain]
        stats.times_seen += 1
        stats.last_seen = datetime.utcnow()

        if classification == Classification.PHISHING:
            stats.phishing_count += 1
        elif classification == Classification.SAFE:
            stats.safe_count += 1
        elif classification == Classification.SUSPICIOUS:
            stats.suspicious_count += 1

    async def get_domain_stats(self, domain: str) -> Optional[DomainStats]:
        return self.domain_stats.get(domain)

    async def get_similar_domains(self, domain: str, limit: int = 10) -> List[DomainStats]:
        # Simple similarity: domains containing similar substrings
        similar = []
        domain_parts = set(domain.lower().split('.'))
        for d, stats in self.domain_stats.items():
            if d != domain:
                d_parts = set(d.lower().split('.'))
                if domain_parts & d_parts:  # Any common parts
                    similar.append(stats)
        return similar[:limit]

    async def get_domain_similarity(self, domain1: str, domain2: str) -> float:
        """
        Calculate simple similarity between two domains.
        Uses basic string comparison for in-memory mode.
        """
        from difflib import SequenceMatcher
        return SequenceMatcher(None, domain1.lower(), domain2.lower()).ratio()

    async def update_registrar_stats(self, registrar: str, is_phishing: bool):
        if registrar not in self.registrar_stats:
            self.registrar_stats[registrar] = RegistrarStats(
                registrar=registrar,
                domains_seen=0,
                phishing_count=0,
                last_updated=datetime.utcnow()
            )
        stats = self.registrar_stats[registrar]
        stats.domains_seen += 1
        if is_phishing:
            stats.phishing_count += 1
        stats.last_updated = datetime.utcnow()

    async def get_registrar_stats(self, registrar: str) -> Optional[RegistrarStats]:
        return self.registrar_stats.get(registrar)

    async def update_ip_stats(self, ip: str, is_phishing: bool):
        if ip not in self.ip_stats:
            self.ip_stats[ip] = IPStats(
                ip=ip,
                domains_hosted=0,
                phishing_count=0,
                last_updated=datetime.utcnow()
            )
        stats = self.ip_stats[ip]
        stats.domains_hosted += 1
        if is_phishing:
            stats.phishing_count += 1
        stats.last_updated = datetime.utcnow()

    async def get_ip_stats(self, ip: str) -> Optional[IPStats]:
        return self.ip_stats.get(ip)

    async def get_classification_context(self, domain: str, registrar: Optional[str] = None, ip: Optional[str] = None) -> Dict[str, Any]:
        context = {
            "domain_phish_rate": None,
            "similar_domains_phish_rate": None,
            "registrar_phish_rate": None,
            "ip_phish_rate": None
        }

        domain_stats = await self.get_domain_stats(domain)
        if domain_stats and domain_stats.times_seen > 0:
            context["domain_phish_rate"] = domain_stats.phishing_rate

        similar = await self.get_similar_domains(domain, limit=5)
        if similar:
            total_seen = sum(d.times_seen for d in similar)
            total_phish = sum(d.phishing_count for d in similar)
            if total_seen > 0:
                context["similar_domains_phish_rate"] = total_phish / total_seen

        if registrar:
            reg_stats = await self.get_registrar_stats(registrar)
            if reg_stats and reg_stats.domains_seen > 0:
                context["registrar_phish_rate"] = reg_stats.phishing_rate

        if ip:
            ip_stats = await self.get_ip_stats(ip)
            if ip_stats and ip_stats.domains_hosted > 0:
                context["ip_phish_rate"] = ip_stats.phishing_rate

        return context

    async def get_network_stats(self) -> NetworkStats:
        total = len(self.classifications)
        phishing = sum(1 for c in self.classifications.values() if c.classification == Classification.PHISHING)
        safe = sum(1 for c in self.classifications.values() if c.classification == Classification.SAFE)
        suspicious = sum(1 for c in self.classifications.values() if c.classification == Classification.SUSPICIOUS)

        analyst_paid = sum(c.analyst_paid_usdc or 0 for c in self.classifications.values())
        policy_paid = sum(b.policy_paid_usdc or 0 for b in self.batches.values())

        return NetworkStats(
            total_urls=total,
            phishing_count=phishing,
            safe_count=safe,
            suspicious_count=suspicious,
            total_batches=len(self.batches),
            total_proofs=total,
            policy_paid_usdc=policy_paid,
            analyst_paid_usdc=analyst_paid,
            total_spent_usdc=policy_paid + analyst_paid,
            running_since=self.first_batch_time
        )

    async def get_top_phishing_tlds(self, limit: int = 10) -> List[Tuple[str, int, float]]:
        tld_stats: Dict[str, Dict[str, int]] = {}
        for record in self.classifications.values():
            tld = record.domain.split('.')[-1] if '.' in record.domain else record.domain
            if tld not in tld_stats:
                tld_stats[tld] = {'total': 0, 'phishing': 0}
            tld_stats[tld]['total'] += 1
            if record.classification == Classification.PHISHING:
                tld_stats[tld]['phishing'] += 1

        results = []
        for tld, stats in tld_stats.items():
            if stats['total'] >= 5:
                rate = stats['phishing'] / stats['total']
                results.append((tld, stats['total'], rate))

        results.sort(key=lambda x: x[2], reverse=True)
        return results[:limit]

    async def get_top_phishing_registrars(self, limit: int = 10) -> List[RegistrarStats]:
        results = [s for s in self.registrar_stats.values() if s.domains_seen >= 5]
        results.sort(key=lambda x: x.phishing_rate, reverse=True)
        return results[:limit]

    async def get_recent_classifications(self, limit: int = 100) -> List[ClassificationRecord]:
        records = list(self.classifications.values())
        records.sort(key=lambda x: x.classified_at, reverse=True)
        return records[:limit]


class Database:
    """
    Async PostgreSQL database client.
    Falls back to in-memory storage if PostgreSQL is not available.
    """

    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or config.database_url
        self._pool: Optional[Pool] = None
        self._in_memory: Optional[InMemoryDatabase] = None
        self._demo_mode = False
        self._connection_error: Optional[str] = None

    def get_status(self) -> Dict[str, Any]:
        """Get database connection status for debugging."""
        # Mask the password in the URL for safety
        masked_url = "not configured"
        if self.database_url:
            import re
            masked_url = re.sub(r':([^:@]+)@', ':***@', self.database_url)

        return {
            "mode": "in-memory" if self._demo_mode else "postgresql",
            "connected": self._pool is not None,
            "asyncpg_available": ASYNCPG_AVAILABLE,
            "database_url_configured": bool(self.database_url),
            "database_url_masked": masked_url,
            "connection_error": self._connection_error,
        }

    async def connect(self):
        """Initialize connection pool or fall back to in-memory mode"""
        if not ASYNCPG_AVAILABLE:
            if config.production_mode:
                logger.warning(
                    "Production mode prefers PostgreSQL but asyncpg is not available. "
                    "Falling back to in-memory for debugging."
                )
            logger.info("asyncpg not available, using in-memory storage")
            self._demo_mode = True
            self._in_memory = InMemoryDatabase()
            await self._in_memory.connect()
            return

        # Retry connection with exponential backoff
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                logger.info(f"Attempting PostgreSQL connection (attempt {attempt + 1}/{max_retries})...")

                # Render requires SSL - asyncpg doesn't parse sslmode from URL
                # so we must set ssl=True explicitly
                import ssl
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE

                # Strip sslmode from URL as we're handling SSL separately
                db_url = self.database_url
                if '?sslmode=' in db_url:
                    db_url = db_url.split('?sslmode=')[0]
                elif '&sslmode=' in db_url:
                    db_url = db_url.replace('&sslmode=require', '').replace('&sslmode=prefer', '')

                self._pool = await asyncpg.create_pool(
                    db_url,
                    min_size=1,
                    max_size=5,
                    command_timeout=60,
                    ssl=ssl_context,
                )

                # Test the connection
                async with self._pool.acquire() as conn:
                    await conn.fetchval('SELECT 1')

                logger.info(f"Connected to PostgreSQL successfully")
                self._connection_error = None

                # Initialize schema
                await self.init_schema()
                return

            except Exception as e:
                self._connection_error = str(e)
                logger.warning(f"PostgreSQL connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff

        # All retries failed - fall back to in-memory
        logger.error(f"All PostgreSQL connection attempts failed: {self._connection_error}")
        if config.production_mode:
            logger.warning(
                f"Production mode prefers PostgreSQL but connection failed. "
                f"Falling back to in-memory for debugging. Check DATABASE_URL configuration."
            )
        logger.info("Falling back to in-memory storage (DEMO MODE)")
        self._demo_mode = True
        self._in_memory = InMemoryDatabase()
        await self._in_memory.connect()

    async def close(self):
        """Close connection pool"""
        if self._pool:
            await self._pool.close()

    @asynccontextmanager
    async def acquire(self):
        """Acquire a connection from the pool"""
        if self._demo_mode:
            yield None
            return
        async with self._pool.acquire() as conn:
            yield conn

    async def init_schema(self):
        """Initialize database schema"""
        if self._demo_mode:
            return
        async with self.acquire() as conn:
            await conn.execute(SCHEMA_SQL)

    # ============ Delegate to in-memory if in demo mode ============

    async def insert_classification(self, record: ClassificationRecord):
        if self._demo_mode:
            return await self._in_memory.insert_classification(record)
        async with self.acquire() as conn:
            await conn.execute("""
                INSERT INTO classifications (
                    id, url, domain, classification, confidence,
                    proof_hash, model_commitment, input_commitment, output_commitment,
                    features, context_used, source, batch_id,
                    analyst_paid_usdc, policy_proof_hash, classified_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
            """,
                record.id, record.url, record.domain, record.classification.value,
                record.confidence, record.proof_hash, record.model_commitment,
                record.input_commitment, record.output_commitment,
                json.dumps(record.features) if record.features else None,
                json.dumps(record.context_used) if record.context_used else None,
                record.source, record.batch_id, record.analyst_paid_usdc,
                record.policy_proof_hash, record.classified_at
            )
            await self._update_domain_stats(conn, record.domain, record.classification)

    async def insert_classifications_batch(self, records: List[ClassificationRecord]):
        if self._demo_mode:
            return await self._in_memory.insert_classifications_batch(records)
        async with self.acquire() as conn:
            async with conn.transaction():
                for record in records:
                    await conn.execute("""
                        INSERT INTO classifications (
                            id, url, domain, classification, confidence,
                            proof_hash, model_commitment, input_commitment, output_commitment,
                            features, context_used, source, batch_id,
                            analyst_paid_usdc, policy_proof_hash, classified_at
                        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                    """,
                        record.id, record.url, record.domain, record.classification.value,
                        record.confidence, record.proof_hash, record.model_commitment,
                        record.input_commitment, record.output_commitment,
                        json.dumps(record.features) if record.features else None,
                        json.dumps(record.context_used) if record.context_used else None,
                        record.source, record.batch_id, record.analyst_paid_usdc,
                        record.policy_proof_hash, record.classified_at
                    )
                    await self._update_domain_stats(conn, record.domain, record.classification)

    async def url_exists(self, url: str) -> bool:
        if self._demo_mode:
            return await self._in_memory.url_exists(url)
        async with self.acquire() as conn:
            result = await conn.fetchval(
                "SELECT EXISTS(SELECT 1 FROM classifications WHERE url = $1)",
                url
            )
            return result

    async def filter_novel_urls(self, urls: List[str]) -> List[str]:
        if self._demo_mode:
            return await self._in_memory.filter_novel_urls(urls)
        if not urls:
            return []
        async with self.acquire() as conn:
            existing = await conn.fetch(
                "SELECT url FROM classifications WHERE url = ANY($1)",
                urls
            )
            existing_set = {row['url'] for row in existing}
            return [url for url in urls if url not in existing_set]

    async def get_classification(self, url: str) -> Optional[ClassificationRecord]:
        if self._demo_mode:
            return await self._in_memory.get_classification(url)
        async with self.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM classifications WHERE url = $1",
                url
            )
            if row:
                return ClassificationRecord(**dict(row))
            return None

    async def insert_batch(self, record: BatchRecord):
        if self._demo_mode:
            return await self._in_memory.insert_batch(record)
        async with self.acquire() as conn:
            await conn.execute("""
                INSERT INTO batches (
                    id, url_count, source, policy_decision, policy_proof_hash,
                    policy_paid_usdc, total_analyst_paid_usdc, created_at, completed_at
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            """,
                record.id, record.url_count, record.source,
                record.policy_decision.value, record.policy_proof_hash,
                record.policy_paid_usdc, record.total_analyst_paid_usdc,
                record.created_at, record.completed_at
            )

    async def complete_batch(self, batch_id: str, total_analyst_paid: float):
        if self._demo_mode:
            return await self._in_memory.complete_batch(batch_id, total_analyst_paid)
        async with self.acquire() as conn:
            await conn.execute("""
                UPDATE batches
                SET completed_at = $1, total_analyst_paid_usdc = $2
                WHERE id = $3
            """, datetime.utcnow(), total_analyst_paid, batch_id)

    async def get_last_batch_time(self) -> Optional[datetime]:
        if self._demo_mode:
            return await self._in_memory.get_last_batch_time()
        async with self.acquire() as conn:
            result = await conn.fetchval(
                "SELECT MAX(created_at) FROM batches"
            )
            return result

    async def _update_domain_stats(self, conn, domain: str, classification: Classification):
        await conn.execute("""
            INSERT INTO domain_stats (domain, times_seen, phishing_count, safe_count, suspicious_count, first_seen, last_seen)
            VALUES ($1, 1, $2, $3, $4, NOW(), NOW())
            ON CONFLICT (domain) DO UPDATE SET
                times_seen = domain_stats.times_seen + 1,
                phishing_count = domain_stats.phishing_count + $2,
                safe_count = domain_stats.safe_count + $3,
                suspicious_count = domain_stats.suspicious_count + $4,
                last_seen = NOW()
        """,
            domain,
            1 if classification == Classification.PHISHING else 0,
            1 if classification == Classification.SAFE else 0,
            1 if classification == Classification.SUSPICIOUS else 0
        )

    async def get_domain_stats(self, domain: str) -> Optional[DomainStats]:
        if self._demo_mode:
            return await self._in_memory.get_domain_stats(domain)
        async with self.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM domain_stats WHERE domain = $1",
                domain
            )
            if row:
                return DomainStats(**dict(row))
            return None

    async def get_similar_domains(self, domain: str, limit: int = 10) -> List[DomainStats]:
        if self._demo_mode:
            return await self._in_memory.get_similar_domains(domain, limit)
        async with self.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM domain_stats
                WHERE domain % $1 AND domain != $1
                ORDER BY similarity(domain, $1) DESC
                LIMIT $2
            """, domain, limit)
            return [DomainStats(**dict(row)) for row in rows]

    async def get_domain_similarity(self, domain1: str, domain2: str) -> float:
        """
        Calculate the similarity between two domains using PostgreSQL's pg_trgm.

        Args:
            domain1: First domain
            domain2: Second domain

        Returns:
            Similarity score between 0 and 1
        """
        if self._demo_mode:
            return await self._in_memory.get_domain_similarity(domain1, domain2)
        async with self.acquire() as conn:
            result = await conn.fetchval(
                "SELECT similarity($1, $2)",
                domain1, domain2
            )
            return float(result) if result else 0.0

    async def update_registrar_stats(self, registrar: str, is_phishing: bool):
        if self._demo_mode:
            return await self._in_memory.update_registrar_stats(registrar, is_phishing)
        async with self.acquire() as conn:
            await conn.execute("""
                INSERT INTO registrar_stats (registrar, domains_seen, phishing_count, last_updated)
                VALUES ($1, 1, $2, NOW())
                ON CONFLICT (registrar) DO UPDATE SET
                    domains_seen = registrar_stats.domains_seen + 1,
                    phishing_count = registrar_stats.phishing_count + $2,
                    last_updated = NOW()
            """, registrar, 1 if is_phishing else 0)

    async def get_registrar_stats(self, registrar: str) -> Optional[RegistrarStats]:
        if self._demo_mode:
            return await self._in_memory.get_registrar_stats(registrar)
        async with self.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM registrar_stats WHERE registrar = $1",
                registrar
            )
            if row:
                return RegistrarStats(**dict(row))
            return None

    async def update_ip_stats(self, ip: str, is_phishing: bool):
        if self._demo_mode:
            return await self._in_memory.update_ip_stats(ip, is_phishing)
        async with self.acquire() as conn:
            await conn.execute("""
                INSERT INTO ip_stats (ip, domains_hosted, phishing_count, last_updated)
                VALUES ($1, 1, $2, NOW())
                ON CONFLICT (ip) DO UPDATE SET
                    domains_hosted = ip_stats.domains_hosted + 1,
                    phishing_count = ip_stats.phishing_count + $2,
                    last_updated = NOW()
            """, ip, 1 if is_phishing else 0)

    async def get_ip_stats(self, ip: str) -> Optional[IPStats]:
        if self._demo_mode:
            return await self._in_memory.get_ip_stats(ip)
        async with self.acquire() as conn:
            row = await conn.fetchrow(
                "SELECT * FROM ip_stats WHERE ip = $1",
                ip
            )
            if row:
                return IPStats(**dict(row))
            return None

    async def get_classification_context(self, domain: str, registrar: Optional[str] = None, ip: Optional[str] = None) -> Dict[str, Any]:
        if self._demo_mode:
            return await self._in_memory.get_classification_context(domain, registrar, ip)

        context = {
            "domain_phish_rate": None,
            "similar_domains_phish_rate": None,
            "registrar_phish_rate": None,
            "ip_phish_rate": None
        }

        domain_stats = await self.get_domain_stats(domain)
        if domain_stats and domain_stats.times_seen > 0:
            context["domain_phish_rate"] = domain_stats.phishing_rate

        similar = await self.get_similar_domains(domain, limit=5)
        if similar:
            total_seen = sum(d.times_seen for d in similar)
            total_phish = sum(d.phishing_count for d in similar)
            if total_seen > 0:
                context["similar_domains_phish_rate"] = total_phish / total_seen

        if registrar:
            reg_stats = await self.get_registrar_stats(registrar)
            if reg_stats and reg_stats.domains_seen > 0:
                context["registrar_phish_rate"] = reg_stats.phishing_rate

        if ip:
            ip_stats = await self.get_ip_stats(ip)
            if ip_stats and ip_stats.domains_hosted > 0:
                context["ip_phish_rate"] = ip_stats.phishing_rate

        return context

    async def get_network_stats(self) -> NetworkStats:
        if self._demo_mode:
            return await self._in_memory.get_network_stats()
        async with self.acquire() as conn:
            counts = await conn.fetchrow("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN classification = 'PHISHING' THEN 1 ELSE 0 END) as phishing,
                    SUM(CASE WHEN classification = 'SAFE' THEN 1 ELSE 0 END) as safe,
                    SUM(CASE WHEN classification = 'SUSPICIOUS' THEN 1 ELSE 0 END) as suspicious,
                    SUM(analyst_paid_usdc) as analyst_paid
                FROM classifications
            """)

            batch_stats = await conn.fetchrow("""
                SELECT
                    COUNT(*) as total_batches,
                    SUM(policy_paid_usdc) as policy_paid
                FROM batches
            """)

            first_batch = await conn.fetchval(
                "SELECT MIN(created_at) FROM batches"
            )

            return NetworkStats(
                total_urls=counts['total'] or 0,
                phishing_count=counts['phishing'] or 0,
                safe_count=counts['safe'] or 0,
                suspicious_count=counts['suspicious'] or 0,
                total_batches=batch_stats['total_batches'] or 0,
                total_proofs=counts['total'] or 0,
                policy_paid_usdc=float(batch_stats['policy_paid'] or 0),
                analyst_paid_usdc=float(counts['analyst_paid'] or 0),
                total_spent_usdc=float((batch_stats['policy_paid'] or 0) + (counts['analyst_paid'] or 0)),
                running_since=first_batch
            )

    async def get_top_phishing_tlds(self, limit: int = 10) -> List[Tuple[str, int, float]]:
        if self._demo_mode:
            return await self._in_memory.get_top_phishing_tlds(limit)
        async with self.acquire() as conn:
            rows = await conn.fetch("""
                SELECT
                    SUBSTRING(domain FROM '\\.([^.]+)$') as tld,
                    COUNT(*) as total,
                    SUM(CASE WHEN classification = 'PHISHING' THEN 1 ELSE 0 END)::float / COUNT(*) as phish_rate
                FROM classifications
                GROUP BY tld
                HAVING COUNT(*) >= 10
                ORDER BY phish_rate DESC
                LIMIT $1
            """, limit)
            return [(row['tld'], row['total'], row['phish_rate']) for row in rows]

    async def get_top_phishing_registrars(self, limit: int = 10) -> List[RegistrarStats]:
        if self._demo_mode:
            return await self._in_memory.get_top_phishing_registrars(limit)
        async with self.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM registrar_stats
                WHERE domains_seen >= 10
                ORDER BY (phishing_count::float / domains_seen) DESC
                LIMIT $1
            """, limit)
            return [RegistrarStats(**dict(row)) for row in rows]

    async def get_recent_classifications(self, limit: int = 100) -> List[ClassificationRecord]:
        if self._demo_mode:
            return await self._in_memory.get_recent_classifications(limit)
        async with self.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM classifications
                ORDER BY classified_at DESC
                LIMIT $1
            """, limit)
            return [ClassificationRecord(**dict(row)) for row in rows]


# Schema SQL
SCHEMA_SQL = """
-- Enable trigram extension for similarity search
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- Classifications table
CREATE TABLE IF NOT EXISTS classifications (
    id UUID PRIMARY KEY,
    url TEXT NOT NULL,
    domain TEXT NOT NULL,
    classification TEXT NOT NULL,
    confidence FLOAT NOT NULL,

    proof_hash TEXT NOT NULL,
    model_commitment TEXT NOT NULL,
    input_commitment TEXT NOT NULL,
    output_commitment TEXT NOT NULL,

    features JSONB,
    context_used JSONB,

    source TEXT,
    batch_id UUID,
    analyst_paid_usdc FLOAT,
    policy_proof_hash TEXT,
    classified_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_classifications_url ON classifications(url);
CREATE INDEX IF NOT EXISTS idx_classifications_domain ON classifications(domain);
CREATE INDEX IF NOT EXISTS idx_classifications_batch ON classifications(batch_id);
CREATE INDEX IF NOT EXISTS idx_classifications_time ON classifications(classified_at DESC);

-- Batches table
CREATE TABLE IF NOT EXISTS batches (
    id UUID PRIMARY KEY,
    url_count INT,
    source TEXT,
    policy_decision TEXT,
    policy_proof_hash TEXT,
    policy_paid_usdc FLOAT,
    total_analyst_paid_usdc FLOAT,
    created_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_batches_time ON batches(created_at DESC);

-- Domain stats
CREATE TABLE IF NOT EXISTS domain_stats (
    domain TEXT PRIMARY KEY,
    times_seen INT DEFAULT 0,
    phishing_count INT DEFAULT 0,
    safe_count INT DEFAULT 0,
    suspicious_count INT DEFAULT 0,
    first_seen TIMESTAMP,
    last_seen TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_domain_stats_trgm ON domain_stats USING gin (domain gin_trgm_ops);

-- Registrar stats
CREATE TABLE IF NOT EXISTS registrar_stats (
    registrar TEXT PRIMARY KEY,
    domains_seen INT DEFAULT 0,
    phishing_count INT DEFAULT 0,
    last_updated TIMESTAMP
);

-- IP stats
CREATE TABLE IF NOT EXISTS ip_stats (
    ip TEXT PRIMARY KEY,
    domains_hosted INT DEFAULT 0,
    phishing_count INT DEFAULT 0,
    last_updated TIMESTAMP
);

-- Treasury tracking
CREATE TABLE IF NOT EXISTS treasury (
    id SERIAL PRIMARY KEY,
    balance_usdc FLOAT DEFAULT 0,
    last_updated TIMESTAMP DEFAULT NOW()
);

INSERT INTO treasury (balance_usdc) VALUES (1000.0) ON CONFLICT DO NOTHING;
"""


# Global database instance
db = Database()
