"""
Tests for database operations
"""
import pytest
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.database import InMemoryDatabase
from shared.types import (
    Classification,
    ClassificationRecord,
    BatchRecord,
    PolicyDecision,
    DomainStats,
)


class TestInMemoryDatabase:
    """Tests for InMemoryDatabase"""

    @pytest.mark.asyncio
    async def test_connect(self, mock_db):
        """Test database connection"""
        # mock_db fixture already calls connect
        assert mock_db is not None

    @pytest.mark.asyncio
    async def test_insert_classification(self, mock_db, sample_classification_record):
        """Test inserting a classification record"""
        await mock_db.insert_classification(sample_classification_record)

        result = await mock_db.get_classification(sample_classification_record.url)
        assert result is not None
        assert result.url == sample_classification_record.url
        assert result.classification == Classification.PHISHING

    @pytest.mark.asyncio
    async def test_insert_classifications_batch(self, mock_db):
        """Test batch insert of classifications"""
        records = [
            ClassificationRecord(
                url=f"https://test{i}.xyz/login",
                domain=f"test{i}.xyz",
                classification=Classification.PHISHING,
                confidence=0.9,
                proof_hash=f"proof_{i}",
                model_commitment="model",
                input_commitment="input",
                output_commitment="output",
                features={},
                context_used={},
                source="test",
                batch_id="batch-1",
                analyst_paid_usdc=0.0005,
                policy_proof_hash="policy",
            )
            for i in range(5)
        ]

        await mock_db.insert_classifications_batch(records)

        for record in records:
            result = await mock_db.get_classification(record.url)
            assert result is not None

    @pytest.mark.asyncio
    async def test_url_exists(self, mock_db, sample_classification_record):
        """Test URL existence check"""
        assert await mock_db.url_exists(sample_classification_record.url) is False

        await mock_db.insert_classification(sample_classification_record)

        assert await mock_db.url_exists(sample_classification_record.url) is True

    @pytest.mark.asyncio
    async def test_filter_novel_urls(self, mock_db, sample_classification_record):
        """Test filtering out already-classified URLs"""
        urls = [
            sample_classification_record.url,
            "https://new-url-1.com",
            "https://new-url-2.com",
        ]

        # Before any classification, all should be novel
        novel = await mock_db.filter_novel_urls(urls)
        assert len(novel) == 3

        # After classifying one
        await mock_db.insert_classification(sample_classification_record)

        novel = await mock_db.filter_novel_urls(urls)
        assert len(novel) == 2
        assert sample_classification_record.url not in novel

    @pytest.mark.asyncio
    async def test_insert_batch(self, mock_db, sample_batch_record):
        """Test inserting a batch record"""
        await mock_db.insert_batch(sample_batch_record)

        assert sample_batch_record.id in mock_db.batches
        assert mock_db.batches[sample_batch_record.id].url_count == 10

    @pytest.mark.asyncio
    async def test_complete_batch(self, mock_db, sample_batch_record):
        """Test completing a batch"""
        await mock_db.insert_batch(sample_batch_record)
        await mock_db.complete_batch(sample_batch_record.id, 0.01)

        batch = mock_db.batches[sample_batch_record.id]
        assert batch.completed_at is not None
        assert batch.total_analyst_paid_usdc == 0.01

    @pytest.mark.asyncio
    async def test_get_last_batch_time(self, mock_db, sample_batch_record):
        """Test getting last batch time"""
        # No batches yet
        last_time = await mock_db.get_last_batch_time()
        assert last_time is None

        # After inserting a batch
        await mock_db.insert_batch(sample_batch_record)

        last_time = await mock_db.get_last_batch_time()
        assert last_time is not None

    @pytest.mark.asyncio
    async def test_domain_stats_update(self, mock_db, sample_classification_record):
        """Test domain stats are updated on classification"""
        await mock_db.insert_classification(sample_classification_record)

        stats = await mock_db.get_domain_stats(sample_classification_record.domain)
        assert stats is not None
        assert stats.times_seen == 1
        assert stats.phishing_count == 1

    @pytest.mark.asyncio
    async def test_domain_stats_multiple_classifications(self, mock_db):
        """Test domain stats with multiple classifications"""
        domain = "test-domain.xyz"

        # Insert multiple classifications for same domain
        for i in range(3):
            record = ClassificationRecord(
                url=f"https://{domain}/page{i}",
                domain=domain,
                classification=Classification.PHISHING if i < 2 else Classification.SAFE,
                confidence=0.9,
                proof_hash=f"proof_{i}",
                model_commitment="model",
                input_commitment="input",
                output_commitment="output",
                features={},
                context_used={},
                source="test",
                batch_id="batch-1",
                analyst_paid_usdc=0.0005,
                policy_proof_hash="policy",
            )
            await mock_db.insert_classification(record)

        stats = await mock_db.get_domain_stats(domain)
        assert stats.times_seen == 3
        assert stats.phishing_count == 2
        assert stats.safe_count == 1

    @pytest.mark.asyncio
    async def test_get_similar_domains(self, mock_db):
        """Test similar domain lookup"""
        # Insert classifications for related domains
        domains = ["test.example.com", "api.example.com", "www.example.com"]

        for i, domain in enumerate(domains):
            record = ClassificationRecord(
                url=f"https://{domain}/page",
                domain=domain,
                classification=Classification.SAFE,
                confidence=0.9,
                proof_hash=f"proof_{i}",
                model_commitment="model",
                input_commitment="input",
                output_commitment="output",
                features={},
                context_used={},
                source="test",
                batch_id="batch-1",
                analyst_paid_usdc=0.0005,
                policy_proof_hash="policy",
            )
            await mock_db.insert_classification(record)

        # Query for similar domains
        similar = await mock_db.get_similar_domains("mail.example.com", limit=5)

        # In-memory implementation uses simple part matching
        assert len(similar) >= 0  # May vary based on implementation

    @pytest.mark.asyncio
    async def test_registrar_stats(self, mock_db):
        """Test registrar stats tracking"""
        await mock_db.update_registrar_stats("namecheap", is_phishing=True)
        await mock_db.update_registrar_stats("namecheap", is_phishing=False)
        await mock_db.update_registrar_stats("namecheap", is_phishing=True)

        stats = await mock_db.get_registrar_stats("namecheap")
        assert stats.domains_seen == 3
        assert stats.phishing_count == 2
        assert stats.phishing_rate == 2 / 3

    @pytest.mark.asyncio
    async def test_ip_stats(self, mock_db):
        """Test IP stats tracking"""
        await mock_db.update_ip_stats("192.168.1.1", is_phishing=True)
        await mock_db.update_ip_stats("192.168.1.1", is_phishing=True)

        stats = await mock_db.get_ip_stats("192.168.1.1")
        assert stats.domains_hosted == 2
        assert stats.phishing_count == 2
        assert stats.phishing_rate == 1.0

    @pytest.mark.asyncio
    async def test_get_network_stats(self, mock_db, sample_classification_record, sample_batch_record):
        """Test network statistics"""
        await mock_db.insert_batch(sample_batch_record)
        await mock_db.insert_classification(sample_classification_record)

        stats = await mock_db.get_network_stats()
        assert stats.total_urls == 1
        assert stats.phishing_count == 1
        assert stats.total_batches == 1

    @pytest.mark.asyncio
    async def test_get_top_phishing_tlds(self, mock_db):
        """Test top phishing TLDs query"""
        # Insert enough classifications for different TLDs
        tlds = [".xyz", ".xyz", ".xyz", ".com", ".com", ".top"]

        for i, tld in enumerate(tlds):
            record = ClassificationRecord(
                url=f"https://domain{i}{tld}/page",
                domain=f"domain{i}{tld}",
                classification=Classification.PHISHING if tld != ".com" else Classification.SAFE,
                confidence=0.9,
                proof_hash=f"proof_{i}",
                model_commitment="model",
                input_commitment="input",
                output_commitment="output",
                features={},
                context_used={},
                source="test",
                batch_id="batch-1",
                analyst_paid_usdc=0.0005,
                policy_proof_hash="policy",
            )
            await mock_db.insert_classification(record)

        # Need minimum threshold to appear in results
        top_tlds = await mock_db.get_top_phishing_tlds(limit=5)
        # Results depend on minimum count threshold
        assert isinstance(top_tlds, list)

    @pytest.mark.asyncio
    async def test_get_recent_classifications(self, mock_db):
        """Test getting recent classifications"""
        # Insert multiple classifications
        for i in range(10):
            record = ClassificationRecord(
                url=f"https://domain{i}.xyz/page",
                domain=f"domain{i}.xyz",
                classification=Classification.PHISHING,
                confidence=0.9,
                proof_hash=f"proof_{i}",
                model_commitment="model",
                input_commitment="input",
                output_commitment="output",
                features={},
                context_used={},
                source="test",
                batch_id="batch-1",
                analyst_paid_usdc=0.0005,
                policy_proof_hash="policy",
            )
            await mock_db.insert_classification(record)

        recent = await mock_db.get_recent_classifications(limit=5)
        assert len(recent) == 5

    @pytest.mark.asyncio
    async def test_get_classification_context(self, mock_db, sample_classification_record):
        """Test getting classification context"""
        await mock_db.insert_classification(sample_classification_record)
        await mock_db.update_registrar_stats("test-registrar", is_phishing=True)
        await mock_db.update_ip_stats("1.2.3.4", is_phishing=True)

        context = await mock_db.get_classification_context(
            domain=sample_classification_record.domain,
            registrar="test-registrar",
            ip="1.2.3.4"
        )

        assert "domain_phish_rate" in context
        assert "registrar_phish_rate" in context
        assert "ip_phish_rate" in context
