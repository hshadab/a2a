# Future Enhancements Roadmap

This document outlines potential enhancements and features for the Threat Intelligence Network.

> **Note**: The core 2-agent system (Scout + Analyst) with self-authorization zkML proofs is complete and deployed. These items represent future improvements.

---

## Current Status (Implemented)

| Feature | Status |
|---------|--------|
| 2-Agent Architecture (Scout + Analyst) | Complete |
| Self-Authorization with zkML Proofs | Complete |
| Circular Payment Flow (x402 on Base) | Complete |
| OpenPhish + URLhaus Sources | Complete |
| Certificate Transparency Monitoring | Complete |
| Typosquat Detection | Complete |
| Quality Scoring with zkML Proof | Complete |
| Classification with zkML Proof | Complete |
| Real-time Dashboard | Complete |
| Render Deployment | Complete |

---

## Planned Enhancements

### Priority 1: Additional URL Sources

| Source | Effort | Value | Dependencies |
|--------|--------|-------|--------------|
| Twitter/X Threat Intel | 3-4 hours | HIGH | X API key |
| Paste Site Monitoring | 2-3 hours | MEDIUM | None |
| VirusTotal Integration | 2 hours | HIGH | VT API key |
| Abuse.ch Feeds | 1-2 hours | HIGH | None |

#### Twitter/X Source
Monitor threat intelligence accounts for reported phishing URLs.

```python
TRUSTED_ACCOUNTS = [
    'PhishTank', 'urlaboratories', 'abuse_ch',
    'JPCERT', 'MalwareTechBlog'
]
```

#### Paste Site Monitoring
Scrape paste sites for leaked credentials and phishing URLs.

---

### Priority 2: Campaign Clustering

**Purpose**: Group related phishing domains into campaigns to identify threat actors.

Features:
- Infrastructure clustering (shared IPs, registrars)
- Domain similarity (typosquatting patterns)
- Temporal clustering (registered/discovered together)

Output:
```json
{
  "campaign_id": "abc123",
  "domains": ["paypa1-login.xyz", "paypal-verify.xyz", "paypa1-secure.tk"],
  "shared_ip": "185.234.x.x",
  "registrar": "NameCheap",
  "confidence": 0.85,
  "threat_actor": "unknown"
}
```

---

### Priority 3: Dynamic Reputation System

Track source accuracy and adjust trust scores over time.

```python
class SourceMetrics:
    source_name: str
    initial_reputation: float
    current_reputation: float
    total_urls: int
    confirmed_correct: int
    confirmed_wrong: int
    false_positives: int
    false_negatives: int
```

Features:
- Accuracy tracking over time
- Reputation decay for stale sources
- Confidence intervals

---

### Priority 4: Enhanced Dashboard

| Feature | Effort | Description |
|---------|--------|-------------|
| Campaign Visualization | 4 hours | Graph view of domain clusters |
| Source Performance | 2 hours | Accuracy metrics per source |
| Historical Charts | 3 hours | URLs over time, classification breakdown |
| Alert System | 4 hours | Slack/Discord notifications for high-confidence threats |

---

### Priority 5: Test Suite

```
tests/
├── test_sources/
│   ├── test_openphish.py
│   ├── test_urlhaus.py
│   └── test_cert_transparency.py
├── test_features/
│   └── test_extraction.py
├── test_prover/
│   └── test_zkml.py
└── test_e2e/
    └── test_pipeline.py
```

Requirements:
```
pytest>=7.0.0
pytest-asyncio>=0.21.0
pytest-cov>=4.0.0
respx>=0.20.0
```

---

## Technical Debt

| Item | Priority | Notes |
|------|----------|-------|
| Add request timeout configuration | LOW | Currently hardcoded |
| Improve error recovery | MEDIUM | Auto-retry failed batches |
| Add metrics/observability | MEDIUM | Prometheus/Grafana |
| Database connection pooling | LOW | For higher throughput |

---

## Environment Variables (Future)

```bash
# For Twitter/X source
TWITTER_BEARER_TOKEN=your_bearer_token

# For VirusTotal
VIRUSTOTAL_API_KEY=your_api_key

# For alerts
SLACK_WEBHOOK_URL=https://hooks.slack.com/...
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
```

---

## Implementation Notes

### Adding a New URL Source

1. Create source file in `agents/scout/sources/`
2. Extend `URLSource` base class
3. Implement `fetch_urls(limit: int) -> List[str]`
4. Set appropriate reputation score
5. Register in `agents/scout/main.py`

```python
class NewSource(URLSource):
    def __init__(self):
        super().__init__(name="new_source", reputation=0.75)

    async def fetch_urls(self, limit: int = 100) -> List[str]:
        # Fetch URLs from source
        return urls
```

### Adding a New zkML Model

1. Train model in PyTorch
2. Export to ONNX using MatMul + Add (not Gemm)
3. Ensure weights are transposed for Jolt Atlas
4. Place in `agents/{agent}/models/jolt/`
5. Update prover configuration

```python
# Export format for Jolt Atlas compatibility
fc1_weight_t = fc1_weight.T  # Transpose!
matmul1 = helper.make_node('MatMul', ['input', 'fc1_weight_t'], ['out'])
add1 = helper.make_node('Add', ['out', 'fc1.bias'], ['output'])
```

---

## Commands

```bash
# Run tests (when implemented)
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Profile zkML models
cd jolt-atlas && cargo run -r -- profile --name authorization
```
