# Threat Intelligence Network

## A Long-Running, Compounding Demo of A2A + x402 + Jolt Atlas zkML

Agents continuously build a threat intelligence database by classifying URLs. The system runs for days/weeks, gets more valuable over time, and demonstrates why zkML is "must have" for trustless agent collaboration.

---

## The Vision

```
Day 1:     1,000 URLs → Small database
Day 30:   50,000 URLs → Patterns emerging
Day 90:  300,000 URLs → Real threat intel
Day 180: 1,000,000 URLs → Competitive product
```

Set it up. Walk away. Come back and watch it grow.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│  ┌─────────┐      ┌─────────┐      ┌─────────┐                  │
│  │  SCOUT  │      │ POLICY  │      │ ANALYST │                  │
│  │         │ ───► │         │ ───► │         │                  │
│  │ Finds   │ A2A  │Approves │ A2A  │Classifies│                 │
│  │ URLs    │      │spending │      │URLs     │                  │
│  │         │      │         │      │         │                  │
│  │ No proof│      │ zkML #1 │      │ zkML #2 │                  │
│  └─────────┘      └─────────┘      └─────────┘                  │
│       │                                 │                        │
│       │                                 │                        │
│       │    ┌───────────────────────────┐│                        │
│       │    │        DATABASE           ││                        │
│       │    │                           ▼│                        │
│       │    │  ┌─────────────────────────┐                       │
│       │    │  │ URLs: 147,832           │                       │
│       │    │  │ Phishing: 24,291        │                       │
│       │    │  │ Safe: 119,847           │                       │
│       │    │  │ Suspicious: 3,694       │                       │
│       │    │  │                         │                       │
│       │    │  │ Proofs: 147,832 ✓       │                       │
│       │    │  │ USDC spent: 7,391.60    │                       │
│       │    │  └─────────────────────────┘                       │
│       │    │             │                                       │
│       │    │             │ Context for                           │
│       │    │             │ future classifications                │
│       │    │             │                                       │
│       │    └─────────────┼───────────────┘                       │
│       │                  │                                       │
│       │                  ▼                                       │
│       │         (recursive feedback)                             │
│       │                                                          │
│       └──────────────── LOOP FOREVER ────────────────────────────┘
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Agents

### 1. Scout Agent

**Purpose:** Continuously discover URLs to analyze

**Sources:**
- PhishTank API (known phishing reports)
- OpenPhish feed
- Certificate Transparency logs (newly registered domains)
- URLhaus (malware URLs)
- Twitter/X security accounts
- Paste sites (Pastebin, etc.)
- Spam trap honeypots (simulated for demo)

**Behavior:**
```
every 5 minutes:
    urls = fetch_from_sources()
    novel_urls = filter_already_classified(urls)
    if len(novel_urls) > 0:
        batch = create_batch(novel_urls)
        send_to_policy_agent(batch)
```

**Outputs:** Batches of 10-100 URLs

**Proof required:** No (just discovery, no computation to verify)

**Paid:** No (Scout is the initiator, funded by treasury)

---

### 2. Policy Agent

**Purpose:** Authorize spending on URL classification

**Inputs:**
- Batch size (number of URLs)
- Current budget remaining
- URL novelty score (how different from existing DB)
- Source reputation (PhishTank vs random paste)
- Time since last batch
- Current threat level (are we seeing a spike?)

**Decision:** APPROVED or DENIED

**Jolt Atlas Model:** `authorization.onnx`
- Same model from Jolt Atlas examples
- Input: 64 features encoding the above
- Output: binary classification + confidence

**Proof required:** YES (zkML)
- Proves the authorization decision was actually computed
- Prevents rubber-stamping or arbitrary denials

**Paid:** 0.10 USDC per decision (paid by Scout/treasury)

---

### 3. Analyst Agent

**Purpose:** Classify URLs as PHISHING / SAFE / SUSPICIOUS

**Inputs:**
- URL string
- Extracted features:
  - Domain age
  - TLD risk (.xyz, .top, .tk = higher risk)
  - Path entropy
  - Query parameter count
  - SSL certificate info
  - WHOIS data (if available)
- **Context from database:**
  - Similar domains seen before
  - Same registrar's phishing rate
  - Same IP's history
  - Same ASN's history

**Outputs:**
- Classification: PHISHING / SAFE / SUSPICIOUS
- Confidence: 0.0 - 1.0
- Features used (for explainability)

**Jolt Atlas Model:** `url_classifier.onnx`
- Need to train or adapt from article classifier
- Input: URL features (numeric)
- Output: 3-class classification

**Proof required:** YES (zkML)
- Proves the classification was actually computed by the model
- Prevents returning cached/fake results

**Paid:** 0.05 USDC per URL (paid by Scout/treasury after Policy approval)

---

### 4. Database (not an agent, but critical)

**Stores:**
```sql
CREATE TABLE classifications (
    id UUID PRIMARY KEY,
    url TEXT NOT NULL,
    domain TEXT NOT NULL,
    classification TEXT NOT NULL,  -- PHISHING, SAFE, SUSPICIOUS
    confidence FLOAT NOT NULL,

    -- Proof data
    proof_hash TEXT NOT NULL,
    model_commitment TEXT NOT NULL,
    input_commitment TEXT NOT NULL,
    output_commitment TEXT NOT NULL,

    -- Context
    features JSONB,
    context_used JSONB,  -- What DB context influenced this

    -- Metadata
    source TEXT,  -- Where Scout found it
    batch_id UUID,
    classified_at TIMESTAMP DEFAULT NOW(),

    -- Economics
    analyst_paid FLOAT,
    policy_proof_hash TEXT
);

CREATE TABLE batches (
    id UUID PRIMARY KEY,
    url_count INT,
    source TEXT,
    policy_decision TEXT,
    policy_proof_hash TEXT,
    policy_paid FLOAT,
    total_analyst_paid FLOAT,
    created_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP
);

CREATE TABLE domain_stats (
    domain TEXT PRIMARY KEY,
    times_seen INT DEFAULT 0,
    phishing_count INT DEFAULT 0,
    safe_count INT DEFAULT 0,
    suspicious_count INT DEFAULT 0,
    first_seen TIMESTAMP,
    last_seen TIMESTAMP
);

CREATE TABLE registrar_stats (
    registrar TEXT PRIMARY KEY,
    domains_seen INT DEFAULT 0,
    phishing_rate FLOAT DEFAULT 0,
    last_updated TIMESTAMP
);

CREATE TABLE ip_stats (
    ip TEXT PRIMARY KEY,
    domains_hosted INT DEFAULT 0,
    phishing_rate FLOAT DEFAULT 0,
    last_updated TIMESTAMP
);
```

**The compounding effect:** As this database grows, the context available for future classifications improves.

---

## The Flow (One Iteration)

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. SCOUT DISCOVERS                                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Scout fetches from PhishTank API                              │
│  → 47 new URLs found                                           │
│  → 12 already in database (skip)                               │
│  → 35 novel URLs ready for analysis                            │
│                                                                 │
│  Batch created: batch-7f3a2b                                   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. POLICY AUTHORIZATION (zkML Proof #1)                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Scout → Policy Agent (A2A):                                   │
│    "Authorize 35 URLs, cost: 1.75 USDC"                        │
│                                                                 │
│  Policy Agent:                                                  │
│    Input features:                                              │
│      batch_size: 35                                            │
│      budget_remaining: 847.25 USDC                             │
│      novelty_score: 0.89 (mostly new domains)                  │
│      source_reputation: 0.95 (PhishTank is trusted)            │
│      time_since_last: 312 seconds                              │
│      threat_level: 0.67 (moderate activity)                    │
│                                                                 │
│    [Run authorization model via Jolt Atlas]                    │
│    [Generate zkML proof - ~20 seconds]                         │
│                                                                 │
│    Decision: AUTHORIZED (confidence: 0.91)                     │
│    Proof: 0x7f3a8b2c...                                        │
│                                                                 │
│  Scout verifies proof → Valid ✓                                │
│  Scout pays Policy Agent: 0.10 USDC                            │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. URL CLASSIFICATION (zkML Proof #2)                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Scout → Analyst Agent (A2A + x402):                           │
│    POST /skills/classify-urls                                  │
│    Body: {urls: [...35 URLs...], batch_id: "batch-7f3a2b"}    │
│                                                                 │
│  Analyst Agent:                                                 │
│    HTTP 402 Payment Required                                   │
│    Amount: 1.75 USDC (35 × 0.05)                              │
│                                                                 │
│  Scout pays via x402: 1.75 USDC                                │
│                                                                 │
│  Analyst Agent processes each URL:                             │
│                                                                 │
│    For URL "http://paypa1-secure.xyz/login":                   │
│      Features extracted:                                        │
│        domain_age: 2 days                                      │
│        tld_risk: 0.85 (.xyz)                                   │
│        path_entropy: 0.23                                      │
│        has_brand_name: true (paypal)                           │
│        typosquat_distance: 1 (paypa1 vs paypal)               │
│                                                                 │
│      Context from database:                                     │
│        similar_domains_phish_rate: 0.92                        │
│        registrar_phish_rate: 0.34                              │
│        ip_phish_rate: 0.78                                     │
│                                                                 │
│      [Run classifier via Jolt Atlas]                           │
│      Classification: PHISHING (confidence: 0.97)               │
│                                                                 │
│    [After all 35 URLs processed]                               │
│    [Generate batch zkML proof - ~30 seconds]                   │
│                                                                 │
│    Results:                                                     │
│      PHISHING: 8                                               │
│      SAFE: 24                                                  │
│      SUSPICIOUS: 3                                             │
│    Proof: 0x8b2c9d4e...                                        │
│                                                                 │
│  Scout verifies proof → Valid ✓                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. DATABASE UPDATE                                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  35 new classifications stored                                  │
│  Domain stats updated                                          │
│  Registrar stats updated                                        │
│  IP stats updated                                              │
│                                                                 │
│  Database now has:                                              │
│    Total URLs: 147,867 (+35)                                   │
│    Total phishing: 24,299 (+8)                                 │
│    Total proofs: 147,867 ✓                                     │
│                                                                 │
│  Context improved for future classifications ↑                 │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. LOOP                                                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Wait 5 minutes...                                             │
│  Scout fetches more URLs...                                    │
│  Repeat forever.                                               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## The Compounding Effect

### Week 1: Cold Start

```
URLs classified: 2,000
Database context: minimal
Classification relies mostly on: URL features alone
Accuracy: ~82%

Example:
  URL: http://amaz0n-verify.com/account
  Features: typosquat, new domain, suspicious TLD
  Context: none (never seen similar)
  Classification: PHISHING (0.79 confidence)
```

### Month 1: Building Mass

```
URLs classified: 50,000
Database context: growing
Classification now uses: URL features + historical patterns

New capabilities:
  - "This registrar has 45% phishing rate"
  - "This IP hosts 12 other phishing sites"
  - "23 similar typosquat domains were phishing"

Accuracy: ~89%

Example:
  URL: http://netfl1x-billing.top/update
  Features: typosquat, new domain, suspicious TLD
  Context:
    - Same registrar: 67% phishing rate
    - Same IP: 4 confirmed phishing sites
    - Similar domains: 8/8 were phishing
  Classification: PHISHING (0.96 confidence)
```

### Month 3: Network Effects

```
URLs classified: 300,000
Database context: rich

New capabilities:
  - Campaign detection (cluster similar URLs)
  - Infrastructure mapping (shared hosting patterns)
  - Temporal patterns (phishing spikes on Fridays)
  - Predictive flagging (domains registered but not yet weaponized)

Accuracy: ~94%

Emerging intelligence:
  - 847 phishing campaigns identified
  - 23 threat actor infrastructure clusters
  - Top 10 bulletproof hosting providers
  - Registrar risk rankings
```

### Month 6+: Serious Asset

```
URLs classified: 1,000,000+
Database context: comprehensive

The database IS the product:
  - Real-time phishing feed (API)
  - Threat actor attribution
  - Predictive domain blocking
  - Brand impersonation monitoring

Could compete with:
  - VirusTotal
  - PhishLabs
  - Proofpoint

All classifications cryptographically verified.
No trust required.
```

---

## What zkML Proves (And Why It's "Must Have")

### Without zkML

```
Analyst Agent could:
  ✗ Return "SAFE" for everything (save compute)
  ✗ Return cached results for new URLs
  ✗ Use a cheaper/worse model
  ✗ Skip the model entirely, guess randomly
  ✗ Be bribed to whitelist phishing domains

Policy Agent could:
  ✗ Approve everything (no budget control)
  ✗ Deny competitors' batches
  ✗ Approve friends' batches regardless of quality

Result: Database full of garbage, no trust in the intel
```

### With zkML

```
Every classification has a proof that:
  ✓ This exact model (commitment: 0xabc...) was used
  ✓ These exact features were input
  ✓ This exact output was produced
  ✓ The computation is reproducible

Every authorization has a proof that:
  ✓ The policy model made the decision
  ✓ Budget constraints were checked
  ✓ The decision matches the inputs

Result:
  - Every entry in database is verifiable
  - Auditors can check any classification
  - No trust in agents required
  - Threat intel is cryptographically sound
```

---

## UI Dashboard

### Main View

```
┌─────────────────────────────────────────────────────────────────┐
│  THREAT INTELLIGENCE NETWORK                    Running: 47d   │
│  A2A + x402 + Jolt Atlas zkML                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │   SCOUT      │  │   POLICY     │  │   ANALYST    │          │
│  │              │  │              │  │              │          │
│  │  ● Active    │  │  ● Active    │  │  ◐ Working   │          │
│  │              │  │              │  │              │          │
│  │  Sources: 5  │  │  Model:      │  │  Model:      │          │
│  │  URLs/hr: 89 │  │  authz.onnx  │  │  classify.   │          │
│  │              │  │              │  │  onnx        │          │
│  │  Last: 2m    │  │  Approved:   │  │              │          │
│  │              │  │  1,247 today │  │  Classified: │          │
│  │              │  │              │  │  1,189 today │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  DATABASE GROWTH                                                │
│                                                                 │
│  Total URLs        Phishing         Safe           Suspicious   │
│    247,832          41,291        198,847            7,694      │
│     (+1,189)         (+203)        (+947)            (+39)      │
│                                                                 │
│  ████████████████████████████████████████████████░░░░░░░░░░    │
│  0                                                      500K    │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  ECONOMICS                                                      │
│                                                                 │
│  Treasury Balance    Total Spent      Policy Paid   Analyst Paid│
│     4,721.35          12,391.60          1,247.83    11,143.77  │
│                                                                 │
│  Cost per URL: $0.05   Cost per batch: $1.85 avg               │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  LIVE FEED                                                      │
│                                                                 │
│  12:47:03  ✓ Batch-8f2a classified: 35 URLs (8 phishing)       │
│  12:47:02  ✓ Work proof verified (143ms)                       │
│  12:46:58  ◐ Analyst processing batch-8f2a...                  │
│  12:46:42  $ Paid Analyst 1.75 USDC (x402)                     │
│  12:46:41  ✓ Policy proof verified (143ms)                     │
│  12:46:21  ◐ Policy evaluating batch-8f2a...                   │
│  12:46:18  → Scout found 35 new URLs (PhishTank)               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Patterns View (After Weeks of Running)

```
┌─────────────────────────────────────────────────────────────────┐
│  EMERGING PATTERNS                              Last 30 days    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  TOP PHISHING TLDs                TOP REGISTRARS (by phish %)  │
│  ┌────────────────────────┐       ┌────────────────────────┐   │
│  │ .xyz     ████████ 34%  │       │ NameSilo    ███████ 45%│   │
│  │ .top     ██████   28%  │       │ Porkbun     █████   31%│   │
│  │ .tk      █████    21%  │       │ Namecheap   ████    24%│   │
│  │ .ml      ███      12%  │       │ GoDaddy     ██      11%│   │
│  │ other    ██        5%  │       │ other       █        8%│   │
│  └────────────────────────┘       └────────────────────────┘   │
│                                                                 │
│  CAMPAIGN CLUSTERS                                              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Cluster #47: PayPal phishing (847 URLs, 12 IPs)         │   │
│  │ Cluster #89: Netflix credential harvest (234 URLs)      │   │
│  │ Cluster #92: Crypto wallet drainer (156 URLs, 3 IPs)    │   │
│  │ [View all 127 clusters]                                  │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ACCURACY IMPROVEMENT (compounding effect)                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                                              ┌───┐      │   │
│  │                                         ┌────┤94%│      │   │
│  │                                    ┌────┤    └───┘      │   │
│  │                               ┌────┤    │               │   │
│  │                          ┌────┤    │    │               │   │
│  │    ┌────┐           ┌────┤    │    │    │               │   │
│  │    │82% │      ┌────┤    │    │    │    │               │   │
│  │    └────┘      │    │    │    │    │    │               │   │
│  │    Day 1    Week 1  Mo 1  Mo 2  Mo 3  Now               │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Proof Explorer

```
┌─────────────────────────────────────────────────────────────────┐
│  PROOF EXPLORER                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Classification: http://paypa1-secure.xyz/login                │
│  Result: PHISHING (confidence: 0.97)                           │
│  Timestamp: 2024-01-15 12:47:03                                │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ WORK PROOF (zkML)                          ✓ Verified   │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │ Model commitment: 0x7a8b3c2d...                         │   │
│  │ Input commitment: 0x9f8e7d6c...                         │   │
│  │ Output commitment: 0x1a2b3c4d...                        │   │
│  │                                                          │   │
│  │ Public inputs:                                          │   │
│  │   domain_age_days: 2                                    │   │
│  │   tld_risk_score: 0.85                                  │   │
│  │   typosquat_distance: 1                                 │   │
│  │   registrar_phish_rate: 0.34                            │   │
│  │   ip_phish_rate: 0.78                                   │   │
│  │   similar_domains_phish_rate: 0.92                      │   │
│  │                                                          │   │
│  │ Proof size: 1.2 KB                                      │   │
│  │ Prove time: 28.4s                                       │   │
│  │ Verify time: 143ms                                      │   │
│  │                                                          │   │
│  │ [Download proof]  [Verify independently]                │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ POLICY PROOF (zkML)                        ✓ Verified   │   │
│  ├─────────────────────────────────────────────────────────┤   │
│  │ Batch: batch-7f3a2b (35 URLs)                           │   │
│  │ Decision: AUTHORIZED                                    │   │
│  │ Budget before: 849.00 USDC                              │   │
│  │ Cost: 1.85 USDC                                         │   │
│  │                                                          │   │
│  │ [Expand details]                                        │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Component | Technology | Notes |
|-----------|------------|-------|
| **Scout Agent** | Python + FastAPI | Async URL fetching |
| **Policy Agent** | Python + FastAPI + Jolt Atlas | zkML proof generation |
| **Analyst Agent** | Python + FastAPI + Jolt Atlas | zkML proof generation |
| **Database** | PostgreSQL | Could use SQLite for MVP |
| **Message Queue** | Redis | Batch coordination |
| **UI** | Next.js + Tailwind | Real-time WebSocket |
| **zkML** | Jolt Atlas (Rust) | Called via CLI wrapper |
| **A2A** | Custom implementation | Based on Google A2A spec |
| **x402** | Custom implementation | Mock USDC for demo |
| **Hosting** | Railway / Docker | Long-running services |

---

## Project Structure

```
threat-intel-network/
├── agents/
│   ├── scout/
│   │   ├── main.py              # Scout agent server
│   │   ├── sources/
│   │   │   ├── phishtank.py     # PhishTank API client
│   │   │   ├── openphish.py     # OpenPhish feed
│   │   │   ├── urlhaus.py       # URLhaus feed
│   │   │   ├── ct_logs.py       # Certificate Transparency
│   │   │   └── honeypot.py      # Simulated spam trap
│   │   ├── agent_card.json
│   │   └── requirements.txt
│   │
│   ├── policy/
│   │   ├── main.py              # Policy agent server
│   │   ├── prover.py            # Jolt Atlas wrapper
│   │   ├── models/
│   │   │   └── authorization.onnx
│   │   ├── agent_card.json
│   │   └── requirements.txt
│   │
│   └── analyst/
│       ├── main.py              # Analyst agent server
│       ├── prover.py            # Jolt Atlas wrapper
│       ├── features.py          # URL feature extraction
│       ├── context.py           # Database context lookup
│       ├── models/
│       │   └── url_classifier.onnx
│       ├── agent_card.json
│       └── requirements.txt
│
├── jolt-atlas/                  # Cloned Jolt Atlas repo
│   └── ...
│
├── database/
│   ├── schema.sql               # PostgreSQL schema
│   ├── migrations/
│   └── seed.py                  # Initial data
│
├── shared/
│   ├── a2a.py                   # A2A protocol client
│   ├── x402.py                  # x402 payment handling
│   ├── types.py                 # Shared Pydantic models
│   ├── config.py                # Configuration
│   └── events.py                # WebSocket event broadcasting
│
├── ui/
│   ├── app/
│   │   ├── page.tsx             # Main dashboard
│   │   ├── patterns/page.tsx    # Patterns view
│   │   ├── proofs/[id]/page.tsx # Proof explorer
│   │   └── layout.tsx
│   ├── components/
│   │   ├── AgentStatus.tsx
│   │   ├── DatabaseStats.tsx
│   │   ├── LiveFeed.tsx
│   │   ├── ProofCard.tsx
│   │   ├── PatternChart.tsx
│   │   └── EconomicsPanel.tsx
│   ├── hooks/
│   │   └── useWebSocket.ts
│   └── package.json
│
├── scripts/
│   ├── train_classifier.py      # Train URL classifier
│   ├── export_onnx.py           # Export to ONNX
│   └── generate_test_data.py    # Synthetic URLs for testing
│
├── docker-compose.yml
├── Makefile
└── README.md
```

---

## Build Plan

### Phase 1: Foundation (Days 1-4)

#### Day 1: Environment Setup
```
□ Clone jolt-atlas repo
□ Build Jolt Atlas (cargo build --release)
□ Run authorization example, understand API
□ Document input/output formats
□ Set up project structure
```

#### Day 2: URL Classifier Model
```
□ Gather training data:
   - PhishTank historical data
   - Legitimate URL samples (Alexa top sites)
□ Define features:
   - Domain age, TLD, path entropy, etc.
□ Train simple classifier (logistic regression or small NN)
□ Export to ONNX format
□ Test with Jolt Atlas
```

#### Day 3: Prover Wrapper
```
□ Create Python CLI wrapper for Jolt Atlas
□ Implement generate_proof(model, inputs)
□ Implement verify_proof(proof, public_inputs)
□ Test both authorization and classifier models
□ Measure timing (prove time, verify time)
```

#### Day 4: Database Setup
```
□ Create PostgreSQL schema
□ Implement database client
□ Create context lookup functions:
   - get_similar_domains()
   - get_registrar_stats()
   - get_ip_stats()
□ Test context enrichment
```

### Phase 2: Agents (Days 5-9)

#### Day 5: Scout Agent
```
□ Implement FastAPI server
□ Create PhishTank source (API client)
□ Create synthetic honeypot source (for demo)
□ Implement batch creation logic
□ Implement deduplication (skip known URLs)
□ Test standalone
```

#### Day 6: Policy Agent
```
□ Implement FastAPI server
□ Create A2A agent card
□ Implement authorization endpoint
□ Integrate Jolt Atlas prover
□ Test: input → decision + proof
```

#### Day 7: Analyst Agent
```
□ Implement FastAPI server
□ Create A2A agent card
□ Implement x402 payment gate
□ Implement feature extraction
□ Implement context lookup
□ Integrate Jolt Atlas prover
□ Test: URLs → classifications + proof
```

#### Day 8: A2A Integration
```
□ Implement A2A discovery
□ Implement skill invocation
□ Wire Scout → Policy flow
□ Wire Scout → Analyst flow
□ Test agent-to-agent communication
```

#### Day 9: x402 Integration
```
□ Implement 402 response generation
□ Implement payment verification
□ Implement mock USDC treasury
□ Wire payment into Analyst flow
□ Test full payment cycle
```

### Phase 3: Integration (Days 10-12)

#### Day 10: End-to-End Flow
```
□ Run full loop once manually
□ Debug issues
□ Add comprehensive logging
□ Test proof verification at each step
```

#### Day 11: Continuous Loop
```
□ Implement Scout scheduler (every 5 min)
□ Implement error handling / retry logic
□ Implement graceful shutdown
□ Test running for 1 hour continuously
```

#### Day 12: Database Growth
```
□ Implement stats aggregation (domain, registrar, IP)
□ Implement context improvement over time
□ Add indexes for performance
□ Test with 1000+ URLs
```

### Phase 4: UI (Days 13-16)

#### Day 13: Basic Dashboard
```
□ Set up Next.js project
□ Create main layout
□ Implement WebSocket connection
□ Show agent status (active/working/idle)
□ Show database stats (totals)
```

#### Day 14: Live Feed
```
□ Implement event streaming from agents
□ Create LiveFeed component
□ Show real-time activity
□ Show proof status (generating/verified/failed)
```

#### Day 15: Economics & Patterns
```
□ Create economics panel (treasury, spent, per-agent)
□ Create patterns view (TLD chart, registrar chart)
□ Implement historical data queries
□ Show compounding effect (accuracy over time)
```

#### Day 16: Proof Explorer
```
□ Create proof detail page
□ Show public inputs
□ Show verification status
□ Allow independent verification
```

### Phase 5: Polish (Days 17-20)

#### Day 17: Failure Modes
```
□ Add "simulate cheating" toggles
□ Implement fake proof generation
□ Show verification failure in UI
□ Document why zkML is "must have"
```

#### Day 18: Performance Tuning
```
□ Optimize proof generation (batching?)
□ Add caching where appropriate
□ Database query optimization
□ Test sustained load
```

#### Day 19: Docker & Deployment
```
□ Create Dockerfiles for each agent
□ Create docker-compose.yml
□ Test full system in containers
□ Deploy to Railway / Render
```

#### Day 20: Documentation & Demo
```
□ Write README
□ Create demo script
□ Record demo video
□ Document API endpoints
```

---

## Timeline Summary

| Phase | Days | Deliverable |
|-------|------|-------------|
| 1. Foundation | 1-4 | Working Jolt Atlas + classifier + database |
| 2. Agents | 5-9 | Three agents with A2A + x402 |
| 3. Integration | 10-12 | Continuous loop running |
| 4. UI | 13-16 | Real-time dashboard |
| 5. Polish | 17-20 | Deployed, documented demo |

**Total: ~20 days (4 weeks)**

---

## Success Criteria

### Technical
- [ ] Scout discovers URLs from multiple sources
- [ ] Policy Agent authorizes with zkML proof
- [ ] Analyst Agent classifies with zkML proof
- [ ] x402 payment flow works
- [ ] A2A discovery works
- [ ] Database grows over time
- [ ] Context improves classifications
- [ ] System runs continuously for 24+ hours

### Demo
- [ ] UI shows real-time agent activity
- [ ] UI shows database growth
- [ ] UI shows proof verification
- [ ] UI shows economic flows
- [ ] Failure mode demonstrates cheating detection
- [ ] Patterns visible after running for hours/days

### Compounding
- [ ] Classification accuracy measurably improves over time
- [ ] Context data (registrar stats, etc.) accumulates
- [ ] Campaign clusters emerge
- [ ] System becomes more valuable the longer it runs

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Jolt Atlas API changes | Pin to specific commit |
| Proof generation too slow | Batch proofs, async processing |
| URL sources rate limited | Multiple sources, caching, backoff |
| Database grows too large | Archival strategy, pruning old data |
| Model accuracy poor | Start with simple model, iterate |

---

## Future Extensions (Post-MVP)

1. **More sources**: Twitter, paste sites, CT logs
2. **Real USDC**: Integrate with Base / Coinbase
3. **API access**: Expose threat intel via paid API
4. **Campaign tracking**: Automated clustering
5. **Predictive blocking**: Flag domains before weaponized
6. **Multi-agent competition**: Multiple Analysts compete on accuracy
7. **Reputation system**: Track agent performance over time
