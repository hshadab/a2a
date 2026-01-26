# Threat Intelligence Network

A long-running, compounding demonstration of **A2A + x402 + Jolt Atlas zkML**.

Two autonomous agents continuously build a threat intelligence database by discovering and classifying URLs. Each agent self-authorizes its spending with zkML proofs, creating a trustless circular micro-economy.

## Live Demo

- **Dashboard**: https://threat-intel-ui.onrender.com
- **Scout Agent**: https://threat-intel-scout.onrender.com
- **Analyst Agent**: https://threat-intel-analyst.onrender.com

## The Vision

```
Day 1:     1,000 URLs → Small database
Day 30:   50,000 URLs → Patterns emerging
Day 90:  300,000 URLs → Real threat intel
Day 180: 1,000,000 URLs → Competitive product
```

Set it up. Walk away. Come back and watch it grow.

## Architecture

```
┌─────────────────────┐         ┌─────────────────────┐
│     URL SCOUT       │◄───────►│   THREAT ANALYST    │
│                     │  A2A    │                     │
│  • Discovers URLs   │  +      │  • Classifies URLs  │
│  • Scores quality   │  x402   │  • Returns verdict  │
│                     │         │                     │
│  zkML Proofs:       │         │  zkML Proofs:       │
│  ├─ Spending Auth   │         │  ├─ Spending Auth   │
│  └─ Quality Score   │         │  └─ Classification  │
│                     │         │                     │
│  Wallet: 0x269C...  │         │  Wallet: 0x7ee8...  │
└─────────────────────┘         └─────────────────────┘
         │                               │
         │      $0.001 / batch           │
         └───────────────────────────────┘
                 (circular flow)

                      │
                      ▼
           ┌──────────────────────┐
           │       DATABASE       │
           │                      │
           │  URLs: 147,832       │
           │  Phishing: 24,291    │
           │  Safe: 119,847       │
           │  Proofs: 147,832 ✓   │
           └──────────────────────┘
```

### Payment Flow

The agents form a circular micro-economy:
- **Analyst → Scout**: $0.001 for URL discovery batches
- **Scout → Analyst**: $0.001 for classification feedback

Net USDC movement: $0 (only gas consumed)
Real value created: A growing, zkML-verified threat intelligence database

## Technologies

| Component | Technology |
|-----------|------------|
| Agent Communication | **A2A v0.3** (Google Agent-to-Agent Protocol) |
| Payments | **x402** on Base Mainnet (USDC) |
| Proofs | **Jolt Atlas zkML** |
| Database | PostgreSQL |
| UI | Next.js + Tailwind |
| Deployment | Render |

## Three Types of zkML Proofs

### 1. Spending Authorization (Both Agents)
- Proves spending is within policy limits
- Uses `authorization.onnx` model (7 inputs → 2 outputs)
- Input: `[amount, balance, daily_spent, hourly_spent, trust_score, hour, is_weekday]`
- Without proof: Agents could overspend or drain wallets

### 2. Quality Score (Scout)
- Proves URL quality tier was computed correctly
- Uses `quality_scorer.onnx` model (32 inputs → 4 outputs)
- Output: HIGH / MEDIUM / LOW / NOISE
- Without proof: Scout could claim high-quality sources

### 3. Classification (Analyst)
- Proves URL classification was actually performed
- Uses `url_classifier.onnx` model (32 inputs → 3 outputs)
- Output: PHISHING / SUSPICIOUS / SAFE
- Without proof: Analyst could return fake classifications

## Quick Start

### Prerequisites
- Docker and Docker Compose
- Node.js 18+ (for UI development)
- Python 3.11+ (for local development)

### 1. Clone and Configure

```bash
cd threat-intel-network

# Install dependencies
make setup

# Configure environment
cp .env.example .env
# Edit .env with your configuration
```

### 2. Install zkML Prover (for real proofs)

```bash
# Download and install the Jolt Atlas zkML prover
make setup-zkml

# Verify installation
make validate
```

The zkML prover is a ~143MB binary that generates cryptographic proofs.
Without it, the system runs in demo mode with simulated proofs.

### 3. Run with Docker

```bash
make build
make run
```

Services:
- **Scout Agent**: http://localhost:8000
- **Analyst Agent**: http://localhost:8001
- **UI Dashboard**: http://localhost:3001

### 4. Watch It Work

Open http://localhost:3001 to see the real-time dashboard.

The system will:
1. Scout discovers URLs from threat feeds (OpenPhish, URLhaus, CT logs)
2. Scout generates spending authorization proof
3. Scout pays Analyst via x402 (USDC on Base)
4. Analyst classifies URLs with zkML proof
5. Scout verifies proof, pays feedback reward
6. Database grows, patterns emerge

## Demo vs Production Mode

### Demo Mode (default)
- Simulated zkML proofs (fast, no binary needed)
- Simulated USDC payments (no wallet needed)
- Falls back to in-memory database if PostgreSQL unavailable
- Perfect for testing and development

### Production Mode
Set `PRODUCTION_MODE=true` in your `.env` file.

**Requirements:**
- zkML prover binary installed (`make setup-zkml`)
- PostgreSQL database configured
- Wallet with USDC on Base mainnet
- All proofs and payments are real

```bash
# Validate production requirements
make validate
```

## Pricing (Configurable)

| Payment | Amount | Direction |
|---------|--------|-----------|
| Discovery | $0.001/batch | Analyst → Scout |
| Feedback | $0.001/batch | Scout → Analyst |

For a batch of 50 URLs: ~$0.002 total (net $0)

## Project Structure

```
threat-intel-network/
├── agents/
│   ├── scout/           # URL discovery + quality scoring
│   │   ├── main.py      # FastAPI server with A2A endpoints
│   │   ├── sources/     # URL sources (OpenPhish, URLhaus, CT logs)
│   │   └── models/      # Quality scorer ONNX model
│   └── analyst/         # URL classification
│       ├── main.py      # FastAPI server with A2A endpoints
│       ├── features.py  # Feature extraction
│       └── models/      # Classifier ONNX model
├── shared/
│   ├── a2a.py           # A2A protocol implementation
│   ├── x402.py          # x402 payment (Base mainnet USDC)
│   ├── database.py      # PostgreSQL client
│   ├── prover.py        # Jolt Atlas zkML wrapper
│   ├── events.py        # WebSocket broadcasting
│   └── config.py        # Environment configuration
├── ui/                  # Next.js dashboard
│   ├── app/             # Next.js 13+ app router
│   └── components/      # React components
├── jolt-atlas/          # zkML prover (submodule)
├── docker-compose.yml
├── render.yaml          # Render deployment config
└── Makefile
```

## URL Sources

| Source | Type | Reputation |
|--------|------|------------|
| OpenPhish | Aggregated phishing feed | 0.90 |
| URLhaus | Malware URL feed | 0.85 |
| Certificate Transparency | Real-time cert monitoring | 0.65 |
| Typosquat Detection | Brand impersonation | 0.70 |

## Development

### Local Development

```bash
# Start dependencies
docker-compose up db redis -d

# Run agents locally (with hot reload)
make dev

# Or run UI only
make dev-ui
```

### Manual Batch Trigger

```bash
make trigger
# or
curl -X POST http://localhost:8000/trigger
```

### Check Health

```bash
make health
```

### View Stats

```bash
make stats
```

## The Compounding Effect

| Time | URLs | What Happens |
|------|------|--------------|
| Day 1 | 1,000 | Cold start, basic features only |
| Week 1 | 10,000 | Registrar patterns emerging |
| Month 1 | 50,000 | "This IP hosts 12 phishing sites" |
| Month 3 | 300,000 | Campaign clusters, threat actor infra |
| Month 6 | 1,000,000+ | Competitive threat intel product |

## Why zkML is "Must Have"

### Without Proofs
- Agents could approve all spending (drain wallets)
- Scout could claim fake high-quality scores
- Analyst could return fake classifications
- No way to verify work was done
- Database could be full of garbage

### With zkML Proofs
- Every spending decision is cryptographically verified
- Every quality score is provably correct
- Every classification is verifiable
- Model commitment binds exact computation
- Trustless agent collaboration

## x402 Payment Flow

```
Scout → Analyst: POST /skills/classify-urls
Analyst → Scout: HTTP 402 Payment Required
                 X-402-Payment: {amount: "0.001", token: "USDC", chain: 8453}
Scout: [Generates spending proof, makes USDC transfer on Base]
Scout → Analyst: POST /skills/classify-urls
                 X-402-Receipt: 0x7f3a8b2c...
Analyst: [Verifies payment, verifies spending proof, processes request]
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | PostgreSQL connection string |
| `REDIS_URL` | Redis connection string |
| `PRIVATE_KEY` | Wallet private key for payments |
| `ANALYST_URL` | Analyst agent endpoint |
| `SCOUT_URL` | Scout agent endpoint |
| `BASE_RPC_URL` | Base mainnet RPC endpoint |
| `PRODUCTION_MODE` | Enable real proofs and payments |

## API Endpoints

### Scout Agent
- `GET /.well-known/agent.json` - A2A agent card
- `POST /skills/discover-url` - Discover and return URLs
- `GET /health` - Health check
- `GET /stats` - Processing statistics

### Analyst Agent
- `GET /.well-known/agent.json` - A2A agent card
- `POST /skills/classify-urls` - Classify URLs (x402 payment required)
- `GET /health` - Health check

## Wallet Addresses

| Agent | Address |
|-------|---------|
| Scout | `0x269CBA662fE55c4fe1212c609090A31844C36ab8` |
| Analyst | `0x7ee88871fA9be48b62552F231a4976A11e559db8` |

View on BaseScan: [Scout](https://basescan.org/address/0x269CBA662fE55c4fe1212c609090A31844C36ab8) | [Analyst](https://basescan.org/address/0x7ee88871fA9be48b62552F231a4976A11e559db8)

## License

MIT
