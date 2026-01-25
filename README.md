# Threat Intelligence Network

A long-running, compounding demonstration of **A2A + x402 + Jolt Atlas zkML**.

Agents continuously build a threat intelligence database by classifying URLs. The system runs for days/weeks, gets more valuable over time, and demonstrates why zkML is "must have" for trustless agent collaboration.

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
┌─────────┐      ┌─────────┐      ┌─────────┐
│  SCOUT  │ ───► │ POLICY  │ ───► │ ANALYST │
│         │ A2A  │         │ A2A  │         │
│ Finds   │      │Approves │      │Classifies│
│ URLs    │      │spending │      │URLs     │
│         │      │         │      │         │
│ No proof│      │ zkML #1 │      │ zkML #2 │
└─────────┘      └─────────┘      └─────────┘
     │                                 │
     │                                 │
     │         ┌──────────────────────┐│
     │         │       DATABASE       ││
     │         │                      ▼│
     │         │  URLs: 147,832       │
     │         │  Phishing: 24,291    │
     │         │  Safe: 119,847       │
     │         │  Proofs: 147,832 ✓   │
     │         └──────────────────────┘
     │                    │
     │                    │ Context improves
     │                    │ future classifications
     └────────────────────┴───────────────────────►
```

## Technologies

| Component | Technology |
|-----------|------------|
| Agent Communication | **A2A** (Google Agent-to-Agent Protocol) |
| Payments | **x402** on Base Mainnet (USDC) |
| Proofs | **Jolt Atlas zkML** |
| Database | PostgreSQL |
| UI | Next.js + Tailwind |

## Two Proofs, Two Trust Problems

### Proof #1: Policy (Authorization)
- Proves spending authorization was computed correctly
- Uses Jolt Atlas `authorization.onnx` model
- Without proof: Policy Agent could approve everything or deny competitors

### Proof #2: Work (Classification)
- Proves URL classification was actually performed
- Uses Jolt Atlas `url_classifier.onnx` model
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

The zkML prover is a 143MB binary that generates cryptographic proofs.
Without it, the system runs in demo mode with simulated proofs.

### 3. Run with Docker

```bash
make build
make run
```

Services:
- **Scout Agent**: http://localhost:8000
- **Policy Agent**: http://localhost:8001
- **Analyst Agent**: http://localhost:8002
- **UI Dashboard**: http://localhost:3001

### 4. Watch It Work

Open http://localhost:3001 to see the real-time dashboard.

The system will:
1. Scout discovers URLs every 5 minutes
2. Policy Agent authorizes with zkML proof
3. Scout pays Analyst via x402 (USDC on Base)
4. Analyst classifies with zkML proof
5. Database grows, context improves

## Demo vs Production Mode

The system has two operating modes:

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

| Agent | Price |
|-------|-------|
| Policy | $0.001 per authorization |
| Analyst | $0.0005 per URL |

For a batch of 50 URLs: ~$0.026 total

## Project Structure

```
threat-intel-network/
├── agents/
│   ├── scout/          # Discovers URLs, orchestrates pipeline
│   ├── policy/         # Authorization with zkML proofs
│   └── analyst/        # Classification with zkML proofs
├── shared/
│   ├── a2a.py          # A2A protocol implementation
│   ├── x402.py         # x402 payment (Base mainnet USDC)
│   ├── database.py     # PostgreSQL client
│   ├── prover.py       # Jolt Atlas wrapper
│   └── events.py       # WebSocket broadcasting
├── ui/                 # Next.js dashboard
├── docker-compose.yml
└── Makefile
```

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
- Policy Agent could approve everything (or deny competitors)
- Analyst could return fake classifications
- No way to verify work was done
- Database could be full of garbage

### With zkML Proofs
- Every authorization is cryptographically verified
- Every classification is provably correct
- Model commitment binds exact computation
- Trustless agent collaboration

## x402 Payment Flow

```
Scout → Analyst: POST /classify-urls
Analyst → Scout: HTTP 402 Payment Required
                 X-402-Payment: {amount: "0.025", token: "USDC", chain: 8453}
Scout: [Makes USDC transfer on Base]
Scout → Analyst: POST /classify-urls
                 X-402-Receipt: 0x7f3a8b2c...
Analyst: [Verifies payment, processes request]
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | PostgreSQL connection string |
| `REDIS_URL` | Redis connection string |
| `PRIVATE_KEY` | Wallet private key for payments |
| `TREASURY_ADDRESS` | Address to receive payments |
| `BASE_RPC_URL` | Base mainnet RPC endpoint |

## License

MIT
