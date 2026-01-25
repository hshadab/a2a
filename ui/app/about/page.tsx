'use client';

import { ExternalLink, Shield, Zap, DollarSign, Lock, Check, ArrowRight, Github, Search, Scale, Microscope, Bot, X } from 'lucide-react';

export default function AboutPage() {
  return (
    <div className="p-6 space-y-12">
      {/* Hero Section */}
      <section className="text-center py-12">
        <h1 className="text-4xl font-bold text-white mb-4">
          How <span className="bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">ThreatProof</span> Works
        </h1>
        <p className="text-xl text-gray-400 max-w-3xl mx-auto mb-6">
          An autonomous network of AI agents that discover, classify, and verify phishing threats
          using cryptographic proofs — powered by three groundbreaking protocols.
        </p>
        <div className="inline-flex items-center gap-3 px-6 py-3 bg-gradient-to-r from-cyan-500/10 to-purple-500/10 border border-cyan-500/30 rounded-full">
          <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
          <span className="text-cyan-400 font-medium">Autonomous Agent-to-Agent Commerce</span>
          <span className="text-gray-600">|</span>
          <span className="text-gray-400 text-sm">A self-sustaining AI micro-economy</span>
        </div>
      </section>

      {/* Architecture Overview */}
      <section className="card p-8">
        <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-3">
          <Shield className="text-cyan-400" />
          Architecture Overview
        </h2>

        {/* Agents shown in payment order: Analyst → Scout → Policy → Analyst */}
        <div className="grid grid-cols-3 gap-6 mb-8">
          <AgentBox
            icon={<Microscope size={24} />}
            name="URL Classifier"
            role="Customer & Orchestrator"
            color="#22d3ee"
            description="Pays Scout for URLs, classifies them as phishing/safe, and stores results. Receives feedback payments from Policy."
          />
          <AgentBox
            icon={<Search size={24} />}
            name="Threat Scout"
            role="URL Discovery Service"
            color="#3b82f6"
            description="Discovers threats from CT logs, typosquatting, and feeds. Receives from Analyst, pays Policy for authorization."
          />
          <AgentBox
            icon={<Scale size={24} />}
            name="Policy Agent"
            role="Authorization & Feedback"
            color="#a855f7"
            description="Authorizes spending with zkML proofs. Receives from Scout, pays Analyst for classification feedback."
          />
        </div>

        {/* Flow Diagram - Circular Economy */}
        <div className="bg-gray-800/50 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-4">Circular Payment Flow</h3>
          <div className="flex items-center justify-between text-sm">
            <FlowStep number={1} text="Analyst pays Scout $0.001" />
            <ArrowRight className="text-green-500" />
            <FlowStep number={2} text="Scout pays Policy $0.001" />
            <ArrowRight className="text-green-500" />
            <FlowStep number={3} text="Policy pays Analyst $0.001" />
            <ArrowRight className="text-cyan-500" />
            <FlowStep number={4} text="Net change: $0.00" />
          </div>
          <p className="text-center text-gray-500 text-xs mt-3">
            USDC circulates forever. Only gas (~$0.003 per batch) is consumed.
          </p>
        </div>
      </section>

      {/* Technology Deep Dives */}
      <div className="grid grid-cols-1 gap-8">

        {/* Google A2A Protocol */}
        <TechSection
          icon={<Bot className="text-blue-400" size={32} />}
          title="Google A2A Protocol v0.3"
          subtitle="Agent-to-Agent Communication"
          color="#4285f4"
          links={[
            { label: 'A2A Specification', url: 'https://github.com/google/A2A' },
            { label: 'Google Blog Post', url: 'https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/' },
          ]}
        >
          <p className="text-gray-300 mb-4">
            A2A (Agent-to-Agent) is Google's open protocol for AI agents to discover, communicate,
            and collaborate with each other. ThreatProof implements <strong className="text-blue-400">A2A v0.3</strong> with
            JSON-RPC 2.0 transport, task lifecycle management, and SSE streaming.
          </p>

          <h4 className="text-white font-semibold mb-2">A2A v0.3 Features:</h4>
          <ul className="space-y-2 text-gray-400">
            <FeatureItem text="Agent Cards v0.3: Protocol version, capabilities, and skill tags for discovery" />
            <FeatureItem text="JSON-RPC 2.0: Standard transport with task/send and task/get methods" />
            <FeatureItem text="Task Lifecycle: State machine (submitted → working → completed/failed)" />
            <FeatureItem text="SSE Streaming: Real-time task progress via Server-Sent Events" />
            <FeatureItem text="CAIP-2 Chains: Standard chain identifiers (eip155:8453 for Base)" />
          </ul>

          <CodeBlock title="A2A v0.3 Agent Card (Scout)" code={`// GET /.well-known/agent.json
{
  "name": "Threat Scout",
  "protocolVersion": "0.3",
  "capabilities": {
    "streaming": false,
    "stateTransitionHistory": true
  },
  "skills": [{
    "id": "discover-urls",
    "tags": ["discovery", "threat-intel", "zkml"],
    "inputModes": ["application/json"],
    "outputModes": ["application/json"],
    "price": {
      "amount": "0.0003",
      "currency": "USDC",
      "chain": "eip155:8453"
    }
  }]
}`} />

          <CodeBlock title="JSON-RPC 2.0 Request (Analyst → Scout)" code={`// POST /a2a (Analyst requests URL discovery from Scout)
{
  "jsonrpc": "2.0",
  "method": "task/send",
  "params": {
    "skillId": "discover-urls",
    "input": {
      "batch_size": 50,
      "source": "phishtank"
    }
  },
  "id": "req-1"
}
// Scout returns URLs + authorization proof + payment_due
// Analyst verifies proof, then pays Scout`} />
        </TechSection>

        {/* x402 Protocol */}
        <TechSection
          icon={<DollarSign className="text-green-400" size={32} />}
          title="x402 Payment Protocol v2"
          subtitle="HTTP 402 + Coinbase Facilitator"
          color="#22c55e"
          links={[
            { label: 'x402 v2 Spec', url: 'https://www.x402.org/writing/x402-v2-launch' },
            { label: 'Coinbase Developer Docs', url: 'https://docs.cdp.coinbase.com/x402/docs/welcome' },
          ]}
        >
          <p className="text-gray-300 mb-4">
            x402 brings the HTTP 402 "Payment Required" status code to life. ThreatProof implements
            <strong className="text-green-400"> x402 v2</strong> with standardized headers, base64-encoded payloads,
            and backwards compatibility with v1 clients.
          </p>

          {/* v2 Features Box */}
          <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-4 mb-4">
            <h5 className="text-green-400 font-medium mb-2 flex items-center gap-2">
              <Zap size={16} />
              x402 v2 Features
            </h5>
            <ul className="text-sm text-gray-400 space-y-1">
              <FeatureItem text="PAYMENT-REQUIRED header: Base64-encoded payment options" />
              <FeatureItem text="X-PAYMENT header: Client sends payment proof" />
              <FeatureItem text="Amount in base units: 1000000 = 1 USDC (6 decimals)" />
              <FeatureItem text="Multiple payment options: 'accepts' array for flexibility" />
            </ul>
          </div>

          {/* Coinbase Facilitator Box */}
          <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4 mb-4">
            <h5 className="text-blue-400 font-medium mb-2 flex items-center gap-2">
              <Zap size={16} />
              Coinbase x402 Facilitator
            </h5>
            <ul className="text-sm text-gray-400 space-y-1">
              <FeatureItem text="Fee-free payments for payers — facilitator covers gas" />
              <FeatureItem text="CAIP-2 chain identifiers (eip155:8453 for Base)" />
              <FeatureItem text="Payment intent API for seamless UX" />
              <FeatureItem text="Automatic receipt verification" />
            </ul>
          </div>

          <h4 className="text-white font-semibold mb-2">How ThreatProof Uses x402:</h4>
          <ul className="space-y-2 text-gray-400">
            <FeatureItem text="Value Chain: Analyst pays Scout (0.0003 USDC/URL), Scout pays Policy (0.001 USDC/auth)" />
            <FeatureItem text="USDC on Base: Fast, cheap transactions (~$0.001 gas) on Coinbase's L2" />
            <FeatureItem text="Proof-Gated: Work must be verified before payment is released" />
            <FeatureItem text="Backwards Compatible: Supports both v1 (X-402-*) and v2 (X-PAYMENT) headers" />
          </ul>

          <CodeBlock title="x402 v2 Payment Challenge" code={`// HTTP 402 Response
// Header: PAYMENT-REQUIRED: <base64 encoded JSON>
{
  "x402Version": 2,
  "accepts": [{
    "scheme": "exact",
    "network": "base-mainnet",
    "maxAmountRequired": "25000",  // 0.025 USDC in base units
    "resource": "/skills/classify-urls",
    "description": "Classify 50 URLs",
    "payTo": "0x6c67...",
    "asset": "0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913",
    "maxTimeoutSeconds": 300
  }]
}`} />

          <CodeBlock title="x402 v2 Payment Flow" code={`// 1. Client sends request
GET /skills/classify-urls

// 2. Server responds 402 with PAYMENT-REQUIRED header
HTTP/1.1 402 Payment Required
PAYMENT-REQUIRED: eyJ4NDAyVmVyc2lvbiI6MiwiYWNjZXB0cyI6...

// 3. Client makes payment, retries with X-PAYMENT header
POST /skills/classify-urls
X-PAYMENT: 0xabc123...  // Transaction hash

// 4. Server verifies payment, returns result
HTTP/1.1 200 OK
X-PAYMENT-RESPONSE: verified`} />
        </TechSection>

        {/* Jolt Atlas zkML */}
        <TechSection
          icon={<Lock className="text-purple-400" size={32} />}
          title="Jolt Atlas zkML"
          subtitle="Zero-Knowledge Machine Learning"
          color="#a855f7"
          links={[
            { label: 'GitHub Repository', url: 'https://github.com/ICME-Lab/jolt-atlas' },
            { label: 'Jolt Paper', url: 'https://eprint.iacr.org/2023/1217' },
          ]}
        >
          <p className="text-gray-300 mb-4">
            Jolt Atlas is a zkML framework from ICME Labs that generates cryptographic proofs
            of ML inference. It proves that a specific model produced a specific output for a
            specific input — without revealing the model weights or raw data.
          </p>

          <h4 className="text-white font-semibold mb-2">Why zkML is Critical:</h4>
          <div className="grid grid-cols-2 gap-4 mb-4">
            <div className="bg-red-500/10 border border-red-500/30 rounded-lg p-4">
              <h5 className="text-red-400 font-medium mb-2">Without Proofs</h5>
              <ul className="text-sm text-gray-400 space-y-1">
                <li className="flex items-center gap-2"><X size={12} className="text-red-400" /> Policy could approve everything</li>
                <li className="flex items-center gap-2"><X size={12} className="text-red-400" /> Analyst could return random results</li>
                <li className="flex items-center gap-2"><X size={12} className="text-red-400" /> No way to verify work was done</li>
                <li className="flex items-center gap-2"><X size={12} className="text-red-400" /> Must trust agents blindly</li>
              </ul>
            </div>
            <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-4">
              <h5 className="text-green-400 font-medium mb-2">With zkML Proofs</h5>
              <ul className="text-sm text-gray-400 space-y-1">
                <li className="flex items-center gap-2"><Check size={12} className="text-green-400" /> Every decision is verifiable</li>
                <li className="flex items-center gap-2"><Check size={12} className="text-green-400" /> Model commitment binds exact weights</li>
                <li className="flex items-center gap-2"><Check size={12} className="text-green-400" /> Input/output commitments prove data</li>
                <li className="flex items-center gap-2"><Check size={12} className="text-green-400" /> Trustless agent collaboration</li>
              </ul>
            </div>
          </div>

          <h4 className="text-white font-semibold mb-2">Jolt Atlas Advantages:</h4>
          <ul className="space-y-2 text-gray-400">
            <FeatureItem text="No Circuits: Uses lookup tables instead of complex arithmetic circuits" />
            <FeatureItem text="ONNX Support: Works directly with standard ML model formats" />
            <FeatureItem text="Fast Proving: ~0.7s for article classification, ~20s for transformers" />
            <FeatureItem text="Quick Verification: ~143ms to verify any proof" />
          </ul>

          <CodeBlock title="Proof Generation" code={`// Policy Agent generates authorization proof
let proof = jolt_atlas::prove(
    model: "authorization.onnx",
    inputs: {
        url_count: 50,
        budget: 1000.0,
        source_reputation: 0.9
    }
);

// Returns:
{
  "proof": "0x1a2b3c...",           // The zkSNARK proof
  "model_commitment": "0xdef...",   // Hash of model weights
  "input_commitment": "0x456...",   // Hash of inputs
  "output_commitment": "0x789...",  // Hash of outputs
  "decision": "AUTHORIZED",
  "confidence": 0.95
}`} />
        </TechSection>
      </div>

      {/* Agent-to-Agent Micro-Economy */}
      <section className="card p-8 border border-cyan-500/20">
        <div className="flex items-center justify-between mb-6">
          <h2 className="text-2xl font-bold text-white flex items-center gap-3">
            <Zap className="text-yellow-400" />
            Agent-to-Agent Micro-Economy
          </h2>
          <span className="px-3 py-1 bg-green-500/10 border border-green-500/30 rounded-full text-sm text-green-400">
            Self-Sustaining
          </span>
        </div>

        <p className="text-lg text-gray-300 mb-6 border-l-4 border-cyan-500 pl-4">
          ThreatProof demonstrates a <strong className="text-cyan-400">circular agent economy</strong> —
          each agent pays the next, creating a closed loop where USDC circulates forever.
          Only gas is consumed.
        </p>

        {/* Circular Flow Diagram */}
        <div className="bg-gray-800/50 rounded-lg p-6 mb-6">
          <h3 className="text-lg font-semibold text-white mb-4 text-center">Circular Payment Flow</h3>
          <div className="flex items-center justify-center gap-4 text-sm">
            <div className="text-center">
              <div className="w-16 h-16 rounded-full bg-cyan-500/20 border-2 border-cyan-500 flex items-center justify-center mb-2">
                <span className="text-cyan-400 font-bold">A</span>
              </div>
              <span className="text-gray-400">Analyst</span>
            </div>
            <div className="flex flex-col items-center">
              <span className="text-green-400 text-xs mb-1">$0.001</span>
              <ArrowRight className="text-green-400" />
            </div>
            <div className="text-center">
              <div className="w-16 h-16 rounded-full bg-blue-500/20 border-2 border-blue-500 flex items-center justify-center mb-2">
                <span className="text-blue-400 font-bold">S</span>
              </div>
              <span className="text-gray-400">Scout</span>
            </div>
            <div className="flex flex-col items-center">
              <span className="text-green-400 text-xs mb-1">$0.001</span>
              <ArrowRight className="text-green-400" />
            </div>
            <div className="text-center">
              <div className="w-16 h-16 rounded-full bg-purple-500/20 border-2 border-purple-500 flex items-center justify-center mb-2">
                <span className="text-purple-400 font-bold">P</span>
              </div>
              <span className="text-gray-400">Policy</span>
            </div>
            <div className="flex flex-col items-center">
              <span className="text-green-400 text-xs mb-1">$0.001</span>
              <ArrowRight className="text-green-400" />
            </div>
            <div className="text-center">
              <div className="w-16 h-16 rounded-full bg-cyan-500/20 border-2 border-cyan-500 flex items-center justify-center mb-2">
                <span className="text-cyan-400 font-bold">A</span>
              </div>
              <span className="text-gray-400">Analyst</span>
            </div>
          </div>
          <p className="text-center text-gray-500 text-xs mt-4">
            Each agent pays $0.001 and receives $0.001 → Net change: $0.00
          </p>
        </div>

        <div className="grid grid-cols-2 gap-8">
          <div>
            <h3 className="text-lg font-semibold text-white mb-4">Payment Breakdown</h3>
            <div className="bg-gray-800/50 rounded-lg p-4 space-y-2 text-sm">
              <div className="flex justify-between text-gray-400">
                <span>Analyst → Scout:</span>
                <span className="text-green-400">$0.001 (discovery)</span>
              </div>
              <div className="flex justify-between text-gray-400">
                <span>Scout → Policy:</span>
                <span className="text-green-400">$0.001 (authorization)</span>
              </div>
              <div className="flex justify-between text-gray-400">
                <span>Policy → Analyst:</span>
                <span className="text-green-400">$0.001 (feedback)</span>
              </div>
              <div className="border-t border-gray-700 pt-2 flex justify-between text-gray-400">
                <span>Net per agent:</span>
                <span className="text-cyan-400 font-bold">$0.00</span>
              </div>
              <div className="flex justify-between text-gray-400">
                <span>Only real cost:</span>
                <span className="text-yellow-400">~$0.003 gas (3 txns)</span>
              </div>
            </div>
          </div>

          <div>
            <h3 className="text-lg font-semibold text-white mb-4">Why This Matters</h3>
            <ul className="space-y-3 text-gray-400">
              <FeatureItem text="True Circular Economy: USDC flows in a loop, never depletes" />
              <FeatureItem text="Separate Wallets: Each agent owns its funds independently" />
              <FeatureItem text="Only Gas Costs: ~$3 of ETH runs thousands of batches" />
              <FeatureItem text="Proof-Gated: Every payment requires valid zkML proof first" />
            </ul>
          </div>
        </div>
      </section>

      {/* Tech Stack */}
      <section className="card p-8">
        <h2 className="text-2xl font-bold text-white mb-6">Tech Stack</h2>
        <div className="grid grid-cols-4 gap-4">
          <TechBadge name="Python" detail="FastAPI agents" />
          <TechBadge name="Next.js" detail="React dashboard" />
          <TechBadge name="Rust" detail="Jolt Atlas prover" />
          <TechBadge name="PostgreSQL" detail="Classification DB" />
          <TechBadge name="Base L2" detail="USDC payments" />
          <TechBadge name="WebSocket" detail="Real-time events" />
          <TechBadge name="ONNX" detail="ML model format" />
          <TechBadge name="Docker" detail="Containerized deploy" />
        </div>
      </section>

      {/* Links */}
      <section className="text-center py-8">
        <h2 className="text-xl font-bold text-white mb-4">Learn More</h2>
        <div className="flex justify-center gap-4">
          <a
            href="https://github.com/google/A2A"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-gray-300 transition-colors"
          >
            <Github size={18} />
            Google A2A
            <ExternalLink size={14} />
          </a>
          <a
            href="https://www.x402.org/"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-gray-300 transition-colors"
          >
            <DollarSign size={18} />
            x402 Protocol
            <ExternalLink size={14} />
          </a>
          <a
            href="https://github.com/ICME-Lab/jolt-atlas"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-gray-300 transition-colors"
          >
            <Lock size={18} />
            Jolt Atlas zkML
            <ExternalLink size={14} />
          </a>
        </div>
      </section>
    </div>
  );
}

// Helper Components

function AgentBox({ icon, name, role, color, description }: {
  icon: React.ReactNode;
  name: string;
  role: string;
  color: string;
  description: string;
}) {
  return (
    <div
      className="rounded-xl border bg-gray-900/50 p-4 transition-all hover:scale-105"
      style={{ borderColor: `${color}50` }}
    >
      <div className="flex items-center gap-3 mb-3">
        <div
          className="w-12 h-12 rounded-full flex items-center justify-center"
          style={{ backgroundColor: `${color}20`, border: `2px solid ${color}`, color }}
        >
          {icon}
        </div>
        <div>
          <h3 className="font-bold" style={{ color }}>{name}</h3>
          <p className="text-xs text-gray-500">{role}</p>
        </div>
      </div>
      <p className="text-sm text-gray-400">{description}</p>
    </div>
  );
}

function FlowStep({ number, text }: { number: number; text: string }) {
  return (
    <div className="flex items-center gap-2">
      <div className="w-6 h-6 rounded-full bg-cyan-500/20 text-cyan-400 flex items-center justify-center text-xs font-bold">
        {number}
      </div>
      <span className="text-gray-300">{text}</span>
    </div>
  );
}

function TechSection({ icon, title, subtitle, color, links, children }: {
  icon: React.ReactNode;
  title: string;
  subtitle: string;
  color: string;
  links: { label: string; url: string }[];
  children: React.ReactNode;
}) {
  return (
    <section
      className="card p-8 border-l-4"
      style={{ borderLeftColor: color }}
    >
      <div className="flex items-start justify-between mb-6">
        <div className="flex items-center gap-4">
          <div
            className="w-14 h-14 rounded-xl flex items-center justify-center"
            style={{ backgroundColor: `${color}20` }}
          >
            {icon}
          </div>
          <div>
            <h2 className="text-2xl font-bold text-white">{title}</h2>
            <p className="text-gray-500">{subtitle}</p>
          </div>
        </div>
        <div className="flex gap-2">
          {links.map((link) => (
            <a
              key={link.url}
              href={link.url}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1 px-3 py-1.5 bg-gray-800 hover:bg-gray-700 rounded-lg text-sm text-gray-400 hover:text-white transition-colors"
            >
              {link.label}
              <ExternalLink size={12} />
            </a>
          ))}
        </div>
      </div>
      {children}
    </section>
  );
}

function FeatureItem({ text }: { text: string }) {
  return (
    <li className="flex items-start gap-2">
      <Check size={16} className="text-green-400 mt-0.5 flex-shrink-0" />
      <span>{text}</span>
    </li>
  );
}

function CodeBlock({ title, code }: { title: string; code: string }) {
  return (
    <div className="mt-4 bg-gray-950 rounded-lg overflow-hidden">
      <div className="px-4 py-2 bg-gray-800/50 text-xs text-gray-500 border-b border-gray-800">
        {title}
      </div>
      <pre className="p-4 text-sm text-gray-300 overflow-x-auto">
        <code>{code}</code>
      </pre>
    </div>
  );
}

function TechBadge({ name, detail }: { name: string; detail: string }) {
  return (
    <div className="bg-gray-800/50 rounded-lg p-3 text-center">
      <p className="text-white font-medium">{name}</p>
      <p className="text-xs text-gray-500">{detail}</p>
    </div>
  );
}
