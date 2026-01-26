'use client';

import { ExternalLink, Shield, Zap, DollarSign, Lock, Check, ArrowRight, Github, Search, Scale, Microscope, Bot, X } from 'lucide-react';

export default function AboutPage() {
  return (
    <div className="p-4 md:p-6 space-y-8 md:space-y-12">
      {/* Hero Section */}
      <section className="text-center py-8 md:py-12">
        <h1 className="text-2xl md:text-4xl font-bold text-white mb-4">
          How <span className="bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">ThreatProof</span> Works
        </h1>
        <p className="text-base md:text-xl text-gray-400 max-w-3xl mx-auto mb-6 px-4">
          An autonomous network of AI agents that discover, classify, and verify phishing threats
          using cryptographic proofs — powered by three groundbreaking protocols.
        </p>
        <div className="inline-flex flex-col md:flex-row items-center gap-2 md:gap-3 px-4 md:px-6 py-3 bg-gradient-to-r from-cyan-500/10 to-purple-500/10 border border-cyan-500/30 rounded-2xl md:rounded-full">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-green-400 animate-pulse" />
            <span className="text-cyan-400 font-medium text-sm md:text-base">Autonomous Agent-to-Agent Commerce</span>
          </div>
          <span className="hidden md:inline text-gray-600">|</span>
          <span className="text-gray-400 text-xs md:text-sm">A self-sustaining AI micro-economy</span>
        </div>
      </section>

      {/* Autonomous Agent-to-Agent Economy */}
      <section className="card p-4 md:p-8 border border-cyan-500/20">
        <div className="flex items-center justify-between mb-4 md:mb-6">
          <h2 className="text-xl md:text-2xl font-bold text-white flex items-center gap-3">
            <Zap className="text-yellow-400" size={24} />
            Autonomous Agent-to-Agent Economy
          </h2>
          <span className="hidden md:inline px-3 py-1 bg-green-500/10 border border-green-500/30 rounded-full text-sm text-green-400">
            Live on Base
          </span>
        </div>

        {/* The Story */}
        <p className="text-base md:text-lg text-gray-300 mb-6 border-l-4 border-cyan-500 pl-4">
          Scout discovers threat URLs → Analyst classifies them → Both get paid.
          <span className="text-cyan-400 font-medium"> No humans in the loop. No trust required.</span>
        </p>

        {/* The Agents */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 md:gap-6 mb-6 md:mb-8">
          <AgentBox
            icon={<Search size={24} />}
            name="Threat Scout"
            role="Discovery + Quality Proof"
            color="#3b82f6"
            description="Discovers suspicious URLs from PhishTank, OpenPhish, and CT logs. Generates zkML proof that quality scoring ran correctly."
          />
          <AgentBox
            icon={<Microscope size={24} />}
            name="Threat Analyst"
            role="Classification + Work Proof"
            color="#22d3ee"
            description="Classifies URLs as phishing/safe/suspicious using ML. Generates zkML proof that classification model executed correctly."
          />
        </div>

        {/* The Payment Flow - Visual */}
        <div className="bg-gray-800/50 rounded-lg p-4 md:p-6 mb-6">
          <h3 className="text-base md:text-lg font-semibold text-white mb-4 text-center">The Payment Loop</h3>

          {/* Desktop: Circular visual */}
          <div className="hidden md:flex items-center justify-center gap-6 text-sm">
            <div className="text-center">
              <div className="w-20 h-20 rounded-full bg-blue-500/20 border-2 border-blue-500 flex items-center justify-center mb-2">
                <Search className="text-blue-400" size={28} />
              </div>
              <span className="text-white font-medium">Scout</span>
              <span className="block text-xs text-gray-500">discovers URL</span>
            </div>

            <div className="flex flex-col items-center">
              <div className="text-green-400 text-xs font-mono mb-1">$0.001</div>
              <ArrowRight className="text-green-400" size={28} />
              <div className="text-gray-500 text-[10px] mt-1">pays for discovery</div>
            </div>

            <div className="text-center">
              <div className="w-20 h-20 rounded-full bg-cyan-500/20 border-2 border-cyan-500 flex items-center justify-center mb-2">
                <Microscope className="text-cyan-400" size={28} />
              </div>
              <span className="text-white font-medium">Analyst</span>
              <span className="block text-xs text-gray-500">classifies URL</span>
            </div>

            <div className="flex flex-col items-center">
              <div className="text-green-400 text-xs font-mono mb-1">$0.001</div>
              <ArrowRight className="text-green-400 rotate-180" size={28} />
              <div className="text-gray-500 text-[10px] mt-1">pays for feedback</div>
            </div>

            <div className="text-center">
              <div className="w-20 h-20 rounded-full bg-blue-500/20 border-2 border-blue-500 flex items-center justify-center mb-2">
                <Search className="text-blue-400" size={28} />
              </div>
              <span className="text-white font-medium">Scout</span>
              <span className="block text-xs text-gray-500">receives payment</span>
            </div>
          </div>

          {/* Mobile: Simplified vertical */}
          <div className="md:hidden space-y-3">
            <div className="flex items-center gap-3 p-3 bg-blue-500/10 rounded-lg">
              <Search className="text-blue-400" size={20} />
              <div>
                <span className="text-white text-sm font-medium">Scout discovers URL</span>
                <span className="block text-xs text-gray-500">generates quality proof</span>
              </div>
            </div>
            <div className="flex justify-center">
              <div className="flex items-center gap-2 text-green-400 text-xs">
                <span>↓</span> <span className="font-mono">$0.001</span> <span>↓</span>
              </div>
            </div>
            <div className="flex items-center gap-3 p-3 bg-cyan-500/10 rounded-lg">
              <Microscope className="text-cyan-400" size={20} />
              <div>
                <span className="text-white text-sm font-medium">Analyst classifies URL</span>
                <span className="block text-xs text-gray-500">generates work proof</span>
              </div>
            </div>
            <div className="flex justify-center">
              <div className="flex items-center gap-2 text-green-400 text-xs">
                <span>↓</span> <span className="font-mono">$0.001</span> <span>↓</span>
              </div>
            </div>
            <div className="text-center text-xs text-gray-500">
              Loop repeats. Net change: $0.00
            </div>
          </div>

          <p className="text-center text-gray-400 text-xs mt-4 font-medium">
            Net per agent: <span className="text-cyan-400">$0.00</span> · Only cost: <span className="text-yellow-400">~$0.002 gas</span>
          </p>
        </div>

        {/* The Trust - Why it's trustless */}
        <div className="bg-gray-900/50 rounded-lg p-4 md:p-6">
          <h3 className="text-base md:text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Lock className="text-purple-400" size={18} />
            No Trust Required
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
            <div className="space-y-3">
              <div className="flex items-start gap-2">
                <Check className="text-green-400 mt-0.5 flex-shrink-0" size={16} />
                <div>
                  <span className="text-white font-medium">Work Proofs</span>
                  <p className="text-gray-500 text-xs">Each agent proves their ML model ran correctly on the input</p>
                </div>
              </div>
              <div className="flex items-start gap-2">
                <Check className="text-green-400 mt-0.5 flex-shrink-0" size={16} />
                <div>
                  <span className="text-white font-medium">Buyer Verifies Seller</span>
                  <p className="text-gray-500 text-xs">Payment only released after proof verification succeeds</p>
                </div>
              </div>
            </div>
            <div className="space-y-3">
              <div className="flex items-start gap-2">
                <Check className="text-green-400 mt-0.5 flex-shrink-0" size={16} />
                <div>
                  <span className="text-white font-medium">Spending Guardrails</span>
                  <p className="text-gray-500 text-xs">zkML proof of policy compliance required before any payment</p>
                </div>
              </div>
              <div className="flex items-start gap-2">
                <Check className="text-green-400 mt-0.5 flex-shrink-0" size={16} />
                <div>
                  <span className="text-white font-medium">Model Commitments</span>
                  <p className="text-gray-500 text-xs">Proof binds to exact ONNX model weights - no bait and switch</p>
                </div>
              </div>
            </div>
          </div>
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

      {/* Tech Stack */}
      <section className="card p-4 md:p-8">
        <h2 className="text-xl md:text-2xl font-bold text-white mb-4 md:mb-6">Tech Stack</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 md:gap-4">
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
      <section className="text-center py-6 md:py-8">
        <h2 className="text-lg md:text-xl font-bold text-white mb-4">Learn More</h2>
        <div className="flex flex-col md:flex-row justify-center gap-3 md:gap-4 px-4">
          <a
            href="https://github.com/google/A2A"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center justify-center gap-2 px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-gray-300 transition-colors text-sm"
          >
            <Github size={16} />
            Google A2A
            <ExternalLink size={12} />
          </a>
          <a
            href="https://www.x402.org/"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center justify-center gap-2 px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-gray-300 transition-colors text-sm"
          >
            <DollarSign size={16} />
            x402 Protocol
            <ExternalLink size={12} />
          </a>
          <a
            href="https://github.com/ICME-Lab/jolt-atlas"
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center justify-center gap-2 px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-gray-300 transition-colors text-sm"
          >
            <Lock size={16} />
            Jolt Atlas zkML
            <ExternalLink size={12} />
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
