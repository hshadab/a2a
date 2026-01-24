'use client';

import { ExternalLink, Shield, Zap, DollarSign, Lock, Check, ArrowRight, Github } from 'lucide-react';

export default function AboutPage() {
  return (
    <div className="p-6 space-y-12">
      {/* Hero Section */}
      <section className="text-center py-12">
        <h1 className="text-4xl font-bold text-white mb-4">
          How <span className="bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">ThreatProof</span> Works
        </h1>
        <p className="text-xl text-gray-400 max-w-3xl mx-auto">
          An autonomous network of AI agents that discover, classify, and verify phishing threats
          using cryptographic proofs — powered by three groundbreaking protocols.
        </p>
      </section>

      {/* Architecture Overview */}
      <section className="card p-8">
        <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-3">
          <Shield className="text-cyan-400" />
          Architecture Overview
        </h2>

        <div className="grid grid-cols-3 gap-6 mb-8">
          <AgentBox
            emoji="🔭"
            name="Scout Agent"
            role="Explorer"
            color="#3b82f6"
            description="Continuously monitors threat feeds (PhishTank, OpenPhish, Certificate Transparency logs) to discover suspicious URLs."
          />
          <AgentBox
            emoji="⚖️"
            name="Policy Agent"
            role="Gatekeeper"
            color="#a855f7"
            description="Evaluates whether to authorize spending on classification. Generates zkML proofs of its decision logic."
          />
          <AgentBox
            emoji="🔬"
            name="Analyst Agent"
            role="Detective"
            color="#22d3ee"
            description="Classifies URLs as phishing/safe using ML. Generates zkML proofs that the classification was computed correctly."
          />
        </div>

        {/* Flow Diagram */}
        <div className="bg-gray-800/50 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-4">Data Flow</h3>
          <div className="flex items-center justify-between text-sm">
            <FlowStep number={1} text="Scout finds URLs" />
            <ArrowRight className="text-gray-600" />
            <FlowStep number={2} text="Policy authorizes + proof" />
            <ArrowRight className="text-gray-600" />
            <FlowStep number={3} text="Scout pays Policy (USDC)" />
            <ArrowRight className="text-gray-600" />
            <FlowStep number={4} text="Analyst classifies + proof" />
            <ArrowRight className="text-gray-600" />
            <FlowStep number={5} text="Scout pays Analyst (USDC)" />
            <ArrowRight className="text-gray-600" />
            <FlowStep number={6} text="Results stored with proofs" />
          </div>
        </div>
      </section>

      {/* Technology Deep Dives */}
      <div className="grid grid-cols-1 gap-8">

        {/* Google A2A Protocol */}
        <TechSection
          icon={<div className="text-3xl">🤖</div>}
          title="Google A2A Protocol"
          subtitle="Agent-to-Agent Communication"
          color="#4285f4"
          links={[
            { label: 'A2A Specification', url: 'https://github.com/google/A2A' },
            { label: 'Google Blog Post', url: 'https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/' },
          ]}
        >
          <p className="text-gray-300 mb-4">
            A2A (Agent-to-Agent) is Google's open protocol for AI agents to discover, communicate,
            and collaborate with each other. It provides a standardized way for autonomous agents
            to negotiate tasks and exchange capabilities.
          </p>

          <h4 className="text-white font-semibold mb-2">How ThreatProof Uses A2A:</h4>
          <ul className="space-y-2 text-gray-400">
            <FeatureItem text="Agent Cards: Each agent publishes a /.well-known/agent.json describing its capabilities and pricing" />
            <FeatureItem text="Skill Discovery: Scout discovers Policy and Analyst agents' skills dynamically" />
            <FeatureItem text="Task Negotiation: Agents negotiate batch sizes, costs, and proof requirements" />
            <FeatureItem text="Interoperability: Any A2A-compatible agent could join the network" />
          </ul>

          <CodeBlock title="Agent Card Example" code={`// GET /.well-known/agent.json
{
  "name": "Policy Agent",
  "description": "Authorizes threat intel spending",
  "skills": [{
    "id": "authorize-batch",
    "price": { "amount": "0.001", "currency": "USDC" },
    "proof_required": true
  }]
}`} />
        </TechSection>

        {/* x402 Protocol */}
        <TechSection
          icon={<DollarSign className="text-green-400" size={32} />}
          title="x402 Payment Protocol"
          subtitle="HTTP 402 Payment Required"
          color="#22c55e"
          links={[
            { label: 'x402 Specification', url: 'https://www.x402.org/' },
            { label: 'Coinbase Developer Docs', url: 'https://docs.cdp.coinbase.com/x402/docs/welcome' },
          ]}
        >
          <p className="text-gray-300 mb-4">
            x402 brings the HTTP 402 "Payment Required" status code to life. It enables
            machine-to-machine payments where APIs can request payment before providing services —
            perfect for autonomous agent economies.
          </p>

          <h4 className="text-white font-semibold mb-2">How ThreatProof Uses x402:</h4>
          <ul className="space-y-2 text-gray-400">
            <FeatureItem text="Pay-per-Request: Scout pays 0.001 USDC per authorization, 0.0005 USDC per URL classification" />
            <FeatureItem text="USDC on Base: Fast, cheap transactions (~$0.001 gas) on Coinbase's L2" />
            <FeatureItem text="Self-Sustaining: All agents share the same treasury, so USDC circulates internally" />
            <FeatureItem text="Transparent Pricing: Costs declared upfront in agent cards" />
          </ul>

          <CodeBlock title="x402 Flow" code={`// 1. Scout requests classification
POST /classify { urls: [...] }

// 2. Analyst responds with 402
HTTP 402 Payment Required
X-Payment-Address: 0x6c67...
X-Payment-Amount: 0.025 USDC

// 3. Scout makes payment on Base
await usdc.transfer(analyst, 0.025)

// 4. Scout retries with receipt
POST /classify
X-Payment-Receipt: 0xabc123...

// 5. Analyst provides service
HTTP 200 { classifications: [...], proof: "..." }`} />
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
                <li>❌ Policy could approve everything</li>
                <li>❌ Analyst could return random results</li>
                <li>❌ No way to verify work was done</li>
                <li>❌ Must trust agents blindly</li>
              </ul>
            </div>
            <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-4">
              <h5 className="text-green-400 font-medium mb-2">With zkML Proofs</h5>
              <ul className="text-sm text-gray-400 space-y-1">
                <li>✓ Every decision is verifiable</li>
                <li>✓ Model commitment binds exact weights</li>
                <li>✓ Input/output commitments prove data</li>
                <li>✓ Trustless agent collaboration</li>
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

      {/* Self-Sustaining Economy */}
      <section className="card p-8">
        <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-3">
          <Zap className="text-yellow-400" />
          Self-Sustaining Agent Economy
        </h2>

        <div className="grid grid-cols-2 gap-8">
          <div>
            <h3 className="text-lg font-semibold text-white mb-4">How It Works</h3>
            <p className="text-gray-400 mb-4">
              All three agents share the same treasury wallet. When Scout "pays" Policy or Analyst,
              the USDC moves from the treasury... back to the treasury. The money circulates internally.
            </p>
            <div className="bg-gray-800/50 rounded-lg p-4 space-y-2 text-sm">
              <div className="flex justify-between text-gray-400">
                <span>Scout pays Policy:</span>
                <span className="text-green-400">0.001 USDC → Treasury</span>
              </div>
              <div className="flex justify-between text-gray-400">
                <span>Scout pays Analyst:</span>
                <span className="text-green-400">0.025 USDC → Treasury</span>
              </div>
              <div className="flex justify-between text-gray-400">
                <span>Net USDC change:</span>
                <span className="text-cyan-400 font-bold">$0.00</span>
              </div>
              <div className="border-t border-gray-700 pt-2 flex justify-between text-gray-400">
                <span>Only real cost:</span>
                <span className="text-yellow-400">~$0.001 gas per batch</span>
              </div>
            </div>
          </div>

          <div>
            <h3 className="text-lg font-semibold text-white mb-4">Why This Matters</h3>
            <ul className="space-y-3 text-gray-400">
              <FeatureItem text="Runs Forever: 1 USDC treasury can fund unlimited batches" />
              <FeatureItem text="Only Gas Costs: ~$3 of ETH funds thousands of batches on Base" />
              <FeatureItem text="Real Payments: The x402 protocol is fully functional, not simulated" />
              <FeatureItem text="Extensible: External customers could pay for API access to fund operations" />
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

function AgentBox({ emoji, name, role, color, description }: {
  emoji: string;
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
          className="w-12 h-12 rounded-full flex items-center justify-center text-2xl"
          style={{ backgroundColor: `${color}20`, border: `2px solid ${color}` }}
        >
          {emoji}
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
