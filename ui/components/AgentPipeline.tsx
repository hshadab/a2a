'use client';

import { useEffect, useState, useMemo } from 'react';
import { Search, Microscope, Wallet, ChevronDown, ChevronUp, Copy, Check, ExternalLink } from 'lucide-react';

type AgentState = 'idle' | 'active' | 'working' | 'proving';

// Agent wallet addresses (2-Agent Model: Scout + Analyst)
const WALLET_ADDRESSES = {
  analyst: '0x7ee88871fA9be48b62552F231a4976A11e559db8',
  scout: '0x269CBA662fE55c4fe1212c609090A31844C36ab8',
};

// Agent A2A endpoints
const AGENT_URLS = {
  analyst: 'https://threat-intel-analyst.onrender.com',
  scout: 'https://threat-intel-scout.onrender.com',
};

interface AgentEvent {
  type: string;
  timestamp: string;
  data: Record<string, any>;
}

interface AgentPipelineProps {
  events: AgentEvent[];
  lastEvent?: AgentEvent | null;
  stats?: {
    total_urls: number;
    phishing_count: number;
    safe_count: number;
    suspicious_count: number;
    total_proofs: number;
    policy_paid_usdc: number;
    analyst_paid_usdc: number;
  } | null;
}

interface PaymentArrow {
  id: number;
  direction: 'left' | 'right';
  amount: number;
}

export default function AgentPipeline({ events, lastEvent, stats }: AgentPipelineProps) {
  const [walletBalances, setWalletBalances] = useState<Record<string, { usdc: number; eth: number }>>({});
  const [paymentArrows, setPaymentArrows] = useState<PaymentArrow[]>([]);
  const [proofStats, setProofStats] = useState({
    scout: { generated: 0, verified: 0, avgGenMs: 0, avgVerMs: 0 },
    analyst: { generated: 0, verified: 0, avgGenMs: 0, avgVerMs: 0 },
  });
  const [lastProof, setLastProof] = useState<{
    scout: { type: string; output: string; confidence: number; genMs: number; verMs: number; commitments: { input: string; output: string; proof: string } } | null;
    analyst: { type: string; output: string; confidence: number; genMs: number; verMs: number; commitments: { input: string; output: string; proof: string } } | null;
  }>({ scout: null, analyst: null });

  // Fetch wallet balances
  useEffect(() => {
    const fetchBalances = async () => {
      const balances: Record<string, { usdc: number; eth: number }> = {};
      const usdcContract = '0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913';
      const balanceOfSelector = '0x70a08231';

      for (const [agent, address] of Object.entries(WALLET_ADDRESSES)) {
        try {
          const ethResp = await fetch('https://mainnet.base.org', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              jsonrpc: '2.0',
              method: 'eth_getBalance',
              params: [address, 'latest'],
              id: 1,
            }),
          });
          const ethData = await ethResp.json();
          const ethBalance = parseInt(ethData.result, 16) / 1e18;

          const paddedAddress = address.slice(2).toLowerCase().padStart(64, '0');
          const usdcResp = await fetch('https://mainnet.base.org', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              jsonrpc: '2.0',
              method: 'eth_call',
              params: [{ to: usdcContract, data: balanceOfSelector + paddedAddress }, 'latest'],
              id: 2,
            }),
          });
          const usdcData = await usdcResp.json();
          const usdcBalance = parseInt(usdcData.result, 16) / 1e6;

          balances[agent] = { usdc: usdcBalance, eth: ethBalance };
        } catch {
          balances[agent] = { usdc: 0, eth: 0 };
        }
      }
      setWalletBalances(balances);
    };

    fetchBalances();
    const interval = setInterval(fetchBalances, 30000);
    return () => clearInterval(interval);
  }, []);

  // Handle payment events - trigger arrow animations
  useEffect(() => {
    if (!lastEvent) return;

    if (lastEvent.type === 'PAYMENT_SENDING') {
      const direction = lastEvent.data.recipient?.includes('scout') ? 'right' : 'left';
      const id = Date.now();
      setPaymentArrows(prev => [...prev, { id, direction, amount: lastEvent.data.amount_usdc }]);

      // Remove arrow after animation completes
      setTimeout(() => {
        setPaymentArrows(prev => prev.filter(a => a.id !== id));
      }, 1500);
    }

    // Track proof stats
    if (lastEvent.type === 'SCOUT_AUTHORIZED') {
      setProofStats(prev => ({
        ...prev,
        scout: {
          ...prev.scout,
          generated: prev.scout.generated + 1,
          avgGenMs: Math.round((prev.scout.avgGenMs * prev.scout.generated + (lastEvent.data.prove_time_ms || 0)) / (prev.scout.generated + 1)),
        }
      }));
      setLastProof(prev => ({
        ...prev,
        scout: {
          type: 'Spending',
          output: lastEvent.data.decision || 'AUTHORIZED',
          confidence: lastEvent.data.confidence || 0,
          genMs: lastEvent.data.prove_time_ms || 0,
          verMs: 0,
          commitments: { input: '', output: '', proof: lastEvent.data.proof_hash || '' }
        }
      }));
    }

    if (lastEvent.type === 'ANALYST_RESPONSE') {
      setProofStats(prev => ({
        ...prev,
        analyst: {
          ...prev.analyst,
          generated: prev.analyst.generated + 1,
          avgGenMs: Math.round((prev.analyst.avgGenMs * prev.analyst.generated + (lastEvent.data.prove_time_ms || 0)) / (prev.analyst.generated + 1)),
        }
      }));
      setLastProof(prev => ({
        ...prev,
        analyst: {
          type: 'Classification',
          output: lastEvent.data.classification || 'PHISHING',
          confidence: lastEvent.data.confidence || 0,
          genMs: lastEvent.data.prove_time_ms || 0,
          verMs: 0,
          commitments: { input: '', output: '', proof: lastEvent.data.proof_hash || '' }
        }
      }));
    }

    if (lastEvent.type === 'SPENDING_PROOF_VERIFIED') {
      const agent = lastEvent.data.agent as 'scout' | 'analyst';
      setProofStats(prev => ({
        ...prev,
        [agent]: {
          ...prev[agent],
          verified: prev[agent].verified + 1,
          avgVerMs: Math.round((prev[agent].avgVerMs * prev[agent].verified + (lastEvent.data.verify_time_ms || 0)) / (prev[agent].verified + 1)),
        }
      }));
    }

    if (lastEvent.type === 'WORK_VERIFIED') {
      setProofStats(prev => ({
        ...prev,
        analyst: {
          ...prev.analyst,
          verified: prev.analyst.verified + 1,
          avgVerMs: Math.round((prev.analyst.avgVerMs * prev.analyst.verified + (lastEvent.data.verify_time_ms || 0)) / (prev.analyst.verified + 1)),
        }
      }));
    }
  }, [lastEvent]);

  // Derive agent states
  const agentStates = useMemo(() => {
    const states = { scout: 'idle' as AgentState, analyst: 'idle' as AgentState };
    if (!lastEvent) return states;

    switch (lastEvent.type) {
      case 'SCOUT_FOUND_URLS':
        states.scout = 'working';
        break;
      case 'SCOUT_AUTHORIZING':
        states.scout = 'proving';
        break;
      case 'SCOUT_AUTHORIZED':
        states.scout = 'active';
        break;
      case 'ANALYST_AUTHORIZING':
        states.analyst = 'proving';
        break;
      case 'ANALYST_AUTHORIZED':
        states.analyst = 'active';
        break;
      case 'ANALYST_PROCESSING':
        states.analyst = 'working';
        break;
      case 'ANALYST_PROVING':
        states.analyst = 'proving';
        break;
      case 'ANALYST_RESPONSE':
      case 'WORK_VERIFIED':
      case 'SPENDING_PROOF_VERIFIED':
        states.analyst = 'active';
        break;
      case 'DATABASE_UPDATED':
        states.scout = 'active';
        break;
    }
    return states;
  }, [lastEvent]);

  // Filter events by agent
  const getAgentEvents = (agent: string) => {
    const prefixes: Record<string, string[]> = {
      scout: ['SCOUT', 'DATABASE'],
      analyst: ['ANALYST', 'WORK_VERIFIED', 'SPENDING_PROOF_VERIFIED'],
    };
    return events
      .filter(e => prefixes[agent]?.some(p => e.type.startsWith(p)))
      .slice(0, 3);
  };

  const formatEventMessage = (event: AgentEvent): string => {
    switch (event.type) {
      case 'SCOUT_FOUND_URLS':
        return event.data.url_count === 1 ? 'Found URL' : `Found ${event.data.url_count} URLs`;
      case 'SCOUT_AUTHORIZING':
        return 'Generating spending proof...';
      case 'SCOUT_AUTHORIZED':
        return `${event.data.decision} (${((event.data.confidence || 0) * 100).toFixed(0)}%)`;
      case 'ANALYST_AUTHORIZING':
        return 'Generating spending proof...';
      case 'ANALYST_AUTHORIZED':
        return `${event.data.decision} (${((event.data.confidence || 0) * 100).toFixed(0)}%)`;
      case 'ANALYST_PROCESSING':
        return event.data.url_count === 1 ? 'Classifying URL...' : `Classifying ${event.data.url_count} URLs`;
      case 'ANALYST_PROVING':
        return 'Generating zkML proof...';
      case 'ANALYST_RESPONSE':
        if (event.data.classification) {
          return `${event.data.classification} (${((event.data.confidence || 0) * 100).toFixed(0)}%)`;
        }
        return `${event.data.phishing_count || 0} phishing found`;
      case 'SPENDING_PROOF_VERIFIED':
        return event.data.valid !== false ? `${event.data.agent} proof verified` : `${event.data.agent} proof FAILED`;
      case 'WORK_VERIFIED':
        return event.data.valid !== false ? `${event.data.quality_tier || 'Work'} verified` : 'Work proof FAILED';
      case 'DATABASE_UPDATED':
        return `+${event.data.urls_added} URL${event.data.urls_added === 1 ? '' : 's'} saved`;
      default:
        return event.type;
    }
  };

  const hasActivePayment = paymentArrows.length > 0;

  return (
    <div className="w-full">
      {/* Pipeline Container - horizontal on desktop, vertical on mobile */}
      <div className="relative flex flex-col lg:flex-row items-center lg:items-start justify-center gap-4">

        {/* Analyst Agent Card */}
        <AgentCard
          name="Threat Analyst Agent"
          tagline="Classifies URLs with verifiable ML"
          whatItDoes={["Runs phishing classifier on each URL", "Returns: phishing / suspicious / safe"]}
          whatItProves={["Classification model ran correctly", "Scout verifies proof → pays $0.001 for analyst feedback"]}
          version="2.1.0"
          icon={<Microscope size={20} />}
          state={agentStates.analyst}
          agentUrl={AGENT_URLS.analyst}
          skillId="classify-urls"
          walletAddress={WALLET_ADDRESSES.analyst}
          walletBalance={walletBalances.analyst}
          stats={{
            urlsProcessed: stats?.phishing_count || 0,
            earned: stats?.analyst_paid_usdc || 0,
            spent: stats?.policy_paid_usdc || 0,
          }}
          models={[
            { name: 'Authorization', size: '4.2 KB', shape: '[7]→[2]', commitment: '44965f00586bff57fa42b9e58ddaf3b2159bc2fd' },
            { name: 'Classifier', size: '12.8 KB', shape: '[32]→[3]', commitment: '7b3e2a1f8c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f' },
          ]}
          proofStats={proofStats.analyst}
          lastProof={lastProof.analyst}
          events={getAgentEvents('analyst')}
          formatEvent={formatEventMessage}
          color="#a855f7"
        />

        {/* Connection Arrows - Desktop: horizontal, Mobile: vertical */}
        {/* Desktop Arrows - Payment Flow */}
        <div className="hidden lg:flex flex-col items-center justify-center h-full min-h-[400px] w-40 relative">
          {/* Payment Flow Label */}
          <div className="absolute -top-2 text-xs font-medium text-green-400 flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
            A2A Payments
          </div>

          {/* Top Arrow: Analyst → Scout (Discovery Payment) - Always glowing */}
          <div className="flex-1 flex items-center justify-center relative w-full">
            <svg width="120" height="50" viewBox="0 0 120 50" className="overflow-visible">
              {/* Glow filter */}
              <defs>
                <filter id="glow-right" x="-50%" y="-50%" width="200%" height="200%">
                  <feGaussianBlur stdDeviation="3" result="blur" />
                  <feMerge>
                    <feMergeNode in="blur" />
                    <feMergeNode in="SourceGraphic" />
                  </feMerge>
                </filter>
              </defs>
              {/* Background line */}
              <line x1="5" y1="25" x2="105" y2="25" stroke="#1f2937" strokeWidth="4" strokeLinecap="round" />
              {/* Always-on glowing arrow */}
              <g filter="url(#glow-right)" className="animate-pulse">
                <line x1="5" y1="25" x2="105" y2="25" stroke="#22c55e" strokeWidth="3" strokeLinecap="round" opacity="0.8" />
                <polygon points="115,25 100,18 100,32" fill="#22c55e" />
              </g>
              {/* Amount label */}
              <text x="60" y="15" textAnchor="middle" fill="#22c55e" fontSize="11" fontFamily="monospace" fontWeight="bold">$0.001</text>
              {/* Active payment overlay */}
              {paymentArrows.filter(a => a.direction === 'right').map(arrow => (
                <g key={arrow.id}>
                  <line x1="5" y1="25" x2="105" y2="25" stroke="#4ade80" strokeWidth="5" className="payment-arrow-animate" strokeLinecap="round" />
                  <polygon points="115,25 100,18 100,32" fill="#4ade80" className="payment-arrow-animate" />
                </g>
              ))}
            </svg>
            <span className="absolute -bottom-2 text-xs text-gray-400 font-medium whitespace-nowrap">Discovery →</span>
          </div>

          {/* Center divider */}
          <div className="w-full h-px bg-gray-800 my-2" />

          {/* Bottom Arrow: Scout → Analyst (Feedback Payment) */}
          <div className="flex-1 flex items-center justify-center relative w-full">
            <svg width="120" height="50" viewBox="0 0 120 50" className="overflow-visible">
              {/* Glow filter */}
              <defs>
                <filter id="glow-left" x="-50%" y="-50%" width="200%" height="200%">
                  <feGaussianBlur stdDeviation="2" result="blur" />
                  <feMerge>
                    <feMergeNode in="blur" />
                    <feMergeNode in="SourceGraphic" />
                  </feMerge>
                </filter>
              </defs>
              {/* Background line */}
              <line x1="115" y1="25" x2="15" y2="25" stroke="#1f2937" strokeWidth="4" strokeLinecap="round" />
              {/* Dimmer arrow (not always glowing) */}
              <g opacity="0.5">
                <line x1="115" y1="25" x2="15" y2="25" stroke="#06b6d4" strokeWidth="2" strokeLinecap="round" strokeDasharray="6 4" />
                <polygon points="5,25 20,18 20,32" fill="#06b6d4" />
              </g>
              {/* Amount label */}
              <text x="60" y="15" textAnchor="middle" fill="#06b6d4" fontSize="11" fontFamily="monospace" opacity="0.7">$0.001</text>
              {/* Active payment overlay */}
              {paymentArrows.filter(a => a.direction === 'left').map(arrow => (
                <g key={arrow.id} filter="url(#glow-left)">
                  <line x1="115" y1="25" x2="15" y2="25" stroke="#22d3ee" strokeWidth="5" className="payment-arrow-animate-reverse" strokeLinecap="round" />
                  <polygon points="5,25 20,18 20,32" fill="#22d3ee" className="payment-arrow-animate-reverse" />
                </g>
              ))}
            </svg>
            <span className="absolute -bottom-2 text-xs text-gray-400 font-medium whitespace-nowrap">← Feedback</span>
          </div>
        </div>

        {/* Mobile Arrows - Vertical between cards */}
        <div className="flex lg:hidden items-center justify-center gap-4 py-2">
          <div className="flex items-center gap-2 text-xs text-gray-500">
            <span className="text-purple-400">↓</span>
            <span>Discovery $0.001</span>
          </div>
          <div className="flex items-center gap-2 text-xs text-gray-500">
            <span className="text-cyan-400">↑</span>
            <span>Feedback $0.001</span>
          </div>
        </div>

        {/* Scout Agent Card */}
        <AgentCard
          name="URL Scout Agent"
          tagline="Discovers threat URLs from intel feeds"
          whatItDoes={["Fetches URLs from PhishTank & OpenPhish", "Scores source quality with ONNX model"]}
          whatItProves={["Quality score was computed correctly", "Analyst verifies proof → pays $0.001 for threat discovery"]}
          version="2.1.0"
          icon={<Search size={20} />}
          state={agentStates.scout}
          agentUrl={AGENT_URLS.scout}
          skillId="discover-url"
          walletAddress={WALLET_ADDRESSES.scout}
          walletBalance={walletBalances.scout}
          stats={{
            urlsProcessed: stats?.total_urls || 0,
            earned: stats?.policy_paid_usdc || 0,
            spent: stats?.analyst_paid_usdc || 0,
          }}
          models={[
            { name: 'Authorization', size: '4.2 KB', shape: '[7]→[2]', commitment: '44965f00586bff57fa42b9e58ddaf3b2159bc2fd' },
            { name: 'QualityScorer', size: '8.1 KB', shape: '[32]→[4]', commitment: '7b3e2a1f8c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f' },
          ]}
          proofStats={proofStats.scout}
          lastProof={lastProof.scout}
          events={getAgentEvents('scout')}
          formatEvent={formatEventMessage}
          color="#06b6d4"
        />
      </div>

      {/* Legend - hidden on mobile, shown on desktop */}
      <div className="hidden md:flex justify-center gap-6 mt-6 text-xs text-gray-500">
        <div className="flex items-center gap-2">
          <div className="w-6 h-0.5 bg-gray-600" style={{ backgroundImage: 'repeating-linear-gradient(90deg, #4b5563 0, #4b5563 4px, transparent 4px, transparent 8px)' }} />
          <span>Idle</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-6 h-0.5 bg-green-500" />
          <span>Payment Active</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-cyan-400">$0.001/URL</span>
          <span>Per-URL Circular Economy</span>
        </div>
      </div>
    </div>
  );
}

// Agent Card Component
function AgentCard({
  name,
  tagline,
  whatItDoes,
  whatItProves,
  version,
  icon,
  state,
  agentUrl,
  skillId,
  walletAddress,
  walletBalance,
  stats,
  models,
  proofStats,
  lastProof,
  events,
  formatEvent,
  color,
}: {
  name: string;
  tagline: string;
  whatItDoes: string[];
  whatItProves: string[];
  version: string;
  icon: React.ReactNode;
  state: AgentState;
  agentUrl: string;
  skillId: string;
  walletAddress: string;
  walletBalance?: { usdc: number; eth: number };
  stats: { urlsProcessed: number; earned: number; spent: number };
  models: { name: string; size: string; shape: string; commitment: string }[];
  proofStats: { generated: number; verified: number; avgGenMs: number; avgVerMs: number };
  lastProof: { type: string; output: string; confidence: number; genMs: number; verMs: number; commitments: { input: string; output: string; proof: string } } | null;
  events: AgentEvent[];
  formatEvent: (e: AgentEvent) => string;
  color: string;
}) {
  const [expandedSections, setExpandedSections] = useState<Record<string, boolean>>({
    models: false,
    proofs: false,
  });
  const [copiedHash, setCopiedHash] = useState<string | null>(null);

  const toggleSection = (section: string) => {
    setExpandedSections(prev => ({ ...prev, [section]: !prev[section] }));
  };

  const copyToClipboard = (text: string, id: string) => {
    navigator.clipboard.writeText(text);
    setCopiedHash(id);
    setTimeout(() => setCopiedHash(null), 2000);
  };

  const isActive = state !== 'idle';
  const isProving = state === 'proving';
  const isWorking = state === 'working';

  const statusText = isProving ? 'PROVING' : isWorking ? 'WORKING' : isActive ? 'ACTIVE' : 'READY';
  const statusColor = isProving ? '#eab308' : isWorking ? '#22c55e' : isActive ? '#3b82f6' : color;

  return (
    <div
      className="w-full lg:flex-1 lg:max-w-md rounded-xl border-2 transition-all duration-300"
      style={{
        borderColor: isActive ? color : `${color}60`,
        boxShadow: isProving ? `0 0 24px ${color}50, inset 0 0 20px ${color}10` : `0 0 0 1px ${color}20, 0 4px 20px rgba(0,0,0,0.3)`,
        background: `linear-gradient(135deg, ${color}08 0%, #0f0f12 50%, #0a0a0f 100%)`,
      }}
    >
      {/* Header */}
      <div className="p-3 md:p-4 border-b rounded-t-xl" style={{ borderColor: `${color}20`, backgroundColor: `${color}12` }}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2 md:gap-3">
            <div
              className="w-8 h-8 md:w-10 md:h-10 rounded-lg flex items-center justify-center border"
              style={{ borderColor: color, backgroundColor: `${color}15`, color }}
            >
              {icon}
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-semibold text-white">{name}</span>
                <span className="text-[10px] text-gray-500">v{version}</span>
              </div>
              <div className="text-xs text-gray-400">{tagline}</div>
            </div>
          </div>
          <div
            className="px-2 py-1 rounded text-xs font-medium flex items-center gap-1.5"
            style={{ backgroundColor: `${statusColor}20`, color: statusColor }}
          >
            <span
              className="w-1.5 h-1.5 rounded-full"
              style={{
                backgroundColor: statusColor,
                animation: isActive ? 'pulse 2s infinite' : 'none',
              }}
            />
            {statusText}
          </div>
        </div>
        {/* A2A + Wallet Links */}
        <div className="flex items-center justify-between mt-2 text-[10px]">
          <div className="flex items-center gap-3">
            <a
              href={`${agentUrl}/.well-known/agent.json`}
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-400 hover:text-blue-300 flex items-center gap-1"
            >
              A2A v0.3
              <ExternalLink size={9} />
            </a>
            <span className="text-gray-500 font-mono">skill:{skillId}</span>
          </div>
          <a
            href={`https://basescan.org/address/${walletAddress}`}
            target="_blank"
            rel="noopener noreferrer"
            className="text-gray-500 font-mono hover:text-gray-400 flex items-center gap-1"
          >
            <Wallet size={10} />
            {walletAddress.slice(0, 6)}...{walletAddress.slice(-4)}
            <ExternalLink size={9} />
          </a>
        </div>
      </div>

      {/* What It Does */}
      <div className="px-4 py-3 border-b" style={{ borderColor: `${color}15`, backgroundColor: '#0d0d10' }}>
        <div className="text-[10px] uppercase tracking-wide mb-1.5" style={{ color: `${color}90` }}>What It Does</div>
        <div className="space-y-1">
          {whatItDoes.map((item, i) => (
            <div key={i} className="text-xs text-gray-200 flex items-start gap-2">
              <span style={{ color: `${color}70` }} className="mt-0.5">•</span>
              <span>{item}</span>
            </div>
          ))}
        </div>
      </div>

      {/* What It Proves */}
      <div className="px-4 py-3 border-b" style={{ borderColor: `${color}15`, backgroundColor: `${color}08` }}>
        <div className="text-[10px] uppercase tracking-wide mb-1.5" style={{ color: `${color}90` }}>What It Proves</div>
        <div className="space-y-1">
          {whatItProves.map((item, i) => (
            <div key={i} className="text-xs text-gray-200 flex items-start gap-2">
              <span style={{ color }} className="mt-0.5">✓</span>
              <span>{item}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Wallet Balance */}
      {walletBalance && (
        <div className="px-4 py-2 border-b" style={{ borderColor: `${color}15`, backgroundColor: '#0c0c0f' }}>
          <div className="flex items-center justify-between text-xs">
            <div className="flex items-center gap-1.5 text-gray-400">
              <Wallet size={12} />
              <span>Balance</span>
            </div>
            <div className="flex items-center gap-3 font-mono">
              <span className="text-green-400">${walletBalance.usdc.toFixed(2)}</span>
              <span className="text-blue-400">{walletBalance.eth.toFixed(4)} ETH</span>
            </div>
          </div>
        </div>
      )}

      {/* Stats Row */}
      <div className="px-4 py-3 border-b" style={{ borderColor: `${color}15`, backgroundColor: '#0d0d10' }}>
        <div className="grid grid-cols-3 gap-4 text-center">
          <div>
            <div className="text-lg font-bold text-white font-mono">{stats.urlsProcessed.toLocaleString()}</div>
            <div className="text-[10px] text-gray-400 uppercase">URLs</div>
          </div>
          <div>
            <div className="text-lg font-bold text-green-400 font-mono">${stats.earned.toFixed(3)}</div>
            <div className="text-[10px] text-gray-400 uppercase">Earned</div>
          </div>
          <div>
            <div className="text-lg font-bold text-red-400 font-mono">${stats.spent.toFixed(3)}</div>
            <div className="text-[10px] text-gray-400 uppercase">Spent</div>
          </div>
        </div>
      </div>

      {/* Models Section - Collapsible */}
      <div className="border-b" style={{ borderColor: `${color}15` }}>
        <button
          onClick={() => toggleSection('models')}
          className="w-full px-4 py-2 flex items-center justify-between text-xs transition-colors"
          style={{ color: `${color}90` }}
        >
          <span className="uppercase tracking-wide">ONNX Models</span>
          {expandedSections.models ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
        </button>
        {expandedSections.models && (
          <div className="px-4 pb-3" style={{ backgroundColor: '#0c0c0f' }}>
            <table className="w-full text-xs">
              <thead>
                <tr style={{ color: `${color}70` }}>
                  <th className="text-left py-1 font-normal">Model</th>
                  <th className="text-left py-1 font-normal">Size</th>
                  <th className="text-left py-1 font-normal">Shape</th>
                </tr>
              </thead>
              <tbody className="font-mono">
                {models.map((model) => (
                  <tr key={model.name} className="text-gray-300">
                    <td className="py-1">{model.name}</td>
                    <td className="py-1">{model.size}</td>
                    <td className="py-1">{model.shape}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="mt-2 space-y-1">
              {models.map((model) => (
                <div key={`${model.name}-commit`} className="flex items-center gap-2 text-[10px]">
                  <span className="w-20" style={{ color: `${color}70` }}>{model.name}:</span>
                  <code className="text-gray-400 px-1.5 py-0.5 rounded flex-1 truncate" style={{ backgroundColor: `${color}10` }}>
                    {model.commitment}
                  </code>
                  <button
                    onClick={() => copyToClipboard(model.commitment, model.name)}
                    className="text-gray-500 hover:text-white transition-colors"
                  >
                    {copiedHash === model.name ? <Check size={12} className="text-green-400" /> : <Copy size={12} />}
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>

      {/* Proofs Section - Collapsible */}
      <div className="border-b" style={{ borderColor: `${color}15` }}>
        <button
          onClick={() => toggleSection('proofs')}
          className="w-full px-4 py-2 flex items-center justify-between text-xs transition-colors"
          style={{ color: `${color}90` }}
        >
          <div className="flex items-center gap-3">
            <span className="uppercase tracking-wide">Proofs</span>
            <span className="font-mono" style={{ color: `${color}60` }}>{proofStats.generated} gen / {proofStats.verified} ver</span>
          </div>
          {expandedSections.proofs ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
        </button>
        {expandedSections.proofs && (
          <div className="px-4 pb-3 space-y-3" style={{ backgroundColor: '#0c0c0f' }}>
            {/* Stats Table */}
            <table className="w-full text-xs">
              <thead>
                <tr style={{ color: `${color}70` }}>
                  <th className="text-left py-1 font-normal">Metric</th>
                  <th className="text-right py-1 font-normal">Value</th>
                </tr>
              </thead>
              <tbody className="font-mono text-gray-300">
                <tr>
                  <td className="py-1">Generated</td>
                  <td className="py-1 text-right">{proofStats.generated}</td>
                </tr>
                <tr>
                  <td className="py-1">Verified</td>
                  <td className="py-1 text-right">{proofStats.verified}</td>
                </tr>
                <tr>
                  <td className="py-1">Avg Gen Time</td>
                  <td className="py-1 text-right">{proofStats.avgGenMs}ms</td>
                </tr>
                <tr>
                  <td className="py-1">Avg Ver Time</td>
                  <td className="py-1 text-right">{proofStats.avgVerMs}ms</td>
                </tr>
                <tr>
                  <td className="py-1">Success Rate</td>
                  <td className="py-1 text-right text-green-400">
                    {proofStats.generated > 0 ? ((proofStats.verified / proofStats.generated) * 100).toFixed(0) : 0}%
                  </td>
                </tr>
              </tbody>
            </table>

            {/* Last Proof */}
            {lastProof && (
              <div className="rounded p-2" style={{ border: `1px solid ${color}25`, backgroundColor: `${color}08` }}>
                <div className="text-[10px] uppercase mb-1" style={{ color: `${color}70` }}>Last Proof</div>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div>
                    <span className="text-gray-500">Type: </span>
                    <span className="text-white font-mono">{lastProof.type}</span>
                  </div>
                  <div>
                    <span className="text-gray-500">Output: </span>
                    <span className="text-white font-mono">{lastProof.output}</span>
                  </div>
                  <div>
                    <span className="text-gray-500">Confidence: </span>
                    <span className="text-white font-mono">{(lastProof.confidence * 100).toFixed(0)}%</span>
                  </div>
                  <div>
                    <span className="text-gray-500">Gen: </span>
                    <span className="text-white font-mono">{lastProof.genMs}ms</span>
                  </div>
                </div>
                {lastProof.commitments.proof && (
                  <div className="mt-2 flex items-center gap-2 text-[10px]">
                    <span style={{ color: `${color}70` }}>Hash:</span>
                    <code className="text-gray-400 px-1.5 py-0.5 rounded flex-1 truncate" style={{ backgroundColor: `${color}10` }}>
                      {lastProof.commitments.proof}
                    </code>
                    <button
                      onClick={() => copyToClipboard(lastProof.commitments.proof, 'proof')}
                      className="text-gray-500 hover:text-white transition-colors"
                    >
                      {copiedHash === 'proof' ? <Check size={12} className="text-green-400" /> : <Copy size={12} />}
                    </button>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Activity Feed */}
      <div className="p-4" style={{ backgroundColor: '#0b0b0e' }}>
        <div className="text-[10px] uppercase tracking-wide mb-2" style={{ color: `${color}80` }}>Activity</div>
        <div className="space-y-1.5 min-h-[60px]">
          {events.length === 0 ? (
            <div className="text-xs text-gray-500 flex items-center gap-2">
              <span className="flex gap-0.5">
                <span className="w-1 h-1 rounded-full animate-pulse" style={{ backgroundColor: color }} />
                <span className="w-1 h-1 rounded-full animate-pulse" style={{ backgroundColor: color, animationDelay: '0.2s' }} />
                <span className="w-1 h-1 rounded-full animate-pulse" style={{ backgroundColor: color, animationDelay: '0.4s' }} />
              </span>
              Monitoring...
            </div>
          ) : (
            events.map((event, i) => (
              <div
                key={`${event.timestamp}-${i}`}
                className={`text-xs flex items-start gap-2 ${i === 0 ? 'text-white' : 'text-gray-500'}`}
              >
                <span className="text-[10px] mt-0.5 font-mono w-12" style={{ color: `${color}50` }}>
                  {new Date(event.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                </span>
                <span className={i === 0 && isActive ? 'animate-pulse' : ''}>
                  {formatEvent(event)}
                </span>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
