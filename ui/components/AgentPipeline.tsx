'use client';

import { useEffect, useState, useMemo } from 'react';
import { Activity, Shield, Zap, DollarSign, ArrowRight, Search, Scale, Microscope, Coins, Check, X, Wallet } from 'lucide-react';
import ProofCard from './ProofCard';
import VerificationChecklist from './VerificationChecklist';

type AgentState = 'idle' | 'active' | 'working' | 'proving';

// Agent wallet addresses (2-Agent Model: Scout + Analyst)
const WALLET_ADDRESSES = {
  analyst: '0x7ee88871fA9be48b62552F231a4976A11e559db8',
  scout: '0x269CBA662fE55c4fe1212c609090A31844C36ab8',
};

interface ProofStage {
  name: string;
  message: string;
  progress_pct: number;
}

interface ProofData {
  proof_hash: string;
  model_commitment: string;
  input_commitment: string;
  output_commitment: string;
  prove_time_ms: number;
  proof_size_bytes: number;
  decision?: string;
  confidence?: number;
  is_real_proof?: boolean;
  stages?: ProofStage[];
}

interface VerificationCheck {
  name: string;
  description: string;
  status: 'pending' | 'checking' | 'passed' | 'failed';
  detail?: string;
}

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

export default function AgentPipeline({ events, lastEvent, stats }: AgentPipelineProps) {
  const [particles, setParticles] = useState<{ id: number; from: string; to: string }[]>([]);
  const [proofProgress, setProofProgress] = useState<{ scout: number; analyst: number }>({ scout: 0, analyst: 0 });
  const [walletBalances, setWalletBalances] = useState<Record<string, { usdc: number; eth: number }>>({});

  // Fetch wallet balances
  useEffect(() => {
    const fetchBalances = async () => {
      const balances: Record<string, { usdc: number; eth: number }> = {};
      const usdcContract = '0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913';
      const balanceOfSelector = '0x70a08231';

      for (const [agent, address] of Object.entries(WALLET_ADDRESSES)) {
        try {
          // Fetch ETH balance
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

          // Fetch USDC balance
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
        } catch (e) {
          balances[agent] = { usdc: 0, eth: 0 };
        }
      }
      setWalletBalances(balances);
    };

    fetchBalances();
    const interval = setInterval(fetchBalances, 30000);
    return () => clearInterval(interval);
  }, []);

  // Track proof state for each agent (2-Agent Model)
  const [scoutSpendingProof, setScoutSpendingProof] = useState<ProofData | null>(null);
  const [analystSpendingProof, setAnalystSpendingProof] = useState<ProofData | null>(null);
  const [analystProof, setAnalystProof] = useState<ProofData | null>(null);
  const [scoutProofStage, setScoutProofStage] = useState<ProofStage | null>(null);
  const [analystProofStage, setAnalystProofStage] = useState<ProofStage | null>(null);
  const [spendingVerifyChecks, setSpendingVerifyChecks] = useState<VerificationCheck[]>([]);
  const [analystVerifyChecks, setAnalystVerifyChecks] = useState<VerificationCheck[]>([]);

  // Track current payment info for annotations
  const [currentPayment, setCurrentPayment] = useState<{ amount: number; recipient: string } | null>(null);

  // Update proof state based on events (2-Agent Model)
  useEffect(() => {
    if (!lastEvent) return;

    switch (lastEvent.type) {
      // Scout self-authorization events
      case 'SCOUT_AUTHORIZING':
        setScoutProofStage({
          name: 'AUTHORIZING',
          message: 'Generating spending proof...',
          progress_pct: 50,
        });
        setProofProgress(prev => ({ ...prev, scout: 50 }));
        break;

      case 'SCOUT_AUTHORIZED':
        setScoutProofStage(null);
        setProofProgress(prev => ({ ...prev, scout: 100 }));
        setScoutSpendingProof({
          proof_hash: lastEvent.data.proof_hash || '',
          model_commitment: '',
          input_commitment: '',
          output_commitment: '',
          prove_time_ms: lastEvent.data.prove_time_ms || 0,
          proof_size_bytes: 0,
          decision: lastEvent.data.decision,
          confidence: lastEvent.data.confidence,
          is_real_proof: true,
        });
        break;

      // Analyst self-authorization events
      case 'ANALYST_AUTHORIZING':
        setAnalystProofStage({
          name: 'AUTHORIZING',
          message: 'Generating spending proof...',
          progress_pct: 30,
        });
        break;

      case 'ANALYST_AUTHORIZED':
        setAnalystSpendingProof({
          proof_hash: lastEvent.data.proof_hash || '',
          model_commitment: '',
          input_commitment: '',
          output_commitment: '',
          prove_time_ms: lastEvent.data.prove_time_ms || 0,
          proof_size_bytes: 0,
          decision: lastEvent.data.decision,
          confidence: lastEvent.data.confidence,
          is_real_proof: true,
        });
        break;

      case 'ANALYST_PROVING':
        setAnalystProofStage({
          name: lastEvent.data.stage || 'PROVING',
          message: lastEvent.data.message || 'Generating zkML proof...',
          progress_pct: lastEvent.data.progress_pct || 50,
        });
        setProofProgress(prev => ({ ...prev, analyst: lastEvent.data.progress_pct || 50 }));
        break;

      case 'ANALYST_RESPONSE':
        setAnalystProofStage(null);
        setProofProgress(prev => ({ ...prev, analyst: 100 }));
        if (lastEvent.data.proof) {
          setAnalystProof({
            proof_hash: lastEvent.data.proof.proof_hash || '',
            model_commitment: lastEvent.data.proof.model_commitment || '',
            input_commitment: lastEvent.data.proof.input_commitment || '',
            output_commitment: lastEvent.data.proof.output_commitment || '',
            prove_time_ms: lastEvent.data.proof.prove_time_ms || 0,
            proof_size_bytes: lastEvent.data.proof.proof_size_bytes || 0,
            decision: lastEvent.data.classification,
            confidence: lastEvent.data.confidence,
            is_real_proof: lastEvent.data.proof.is_real_proof || false,
            stages: lastEvent.data.proof.stages,
          });
        }
        break;

      // Spending proof verification
      case 'SPENDING_PROOF_VERIFIED':
        if (lastEvent.data.checks) {
          setSpendingVerifyChecks(lastEvent.data.checks);
        }
        break;

      case 'WORK_VERIFIED':
        if (lastEvent.data.checks) {
          setAnalystVerifyChecks(lastEvent.data.checks);
        }
        break;

      case 'PAYMENT_SENDING':
        setCurrentPayment({
          amount: lastEvent.data.amount_usdc,
          recipient: lastEvent.data.recipient,
        });
        break;

      case 'PAYMENT_SENT':
        setTimeout(() => setCurrentPayment(null), 2000);
        break;
    }
  }, [lastEvent]);

  // Derive agent states from last event (2-Agent Model)
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

  // Determine active flow direction (2-Agent Model: Scout ←→ Analyst)
  const activeFlow = useMemo(() => {
    if (!lastEvent) return null;

    // Analyst → Scout: Analyst pays Scout for discovery
    if (lastEvent.type === 'ANALYST_AUTHORIZING' ||
        (lastEvent.type === 'PAYMENT_SENDING' && lastEvent.data.recipient?.includes('scout'))) {
      return 'analyst-scout';
    }

    // Scout → Analyst: Scout returns URLs, Scout pays Analyst feedback
    if (lastEvent.type === 'SCOUT_AUTHORIZING' ||
        lastEvent.type === 'SCOUT_AUTHORIZED' ||
        (lastEvent.type === 'PAYMENT_SENDING' && lastEvent.data.recipient?.includes('analyst'))) {
      return 'scout-analyst';
    }

    if (lastEvent.type === 'ANALYST_PROCESSING' ||
        lastEvent.type === 'ANALYST_PROVING' ||
        lastEvent.type === 'SPENDING_PROOF_VERIFIED') {
      return 'scout-analyst';
    }

    if (lastEvent.type === 'ANALYST_RESPONSE' || lastEvent.type === 'WORK_VERIFIED') {
      return 'analyst-scout';
    }

    return null;
  }, [lastEvent]);

  // Spawn particles on flow
  useEffect(() => {
    if (activeFlow) {
      const [from, to] = activeFlow.split('-');
      const id = Date.now();
      setParticles(prev => [...prev, { id, from, to }]);
      setTimeout(() => {
        setParticles(prev => prev.filter(p => p.id !== id));
      }, 2000);
    }
  }, [activeFlow, lastEvent?.timestamp]);

  // Filter events by agent (2-Agent Model)
  const getAgentEvents = (agent: string) => {
    const prefixes: Record<string, string[]> = {
      scout: ['SCOUT', 'DATABASE'],
      analyst: ['ANALYST', 'WORK_VERIFIED', 'SPENDING_PROOF_VERIFIED'],
    };
    return events
      .filter(e => prefixes[agent]?.some(p => e.type.startsWith(p)))
      .slice(0, 4);
  };

  const formatEventMessage = (event: AgentEvent): string => {
    switch (event.type) {
      case 'SCOUT_FOUND_URLS':
        return `Found ${event.data.url_count} URLs`;
      case 'SCOUT_AUTHORIZING':
        return 'Generating spending proof...';
      case 'SCOUT_AUTHORIZED':
        return `${event.data.decision} (${(event.data.confidence * 100).toFixed(0)}%)`;
      case 'ANALYST_AUTHORIZING':
        return 'Generating spending proof...';
      case 'ANALYST_AUTHORIZED':
        return `${event.data.decision} (${(event.data.confidence * 100).toFixed(0)}%)`;
      case 'ANALYST_PROCESSING':
        return `Classifying ${event.data.url_count} URLs`;
      case 'ANALYST_PROVING':
        return 'Generating zkML proof...';
      case 'ANALYST_RESPONSE':
        return `${event.data.phishing_count} phishing found`;
      case 'SPENDING_PROOF_VERIFIED':
        return event.data.valid !== false ? `${event.data.agent} proof verified` : `${event.data.agent} proof FAILED`;
      case 'WORK_VERIFIED':
        return event.data.valid !== false ? 'Work proof verified' : 'Work proof FAILED';
      case 'DATABASE_UPDATED':
        return `+${event.data.urls_added} URLs saved`;
      default:
        return event.type;
    }
  };

  return (
    <div className="w-full network-alive">
      {/* Pipeline Container with SVG Connections (2-Agent Model) */}
      <div className="relative flex items-stretch justify-center gap-8">
        {/* SVG Layer for Connection Arrows */}
        <svg
          className="absolute inset-0 w-full h-full pointer-events-none"
          style={{ zIndex: 0 }}
          viewBox="0 0 100 100"
          preserveAspectRatio="none"
        >
          <defs>
            {/* Arrow markers */}
            <marker id="arrowRight" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">
              <path d="M 0 0 L 10 5 L 0 10 z" fill="#3b82f6" className="arrow-marker" />
            </marker>
            <marker id="arrowRightActive" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto">
              <path d="M 0 0 L 10 5 L 0 10 z" fill="#fbbf24" />
            </marker>
            <marker id="arrowLeft" markerWidth="10" markerHeight="10" refX="2" refY="5" orient="auto-start-reverse">
              <path d="M 10 0 L 0 5 L 10 10 z" fill="#22d3ee" />
            </marker>
            {/* Glow filter */}
            <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="2" result="coloredBlur" />
              <feMerge>
                <feMergeNode in="coloredBlur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          {/* Connection: Analyst to Scout - Payment flow (Analyst pays Scout for discovery) */}
          <line
            x1="30" y1="42"
            x2="70" y2="42"
            stroke={activeFlow === 'analyst-scout' ? '#fbbf24' : '#22d3ee'}
            strokeWidth="0.5"
            strokeOpacity={activeFlow === 'analyst-scout' ? 1 : 0.6}
            markerEnd={activeFlow === 'analyst-scout' ? 'url(#arrowRightActive)' : 'url(#arrowRight)'}
            className={activeFlow === 'analyst-scout' ? '' : 'flow-line'}
            filter={activeFlow === 'analyst-scout' ? 'url(#glow)' : 'none'}
          />
          {/* Scout to Analyst - URLs + proof, Scout pays Analyst feedback */}
          <line
            x1="70" y1="48"
            x2="30" y2="48"
            stroke={activeFlow === 'scout-analyst' ? '#fbbf24' : '#3b82f6'}
            strokeWidth="0.5"
            strokeOpacity={activeFlow === 'scout-analyst' ? 1 : 0.4}
            markerEnd="url(#arrowLeft)"
            className={activeFlow === 'scout-analyst' ? '' : 'flow-line'}
            style={{ animationDirection: 'reverse' }}
            filter={activeFlow === 'scout-analyst' ? 'url(#glow)' : 'none'}
          />

        </svg>

        {/* Payment Annotation Overlay */}
        {currentPayment && (
          <div className="absolute top-0 left-1/2 -translate-x-1/2 z-20 float-annotation">
            <div className="bg-green-500/20 border border-green-500/50 rounded-lg px-3 py-1.5 backdrop-blur-sm">
              <div className="flex items-center gap-2 text-green-400">
                <Coins size={16} className="text-green-400" />
                <span className="font-mono text-sm font-bold">{currentPayment.amount.toFixed(4)} USDC</span>
              </div>
            </div>
          </div>
        )}

        {/* Flow Label Annotations (2-Agent Model) */}
        {activeFlow === 'analyst-scout' && (
          <div className="absolute top-4 left-1/2 -translate-x-1/2 z-10">
            <span className="text-xs text-yellow-400 bg-yellow-400/10 px-2 py-1 rounded animate-pulse">
              $0.001 + Discovery Request
            </span>
          </div>
        )}
        {activeFlow === 'scout-analyst' && (
          <div className="absolute top-4 left-1/2 -translate-x-1/2 z-10">
            <span className="text-xs text-cyan-400 bg-cyan-400/10 px-2 py-1 rounded animate-pulse">
              URLs + Spending Proof + $0.001 Feedback
            </span>
          </div>
        )}

        {/* Analyst Agent - Pays Scout for discovery, receives feedback payment */}
        <AgentCard
          name="Threat Analysis Agent"
          role="Self-Auth Spending + Classification"
          icon={<Microscope size={24} />}
          color="cyan"
          colorHex="#22d3ee"
          state={agentStates.analyst}
          proofProgress={agentStates.analyst === 'proving' ? proofProgress.analyst : 0}
          stats={[
            { label: 'Phishing', value: stats?.phishing_count?.toLocaleString() || '0' },
            { label: 'Earned', value: `$${(stats?.analyst_paid_usdc || 0).toFixed(3)}` },
          ]}
          events={getAgentEvents('analyst')}
          formatEvent={formatEventMessage}
          walletBalance={walletBalances.analyst}
          walletAddress={WALLET_ADDRESSES.analyst}
        />

        {/* Spacer for SVG lines */}
        <div className="w-32" />

        {/* Scout Agent - Self-authorizes, receives from Analyst, pays Analyst feedback */}
        <AgentCard
          name="Scout Agent"
          role="Self-Auth Spending + Discovery"
          icon={<Search size={24} />}
          color="blue"
          colorHex="#3b82f6"
          state={agentStates.scout}
          proofProgress={agentStates.scout === 'proving' ? proofProgress.scout : 0}
          stats={[
            { label: 'URLs Found', value: stats?.total_urls?.toLocaleString() || '0' },
            { label: 'Sources', value: '4' },
          ]}
          events={getAgentEvents('scout')}
          formatEvent={formatEventMessage}
          walletBalance={walletBalances.scout}
          walletAddress={WALLET_ADDRESSES.scout}
        />
      </div>

      {/* Flow Legend (2-Agent Model) */}
      <div className="flex justify-center gap-6 mt-6 text-xs text-gray-500">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-yellow-400 shadow-[0_0_8px_#fbbf24] status-active" />
          <span>Discovery Request</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-cyan-400 shadow-[0_0_8px_#22d3ee] status-active" style={{ animationDelay: '0.3s' }} />
          <span>URLs + Spending Proof</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-green-400 shadow-[0_0_8px_#22c55e] status-active" style={{ animationDelay: '0.6s' }} />
          <span>$0.001 Payments</span>
        </div>
        <div className="flex items-center gap-2 ml-4 border-l border-gray-700 pl-4">
          <span className="text-cyan-400">2-Agent Circular Economy</span>
          <span className="flex gap-0.5">
            <span className="w-1 h-1 rounded-full bg-cyan-400 stream-dot" />
            <span className="w-1 h-1 rounded-full bg-cyan-400 stream-dot" />
            <span className="w-1 h-1 rounded-full bg-cyan-400 stream-dot" />
          </span>
        </div>
      </div>

      {/* Proof Details Section (2-Agent Model) */}
      {(scoutSpendingProof || analystProof || agentStates.scout === 'proving' || agentStates.analyst === 'proving') && (
        <div className="mt-6 grid grid-cols-2 gap-4">
          {/* Scout Spending Proof Card */}
          <div className="space-y-3">
            <ProofCard
              title="Scout Spending Proof"
              proof={scoutSpendingProof}
              isGenerating={agentStates.scout === 'proving'}
              currentStage={scoutProofStage}
              color="blue"
              colorHex="#3b82f6"
            />
            {spendingVerifyChecks.length > 0 && (
              <VerificationChecklist
                checks={spendingVerifyChecks}
                verifyTimeMs={lastEvent?.type === 'SPENDING_PROOF_VERIFIED' ? lastEvent.data.verify_time_ms : undefined}
              />
            )}
          </div>

          {/* Analyst Classification Proof Card */}
          <div className="space-y-3">
            <ProofCard
              title="Classification Proof"
              proof={analystProof}
              isGenerating={agentStates.analyst === 'proving' && !analystSpendingProof}
              currentStage={analystProofStage}
              color="cyan"
              colorHex="#22d3ee"
            />
            {analystVerifyChecks.length > 0 && (
              <VerificationChecklist
                checks={analystVerifyChecks}
                verifyTimeMs={lastEvent?.type === 'WORK_VERIFIED' ? lastEvent.data.verify_time_ms : undefined}
              />
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// Agent Card Component with Progress Ring
function AgentCard({
  name,
  role,
  icon,
  color,
  colorHex,
  state,
  proofProgress,
  stats,
  events,
  formatEvent,
  walletBalance,
  walletAddress,
}: {
  name: string;
  role: string;
  icon: React.ReactNode;
  color: string;
  colorHex: string;
  state: AgentState;
  proofProgress: number;
  stats: { label: string; value: string }[];
  events: AgentEvent[];
  formatEvent: (e: AgentEvent) => string;
  walletBalance?: { usdc: number; eth: number };
  walletAddress?: string;
}) {
  const isActive = state !== 'idle';
  const isProving = state === 'proving';
  const isWorking = state === 'working';

  // Calculate progress ring values
  const radius = 32;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (proofProgress / 100) * circumference;

  return (
    <div
      className={`flex-1 rounded-xl border bg-gray-900/50 backdrop-blur-sm transition-all duration-500 relative z-10 ${
        isProving ? 'glow-breathe' : isWorking ? '' : 'idle-glow'
      }`}
      style={{
        borderColor: isActive ? colorHex : `${colorHex}60`,
        ['--glow-color' as any]: colorHex,
        boxShadow: isWorking
          ? `0 0 30px ${colorHex}40`
          : isActive
          ? `0 0 15px ${colorHex}20`
          : `0 0 10px ${colorHex}15`,
      }}
    >
      {/* Scan effect overlay when idle */}
      {!isActive && (
        <div className="absolute inset-0 rounded-xl overflow-hidden pointer-events-none scan-effect" />
      )}

      {/* Ripple rings when proving */}
      {isProving && (
        <>
          <div className="ripple-ring" style={{ ['--glow-color' as any]: colorHex, borderColor: colorHex }} />
          <div className="ripple-ring" style={{ ['--glow-color' as any]: colorHex, borderColor: colorHex }} />
          <div className="ripple-ring" style={{ ['--glow-color' as any]: colorHex, borderColor: colorHex }} />
        </>
      )}

      {/* Header */}
      <div className={`p-4 border-b border-gray-800`} style={{ backgroundColor: `${colorHex}10` }}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            {/* Avatar with Progress Ring */}
            <div className="relative">
              <div
                className={`w-16 h-16 rounded-full flex items-center justify-center border-2 relative ${
                  isWorking ? 'animate-spin-slow' : isProving ? '' : 'avatar-idle'
                }`}
                style={{
                  borderColor: colorHex,
                  backgroundColor: `${colorHex}20`,
                }}
              >
                <span style={{ color: colorHex }}>{icon}</span>
              </div>

              {/* Progress Ring SVG */}
              {isProving && (
                <svg
                  className="absolute -inset-1 w-[72px] h-[72px] progress-ring"
                  viewBox="0 0 72 72"
                >
                  {/* Background ring */}
                  <circle
                    cx="36"
                    cy="36"
                    r={radius}
                    fill="none"
                    stroke={`${colorHex}30`}
                    strokeWidth="4"
                  />
                  {/* Progress ring */}
                  <circle
                    cx="36"
                    cy="36"
                    r={radius}
                    fill="none"
                    stroke={colorHex}
                    strokeWidth="4"
                    strokeLinecap="round"
                    strokeDasharray={circumference}
                    strokeDashoffset={proofProgress > 0 ? strokeDashoffset : circumference}
                    className={proofProgress === 0 ? 'progress-ring-indeterminate' : 'transition-all duration-300'}
                    style={{ filter: `drop-shadow(0 0 6px ${colorHex})` }}
                  />
                </svg>
              )}

              {/* Spinning dashed ring when working */}
              {isWorking && (
                <div
                  className="absolute -inset-2 rounded-full border-2 border-dashed animate-spin"
                  style={{ borderColor: colorHex, animationDuration: '3s' }}
                />
              )}
            </div>

            <div>
              <h3 className="font-bold text-lg" style={{ color: colorHex }}>
                {name}
              </h3>
              <p className="text-xs text-gray-500">{role}</p>
            </div>
          </div>

          {/* Status Badge */}
          <div
            className={`px-2 py-1 rounded-full text-xs font-medium flex items-center gap-1 ${
              isProving
                ? 'bg-yellow-500/20 text-yellow-400'
                : isWorking
                ? 'bg-green-500/20 text-green-400'
                : isActive
                ? 'bg-blue-500/20 text-blue-400'
                : 'bg-gray-700/50'
            }`}
            style={{ color: !isActive ? colorHex : undefined }}
          >
            {isProving && (
              <span className="w-2 h-2 rounded-full bg-yellow-400 status-active" />
            )}
            {isWorking && (
              <span className="w-2 h-2 rounded-full bg-green-400 status-active" />
            )}
            {isActive && !isProving && !isWorking && (
              <span className="w-2 h-2 rounded-full bg-blue-400" />
            )}
            {!isActive && (
              <span className="w-2 h-2 rounded-full status-active" style={{ backgroundColor: colorHex }} />
            )}
            <span>
              {isProving ? 'PROVING' : isWorking ? 'WORKING' : isActive ? 'ACTIVE' : 'READY'}
            </span>
          </div>
        </div>
      </div>

      {/* Wallet Balance */}
      {walletBalance && (
        <div className="px-4 py-2 border-b border-gray-800 bg-gray-800/30">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-1.5">
              <Wallet size={12} className="text-gray-500" />
              <span className="text-xs text-gray-500 font-mono">
                {walletAddress ? `${walletAddress.slice(0, 6)}...${walletAddress.slice(-4)}` : ''}
              </span>
            </div>
            <div className="flex items-center gap-3 text-xs">
              <span className="text-green-400 font-mono font-medium">
                ${walletBalance.usdc.toFixed(2)} USDC
              </span>
              <span className="text-blue-400 font-mono">
                {walletBalance.eth.toFixed(4)} ETH
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Stats */}
      <div className="p-4 border-b border-gray-800">
        <div className="grid grid-cols-2 gap-4">
          {stats.map((stat) => (
            <div key={stat.label}>
              <p className="text-xs text-gray-500 uppercase tracking-wide">{stat.label}</p>
              <p className="text-xl font-bold text-white">{stat.value}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Activity Feed */}
      <div className="p-4">
        <p className="text-xs text-gray-500 uppercase tracking-wide mb-2">Recent Activity</p>
        <div className="space-y-2 min-h-[100px]">
          {events.length === 0 ? (
            <div className="flex items-center gap-2 text-sm" style={{ color: colorHex }}>
              <span className="flex gap-1">
                <span className="w-1.5 h-1.5 rounded-full stream-dot" style={{ backgroundColor: colorHex }} />
                <span className="w-1.5 h-1.5 rounded-full stream-dot" style={{ backgroundColor: colorHex }} />
                <span className="w-1.5 h-1.5 rounded-full stream-dot" style={{ backgroundColor: colorHex }} />
              </span>
              <span className="opacity-70">Monitoring network</span>
            </div>
          ) : (
            events.map((event, i) => (
              <div
                key={`${event.timestamp}-${i}`}
                className={`text-sm flex items-start gap-2 ${i === 0 ? 'text-white' : 'text-gray-500'}`}
              >
                <span className="text-gray-600 text-xs mt-0.5">
                  {new Date(event.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </span>
                <span className={i === 0 ? 'animate-pulse' : ''}>
                  {formatEvent(event)}
                  {i === 0 && event.type.includes('VERIFIED') && (
                    <Check size={12} className="ml-1 text-green-400 inline" />
                  )}
                </span>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
