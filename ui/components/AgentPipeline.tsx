'use client';

import { useEffect, useState, useMemo } from 'react';
import { Activity, Shield, Zap, DollarSign, ArrowRight } from 'lucide-react';
import ProofCard from './ProofCard';
import VerificationChecklist from './VerificationChecklist';

type AgentState = 'idle' | 'active' | 'working' | 'proving';

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
  const [proofProgress, setProofProgress] = useState<{ policy: number; analyst: number }>({ policy: 0, analyst: 0 });

  // Track proof state for each agent
  const [policyProof, setPolicyProof] = useState<ProofData | null>(null);
  const [analystProof, setAnalystProof] = useState<ProofData | null>(null);
  const [policyProofStage, setPolicyProofStage] = useState<ProofStage | null>(null);
  const [analystProofStage, setAnalystProofStage] = useState<ProofStage | null>(null);
  const [policyVerifyChecks, setPolicyVerifyChecks] = useState<VerificationCheck[]>([]);
  const [analystVerifyChecks, setAnalystVerifyChecks] = useState<VerificationCheck[]>([]);

  // Track current payment info for annotations
  const [currentPayment, setCurrentPayment] = useState<{ amount: number; recipient: string } | null>(null);

  // Update proof state based on events
  useEffect(() => {
    if (!lastEvent) return;

    switch (lastEvent.type) {
      case 'POLICY_PROVING':
        setPolicyProofStage({
          name: lastEvent.data.stage || 'PROVING',
          message: lastEvent.data.message || 'Generating zkML proof...',
          progress_pct: lastEvent.data.progress_pct || 50,
        });
        setProofProgress(prev => ({ ...prev, policy: lastEvent.data.progress_pct || 50 }));
        break;

      case 'POLICY_RESPONSE':
        setPolicyProofStage(null);
        setProofProgress(prev => ({ ...prev, policy: 100 }));
        if (lastEvent.data.proof) {
          setPolicyProof({
            proof_hash: lastEvent.data.proof.proof_hash || '',
            model_commitment: lastEvent.data.proof.model_commitment || '',
            input_commitment: lastEvent.data.proof.input_commitment || '',
            output_commitment: lastEvent.data.proof.output_commitment || '',
            prove_time_ms: lastEvent.data.proof.prove_time_ms || 0,
            proof_size_bytes: lastEvent.data.proof.proof_size_bytes || 0,
            decision: lastEvent.data.decision,
            confidence: lastEvent.data.confidence,
            is_real_proof: lastEvent.data.proof.is_real_proof || false,
            stages: lastEvent.data.proof.stages,
          });
        }
        break;

      case 'POLICY_VERIFIED':
        if (lastEvent.data.checks) {
          setPolicyVerifyChecks(lastEvent.data.checks);
        }
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

  // Derive agent states from last event
  const agentStates = useMemo(() => {
    const states = { scout: 'idle' as AgentState, policy: 'idle' as AgentState, analyst: 'idle' as AgentState };

    if (!lastEvent) return states;

    switch (lastEvent.type) {
      case 'SCOUT_FOUND_URLS':
        states.scout = 'working';
        break;
      case 'POLICY_REQUESTING':
        states.scout = 'active';
        states.policy = 'working';
        break;
      case 'POLICY_PROVING':
        states.policy = 'proving';
        break;
      case 'POLICY_RESPONSE':
      case 'POLICY_VERIFIED':
        states.policy = 'active';
        break;
      case 'ANALYST_PROCESSING':
        states.analyst = 'working';
        break;
      case 'ANALYST_PROVING':
        states.analyst = 'proving';
        break;
      case 'ANALYST_RESPONSE':
      case 'WORK_VERIFIED':
        states.analyst = 'active';
        break;
      case 'DATABASE_UPDATED':
        states.scout = 'active';
        break;
    }
    return states;
  }, [lastEvent]);

  // Determine active flow direction
  const activeFlow = useMemo(() => {
    if (!lastEvent) return null;

    if (lastEvent.type === 'POLICY_REQUESTING' || lastEvent.type === 'PAYMENT_SENDING') {
      if (lastEvent.data.recipient?.includes('policy')) return 'scout-policy';
      if (lastEvent.data.recipient?.includes('analyst')) return 'policy-analyst';
      return 'scout-policy';
    }
    if (lastEvent.type === 'POLICY_RESPONSE' || lastEvent.type === 'POLICY_VERIFIED') return 'policy-scout';
    if (lastEvent.type === 'ANALYST_PROCESSING') return 'policy-analyst';
    if (lastEvent.type === 'ANALYST_RESPONSE' || lastEvent.type === 'WORK_VERIFIED') return 'analyst-scout';
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

  // Filter events by agent
  const getAgentEvents = (agent: string) => {
    const prefixes: Record<string, string[]> = {
      scout: ['SCOUT', 'DATABASE'],
      policy: ['POLICY'],
      analyst: ['ANALYST', 'WORK_VERIFIED'],
    };
    return events
      .filter(e => prefixes[agent]?.some(p => e.type.startsWith(p)))
      .slice(0, 4);
  };

  const formatEventMessage = (event: AgentEvent): string => {
    switch (event.type) {
      case 'SCOUT_FOUND_URLS':
        return `Found ${event.data.url_count} URLs`;
      case 'POLICY_REQUESTING':
        return `Reviewing ${event.data.url_count} URLs`;
      case 'POLICY_PROVING':
        return 'Generating zkML proof...';
      case 'POLICY_RESPONSE':
        return `${event.data.decision} (${(event.data.confidence * 100).toFixed(0)}%)`;
      case 'POLICY_VERIFIED':
        return `Proof verified`;
      case 'ANALYST_PROCESSING':
        return `Classifying ${event.data.url_count} URLs`;
      case 'ANALYST_PROVING':
        return 'Generating zkML proof...';
      case 'ANALYST_RESPONSE':
        return `${event.data.phishing_count} phishing found`;
      case 'WORK_VERIFIED':
        return 'Work proof verified';
      case 'DATABASE_UPDATED':
        return `+${event.data.urls_added} URLs saved`;
      default:
        return event.type;
    }
  };

  return (
    <div className="w-full network-alive">
      {/* Pipeline Container with SVG Connections */}
      <div className="relative flex items-stretch justify-between gap-2">
        {/* SVG Layer for Connection Lines */}
        <svg
          className="absolute inset-0 w-full h-full pointer-events-none"
          style={{ zIndex: 0 }}
        >
          <defs>
            {/* Gradient for active flow */}
            <linearGradient id="flowGradientRight" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#fbbf24" stopOpacity="0.2" />
              <stop offset="50%" stopColor="#fbbf24" stopOpacity="1" />
              <stop offset="100%" stopColor="#fbbf24" stopOpacity="0.2" />
            </linearGradient>
            <linearGradient id="flowGradientLeft" x1="100%" y1="0%" x2="0%" y2="0%">
              <stop offset="0%" stopColor="#22d3ee" stopOpacity="0.2" />
              <stop offset="50%" stopColor="#22d3ee" stopOpacity="1" />
              <stop offset="100%" stopColor="#22d3ee" stopOpacity="0.2" />
            </linearGradient>
            <linearGradient id="paymentGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#22c55e" stopOpacity="0.2" />
              <stop offset="50%" stopColor="#22c55e" stopOpacity="1" />
              <stop offset="100%" stopColor="#22c55e" stopOpacity="0.2" />
            </linearGradient>
            {/* Glow filter */}
            <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur stdDeviation="3" result="coloredBlur" />
              <feMerge>
                <feMergeNode in="coloredBlur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
          </defs>

          {/* Connection: Scout to Policy */}
          <g className="connection-scout-policy">
            {/* Background ambient line - always visible with glow */}
            <path
              d="M 25% 50% Q 32% 35% 40% 50%"
              fill="none"
              stroke="#3b82f6"
              strokeWidth="2"
              strokeOpacity="0.3"
              className="connection-ambient"
            />
            {/* Active flow line */}
            <path
              d="M 25% 50% Q 32% 35% 40% 50%"
              fill="none"
              stroke={activeFlow === 'scout-policy' ? 'url(#flowGradientRight)' : activeFlow === 'policy-scout' ? 'url(#flowGradientLeft)' : '#3b82f6'}
              strokeWidth={activeFlow?.includes('policy') && activeFlow?.includes('scout') ? 3 : 2}
              strokeOpacity={activeFlow?.includes('policy') && activeFlow?.includes('scout') ? 1 : 0.5}
              className={activeFlow?.includes('policy') && activeFlow?.includes('scout') ? 'dash-flow' : 'continuous-flow'}
              filter={activeFlow?.includes('policy') && activeFlow?.includes('scout') ? 'url(#glow)' : 'none'}
            />
            {/* Ambient particles - always flowing (dimmer when idle) */}
            {[0, 1, 2].map(i => (
              <circle
                key={`ambient-sp-${i}`}
                r={activeFlow === 'scout-policy' ? 4 : 2}
                fill={activeFlow === 'scout-policy' ? '#fbbf24' : '#3b82f6'}
                opacity={activeFlow === 'scout-policy' ? 1 : 0.4}
                filter={activeFlow === 'scout-policy' ? 'url(#glow)' : 'none'}
                style={{
                  offsetPath: "path('M 25% 50% Q 32% 35% 40% 50%')",
                  offsetRotate: '0deg',
                } as any}
                className={activeFlow === 'scout-policy' ? 'path-particle' : 'ambient-particle'}
              />
            ))}
            {/* Reverse flow particles */}
            {activeFlow === 'policy-scout' && (
              <>
                {[0, 1, 2].map(i => (
                  <circle
                    key={i}
                    r="4"
                    fill="#22d3ee"
                    filter="url(#glow)"
                    style={{
                      offsetPath: "path('M 40% 50% Q 32% 35% 25% 50%')",
                      offsetRotate: '0deg',
                    } as any}
                    className="path-particle"
                  />
                ))}
              </>
            )}
          </g>

          {/* Connection: Policy to Analyst */}
          <g className="connection-policy-analyst">
            {/* Background ambient line - always visible with glow */}
            <path
              d="M 60% 50% Q 68% 35% 75% 50%"
              fill="none"
              stroke="#a855f7"
              strokeWidth="2"
              strokeOpacity="0.3"
              className="connection-ambient"
              style={{ animationDelay: '1.5s' }}
            />
            {/* Active flow line */}
            <path
              d="M 60% 50% Q 68% 35% 75% 50%"
              fill="none"
              stroke={activeFlow === 'policy-analyst' ? 'url(#flowGradientRight)' : activeFlow === 'analyst-scout' ? 'url(#flowGradientLeft)' : '#a855f7'}
              strokeWidth={activeFlow === 'policy-analyst' || activeFlow === 'analyst-scout' ? 3 : 2}
              strokeOpacity={activeFlow === 'policy-analyst' || activeFlow === 'analyst-scout' ? 1 : 0.5}
              className={activeFlow === 'policy-analyst' || activeFlow === 'analyst-scout' ? 'dash-flow' : 'continuous-flow'}
              filter={activeFlow === 'policy-analyst' || activeFlow === 'analyst-scout' ? 'url(#glow)' : 'none'}
            />
            {/* Ambient particles - always flowing */}
            {[0, 1, 2].map(i => (
              <circle
                key={`ambient-pa-${i}`}
                r={activeFlow === 'policy-analyst' ? 4 : 2}
                fill={activeFlow === 'policy-analyst' ? '#fbbf24' : '#a855f7'}
                opacity={activeFlow === 'policy-analyst' ? 1 : 0.4}
                filter={activeFlow === 'policy-analyst' ? 'url(#glow)' : 'none'}
                style={{
                  offsetPath: "path('M 60% 50% Q 68% 35% 75% 50%')",
                  offsetRotate: '0deg',
                  animationDelay: `${i * 2 + 1}s`,
                } as any}
                className={activeFlow === 'policy-analyst' ? 'path-particle' : 'ambient-particle'}
              />
            ))}
            {/* Reverse flow particles */}
            {activeFlow === 'analyst-scout' && (
              <>
                {[0, 1, 2].map(i => (
                  <circle
                    key={i}
                    r="4"
                    fill="#22d3ee"
                    filter="url(#glow)"
                    style={{
                      offsetPath: "path('M 75% 50% Q 68% 35% 60% 50%')",
                      offsetRotate: '0deg',
                    } as any}
                    className="path-particle"
                  />
                ))}
              </>
            )}
          </g>
        </svg>

        {/* Payment Annotation Overlay */}
        {currentPayment && (
          <div className="absolute top-0 left-1/2 -translate-x-1/2 z-20 float-annotation">
            <div className="bg-green-500/20 border border-green-500/50 rounded-lg px-3 py-1.5 backdrop-blur-sm">
              <div className="flex items-center gap-2 text-green-400">
                <span className="coin-flip">💰</span>
                <span className="font-mono text-sm font-bold">{currentPayment.amount.toFixed(4)} USDC</span>
              </div>
            </div>
          </div>
        )}

        {/* Flow Label Annotations */}
        {activeFlow === 'scout-policy' && (
          <div className="absolute top-4 left-[30%] z-10">
            <span className="text-xs text-yellow-400 bg-yellow-400/10 px-2 py-1 rounded animate-pulse">
              Auth Request
            </span>
          </div>
        )}
        {activeFlow === 'policy-scout' && (
          <div className="absolute top-4 left-[30%] z-10">
            <span className="text-xs text-cyan-400 bg-cyan-400/10 px-2 py-1 rounded animate-pulse">
              Authorized + Proof
            </span>
          </div>
        )}
        {activeFlow === 'policy-analyst' && (
          <div className="absolute top-4 left-[65%] z-10">
            <span className="text-xs text-yellow-400 bg-yellow-400/10 px-2 py-1 rounded animate-pulse">
              Classify URLs
            </span>
          </div>
        )}
        {activeFlow === 'analyst-scout' && (
          <div className="absolute top-4 left-[65%] z-10">
            <span className="text-xs text-cyan-400 bg-cyan-400/10 px-2 py-1 rounded animate-pulse">
              Results + Proof
            </span>
          </div>
        )}

        {/* Scout Agent */}
        <AgentCard
          name="Scout"
          role="Explorer"
          icon={<Activity size={24} />}
          emoji="🔭"
          color="blue"
          colorHex="#3b82f6"
          state={agentStates.scout}
          proofProgress={0}
          stats={[
            { label: 'URLs Found', value: stats?.total_urls?.toLocaleString() || '0' },
            { label: 'Sources', value: '5' },
          ]}
          events={getAgentEvents('scout')}
          formatEvent={formatEventMessage}
        />

        {/* Spacer for SVG lines */}
        <div className="w-16" />

        {/* Policy Agent */}
        <AgentCard
          name="Policy"
          role="Gatekeeper"
          icon={<Shield size={24} />}
          emoji="⚖️"
          color="purple"
          colorHex="#a855f7"
          state={agentStates.policy}
          proofProgress={agentStates.policy === 'proving' ? proofProgress.policy : 0}
          stats={[
            { label: 'Proofs', value: stats?.total_proofs?.toLocaleString() || '0' },
            { label: 'Earned', value: `$${(stats?.policy_paid_usdc || 0).toFixed(3)}` },
          ]}
          events={getAgentEvents('policy')}
          formatEvent={formatEventMessage}
        />

        {/* Spacer for SVG lines */}
        <div className="w-16" />

        {/* Analyst Agent */}
        <AgentCard
          name="Analyst"
          role="Detective"
          icon={<Zap size={24} />}
          emoji="🔬"
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
        />
      </div>

      {/* Flow Legend */}
      <div className="flex justify-center gap-6 mt-6 text-xs text-gray-500">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-yellow-400 shadow-[0_0_8px_#fbbf24] status-active" />
          <span>Request/Data</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-cyan-400 shadow-[0_0_8px_#22d3ee] status-active" style={{ animationDelay: '0.3s' }} />
          <span>Response/Proof</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-green-400 shadow-[0_0_8px_#22c55e] status-active" style={{ animationDelay: '0.6s' }} />
          <span>Payment</span>
        </div>
        <div className="flex items-center gap-2 ml-4 border-l border-gray-700 pl-4">
          <span className="text-cyan-400">Network Active</span>
          <span className="flex gap-0.5">
            <span className="w-1 h-1 rounded-full bg-cyan-400 stream-dot" />
            <span className="w-1 h-1 rounded-full bg-cyan-400 stream-dot" />
            <span className="w-1 h-1 rounded-full bg-cyan-400 stream-dot" />
          </span>
        </div>
      </div>

      {/* Proof Details Section */}
      {(policyProof || analystProof || agentStates.policy === 'proving' || agentStates.analyst === 'proving') && (
        <div className="mt-6 grid grid-cols-2 gap-4">
          {/* Policy Proof Card */}
          <div className="space-y-3">
            <ProofCard
              title="Authorization Proof"
              proof={policyProof}
              isGenerating={agentStates.policy === 'proving'}
              currentStage={policyProofStage}
              color="purple"
              colorHex="#a855f7"
            />
            {policyVerifyChecks.length > 0 && (
              <VerificationChecklist
                checks={policyVerifyChecks}
                verifyTimeMs={lastEvent?.type === 'POLICY_VERIFIED' ? lastEvent.data.verify_time_ms : undefined}
              />
            )}
          </div>

          {/* Analyst Proof Card */}
          <div className="space-y-3">
            <ProofCard
              title="Classification Proof"
              proof={analystProof}
              isGenerating={agentStates.analyst === 'proving'}
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
  emoji,
  color,
  colorHex,
  state,
  proofProgress,
  stats,
  events,
  formatEvent,
}: {
  name: string;
  role: string;
  icon: React.ReactNode;
  emoji: string;
  color: string;
  colorHex: string;
  state: AgentState;
  proofProgress: number;
  stats: { label: string; value: string }[];
  events: AgentEvent[];
  formatEvent: (e: AgentEvent) => string;
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
                <span className="text-2xl">{emoji}</span>
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
                    <span className="ml-1 text-green-400">✓</span>
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
