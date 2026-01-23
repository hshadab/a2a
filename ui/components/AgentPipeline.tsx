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

  // Track proof state for each agent
  const [policyProof, setPolicyProof] = useState<ProofData | null>(null);
  const [analystProof, setAnalystProof] = useState<ProofData | null>(null);
  const [policyProofStage, setPolicyProofStage] = useState<ProofStage | null>(null);
  const [analystProofStage, setAnalystProofStage] = useState<ProofStage | null>(null);
  const [policyVerifyChecks, setPolicyVerifyChecks] = useState<VerificationCheck[]>([]);
  const [analystVerifyChecks, setAnalystVerifyChecks] = useState<VerificationCheck[]>([]);

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
        break;

      case 'POLICY_RESPONSE':
        setPolicyProofStage(null);
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
        break;

      case 'ANALYST_RESPONSE':
        setAnalystProofStage(null);
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

    if (lastEvent.type === 'POLICY_REQUESTING') return 'scout-policy';
    if (lastEvent.type === 'POLICY_RESPONSE') return 'policy-scout';
    if (lastEvent.type === 'ANALYST_PROCESSING' || lastEvent.type === 'PAYMENT_SENDING') return 'scout-analyst';
    if (lastEvent.type === 'ANALYST_RESPONSE') return 'analyst-scout';
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
      }, 1500);
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
        return `Proof verified ✓`;
      case 'ANALYST_PROCESSING':
        return `Classifying ${event.data.url_count} URLs`;
      case 'ANALYST_PROVING':
        return 'Generating zkML proof...';
      case 'ANALYST_RESPONSE':
        return `${event.data.phishing_count} phishing found`;
      case 'WORK_VERIFIED':
        return 'Work proof verified ✓';
      case 'DATABASE_UPDATED':
        return `+${event.data.urls_added} URLs saved`;
      default:
        return event.type;
    }
  };

  const getGlowClass = (state: AgentState, color: string) => {
    if (state === 'proving') return `shadow-[0_0_30px_${color},0_0_60px_${color}] animate-pulse`;
    if (state === 'working') return `shadow-[0_0_20px_${color}]`;
    if (state === 'active') return `shadow-[0_0_10px_${color}]`;
    return '';
  };

  return (
    <div className="w-full">
      {/* Pipeline Container */}
      <div className="relative flex items-stretch justify-between gap-4">

        {/* Scout Agent */}
        <AgentCard
          name="Scout"
          role="Explorer"
          icon={<Activity size={24} />}
          emoji="🔭"
          color="blue"
          colorHex="#3b82f6"
          state={agentStates.scout}
          stats={[
            { label: 'URLs Found', value: stats?.total_urls?.toLocaleString() || '0' },
            { label: 'Sources', value: '5' },
          ]}
          events={getAgentEvents('scout')}
          formatEvent={formatEventMessage}
        />

        {/* Flow Arrow: Scout → Policy */}
        <FlowArrow
          active={activeFlow === 'scout-policy' || activeFlow === 'policy-scout'}
          direction={activeFlow === 'policy-scout' ? 'left' : 'right'}
          particles={particles.filter(p =>
            (p.from === 'scout' && p.to === 'policy') ||
            (p.from === 'policy' && p.to === 'scout')
          )}
          color={activeFlow === 'policy-scout' ? '#22d3ee' : '#fbbf24'}
          label="Auth Request"
        />

        {/* Policy Agent */}
        <AgentCard
          name="Policy"
          role="Gatekeeper"
          icon={<Shield size={24} />}
          emoji="⚖️"
          color="purple"
          colorHex="#a855f7"
          state={agentStates.policy}
          stats={[
            { label: 'Proofs', value: stats?.total_proofs?.toLocaleString() || '0' },
            { label: 'Earned', value: `$${(stats?.policy_paid_usdc || 0).toFixed(3)}` },
          ]}
          events={getAgentEvents('policy')}
          formatEvent={formatEventMessage}
        />

        {/* Flow Arrow: Scout → Analyst (via Policy approval) */}
        <FlowArrow
          active={activeFlow === 'scout-analyst' || activeFlow === 'analyst-scout'}
          direction={activeFlow === 'analyst-scout' ? 'left' : 'right'}
          particles={particles.filter(p =>
            (p.from === 'scout' && p.to === 'analyst') ||
            (p.from === 'analyst' && p.to === 'scout')
          )}
          color={activeFlow === 'analyst-scout' ? '#22d3ee' : '#fbbf24'}
          label="Classify"
        />

        {/* Analyst Agent */}
        <AgentCard
          name="Analyst"
          role="Detective"
          icon={<Zap size={24} />}
          emoji="🔬"
          color="cyan"
          colorHex="#22d3ee"
          state={agentStates.analyst}
          stats={[
            { label: 'Phishing', value: stats?.phishing_count?.toLocaleString() || '0' },
            { label: 'Earned', value: `$${(stats?.analyst_paid_usdc || 0).toFixed(3)}` },
          ]}
          events={getAgentEvents('analyst')}
          formatEvent={formatEventMessage}
        />
      </div>

      {/* Flow Legend */}
      <div className="flex justify-center gap-6 mt-4 text-xs text-gray-500">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-yellow-400" />
          <span>Data/Request</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-cyan-400" />
          <span>Proof/Response</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-purple-400" />
          <span>Payment</span>
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

// Agent Card Component
function AgentCard({
  name,
  role,
  icon,
  emoji,
  color,
  colorHex,
  state,
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
  stats: { label: string; value: string }[];
  events: AgentEvent[];
  formatEvent: (e: AgentEvent) => string;
}) {
  const isActive = state !== 'idle';
  const isProving = state === 'proving';
  const isWorking = state === 'working';

  const glowStyle = useMemo(() => {
    if (isProving) return { boxShadow: `0 0 30px ${colorHex}, 0 0 60px ${colorHex}40` };
    if (isWorking) return { boxShadow: `0 0 20px ${colorHex}80` };
    if (isActive) return { boxShadow: `0 0 10px ${colorHex}40` };
    return {};
  }, [isProving, isWorking, isActive, colorHex]);

  return (
    <div
      className={`flex-1 rounded-xl border bg-gray-900/50 backdrop-blur-sm transition-all duration-300 ${
        isProving ? 'animate-pulse' : ''
      }`}
      style={{
        borderColor: isActive ? colorHex : '#374151',
        ...glowStyle,
      }}
    >
      {/* Header */}
      <div className={`p-4 border-b border-gray-800 bg-${color}-500/10`}>
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            {/* Avatar */}
            <div
              className={`w-14 h-14 rounded-full flex items-center justify-center border-2 relative ${
                isWorking || isProving ? 'animate-spin-slow' : ''
              }`}
              style={{
                borderColor: colorHex,
                backgroundColor: `${colorHex}20`,
              }}
            >
              <span className="text-2xl">{emoji}</span>
              {(isWorking || isProving) && (
                <div
                  className="absolute inset-0 rounded-full border-2 border-dashed animate-spin"
                  style={{ borderColor: colorHex }}
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
            className={`px-2 py-1 rounded-full text-xs font-medium ${
              isProving
                ? 'bg-yellow-500/20 text-yellow-400 animate-pulse'
                : isWorking
                ? 'bg-green-500/20 text-green-400'
                : isActive
                ? 'bg-blue-500/20 text-blue-400'
                : 'bg-gray-700/50 text-gray-500'
            }`}
          >
            {isProving ? '🔐 PROVING' : isWorking ? '⚡ WORKING' : isActive ? '● ACTIVE' : '○ IDLE'}
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
            <p className="text-gray-600 text-sm italic">Waiting...</p>
          ) : (
            events.map((event, i) => (
              <div
                key={`${event.timestamp}-${i}`}
                className={`text-sm flex items-start gap-2 ${i === 0 ? 'text-white' : 'text-gray-500'}`}
              >
                <span className="text-gray-600 text-xs mt-0.5">
                  {new Date(event.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </span>
                <span className={i === 0 ? 'animate-pulse' : ''}>{formatEvent(event)}</span>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}

// Flow Arrow Component
function FlowArrow({
  active,
  direction,
  particles,
  color,
  label,
}: {
  active: boolean;
  direction: 'left' | 'right';
  particles: { id: number; from: string; to: string }[];
  color: string;
  label: string;
}) {
  return (
    <div className="flex flex-col items-center justify-center w-20 relative">
      {/* Label */}
      <span className="text-xs text-gray-600 mb-2">{label}</span>

      {/* Arrow Line */}
      <div className="relative w-full h-8 flex items-center">
        <div
          className={`h-0.5 w-full transition-all duration-300 ${
            active ? '' : 'opacity-30'
          }`}
          style={{
            background: active
              ? `linear-gradient(${direction === 'right' ? '90deg' : '270deg'}, transparent, ${color}, transparent)`
              : '#374151',
          }}
        />

        {/* Arrow Head */}
        <ArrowRight
          size={16}
          className={`absolute transition-all duration-300 ${
            direction === 'left' ? 'left-0 rotate-180' : 'right-0'
          }`}
          style={{
            color: active ? color : '#374151',
            filter: active ? `drop-shadow(0 0 4px ${color})` : 'none',
          }}
        />

        {/* Particles */}
        {particles.map((p) => (
          <div
            key={p.id}
            className="absolute w-2 h-2 rounded-full animate-flow-particle"
            style={{
              backgroundColor: color,
              boxShadow: `0 0 8px ${color}`,
              animationDirection: direction === 'left' ? 'reverse' : 'normal',
            }}
          />
        ))}
      </div>
    </div>
  );
}
