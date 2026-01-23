'use client';

import { useEffect, useState, useMemo } from 'react';
import { DollarSign } from 'lucide-react';

type AgentState = 'idle' | 'active' | 'working' | 'proving' | 'sending' | 'receiving';
type ConnectionState = 'idle' | 'active' | 'data' | 'payment' | 'proof';

interface AgentTriangleProps {
  scoutState: AgentState;
  policyState: AgentState;
  analystState: AgentState;
  activeConnection?: 'scout-policy' | 'policy-scout' | 'scout-analyst' | 'analyst-scout' | 'policy-analyst' | null;
  connectionType?: ConnectionState;
  lastEvent?: { type: string; data: Record<string, any> } | null;
  stats?: {
    policyPaid: number;
    analystPaid: number;
    totalUrls: number;
  };
}

export default function AgentTriangle({
  scoutState,
  policyState,
  analystState,
  activeConnection,
  connectionType = 'idle',
  lastEvent,
  stats,
}: AgentTriangleProps) {
  const [particles, setParticles] = useState<{ id: number; connection: string; type: string }[]>([]);
  const [pulsingAgent, setPulsingAgent] = useState<string | null>(null);

  // Determine states from last event
  const derivedStates = useMemo(() => {
    if (!lastEvent) return { scout: scoutState, policy: policyState, analyst: analystState, conn: null, connType: 'idle' as ConnectionState };

    let scout = scoutState;
    let policy = policyState;
    let analyst = analystState;
    let conn: typeof activeConnection = null;
    let connType: ConnectionState = 'idle';

    switch (lastEvent.type) {
      case 'SCOUT_FOUND_URLS':
        scout = 'active';
        break;
      case 'POLICY_REQUESTING':
        scout = 'sending';
        policy = 'receiving';
        conn = 'scout-policy';
        connType = 'data';
        break;
      case 'POLICY_PROVING':
        policy = 'proving';
        break;
      case 'POLICY_RESPONSE':
        policy = 'sending';
        scout = 'receiving';
        conn = 'policy-scout';
        connType = 'proof';
        break;
      case 'PAYMENT_SENDING':
        conn = lastEvent.data.recipient?.includes('policy') ? 'scout-policy' : 'scout-analyst';
        connType = 'payment';
        break;
      case 'ANALYST_PROCESSING':
        analyst = 'working';
        break;
      case 'ANALYST_PROVING':
        analyst = 'proving';
        break;
      case 'ANALYST_RESPONSE':
        analyst = 'sending';
        scout = 'receiving';
        conn = 'analyst-scout';
        connType = 'proof';
        break;
      case 'DATABASE_UPDATED':
        scout = 'active';
        break;
    }

    return { scout, policy, analyst, conn, connType };
  }, [lastEvent, scoutState, policyState, analystState]);

  // Spawn particles when connection is active
  useEffect(() => {
    if (derivedStates.conn && derivedStates.connType !== 'idle') {
      const id = Date.now();
      setParticles(prev => [...prev, { id, connection: derivedStates.conn!, type: derivedStates.connType }]);

      // Clean up old particles
      setTimeout(() => {
        setParticles(prev => prev.filter(p => p.id !== id));
      }, 2000);
    }
  }, [derivedStates.conn, derivedStates.connType, lastEvent?.type]);

  // Pulse agent on activity
  useEffect(() => {
    if (lastEvent?.type) {
      if (lastEvent.type.includes('SCOUT')) setPulsingAgent('scout');
      else if (lastEvent.type.includes('POLICY')) setPulsingAgent('policy');
      else if (lastEvent.type.includes('ANALYST')) setPulsingAgent('analyst');

      const timeout = setTimeout(() => setPulsingAgent(null), 1000);
      return () => clearTimeout(timeout);
    }
  }, [lastEvent]);

  // Get eye direction based on connection
  const getEyeDirection = (agent: string) => {
    if (!derivedStates.conn) return 'center';

    if (agent === 'scout') {
      if (derivedStates.conn.includes('policy')) return 'left';
      if (derivedStates.conn.includes('analyst')) return 'right';
    }
    if (agent === 'policy') {
      if (derivedStates.conn.includes('scout')) return 'up';
      if (derivedStates.conn.includes('analyst')) return 'right';
    }
    if (agent === 'analyst') {
      if (derivedStates.conn.includes('scout')) return 'up';
      if (derivedStates.conn.includes('policy')) return 'left';
    }
    return 'center';
  };

  const getConnectionColor = (type: ConnectionState) => {
    switch (type) {
      case 'data': return '#fbbf24'; // yellow
      case 'payment': return '#a855f7'; // purple
      case 'proof': return '#22d3ee'; // cyan
      default: return '#374151'; // gray
    }
  };

  return (
    <div className="agent-triangle bg-gray-900/30 rounded-xl border border-gray-800 p-6">
      {/* SVG for connection lines */}
      <svg className="absolute inset-0 w-full h-full pointer-events-none" style={{ zIndex: 0 }}>
        <defs>
          <linearGradient id="gradient-yellow" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#fbbf24" stopOpacity="0" />
            <stop offset="50%" stopColor="#fbbf24" stopOpacity="1" />
            <stop offset="100%" stopColor="#fbbf24" stopOpacity="0" />
          </linearGradient>
          <linearGradient id="gradient-purple" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#a855f7" stopOpacity="0" />
            <stop offset="50%" stopColor="#a855f7" stopOpacity="1" />
            <stop offset="100%" stopColor="#a855f7" stopOpacity="0" />
          </linearGradient>
          <linearGradient id="gradient-cyan" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#22d3ee" stopOpacity="0" />
            <stop offset="50%" stopColor="#22d3ee" stopOpacity="1" />
            <stop offset="100%" stopColor="#22d3ee" stopOpacity="0" />
          </linearGradient>
        </defs>

        {/* Scout to Policy line */}
        <path
          d="M 200 120 Q 120 200 100 280"
          className={`connection-path ${derivedStates.conn?.includes('scout') && derivedStates.conn?.includes('policy') ? 'active' : 'idle'}`}
          style={{
            stroke: derivedStates.conn?.includes('scout') && derivedStates.conn?.includes('policy')
              ? getConnectionColor(derivedStates.connType)
              : '#374151',
            strokeDasharray: derivedStates.conn?.includes('scout') && derivedStates.conn?.includes('policy') ? 'none' : '8 4',
          }}
        />

        {/* Scout to Analyst line */}
        <path
          d="M 280 120 Q 360 200 380 280"
          className={`connection-path ${derivedStates.conn?.includes('scout') && derivedStates.conn?.includes('analyst') ? 'active' : 'idle'}`}
          style={{
            stroke: derivedStates.conn?.includes('scout') && derivedStates.conn?.includes('analyst')
              ? getConnectionColor(derivedStates.connType)
              : '#374151',
            strokeDasharray: derivedStates.conn?.includes('scout') && derivedStates.conn?.includes('analyst') ? 'none' : '8 4',
          }}
        />

        {/* Policy to Analyst line */}
        <path
          d="M 140 320 L 340 320"
          className={`connection-path ${derivedStates.conn?.includes('policy') && derivedStates.conn?.includes('analyst') ? 'active' : 'idle'}`}
          style={{
            stroke: derivedStates.conn?.includes('policy') && derivedStates.conn?.includes('analyst')
              ? getConnectionColor(derivedStates.connType)
              : '#374151',
            strokeDasharray: derivedStates.conn?.includes('policy') && derivedStates.conn?.includes('analyst') ? 'none' : '8 4',
          }}
        />

        {/* Animated particles */}
        {particles.map(particle => (
          <circle
            key={particle.id}
            r="5"
            fill={getConnectionColor(particle.type as ConnectionState)}
            style={{
              filter: `drop-shadow(0 0 6px ${getConnectionColor(particle.type as ConnectionState)})`,
            }}
          >
            <animateMotion
              dur="1.5s"
              repeatCount="1"
              path={
                particle.connection === 'scout-policy' ? "M 200 120 Q 120 200 100 280" :
                particle.connection === 'policy-scout' ? "M 100 280 Q 120 200 200 120" :
                particle.connection === 'scout-analyst' ? "M 280 120 Q 360 200 380 280" :
                particle.connection === 'analyst-scout' ? "M 380 280 Q 360 200 280 120" :
                "M 140 320 L 340 320"
              }
            />
          </circle>
        ))}
      </svg>

      {/* Scout Agent - Top */}
      <div
        className={`agent-node ${derivedStates.scout !== 'idle' ? 'active' : ''}`}
        style={{ top: '20px', left: '50%', transform: 'translateX(-50%)', color: '#3b82f6' }}
      >
        {pulsingAgent === 'scout' && <div className="pulse-ring" />}
        <div className={`agent-avatar bg-blue-500/20 border-2 border-blue-500 ${derivedStates.scout === 'proving' ? 'proving' : ''} ${derivedStates.scout === 'working' ? 'working' : ''}`}>
          <div className="flex flex-col items-center">
            <AgentFace
              expression={derivedStates.scout === 'working' ? 'focused' : derivedStates.scout === 'active' ? 'happy' : 'neutral'}
              eyeDirection={getEyeDirection('scout')}
              color="#3b82f6"
            />
            <span className="text-2xl mt-1">🔭</span>
          </div>
        </div>
        <span className="mt-2 font-semibold text-blue-400">Scout</span>
        <span className="text-xs text-gray-500">Explorer</span>
        {derivedStates.scout === 'working' && (
          <span className="text-xs text-yellow-400 mt-1 animate-pulse">Discovering...</span>
        )}
      </div>

      {/* Policy Agent - Bottom Left */}
      <div
        className={`agent-node ${derivedStates.policy !== 'idle' ? 'active' : ''}`}
        style={{ bottom: '60px', left: '60px', color: '#a855f7' }}
      >
        {pulsingAgent === 'policy' && <div className="pulse-ring" />}
        <div className={`agent-avatar bg-purple-500/20 border-2 border-purple-500 ${derivedStates.policy === 'proving' ? 'proving' : ''} ${derivedStates.policy === 'working' ? 'working' : ''}`}>
          <div className="flex flex-col items-center">
            <AgentFace
              expression={derivedStates.policy === 'proving' ? 'thinking' : derivedStates.policy === 'active' ? 'stern' : 'neutral'}
              eyeDirection={getEyeDirection('policy')}
              color="#a855f7"
            />
            <span className="text-2xl mt-1">⚖️</span>
          </div>
        </div>
        <span className="mt-2 font-semibold text-purple-400">Policy</span>
        <span className="text-xs text-gray-500">Gatekeeper</span>
        {derivedStates.policy === 'proving' && (
          <span className="text-xs text-cyan-400 mt-1 animate-pulse">Generating proof...</span>
        )}
        {stats && (
          <span className="text-xs text-purple-300 mt-1">${stats.policyPaid.toFixed(3)} earned</span>
        )}
      </div>

      {/* Analyst Agent - Bottom Right */}
      <div
        className={`agent-node ${derivedStates.analyst !== 'idle' ? 'active' : ''}`}
        style={{ bottom: '60px', right: '60px', color: '#22d3ee' }}
      >
        {pulsingAgent === 'analyst' && <div className="pulse-ring" />}
        <div className={`agent-avatar bg-cyan-500/20 border-2 border-cyan-500 ${derivedStates.analyst === 'proving' ? 'proving' : ''} ${derivedStates.analyst === 'working' ? 'working' : ''}`}>
          <div className="flex flex-col items-center">
            <AgentFace
              expression={derivedStates.analyst === 'working' ? 'focused' : derivedStates.analyst === 'proving' ? 'thinking' : 'neutral'}
              eyeDirection={getEyeDirection('analyst')}
              color="#22d3ee"
            />
            <span className="text-2xl mt-1">🔬</span>
          </div>
        </div>
        <span className="mt-2 font-semibold text-cyan-400">Analyst</span>
        <span className="text-xs text-gray-500">Detective</span>
        {derivedStates.analyst === 'working' && (
          <span className="text-xs text-yellow-400 mt-1 animate-pulse">Classifying...</span>
        )}
        {derivedStates.analyst === 'proving' && (
          <span className="text-xs text-cyan-400 mt-1 animate-pulse">Generating proof...</span>
        )}
        {stats && (
          <span className="text-xs text-cyan-300 mt-1">${stats.analystPaid.toFixed(3)} earned</span>
        )}
      </div>

      {/* Treasury - Center Bottom */}
      <div className="treasury-node flex flex-col items-center" style={{ bottom: '10px' }}>
        <div className="w-12 h-12 rounded-full bg-yellow-500/20 border-2 border-yellow-500 flex items-center justify-center">
          <DollarSign className="text-yellow-400" size={24} />
        </div>
        <span className="text-xs text-yellow-400 mt-1">Treasury</span>
      </div>

      {/* Legend */}
      <div className="absolute bottom-4 right-4 flex flex-col gap-1 text-xs">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-yellow-400" />
          <span className="text-gray-500">Data</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-purple-400" />
          <span className="text-gray-500">Payment</span>
        </div>
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full bg-cyan-400" />
          <span className="text-gray-500">Proof</span>
        </div>
      </div>
    </div>
  );
}

// Simple face component for agent expressions
function AgentFace({
  expression,
  eyeDirection,
  color
}: {
  expression: 'neutral' | 'happy' | 'focused' | 'thinking' | 'stern';
  eyeDirection: string;
  color: string;
}) {
  const getEyeStyle = () => {
    const base = { width: 8, height: 8, background: color, borderRadius: '50%' };
    return base;
  };

  const getPupilOffset = () => {
    switch (eyeDirection) {
      case 'left': return { left: 0 };
      case 'right': return { right: 0 };
      case 'up': return { top: 0 };
      case 'down': return { bottom: 0 };
      default: return { left: 2, top: 2 };
    }
  };

  const getMouth = () => {
    switch (expression) {
      case 'happy':
        return <div className="w-4 h-2 border-b-2 rounded-b-full" style={{ borderColor: color }} />;
      case 'focused':
        return <div className="w-3 h-0.5" style={{ background: color }} />;
      case 'thinking':
        return <div className="w-2 h-2 rounded-full border-2" style={{ borderColor: color }} />;
      case 'stern':
        return <div className="w-4 h-0.5" style={{ background: color }} />;
      default:
        return <div className="w-3 h-0.5 rounded" style={{ background: color, opacity: 0.5 }} />;
    }
  };

  return (
    <div className="flex flex-col items-center gap-1">
      <div className="flex gap-3">
        <div className="relative" style={getEyeStyle()}>
          <div
            className="absolute w-2 h-2 bg-gray-900 rounded-full"
            style={getPupilOffset()}
          />
        </div>
        <div className="relative" style={getEyeStyle()}>
          <div
            className="absolute w-2 h-2 bg-gray-900 rounded-full"
            style={getPupilOffset()}
          />
        </div>
      </div>
      {getMouth()}
    </div>
  );
}
