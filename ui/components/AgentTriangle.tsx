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

  // Fixed dimensions for consistent layout
  const WIDTH = 480;
  const HEIGHT = 380;

  // Agent positions (fixed coordinates)
  const positions = {
    scout: { x: WIDTH / 2, y: 60 },
    policy: { x: 100, y: 280 },
    analyst: { x: WIDTH - 100, y: 280 },
    treasury: { x: WIDTH / 2, y: 340 },
  };

  return (
    <div className="flex justify-center">
      <div
        className="bg-gray-900/30 rounded-xl border border-gray-800 p-4 relative"
        style={{ width: WIDTH, height: HEIGHT }}
      >
        {/* SVG for connection lines */}
        <svg
          className="absolute inset-0 pointer-events-none"
          width={WIDTH}
          height={HEIGHT}
          style={{ zIndex: 0 }}
        >
          {/* Scout to Policy line */}
          <line
            x1={positions.scout.x}
            y1={positions.scout.y + 50}
            x2={positions.policy.x}
            y2={positions.policy.y - 50}
            className="connection-path"
            style={{
              stroke: derivedStates.conn?.includes('scout') && derivedStates.conn?.includes('policy')
                ? getConnectionColor(derivedStates.connType)
                : '#374151',
              strokeWidth: derivedStates.conn?.includes('scout') && derivedStates.conn?.includes('policy') ? 3 : 2,
              strokeDasharray: derivedStates.conn?.includes('scout') && derivedStates.conn?.includes('policy') ? 'none' : '8 4',
            }}
          />

          {/* Scout to Analyst line */}
          <line
            x1={positions.scout.x}
            y1={positions.scout.y + 50}
            x2={positions.analyst.x}
            y2={positions.analyst.y - 50}
            className="connection-path"
            style={{
              stroke: derivedStates.conn?.includes('scout') && derivedStates.conn?.includes('analyst')
                ? getConnectionColor(derivedStates.connType)
                : '#374151',
              strokeWidth: derivedStates.conn?.includes('scout') && derivedStates.conn?.includes('analyst') ? 3 : 2,
              strokeDasharray: derivedStates.conn?.includes('scout') && derivedStates.conn?.includes('analyst') ? 'none' : '8 4',
            }}
          />

          {/* Policy to Analyst line */}
          <line
            x1={positions.policy.x + 50}
            y1={positions.policy.y}
            x2={positions.analyst.x - 50}
            y2={positions.analyst.y}
            className="connection-path"
            style={{
              stroke: derivedStates.conn?.includes('policy') && derivedStates.conn?.includes('analyst')
                ? getConnectionColor(derivedStates.connType)
                : '#374151',
              strokeWidth: derivedStates.conn?.includes('policy') && derivedStates.conn?.includes('analyst') ? 3 : 2,
              strokeDasharray: derivedStates.conn?.includes('policy') && derivedStates.conn?.includes('analyst') ? 'none' : '8 4',
            }}
          />

          {/* Animated particles */}
          {particles.map(particle => {
            const getPath = () => {
              switch (particle.connection) {
                case 'scout-policy':
                  return `M ${positions.scout.x} ${positions.scout.y + 50} L ${positions.policy.x} ${positions.policy.y - 50}`;
                case 'policy-scout':
                  return `M ${positions.policy.x} ${positions.policy.y - 50} L ${positions.scout.x} ${positions.scout.y + 50}`;
                case 'scout-analyst':
                  return `M ${positions.scout.x} ${positions.scout.y + 50} L ${positions.analyst.x} ${positions.analyst.y - 50}`;
                case 'analyst-scout':
                  return `M ${positions.analyst.x} ${positions.analyst.y - 50} L ${positions.scout.x} ${positions.scout.y + 50}`;
                default:
                  return `M ${positions.policy.x + 50} ${positions.policy.y} L ${positions.analyst.x - 50} ${positions.analyst.y}`;
              }
            };
            return (
              <circle
                key={particle.id}
                r="6"
                fill={getConnectionColor(particle.type as ConnectionState)}
                style={{
                  filter: `drop-shadow(0 0 8px ${getConnectionColor(particle.type as ConnectionState)})`,
                }}
              >
                <animateMotion dur="1s" repeatCount="1" path={getPath()} />
              </circle>
            );
          })}
        </svg>

        {/* Scout Agent - Top */}
        <div
          className={`absolute flex flex-col items-center ${derivedStates.scout !== 'idle' ? 'active' : ''}`}
          style={{
            left: positions.scout.x,
            top: positions.scout.y,
            transform: 'translate(-50%, -50%)',
            color: '#3b82f6'
          }}
        >
          {pulsingAgent === 'scout' && <div className="pulse-ring" style={{ width: 80, height: 80 }} />}
          <div className={`w-20 h-20 rounded-full bg-blue-500/20 border-2 border-blue-500 flex items-center justify-center ${derivedStates.scout === 'proving' ? 'proving' : ''} ${derivedStates.scout === 'working' ? 'working' : ''}`}>
            <div className="flex flex-col items-center">
              <AgentFace
                expression={derivedStates.scout === 'working' ? 'focused' : derivedStates.scout === 'active' ? 'happy' : 'neutral'}
                eyeDirection={getEyeDirection('scout')}
                color="#3b82f6"
              />
              <span className="text-lg">🔭</span>
            </div>
          </div>
          <span className="mt-1 text-sm font-semibold text-blue-400">Scout</span>
          <span className="text-xs text-gray-500">Explorer</span>
        </div>

        {/* Policy Agent - Bottom Left */}
        <div
          className={`absolute flex flex-col items-center ${derivedStates.policy !== 'idle' ? 'active' : ''}`}
          style={{
            left: positions.policy.x,
            top: positions.policy.y,
            transform: 'translate(-50%, -50%)',
            color: '#a855f7'
          }}
        >
          {pulsingAgent === 'policy' && <div className="pulse-ring" style={{ width: 80, height: 80 }} />}
          <div className={`w-20 h-20 rounded-full bg-purple-500/20 border-2 border-purple-500 flex items-center justify-center ${derivedStates.policy === 'proving' ? 'proving' : ''} ${derivedStates.policy === 'working' ? 'working' : ''}`}>
            <div className="flex flex-col items-center">
              <AgentFace
                expression={derivedStates.policy === 'proving' ? 'thinking' : derivedStates.policy === 'active' ? 'stern' : 'neutral'}
                eyeDirection={getEyeDirection('policy')}
                color="#a855f7"
              />
              <span className="text-lg">⚖️</span>
            </div>
          </div>
          <span className="mt-1 text-sm font-semibold text-purple-400">Policy</span>
          <span className="text-xs text-gray-500">Gatekeeper</span>
          {stats && (
            <span className="text-xs text-purple-300">${stats.policyPaid.toFixed(3)}</span>
          )}
        </div>

        {/* Analyst Agent - Bottom Right */}
        <div
          className={`absolute flex flex-col items-center ${derivedStates.analyst !== 'idle' ? 'active' : ''}`}
          style={{
            left: positions.analyst.x,
            top: positions.analyst.y,
            transform: 'translate(-50%, -50%)',
            color: '#22d3ee'
          }}
        >
          {pulsingAgent === 'analyst' && <div className="pulse-ring" style={{ width: 80, height: 80 }} />}
          <div className={`w-20 h-20 rounded-full bg-cyan-500/20 border-2 border-cyan-500 flex items-center justify-center ${derivedStates.analyst === 'proving' ? 'proving' : ''} ${derivedStates.analyst === 'working' ? 'working' : ''}`}>
            <div className="flex flex-col items-center">
              <AgentFace
                expression={derivedStates.analyst === 'working' ? 'focused' : derivedStates.analyst === 'proving' ? 'thinking' : 'neutral'}
                eyeDirection={getEyeDirection('analyst')}
                color="#22d3ee"
              />
              <span className="text-lg">🔬</span>
            </div>
          </div>
          <span className="mt-1 text-sm font-semibold text-cyan-400">Analyst</span>
          <span className="text-xs text-gray-500">Detective</span>
          {stats && (
            <span className="text-xs text-cyan-300">${stats.analystPaid.toFixed(3)}</span>
          )}
        </div>

        {/* Treasury - Center Bottom */}
        <div
          className="absolute flex flex-col items-center"
          style={{
            left: positions.treasury.x,
            top: positions.treasury.y,
            transform: 'translate(-50%, -50%)'
          }}
        >
          <div className="w-10 h-10 rounded-full bg-yellow-500/20 border-2 border-yellow-500 flex items-center justify-center">
            <DollarSign className="text-yellow-400" size={20} />
          </div>
          <span className="text-xs text-yellow-400 mt-1">Treasury</span>
        </div>

        {/* Legend */}
        <div className="absolute bottom-3 right-3 flex gap-4 text-xs">
          <div className="flex items-center gap-1">
            <div className="w-2 h-2 rounded-full bg-yellow-400" />
            <span className="text-gray-500">Data</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-2 h-2 rounded-full bg-purple-400" />
            <span className="text-gray-500">Payment</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-2 h-2 rounded-full bg-cyan-400" />
            <span className="text-gray-500">Proof</span>
          </div>
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
