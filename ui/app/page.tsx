'use client';

import { useState, useEffect } from 'react';
import { useWebSocket } from '@/hooks/useWebSocket';
import {
  getNetworkStats,
  getHealthStatus,
  triggerBatch,
  formatUSDC,
  formatNumber,
  formatDuration,
  getEventIcon,
  getEventColor,
  NetworkStats,
  HealthStatus,
} from '@/lib/api';
import {
  Activity,
  Shield,
  Zap,
  Database,
  DollarSign,
  AlertTriangle,
  CheckCircle,
  Clock,
  RefreshCw,
} from 'lucide-react';
import AgentTriangle from '@/components/AgentTriangle';

const WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000/ws';

export default function Dashboard() {
  const { events, isConnected, lastEvent } = useWebSocket(WS_URL);
  const [stats, setStats] = useState<NetworkStats | null>(null);
  const [health, setHealth] = useState<HealthStatus | null>(null);
  const [loading, setLoading] = useState(true);

  // Fetch initial stats
  useEffect(() => {
    async function fetchData() {
      try {
        const [statsData, healthData] = await Promise.all([
          getNetworkStats(),
          getHealthStatus(),
        ]);
        setStats(statsData);
        setHealth(healthData);
      } catch (e) {
        console.error('Failed to fetch data:', e);
      } finally {
        setLoading(false);
      }
    }

    fetchData();
    const interval = setInterval(fetchData, 10000); // Refresh every 10s
    return () => clearInterval(interval);
  }, []);

  // Update stats on database update events
  useEffect(() => {
    if (lastEvent?.type === 'DATABASE_UPDATED') {
      setStats((prev) =>
        prev
          ? {
              ...prev,
              total_urls: lastEvent.data.total_urls,
              phishing_count: lastEvent.data.total_phishing,
              safe_count: lastEvent.data.total_safe,
              suspicious_count: lastEvent.data.total_suspicious,
            }
          : prev
      );
    }
  }, [lastEvent]);

  const handleTrigger = async () => {
    try {
      await triggerBatch();
    } catch (e) {
      console.error('Failed to trigger batch:', e);
    }
  };

  return (
    <div className="min-h-screen p-6">
      {/* Header */}
      <header className="mb-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-white flex items-center gap-3">
              <Shield className="text-cyber-blue" />
              Threat Intelligence Network
            </h1>
            <p className="text-gray-400 mt-1">
              A2A + x402 + Jolt Atlas zkML
            </p>
          </div>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <div
                className={`w-3 h-3 rounded-full ${
                  isConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'
                }`}
              />
              <span className="text-sm text-gray-400">
                {isConnected ? 'Live' : 'Disconnected'}
              </span>
            </div>
            {stats?.running_since && (
              <div className="text-sm text-gray-400">
                Running: {formatDuration(stats.running_since)}
              </div>
            )}
            <button
              onClick={handleTrigger}
              className="px-4 py-2 bg-cyber-blue/20 border border-cyber-blue text-cyber-blue rounded-lg hover:bg-cyber-blue/30 transition-colors flex items-center gap-2"
            >
              <RefreshCw size={16} />
              Trigger Batch
            </button>
          </div>
        </div>
      </header>

      {/* Agent Triangle Visualization */}
      <section className="mb-8">
        <AgentTriangle
          scoutState={health?.running ? 'active' : 'idle'}
          policyState={lastEvent?.type.includes('POLICY') ? (lastEvent?.type === 'POLICY_PROVING' ? 'proving' : 'working') : 'active'}
          analystState={lastEvent?.type.includes('ANALYST') ? (lastEvent?.type === 'ANALYST_PROVING' ? 'proving' : 'working') : 'active'}
          lastEvent={lastEvent}
          stats={stats ? {
            policyPaid: stats.policy_paid_usdc,
            analystPaid: stats.analyst_paid_usdc,
            totalUrls: stats.total_urls,
          } : undefined}
        />
      </section>

      {/* Agent Details */}
      <section className="grid grid-cols-3 gap-4 mb-8">
        <AgentCard
          name="Scout"
          status={health?.running ? 'active' : 'idle'}
          icon={<Activity className="text-blue-400" />}
          stats={[
            { label: 'Sources', value: '5' },
            { label: 'URLs/batch', value: '50' },
          ]}
          color="blue"
        />
        <AgentCard
          name="Policy"
          status={lastEvent?.type.includes('POLICY') ? 'working' : 'active'}
          icon={<Shield className="text-purple-400" />}
          stats={[
            { label: 'Model', value: 'authz.onnx' },
            { label: 'Price', value: '$0.001' },
          ]}
          color="purple"
          isProving={lastEvent?.type === 'POLICY_PROVING'}
        />
        <AgentCard
          name="Analyst"
          status={lastEvent?.type.includes('ANALYST') ? 'working' : 'active'}
          icon={<Zap className="text-cyan-400" />}
          stats={[
            { label: 'Model', value: 'classify.onnx' },
            { label: 'Price', value: '$0.0005/URL' },
          ]}
          color="cyan"
          isProving={lastEvent?.type === 'ANALYST_PROVING'}
        />
      </section>

      {/* Stats Grid */}
      <section className="grid grid-cols-4 gap-4 mb-8">
        <StatCard
          label="Total URLs"
          value={stats ? formatNumber(stats.total_urls) : '-'}
          icon={<Database className="text-cyber-blue" />}
          change={lastEvent?.type === 'DATABASE_UPDATED' ? `+${lastEvent.data.urls_added}` : undefined}
        />
        <StatCard
          label="Phishing Detected"
          value={stats ? formatNumber(stats.phishing_count) : '-'}
          icon={<AlertTriangle className="text-cyber-red" />}
          subtext={stats ? `${((stats.phishing_count / Math.max(1, stats.total_urls)) * 100).toFixed(1)}%` : '-'}
        />
        <StatCard
          label="Safe URLs"
          value={stats ? formatNumber(stats.safe_count) : '-'}
          icon={<CheckCircle className="text-cyber-green" />}
        />
        <StatCard
          label="Total Proofs"
          value={stats ? formatNumber(stats.total_proofs) : '-'}
          icon={<Shield className="text-cyber-purple" />}
        />
      </section>

      {/* Economics */}
      <section className="grid grid-cols-4 gap-4 mb-8">
        <StatCard
          label="Total Spent"
          value={stats ? formatUSDC(stats.total_spent_usdc) : '-'}
          icon={<DollarSign className="text-yellow-400" />}
        />
        <StatCard
          label="Policy Paid"
          value={stats ? formatUSDC(stats.policy_paid_usdc) : '-'}
          subtext="Authorization proofs"
        />
        <StatCard
          label="Analyst Paid"
          value={stats ? formatUSDC(stats.analyst_paid_usdc) : '-'}
          subtext="Classification proofs"
        />
        <StatCard
          label="Cost per URL"
          value={stats && stats.total_urls > 0 ? formatUSDC(stats.total_spent_usdc / stats.total_urls) : '-'}
          subtext="Average"
        />
      </section>

      {/* Live Feed */}
      <section className="card p-4">
        <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <Activity className="text-cyber-blue" />
          Live Feed
        </h2>
        <div className="space-y-2 max-h-96 overflow-y-auto">
          {events.length === 0 ? (
            <p className="text-gray-500 text-center py-8">
              Waiting for events...
            </p>
          ) : (
            events.map((event, i) => (
              <EventRow key={`${event.timestamp}-${i}`} event={event} />
            ))
          )}
        </div>
      </section>

      {/* Why zkML Section */}
      <section className="mt-8 card p-6">
        <h2 className="text-lg font-semibold mb-4">Why zkML is "Must Have"</h2>
        <div className="grid grid-cols-2 gap-6">
          <div className="space-y-3">
            <h3 className="text-cyber-red font-medium">Without Proofs</h3>
            <ul className="space-y-2 text-sm text-gray-400">
              <li className="flex items-center gap-2">
                <span className="text-red-500">✗</span>
                Policy Agent could approve everything
              </li>
              <li className="flex items-center gap-2">
                <span className="text-red-500">✗</span>
                Analyst could return fake classifications
              </li>
              <li className="flex items-center gap-2">
                <span className="text-red-500">✗</span>
                No way to verify work was done
              </li>
              <li className="flex items-center gap-2">
                <span className="text-red-500">✗</span>
                Database could be full of garbage
              </li>
            </ul>
          </div>
          <div className="space-y-3">
            <h3 className="text-cyber-green font-medium">With zkML Proofs</h3>
            <ul className="space-y-2 text-sm text-gray-400">
              <li className="flex items-center gap-2">
                <span className="text-green-500">✓</span>
                Every authorization is cryptographically verified
              </li>
              <li className="flex items-center gap-2">
                <span className="text-green-500">✓</span>
                Every classification is provably correct
              </li>
              <li className="flex items-center gap-2">
                <span className="text-green-500">✓</span>
                Model commitment binds exact computation
              </li>
              <li className="flex items-center gap-2">
                <span className="text-green-500">✓</span>
                Trustless agent collaboration
              </li>
            </ul>
          </div>
        </div>
      </section>
    </div>
  );
}

// ============ Components ============

function AgentCard({
  name,
  status,
  icon,
  stats,
  color,
  isProving,
}: {
  name: string;
  status: 'idle' | 'active' | 'working';
  icon: React.ReactNode;
  stats: { label: string; value: string }[];
  color: string;
  isProving?: boolean;
}) {
  const statusColors = {
    idle: 'bg-gray-500',
    active: 'bg-green-500',
    working: 'bg-yellow-500 animate-pulse',
  };

  return (
    <div className={`card p-4 border-${color}-500/30`}>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          {icon}
          <span className="font-semibold">{name}</span>
        </div>
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${statusColors[status]}`} />
          <span className="text-sm text-gray-400 capitalize">{status}</span>
        </div>
      </div>
      {isProving && (
        <div className="mb-3">
          <div className="flex items-center gap-2 text-sm text-yellow-400 mb-1">
            <Shield size={14} />
            Generating zkML proof...
          </div>
          <div className="h-1 bg-gray-800 rounded-full overflow-hidden">
            <div className="h-full bg-yellow-400 animate-pulse w-2/3" />
          </div>
        </div>
      )}
      <div className="space-y-1">
        {stats.map((stat) => (
          <div key={stat.label} className="flex justify-between text-sm">
            <span className="text-gray-500">{stat.label}</span>
            <span className="text-gray-300">{stat.value}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function StatCard({
  label,
  value,
  icon,
  change,
  subtext,
}: {
  label: string;
  value: string;
  icon?: React.ReactNode;
  change?: string;
  subtext?: string;
}) {
  return (
    <div className="card p-4">
      <div className="flex items-center justify-between mb-2">
        <span className="stat-label">{label}</span>
        {icon}
      </div>
      <div className="flex items-baseline gap-2">
        <span className="stat-value text-white">{value}</span>
        {change && (
          <span className="text-sm text-cyber-green animate-slide-in">
            {change}
          </span>
        )}
      </div>
      {subtext && <p className="text-xs text-gray-500 mt-1">{subtext}</p>}
    </div>
  );
}

function EventRow({ event }: { event: { type: string; timestamp: string; data: Record<string, any> } }) {
  const time = new Date(event.timestamp).toLocaleTimeString();
  const icon = getEventIcon(event.type);
  const color = getEventColor(event.type);

  const getMessage = () => {
    switch (event.type) {
      case 'SCOUT_FOUND_URLS':
        return `Found ${event.data.url_count} URLs from ${event.data.source}`;
      case 'POLICY_REQUESTING':
        return `Requesting authorization for ${event.data.url_count} URLs`;
      case 'POLICY_PROVING':
        return 'Generating policy proof...';
      case 'POLICY_RESPONSE':
        return `${event.data.decision} (${(event.data.confidence * 100).toFixed(0)}% confidence)`;
      case 'POLICY_VERIFIED':
        return `Policy proof verified in ${event.data.verify_time_ms}ms`;
      case 'PAYMENT_SENDING':
        return `Sending ${event.data.amount_usdc} USDC`;
      case 'PAYMENT_SENT':
        return `Paid ${event.data.amount_usdc} USDC (tx: ${event.data.tx_hash?.slice(0, 10)}...)`;
      case 'ANALYST_PROCESSING':
        return `Processing ${event.data.url_count} URLs${event.data.progress ? ` (${event.data.progress}/${event.data.url_count})` : ''}`;
      case 'ANALYST_PROVING':
        return 'Generating classification proof...';
      case 'ANALYST_RESPONSE':
        return `Classified: ${event.data.phishing_count} phishing, ${event.data.safe_count} safe, ${event.data.suspicious_count} suspicious`;
      case 'WORK_VERIFIED':
        return `Work proof verified in ${event.data.verify_time_ms}ms`;
      case 'DATABASE_UPDATED':
        return `Added ${event.data.urls_added} URLs (total: ${event.data.total_urls})`;
      case 'ERROR':
        return event.data.error;
      default:
        return JSON.stringify(event.data);
    }
  };

  return (
    <div className={`flex items-start gap-3 text-sm py-2 border-b border-gray-800/50 animate-slide-in ${color}`}>
      <span className="text-gray-500 w-20 flex-shrink-0">{time}</span>
      <span className="w-6">{icon}</span>
      <span className="flex-1">{getMessage()}</span>
      {event.data.batch_id && (
        <span className="text-gray-600 text-xs font-mono">
          {event.data.batch_id.slice(0, 8)}
        </span>
      )}
    </div>
  );
}
