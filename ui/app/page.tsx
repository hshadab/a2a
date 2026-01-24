'use client';

import { useState, useEffect } from 'react';
import { useWebSocket } from '@/hooks/useWebSocket';
import {
  getNetworkStats,
  getHealthStatus,
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
  Database,
  DollarSign,
  AlertTriangle,
  CheckCircle,
  Cpu,
} from 'lucide-react';
import AgentPipeline from '@/components/AgentPipeline';
import TreasuryWidget from '@/components/TreasuryWidget';

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

  return (
    <div className="min-h-screen p-6">
      {/* Header */}
      <header className="mb-8">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-white flex items-center gap-3">
              <Shield className="text-cyber-blue" />
              ThreatProof
            </h1>
            <p className="text-gray-400 mt-1">
              Verifiable Threat Intelligence |{' '}
              <span className="text-cyan-400">Google A2A</span> +{' '}
              <span className="text-green-400">x402</span> +{' '}
              <a
                href="https://github.com/ICME-Lab/jolt-atlas"
                target="_blank"
                rel="noopener noreferrer"
                className="text-purple-400 hover:text-purple-300 underline decoration-dotted"
              >
                Jolt Atlas zkML
              </a>
            </p>
          </div>
          <div className="flex items-center gap-4">
            {/* WebSocket Connection Status */}
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

            {/* Autonomous Mode Indicator */}
            <div className="flex items-center gap-2 px-3 py-1.5 bg-cyan-500/10 border border-cyan-500/30 rounded-lg">
              <Cpu size={14} className="text-cyan-400 animate-pulse" />
              <span className="text-sm text-cyan-400">Autonomous</span>
            </div>

            {/* Running Time */}
            {stats?.running_since && (
              <div className="text-sm text-gray-400">
                Uptime: {formatDuration(stats.running_since)}
              </div>
            )}
          </div>
        </div>

        {/* Treasury Widget */}
        <div className="mt-4">
          <TreasuryWidget />
        </div>
      </header>

      {/* Agent Pipeline Visualization */}
      <section className="mb-8">
        <AgentPipeline
          events={events}
          lastEvent={lastEvent}
          stats={stats}
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

      {/* Autonomous Economy Section */}
      <section className="mt-8 card p-6 border border-cyan-500/20">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-10 h-10 rounded-full bg-gradient-to-br from-cyan-500/20 to-purple-500/20 flex items-center justify-center">
            <Cpu className="text-cyan-400" size={20} />
          </div>
          <div>
            <h2 className="text-lg font-semibold text-white">Autonomous Agent-to-Agent Commerce</h2>
            <p className="text-sm text-gray-500">A self-sustaining AI micro-economy</p>
          </div>
        </div>
        <div className="grid grid-cols-3 gap-6">
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-cyan-400" />
              <span className="text-sm font-medium text-cyan-400">Discover</span>
            </div>
            <p className="text-xs text-gray-400">
              Scout Agent autonomously discovers URLs from threat feeds and pays Policy Agent for authorization decisions.
            </p>
          </div>
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-green-400" />
              <span className="text-sm font-medium text-green-400">Pay & Verify</span>
            </div>
            <p className="text-xs text-gray-400">
              Agents exchange USDC via x402 protocol. Every payment triggers zkML proof generation to verify honest work.
            </p>
          </div>
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-purple-400" />
              <span className="text-sm font-medium text-purple-400">Self-Sustaining</span>
            </div>
            <p className="text-xs text-gray-400">
              USDC circulates between agents in a closed loop. Only gas is consumed—the economy runs indefinitely.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}

// ============ Components ============

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
