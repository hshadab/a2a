'use client';

import { useState, useEffect } from 'react';
import { useWebSocket } from '@/hooks/useWebSocket';
import {
  getNetworkStats,
  getHealthStatus,
  formatUSDC,
  formatNumber,
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
    <div className="min-h-screen p-4 md:p-6 pt-6 md:pt-8">
      {/* Agent Pipeline Visualization */}
      <section className="mb-6 md:mb-8">
        <AgentPipeline
          events={events}
          lastEvent={lastEvent}
          stats={stats}
        />
      </section>

      {/* Stats Grid - 2 cols on mobile, 4 on desktop */}
      <section className="grid grid-cols-2 md:grid-cols-4 gap-3 md:gap-4 mb-6 md:mb-8">
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

      {/* Economics (2-Agent Model) - 2 cols on mobile, 4 on desktop */}
      <section className="grid grid-cols-2 md:grid-cols-4 gap-3 md:gap-4 mb-6 md:mb-8">
        <StatCard
          label="Total Spent"
          value={stats ? formatUSDC(stats.total_spent_usdc) : '-'}
          icon={<DollarSign className="text-yellow-400" />}
        />
        <StatCard
          label="Analyst → Scout"
          value={stats ? formatUSDC(stats.analyst_paid_usdc) : '-'}
          subtext="URL discovery"
        />
        <StatCard
          label="Scout → Analyst"
          value={stats ? formatUSDC(stats.policy_paid_usdc) : '-'}
          subtext="Classification feedback"
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

      {/* Agent-to-Agent Micro-Economy (2-Agent Model) */}
      <section className="mt-6 md:mt-8 card p-4 md:p-6 border border-cyan-500/20">
        <div className="flex items-center gap-3 mb-4">
          <div className="w-8 h-8 md:w-10 md:h-10 rounded-full bg-gradient-to-br from-cyan-500/20 to-blue-500/20 flex items-center justify-center">
            <Cpu className="text-cyan-400" size={18} />
          </div>
          <div>
            <h2 className="text-base md:text-lg font-semibold text-white">Per-URL Circular Economy</h2>
            <p className="text-xs md:text-sm text-gray-500">Mutual work verification (Net: $0.00)</p>
          </div>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 md:gap-6">
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-cyan-400" />
              <span className="text-sm font-medium text-cyan-400">Analyst → Scout ($0.001/URL)</span>
            </div>
            <p className="text-xs text-gray-400">
              Analyst verifies Scout&apos;s quality work proof, then pays for URL discovery.
            </p>
          </div>
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-blue-400" />
              <span className="text-sm font-medium text-blue-400">Scout → Analyst ($0.001/URL)</span>
            </div>
            <p className="text-xs text-gray-400">
              Scout verifies Analyst&apos;s classification proof, then pays feedback.
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
    <div className="card p-3 md:p-4">
      <div className="flex items-center justify-between mb-1 md:mb-2">
        <span className="stat-label text-[10px] md:text-xs">{label}</span>
        <span className="scale-75 md:scale-100">{icon}</span>
      </div>
      <div className="flex items-baseline gap-1 md:gap-2">
        <span className="stat-value text-white text-lg md:text-2xl">{value}</span>
        {change && (
          <span className="text-xs md:text-sm text-cyber-green animate-slide-in">
            {change}
          </span>
        )}
      </div>
      {subtext && <p className="text-[10px] md:text-xs text-gray-500 mt-1">{subtext}</p>}
    </div>
  );
}

function EventRow({ event }: { event: { type: string; timestamp: string; data: Record<string, any> } }) {
  const time = new Date(event.timestamp).toLocaleTimeString();
  const icon = getEventIcon(event.type, event.data);
  const color = getEventColor(event.type, event.data);

  const getMessage = () => {
    switch (event.type) {
      case 'SCOUT_FOUND_URLS':
        return event.data.url_count === 1
          ? `Found URL from ${event.data.source}`
          : `Found ${event.data.url_count} URLs from ${event.data.source}`;
      case 'SCOUT_AUTHORIZING':
        return `Scout self-authorizing ($${event.data.estimated_cost})`;
      case 'SCOUT_AUTHORIZED':
        return `Scout self-authorized (${(event.data.confidence * 100).toFixed(0)}% confidence)`;
      case 'ANALYST_AUTHORIZING':
        return `Analyst self-authorizing ($${event.data.estimated_cost})`;
      case 'ANALYST_AUTHORIZED':
        return `Analyst self-authorized (${(event.data.confidence * 100).toFixed(0)}% confidence)`;
      case 'SPENDING_PROOF_VERIFIED':
        return event.data.valid !== false
          ? `${event.data.agent} spending proof verified in ${event.data.verify_time_ms}ms`
          : `${event.data.agent} spending proof failed (${event.data.verify_time_ms}ms)`;
      case 'PAYMENT_SENDING':
        return `Sending $${event.data.amount_usdc} USDC`;
      case 'PAYMENT_SENT':
        return `Paid $${event.data.amount_usdc} USDC (tx: ${event.data.tx_hash?.slice(0, 10)}...)`;
      case 'ANALYST_PROCESSING':
        return event.data.url_count === 1
          ? 'Processing URL...'
          : `Processing ${event.data.url_count} URLs${event.data.progress ? ` (${event.data.progress}/${event.data.url_count})` : ''}`;
      case 'ANALYST_PROVING':
        return 'Generating classification proof...';
      case 'ANALYST_RESPONSE':
        // Single URL response
        if (event.data.classification) {
          return `Classified: ${event.data.classification} (${(event.data.confidence * 100).toFixed(0)}% confidence)`;
        }
        // Legacy batch response
        return `Classified: ${event.data.phishing_count} phishing, ${event.data.safe_count} safe, ${event.data.suspicious_count} suspicious`;
      case 'WORK_VERIFIED':
        return event.data.valid !== false
          ? `${event.data.quality_tier || 'Work'} proof verified in ${event.data.verify_time_ms}ms`
          : `Work proof verification failed (${event.data.verify_time_ms}ms)`;
      case 'DATABASE_UPDATED':
        return event.data.urls_added === 1
          ? `Added URL (total: ${event.data.total_urls})`
          : `Added ${event.data.urls_added} URLs (total: ${event.data.total_urls})`;
      case 'ERROR':
        return event.data.error;
      default:
        return JSON.stringify(event.data);
    }
  };

  // Check for sample URLs to display
  const sampleUrls = event.data.sample_urls || event.data.sample_results;
  const hasSamples = sampleUrls && sampleUrls.length > 0;

  return (
    <div className={`text-xs md:text-sm py-2 border-b border-gray-800/50 animate-slide-in ${color}`}>
      <div className="flex items-start gap-2 md:gap-3">
        <span className="text-gray-500 w-14 md:w-20 flex-shrink-0 text-[10px] md:text-sm">{time}</span>
        <span className="w-4 md:w-6">{icon}</span>
        <span className="flex-1 break-words">{getMessage()}</span>
        {(event.data.request_id || event.data.batch_id) && (
          <span className="hidden md:inline text-gray-600 text-xs font-mono">
            {(event.data.request_id || event.data.batch_id).slice(0, 8)}
          </span>
        )}
      </div>
      {/* Show sample URLs if available - hidden on mobile */}
      {hasSamples && (
        <div className="hidden md:block ml-[116px] mt-1.5 space-y-0.5">
          {sampleUrls.slice(0, 3).map((item: string | { url: string; classification: string }, i: number) => {
            const url = typeof item === 'string' ? item : item.url;
            const classification = typeof item === 'object' ? item.classification : null;
            const displayUrl = url.length > 60 ? url.slice(0, 57) + '...' : url;
            return (
              <div key={i} className="flex items-center gap-2 text-xs">
                <span className="text-gray-600">-</span>
                <span className="font-mono text-gray-400 break-all">{displayUrl}</span>
                {classification && (
                  <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${
                    classification === 'phishing' ? 'bg-red-500/20 text-red-400' :
                    classification === 'safe' ? 'bg-green-500/20 text-green-400' :
                    'bg-yellow-500/20 text-yellow-400'
                  }`}>
                    {classification}
                  </span>
                )}
              </div>
            );
          })}
          {sampleUrls.length > 3 && (
            <span className="text-xs text-gray-600 ml-4">
              +{sampleUrls.length - 3} more...
            </span>
          )}
        </div>
      )}
    </div>
  );
}
