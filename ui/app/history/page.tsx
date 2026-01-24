'use client';

import { useState, useEffect } from 'react';
import { useWebSocket } from '@/hooks/useWebSocket';
import {
  getNetworkStats,
  getEventIcon,
  getEventColor,
  formatNumber,
  NetworkStats,
} from '@/lib/api';
import {
  Activity,
  Clock,
  Shield,
  AlertTriangle,
  CheckCircle,
  XCircle,
  Filter,
  Download,
} from 'lucide-react';

const WS_URL = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000/ws';

interface AgentEvent {
  type: string;
  timestamp: string;
  data: Record<string, any>;
}

export default function HistoryPage() {
  const { events } = useWebSocket(WS_URL);
  const [stats, setStats] = useState<NetworkStats | null>(null);
  const [filter, setFilter] = useState<string>('all');
  const [expandedEvent, setExpandedEvent] = useState<number | null>(null);

  useEffect(() => {
    async function fetchStats() {
      try {
        const data = await getNetworkStats();
        setStats(data);
      } catch (e) {
        console.error('Failed to fetch stats:', e);
      }
    }
    fetchStats();
    const interval = setInterval(fetchStats, 30000);
    return () => clearInterval(interval);
  }, []);

  const filterOptions = [
    { value: 'all', label: 'All Events' },
    { value: 'SCOUT', label: 'Scout' },
    { value: 'POLICY', label: 'Spending Policy' },
    { value: 'ANALYST', label: 'Classifier' },
    { value: 'PAYMENT', label: 'Payments' },
    { value: 'DATABASE', label: 'Database' },
    { value: 'ERROR', label: 'Errors' },
  ];

  const filteredEvents = events.filter((event) => {
    if (filter === 'all') return true;
    return event.type.startsWith(filter);
  });

  const getEventDetails = (event: AgentEvent) => {
    const details: { label: string; value: string }[] = [];

    if (event.data.batch_id) {
      details.push({ label: 'Batch ID', value: event.data.batch_id });
    }
    if (event.data.url_count !== undefined) {
      details.push({ label: 'URL Count', value: event.data.url_count.toString() });
    }
    if (event.data.source) {
      details.push({ label: 'Source', value: event.data.source });
    }
    if (event.data.decision) {
      details.push({ label: 'Decision', value: event.data.decision });
    }
    if (event.data.confidence !== undefined) {
      details.push({ label: 'Confidence', value: `${(event.data.confidence * 100).toFixed(1)}%` });
    }
    if (event.data.amount_usdc !== undefined) {
      details.push({ label: 'Amount', value: `${event.data.amount_usdc} USDC` });
    }
    if (event.data.tx_hash) {
      details.push({ label: 'TX Hash', value: event.data.tx_hash });
    }
    if (event.data.proof_hash) {
      details.push({ label: 'Proof Hash', value: event.data.proof_hash });
    }
    if (event.data.prove_time_ms !== undefined) {
      details.push({ label: 'Prove Time', value: `${event.data.prove_time_ms}ms` });
    }
    if (event.data.verify_time_ms !== undefined) {
      details.push({ label: 'Verify Time', value: `${event.data.verify_time_ms}ms` });
    }
    if (event.data.phishing_count !== undefined) {
      details.push({ label: 'Phishing', value: event.data.phishing_count.toString() });
    }
    if (event.data.safe_count !== undefined) {
      details.push({ label: 'Safe', value: event.data.safe_count.toString() });
    }
    if (event.data.error) {
      details.push({ label: 'Error', value: event.data.error });
    }

    return details;
  };

  const getMessage = (event: AgentEvent) => {
    switch (event.type) {
      case 'SCOUT_FOUND_URLS':
        return `Discovered ${event.data.url_count} URLs from ${event.data.source}`;
      case 'POLICY_REQUESTING':
        return `Requesting spending authorization for ${event.data.url_count} URLs`;
      case 'POLICY_PROVING':
        return 'Generating spending policy proof...';
      case 'POLICY_RESPONSE':
        return `Spending ${event.data.decision} (${(event.data.confidence * 100).toFixed(0)}% confidence)`;
      case 'POLICY_VERIFIED':
        return `Spending policy proof verified in ${event.data.verify_time_ms}ms`;
      case 'PAYMENT_SENDING':
        return `Sending ${event.data.amount_usdc} USDC payment`;
      case 'PAYMENT_SENT':
        return `Payment of ${event.data.amount_usdc} USDC confirmed`;
      case 'ANALYST_PROCESSING':
        return `Classifying ${event.data.url_count} URLs`;
      case 'ANALYST_PROVING':
        return 'Generating classification proof...';
      case 'ANALYST_RESPONSE':
        return `Classified: ${event.data.phishing_count} phishing, ${event.data.safe_count} safe`;
      case 'WORK_VERIFIED':
        return `Classification proof verified in ${event.data.verify_time_ms}ms`;
      case 'DATABASE_UPDATED':
        return `Stored ${event.data.urls_added} URLs (total: ${event.data.total_urls})`;
      case 'ERROR':
        return `Error: ${event.data.error}`;
      default:
        return event.type;
    }
  };

  return (
    <div className="min-h-screen p-6 pt-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-bold text-white flex items-center gap-3">
            <Clock className="text-cyan-400" />
            Activity History
          </h1>
          <p className="text-gray-400 mt-1">
            Complete log of agent actions, proofs, and payments
          </p>
        </div>

        {/* Stats Summary */}
        <div className="flex items-center gap-6">
          <div className="text-center">
            <p className="text-2xl font-bold text-white">{formatNumber(stats?.total_urls || 0)}</p>
            <p className="text-xs text-gray-500">Total URLs</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold text-red-400">{formatNumber(stats?.phishing_count || 0)}</p>
            <p className="text-xs text-gray-500">Phishing</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold text-green-400">{formatNumber(stats?.safe_count || 0)}</p>
            <p className="text-xs text-gray-500">Safe</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold text-purple-400">{formatNumber(stats?.total_proofs || 0)}</p>
            <p className="text-xs text-gray-500">Proofs</p>
          </div>
        </div>
      </div>

      {/* Filter Bar */}
      <div className="flex items-center gap-4 mb-6">
        <div className="flex items-center gap-2 text-gray-400">
          <Filter size={16} />
          <span className="text-sm">Filter:</span>
        </div>
        <div className="flex gap-2">
          {filterOptions.map((option) => (
            <button
              key={option.value}
              onClick={() => setFilter(option.value)}
              className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-all ${
                filter === option.value
                  ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30'
                  : 'bg-gray-800/50 text-gray-400 hover:bg-gray-800 hover:text-white'
              }`}
            >
              {option.label}
            </button>
          ))}
        </div>
        <div className="flex-1" />
        <span className="text-sm text-gray-500">
          {filteredEvents.length} events
        </span>
      </div>

      {/* Events List */}
      <div className="card">
        <div className="divide-y divide-gray-800">
          {filteredEvents.length === 0 ? (
            <div className="p-12 text-center text-gray-500">
              <Activity className="mx-auto mb-4 opacity-50" size={48} />
              <p>No events yet. Waiting for agent activity...</p>
            </div>
          ) : (
            filteredEvents.map((event, i) => {
              const isExpanded = expandedEvent === i;
              const details = getEventDetails(event);
              const sampleUrls = event.data.sample_urls || event.data.sample_results;

              return (
                <div
                  key={`${event.timestamp}-${i}`}
                  className={`p-4 hover:bg-gray-800/30 cursor-pointer transition-colors ${
                    isExpanded ? 'bg-gray-800/50' : ''
                  }`}
                  onClick={() => setExpandedEvent(isExpanded ? null : i)}
                >
                  <div className="flex items-start gap-4">
                    {/* Icon */}
                    <div className="text-2xl w-8 flex-shrink-0">
                      {getEventIcon(event.type)}
                    </div>

                    {/* Content */}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-3">
                        <span className={`font-medium ${getEventColor(event.type)}`}>
                          {getMessage(event)}
                        </span>
                        {event.type.includes('VERIFIED') && (
                          <CheckCircle size={16} className="text-green-400" />
                        )}
                        {event.type === 'ERROR' && (
                          <XCircle size={16} className="text-red-400" />
                        )}
                      </div>

                      {/* Expanded Details */}
                      {isExpanded && (
                        <div className="mt-4 space-y-3">
                          {/* Details Grid */}
                          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                            {details.map((detail, j) => (
                              <div key={j} className="bg-gray-900/50 rounded-lg p-3">
                                <p className="text-xs text-gray-500 uppercase">{detail.label}</p>
                                <p className="text-sm text-white font-mono truncate" title={detail.value}>
                                  {detail.value.length > 20 ? detail.value.slice(0, 17) + '...' : detail.value}
                                </p>
                              </div>
                            ))}
                          </div>

                          {/* Sample URLs */}
                          {sampleUrls && sampleUrls.length > 0 && (
                            <div className="bg-gray-900/50 rounded-lg p-3">
                              <p className="text-xs text-gray-500 uppercase mb-2">Sample URLs</p>
                              <div className="space-y-1">
                                {sampleUrls.slice(0, 5).map((item: string | { url: string; classification: string }, j: number) => {
                                  const url = typeof item === 'string' ? item : item.url;
                                  const classification = typeof item === 'object' ? item.classification : null;
                                  return (
                                    <div key={j} className="flex items-center gap-2 text-xs">
                                      <span className="font-mono text-gray-400 truncate flex-1">{url}</span>
                                      {classification && (
                                        <span className={`px-2 py-0.5 rounded text-[10px] font-medium ${
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
                              </div>
                            </div>
                          )}
                        </div>
                      )}
                    </div>

                    {/* Timestamp */}
                    <div className="text-right flex-shrink-0">
                      <p className="text-sm text-gray-400">
                        {new Date(event.timestamp).toLocaleTimeString()}
                      </p>
                      <p className="text-xs text-gray-600">
                        {new Date(event.timestamp).toLocaleDateString()}
                      </p>
                    </div>
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
}
