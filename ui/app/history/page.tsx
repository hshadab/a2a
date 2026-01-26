'use client';

import { useState, useEffect } from 'react';
import {
  getNetworkStats,
  getClassificationHistory,
  formatNumber,
  NetworkStats,
  ClassificationHistoryItem,
} from '@/lib/api';
import {
  Activity,
  Clock,
  Shield,
  AlertTriangle,
  CheckCircle,
  ExternalLink,
  RefreshCw,
} from 'lucide-react';

export default function HistoryPage() {
  const [stats, setStats] = useState<NetworkStats | null>(null);
  const [history, setHistory] = useState<ClassificationHistoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [filter, setFilter] = useState<string>('all');

  const fetchData = async () => {
    try {
      const [statsData, historyData] = await Promise.all([
        getNetworkStats(),
        getClassificationHistory(100),
      ]);
      setStats(statsData);
      setHistory(historyData.classifications);
    } catch (e) {
      console.error('Failed to fetch data:', e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 10000);
    return () => clearInterval(interval);
  }, []);

  const filterOptions = [
    { value: 'all', label: 'All' },
    { value: 'phishing', label: 'Phishing' },
    { value: 'safe', label: 'Safe' },
    { value: 'suspicious', label: 'Suspicious' },
  ];

  const filteredHistory = history.filter((item) => {
    if (filter === 'all') return true;
    return item.classification === filter;
  });

  const getClassificationIcon = (classification: string) => {
    switch (classification) {
      case 'phishing':
        return <AlertTriangle className="text-red-400" size={18} />;
      case 'safe':
        return <CheckCircle className="text-green-400" size={18} />;
      default:
        return <Shield className="text-yellow-400" size={18} />;
    }
  };

  const getClassificationColor = (classification: string) => {
    switch (classification) {
      case 'phishing':
        return 'text-red-400 bg-red-500/10 border-red-500/30';
      case 'safe':
        return 'text-green-400 bg-green-500/10 border-green-500/30';
      default:
        return 'text-yellow-400 bg-yellow-500/10 border-yellow-500/30';
    }
  };

  return (
    <div className="min-h-screen p-4 md:p-6 pt-6 md:pt-8">
      {/* Header */}
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-6">
        <div>
          <h1 className="text-xl md:text-2xl font-bold text-white flex items-center gap-3">
            <Clock className="text-cyan-400" />
            Classification History
          </h1>
          <p className="text-gray-400 mt-1 text-sm">
            Recent URL classifications with zkML proofs
          </p>
        </div>

        {/* Stats Summary */}
        <div className="flex items-center gap-4 md:gap-6">
          <div className="text-center">
            <p className="text-xl md:text-2xl font-bold text-white">{formatNumber(stats?.total_urls || 0)}</p>
            <p className="text-[10px] md:text-xs text-gray-500">Total</p>
          </div>
          <div className="text-center">
            <p className="text-xl md:text-2xl font-bold text-red-400">{formatNumber(stats?.phishing_count || 0)}</p>
            <p className="text-[10px] md:text-xs text-gray-500">Phishing</p>
          </div>
          <div className="text-center">
            <p className="text-xl md:text-2xl font-bold text-green-400">{formatNumber(stats?.safe_count || 0)}</p>
            <p className="text-[10px] md:text-xs text-gray-500">Safe</p>
          </div>
          <button
            onClick={fetchData}
            className="p-2 text-gray-400 hover:text-white transition-colors"
            title="Refresh"
          >
            <RefreshCw size={18} className={loading ? 'animate-spin' : ''} />
          </button>
        </div>
      </div>

      {/* Filter Bar */}
      <div className="flex items-center gap-3 mb-4">
        <span className="text-sm text-gray-500">Filter:</span>
        <div className="flex gap-2">
          {filterOptions.map((option) => (
            <button
              key={option.value}
              onClick={() => setFilter(option.value)}
              className={`px-3 py-1 rounded text-xs font-medium transition-all ${
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
        <span className="text-xs text-gray-500">
          {filteredHistory.length} classifications
        </span>
      </div>

      {/* Classifications List */}
      <div className="card overflow-hidden">
        {loading && history.length === 0 ? (
          <div className="p-12 text-center text-gray-500">
            <RefreshCw className="mx-auto mb-4 animate-spin" size={32} />
            <p>Loading classifications...</p>
          </div>
        ) : filteredHistory.length === 0 ? (
          <div className="p-12 text-center text-gray-500">
            <Activity className="mx-auto mb-4 opacity-50" size={48} />
            <p>No classifications yet. Trigger the analyst to process URLs.</p>
          </div>
        ) : (
          <div className="divide-y divide-gray-800/50">
            {filteredHistory.map((item, i) => (
              <div
                key={item.request_id}
                className="p-3 md:p-4 hover:bg-gray-800/30 transition-colors"
              >
                <div className="flex items-start gap-3">
                  {/* Classification Icon */}
                  <div className="flex-shrink-0 mt-0.5">
                    {getClassificationIcon(item.classification)}
                  </div>

                  {/* Content */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2 flex-wrap">
                      <span className={`px-2 py-0.5 rounded text-xs font-medium border ${getClassificationColor(item.classification)}`}>
                        {item.classification.toUpperCase()}
                      </span>
                      <span className="text-xs text-gray-500">
                        {(item.confidence * 100).toFixed(0)}% confidence
                      </span>
                    </div>
                    <p className="text-sm text-white font-mono mt-1 truncate" title={item.url}>
                      {item.url}
                    </p>
                    <div className="flex items-center gap-3 mt-1 text-xs text-gray-500">
                      <span>{item.domain}</span>
                      <span className="text-gray-700">•</span>
                      <span className="font-mono">{item.proof_hash.slice(0, 12)}...</span>
                    </div>
                  </div>

                  {/* Timestamp */}
                  <div className="text-right flex-shrink-0">
                    <p className="text-xs text-gray-400">
                      {new Date(item.timestamp).toLocaleTimeString()}
                    </p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
