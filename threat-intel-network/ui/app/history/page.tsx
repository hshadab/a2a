'use client';

import { useState, useEffect } from 'react';
import {
  getNetworkStats,
  getClassificationHistory,
  getActivities,
  getPaymentActivities,
  formatNumber,
  formatUSDC,
  getActivityIcon,
  getActivityColor,
  getCategoryColor,
  NetworkStats,
  ClassificationHistoryItem,
  Activity,
} from '@/lib/api';
import {
  Clock,
  Shield,
  AlertTriangle,
  CheckCircle,
  ExternalLink,
  RefreshCw,
  DollarSign,
  Lock,
  Search,
  Database,
} from 'lucide-react';

type TabType = 'all' | 'classifications' | 'payments' | 'proofs';

export default function HistoryPage() {
  const [stats, setStats] = useState<NetworkStats | null>(null);
  const [activities, setActivities] = useState<Activity[]>([]);
  const [history, setHistory] = useState<ClassificationHistoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<TabType>('all');
  const [totalPayments, setTotalPayments] = useState(0);

  const fetchData = async () => {
    try {
      const statsData = await getNetworkStats();
      setStats(statsData);

      // Fetch based on active tab
      if (activeTab === 'all') {
        const activitiesData = await getActivities(100);
        setActivities(activitiesData.activities);
      } else if (activeTab === 'classifications') {
        const activitiesData = await getActivities(100, 'classification');
        setActivities(activitiesData.activities);
      } else if (activeTab === 'payments') {
        const paymentsData = await getPaymentActivities(100);
        setActivities(paymentsData.payments);
        setTotalPayments(paymentsData.total_usdc);
      } else if (activeTab === 'proofs') {
        // Get authorization + verification activities
        const [authData, verifyData] = await Promise.all([
          getActivities(50, 'authorization'),
          getActivities(50, 'verification'),
        ]);
        // Merge and sort by timestamp
        const merged = [...authData.activities, ...verifyData.activities]
          .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
          .slice(0, 100);
        setActivities(merged);
      }

      // Also fetch traditional history for fallback
      const historyData = await getClassificationHistory(100);
      setHistory(historyData.classifications);
    } catch (e) {
      console.error('Failed to fetch data:', e);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    setLoading(true);
    fetchData();
    const interval = setInterval(fetchData, 10000);
    return () => clearInterval(interval);
  }, [activeTab]);

  const tabs: { id: TabType; label: string; icon: React.ReactNode }[] = [
    { id: 'all', label: 'All', icon: <Database size={14} /> },
    { id: 'classifications', label: 'Classifications', icon: <Shield size={14} /> },
    { id: 'payments', label: 'Payments', icon: <DollarSign size={14} /> },
    { id: 'proofs', label: 'Proofs', icon: <Lock size={14} /> },
  ];

  const getClassificationIcon = (classification: string) => {
    switch (classification?.toLowerCase()) {
      case 'phishing':
        return <AlertTriangle className="text-red-400" size={18} />;
      case 'safe':
        return <CheckCircle className="text-green-400" size={18} />;
      default:
        return <Shield className="text-yellow-400" size={18} />;
    }
  };

  const getClassificationColor = (classification: string) => {
    switch (classification?.toLowerCase()) {
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
            Activity History
          </h1>
          <p className="text-gray-400 mt-1 text-sm">
            Pipeline activity with zkML proofs and blockchain transactions
          </p>
        </div>

        {/* Stats Summary */}
        <div className="flex items-center gap-4 md:gap-6">
          <div className="text-center">
            <p className="text-xl md:text-2xl font-bold text-white">{formatNumber(stats?.total_urls || 0)}</p>
            <p className="text-[10px] md:text-xs text-gray-500">Total URLs</p>
          </div>
          <div className="text-center">
            <p className="text-xl md:text-2xl font-bold text-red-400">{formatNumber(stats?.phishing_count || 0)}</p>
            <p className="text-[10px] md:text-xs text-gray-500">Phishing</p>
          </div>
          {activeTab === 'payments' && (
            <div className="text-center">
              <p className="text-xl md:text-2xl font-bold text-yellow-400">{formatUSDC(totalPayments)}</p>
              <p className="text-[10px] md:text-xs text-gray-500">Total Paid</p>
            </div>
          )}
          <button
            onClick={fetchData}
            className="p-2 text-gray-400 hover:text-white transition-colors"
            title="Refresh"
          >
            <RefreshCw size={18} className={loading ? 'animate-spin' : ''} />
          </button>
        </div>
      </div>

      {/* Tab Bar */}
      <div className="flex items-center gap-2 mb-4 overflow-x-auto pb-2">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all whitespace-nowrap ${
              activeTab === tab.id
                ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30'
                : 'bg-gray-800/50 text-gray-400 hover:bg-gray-800 hover:text-white'
            }`}
          >
            {tab.icon}
            {tab.label}
          </button>
        ))}
        <div className="flex-1" />
        <span className="text-xs text-gray-500">
          {activities.length} activities
        </span>
      </div>

      {/* Activity List */}
      <div className="card overflow-hidden">
        {loading && activities.length === 0 ? (
          <div className="p-12 text-center text-gray-500">
            <RefreshCw className="mx-auto mb-4 animate-spin" size={32} />
            <p>Loading activities...</p>
          </div>
        ) : activities.length === 0 ? (
          <div className="p-12 text-center text-gray-500">
            <Search className="mx-auto mb-4 opacity-50" size={48} />
            <p>No activities yet. Trigger the pipeline to process URLs.</p>
          </div>
        ) : (
          <div className="divide-y divide-gray-800/50">
            {activities.map((activity) => (
              <ActivityListItem key={activity.id} activity={activity} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function ActivityListItem({ activity }: { activity: Activity }) {
  const icon = getActivityIcon(activity);
  const color = getActivityColor(activity);

  const getCategoryBadge = () => {
    const categoryColors = getCategoryColor(activity.category);
    return (
      <span className={`px-2 py-0.5 rounded text-xs font-medium border ${categoryColors}`}>
        {activity.category.toUpperCase()}
      </span>
    );
  };

  const getClassificationBadge = () => {
    if (!activity.classification) return null;
    const colors = activity.classification === 'PHISHING'
      ? 'text-red-400 bg-red-500/10 border-red-500/30'
      : activity.classification === 'SAFE'
      ? 'text-green-400 bg-green-500/10 border-green-500/30'
      : 'text-yellow-400 bg-yellow-500/10 border-yellow-500/30';

    return (
      <span className={`px-2 py-0.5 rounded text-xs font-medium border ${colors}`}>
        {activity.classification}
      </span>
    );
  };

  return (
    <div className="p-3 md:p-4 hover:bg-gray-800/30 transition-colors">
      <div className="flex items-start gap-3">
        {/* Icon */}
        <div className={`flex-shrink-0 mt-0.5 text-lg ${color}`}>
          {icon}
        </div>

        {/* Content */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            {getCategoryBadge()}
            {getClassificationBadge()}
            {activity.confidence && (
              <span className="text-xs text-gray-500">
                {(activity.confidence * 100).toFixed(0)}% confidence
              </span>
            )}
            {activity.amount_usdc && (
              <span className="text-xs text-yellow-400 font-medium">
                {formatUSDC(activity.amount_usdc)}
              </span>
            )}
          </div>

          <p className="text-sm text-white font-medium mt-1">
            {activity.title}
          </p>
          <p className="text-xs text-gray-400 mt-0.5">
            {activity.description}
          </p>

          {/* URL display */}
          {activity.url && (
            <p className="text-xs text-gray-500 font-mono mt-1 truncate" title={activity.url}>
              {activity.url}
            </p>
          )}

          {/* Proof/Transaction details */}
          <div className="flex items-center gap-3 mt-2 text-xs text-gray-500 flex-wrap">
            {activity.proof_hash && (
              <span className="font-mono">
                Proof: {activity.proof_hash.slice(0, 12)}...
              </span>
            )}
            {activity.prove_time_ms && (
              <span>
                {activity.prove_time_ms}ms
              </span>
            )}
            {activity.tx_hash && activity.tx_hash !== 'simulated' && (
              <span className="font-mono">
                Tx: {activity.tx_hash.slice(0, 10)}...
              </span>
            )}
            {activity.tx_hash === 'simulated' && (
              <span className="text-gray-600 italic">
                (simulated)
              </span>
            )}
          </div>

          {/* Blockchain explorer link */}
          {activity.explorer_url && (
            <a
              href={activity.explorer_url}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 mt-2 text-xs text-blue-400 hover:text-blue-300 transition-colors"
            >
              <ExternalLink size={12} />
              View on Basescan
            </a>
          )}
        </div>

        {/* Timestamp and Agent */}
        <div className="text-right flex-shrink-0">
          <p className="text-xs text-gray-400">
            {new Date(activity.timestamp).toLocaleString('en-US', { timeZone: 'America/New_York', month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit', second: '2-digit', hour12: true })} EST
          </p>
          <p className="text-[10px] text-gray-600 mt-1 px-1.5 py-0.5 bg-gray-800/50 rounded inline-block">
            {activity.agent}
          </p>
        </div>
      </div>
    </div>
  );
}
