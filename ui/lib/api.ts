const SCOUT_URL = process.env.NEXT_PUBLIC_SCOUT_URL || 'http://localhost:8000';

export interface NetworkStats {
  total_urls: number;
  phishing_count: number;
  safe_count: number;
  suspicious_count: number;
  total_batches: number;
  total_proofs: number;
  treasury_balance_usdc: number;
  total_spent_usdc: number;
  policy_paid_usdc: number;
  analyst_paid_usdc: number;
  running_since: string | null;
}

export interface HealthStatus {
  status: string;
  running: boolean;
  batches_processed: number;
  urls_processed: number;
}

export async function getNetworkStats(): Promise<NetworkStats> {
  const response = await fetch(`${SCOUT_URL}/stats`);
  if (!response.ok) {
    throw new Error('Failed to fetch stats');
  }
  return response.json();
}

export async function getHealthStatus(): Promise<HealthStatus> {
  const response = await fetch(`${SCOUT_URL}/health`);
  if (!response.ok) {
    throw new Error('Failed to fetch health');
  }
  return response.json();
}

export async function triggerBatch(): Promise<{ batch_id: string | null }> {
  const response = await fetch(`${SCOUT_URL}/trigger`, {
    method: 'POST',
  });
  if (!response.ok) {
    throw new Error('Failed to trigger batch');
  }
  return response.json();
}

export function formatUSDC(amount: number): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
    maximumFractionDigits: 6,
  }).format(amount);
}

export function formatNumber(num: number): string {
  return new Intl.NumberFormat('en-US').format(num);
}

export function formatDuration(startTime: string | null): string {
  if (!startTime) return '-';

  const start = new Date(startTime);
  const now = new Date();
  const diffMs = now.getTime() - start.getTime();

  const days = Math.floor(diffMs / (1000 * 60 * 60 * 24));
  const hours = Math.floor((diffMs % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60));
  const minutes = Math.floor((diffMs % (1000 * 60 * 60)) / (1000 * 60));

  if (days > 0) return `${days}d ${hours}h`;
  if (hours > 0) return `${hours}h ${minutes}m`;
  return `${minutes}m`;
}

export function getEventIcon(type: string): string {
  switch (type) {
    case 'SCOUT_FOUND_URLS':
      return '🔍';
    case 'POLICY_REQUESTING':
      return '📋';
    case 'POLICY_PROVING':
      return '🔐';
    case 'POLICY_RESPONSE':
      return '✅';
    case 'POLICY_VERIFIED':
      return '✓';
    case 'PAYMENT_SENDING':
      return '💸';
    case 'PAYMENT_SENT':
      return '💰';
    case 'ANALYST_PROCESSING':
      return '🔬';
    case 'ANALYST_PROVING':
      return '🔐';
    case 'ANALYST_RESPONSE':
      return '📊';
    case 'WORK_VERIFIED':
      return '✓';
    case 'DATABASE_UPDATED':
      return '💾';
    case 'ERROR':
      return '❌';
    default:
      return '•';
  }
}

export function getEventColor(type: string): string {
  switch (type) {
    case 'SCOUT_FOUND_URLS':
      return 'text-blue-400';
    case 'POLICY_REQUESTING':
    case 'POLICY_PROVING':
      return 'text-purple-400';
    case 'POLICY_RESPONSE':
    case 'POLICY_VERIFIED':
      return 'text-green-400';
    case 'PAYMENT_SENDING':
    case 'PAYMENT_SENT':
      return 'text-yellow-400';
    case 'ANALYST_PROCESSING':
    case 'ANALYST_PROVING':
      return 'text-cyan-400';
    case 'ANALYST_RESPONSE':
    case 'WORK_VERIFIED':
      return 'text-green-400';
    case 'DATABASE_UPDATED':
      return 'text-emerald-400';
    case 'ERROR':
      return 'text-red-400';
    default:
      return 'text-gray-400';
  }
}
