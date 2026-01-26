const SCOUT_URL = process.env.NEXT_PUBLIC_SCOUT_URL || 'http://localhost:8000';
const POLICY_URL = process.env.NEXT_PUBLIC_POLICY_URL || 'http://localhost:8001';
const ANALYST_URL = process.env.NEXT_PUBLIC_ANALYST_URL || 'http://localhost:8002';

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
  const data = await response.json();
  // API returns nested structure {network: {...}, scout: {...}}
  return data.network || data;
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

export function formatUSDC(amount: number | null | undefined): string {
  if (amount == null || isNaN(amount)) return '$0.00';
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
    maximumFractionDigits: 6,
  }).format(amount);
}

export function formatNumber(num: number | null | undefined): string {
  if (num == null || isNaN(num)) return '0';
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

export function getEventIcon(type: string, data?: Record<string, any>): string {
  switch (type) {
    case 'SCOUT_FOUND_URLS':
      return '→';
    case 'SCOUT_AUTHORIZING':
      return '○';
    case 'SCOUT_AUTHORIZED':
      return '●';
    case 'SCOUT_PROVING':
      return '◐';
    case 'ANALYST_AUTHORIZING':
      return '○';
    case 'ANALYST_AUTHORIZED':
      return '●';
    case 'SPENDING_PROOF_VERIFIED':
      return data?.valid === false ? '✗' : '✓';
    case 'PAYMENT_SENDING':
      return '$';
    case 'PAYMENT_SENT':
      return '$';
    case 'ANALYST_PROCESSING':
      return '◎';
    case 'ANALYST_PROVING':
      return '◐';
    case 'ANALYST_RESPONSE':
      return '●';
    case 'WORK_VERIFIED':
      return data?.valid === false ? '✗' : '✓';
    case 'DATABASE_UPDATED':
      return '▣';
    case 'ERROR':
      return '✗';
    default:
      return '•';
  }
}

export function getEventColor(type: string, data?: Record<string, any>): string {
  switch (type) {
    case 'SCOUT_FOUND_URLS':
      return 'text-blue-400';
    case 'SCOUT_AUTHORIZING':
    case 'SCOUT_PROVING':
      return 'text-blue-400';
    case 'SCOUT_AUTHORIZED':
      return 'text-green-400';
    case 'ANALYST_AUTHORIZING':
    case 'ANALYST_PROVING':
      return 'text-cyan-400';
    case 'ANALYST_AUTHORIZED':
      return 'text-green-400';
    case 'SPENDING_PROOF_VERIFIED':
      return data?.valid === false ? 'text-red-400' : 'text-green-400';
    case 'PAYMENT_SENDING':
    case 'PAYMENT_SENT':
      return 'text-yellow-400';
    case 'ANALYST_PROCESSING':
      return 'text-cyan-400';
    case 'ANALYST_RESPONSE':
      return 'text-green-400';
    case 'WORK_VERIFIED':
      return data?.valid === false ? 'text-red-400' : 'text-green-400';
    case 'DATABASE_UPDATED':
      return 'text-emerald-400';
    case 'ERROR':
      return 'text-red-400';
    default:
      return 'text-gray-400';
  }
}

// ============ A2A Task Types ============

export type TaskState = 'submitted' | 'working' | 'completed' | 'failed';

export interface TaskArtifact {
  id: string;
  name: string;
  mimeType: string;
  data?: any;
  createdAt: string;
}

export interface TaskError {
  code: string;
  message: string;
  details?: Record<string, any>;
}

export interface A2ATask {
  id: string;
  contextId?: string;
  skillId: string;
  state: TaskState;
  createdAt: string;
  updatedAt: string;
  completedAt?: string;
  output?: Record<string, any>;
  artifacts?: TaskArtifact[];
  error?: TaskError;
}

export interface TaskListResponse {
  tasks: A2ATask[];
  total: number;
}

// ============ JSON-RPC Types ============

interface JSONRPCRequest {
  jsonrpc: '2.0';
  method: string;
  params?: Record<string, any>;
  id: string;
}

interface JSONRPCResponse<T = any> {
  jsonrpc: '2.0';
  result?: T;
  error?: {
    code: number;
    message: string;
    data?: any;
  };
  id: string;
}

// ============ A2A Task Functions ============

async function jsonRpcCall<T>(url: string, method: string, params?: Record<string, any>): Promise<T> {
  const request: JSONRPCRequest = {
    jsonrpc: '2.0',
    method,
    params,
    id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
  };

  const response = await fetch(`${url}/a2a`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    throw new Error(`HTTP error: ${response.status}`);
  }

  const data: JSONRPCResponse<T> = await response.json();

  if (data.error) {
    throw new Error(data.error.message);
  }

  return data.result as T;
}

// Policy Agent Tasks
export async function getPolicyTask(taskId: string): Promise<A2ATask> {
  return jsonRpcCall<A2ATask>(POLICY_URL, 'task/get', { taskId });
}

export async function listPolicyTasks(limit: number = 100): Promise<TaskListResponse> {
  return jsonRpcCall<TaskListResponse>(POLICY_URL, 'tasks/list', { limit });
}

export async function sendPolicyTask(input: {
  batch_id: string;
  url_count: number;
  estimated_cost_usdc: number;
  budget_remaining_usdc: number;
  source_reputation?: number;
  novelty_score?: number;
  time_since_last_batch_seconds?: number;
  threat_level?: number;
}): Promise<A2ATask> {
  return jsonRpcCall<A2ATask>(POLICY_URL, 'task/send', {
    skillId: 'authorize-batch',
    input,
  });
}

// Analyst Agent Tasks
export async function getAnalystTask(taskId: string): Promise<A2ATask> {
  return jsonRpcCall<A2ATask>(ANALYST_URL, 'task/get', { taskId });
}

export async function listAnalystTasks(limit: number = 100): Promise<TaskListResponse> {
  return jsonRpcCall<TaskListResponse>(ANALYST_URL, 'tasks/list', { limit });
}

export async function sendAnalystTask(
  input: {
    batch_id: string;
    urls: string[];
    policy_proof_hash: string;
  },
  paymentReceipt?: string
): Promise<A2ATask> {
  return jsonRpcCall<A2ATask>(ANALYST_URL, 'task/send', {
    skillId: 'classify-urls',
    input,
    paymentReceipt,
  });
}

// ============ Agent Card Functions ============

export interface AgentCardV3 {
  name: string;
  description: string;
  url: string;
  version: string;
  protocolVersion: string;
  capabilities: {
    streaming: boolean;
    pushNotifications: boolean;
    stateTransitionHistory: boolean;
  };
  skills: Array<{
    id: string;
    name: string;
    description: string;
    tags: string[];
    inputModes: string[];
    outputModes: string[];
    price?: {
      amount: string;
      currency: string;
      per: string;
      chain: string;
    };
  }>;
  provider: string;
  documentationUrl?: string;
  defaultPaymentAddress?: string;
  supportedPaymentMethods: string[];
}

export async function getAgentCard(agentUrl: string): Promise<AgentCardV3> {
  const response = await fetch(`${agentUrl}/.well-known/agent.json`);
  if (!response.ok) {
    throw new Error(`Failed to fetch agent card: ${response.status}`);
  }
  return response.json();
}

export async function getScoutCard(): Promise<AgentCardV3> {
  return getAgentCard(SCOUT_URL);
}

export async function getPolicyCard(): Promise<AgentCardV3> {
  return getAgentCard(POLICY_URL);
}

export async function getAnalystCard(): Promise<AgentCardV3> {
  return getAgentCard(ANALYST_URL);
}

// ============ SSE Streaming ============

export interface TaskSSEEvent {
  type: 'task/status' | 'task/artifact' | 'task/complete' | 'task/error';
  data: A2ATask | { taskId: string; state: string } | { taskId: string; error: TaskError };
}

export function streamTask(
  agentUrl: string,
  taskId: string,
  onEvent: (event: TaskSSEEvent) => void,
  onError?: (error: Error) => void
): () => void {
  const eventSource = new EventSource(`${agentUrl}/tasks/${taskId}/stream`);

  eventSource.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      onEvent({ type: event.type as TaskSSEEvent['type'], data });
    } catch (e) {
      // Ignore parse errors (heartbeats, etc.)
    }
  };

  eventSource.addEventListener('task/status', (event: MessageEvent) => {
    onEvent({ type: 'task/status', data: JSON.parse(event.data) });
  });

  eventSource.addEventListener('task/artifact', (event: MessageEvent) => {
    onEvent({ type: 'task/artifact', data: JSON.parse(event.data) });
  });

  eventSource.addEventListener('task/complete', (event: MessageEvent) => {
    onEvent({ type: 'task/complete', data: JSON.parse(event.data) });
    eventSource.close();
  });

  eventSource.addEventListener('task/error', (event: MessageEvent) => {
    onEvent({ type: 'task/error', data: JSON.parse(event.data) });
    eventSource.close();
  });

  eventSource.onerror = (error) => {
    onError?.(new Error('SSE connection error'));
    eventSource.close();
  };

  // Return cleanup function
  return () => eventSource.close();
}

export function streamPolicyTask(
  taskId: string,
  onEvent: (event: TaskSSEEvent) => void,
  onError?: (error: Error) => void
): () => void {
  return streamTask(POLICY_URL, taskId, onEvent, onError);
}

export function streamAnalystTask(
  taskId: string,
  onEvent: (event: TaskSSEEvent) => void,
  onError?: (error: Error) => void
): () => void {
  return streamTask(ANALYST_URL, taskId, onEvent, onError);
}
