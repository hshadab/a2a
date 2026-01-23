'use client';

import { useState } from 'react';
import { Shield, Clock, Database, Hash, ChevronDown, ChevronUp, CheckCircle, Loader2 } from 'lucide-react';

interface ProofStage {
  name: string;
  message: string;
  progress_pct: number;
  timestamp?: number;
}

interface ProofData {
  proof_hash: string;
  model_commitment: string;
  input_commitment: string;
  output_commitment: string;
  prove_time_ms: number;
  proof_size_bytes: number;
  decision?: string;
  confidence?: number;
  is_real_proof?: boolean;
  stages?: ProofStage[];
}

interface ProofCardProps {
  title: string;
  proof: ProofData | null;
  isGenerating?: boolean;
  currentStage?: ProofStage | null;
  color?: string;
  colorHex?: string;
}

export default function ProofCard({
  title,
  proof,
  isGenerating = false,
  currentStage = null,
  color = 'purple',
  colorHex = '#a855f7',
}: ProofCardProps) {
  const [expanded, setExpanded] = useState(false);

  if (!proof && !isGenerating) {
    return (
      <div className={`rounded-lg border border-gray-800 bg-gray-900/30 p-4`}>
        <div className="flex items-center gap-2 text-gray-500">
          <Shield size={16} />
          <span className="text-sm">{title}</span>
        </div>
        <p className="text-xs text-gray-600 mt-2">No proof generated yet</p>
      </div>
    );
  }

  return (
    <div
      className={`rounded-lg border bg-gray-900/50 p-4 transition-all ${
        isGenerating ? 'border-yellow-500/50 shadow-[0_0_15px_rgba(234,179,8,0.2)]' : `border-${color}-500/30`
      }`}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Shield size={16} style={{ color: isGenerating ? '#eab308' : colorHex }} />
          <span className="text-sm font-medium" style={{ color: isGenerating ? '#eab308' : colorHex }}>
            {title}
          </span>
        </div>
        {proof?.is_real_proof && (
          <span className="text-xs px-2 py-0.5 rounded-full bg-green-500/20 text-green-400">
            REAL PROOF
          </span>
        )}
        {isGenerating && (
          <span className="text-xs px-2 py-0.5 rounded-full bg-yellow-500/20 text-yellow-400 flex items-center gap-1">
            <Loader2 size={10} className="animate-spin" />
            PROVING
          </span>
        )}
      </div>

      {/* Progress Bar (when generating) */}
      {isGenerating && currentStage && (
        <div className="mb-4">
          <div className="flex items-center justify-between text-xs mb-1">
            <span className="text-yellow-400">{currentStage.message}</span>
            <span className="text-gray-500">{currentStage.progress_pct}%</span>
          </div>
          <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-yellow-500 to-yellow-400 transition-all duration-300"
              style={{ width: `${currentStage.progress_pct}%` }}
            />
          </div>
          <div className="flex justify-between text-xs text-gray-600 mt-1">
            <span>Stage: {currentStage.name}</span>
          </div>
        </div>
      )}

      {/* Proof Summary */}
      {proof && (
        <>
          <div className="grid grid-cols-2 gap-3 mb-3">
            <div className="flex items-center gap-2">
              <Clock size={12} className="text-gray-500" />
              <div>
                <p className="text-xs text-gray-500">Prove Time</p>
                <p className="text-sm font-mono text-white">{proof.prove_time_ms}ms</p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Database size={12} className="text-gray-500" />
              <div>
                <p className="text-xs text-gray-500">Proof Size</p>
                <p className="text-sm font-mono text-white">
                  {proof.proof_size_bytes > 1000
                    ? `${(proof.proof_size_bytes / 1000).toFixed(1)}KB`
                    : `${proof.proof_size_bytes}B`}
                </p>
              </div>
            </div>
          </div>

          {/* Decision (if applicable) */}
          {proof.decision && (
            <div className="flex items-center justify-between p-2 rounded bg-gray-800/50 mb-3">
              <span className="text-xs text-gray-400">Decision</span>
              <span
                className={`text-sm font-bold ${
                  proof.decision === 'AUTHORIZED' || proof.decision === 'SAFE'
                    ? 'text-green-400'
                    : proof.decision === 'DENIED' || proof.decision === 'PHISHING'
                    ? 'text-red-400'
                    : 'text-yellow-400'
                }`}
              >
                {proof.decision} ({((proof.confidence || 0) * 100).toFixed(0)}%)
              </span>
            </div>
          )}

          {/* Expandable Commitments */}
          <button
            onClick={() => setExpanded(!expanded)}
            className="flex items-center gap-1 text-xs text-gray-500 hover:text-gray-300 transition-colors"
          >
            {expanded ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
            {expanded ? 'Hide' : 'Show'} Commitments
          </button>

          {expanded && (
            <div className="mt-3 space-y-2 text-xs">
              <CommitmentRow label="Proof Hash" value={proof.proof_hash} icon={<Hash size={10} />} />
              <CommitmentRow label="Model" value={proof.model_commitment} icon={<Database size={10} />} />
              <CommitmentRow label="Input" value={proof.input_commitment} icon={<Hash size={10} />} />
              <CommitmentRow label="Output" value={proof.output_commitment} icon={<Hash size={10} />} />
            </div>
          )}

          {/* Proof Stages (if available) */}
          {expanded && proof.stages && proof.stages.length > 0 && (
            <div className="mt-3 pt-3 border-t border-gray-800">
              <p className="text-xs text-gray-500 mb-2">Proof Generation Stages</p>
              <div className="space-y-1">
                {proof.stages.map((stage, i) => (
                  <div key={i} className="flex items-center gap-2 text-xs">
                    <CheckCircle size={10} className="text-green-500" />
                    <span className="text-gray-400 w-24">{stage.name}</span>
                    <span className="text-gray-600">{stage.message}</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}

function CommitmentRow({
  label,
  value,
  icon,
}: {
  label: string;
  value: string;
  icon: React.ReactNode;
}) {
  return (
    <div className="flex items-center gap-2 p-2 rounded bg-gray-800/30">
      <span className="text-gray-600">{icon}</span>
      <span className="text-gray-500 w-16">{label}</span>
      <code className="text-gray-400 font-mono text-[10px] truncate flex-1">
        {value.slice(0, 16)}...{value.slice(-8)}
      </code>
    </div>
  );
}
