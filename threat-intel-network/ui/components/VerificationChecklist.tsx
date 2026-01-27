'use client';

import { useEffect, useState } from 'react';
import { CheckCircle, XCircle, Loader2, Shield } from 'lucide-react';

interface VerificationCheck {
  name: string;
  description: string;
  status: 'pending' | 'checking' | 'passed' | 'failed';
  detail?: string;
}

interface VerificationChecklistProps {
  checks: VerificationCheck[];
  isVerifying?: boolean;
  verifyTimeMs?: number;
  onComplete?: (allPassed: boolean) => void;
}

export default function VerificationChecklist({
  checks,
  isVerifying = false,
  verifyTimeMs,
  onComplete,
}: VerificationChecklistProps) {
  const [animatedChecks, setAnimatedChecks] = useState<VerificationCheck[]>([]);

  // Animate checks appearing one by one
  useEffect(() => {
    if (checks.length === 0) {
      setAnimatedChecks([]);
      return;
    }

    setAnimatedChecks([]);
    let index = 0;

    const interval = setInterval(() => {
      if (index < checks.length) {
        setAnimatedChecks((prev) => [...prev, checks[index]]);
        index++;
      } else {
        clearInterval(interval);
        const allPassed = checks.every((c) => c.status === 'passed');
        onComplete?.(allPassed);
      }
    }, 150);

    return () => clearInterval(interval);
  }, [checks, onComplete]);

  const allPassed = checks.length > 0 && checks.every((c) => c.status === 'passed');
  const anyFailed = checks.some((c) => c.status === 'failed');

  return (
    <div className="rounded-lg border border-gray-800 bg-gray-900/30 p-4">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Shield size={16} className={allPassed ? 'text-green-400' : anyFailed ? 'text-red-400' : 'text-cyan-400'} />
          <span className="text-sm font-medium text-white">Verification Checklist</span>
        </div>
        {isVerifying && (
          <span className="text-xs text-cyan-400 flex items-center gap-1">
            <Loader2 size={10} className="animate-spin" />
            Verifying...
          </span>
        )}
        {!isVerifying && verifyTimeMs !== undefined && (
          <span className="text-xs text-gray-500">{verifyTimeMs}ms</span>
        )}
      </div>

      {/* Checks */}
      <div className="space-y-2">
        {animatedChecks.map((check, i) => (
          <CheckRow key={i} check={check} index={i} />
        ))}
      </div>

      {/* Summary */}
      {!isVerifying && checks.length > 0 && (
        <div
          className={`mt-4 p-3 rounded-lg flex items-center justify-between ${
            allPassed
              ? 'bg-green-500/10 border border-green-500/30'
              : anyFailed
              ? 'bg-red-500/10 border border-red-500/30'
              : 'bg-gray-800/50'
          }`}
        >
          <div className="flex items-center gap-2">
            {allPassed ? (
              <>
                <CheckCircle size={18} className="text-green-400" />
                <span className="text-green-400 font-medium">PROOF VALID</span>
              </>
            ) : anyFailed ? (
              <>
                <XCircle size={18} className="text-red-400" />
                <span className="text-red-400 font-medium">VERIFICATION FAILED</span>
              </>
            ) : (
              <>
                <Loader2 size={18} className="text-gray-400 animate-spin" />
                <span className="text-gray-400">Pending...</span>
              </>
            )}
          </div>
          {verifyTimeMs !== undefined && (
            <span className="text-xs text-gray-500">
              Verified in {verifyTimeMs}ms
            </span>
          )}
        </div>
      )}
    </div>
  );
}

function CheckRow({ check, index }: { check: VerificationCheck; index: number }) {
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => setVisible(true), 50);
    return () => clearTimeout(timer);
  }, []);

  const getStatusIcon = () => {
    switch (check.status) {
      case 'passed':
        return <CheckCircle size={14} className="text-green-400" />;
      case 'failed':
        return <XCircle size={14} className="text-red-400" />;
      case 'checking':
        return <Loader2 size={14} className="text-cyan-400 animate-spin" />;
      default:
        return <div className="w-3.5 h-3.5 rounded-full border border-gray-600" />;
    }
  };

  const getStatusColor = () => {
    switch (check.status) {
      case 'passed':
        return 'text-green-400';
      case 'failed':
        return 'text-red-400';
      case 'checking':
        return 'text-cyan-400';
      default:
        return 'text-gray-500';
    }
  };

  return (
    <div
      className={`flex items-start gap-3 p-2 rounded transition-all duration-300 ${
        visible ? 'opacity-100 translate-x-0' : 'opacity-0 -translate-x-4'
      } ${check.status === 'passed' ? 'bg-green-500/5' : check.status === 'failed' ? 'bg-red-500/5' : ''}`}
      style={{ transitionDelay: `${index * 50}ms` }}
    >
      <div className="mt-0.5">{getStatusIcon()}</div>
      <div className="flex-1 min-w-0">
        <div className="flex items-center justify-between">
          <span className={`text-sm font-medium ${getStatusColor()}`}>{check.name}</span>
          {check.status === 'passed' && (
            <CheckCircle size={12} className="text-green-600" />
          )}
        </div>
        <p className="text-xs text-gray-500">{check.description}</p>
        {check.detail && (
          <code className="text-[10px] text-gray-600 font-mono mt-1 block truncate">
            {check.detail}
          </code>
        )}
      </div>
    </div>
  );
}

// Pre-built verification checks for common scenarios
export const standardVerificationChecks: VerificationCheck[] = [
  {
    name: 'Proof structure',
    description: 'Verify proof format and size',
    status: 'pending',
  },
  {
    name: 'Model commitment',
    description: 'Verify model hash matches',
    status: 'pending',
  },
  {
    name: 'Input commitment',
    description: 'Verify input hash matches',
    status: 'pending',
  },
  {
    name: 'Output commitment',
    description: 'Verify output hash matches',
    status: 'pending',
  },
  {
    name: 'Cryptographic binding',
    description: 'Verify SNARK proof',
    status: 'pending',
  },
];
