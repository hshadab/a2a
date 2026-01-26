'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Activity, Info, Cpu, History } from 'lucide-react';

export default function Navigation() {
  const pathname = usePathname();

  const tabs = [
    { href: '/', label: 'Dashboard', icon: Activity },
    { href: '/history', label: 'History', icon: History },
    { href: '/about', label: 'About', icon: Info },
  ];

  return (
    <nav className="border-b border-[#2a2a2a] bg-[#0a0a0a] sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-6">
        {/* Top Row: Logo, Title, Status */}
        <div className="flex items-center justify-between h-20 border-b border-gray-800/50">
          {/* Logo & Title */}
          <div className="flex items-center gap-4">
            <Link href="/" className="flex items-center gap-3">
              {/* Novanet Logo - 2/3 size */}
              <img
                src="https://cdn.prod.website-files.com/65d52b07d5bc41614daa723f/665df12739c532f45b665fe7_logo-novanet.svg"
                alt="Novanet"
                className="h-5 w-auto"
              />
              <span className="text-2xl font-bold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
                ThreatProof
              </span>
            </Link>
            <div className="flex items-center gap-2 text-sm">
              <span className="text-cyan-400">Google A2A</span>
              <span className="text-green-400">x402</span>
              <a
                href="https://github.com/ICME-Lab/jolt-atlas"
                target="_blank"
                rel="noopener noreferrer"
                className="text-purple-400 hover:text-purple-300"
              >
                Jolt Atlas zkML
              </a>
            </div>
          </div>

          {/* Status Indicators */}
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-green-500 animate-pulse" />
              <span className="text-sm text-green-400 font-medium">Live</span>
            </div>
            <div className="flex items-center gap-2 px-3 py-1 bg-cyan-500/10 border border-cyan-500/30 rounded-lg">
              <Cpu size={14} className="text-cyan-400" />
              <span className="text-sm text-cyan-400 font-medium">Autonomous</span>
            </div>
            {/* Tabs */}
            <div className="flex items-center gap-2 ml-3">
              {tabs.map((tab) => {
                const isActive = pathname === tab.href;
                const Icon = tab.icon;
                return (
                  <Link
                    key={tab.href}
                    href={tab.href}
                    className={`flex items-center gap-2 px-4 py-1.5 rounded-lg text-sm font-medium transition-all ${
                      isActive
                        ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30'
                        : 'text-gray-400 hover:text-white hover:bg-gray-800'
                    }`}
                  >
                    <Icon size={16} />
                    {tab.label}
                  </Link>
                );
              })}
            </div>
          </div>
        </div>

        {/* Bottom Row: Tagline */}
        <div className="flex items-center justify-center h-10">
          <div className="flex items-center gap-4 text-xs text-gray-400">
            <span className="text-cyan-400">Autonomous Agent-to-Agent commerce</span>
            <span className="text-gray-500">•</span>
            <span>zkML proofs verify correct work and spending guardrails</span>
          </div>
        </div>
      </div>
    </nav>
  );
}
