'use client';

import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Activity, Info, Shield } from 'lucide-react';

export default function Navigation() {
  const pathname = usePathname();

  const tabs = [
    { href: '/', label: 'Dashboard', icon: Activity },
    { href: '/about', label: 'About', icon: Info },
  ];

  return (
    <nav className="border-b border-gray-800 bg-gray-900/50 backdrop-blur-sm sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-6">
        <div className="flex items-center justify-between h-14">
          {/* Logo */}
          <Link href="/" className="flex items-center gap-2 text-white font-bold text-lg">
            <Shield className="text-cyan-400" size={24} />
            <span className="bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
              ThreatProof
            </span>
          </Link>

          {/* Tabs */}
          <div className="flex items-center gap-1">
            {tabs.map((tab) => {
              const isActive = pathname === tab.href;
              const Icon = tab.icon;
              return (
                <Link
                  key={tab.href}
                  href={tab.href}
                  className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
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

          {/* Status indicator */}
          <div className="flex items-center gap-2 text-xs text-gray-500">
            <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
            <span>Base Mainnet</span>
          </div>
        </div>
      </div>
    </nav>
  );
}
