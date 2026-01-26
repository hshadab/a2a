'use client';

import { useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Activity, Info, Cpu, History, Menu, X } from 'lucide-react';

export default function Navigation() {
  const pathname = usePathname();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const tabs = [
    { href: '/', label: 'Dashboard', icon: Activity },
    { href: '/history', label: 'History', icon: History },
    { href: '/about', label: 'About', icon: Info },
  ];

  return (
    <nav className="border-b border-[#2a2a2a] bg-[#0a0a0a] sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 md:px-6">
        {/* Main Row */}
        <div className="flex items-center justify-between h-16 md:h-20">
          {/* Logo & Title */}
          <div className="flex items-center gap-2 md:gap-4">
            <Link href="/" className="flex items-center gap-2 md:gap-3">
              <img
                src="https://cdn.prod.website-files.com/65d52b07d5bc41614daa723f/665df12739c532f45b665fe7_logo-novanet.svg"
                alt="Novanet"
                className="h-4 md:h-5 w-auto"
              />
              <span className="text-xl md:text-2xl font-bold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
                ThreatProof
              </span>
            </Link>
            {/* Protocol badges - hidden on mobile */}
            <div className="hidden lg:flex items-center gap-2 text-sm">
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

          {/* Desktop: Status + Tabs */}
          <div className="hidden md:flex items-center gap-4">
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-green-500 animate-pulse" />
              <span className="text-sm text-green-400 font-medium">Live</span>
            </div>
            <div className="flex items-center gap-2 px-3 py-1 bg-cyan-500/10 border border-cyan-500/30 rounded-lg">
              <Cpu size={14} className="text-cyan-400" />
              <span className="text-sm text-cyan-400 font-medium">Autonomous</span>
            </div>
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

          {/* Mobile: Status + Hamburger */}
          <div className="flex md:hidden items-center gap-3">
            <div className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
              <span className="text-xs text-green-400 font-medium">Live</span>
            </div>
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="p-2 text-gray-400 hover:text-white"
            >
              {mobileMenuOpen ? <X size={24} /> : <Menu size={24} />}
            </button>
          </div>
        </div>

        {/* Mobile Menu */}
        {mobileMenuOpen && (
          <div className="md:hidden border-t border-[#2a2a2a] py-4 space-y-2">
            {tabs.map((tab) => {
              const isActive = pathname === tab.href;
              const Icon = tab.icon;
              return (
                <Link
                  key={tab.href}
                  href={tab.href}
                  onClick={() => setMobileMenuOpen(false)}
                  className={`flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all ${
                    isActive
                      ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30'
                      : 'text-gray-400 hover:text-white hover:bg-gray-800'
                  }`}
                >
                  <Icon size={18} />
                  {tab.label}
                </Link>
              );
            })}
            {/* Protocol badges in mobile menu */}
            <div className="flex items-center gap-3 px-4 pt-3 border-t border-[#2a2a2a] mt-3">
              <span className="text-xs text-cyan-400">A2A</span>
              <span className="text-xs text-green-400">x402</span>
              <span className="text-xs text-purple-400">zkML</span>
            </div>
          </div>
        )}

        {/* Tagline - left aligned, closer to nav */}
        <div className="flex items-center justify-start h-6 md:h-7 -mt-1">
          <div className="flex flex-row items-center gap-2 text-xs md:text-sm text-gray-400">
            <span className="text-cyan-400 font-medium">Autonomous Agent-to-Agent commerce</span>
            <span className="text-gray-600">•</span>
            <span>zkML proofs verify correct work and spending guardrails</span>
          </div>
        </div>
      </div>
    </nav>
  );
}
