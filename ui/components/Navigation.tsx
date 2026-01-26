'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Activity, Info, Shield, Fuel, Wallet, ExternalLink, Cpu, History, Banknote } from 'lucide-react';

interface Balance {
  usdc: number;
  eth: number;
  usdcFormatted: string;
  ethFormatted: string;
}

const TREASURY_ADDRESS = '0x6c67DBBa573326318CdE33dDa4e6D3b34f8dC303';

export default function Navigation() {
  const pathname = usePathname();
  const [balance, setBalance] = useState<Balance | null>(null);
  const [loading, setLoading] = useState(true);

  const tabs = [
    { href: '/', label: 'Dashboard', icon: Activity },
    { href: '/history', label: 'History', icon: History },
    { href: '/about', label: 'About', icon: Info },
  ];

  // Fetch treasury balance
  useEffect(() => {
    const fetchBalance = async () => {
      try {
        const ethResponse = await fetch('https://mainnet.base.org', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            jsonrpc: '2.0',
            method: 'eth_getBalance',
            params: [TREASURY_ADDRESS, 'latest'],
            id: 1,
          }),
        });
        const ethData = await ethResponse.json();
        const ethWei = parseInt(ethData.result, 16);
        const ethBalance = ethWei / 1e18;

        const usdcContract = '0x833589fCD6eDb6E08f4c7C32D4f71b54bdA02913';
        const balanceOfSelector = '0x70a08231';
        const paddedAddress = TREASURY_ADDRESS.slice(2).padStart(64, '0');

        const usdcResponse = await fetch('https://mainnet.base.org', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            jsonrpc: '2.0',
            method: 'eth_call',
            params: [{ to: usdcContract, data: balanceOfSelector + paddedAddress }, 'latest'],
            id: 2,
          }),
        });
        const usdcData = await usdcResponse.json();
        const usdcRaw = parseInt(usdcData.result, 16);
        const usdcBalance = usdcRaw / 1e6;

        setBalance({
          usdc: usdcBalance,
          eth: ethBalance,
          usdcFormatted: usdcBalance.toFixed(4),
          ethFormatted: ethBalance.toFixed(6),
        });
      } catch (err) {
        setBalance({ usdc: 1.0, eth: 0.001, usdcFormatted: '1.0000', ethFormatted: '0.001000' });
      } finally {
        setLoading(false);
      }
    };

    fetchBalance();
    const interval = setInterval(fetchBalance, 30000);
    return () => clearInterval(interval);
  }, []);

  const shortAddress = `${TREASURY_ADDRESS.slice(0, 6)}...${TREASURY_ADDRESS.slice(-4)}`;

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

        {/* Bottom Row: Treasury Stats */}
        <div className="flex items-center justify-between h-14">
          <div className="flex items-center gap-5">
            {/* USDC Balance */}
            <div className="flex items-center gap-2">
              <Banknote size={18} className="text-green-400" />
              <div className="flex items-center gap-1.5">
                <span className="text-xs text-gray-500 uppercase">Treasury</span>
                <span className="text-base text-green-400 font-mono font-bold">
                  {loading ? '...' : `${balance?.usdcFormatted} USDC`}
                </span>
              </div>
            </div>

            {/* ETH Balance */}
            <div className="flex items-center gap-2">
              <Fuel size={18} className="text-blue-400" />
              <div className="flex items-center gap-1.5">
                <span className="text-xs text-gray-500 uppercase">Gas</span>
                <span className="text-base text-blue-400 font-mono font-bold">
                  {loading ? '...' : `${balance?.ethFormatted} ETH`}
                </span>
              </div>
            </div>

            {/* Basescan Link */}
            <a
              href={`https://basescan.org/address/${TREASURY_ADDRESS}`}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1.5 text-sm text-gray-500 hover:text-gray-300"
            >
              <Wallet size={14} />
              <span className="font-mono">{shortAddress}</span>
              <ExternalLink size={12} />
            </a>
          </div>

          {/* Economy info */}
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
