'use client';

import { useState, useEffect } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Activity, Info, Shield, Fuel, Wallet, ExternalLink, Cpu } from 'lucide-react';

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

  const estimatedBatches = balance ? Math.floor(balance.eth / 0.000001) : 0;
  const shortAddress = `${TREASURY_ADDRESS.slice(0, 6)}...${TREASURY_ADDRESS.slice(-4)}`;

  return (
    <nav className="border-b border-gray-800 bg-gray-900/80 backdrop-blur-sm sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4">
        {/* Top Row: Logo, Title, Status */}
        <div className="flex items-center justify-between h-12 border-b border-gray-800/50">
          {/* Logo & Title */}
          <div className="flex items-center gap-3">
            <Link href="/" className="flex items-center gap-2">
              <Shield className="text-cyan-400" size={22} />
              <span className="text-lg font-bold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
                ThreatProof
              </span>
            </Link>
            <span className="text-gray-600">|</span>
            <span className="text-xs text-gray-400">
              <span className="text-cyan-400">Google A2A</span>
              {' + '}
              <span className="text-green-400">x402</span>
              {' + '}
              <a
                href="https://github.com/ICME-Lab/jolt-atlas"
                target="_blank"
                rel="noopener noreferrer"
                className="text-purple-400 hover:text-purple-300"
              >
                Jolt Atlas zkML
              </a>
            </span>
          </div>

          {/* Status Indicators */}
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
              <span className="text-xs text-green-400">Live</span>
            </div>
            <div className="flex items-center gap-1.5 px-2 py-0.5 bg-cyan-500/10 border border-cyan-500/30 rounded">
              <Cpu size={12} className="text-cyan-400" />
              <span className="text-xs text-cyan-400">Autonomous</span>
            </div>
            {/* Tabs */}
            <div className="flex items-center gap-1 ml-2">
              {tabs.map((tab) => {
                const isActive = pathname === tab.href;
                const Icon = tab.icon;
                return (
                  <Link
                    key={tab.href}
                    href={tab.href}
                    className={`flex items-center gap-1.5 px-3 py-1 rounded text-xs font-medium transition-all ${
                      isActive
                        ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30'
                        : 'text-gray-400 hover:text-white hover:bg-gray-800'
                    }`}
                  >
                    <Icon size={14} />
                    {tab.label}
                  </Link>
                );
              })}
            </div>
          </div>
        </div>

        {/* Bottom Row: Treasury Stats */}
        <div className="flex items-center justify-between h-10">
          <div className="flex items-center gap-4">
            {/* USDC Balance */}
            <div className="flex items-center gap-2">
              <span className="text-sm">💵</span>
              <div className="flex items-center gap-1">
                <span className="text-[10px] text-gray-500 uppercase">Treasury</span>
                <span className="text-sm text-green-400 font-mono font-bold">
                  {loading ? '...' : `${balance?.usdcFormatted} USDC`}
                </span>
              </div>
            </div>

            <span className="text-gray-700">|</span>

            {/* ETH Balance */}
            <div className="flex items-center gap-2">
              <Fuel size={14} className="text-blue-400" />
              <div className="flex items-center gap-1">
                <span className="text-[10px] text-gray-500 uppercase">Gas</span>
                <span className="text-sm text-blue-400 font-mono font-bold">
                  {loading ? '...' : `${balance?.ethFormatted} ETH`}
                </span>
              </div>
            </div>

            <span className="text-gray-700">|</span>

            {/* Runway */}
            <div className="flex items-center gap-1">
              <span className="text-[10px] text-gray-500 uppercase">Runway</span>
              <span className="text-sm text-purple-400 font-mono font-bold">
                {loading ? '...' : `~${estimatedBatches.toLocaleString()} batches`}
              </span>
            </div>

            <span className="text-gray-700">|</span>

            {/* Basescan Link */}
            <a
              href={`https://basescan.org/address/${TREASURY_ADDRESS}`}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1 text-xs text-gray-500 hover:text-gray-300"
            >
              <Wallet size={12} />
              <span className="font-mono">{shortAddress}</span>
              <ExternalLink size={10} />
            </a>
          </div>

          {/* Economy info */}
          <div className="flex items-center gap-2 text-[10px] text-gray-500">
            <span className="flex items-center gap-1">
              <span className="w-1.5 h-1.5 rounded-full bg-green-500" />
              Self-sustaining
            </span>
            <span className="text-gray-700">|</span>
            <span>Batches every 5 min</span>
            <span className="text-gray-700">|</span>
            <span>USDC circulates internally, only gas consumed</span>
          </div>
        </div>
      </div>
    </nav>
  );
}
