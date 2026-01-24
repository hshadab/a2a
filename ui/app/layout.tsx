import type { Metadata } from 'next'
import './globals.css'
import Navigation from '@/components/Navigation'

export const metadata: Metadata = {
  title: 'ThreatProof - Verifiable Threat Intelligence',
  description: 'Autonomous threat intelligence with zkML-verified classifications using A2A, x402, and Jolt Atlas',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-[#0a0a0f]">
        <Navigation />
        <main className="max-w-7xl mx-auto">
          {children}
        </main>
      </body>
    </html>
  )
}
