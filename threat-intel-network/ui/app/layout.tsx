import type { Metadata } from 'next'
import './globals.css'
import Navigation from '@/components/Navigation'

export const metadata: Metadata = {
  title: 'ThreatProof - Verifiable Threat Intelligence',
  description: 'Autonomous Agent-to-Agent commerce secured by zkML. Agents discover and classify phishing threats with cryptographic proofs of correct work.',
  metadataBase: new URL('https://www.threatproof.ai'),
  openGraph: {
    title: 'ThreatProof - Verifiable Threat Intelligence',
    description: 'Autonomous Agent-to-Agent commerce secured by zkML. Powered by Google A2A, x402, and Jolt Atlas.',
    url: 'https://www.threatproof.ai',
    siteName: 'ThreatProof',
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'ThreatProof - Verifiable Threat Intelligence',
    description: 'Autonomous Agent-to-Agent commerce secured by zkML.',
  },
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
