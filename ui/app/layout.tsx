import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Threat Intelligence Network',
  description: 'A2A + x402 + zkML Threat Intelligence Demo',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className="min-h-screen bg-[#0a0a0f]">{children}</body>
    </html>
  )
}
