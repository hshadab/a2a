/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  output: 'standalone',
  env: {
    NEXT_PUBLIC_SCOUT_URL: process.env.NEXT_PUBLIC_SCOUT_URL || 'http://localhost:8000',
    NEXT_PUBLIC_POLICY_URL: process.env.NEXT_PUBLIC_POLICY_URL || 'http://localhost:8001',
    NEXT_PUBLIC_ANALYST_URL: process.env.NEXT_PUBLIC_ANALYST_URL || 'http://localhost:8002',
    NEXT_PUBLIC_WS_URL: process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:8000/ws',
  },
  // Allow connections from any origin for WebSocket
  async headers() {
    return [
      {
        source: '/api/:path*',
        headers: [
          { key: 'Access-Control-Allow-Origin', value: '*' },
          { key: 'Access-Control-Allow-Methods', value: 'GET, POST, OPTIONS' },
          { key: 'Access-Control-Allow-Headers', value: 'Content-Type, Authorization' },
        ],
      },
    ]
  },
}

module.exports = nextConfig
