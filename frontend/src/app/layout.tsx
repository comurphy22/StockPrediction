import type { Metadata } from 'next'
import './globals.css'

export const metadata: Metadata = {
  title: 'Stock Prediction | ML-Powered Trading Signals',
  description: 'Stock predictions using politician trading signals, news sentiment, and technical analysis',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <head>
        <link 
          href="https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" 
          rel="stylesheet" 
        />
      </head>
      <body className="min-h-screen bg-midnight-950 bg-grid antialiased">
        <div className="fixed inset-0 bg-gradient-to-br from-midnight-950 via-midnight-900 to-midnight-950 -z-10" />
        <div className="fixed top-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-profit-500/50 to-transparent" />
        {children}
      </body>
    </html>
  )
}

