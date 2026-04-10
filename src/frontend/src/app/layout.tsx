import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import type { Metadata } from "next"
import "./globals.css"

export const metadata: Metadata = {
  title: "AI Тренер — Фигурное катание",
  description: "ML-based AI coach for figure skating",
}

const queryClient = new QueryClient()

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ru" suppressHydrationWarning>
      <body className="min-h-screen bg-background text-foreground">
        <QueryClientProvider client={queryClient}>
          <header className="border-b border-border px-6 py-3">
            <h1 className="text-lg font-semibold">AI Тренер — Фигурное катание</h1>
          </header>
          <main>{children}</main>
        </QueryClientProvider>
      </body>
    </html>
  )
}
