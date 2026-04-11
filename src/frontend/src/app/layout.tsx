import { Activity } from "lucide-react"
import type { Metadata } from "next"
import Link from "next/link"
import { NextIntlClientProvider } from "next-intl"
import { getLocale, getMessages, getTranslations } from "next-intl/server"
import { AppNav } from "@/components/app-nav"
import { Toaster } from "@/components/ui/sonner"
import { Providers } from "./providers"
import "./globals.css"

export async function generateMetadata(): Promise<Metadata> {
  const t = await getTranslations("app")
  return {
    title: t("titleFull"),
    description: "ML-based AI coach for figure skating",
  }
}

export default async function RootLayout({ children }: { children: React.ReactNode }) {
  const locale = await getLocale()
  const messages = await getMessages()
  const t = await getTranslations("app")

  return (
    <html lang={locale} suppressHydrationWarning>
      <body className="min-h-screen bg-background text-foreground">
        <NextIntlClientProvider messages={messages}>
          <Providers>
            <header className="sticky top-0 z-50 border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
              <div className="mx-auto flex h-12 max-w-6xl items-center justify-between px-4">
                <Link href="/" className="flex items-center gap-2 font-semibold">
                  <Activity className="h-5 w-5" />
                  <span className="hidden sm:inline">{t("title")}</span>
                </Link>
                <AppNav />
              </div>
            </header>
            <main className="mx-auto w-full max-w-6xl p-4 sm:p-6">{children}</main>
            <footer className="border-t border-border px-4 py-3 text-center text-xs text-muted-foreground">
              {t("footer")}
            </footer>
            <Toaster richColors position="bottom-right" />
          </Providers>
        </NextIntlClientProvider>
      </body>
    </html>
  )
}
