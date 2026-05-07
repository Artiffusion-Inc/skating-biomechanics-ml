import { NextResponse } from "next/server"
import type { NextRequest } from "next/server"

const cspDirectives: Record<string, string[]> = {
  "default-src": ["'self'"],
  "script-src": ["'self'", "'strict-dynamic'", "'unsafe-inline'", "https://cdn.jsdelivr.net"],
  "style-src": ["'self'", "'unsafe-inline'"],
  "img-src": ["'self'", "data:", "blob:"],
  "media-src": ["'self'", "blob:"],
  "connect-src": ["'self'", "blob:", "https://*.r2.cloudflarestorage.com", "http://localhost:8000"],
  "font-src": ["'self'"],
  "object-src": ["'none'"],
  "frame-ancestors": ["'none'"],
  "base-uri": ["'self'"],
  "form-action": ["'self'"],
  "worker-src": ["'self'", "blob:"],
}

function buildCsp(nonce: string, isDev: boolean): string {
  const directives = { ...cspDirectives }

  // Add nonce to script-src
  directives["script-src"] = [`'nonce-${nonce}'`, ...directives["script-src"]]

  // Dev: allow eval for React HMR
  if (isDev) {
    directives["script-src"] = [...directives["script-src"], "'unsafe-eval'"]
    // Dev: allow Vite HMR websocket
    directives["connect-src"] = [...directives["connect-src"], "ws://localhost:*"]
  }

  return Object.entries(directives)
    .map(([key, values]) => `${key} ${values.join(" ")}`)
    .join("; ")
}

export function proxy(_request: NextRequest) {
  const nonce = crypto.randomUUID().replace(/-/g, "")
  const isDev = process.env.NODE_ENV === "development"
  const csp = buildCsp(nonce, isDev)

  const response = NextResponse.next()

  response.headers.set("Content-Security-Policy", csp)
  response.headers.set("X-Nonce", nonce)

  return response
}

export const config = {
  // Skip static assets and Next.js internals — they don't need CSP nonce
  matcher: [
    {
      source: "/((?!_next/static|_next/image|favicon.ico|sitemap.xml|robots.txt).*)",
    },
  ],
}
