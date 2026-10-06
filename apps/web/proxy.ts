// Runs before every request:
// 1. Optional site-wide password gate for private demos (off unless SITE_PASSWORD is set). HTTP
//    Basic auth is re-sent by the browser for every same-origin request, so no app changes are needed.
// 2. Content-Security-Policy, built at request time so the realtime API origin (PUBLIC_API_URL, which
//    differs per deployment) can be allowed for the WebSocket without rebuilding the app.
import { NextResponse, type NextRequest } from "next/server";

function safeEqual(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}

export function csp(): string {
  const api = (process.env.PUBLIC_API_URL || process.env.API_BASE_URL || "").replace(/\/$/, "");
  const wsOrigin = api ? api.replace(/^http/, "ws") : "";
  const dev = process.env.NODE_ENV !== "production";
  const connect = ["'self'", "blob:", "data:", wsOrigin, dev ? "ws://localhost:* ws://127.0.0.1:*" : ""].filter(Boolean).join(" ");
  return [
    "default-src 'self'",
    // 'unsafe-inline' covers the theme bootstrap and the import map; 'wasm-unsafe-eval' runs the VAD,
    // vision and lip-sync WebAssembly. No third-party script origins are allowed.
    `script-src 'self' 'unsafe-inline' 'wasm-unsafe-eval'${dev ? " 'unsafe-eval'" : ""}`,
    "style-src 'self' 'unsafe-inline'",
    "img-src 'self' data: blob:",
    "media-src 'self' data: blob:",
    "worker-src 'self' blob:",
    `connect-src ${connect}`,
    "font-src 'self'",
    "frame-ancestors 'none'",
    "base-uri 'self'",
    "form-action 'self'",
  ].join("; ");
}

export function proxy(req: NextRequest) {
  const password = process.env.SITE_PASSWORD;
  if (password) {
    const auth = req.headers.get("authorization") ?? "";
    let ok = false;
    if (auth.startsWith("Basic ")) {
      try {
        const decoded = atob(auth.slice(6));
        ok = safeEqual(decoded.slice(decoded.indexOf(":") + 1), password);
      } catch {
        ok = false; // malformed header: challenge again
      }
    }
    if (!ok) {
      return new NextResponse("Authentication required", {
        status: 401,
        headers: { "WWW-Authenticate": 'Basic realm="Interview Coach (private preview)", charset="UTF-8"' },
      });
    }
  }
  const res = NextResponse.next();
  res.headers.set("Content-Security-Policy", csp());
  return res;
}
