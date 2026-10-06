// Server-side API access for server components: same upstream and headers as the /api proxy,
// forwarding the signed-in user's cookie. Returns null on any non-2xx so pages can fall back to
// their client-side loading path.
import { cookies, headers } from "next/headers";

export async function serverApi<T>(path: string): Promise<T | null> {
  const base = (process.env.API_BASE_URL ?? "http://localhost:8000").replace(/\/$/, "");
  const h = new Headers({ accept: "application/json" });
  const cookie = (await cookies()).toString();
  if (cookie) h.set("cookie", cookie);
  const shared = process.env.API_SHARED_SECRET;
  if (shared) h.set("x-ic-internal", shared);
  h.set("x-forwarded-for", (await headers()).get("x-forwarded-for") ?? "unknown");
  try {
    const r = await fetch(`${base}${path}`, { headers: h, cache: "no-store" });
    return r.ok ? ((await r.json()) as T) : null;
  } catch {
    return null;
  }
}
