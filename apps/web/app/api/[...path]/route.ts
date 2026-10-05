// Same-origin proxy to the FastAPI service. Resolved at request time (API_BASE_URL), so one
// build works in every environment, and auth cookies stay first-party.
import type { NextRequest } from "next/server";

export const dynamic = "force-dynamic";

const FORWARD_REQ = ["cookie", "content-type", "x-ic-csrf", "user-agent", "accept"];
const FORWARD_RES = ["content-type", "set-cookie", "retry-after", "cache-control"];

async function proxy(req: NextRequest, ctx: { params: Promise<{ path: string[] }> }) {
  const { path } = await ctx.params;
  const base = (process.env.API_BASE_URL ?? "http://localhost:8000").replace(/\/$/, "");
  const url = `${base}/${path.map(encodeURIComponent).join("/")}${req.nextUrl.search}`;
  const headers = new Headers();
  for (const h of FORWARD_REQ) {
    const v = req.headers.get(h);
    if (v) headers.set(h, v);
  }
  const fwd = req.headers.get("x-forwarded-for");
  headers.set("x-forwarded-for", fwd ?? "unknown");
  const hasBody = !["GET", "HEAD"].includes(req.method);
  const body = hasBody ? await req.arrayBuffer() : undefined;
  if (body) headers.set("content-length", String(body.byteLength));
  let upstream: Response;
  try {
    upstream = await fetch(url, { method: req.method, headers, body, redirect: "manual", cache: "no-store" });
  } catch {
    return Response.json({ detail: "service unavailable" }, { status: 503 });
  }
  const out = new Headers();
  for (const h of FORWARD_RES) {
    if (h === "set-cookie") upstream.headers.getSetCookie().forEach((c) => out.append("set-cookie", c));
    else {
      const v = upstream.headers.get(h);
      if (v) out.set(h, v);
    }
  }
  return new Response(upstream.body, { status: upstream.status, headers: out });
}

export { proxy as GET, proxy as POST, proxy as PUT, proxy as PATCH, proxy as DELETE };
