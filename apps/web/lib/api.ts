// Same-origin API client (app/api/[...path] proxies to the FastAPI service).

export class ApiError extends Error {
  constructor(public status: number, message: string, public detail?: unknown) {
    super(message);
  }
}

function messageOf(detail: unknown, fallback: string): string {
  if (typeof detail === "string") return detail;
  if (detail && typeof detail === "object" && "message" in detail) return String((detail as { message: unknown }).message);
  if (Array.isArray(detail) && detail[0] && typeof detail[0] === "object" && "msg" in detail[0]) return String(detail[0].msg);
  return fallback;
}

export async function api<T = unknown>(path: string, init: RequestInit & { json?: unknown } = {}): Promise<T> {
  const headers = new Headers(init.headers);
  headers.set("x-ic-csrf", "1");
  let body = init.body;
  if (init.json !== undefined) {
    headers.set("content-type", "application/json");
    body = JSON.stringify(init.json);
  }
  const res = await fetch(`/api${path}`, { ...init, body, headers, credentials: "include", cache: "no-store" });
  const text = await res.text();
  const data = text ? JSON.parse(text) : null;
  if (!res.ok) throw new ApiError(res.status, messageOf(data?.detail, res.statusText), data?.detail);
  return data as T;
}

export type User = { id: string; email: string; name: string; plan: string; role: string };
export type Capabilities = {
  llm: string | null; face_verification: boolean; voice_verification: boolean;
  server_asr: string | null; server_tts: string | null; code_execution: boolean; neural_avatar: boolean;
};
