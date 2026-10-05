"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { api, ApiError } from "@/lib/api";

export function AuthForm({ mode }: { mode: "login" | "signup" }) {
  const router = useRouter();
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  async function submit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const f = new FormData(e.currentTarget);
    setBusy(true);
    setError("");
    try {
      if (mode === "signup") {
        await api("/auth/register", { method: "POST", json: {
          email: f.get("email"), password: f.get("password"), name: f.get("name") ?? "", accept_terms: f.get("terms") === "on", age_confirmed: f.get("adult") === "on" } });
        router.push("/onboarding");
      } else {
        await api("/auth/login", { method: "POST", json: { email: f.get("email"), password: f.get("password") } });
        router.push("/dashboard");
      }
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Something went wrong. Please try again.");
    } finally {
      setBusy(false);
    }
  }
  return (
    <form className="card stack" style={{ maxWidth: 420 }} onSubmit={submit}>
      <h1>{mode === "signup" ? "Create your account" : "Sign in"}</h1>
      {mode === "signup" && <label className="field"><span>Name</span><input name="name" autoComplete="name" /></label>}
      <label className="field"><span>Email</span><input name="email" type="email" required autoComplete="email" /></label>
      <label className="field"><span>Password (10+ characters)</span>
        <input name="password" type="password" required minLength={10} autoComplete={mode === "signup" ? "new-password" : "current-password"} /></label>
      {mode === "signup" && (
        <label className="check"><input name="adult" type="checkbox" required />
          <span>I am 18 or older.</span></label>
      )}
      {mode === "signup" && (
        <label className="check"><input name="terms" type="checkbox" required />
          <span>I agree to the <Link href="/legal/terms">terms</Link> and have read the <Link href="/legal/privacy">privacy notice</Link>.</span></label>
      )}
      {error && <p className="notice error" role="alert">{error}</p>}
      <button className="btn primary" disabled={busy}>{mode === "signup" ? "Create account" : "Sign in"}</button>
      <p className="small muted">{mode === "signup" ? <>Already have an account? <Link href="/login">Sign in</Link></> : <>New here? <Link href="/signup">Create an account</Link></>}</p>
    </form>
  );
}
