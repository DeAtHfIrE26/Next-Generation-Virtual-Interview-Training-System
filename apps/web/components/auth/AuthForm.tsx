"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { Alert, Button, Checkbox, Field, Input } from "@/components/ui";
import { api, ApiError } from "@/lib/api";

export function AuthForm({ mode }: { mode: "login" | "signup" }) {
  const router = useRouter();
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const signup = mode === "signup";

  async function submit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const f = new FormData(e.currentTarget);
    setBusy(true);
    setError("");
    try {
      if (signup) {
        await api("/auth/register", {
          method: "POST",
          json: { email: f.get("email"), password: f.get("password"), name: f.get("name") ?? "", accept_terms: f.get("terms") === "on", age_confirmed: f.get("adult") === "on" },
        });
        router.push("/onboarding");
      } else {
        await api("/auth/login", { method: "POST", json: { email: f.get("email"), password: f.get("password") } });
        router.push("/dashboard");
      }
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Something went wrong. Please try again.");
      setBusy(false);
    }
  }

  return (
    <form onSubmit={submit} className="flex w-full max-w-[400px] flex-col gap-5">
      <div>
        <h1 className="text-[28px] font-semibold leading-9">{signup ? "Create your account" : "Welcome back"}</h1>
        <p className="mt-1 text-[14px] text-fg-muted">{signup ? "Free to start. Your first interview takes about two minutes to set up." : "Sign in to continue practising."}</p>
      </div>
      {signup && <Field label="Name" htmlFor="name" optional><Input id="name" name="name" autoComplete="name" /></Field>}
      <Field label="Email" htmlFor="email"><Input id="email" name="email" type="email" required autoComplete="email" /></Field>
      <Field label="Password" htmlFor="password" hint={signup ? "At least 10 characters." : undefined}>
        <Input id="password" name="password" type="password" required minLength={10} autoComplete={signup ? "new-password" : "current-password"} />
      </Field>
      {signup && (
        <div className="flex flex-col gap-3">
          <Checkbox name="adult" required label="I am 18 or older." />
          <Checkbox name="terms" required label={<>I agree to the <Link href="/legal/terms" className="underline">terms</Link> and have read the <Link href="/legal/privacy" className="underline">privacy notice</Link>.</>} />
        </div>
      )}
      {error && <Alert tone="danger">{error}</Alert>}
      <Button type="submit" size="lg" loading={busy}>{signup ? "Create account" : "Sign in"}</Button>
      <p className="text-center text-[13px] text-fg-muted">
        {signup ? <>Already have an account? <Link href="/login" className="font-medium text-fg hover:text-accent">Sign in</Link></> : <>New here? <Link href="/signup" className="font-medium text-fg hover:text-accent">Create an account</Link></>}
      </p>
    </form>
  );
}
