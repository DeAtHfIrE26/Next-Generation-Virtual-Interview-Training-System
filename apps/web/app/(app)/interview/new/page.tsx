"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { api, ApiError } from "@/lib/api";
import { useUser } from "@/lib/hooks/useUser";

export default function NewInterview() {
  const { user } = useUser();
  const router = useRouter();
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  if (!user) return null;
  async function submit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    setBusy(true);
    setError("");
    const form = new FormData(e.currentTarget);
    const file = form.get("resume");
    if (file instanceof File && file.size === 0) form.delete("resume");
    try {
      const r = await api<{ id: string }>("/sessions", { method: "POST", body: form });
      router.push(`/interview/${r.id}`);
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not start the session.");
      if (err instanceof ApiError && err.status === 403) router.push("/onboarding");
    } finally {
      setBusy(false);
    }
  }
  return (
    <form className="card stack" style={{ maxWidth: 720 }} onSubmit={submit}>
      <h1>New practice interview</h1>
      <div className="grid two">
        <label className="field"><span>Target role</span><input name="role" required minLength={2} maxLength={120} placeholder="e.g. Backend Software Engineer" /></label>
        <label className="field"><span>Seniority</span>
          <select name="seniority" defaultValue="mid">
            <option value="intern">Intern</option><option value="junior">Junior</option><option value="mid">Mid-level</option>
            <option value="senior">Senior</option><option value="lead">Lead / manager</option>
          </select></label>
      </div>
      <label className="field"><span>Job description (optional, improves question relevance)</span>
        <textarea name="job_description" rows={5} maxLength={6000} /></label>
      <label className="field"><span>Resume PDF (optional, max 2 MB). Contact details are removed before any AI processing.</span>
        <input name="resume" type="file" accept="application/pdf" /></label>
      <div className="grid two">
        <label className="field"><span>Length</span>
          <select name="length" defaultValue="8"><option value="5">Short (5 questions)</option><option value="8">Standard (8)</option><option value="12">Long (12)</option></select></label>
        <label className="field"><span>Mode</span>
          <select name="mode" defaultValue="coaching">
            <option value="coaching">Coaching: notices only</option>
            <option value="proctored">Strict practice: repeated integrity issues end the session</option>
          </select></label>
      </div>
      {error && <p className="notice error" role="alert">{error}</p>}
      <button className="btn primary" disabled={busy}>Start interview</button>
    </form>
  );
}
