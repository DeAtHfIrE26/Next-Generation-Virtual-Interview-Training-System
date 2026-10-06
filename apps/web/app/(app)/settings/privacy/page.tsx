"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { ConsentForm } from "@/components/ConsentForm";
import { api } from "@/lib/api";
import { useUser } from "@/lib/hooks/useUser";

export default function Privacy() {
  const { user } = useUser();
  const router = useRouter();
  const [confirm, setConfirm] = useState("");
  if (!user) return null;
  async function exportData() {
    const data = await api("/privacy/export");
    const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = "interview-coach-export.json";
    a.click();
    URL.revokeObjectURL(url);
  }
  return (
    <div className="stack" style={{ maxWidth: 760 }}>
      <h1>Privacy and data</h1>
      <h2>Permissions</h2>
      <ConsentForm />
      <h2>Your data</h2>
      <div className="row">
        <button className="btn" onClick={exportData}>Download all my data</button>
        <button className="btn" onClick={async () => { await api("/enrollment", { method: "DELETE" }); alert("Face and voice templates deleted."); }}>Delete face and voice templates</button>
      </div>
      <h2>Delete account</h2>
      <p className="muted">This permanently deletes your account, sessions, reports and templates. Type DELETE to confirm.</p>
      <div className="row">
        <input aria-label="Type DELETE to confirm" value={confirm} onChange={(e) => setConfirm(e.target.value)} style={{ maxWidth: 200 }} />
        <button className="btn danger" disabled={confirm !== "DELETE"} onClick={async () => { await api("/privacy/account", { method: "DELETE" }); router.push("/"); }}>Delete my account</button>
      </div>
    </div>
  );
}
