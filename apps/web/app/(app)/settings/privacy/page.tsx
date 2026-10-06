"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { ArrowLeft, Download, Fingerprint, Trash2 } from "lucide-react";
import { ConsentForm } from "@/components/auth/ConsentForm";
import { Alert, Button, Card, Field, Input, Skeleton } from "@/components/ui";
import { api, ApiError } from "@/lib/api";
import { useUser } from "@/lib/hooks/useUser";

export default function Privacy() {
  const { user } = useUser();
  const router = useRouter();
  const [confirm, setConfirm] = useState("");
  const [busy, setBusy] = useState<"export" | "templates" | "account" | null>(null);
  const [notice, setNotice] = useState<{ tone: "success" | "danger"; text: string } | null>(null);

  async function exportData() {
    setBusy("export");
    setNotice(null);
    try {
      const data = await api("/privacy/export");
      const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }));
      const a = document.createElement("a");
      a.href = url;
      a.download = "interview-coach-export.json";
      a.click();
      URL.revokeObjectURL(url);
    } catch (e) {
      setNotice({ tone: "danger", text: e instanceof ApiError ? e.message : "Could not export your data." });
    } finally {
      setBusy(null);
    }
  }

  async function deleteTemplates() {
    setBusy("templates");
    setNotice(null);
    try {
      await api("/enrollment", { method: "DELETE" });
      setNotice({ tone: "success", text: "Face and voice templates deleted." });
    } catch (e) {
      setNotice({ tone: "danger", text: e instanceof ApiError ? e.message : "Could not delete the templates." });
    } finally {
      setBusy(null);
    }
  }

  async function deleteAccount() {
    setBusy("account");
    setNotice(null);
    try {
      await api("/privacy/account", { method: "DELETE" });
      router.push("/");
    } catch (e) {
      setNotice({ tone: "danger", text: e instanceof ApiError ? e.message : "Could not delete your account." });
      setBusy(null);
    }
  }

  return (
    <div className="mx-auto flex max-w-[1100px] flex-col gap-6 px-5 py-8">
      <header>
        <Link href="/settings" className="mb-3 inline-flex items-center gap-1 text-[13px] text-fg-muted hover:text-fg"><ArrowLeft className="size-3.5" aria-hidden /> Settings</Link>
        <p className="text-[13px] font-medium text-accent">Settings</p>
        <h1 className="mt-1 text-[30px] font-semibold leading-9">Privacy and data</h1>
      </header>

      {!user ? (
        <div className="flex max-w-[760px] flex-col gap-4" aria-busy="true">
          <Skeleton className="h-64" />
          <Skeleton className="h-32" />
        </div>
      ) : (
        <div className="flex max-w-[760px] flex-col gap-6">
          <Card as="section" aria-labelledby="perm-heading" className="flex flex-col gap-4 p-5">
            <div>
              <h2 id="perm-heading" className="text-[16px] font-semibold">Permissions</h2>
              <p className="mt-1 text-[13px] text-fg-muted">Choose what we may do with your data. Changes save immediately.</p>
            </div>
            <ConsentForm />
          </Card>

          <Card as="section" aria-labelledby="data-heading" className="flex flex-col gap-4 p-5">
            <div>
              <h2 id="data-heading" className="text-[16px] font-semibold">Your data</h2>
              <p className="mt-1 text-[13px] text-fg-muted">Download everything we hold about you as JSON, or remove your identity-check templates.</p>
            </div>
            <div className="flex flex-col gap-3 sm:flex-row sm:flex-wrap">
              <Button variant="secondary" onClick={() => void exportData()} loading={busy === "export"} disabled={busy !== null}>
                {busy !== "export" && <Download className="size-4" aria-hidden />} Download all my data
              </Button>
              <Button variant="secondary" onClick={() => void deleteTemplates()} loading={busy === "templates"} disabled={busy !== null}>
                {busy !== "templates" && <Fingerprint className="size-4" aria-hidden />} Delete face and voice templates
              </Button>
            </div>
          </Card>

          {notice && <Alert tone={notice.tone}>{notice.text}</Alert>}

          <Card as="section" aria-labelledby="delete-heading" className="flex flex-col gap-4 border-danger/30 p-5">
            <div>
              <h2 id="delete-heading" className="text-[16px] font-semibold text-danger">Delete account</h2>
              <p className="mt-1 text-[13px] text-fg-muted">This permanently deletes your account, sessions, reports and templates. Type DELETE to confirm.</p>
            </div>
            <form
              className="flex flex-col gap-3 sm:flex-row sm:items-end"
              onSubmit={(e) => { e.preventDefault(); if (confirm === "DELETE") void deleteAccount(); }}
            >
              <div className="sm:w-56">
                <Field label="Type DELETE to confirm" htmlFor="delete-confirm">
                  <Input id="delete-confirm" value={confirm} onChange={(e) => setConfirm(e.target.value)} autoComplete="off" spellCheck={false} placeholder="DELETE" />
                </Field>
              </div>
              <Button type="submit" variant="danger" disabled={confirm !== "DELETE" || busy !== null} loading={busy === "account"}>
                {busy !== "account" && <Trash2 className="size-4" aria-hidden />} Delete my account
              </Button>
            </form>
          </Card>
        </div>
      )}
    </div>
  );
}
