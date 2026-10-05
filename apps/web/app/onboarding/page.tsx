"use client";

import { useRouter } from "next/navigation";
import { ConsentForm } from "@/components/ConsentForm";
import { useUser } from "@/lib/hooks/useUser";

export default function Onboarding() {
  const router = useRouter();
  const { user } = useUser();
  if (!user) return null;
  return (
    <div className="stack" style={{ maxWidth: 720 }}>
      <h1>Before you start</h1>
      <p className="lead">Choose what the coach may do. Only the first item is needed to practise.</p>
      <ConsentForm onDone={() => router.push("/dashboard")} />
    </div>
  );
}
