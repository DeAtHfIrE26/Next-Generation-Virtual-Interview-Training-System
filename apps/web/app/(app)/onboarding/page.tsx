"use client";

import { useRouter } from "next/navigation";
import { ConsentForm } from "@/components/auth/ConsentForm";

export default function Onboarding() {
  const router = useRouter();
  return (
    <div className="mx-auto flex max-w-[720px] flex-col gap-6 px-5 py-10">
      <header>
        <p className="text-[13px] font-medium text-accent">Step 1 of 2 · Your choices</p>
        <h1 className="mt-1 text-[30px] font-semibold leading-9">Before you start</h1>
        <p className="mt-2 text-[15px] text-fg-muted">Choose what the coach may do. Only the first item is needed to practise; everything else is optional and can be changed later.</p>
      </header>
      <ConsentForm onDone={() => router.push("/interview/new")} />
    </div>
  );
}
