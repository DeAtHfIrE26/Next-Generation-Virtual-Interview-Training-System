"use client";

import { AppHeader, Footer } from "@/components/shell";
import { useUser } from "@/lib/hooks/useUser";

export default function AppLayout({ children }: { children: React.ReactNode }) {
  const { user } = useUser();
  return (
    <div className="flex min-h-dvh flex-col">
      <AppHeader user={user} />
      <main className="mx-auto w-full max-w-[1200px] flex-1 px-5 py-10">{children}</main>
      <Footer />
    </div>
  );
}
