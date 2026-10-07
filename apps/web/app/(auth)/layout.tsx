import { BrandMark, ThemeToggle } from "@/components/shell";

export default function AuthLayout({ children }: { children: React.ReactNode }) {
  return (
    <div className="relative flex min-h-dvh flex-col">
      <div className="pointer-events-none absolute inset-0 -z-10" style={{ background: "var(--stage-glow)" }} />
      <header className="mx-auto flex h-14 w-full max-w-[1200px] items-center justify-between px-5">
        <BrandMark />
        <ThemeToggle />
      </header>
      <main className="flex flex-1 items-center justify-center px-5 py-10">{children}</main>
    </div>
  );
}
