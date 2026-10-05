import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI Interview Coach",
  description: "Practise interviews with an adaptive AI interviewer and get explainable feedback.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <div className="shell">
          <nav className="nav" aria-label="Main">
            <Link className="brand" href="/">AI Interview Coach</Link>
            <Link href="/dashboard">Dashboard</Link>
            <Link href="/interview/new">Practise</Link>
            <Link href="/pricing">Pricing</Link>
            <Link href="/settings/privacy">Privacy</Link>
          </nav>
          <main>{children}</main>
          <footer>
            <div className="row" style={{ justifyContent: "space-between" }}>
              <span>Patent pending (IN 202541122226). For interview practice only, not for hiring decisions.</span>
              <span className="row"><Link href="/legal/privacy">Privacy notice</Link><Link href="/legal/terms">Terms</Link></span>
            </div>
          </footer>
        </div>
      </body>
    </html>
  );
}
