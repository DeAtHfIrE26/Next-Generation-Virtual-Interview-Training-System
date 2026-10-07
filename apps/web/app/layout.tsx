import type { Metadata, Viewport } from "next";
import { GeistMono } from "geist/font/mono";
import { GeistSans } from "geist/font/sans";
import { THEME_SCRIPT } from "@/components/shell";
import "./globals.css";

export const metadata: Metadata = {
  title: { default: "Interview Coach: practise with a live AI interviewer", template: "%s · Interview Coach" },
  description:
    "Rehearse real interviews out loud with an adaptive AI interviewer that listens, follows up and gives feedback grounded in your own words.",
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: dark)", color: "#09090b" },
    { media: "(prefers-color-scheme: light)", color: "#fafafa" },
  ],
};

// three.js and TalkingHead are served as native ES modules from /vendor (scripts/vendor-3d.mjs).
const IMPORT_MAP = JSON.stringify({
  imports: { three: "/vendor/three/three.module.js", "three/addons/": "/vendor/three/addons/" },
});

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className={`${GeistSans.variable} ${GeistMono.variable}`} suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_SCRIPT }} />
        <script type="importmap" dangerouslySetInnerHTML={{ __html: IMPORT_MAP }} />
      </head>
      <body className="min-h-dvh">{children}</body>
    </html>
  );
}
