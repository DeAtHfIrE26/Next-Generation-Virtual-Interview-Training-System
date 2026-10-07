import { Check } from "lucide-react";
import { Badge, ButtonLink } from "@/components/ui";

// Server component: only render client components from the UI kit (helpers such as cn are client references).
const cx = (...parts: (string | false | undefined)[]) => parts.filter(Boolean).join(" ");

type Plan = {
  name: string;
  summary: string;
  features: string[];
  cta: { label: string; href: string };
  featured?: boolean;
};

// Same plans and claims as before; prices are shown at checkout, limits are set by the operator.
const plans: Plan[] = [
  {
    name: "Free",
    summary: "A few practice sessions each month with automatic and AI feedback, browser voice.",
    features: ["A few practice sessions each month", "Automatic and AI feedback", "Browser voice"],
    cta: { label: "Start free", href: "/signup" },
  },
  {
    name: "Pro",
    summary: "More sessions, natural interviewer voice, longer interviews, full report history.",
    features: ["More sessions", "Natural interviewer voice", "Longer interviews", "Full report history"],
    cta: { label: "Upgrade", href: "/settings/billing" },
    featured: true,
  },
  {
    name: "Teams and colleges",
    summary: "Seats for placement cells and bootcamps, for candidate practice only.",
    features: ["Seats for placement cells and bootcamps", "For candidate practice only"],
    cta: { label: "Contact us", href: "mailto:sales@example.com" },
  },
];

export default function Pricing() {
  return (
    <div className="mx-auto flex max-w-[1100px] flex-col gap-10 px-5 py-16 md:py-20">
      <header className="flex flex-col items-center gap-3 text-center">
        <p className="text-[13px] font-medium text-accent">Pricing</p>
        <h1 className="text-[36px] font-semibold leading-[44px] tracking-[-0.02em] sm:text-[48px] sm:leading-[52px] sm:tracking-[-0.03em]">Plans for every stage of prep</h1>
        <p className="max-w-xl text-[16px] leading-6 text-fg-muted">Start free. Upgrade when you want more practice.</p>
      </header>

      <ul className="grid gap-4 md:grid-cols-3" aria-label="Plans">
        {plans.map((p) => (
          <li
            key={p.name}
            className={cx(
              "relative flex flex-col gap-6 rounded-[16px] border bg-surface p-6 shadow-[var(--highlight)]",
              p.featured ? "border-accent/50 ring-1 ring-accent/30" : "border-line",
            )}
          >
            <div className="flex flex-col gap-2">
              <div className="flex items-center justify-between gap-2">
                <h2 className="text-[18px] font-semibold leading-7">{p.name}</h2>
                {p.featured && <Badge tone="accent">Recommended</Badge>}
              </div>
              <p className="text-[14px] leading-5 text-fg-muted">{p.summary}</p>
            </div>
            <ul className="flex flex-1 flex-col gap-3 border-t border-line pt-5 text-[14px]" aria-label={`${p.name} includes`}>
              {p.features.map((f) => (
                <li key={f} className="flex items-start gap-2.5">
                  <Check className={cx("mt-0.5 size-4 shrink-0", p.featured ? "text-accent" : "text-fg-subtle")} aria-hidden />
                  <span className="text-fg">{f}</span>
                </li>
              ))}
            </ul>
            <ButtonLink href={p.cta.href} variant={p.featured ? "primary" : "secondary"} className="w-full">{p.cta.label}</ButtonLink>
          </li>
        ))}
      </ul>

      <p className="text-center text-[13px] leading-5 text-fg-muted">Prices are shown at checkout in INR (Razorpay) or USD (Stripe). Limits per plan are configured by the operator.</p>
    </div>
  );
}
