import Link from "next/link";

export default function Pricing() {
  return (
    <div className="stack">
      <h1>Pricing</h1>
      <div className="grid three">
        <div className="card stack"><h3>Free</h3><p className="muted">A few practice sessions each month with automatic and AI feedback, browser voice.</p>
          <Link className="btn" href="/signup">Start free</Link></div>
        <div className="card stack"><h3>Pro</h3><p className="muted">More sessions, natural interviewer voice, longer interviews, full report history.</p>
          <Link className="btn primary" href="/settings/billing">Upgrade</Link></div>
        <div className="card stack"><h3>Teams and colleges</h3><p className="muted">Seats for placement cells and bootcamps, for candidate practice only.</p>
          <a className="btn" href="mailto:sales@example.com">Contact us</a></div>
      </div>
      <p className="small muted">Prices are shown at checkout in INR (Razorpay) or USD (Stripe). Limits per plan are configured by the operator.</p>
    </div>
  );
}
