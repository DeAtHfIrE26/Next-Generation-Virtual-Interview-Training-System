"use client";

// Design-system primitives (docs/DESIGN.md). Small, accessible, token-driven.
import Link from "next/link";
import { Loader2 } from "lucide-react";
import {
  forwardRef,
  useEffect,
  useId,
  useRef,
  type ButtonHTMLAttributes,
  type InputHTMLAttributes,
  type ReactNode,
  type SelectHTMLAttributes,
  type TextareaHTMLAttributes,
} from "react";

export function cn(...parts: (string | false | null | undefined)[]): string {
  return parts.filter(Boolean).join(" ");
}

// ----------------------------------------------------------------------------- Button

type Variant = "primary" | "secondary" | "ghost" | "danger" | "subtle";
type Size = "sm" | "md" | "lg";

const base =
  "inline-flex items-center justify-center gap-2 whitespace-nowrap font-medium transition-[background,border-color,color,transform,box-shadow] duration-[120ms] ease-[var(--ease-out-soft)] active:scale-[0.98] disabled:pointer-events-none disabled:opacity-50 select-none";
const variants: Record<Variant, string> = {
  primary: "bg-accent text-accent-ink hover:bg-accent-strong shadow-[var(--highlight)]",
  secondary: "border border-line bg-surface-2 text-fg hover:border-line-strong hover:bg-surface-3",
  ghost: "text-fg-muted hover:text-fg hover:bg-surface-2",
  danger: "bg-danger/90 text-white hover:bg-danger",
  subtle: "bg-surface-2 text-fg hover:bg-surface-3",
};
const sizes: Record<Size, string> = {
  sm: "h-8 rounded-[10px] px-3 text-[13px]",
  md: "h-10 rounded-[12px] px-4 text-sm",
  lg: "h-12 rounded-[14px] px-6 text-[15px]",
};

export function buttonClass(variant: Variant = "primary", size: Size = "md", extra?: string) {
  return cn(base, variants[variant], sizes[size], extra);
}

type ButtonProps = ButtonHTMLAttributes<HTMLButtonElement> & { variant?: Variant; size?: Size; loading?: boolean };

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { variant = "primary", size = "md", loading, className, children, disabled, ...rest },
  ref,
) {
  return (
    <button ref={ref} className={buttonClass(variant, size, className)} disabled={disabled || loading} aria-busy={loading || undefined} {...rest}>
      {loading && <Loader2 className="size-4 animate-spin" aria-hidden />}
      {children}
    </button>
  );
});

export function ButtonLink({
  href, variant = "primary", size = "md", className, children, ...rest
}: { href: string; variant?: Variant; size?: Size; className?: string; children: ReactNode } & Omit<React.ComponentProps<typeof Link>, "href" | "className">) {
  return <Link href={href} className={buttonClass(variant, size, className)} {...rest}>{children}</Link>;
}

// ----------------------------------------------------------------------------- Surfaces

export function Card({ className, children, as: As = "div", ...rest }: { className?: string; children: ReactNode; as?: "div" | "section" | "article" | "form" } & React.HTMLAttributes<HTMLElement>) {
  return (
    <As className={cn("rounded-[16px] border border-line bg-surface shadow-[var(--highlight)]", className)} {...rest}>
      {children}
    </As>
  );
}

export function Badge({ tone = "neutral", children, className }: { tone?: "neutral" | "accent" | "live" | "warn" | "danger" | "success"; children: ReactNode; className?: string }) {
  const tones = {
    neutral: "border-line bg-surface-2 text-fg-muted",
    accent: "border-accent/30 bg-accent/10 text-accent",
    live: "border-live/30 bg-live/10 text-live",
    warn: "border-warn/30 bg-warn/10 text-warn",
    danger: "border-danger/30 bg-danger/10 text-danger",
    success: "border-success/30 bg-success/10 text-success",
  } as const;
  return <span className={cn("inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[12px] font-medium leading-4", tones[tone], className)}>{children}</span>;
}

export function Kbd({ children }: { children: ReactNode }) {
  return <kbd className="rounded-[6px] border border-line bg-surface-2 px-1.5 py-0.5 font-mono text-[11px] text-fg-muted">{children}</kbd>;
}

export function Spinner({ className }: { className?: string }) {
  return <Loader2 className={cn("size-4 animate-spin text-fg-muted", className)} aria-hidden />;
}

export function Skeleton({ className }: { className?: string }) {
  return (
    <div
      className={cn("rounded-[10px] bg-[linear-gradient(90deg,var(--surface-2),var(--surface-3),var(--surface-2))] bg-[length:200%_100%] animate-shimmer", className)}
      aria-hidden
    />
  );
}

// ----------------------------------------------------------------------------- Form controls

const control =
  "w-full rounded-[12px] border border-line bg-surface-2 px-3 text-sm text-fg placeholder:text-fg-subtle transition-colors duration-[120ms] hover:border-line-strong focus:border-accent focus:outline-none focus-visible:outline-none focus:ring-2 focus:ring-accent/30 disabled:opacity-60";

export function Field({ label, hint, error, children, htmlFor, optional }: { label: string; hint?: ReactNode; error?: string | null; children: ReactNode; htmlFor?: string; optional?: boolean }) {
  return (
    <div className="flex flex-col gap-1.5">
      <label htmlFor={htmlFor} className="flex items-baseline justify-between text-[13px] font-medium text-fg">
        <span>{label}</span>
        {optional && <span className="text-[12px] font-normal text-fg-subtle">Optional</span>}
      </label>
      {children}
      {error ? <p className="text-[12px] text-danger" role="alert">{error}</p> : hint ? <p className="text-[12px] text-fg-subtle">{hint}</p> : null}
    </div>
  );
}

export const Input = forwardRef<HTMLInputElement, InputHTMLAttributes<HTMLInputElement>>(function Input({ className, ...rest }, ref) {
  return <input ref={ref} className={cn(control, "h-10", className)} {...rest} />;
});

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaHTMLAttributes<HTMLTextAreaElement>>(function Textarea({ className, ...rest }, ref) {
  return <textarea ref={ref} className={cn(control, "min-h-24 py-2.5 leading-5", className)} {...rest} />;
});

export const Select = forwardRef<HTMLSelectElement, SelectHTMLAttributes<HTMLSelectElement>>(function Select({ className, children, ...rest }, ref) {
  return (
    <select ref={ref} className={cn(control, "h-10 appearance-none bg-[url('data:image/svg+xml;utf8,<svg xmlns=%22http://www.w3.org/2000/svg%22 width=%2212%22 height=%2212%22 fill=%22none%22 stroke=%22%2371717a%22 stroke-width=%221.6%22><path d=%22M3 4.5l3 3 3-3%22/></svg>')] bg-[right_12px_center] bg-no-repeat pr-9", className)} {...rest}>
      {children}
    </select>
  );
});

export function Checkbox({ label, description, className, ...rest }: InputHTMLAttributes<HTMLInputElement> & { label: ReactNode; description?: ReactNode }) {
  const id = useId();
  return (
    <label htmlFor={rest.id ?? id} className={cn("group flex cursor-pointer items-start gap-3 rounded-[12px] p-1 text-sm", className)}>
      <input
        id={rest.id ?? id}
        type="checkbox"
        className="mt-0.5 size-4 shrink-0 cursor-pointer appearance-none rounded-[5px] border border-line-strong bg-surface-2 transition-colors checked:border-accent checked:bg-accent checked:bg-[url('data:image/svg+xml;utf8,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 16 16%22 fill=%22none%22 stroke=%22white%22 stroke-width=%222.2%22><path d=%22M4 8.5l2.5 2.5L12 5.5%22/></svg>')] bg-center"
        {...rest}
      />
      <span className="flex flex-col gap-0.5">
        <span className="text-fg">{label}</span>
        {description && <span className="text-[13px] text-fg-muted">{description}</span>}
      </span>
    </label>
  );
}

export function Switch({ checked, onChange, label, disabled }: { checked: boolean; onChange: (v: boolean) => void; label: string; disabled?: boolean }) {
  return (
    <button
      type="button"
      role="switch"
      aria-checked={checked}
      aria-label={label}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className={cn(
        "relative inline-flex h-6 w-10 shrink-0 items-center rounded-full border transition-colors duration-[200ms] disabled:opacity-50",
        checked ? "border-accent bg-accent" : "border-line-strong bg-surface-3",
      )}
    >
      <span className={cn("inline-block size-[18px] rounded-full bg-white shadow transition-transform duration-[200ms] ease-[var(--ease-out-soft)]", checked ? "translate-x-[18px]" : "translate-x-[2px]")} />
    </button>
  );
}

export function Segmented<T extends string>({ value, onChange, options, label, size = "md" }: { value: T; onChange: (v: T) => void; options: { value: T; label: ReactNode; hint?: string }[]; label: string; size?: "sm" | "md" }) {
  return (
    <div role="radiogroup" aria-label={label} className="inline-flex w-full flex-wrap gap-1 rounded-[12px] border border-line bg-surface-2 p-1">
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          role="radio"
          aria-checked={value === o.value}
          title={o.hint}
          onClick={() => onChange(o.value)}
          className={cn(
            "flex-1 rounded-[9px] px-3 font-medium transition-[background,color,box-shadow] duration-[120ms]",
            size === "sm" ? "h-7 text-[12px]" : "h-8 text-[13px]",
            value === o.value ? "bg-surface text-fg shadow-[var(--highlight),0_1px_2px_rgb(0_0_0/0.2)] ring-1 ring-line" : "text-fg-muted hover:text-fg",
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

// ----------------------------------------------------------------------------- Data display

export function ProgressBar({ value, tone = "accent", label, className }: { value: number; tone?: "accent" | "live" | "warn" | "success"; label?: string; className?: string }) {
  const color = { accent: "bg-accent", live: "bg-live", warn: "bg-warn", success: "bg-success" }[tone];
  return (
    <div className={cn("h-1.5 w-full overflow-hidden rounded-full bg-surface-3", className)} role="progressbar" aria-label={label} aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(value * 100)}>
      <div className={cn("h-full rounded-full transition-[width] duration-[320ms] ease-[var(--ease-out-soft)]", color)} style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }} />
    </div>
  );
}

/** Live level meter (0..1), e.g. microphone input. */
export function LevelMeter({ level, bars = 12, className }: { level: number; bars?: number; className?: string }) {
  return (
    <div className={cn("flex h-5 items-end gap-[3px]", className)} aria-hidden>
      {Array.from({ length: bars }, (_, i) => {
        const on = level * bars > i;
        return <span key={i} className={cn("w-1 rounded-full transition-[height,background] duration-75", on ? "bg-live" : "bg-surface-3")} style={{ height: `${30 + (i / bars) * 70}%` }} />;
      })}
    </div>
  );
}

export function Stat({ label, value, sub, className }: { label: string; value: ReactNode; sub?: ReactNode; className?: string }) {
  return (
    <div className={cn("flex flex-col gap-1", className)}>
      <span className="text-[12px] font-medium uppercase tracking-[0.06em] text-fg-subtle">{label}</span>
      <span className="font-mono text-[22px] leading-7 tabular text-fg">{value}</span>
      {sub && <span className="text-[12px] text-fg-muted">{sub}</span>}
    </div>
  );
}

export function EmptyState({ icon, title, body, action }: { icon?: ReactNode; title: string; body?: ReactNode; action?: ReactNode }) {
  return (
    <div className="flex flex-col items-center gap-3 rounded-[16px] border border-dashed border-line px-6 py-12 text-center">
      {icon && <div className="text-fg-subtle">{icon}</div>}
      <h3 className="text-[15px] font-semibold">{title}</h3>
      {body && <p className="max-w-sm text-[13px] text-fg-muted">{body}</p>}
      {action}
    </div>
  );
}

export function Alert({ tone = "neutral", title, children, action }: { tone?: "neutral" | "warn" | "danger" | "success" | "accent"; title?: string; children?: ReactNode; action?: ReactNode }) {
  const tones = {
    neutral: "border-line bg-surface-2",
    warn: "border-warn/30 bg-warn/[0.07]",
    danger: "border-danger/30 bg-danger/[0.07]",
    success: "border-success/30 bg-success/[0.07]",
    accent: "border-accent/30 bg-accent/[0.07]",
  } as const;
  return (
    <div role={tone === "danger" ? "alert" : "status"} className={cn("flex items-start justify-between gap-4 rounded-[12px] border px-4 py-3 text-[13px]", tones[tone])}>
      <div className="flex flex-col gap-0.5">
        {title && <strong className="font-semibold text-fg">{title}</strong>}
        {children && <div className="text-fg-muted">{children}</div>}
      </div>
      {action}
    </div>
  );
}

// ----------------------------------------------------------------------------- Dialog (native <dialog>)

export function Dialog({ open, onClose, title, children, footer }: { open: boolean; onClose: () => void; title: string; children: ReactNode; footer?: ReactNode }) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (open && !d.open) d.showModal();
    if (!open && d.open) d.close();
  }, [open]);
  return (
    <dialog
      ref={ref}
      onClose={onClose}
      onClick={(e) => e.target === ref.current && onClose()}
      className="m-auto w-[min(92vw,440px)] rounded-[20px] border border-line bg-surface p-0 text-fg shadow-[var(--shadow-float)] backdrop:bg-black/50 backdrop:backdrop-blur-sm"
    >
      <div className="flex flex-col gap-4 p-6">
        <h2 className="text-[18px] font-semibold">{title}</h2>
        <div className="text-[14px] text-fg-muted">{children}</div>
        {footer && <div className="flex justify-end gap-2 pt-2">{footer}</div>}
      </div>
    </dialog>
  );
}
