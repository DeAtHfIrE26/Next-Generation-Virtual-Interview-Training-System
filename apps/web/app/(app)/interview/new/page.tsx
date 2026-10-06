"use client";

import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { ArrowRight, Building2, Check, FileText, Languages, Sparkles, Upload, UserRound, X } from "lucide-react";
import { Alert, Button, Card, cn, Field, Input, Segmented, Select, Skeleton, Textarea } from "@/components/ui";
import { api, ApiError } from "@/lib/api";

interface Persona { id: string; name: string; title: string; look: Record<string, string> }
interface Options {
  seniorities: string[]; interview_types: string[]; rounds: string[]; languages: Record<string, string>; personas: Persona[];
  capabilities: { llm: string | null };
}

const TYPE_HINT: Record<string, string> = {
  technical: "Fundamentals, problem solving and depth on your projects",
  behavioral: "Ownership, collaboration and conflict, with STAR follow-ups",
  system_design: "Requirements, architecture, trade-offs and scale",
  hr: "Motivation, values and career goals",
  case: "Structuring an ambiguous problem and recommending",
  mixed: "A realistic blend for the role",
};
const SPECIAL: Record<string, string> = { hr: "HR and culture", system_design: "System design" };
const label = (s: string) => SPECIAL[s] ?? s.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());

export default function NewInterview() {
  const router = useRouter();
  const [opts, setOpts] = useState<Options | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [role, setRole] = useState("");
  const [seniority, setSeniority] = useState("mid");
  const [company, setCompany] = useState("");
  const [companyStyle, setCompanyStyle] = useState("");
  const [type, setType] = useState("mixed");
  const [round, setRound] = useState("technical");
  const [jd, setJd] = useState("");
  const [skills, setSkills] = useState<string[]>([]);
  const [skillDraft, setSkillDraft] = useState("");
  const [resume, setResume] = useState<File | null>(null);
  const [duration, setDuration] = useState(20);
  const [difficulty, setDifficulty] = useState("auto");
  const [language, setLanguage] = useState("en");
  const [persona, setPersona] = useState("maya");
  const [mode, setMode] = useState("coaching");
  const fileRef = useRef<HTMLInputElement>(null);

  useEffect(() => { void api<Options>("/sessions/options").then(setOpts).catch(() => setError("Could not load the interview options.")); }, []);

  function addSkill() {
    const parts = skillDraft.split(",").map((s) => s.trim()).filter(Boolean);
    if (parts.length) setSkills((s) => Array.from(new Set([...s, ...parts])).slice(0, 12));
    setSkillDraft("");
  }

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError("");
    const form = new FormData();
    Object.entries({
      role, seniority, company, company_style: companyStyle, interview_type: type, round, job_description: jd,
      skills: [...skills, ...skillDraft.split(",").map((s) => s.trim()).filter(Boolean)].join(","),
      duration_minutes: String(duration), difficulty, language, persona, mode,
    }).forEach(([k, v]) => form.set(k, v));
    if (resume) form.set("resume", resume);
    try {
      const r = await api<{ id: string }>("/sessions", { method: "POST", body: form });
      router.push(`/interview/${r.id}`);
    } catch (err) {
      if (err instanceof ApiError && err.status === 403) { router.push("/onboarding"); return; }
      setError(err instanceof ApiError ? err.message : "Could not start the interview.");
      setBusy(false);
    }
  }

  return (
    <form onSubmit={submit} className="mx-auto grid max-w-[1200px] gap-6 px-5 py-8 lg:grid-cols-[1fr_340px]">
      <div className="flex flex-col gap-6">
        <header>
          <p className="text-[13px] font-medium text-accent">New interview</p>
          <h1 className="mt-1 text-[30px] font-semibold leading-9">Set the scene</h1>
          <p className="mt-1 text-[14px] text-fg-muted">The more context you give, the more the questions sound like the real thing. Only the role is required.</p>
        </header>

        <Section icon={<UserRound className="size-4" />} title="The role">
          <div className="grid gap-4 sm:grid-cols-[1fr_200px]">
            <Field label="Target role" htmlFor="role"><Input id="role" required minLength={2} maxLength={120} value={role} onChange={(e) => setRole(e.target.value)} placeholder="e.g. Backend Software Engineer" /></Field>
            <Field label="Seniority" htmlFor="seniority">
              <Select id="seniority" value={seniority} onChange={(e) => setSeniority(e.target.value)}>
                {(opts?.seniorities ?? ["mid"]).map((s) => <option key={s} value={s}>{label(s)}</option>)}
              </Select>
            </Field>
          </div>
        </Section>

        <Section icon={<Building2 className="size-4" />} title="Company" optional>
          <div className="grid gap-4 sm:grid-cols-2">
            <Field label="Company" htmlFor="company"><Input id="company" maxLength={120} value={company} onChange={(e) => setCompany(e.target.value)} placeholder="e.g. Razorpay" /></Field>
            <Field label="How they interview" htmlFor="style" hint="Tone, format, what they're known to probe."><Input id="style" maxLength={400} value={companyStyle} onChange={(e) => setCompanyStyle(e.target.value)} placeholder="e.g. bar-raiser style, deep on ownership" /></Field>
          </div>
        </Section>

        <Section icon={<Sparkles className="size-4" />} title="Format">
          <div className="grid gap-2 sm:grid-cols-3">
            {(opts?.interview_types ?? []).map((t) => (
              <button key={t} type="button" onClick={() => setType(t)} aria-pressed={type === t}
                className={cn("flex flex-col gap-1 rounded-[14px] border p-3 text-left transition-colors", type === t ? "border-accent bg-accent/10" : "border-line hover:border-line-strong")}>
                <span className="flex items-center justify-between text-[14px] font-medium">{label(t)}{type === t && <Check className="size-4 text-accent" />}</span>
                <span className="text-[12px] leading-4 text-fg-muted">{TYPE_HINT[t]}</span>
              </button>
            ))}
            {!opts && Array.from({ length: 6 }).map((_, i) => <Skeleton key={i} className="h-20" />)}
          </div>
          <div className="mt-4 grid gap-4 sm:grid-cols-2">
            <Field label="Round" htmlFor="round">
              <Select id="round" value={round} onChange={(e) => setRound(e.target.value)}>
                {(opts?.rounds ?? ["technical"]).map((r) => <option key={r} value={r}>{label(r)}</option>)}
              </Select>
            </Field>
            <Field label={`Length: ${duration} minutes`} htmlFor="duration">
              <input id="duration" type="range" min={5} max={60} step={5} value={duration} onChange={(e) => setDuration(+e.target.value)} className="mt-3 w-full accent-[var(--accent)]" />
            </Field>
          </div>
          <div className="mt-4 flex flex-col gap-2">
            <span className="text-[13px] font-medium">Difficulty</span>
            <Segmented label="Difficulty" value={difficulty} onChange={setDifficulty}
              options={[{ value: "auto", label: "Adaptive" }, { value: "1", label: "1" }, { value: "2", label: "2" }, { value: "3", label: "3" }, { value: "4", label: "4" }, { value: "5", label: "5" }]} />
            <span className="text-[12px] text-fg-subtle">{difficulty === "auto" ? "Starts where your profile suggests and adapts after every answer." : "Stays within one level of your choice, adapting to your answers."}</span>
          </div>
        </Section>

        <Section icon={<FileText className="size-4" />} title="Context" optional>
          <Field label="Job description" htmlFor="jd" hint="Paste the posting. Questions will target what it asks for.">
            <Textarea id="jd" rows={5} maxLength={6000} value={jd} onChange={(e) => setJd(e.target.value)} />
          </Field>
          <div className="mt-4 grid gap-4 sm:grid-cols-2">
            <Field label="Skills to probe" htmlFor="skills" hint="Press Enter or comma to add.">
              <div className="flex flex-wrap items-center gap-1.5 rounded-[12px] border border-line bg-surface px-2 py-1.5 focus-within:border-accent">
                {skills.map((s) => (
                  <span key={s} className="inline-flex items-center gap-1 rounded-full bg-surface-3 px-2 py-0.5 text-[12px]">
                    {s}<button type="button" aria-label={`Remove ${s}`} onClick={() => setSkills((x) => x.filter((y) => y !== s))}><X className="size-3" /></button>
                  </span>
                ))}
                <input id="skills" value={skillDraft} onChange={(e) => setSkillDraft(e.target.value)} onBlur={addSkill}
                  onKeyDown={(e) => { if (e.key === "Enter" || e.key === ",") { e.preventDefault(); addSkill(); } }}
                  placeholder={skills.length ? "" : "e.g. Postgres, API design"} className="min-w-[120px] flex-1 bg-transparent py-1 text-[14px] outline-none" />
              </div>
            </Field>
            <Field label="Resume (PDF, up to 2 MB)" htmlFor="resume" hint="Contact details are removed before any AI sees it.">
              <input ref={fileRef} id="resume" type="file" accept="application/pdf" className="sr-only" onChange={(e) => setResume(e.target.files?.[0] ?? null)} />
              <Button type="button" variant="secondary" className="w-full justify-start" onClick={() => fileRef.current?.click()}>
                <Upload className="size-4" /> <span className="truncate">{resume ? resume.name : "Upload resume"}</span>
              </Button>
            </Field>
          </div>
        </Section>

        <Section icon={<UserRound className="size-4" />} title="Your interviewer">
          <div className="grid gap-2 sm:grid-cols-2">
            {(opts?.personas ?? []).map((p) => (
              <button key={p.id} type="button" onClick={() => setPersona(p.id)} aria-pressed={persona === p.id}
                className={cn("flex items-center gap-3 rounded-[14px] border p-3 text-left transition-colors", persona === p.id ? "border-accent bg-accent/10" : "border-line hover:border-line-strong")}>
                <span className="grid size-10 place-items-center rounded-full text-[14px] font-semibold text-white" style={{ background: `linear-gradient(135deg, ${p.look.accent ?? "#7c9cff"}, ${p.look.top ?? "#1f2a44"})` }}>{p.name[0]}</span>
                <span className="flex-1"><span className="block text-[14px] font-medium">{p.name}</span><span className="block text-[12px] text-fg-muted">{p.title}</span></span>
                {persona === p.id && <Check className="size-4 text-accent" />}
              </button>
            ))}
          </div>
          <div className="mt-4 grid gap-4 sm:grid-cols-2">
            <Field label="Language" htmlFor="lang">
              <div className="relative">
                <Languages className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-fg-subtle" />
                <Select id="lang" value={language} onChange={(e) => setLanguage(e.target.value)} className="pl-9">
                  {Object.entries(opts?.languages ?? { en: "English" }).map(([k, v]) => <option key={k} value={k}>{v}</option>)}
                </Select>
              </div>
            </Field>
            <Field label="Integrity mode" htmlFor="mode" hint={mode === "coaching" ? "Notices are shown but never affect your score." : "Repeated integrity issues end the session."}>
              <Select id="mode" value={mode} onChange={(e) => setMode(e.target.value)}>
                <option value="coaching">Coaching</option>
                <option value="proctored">Strict practice</option>
              </Select>
            </Field>
          </div>
        </Section>
      </div>

      {/* Summary */}
      <aside className="lg:sticky lg:top-20 lg:self-start">
        <Card className="flex flex-col gap-4 p-5">
          <h2 className="text-[16px] font-semibold">Summary</h2>
          <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-2 text-[13px]">
            <dt className="text-fg-subtle">Role</dt><dd className="truncate">{role || "–"}{seniority ? ` (${label(seniority)})` : ""}</dd>
            <dt className="text-fg-subtle">Company</dt><dd className="truncate">{company || "–"}</dd>
            <dt className="text-fg-subtle">Format</dt><dd>{label(type)} · {label(round)}</dd>
            <dt className="text-fg-subtle">Length</dt><dd>{duration} min</dd>
            <dt className="text-fg-subtle">Difficulty</dt><dd>{difficulty === "auto" ? "Adaptive" : `Level ${difficulty}`}</dd>
            <dt className="text-fg-subtle">Interviewer</dt><dd>{opts?.personas.find((p) => p.id === persona)?.name ?? "–"}</dd>
            <dt className="text-fg-subtle">Context</dt><dd>{[jd && "job description", resume && "resume", (skills.length || skillDraft) && "skills"].filter(Boolean).join(", ") || "none yet"}</dd>
          </dl>
          {opts && !opts.capabilities.llm && (
            <Alert tone="warn">The AI interviewer isn&rsquo;t configured on this server, so you&rsquo;ll get flagged backup questions.</Alert>
          )}
          {error && <Alert tone="danger">{error}</Alert>}
          <Button type="submit" size="lg" loading={busy} disabled={role.trim().length < 2}>Start interview <ArrowRight className="size-4" /></Button>
          <p className="text-[12px] leading-5 text-fg-subtle">Next you&rsquo;ll check your camera and microphone. Video is processed on your device and never uploaded.</p>
        </Card>
      </aside>
    </form>
  );
}

function Section({ icon, title, optional, children }: { icon: React.ReactNode; title: string; optional?: boolean; children: React.ReactNode }) {
  return (
    <Card className="p-5">
      <h2 className="mb-4 flex items-center gap-2 text-[15px] font-semibold"><span className="text-accent">{icon}</span>{title}{optional && <span className="text-[12px] font-normal text-fg-subtle">optional</span>}</h2>
      {children}
    </Card>
  );
}
