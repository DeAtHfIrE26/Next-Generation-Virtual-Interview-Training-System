"use client";

import Link from "next/link";
import {
  ArrowRight, AudioLines, BadgeCheck, Brain, Building2, Code2, Eye, FileText, Lock, MessagesSquare, Mic, ShieldCheck, Sparkles, Trash2, Users,
} from "lucide-react";
import { AvatarStage } from "@/components/avatar/AvatarStage";
import { Badge, ButtonLink, Card } from "@/components/ui";

const steps = [
  { icon: FileText, title: "Set the scene", body: "Pick the role, company and round. Paste the job description and add your resume. The interviewer builds a plan of what to probe and how to spend the time." },
  { icon: Mic, title: "Talk it through, out loud", body: "Answer by voice, the way you will on the day. It listens, follows up on what you actually said, pushes back on vague answers and adjusts the difficulty as you go." },
  { icon: BadgeCheck, title: "Get feedback you can check", body: "Every score points to your own words. See a breakdown per skill, your strongest and weakest moments, pace and filler words, and concrete fixes." },
];

const measures = [
  { icon: MessagesSquare, title: "Content", body: "Relevance, structure (STAR), depth and technical accuracy, each backed by quotes from your answer that are checked against the transcript." },
  { icon: AudioLines, title: "Delivery", body: "Speaking pace, long pauses and filler words, measured from word-level timings of your actual speech." },
  { icon: Eye, title: "Presence", body: "Eye contact with the camera and framing, computed on your device. Video never leaves your browser." },
  { icon: ShieldCheck, title: "Integrity checks", body: "Lip-sync verification and phone or second-person detection. In practice mode these are informational and never change your score." },
];

const types = [
  { icon: Code2, label: "Technical" },
  { icon: Users, label: "Behavioural" },
  { icon: Building2, label: "System design" },
  { icon: Brain, label: "Case" },
  { icon: Sparkles, label: "HR and culture" },
];

export default function Landing() {
  return (
    <>
      {/* Hero */}
      <section className="relative overflow-hidden">
        <div className="pointer-events-none absolute inset-0 -z-10 opacity-80" style={{ background: "var(--stage-glow)" }} />
        <div className="mx-auto grid max-w-[1200px] items-center gap-12 px-5 pb-20 pt-16 md:pt-24 lg:grid-cols-[1.05fr_1fr]">
          <div className="flex flex-col gap-6 animate-fade-up">
            <Badge tone="accent" className="w-fit"><Sparkles className="size-3" /> Live AI interviewer · Patent pending</Badge>
            <h1 className="text-[44px] font-semibold leading-[48px] tracking-[-0.035em] sm:text-[60px] sm:leading-[64px]">
              Rehearse the interview<br className="hidden sm:block" /> before it counts.
            </h1>
            <p className="max-w-xl text-[17px] leading-7 text-fg-muted">
              Speak with an AI interviewer that reads your resume, listens to every answer and follows up like a real one.
              Then see exactly where you are strong, in your own words.
            </p>
            <div className="flex flex-wrap gap-3">
              <ButtonLink href="/signup" size="lg">Start a practice interview <ArrowRight className="size-4" /></ButtonLink>
              <ButtonLink href="#how" size="lg" variant="secondary">How it works</ButtonLink>
            </div>
            <ul className="flex flex-wrap gap-x-6 gap-y-2 pt-2 text-[13px] text-fg-muted">
              <li className="flex items-center gap-2"><Lock className="size-3.5" /> Video stays on your device</li>
              <li className="flex items-center gap-2"><Trash2 className="size-3.5" /> Delete everything in one click</li>
              <li className="flex items-center gap-2"><ShieldCheck className="size-3.5" /> No emotion or personality guessing</li>
            </ul>
          </div>
          <div className="relative">
            <AvatarStage deferUntilIdle look={{ skin: "#c99a7a", hair: "#2b1d16", top: "#1f2a44" }} state="idle" className="aspect-[4/5] w-full sm:aspect-[5/5]" name="Maya · Engineering Manager" />
            <div className="absolute -bottom-5 left-4 right-4 sm:left-10 sm:right-10">
              <Card className="px-4 py-3 shadow-[var(--shadow-float)]">
                <p className="text-[12px] font-medium text-fg-subtle">Example follow-up</p>
                <p className="mt-1 text-[14px] leading-5">&ldquo;You said you cut the nightly job from six hours to forty minutes. What made the old job slow, and how did you verify the fix?&rdquo;</p>
              </Card>
            </div>
          </div>
        </div>
      </section>

      {/* How it works */}
      <section id="how" className="border-t border-line">
        <div className="mx-auto max-w-[1200px] px-5 py-24">
          <div className="max-w-2xl">
            <p className="text-[13px] font-medium text-accent">How it works</p>
            <h2 className="mt-2 text-[36px] font-semibold leading-[44px]">A real interview, not a quiz.</h2>
            <p className="mt-3 text-[16px] leading-6 text-fg-muted">Every question is written live for you. No fixed question list, no repeats.</p>
          </div>
          <div className="mt-12 grid gap-4 md:grid-cols-3">
            {steps.map((s, i) => (
              <Card key={s.title} className="flex flex-col gap-4 p-6">
                <div className="flex items-center justify-between">
                  <span className="grid size-10 place-items-center rounded-[12px] border border-line bg-surface-2"><s.icon className="size-5 text-fg" /></span>
                  <span className="font-mono text-[12px] text-fg-subtle">0{i + 1}</span>
                </div>
                <h3 className="text-[18px] font-semibold">{s.title}</h3>
                <p className="text-[14px] leading-6 text-fg-muted">{s.body}</p>
              </Card>
            ))}
          </div>
          <div className="mt-10 flex flex-wrap items-center gap-2">
            <span className="mr-2 text-[13px] text-fg-muted">Interview types:</span>
            {types.map((t) => (
              <span key={t.label} className="inline-flex items-center gap-2 rounded-full border border-line bg-surface px-3 py-1.5 text-[13px]"><t.icon className="size-3.5 text-fg-muted" />{t.label}</span>
            ))}
          </div>
        </div>
      </section>

      {/* What we measure */}
      <section id="proof" className="border-t border-line bg-surface/40">
        <div className="mx-auto max-w-[1200px] px-5 py-24">
          <div className="grid gap-12 lg:grid-cols-[0.9fr_1.1fr]">
            <div className="max-w-md">
              <p className="text-[13px] font-medium text-accent">What we measure</p>
              <h2 className="mt-2 text-[36px] font-semibold leading-[44px]">Feedback you can verify.</h2>
              <p className="mt-3 text-[16px] leading-6 text-fg-muted">
                Scores quote your own answer, and quotes are checked against the transcript. Scores stay labelled
                <em> experimental</em> until they are validated against human raters, and we publish how that is measured.
              </p>
              <Link href="/pricing" className="mt-6 inline-flex items-center gap-1 text-[14px] font-medium text-fg hover:text-accent">See plans <ArrowRight className="size-4" /></Link>
            </div>
            <div className="grid gap-4 sm:grid-cols-2">
              {measures.map((m) => (
                <Card key={m.title} className="flex flex-col gap-3 p-6">
                  <m.icon className="size-5 text-accent" />
                  <h3 className="text-[16px] font-semibold">{m.title}</h3>
                  <p className="text-[14px] leading-6 text-fg-muted">{m.body}</p>
                </Card>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="border-t border-line">
        <div className="mx-auto flex max-w-[1200px] flex-col items-start gap-6 px-5 py-24 md:flex-row md:items-center md:justify-between">
          <div>
            <h2 className="text-[32px] font-semibold leading-10">Your next interview is a rehearsal away.</h2>
            <p className="mt-2 text-[16px] text-fg-muted">Free to start. About two minutes to set up.</p>
          </div>
          <ButtonLink href="/signup" size="lg">Start practising <ArrowRight className="size-4" /></ButtonLink>
        </div>
      </section>
    </>
  );
}
