# Design direction

The product should feel like a calm, expensive piece of software: the restraint of Linear, the typographic clarity of Vercel, and Apple's sense of space and motion. During an interview, the interviewer (the avatar) and the conversation are the whole experience; everything else steps back.

## Principles

1. **One focal point per screen.** The room is a stage for the interviewer. The report leads with one number and one sentence. Landing has one call to action.
2. **Honest UI.** Live states (listening, thinking, speaking, reconnecting) are always visible and always true. Experimental scores are labelled. Emergency (non-LLM) questions show a small badge.
3. **Dark first, light equal.** Both themes are designed, not inverted. The system preference decides; the user can override.
4. **Typography does the work.** Hierarchy comes from size, weight and spacing. Colour is reserved for state and the single accent.
5. **Motion explains.** Motion shows a change of state (a question arriving, a phase change, a score filling in). There are no decorative loops. `prefers-reduced-motion` turns transforms off.
6. **Accessible by default.** WCAG 2.2 AA contrast, visible focus, full keyboard control, captions always available, live regions for questions and state.

## References studied

| Reference | What we take |
|---|---|
| Linear | Dense but airy layouts, hairline borders instead of shadows, an exact grey scale, keyboard-first controls |
| Vercel / Geist | Geist Sans and Mono, monochrome UI with a single accent, crisp 1 px borders, numbers in mono |
| Apple (product pages, FaceTime) | Generous whitespace, cinematic hero, soft spotlight behind a person, a floating control dock |
| Zoom, Google Meet | Familiar room conventions: self-view picture-in-picture, a bottom control bar, and a mic level meter on the mic button |

## Colour

Tokens live in `apps/web/app/globals.css` (`--color-*`) and are exposed to Tailwind through `@theme`.

| Token | Dark | Light | Use |
|---|---|---|---|
| `canvas` | `#09090B` | `#FAFAFA` | page background |
| `surface` | `#111114` | `#FFFFFF` | cards, panels |
| `surface-2` | `#18181C` | `#F4F4F5` | inputs, nested panels |
| `line` | `rgba(255,255,255,.08)` | `rgba(9,9,11,.08)` | hairline borders |
| `line-strong` | `rgba(255,255,255,.14)` | `rgba(9,9,11,.14)` | hover and focus borders |
| `fg` | `#EDEDEF` | `#09090B` | primary text |
| `fg-muted` | `#A1A1AA` | `#52525B` | secondary text (AA on surface) |
| `fg-subtle` | `#71717A` | `#71717A` | tertiary text, large sizes only |
| `accent` | `#8B8BFF` | `#5B5BEF` | the single brand accent: primary actions, focus, the speaking ring |
| `live` | `#2DD4BF` | `#0D9488` | listening and active microphone |
| `warn` | `#F5A524` | `#B45309` | thinking, reconnecting, experimental |
| `danger` | `#F87171` | `#DC2626` | end interview, errors |
| `success` | `#4ADE80` | `#16A34A` | passed checks |

The accent gradient (`accent` to `live`, 120°) is used in exactly two places: the brand mark and the avatar stage spotlight.

## Typography

- **Geist Sans** for UI and **Geist Mono** for timers, scores and metrics (self-hosted from the `geist` package, OFL).
- Scale (px / line-height / tracking):
  - 12/16
  - 13/18
  - 14/20 (UI default)
  - 16/24 (body)
  - 18/28
  - 22/30 at −0.01em
  - 28/36 at −0.02em
  - 36/44 at −0.02em
  - 48/52 at −0.03em
  - 64/68 at −0.035em (hero)
- Weights: 400 body, 500 UI labels, 600 headings. Never 700+ except the hero.
- Numbers are tabular (`font-variant-numeric: tabular-nums`) wherever they update live.

## Space, shape, elevation

- 4 px base grid. Component padding is 12/16/20; sections are 96 px (desktop) and 64 px (mobile); content max width is 1200 px. The room uses the full viewport.
- Radius: 8 for small controls, 12 for inputs and buttons, 16 for cards, 24 for the stage, 999 for pills.
- Elevation comes from borders plus a 1 px inner highlight (`inset 0 1px 0 rgba(255,255,255,.04)`). Shadows appear only on floating layers (dock, dialogs, toasts).

## Motion

| Token | Duration | Easing | Use |
|---|---|---|---|
| `micro` | 120 ms | `cubic-bezier(.2,.8,.2,1)` | hover, press, toggles |
| `ui` | 200 ms | same | panels, chips, tooltips |
| `layout` | 320 ms | same | page sections, the question card arriving |
| `presence` | spring (stiffness 260, damping 30) | — | phase chip, dock |

Specific behaviours:
- The phase chip morphs between Listening (teal pulse), Thinking (amber shimmer) and Speaking (accent ring that follows audio level).
- The caption text fades in word by word as partial transcripts arrive. Final text settles without jumping.
- With reduced motion, the pulses and shimmers become static colour changes.

## Screens

| Screen | Purpose and key elements |
|---|---|
| Landing | Hero with a live-rendered interviewer, the promise in one line, one CTA. "How it works" in three steps. Proof (what is measured and how). Privacy. Pricing teaser. |
| Sign up / log in | Single card, no distractions. Adult and terms checkboxes. Clear errors. |
| Onboarding | Consent per purpose in plain language, with sensible defaults (only data processing is required). |
| New interview | A two-step set-up: (1) role, company, JD, resume; (2) type, round, difficulty, duration, language, interviewer persona. Live summary on the side. |
| Device check | Camera preview, mic level meter with device picker, speaker test (plays the persona's voice), connection test. Every permission error has its own state with fix-it steps. |
| Interview room | Full-bleed stage with the 3D interviewer and spotlight. Captions bottom-centre. Top bar: role, timer, competency progress. Self-view picture-in-picture. Floating dock: mic (with level ring), type answer, repeat, captions, end. Phase chip. `?debug=1` diagnostics drawer. |
| Report | Overall score and summary sentence. Per-skill bars. Highlighted moments with the candidate's own words. Tips. The full transcript with evidence quotes highlighted. Delivery metrics. Share. |
| Dashboard | Start-interview card, history list, progress-over-time chart per dimension. |
| Pricing, Settings | Plain cards; settings split into Account, Privacy, Billing. |

## Design QA loop

Every screen is screenshotted at 1440×900 (desktop), 834×1112 (tablet) and 390×844 (mobile), in dark and light. Each screenshot is critiqued against this document and iterated on. Final screenshots go in `docs/evidence/ui/`, named `<screen>-<viewport>-<theme>.png`.
