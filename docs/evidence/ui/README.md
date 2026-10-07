# Design QA screenshots (docs/DESIGN.md, "Design QA loop")

12 screens × 3 viewports × 2 themes, 72 images in total. They were captured from the production build (`docker compose up`) on 2026-10-06 by headless Chromium with software WebGL.

- Screens: landing, pricing, login, signup, onboarding, dashboard, new interview, device check, report, settings, privacy, billing.
- Viewports: desktop 1440×900, tablet 834×1112, mobile 390×844.
- Themes: dark and light.
- Naming: `<screen>--<viewport>--<theme>.jpg`.
- The signed-in screens belong to a throwaway local test user. The report is a real finished session with typed answers and no LLM configured, so its questions are flagged backup questions and it uses offline scoring.
- The live room (listening, speaking, barge-in, captions, diagnostics) is shown in `docs/evidence/e2e/compose-no-llm/`.

The QA pass on these images found and fixed:
- the "Hr" label (now "HR and culture");
- duplicate "optional" markers inside optional sections;
- repeated tips in the report;
- three-word evidence fragments from the offline scorer (now whole sentences, still verbatim);
- light-theme colours below AA contrast (see `docs/evidence/lighthouse/README.md`).
