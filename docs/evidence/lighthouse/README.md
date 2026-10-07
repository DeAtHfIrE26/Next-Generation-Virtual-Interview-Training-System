# Lighthouse (DoD #5: performance and accessibility ≥ 90)

Lighthouse 13.5.0, default mobile emulation (simulated throttling), headless Chromium 1194, run against the production build served by `docker compose up` on 2026-10-06. The report page belonged to a real finished session with two answered questions and was fetched with that test user's session (the cookie is redacted in the JSON). Open the JSON files in https://googlechrome.github.io/lighthouse/viewer/.

| Page | Performance | Accessibility | FCP | LCP | TBT | CLS |
|---|---|---|---|---|---|---|
| landing (`/`) | **94** | **100** | 0.8 s | 2.3 s | 240 ms | 0.005 |
| report (`/reports/<session-id>`) | **98** | **100** | 0.8 s | 2.3 s | 40 ms | 0 |

Fixes made to reach this, measured before and after:

| Page | Before (performance / accessibility) | After |
|---|---|---|
| landing | 90 / 96 | 94 / 100 |
| report | 88 / 91 | 98 / 100 |

- The 3D avatar on the landing page loads once the page is idle.
- The report is server-rendered.
- Light-theme accent and status colours were darkened to at least 4.5:1 on their tints.
- Faded (`opacity`) text was removed from uncovered skill cards.
- Meters with no value no longer claim the `meter` role.

Remaining non-passing audit: `bf-cache`. Pages that hold a WebSocket or `no-store` responses cannot use the back/forward cache by design.
