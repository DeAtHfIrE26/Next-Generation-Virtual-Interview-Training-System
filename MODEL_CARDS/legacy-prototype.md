# Legacy prototype (reference only)

`legacy/desktop` and `legacy/futuristic` are the original research prototypes. Their algorithms are re-implemented exactly in `interview_core.legacy` and pinned by characterization tests, so the original patent-element behaviour stays reproducible and serves as the "before" system in every evaluation suite.

Do not deploy the legacy code:
- `yolov8n.pt` is AGPL-3.0; InsightFace `buffalo_l` is non-commercial; PyMuPDF is AGPL-3.0; the free Google speech endpoint is not licensed for production.
- Its histogram "face recognition" cannot tell people apart, its lip-sync check cannot fire, and its report padded missing data with random values. These defects are documented in `docs/CLAIM_MAP.md`.
