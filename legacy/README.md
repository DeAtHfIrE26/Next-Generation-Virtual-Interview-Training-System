# Legacy desktop prototypes (frozen reference)

These directories hold the original research prototypes, kept for history and as the behavioural reference for the patented mechanisms. They are **not** the product and are not maintained.

| Path | Origin | Notes |
|---|---|---|
| `desktop/` | `Next-Generation-Virtual-Interview-Training-System` @ `d821924` (moved with `git mv`) | Tkinter app described in the ICCCNT-2025 paper. Entry point: `main.py`. |
| `futuristic/` | `futuristic-ai-interviewer` @ `358ea09` (imported with `git subtree add`, full history) | Later prototype with InsightFace and Resemblyzer. Entry point: `GPTUpdate.py`. |

Changes made here since import (milestone M0):
- Hard-coded API keys replaced with `MISTRAL_API_KEY` and `JUDGE0_RAPIDAPI_KEY` environment variables.
- Session logs, real-session reports, voice recordings and synthetic reports/resumes removed from the tree.
- `futuristic/chatGPT.py` syntax errors fixed. `futuristic/fix.py` converted from UTF-16 to UTF-8.

**Known limitation:** `desktop/requirements.txt` pins (for example `mediapipe==0.10.3`, `pyannote.audio==2.1.1`) do not resolve on Python 3.11 or later. They are left as originally committed. The supported implementation is `packages/core` plus `services/api` plus `apps/web`. See `docs/CLAIM_MAP.md` for where each patented element now lives.

Licensing: `yolov8n.pt` is AGPL-3.0 (Ultralytics), and InsightFace `buffalo_l` weights are non-commercial. Neither is used by the product code.
