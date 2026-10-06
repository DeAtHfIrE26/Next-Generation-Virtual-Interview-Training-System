# What I need from you (Kashyap)

Everything here physically requires you: an account, a key, a payment, a recording, or a legal decision.
I keep working on everything else in the meantime. Items are ordered by how much they unblock.

**Never paste a key into chat, a GitHub comment or a commit.** Put keys in one of two places:

- **Your machine (docker compose):** the `.env` file at the repo root. Copy `.env.example` to `.env` first. `.env` is git-ignored.
- **This cloud workspace (so I can test with real providers here):** open the environment menu in the session title bar, choose **Edit**, and add the key as an **environment variable** with the exact name given below. A **new session** picks it up, so after adding keys, start a new session and tell me "keys added".
- **Hosted (Vercel):** Project `interview-coach-api`, then **Settings → Environment Variables**, same names, then redeploy.

Status legend: **BLOCKING** means a Definition-of-Done item cannot be proven without it. **UPGRADE** means it works today with a free or local fallback, but the premium version needs this.

---

## 1. An LLM key — BLOCKING (question engine evidence, real end-to-end test)

The interviewer is now 100% LLM-generated, and there is no question bank. To produce the 20 mock-interview transcripts and the end-to-end video here, I need one working LLM. Pick **one** (A is recommended; B is free).

**A. Anthropic Claude (recommended: best interviewer quality, reachable from this workspace)**
1. Go to https://console.anthropic.com and sign up or log in.
2. **Settings → Billing**: add a card and buy $10 of credits. 20 mock interviews plus testing should cost well under $10.
3. **Settings → API Keys → Create Key**, named `interview-coach`.
4. Add these variables:
   - `ANTHROPIC_API_KEY` = the key (starts `sk-ant-`)
   - `LLM_PROVIDER` = `anthropic`

**B. Google Gemini (free tier, also reachable from this workspace)**
1. Go to https://aistudio.google.com/apikey and sign in with a Google account.
2. Click **Create API key**. The free tier is fine for testing.
3. Add these variables:
   - `GEMINI_API_KEY` = the key
   - `LLM_PROVIDER` = `gemini`

**C. Fully local, no key (your machine only)**
- `docker compose --profile local-llm up` starts Ollama and pulls a local model automatically.
- On CPU it is noticeably slower and less sharp than A or B, but it is free and private.

**Best setup:** set both A and B, plus `LLM_FALLBACK_PROVIDER` = `gemini`, so a Claude outage falls back to Gemini instead of the emergency question.

## 2. Allow model downloads in this workspace — BLOCKING for running the local LLM here

This workspace's network policy blocks the hosts that serve local models and some voice providers. To let me run the free/local stack and the premium voice providers here:

1. Open the environment menu in the session title bar, choose **Edit**, then **Network access**.
2. Choose **Custom** and keep the default package-manager list.
3. Add these allowed domains: `huggingface.co`, `cdn-lfs.huggingface.co`, `cas-bridge.xethub.hf.co`, `ollama.com`, `registry.ollama.ai`, `api.deepgram.com`, `api.elevenlabs.io`, `api.openai.com`.
4. Start a new session.

Steps are documented at https://code.claude.com/docs/en/cloud-environments#network-access.

## 3. Premium speech-to-text: Deepgram — UPGRADE

Today the pipeline uses sherpa-onnx streaming recognition, running locally and free with real partial transcripts. Deepgram Nova-3 is more accurate, especially on Indian-English accents.

1. Go to https://console.deepgram.com and sign up. New accounts get free credit; no card is needed to start.
2. **API Keys → Create a New API Key**, role **Member**.
3. Add these variables:
   - `DEEPGRAM_API_KEY` = the key
   - `STT_PROVIDER` = `deepgram`

## 4. Premium, human-sounding voice — UPGRADE (pick one)

Today the voice is Kokoro, running locally and free. Lip-sync comes from the actual audio.

**A. ElevenLabs (most natural voice; returns character timestamps for exact lip-sync)**
1. Go to https://elevenlabs.io and sign up. The Starter plan (about $5 a month) is enough for testing and includes a commercial licence.
2. **Profile → API Keys → Create**.
3. Pick a voice in the Voice Library and copy its **Voice ID**.
4. Add these variables:
   - `ELEVENLABS_API_KEY` = the key
   - `ELEVENLABS_VOICE_ID` = the voice ID
   - `TTS_PROVIDER` = `elevenlabs`

**B. Amazon Polly Neural/Generative (returns viseme timings; reachable from this workspace, so I can test it here)**
1. In the AWS console, open **IAM → Users → Create user** and name it `interview-coach-polly`.
2. Attach an inline policy that allows only `polly:SynthesizeSpeech` and `polly:DescribeVoices`.
3. **Security credentials → Create access key → Application running outside AWS.**
4. Add these variables:
   - `AWS_ACCESS_KEY_ID` = the access key ID
   - `AWS_SECRET_ACCESS_KEY` = the secret access key
   - `AWS_REGION` = `ap-south-1`
   - `TTS_PROVIDER` = `polly`

## 5. Avatar — decision pending research

I am benchmarking 3D avatar sources and photoreal streaming-avatar services, checking each one's licence and availability. The recommendation and any purchase or key it needs will be added here, with exact steps. Until then, the 3D avatar ships with a commercially licensed default.

## 6. Real voice recordings for testing — UPGRADE (accuracy evidence)

I can test with synthesized speech, but real people's voices are the honest test for Indian-English accents. I will not present synthesized audio as real recordings.

1. Record 10–20 short answers (20–60 seconds each) from 4–6 people with different Indian-English accents. Mix men and women, and quiet and noisy rooms.
2. Each person signs the consent form in `eval/DATA_COLLECTION.md` (the consent ID goes in the manifest).
3. Format: WAV, 16 kHz, mono. Name each file `speakerID_answerN.wav`. Add a `.txt` file next to each one with the exact words spoken.
4. Upload them to a private Google Drive folder and tell me when they are there. Do not commit them to the public repo.

## 7. Carried over from the last pass (still open)

- **Rotate the leaked keys now.** That means the old Mistral key and the RapidAPI/Judge0 key; both are in the repo's git history.
- **Rotate the preview `API_SHARED_SECRET`** on both Vercel projects. It appeared in an earlier session log.
- **Patent:** send the filed claims (not just the abstract), so the Claim Map can be keyed to the actual claim numbers.
- **IP ownership:** settle VIT ownership and licensing of the patent and the code. This is the largest commercial blocker.
- **Licence:** choose a licence for the repo. See `docs/LICENSE_DECISION.md`.
- **Lawyer review** of `COMPLIANCE.md` and the privacy notice before any public launch.
- **Persistent database for the hosted preview (optional):**
  1. In Vercel, open **Storage → Create → Neon (Postgres)**, free plan.
  2. Connect it to `interview-coach-api`.
  3. Set `DATABASE_URL` to the `postgresql+psycopg://...` form of the connection string.
