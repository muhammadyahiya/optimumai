# Deploy

`optimumai dashboard` runs the Streamlit progress dashboard locally
(`pip install "optimumai[dashboard]"`). To share it as a public link — for
teaching, demos, or a portfolio — deploy it to a free host. Both options below
run the same app: `src/optimumai/dashboard/app.py`.

```bash
pip install "optimumai[dashboard]"
optimumai dashboard                    # localhost:8501
optimumai dashboard --port 8888
```

---

## Option 1 — Streamlit Community Cloud (recommended)

1. Push this repo to GitHub (already done for the canonical project).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Point it at the repo, branch `main`, main file
   `src/optimumai/dashboard/app.py`.
4. Add a `requirements.txt` (or reuse the project) containing
   `optimumai[dashboard]`.
5. Click **Deploy** — you get a public `*.streamlit.app` URL in under a
   minute.

---

## Option 2 — Hugging Face Spaces

1. Create a new **Space** → SDK: **Streamlit**.
2. In the Space repo, add a one-line `app.py`:

   ```python
   from optimumai.dashboard import app   # runs the Streamlit script on import
   ```

3. Add `requirements.txt` with `optimumai[dashboard]`.
4. Push — the Space builds automatically and serves a public URL.

---

Both options are free and give you a shareable link without running a server
yourself. The dashboard is read-only (it only reads
`~/.optimumai/progress.json` on the server side), so there are no security
concerns with public deployment.

---

## What the dashboard shows

- Per-track completion breakdown (progress bars per track)
- Overall completion percentage
- "What's next" recommendation (the first incomplete lesson in the next track)
- Quiz performance history
- Spaced-repetition review schedule (what's due and when)

---

## Custom progress file path

If you want the deployed dashboard to reflect a different user's progress,
set the `OPTIMUMAI_PROGRESS_PATH` environment variable in your Space or
Streamlit app settings before deploying.
