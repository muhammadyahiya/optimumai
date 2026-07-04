# Deploy the dashboard

`optimumai dashboard` runs the Streamlit progress dashboard locally
(`pip install "optimumai[dashboard]"`). To share it as a public link — handy for
teaching, demos, or a portfolio — deploy it to a free host. Both options below run
the *same* app: `src/optimumai/dashboard/app.py`.

## Streamlit Community Cloud

1. Push this repo to GitHub (already done for the canonical project).
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**.
3. Point it at the repo, branch `main`, main file
   `src/optimumai/dashboard/app.py`.
4. Add a `requirements.txt` (or reuse the project) containing `optimumai[dashboard]`.
5. Deploy — you get a public `*.streamlit.app` URL.

## Hugging Face Spaces

1. Create a new **Space** → SDK: **Streamlit**.
2. In the Space repo, add a one-line `app.py`:

   ```python
   from optimumai.dashboard import app  # runs the Streamlit script on import
   ```

3. Add `requirements.txt` with `optimumai[dashboard]`.
4. Push — the Space builds and serves a public URL.

Both are free and give you a shareable link without running a server yourself.
