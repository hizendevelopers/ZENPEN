# ZENPEN

AI article, blog, and subtitle generation platform built with a FastAPI backend and a lightweight web frontend.

## Run locally

```bash
.\.venv\Scripts\python.exe -m pip install -r backend/requirements.txt
.\.venv\Scripts\python.exe -m uvicorn backend.app:app --host 127.0.0.1 --port 8000
```

Then open http://127.0.0.1:8000/

Notes:
- `ffmpeg` must be available on your `PATH` for YouTube audio downloads and video-to-audio extraction.
- The backend loads `GEMINI_API_KEY` from a root `.env` file or your shell environment, and falls back to a local summary if it is unavailable.
- To use Gemini locally, create a `.env` file in the project root based on `.env.example`.
- To connect Supabase properly, add `SUPABASE_URL`, `SUPABASE_PUBLISHABLE_KEY`, and `SUPABASE_SECRET_KEY` to `.env`, then run the SQL in [supabase/schema.sql](./supabase/schema.sql) inside the Supabase SQL editor.
- `Upload file` analysis stays synchronous.
- `URL analysis` can run through a Redis-backed background queue so YouTube and remote downloads do not block the main request thread.
- If `APIFY_TOKEN` is configured, the worker tries Apify's `streamers/youtube-video-downloader` actor before falling back to local `yt-dlp`.
- If YouTube media download fails on the server, the worker falls back to fetching the YouTube transcript directly when available.
- For cloud IP blocks, configure a rotating residential proxy through `YOUTUBE_PROXY_HTTP` and `YOUTUBE_PROXY_HTTPS`.
- Some YouTube videos may still require authenticated cookies. For cloud deployments, add `YOUTUBE_COOKIES_B64` as a base64-encoded Netscape cookies file exported from a browser session that can access the target YouTube videos.
- When Supabase is configured, login/sign-up use Supabase Auth and analysis history is stored in the `analysis_jobs`, `analysis_topics`, and `generated_articles` tables.
- If you want the safest startup path on Windows, run `.\run.ps1` from the project root. It uses the project virtual environment automatically.

## Run with Docker

```bash
docker build -t zenpen .
docker run -p 8000:8000 zenpen
```

## Run the queue worker locally

For URL analysis jobs, start Redis first and then run:

```bash
.\.venv\Scripts\python.exe -m backend.worker
```

The web app will automatically enqueue URL jobs whenever `REDIS_URL` points to a working Redis instance.

## Recommended cloud deployment

This app uses heavy Python/media dependencies such as Whisper, FFmpeg, MoviePy, FAISS, sentence-transformers, and YouTube downloader/transcript tooling. Those are not a good fit for Vercel Serverless Functions.

For a complete working deployment, use a container-friendly platform such as Render, Railway, Fly.io, or any VM/container host.

### Production layout

1. `Main web app`
   - serves the SPA + FastAPI API
   - handles authentication, uploads, history, and article generation
2. `Worker service`
   - pulls URL analysis jobs from Redis
   - handles YouTube transcript/download retries away from the request thread
3. `Redis`
   - stores queued URL jobs and retry state
4. `Optional residential proxy`
   - routes YouTube transcript/download requests through non-cloud IPs for better reliability

### Render

1. Push this repository to GitHub.
2. Create services from [render.yaml](./render.yaml).
3. Render will provision:
   - `zenpen-web`
   - `zenpen-worker`
   - `zenpen-redis`
4. Add these environment variables to both web and worker services:
   - `GEMINI_API_KEY`
   - `SUPABASE_URL`
   - `SUPABASE_PUBLISHABLE_KEY`
   - `SUPABASE_SECRET_KEY`
   - `SUPABASE_JWKS_URL`
   - `APIFY_TOKEN`
   - `YOUTUBE_COOKIES_B64`
   - `YOUTUBE_PROXY_HTTP` and `YOUTUBE_PROXY_HTTPS` when using a residential proxy

### Railway

Create three services from the same repo:

1. `zenpen-web`
   - Start command: `python -m uvicorn backend.app:app --host 0.0.0.0 --port $PORT`
2. `zenpen-worker`
   - Start command: `python -m backend.worker`
3. `redis`
   - Railway Redis service

Then set `REDIS_URL` on both the web and worker services, plus the same app env vars listed above.
