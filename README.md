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
- Some YouTube videos now require authenticated cookies. For cloud deployments, add `YOUTUBE_COOKIES_B64` as a base64-encoded Netscape cookies file exported from a browser session that can access the target YouTube videos.
- The backend loads `GEMINI_API_KEY` from a root `.env` file or your shell environment, and falls back to a local summary if it is unavailable.
- To use Gemini locally, create a `.env` file in the project root based on `.env.example`.
- To connect Supabase properly, add `SUPABASE_URL`, `SUPABASE_PUBLISHABLE_KEY`, and `SUPABASE_SECRET_KEY` to `.env`, then run the SQL in [supabase/schema.sql](./supabase/schema.sql) inside the Supabase SQL editor.
- If `APIFY_TOKEN` is configured, the backend will try Apify's `streamers/youtube-video-downloader` actor first for YouTube URLs, then fall back to local `yt-dlp`.
- When Supabase is configured, login/sign-up use Supabase Auth and analysis history is stored in the `analysis_jobs`, `analysis_topics`, and `generated_articles` tables.
- If you want the safest startup path on Windows, run `.\run.ps1` from the project root. It uses the project virtual environment automatically.

## Run with Docker

```bash
docker build -t zenpen .
docker run -p 8000:8000 zenpen
```

## Recommended cloud deployment

This app uses heavy Python/media dependencies such as Whisper, FFmpeg, MoviePy, FAISS, and sentence-transformers. Those are not a good fit for Vercel Serverless Functions.

For a complete working deployment, use a container-friendly platform such as Render, Railway, Fly.io, or any VM/container host.

### Render

1. Push this repository to GitHub.
2. Create a new Render Web Service from the repo.
3. Render will detect [render.yaml](./render.yaml) and the [Dockerfile](./Dockerfile).
4. Add these environment variables in Render:
   - `GEMINI_API_KEY`
   - `SUPABASE_URL`
   - `SUPABASE_PUBLISHABLE_KEY`
   - `SUPABASE_SECRET_KEY`
   - `SUPABASE_JWKS_URL`
   - `APIFY_TOKEN` (recommended for more reliable YouTube downloads)
   - `YOUTUBE_COOKIES_B64` (optional, but recommended if YouTube asks for sign-in)
5. Deploy.

The Docker image installs `ffmpeg` and `libgomp1`, uses Python 3.11, and runs the full FastAPI app so the frontend and backend stay on the same origin.
