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
- When Supabase is configured, login/sign-up use Supabase Auth and analysis history is stored in the `analysis_jobs`, `analysis_topics`, and `generated_articles` tables.
- If you want the safest startup path on Windows, run `.\run.ps1` from the project root. It uses the project virtual environment automatically.

## Run with Docker

```bash
docker build -t zenpen .
docker run -p 8000:8000 zenpen
```
