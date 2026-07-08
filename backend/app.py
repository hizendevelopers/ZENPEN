from __future__ import annotations

import base64
import importlib
import importlib.util
import json
import os
import re
import shutil
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import nltk
import numpy as np
import httpx
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel
from starlette.staticfiles import StaticFiles
from nltk.tokenize import sent_tokenize
from dotenv import load_dotenv
from rq.job import Job

from backend.queueing import fetch_job, queue_is_available, queue_is_configured
from backend.url_jobs import enqueue_url_analysis

BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
HISTORY_FILE = BASE_DIR / "backend" / "history.json"
WHISPER_CACHE_DIR = BASE_DIR / "backend" / ".cache" / "whisper"
ENV_FILE = BASE_DIR / ".env"

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
load_dotenv(ENV_FILE)

app = FastAPI(title="Media Transcriber & Analyzer")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_PUBLISHABLE_KEY = os.getenv("SUPABASE_PUBLISHABLE_KEY", "")
SUPABASE_SECRET_KEY = os.getenv("SUPABASE_SECRET_KEY", "")
YOUTUBE_COOKIES_B64 = os.getenv("YOUTUBE_COOKIES_B64", "").strip()
APIFY_TOKEN = os.getenv("APIFY_TOKEN", "").strip()
REDIS_URL = os.getenv("REDIS_URL", "").strip()
YOUTUBE_PROXY_URL = os.getenv("YOUTUBE_PROXY_URL", "").strip()
YOUTUBE_PROXY_HTTP = os.getenv("YOUTUBE_PROXY_HTTP", "").strip()
YOUTUBE_PROXY_HTTPS = os.getenv("YOUTUBE_PROXY_HTTPS", "").strip()
FAST_ANALYSIS_MODE = os.getenv("FAST_ANALYSIS_MODE", "true").strip().lower() not in {"0", "false", "no", "off"}
FAST_ANALYSIS_TRANSCRIPT_LIMIT = int(os.getenv("FAST_ANALYSIS_TRANSCRIPT_LIMIT", "5000"))
FAST_ANALYSIS_CHUNK_LIMIT = int(os.getenv("FAST_ANALYSIS_CHUNK_LIMIT", "6"))
ENABLE_BACKGROUND_URL_JOBS = os.getenv("ENABLE_BACKGROUND_URL_JOBS", "false").strip().lower() in {"1", "true", "yes", "on"}
GEMINI_BACKOFF_UNTIL = 0.0

WHISPER_MODEL = None
EMBEDDER = None
GENAI_CLIENT = None
FAISS_MODULE = None
YT_DLP_MODULE = None
VIDEO_FILE_CLIP_CLASS = None
SENTENCE_TRANSFORMER_CLASS = None
WHISPER_MODULE = None
GENAI_MODULE = None
APIFY_CLIENT_CLASS = None
YOUTUBE_TRANSCRIPT_API_CLASS = None


class ArticlesRequest(BaseModel):
    headline: str
    summary: str
    topics: list[str] = []
    selected_topics: list[str]
    article_count: int = 1


class QuietYtdlpLogger:
    def debug(self, msg: str) -> None:
        return None

    def warning(self, msg: str) -> None:
        return None

    def error(self, msg: str) -> None:
        return None


class SignupRequest(BaseModel):
    name: str
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


COMMON_TOPIC_STOPWORDS = {
    "about", "after", "again", "also", "amid", "among", "around", "being", "below", "between",
    "both", "could", "first", "from", "further", "here", "into", "just", "main", "more", "most",
    "news", "only", "other", "over", "same", "some", "such", "than", "that", "their", "there",
    "these", "they", "this", "those", "through", "under", "using", "very", "what", "when", "where",
    "which", "while", "with", "would", "your", "have", "were", "been", "them", "because", "content",
    "video", "transcript", "summary", "give", "breaking", "points", "said", "says", "show", "kind",
    "then", "taken", "single", "generation", "musical", "story", "system", "shown", "last", "this", "that",
}

ALLOWED_ARTICLE_TAGS = {"h2", "h3", "p", "ul", "li", "strong", "br"}


def dependency_status() -> dict[str, bool | str]:
    return {
        "python": sys.executable,
        "yt_dlp": module_available("yt_dlp"),
        "whisper": module_available("whisper"),
        "moviepy": module_available("moviepy"),
        "faiss": module_available("faiss"),
        "sentence_transformers": module_available("sentence_transformers"),
        "gemini": module_available("google.genai"),
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "gemini_api_key_configured": bool(GEMINI_API_KEY),
        "apify_configured": bool(APIFY_TOKEN),
        "youtube_transcript_api": module_available("youtube_transcript_api"),
        "supabase_configured": supabase_is_configured(),
        "redis_queue_configured": queue_is_configured(),
        "redis_queue_available": queue_is_available(),
        "background_url_jobs_enabled": ENABLE_BACKGROUND_URL_JOBS,
        "youtube_proxy_configured": bool(get_proxy_url("http") or get_proxy_url("https")),
    }


def module_available(module_name: str) -> bool:
    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


def get_genai_module():
    global GENAI_MODULE
    if GENAI_MODULE is None:
        try:
            GENAI_MODULE = importlib.import_module("google.genai")
        except Exception as exc:
            raise RuntimeError(build_missing_dependency_message("google-genai")) from exc
    return GENAI_MODULE


def get_whisper_module():
    global WHISPER_MODULE
    if WHISPER_MODULE is None:
        try:
            WHISPER_MODULE = importlib.import_module("whisper")
        except Exception as exc:
            raise RuntimeError(build_missing_dependency_message("Whisper")) from exc
    return WHISPER_MODULE


def get_sentence_transformer_class():
    global SENTENCE_TRANSFORMER_CLASS
    if SENTENCE_TRANSFORMER_CLASS is None:
        try:
            module = importlib.import_module("sentence_transformers")
            SENTENCE_TRANSFORMER_CLASS = getattr(module, "SentenceTransformer")
        except Exception as exc:
            raise RuntimeError(build_missing_dependency_message("sentence-transformers")) from exc
    return SENTENCE_TRANSFORMER_CLASS


def get_yt_dlp_module():
    global YT_DLP_MODULE
    if YT_DLP_MODULE is None:
        try:
            YT_DLP_MODULE = importlib.import_module("yt_dlp")
        except Exception as exc:
            raise RuntimeError(build_missing_dependency_message("yt-dlp")) from exc
    return YT_DLP_MODULE


def get_video_file_clip_class():
    global VIDEO_FILE_CLIP_CLASS
    if VIDEO_FILE_CLIP_CLASS is None:
        try:
            module = importlib.import_module("moviepy.editor")
            VIDEO_FILE_CLIP_CLASS = getattr(module, "VideoFileClip")
        except Exception:
            try:
                module = importlib.import_module("moviepy")
                VIDEO_FILE_CLIP_CLASS = getattr(module, "VideoFileClip")
            except Exception as exc:
                raise RuntimeError(build_missing_dependency_message("moviepy")) from exc
    return VIDEO_FILE_CLIP_CLASS


def get_faiss_module():
    global FAISS_MODULE
    if FAISS_MODULE is None:
        try:
            FAISS_MODULE = importlib.import_module("faiss")
        except Exception as exc:
            raise RuntimeError(build_missing_dependency_message("faiss-cpu")) from exc
    return FAISS_MODULE


def get_apify_client_class():
    global APIFY_CLIENT_CLASS
    if APIFY_CLIENT_CLASS is None:
        try:
            module = importlib.import_module("apify_client")
            APIFY_CLIENT_CLASS = getattr(module, "ApifyClient")
        except Exception as exc:
            raise RuntimeError(build_missing_dependency_message("apify-client")) from exc
    return APIFY_CLIENT_CLASS


def get_youtube_transcript_api_class():
    global YOUTUBE_TRANSCRIPT_API_CLASS
    if YOUTUBE_TRANSCRIPT_API_CLASS is None:
        try:
            module = importlib.import_module("youtube_transcript_api")
            YOUTUBE_TRANSCRIPT_API_CLASS = getattr(module, "YouTubeTranscriptApi")
        except Exception as exc:
            raise RuntimeError(build_missing_dependency_message("youtube-transcript-api")) from exc
    return YOUTUBE_TRANSCRIPT_API_CLASS


def supabase_is_configured() -> bool:
    return bool(SUPABASE_URL and SUPABASE_PUBLISHABLE_KEY and SUPABASE_SECRET_KEY)


def get_api_config() -> dict[str, object]:
    return {
        "supabase": {
            "enabled": bool(SUPABASE_URL and SUPABASE_PUBLISHABLE_KEY),
            "url": SUPABASE_URL or None,
            "publishableKey": SUPABASE_PUBLISHABLE_KEY or None,
        },
        "queue": {
            "enabled": False,
        },
    }


def background_url_jobs_available() -> bool:
    return ENABLE_BACKGROUND_URL_JOBS and queue_is_available()


def get_bearer_token(request: Request) -> Optional[str]:
    authorization = request.headers.get("Authorization", "").strip()
    if not authorization.lower().startswith("bearer "):
        return None
    token = authorization.split(" ", 1)[1].strip()
    return token or None


def supabase_auth_headers(token: str) -> dict[str, str]:
    return {
        "apikey": SUPABASE_PUBLISHABLE_KEY,
        "Authorization": f"Bearer {token}",
    }


def supabase_service_headers() -> dict[str, str]:
    return {
        "apikey": SUPABASE_SECRET_KEY,
        "Authorization": f"Bearer {SUPABASE_SECRET_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


def supabase_public_headers() -> dict[str, str]:
    return {
        "apikey": SUPABASE_PUBLISHABLE_KEY,
        "Content-Type": "application/json",
    }


def supabase_auth_request(
    method: str,
    path: str,
    *,
    params: Optional[dict[str, str]] = None,
    json_body: Optional[object] = None,
    admin: bool = False,
) -> httpx.Response:
    if not supabase_is_configured():
        raise RuntimeError("Supabase is not configured")

    headers = supabase_service_headers() if admin else supabase_public_headers()
    response = httpx.request(
        method,
        f"{SUPABASE_URL}/auth/v1/{path}",
        headers=headers,
        params=params,
        json=json_body,
        timeout=20.0,
    )
    return response


def extract_supabase_error(response: httpx.Response, default_message: str) -> str:
    try:
        payload = response.json()
    except Exception:
        return default_message

    if isinstance(payload, dict):
        for key in ("msg", "message", "error_description", "error"):
            value = payload.get(key)
            if value:
                return str(value)
    return default_message


def create_supabase_user(name: str, email: str, password: str) -> dict[str, object]:
    response = supabase_auth_request(
        "POST",
        "admin/users",
        json_body={
            "email": email,
            "password": password,
            "email_confirm": True,
            "user_metadata": {
                "full_name": name,
            },
        },
        admin=True,
    )

    if response.status_code >= 400:
        raise HTTPException(
            status_code=400,
            detail=extract_supabase_error(response, "Could not create your account."),
        )

    payload = response.json()
    return payload if isinstance(payload, dict) else {}


def sign_in_supabase_user(email: str, password: str) -> dict[str, object]:
    response = supabase_auth_request(
        "POST",
        "token",
        params={"grant_type": "password"},
        json_body={
            "email": email,
            "password": password,
        },
    )

    if response.status_code >= 400:
        raise HTTPException(
            status_code=400,
            detail=extract_supabase_error(response, "Invalid email or password."),
        )

    payload = response.json()
    return payload if isinstance(payload, dict) else {}


def resolve_supabase_user(request: Request) -> Optional[dict[str, object]]:
    if not supabase_is_configured():
        return None

    token = get_bearer_token(request)
    if not token:
        return None

    try:
        response = httpx.get(
            f"{SUPABASE_URL}/auth/v1/user",
            headers=supabase_auth_headers(token),
            timeout=15.0,
        )
        if response.status_code != 200:
            return None
        payload = response.json()
        return payload if isinstance(payload, dict) and payload.get("id") else None
    except Exception:
        return None


def supabase_rest_request(
    method: str,
    path: str,
    *,
    params: Optional[dict[str, str]] = None,
    json_body: Optional[object] = None,
) -> object:
    if not supabase_is_configured():
        raise RuntimeError("Supabase is not configured")

    response = httpx.request(
        method,
        f"{SUPABASE_URL}/rest/v1/{path}",
        headers=supabase_service_headers(),
        params=params,
        json=json_body,
        timeout=20.0,
    )
    response.raise_for_status()
    if not response.content:
        return None
    return response.json()


@lru_cache(maxsize=8)
def get_product_id_by_slug(slug: str) -> Optional[str]:
    rows = supabase_rest_request(
        "GET",
        "products",
        params={
            "slug": f"eq.{slug}",
            "select": "id",
            "limit": "1",
        },
    )
    if isinstance(rows, list) and rows:
        product_id = rows[0].get("id")
        return str(product_id) if product_id else None
    return None


def load_history_from_supabase(user_id: str) -> list[dict[str, object]]:
    rows = supabase_rest_request(
        "GET",
        "analysis_jobs",
        params={
            "user_id": f"eq.{user_id}",
            "select": "id,source_type,source_url,source_file_name,headline,summary,created_at",
            "order": "created_at.desc",
            "limit": "10",
        },
    )
    history_items: list[dict[str, object]] = []
    if not isinstance(rows, list):
        return history_items

    for row in rows:
        source_type = row.get("source_type") or "url"
        source_label = "youtube-url" if source_type == "url" else "uploaded-file"
        history_items.append(
            {
                "id": row.get("id"),
                "source": source_label,
                "headline": row.get("headline") or "Media summary",
                "summary": row.get("summary") or "",
                "created_at": row.get("created_at"),
            }
        )
    return history_items


def persist_analysis_to_supabase(
    *,
    user_id: str,
    source_type: str,
    source_url: Optional[str],
    source_file_name: Optional[str],
    source_mime_type: Optional[str],
    query: str,
    result: dict[str, object],
    selected_topics: list[str],
) -> None:
    product_id = get_product_id_by_slug("article-generator")
    if not product_id:
        raise RuntimeError("The article-generator product row is missing in Supabase.")

    headline = str(result.get("headline") or "Media summary")
    summary = str(result.get("summary") or "")
    topics = result.get("topics") if isinstance(result.get("topics"), list) else []
    articles = result.get("articles") if isinstance(result.get("articles"), list) else []

    inserted_jobs = supabase_rest_request(
        "POST",
        "analysis_jobs",
        json_body=[
            {
                "user_id": user_id,
                "product_id": product_id,
                "source_type": source_type,
                "source_url": source_url,
                "source_file_name": source_file_name,
                "source_mime_type": source_mime_type,
                "query": query,
                "status": "completed",
                "headline": headline,
                "summary": summary,
                "raw_result": result,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
        ],
    )
    if not isinstance(inserted_jobs, list) or not inserted_jobs:
        raise RuntimeError("Supabase did not return the created analysis job.")

    analysis_job_id = inserted_jobs[0].get("id")
    if not analysis_job_id:
        raise RuntimeError("Supabase did not return an analysis job id.")

    topic_rows = []
    selected_lookup = {topic.lower() for topic in selected_topics}
    for index, topic in enumerate(topics):
        topic_text = str(topic).strip()
        if not topic_text:
            continue
        topic_rows.append(
            {
                "analysis_job_id": analysis_job_id,
                "topic": topic_text,
                "sort_order": index,
                "selected": topic_text.lower() in selected_lookup,
            }
        )

    inserted_topics: list[dict[str, object]] = []
    if topic_rows:
        rows = supabase_rest_request("POST", "analysis_topics", json_body=topic_rows)
        if isinstance(rows, list):
            inserted_topics = rows

    topic_id_by_name = {
        str(item.get("topic")).lower(): item.get("id")
        for item in inserted_topics
        if item.get("topic") and item.get("id")
    }

    article_rows = []
    for index, article in enumerate(articles):
        if not isinstance(article, dict):
            continue
        topic = str(article.get("topic") or "").strip()
        content = str(article.get("content") or "").strip()
        if not content:
            continue
        article_rows.append(
            {
                "analysis_job_id": analysis_job_id,
                "topic_id": topic_id_by_name.get(topic.lower()) if topic else None,
                "title": headline,
                "topic": topic or "Main topic",
                "content": content,
                "image_url": article.get("image_url"),
                "sort_order": index,
            }
        )

    if article_rows:
        supabase_rest_request("POST", "generated_articles", json_body=article_rows)


def build_missing_dependency_message(package_name: str, extra_hint: Optional[str] = None) -> str:
    message = (
        f"{package_name} is not available in the Python interpreter running this server: {sys.executable}. "
        "Start the app with .\\.venv\\Scripts\\python.exe or activate the virtual environment first."
    )
    if extra_hint:
        message = f"{message} {extra_hint}"
    return message


def get_proxy_url(scheme: str = "https") -> Optional[str]:
    if scheme == "https":
        return YOUTUBE_PROXY_HTTPS or YOUTUBE_PROXY_URL or YOUTUBE_PROXY_HTTP or None
    return YOUTUBE_PROXY_HTTP or YOUTUBE_PROXY_URL or YOUTUBE_PROXY_HTTPS or None


def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg is not available on PATH. Install ffmpeg and restart the server."
        )


def get_genai_client():
    global GENAI_CLIENT
    if not GEMINI_API_KEY:
        raise RuntimeError("Gemini is not configured")
    if GENAI_CLIENT is None:
        genai = get_genai_module()
        GENAI_CLIENT = genai.Client(api_key=GEMINI_API_KEY)
    return GENAI_CLIENT


def parse_retry_delay_seconds(message: str) -> int:
    match = re.search(r"retry in\s+(\d+)", message, re.IGNORECASE)
    if match:
        return max(int(match.group(1)), 30)
    return 60


def should_skip_gemini() -> bool:
    return GEMINI_BACKOFF_UNTIL > time.time()


def gemini_generate_text(prompt: str) -> str:
    global GEMINI_BACKOFF_UNTIL
    if should_skip_gemini():
        raise RuntimeError("Gemini is temporarily cooling down after a quota/rate-limit response.")

    client = get_genai_client()
    try:
        response = client.models.generate_content(
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            contents=prompt,
        )
        return getattr(response, "text", "").strip()
    except Exception as exc:
        error_text = str(exc)
        if "429" in error_text or "RESOURCE_EXHAUSTED" in error_text or "rate limit" in error_text.lower():
            GEMINI_BACKOFF_UNTIL = time.time() + parse_retry_delay_seconds(error_text)
        raise


def load_history() -> list[dict]:
    if not HISTORY_FILE.exists():
        return []
    try:
        with HISTORY_FILE.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
            return data if isinstance(data, list) else []
    except Exception:
        return []


def save_history(items: list[dict]) -> None:
    try:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with HISTORY_FILE.open("w", encoding="utf-8") as handle:
            json.dump(items, handle, indent=2)
    except OSError:
        # Read-only filesystems such as serverless runtimes should not crash the app.
        return None


def add_history_entry(result: dict[str, str], source: str) -> None:
    history = load_history()
    entry = {
        "id": int(time.time() * 1000),
        "source": source,
        "headline": result.get("headline", "Media summary"),
        "summary": result.get("summary", ""),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    history.insert(0, entry)
    save_history(history[:10])


def get_whisper_model():
    global WHISPER_MODEL
    if WHISPER_MODEL is None:
        whisper = get_whisper_module()
        WHISPER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        WHISPER_MODEL = whisper.load_model("tiny", download_root=str(WHISPER_CACHE_DIR))
    return WHISPER_MODEL


def get_embedder():
    global EMBEDDER
    if EMBEDDER is None:
        sentence_transformer = get_sentence_transformer_class()
        EMBEDDER = sentence_transformer("all-MiniLM-L6-v2")
    return EMBEDDER


def maybe_write_youtube_cookies_file(output_dir: Path) -> Optional[Path]:
    if not YOUTUBE_COOKIES_B64:
        return None
    try:
        decoded = base64.b64decode(YOUTUBE_COOKIES_B64).decode("utf-8")
    except Exception as exc:
        raise RuntimeError("YOUTUBE_COOKIES_B64 is not valid base64-encoded Netscape cookie data.") from exc

    cookies_path = output_dir / "youtube-cookies.txt"
    cookies_path.write_text(decoded, encoding="utf-8")
    return cookies_path


def build_transcript_api_instance():
    transcript_api_class = get_youtube_transcript_api_class()
    http_proxy = get_proxy_url("http")
    https_proxy = get_proxy_url("https")
    if http_proxy or https_proxy:
        proxies_module = importlib.import_module("youtube_transcript_api.proxies")
        generic_proxy_config = getattr(proxies_module, "GenericProxyConfig")
        try:
            return transcript_api_class(
                proxy_config=generic_proxy_config(
                    http_url=http_proxy,
                    https_url=https_proxy,
                )
            )
        except TypeError:
            # Supports lightweight test doubles that do not accept proxy args.
            return transcript_api_class()
    return transcript_api_class()


def is_youtube_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return "youtube.com" in host or "youtu.be" in host


def extract_youtube_video_id(url: str) -> Optional[str]:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if "youtu.be" in host:
        candidate = parsed.path.strip("/").split("/")[0]
        return candidate or None
    if "youtube.com" in host:
        if parsed.path == "/watch":
            for pair in parsed.query.split("&"):
                if pair.startswith("v="):
                    return pair.split("=", 1)[1] or None
        path_parts = [part for part in parsed.path.split("/") if part]
        if len(path_parts) >= 2 and path_parts[0] in {"shorts", "embed", "live"}:
            return path_parts[1]
    return None


def fetch_youtube_transcript_text(url: str) -> str:
    video_id = extract_youtube_video_id(url)
    if not video_id:
        raise RuntimeError("Could not extract a valid YouTube video id from the URL.")

    transcript_api = build_transcript_api_instance()
    try:
        transcript_entries = transcript_api.fetch(video_id, languages=["en", "en-US", "en-GB"])
    except Exception:
        transcript_entries = transcript_api.fetch(video_id)

    transcript_text = " ".join(
        str(getattr(item, "text", "")).replace("\n", " ").strip()
        for item in transcript_entries
        if getattr(item, "text", "")
    ).strip()
    if not transcript_text:
        raise RuntimeError("No transcript text was available for this YouTube video.")
    return transcript_text


def clean_vtt_caption_text(raw_text: str) -> str:
    cleaned_lines: list[str] = []
    seen_recent: str = ""
    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line == "WEBVTT" or line.startswith(("Kind:", "Language:")):
            continue
        if "-->" in line:
            continue
        line = re.sub(r"<\d{2}:\d{2}:\d{2}\.\d{3}>", " ", line)
        line = re.sub(r"</?c[^>]*>", "", line)
        line = re.sub(r"</?[^>]+>", "", line)
        line = " ".join(line.split()).strip()
        if not line:
            continue
        if line == seen_recent:
            continue
        cleaned_lines.append(line)
        seen_recent = line
    return " ".join(cleaned_lines).strip()


def convert_asterisk_bold_to_html(text: str) -> str:
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text or "")


def strip_html_tags(text: str) -> str:
    no_tags = re.sub(r"<[^>]+>", " ", text or "")
    return " ".join(no_tags.split()).strip()


def sanitize_article_html(raw_html: str) -> str:
    if not raw_html.strip():
        return ""

    html = convert_asterisk_bold_to_html(raw_html.strip())
    html = re.sub(r"(?m)^\s*#{3}\s*(.+)$", r"<h3>\1</h3>", html)
    html = re.sub(r"(?m)^\s*#{2}\s*(.+)$", r"<h2>\1</h2>", html)

    if not re.search(r"<(?:h2|h3|p|ul|li)\b", html):
        blocks: list[str] = []
        paragraphs = [block.strip() for block in re.split(r"\n\s*\n", html) if block.strip()]
        for index, paragraph in enumerate(paragraphs):
            cleaned = paragraph.strip()
            if cleaned.startswith("- "):
                items = [item.strip()[2:].strip() for item in cleaned.splitlines() if item.strip().startswith("- ")]
                blocks.append("<ul>" + "".join(f"<li>{item}</li>" for item in items if item) + "</ul>")
            elif index == 0:
                blocks.append(f"<h2>{cleaned}</h2>")
            else:
                blocks.append(f"<p>{cleaned}</p>")
        html = "\n".join(blocks)

    html = re.sub(r"\n{2,}", "\n", html)
    html = re.sub(r"</?(script|style)[^>]*>", "", html, flags=re.IGNORECASE)
    html = re.sub(
        r"</?([a-zA-Z0-9]+)(?:\s+[^>]*)?>",
        lambda match: match.group(0) if match.group(1).lower() in ALLOWED_ARTICLE_TAGS else "",
        html,
    )
    return html.strip()


def sanitize_inline_html(text: str) -> str:
    html = convert_asterisk_bold_to_html(text or "")
    html = re.sub(r"</?(script|style)[^>]*>", "", html, flags=re.IGNORECASE)
    html = re.sub(
        r"</?([a-zA-Z0-9]+)(?:\s+[^>]*)?>",
        lambda match: match.group(0) if match.group(1).lower() == "strong" else "",
        html,
    )
    return html.strip()


def fetch_youtube_subtitles_text(url: str, output_dir: str) -> str:
    yt_dlp = get_yt_dlp_module()
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    cookies_file = maybe_write_youtube_cookies_file(output_dir_path)

    def build_subtitle_opts(proxy_url: Optional[str]) -> dict[str, object]:
        opts: dict[str, object] = {
            "skip_download": True,
            "writesubtitles": True,
            "writeautomaticsub": True,
            "subtitleslangs": ["en", "en-US", "en-GB"],
            "subtitlesformat": "vtt",
            "outtmpl": str(output_dir_path / "subtitle-download.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "socket_timeout": 30,
            "retries": 2,
            "logger": QuietYtdlpLogger(),
            "http_headers": {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/127.0.0.0 Safari/537.36"
                )
            },
        }
        if cookies_file:
            opts["cookiefile"] = str(cookies_file)
        if proxy_url:
            opts["proxy"] = proxy_url
        return opts

    direct_error: Optional[Exception] = None
    try:
        with yt_dlp.YoutubeDL(build_subtitle_opts(None)) as ydl:
            ydl.extract_info(url, download=True)
    except Exception as exc:
        direct_error = exc
        proxy_url = get_proxy_url("https")
        if not proxy_url:
            raise
        with yt_dlp.YoutubeDL(build_subtitle_opts(proxy_url)) as ydl:
            ydl.extract_info(url, download=True)

    subtitle_files = sorted(output_dir_path.glob("*.vtt"))
    if not subtitle_files:
        if direct_error:
            raise RuntimeError(f"No YouTube subtitle file was available for this video. Direct subtitle fetch failed: {direct_error}")
        raise RuntimeError("No YouTube subtitle file was available for this video.")

    best_text = ""
    for subtitle_file in subtitle_files:
        subtitle_text = clean_vtt_caption_text(subtitle_file.read_text(encoding="utf-8", errors="ignore"))
        if len(subtitle_text) > len(best_text):
            best_text = subtitle_text
    if not best_text:
        raise RuntimeError("YouTube subtitle download succeeded, but no readable subtitle text was found.")
    return best_text


def select_apify_download_url(item: dict[str, object]) -> Optional[str]:
    candidate_keys = (
        "downloadUrl",
        "download_url",
        "apify_storage_url",
        "storageUrl",
        "fileUrl",
        "file_url",
        "url",
    )
    for key in candidate_keys:
        value = item.get(key)
        if isinstance(value, str) and value.startswith(("http://", "https://")):
            if key == "url" and is_youtube_url(value):
                continue
            return value
    return None


def find_http_urls_in_payload(payload: object) -> list[str]:
    urls: list[str] = []

    def walk(value: object) -> None:
        if isinstance(value, str):
            if value.startswith(("http://", "https://")):
                urls.append(value)
            return
        if isinstance(value, dict):
            for nested_value in value.values():
                walk(nested_value)
            return
        if isinstance(value, list):
            for nested_value in value:
                walk(nested_value)

    walk(payload)
    return urls


def select_any_non_youtube_url(payload: object) -> Optional[str]:
    for value in find_http_urls_in_payload(payload):
        if not is_youtube_url(value):
            return value
    return None


def parse_json_response_or_raise(response: httpx.Response, context: str) -> object:
    try:
        return response.json()
    except ValueError as exc:
        text = response.text.strip()
        preview = text[:300] if text else "empty response"
        raise RuntimeError(f"{context} returned a non-JSON response: {preview}") from exc


def infer_download_suffix(download_url: str, fallback: str = ".mp3") -> str:
    path = urlparse(download_url).path
    suffix = Path(path).suffix.lower()
    return suffix if suffix else fallback


def download_file_to_path(download_url: str, destination: Path) -> Path:
    with httpx.stream(
        "GET",
        download_url,
        follow_redirects=True,
        timeout=120.0,
        proxy=get_proxy_url("https"),
    ) as response:
        if response.status_code >= 400:
            text = response.text.strip()
            preview = text[:300] if text else f"HTTP {response.status_code}"
            raise RuntimeError(f"Media download failed from upstream ({response.status_code}): {preview}")
        content_type = (response.headers.get("content-type") or "").lower()
        if "application/json" in content_type or "text/plain" in content_type:
            text = response.text.strip()
            if text and len(text) < 500 and "upstream error" in text.lower():
                raise RuntimeError(f"Media download failed from upstream: {text}")
        with destination.open("wb") as file_handle:
            for chunk in response.iter_bytes():
                file_handle.write(chunk)
    return destination


def apify_kv_record_url(store_id: str, record_key: str) -> str:
    return f"https://api.apify.com/v2/key-value-stores/{store_id}/records/{record_key}?disableRedirect=true&token={APIFY_TOKEN}"


def select_apify_kv_media_url(store_id: str) -> Optional[str]:
    response = httpx.get(
        f"https://api.apify.com/v2/key-value-stores/{store_id}/keys",
        params={"token": APIFY_TOKEN},
        timeout=30.0,
        proxy=get_proxy_url("https"),
    )
    response.raise_for_status()
    payload = parse_json_response_or_raise(response, "Apify key-value store listing")
    items = payload.get("items") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return None

    def is_media_item(item: dict[str, object]) -> bool:
        key = str(item.get("key") or "").lower()
        content_type = str(item.get("contentType") or "").lower()
        return (
            content_type.startswith("audio/")
            or content_type.startswith("video/")
            or key.endswith((".mp3", ".mp4", ".m4a", ".wav", ".webm", ".aac", ".ogg", ".opus"))
        )

    for item in items:
        if isinstance(item, dict) and is_media_item(item):
            key = item.get("key")
            if isinstance(key, str) and key:
                return apify_kv_record_url(store_id, key)
    return None


def download_audio_via_apify(url: str, output_dir: str) -> str:
    if not APIFY_TOKEN:
        raise RuntimeError("Apify token is not configured")

    apify_client_class = get_apify_client_class()
    client = apify_client_class(APIFY_TOKEN)
    run_input = {
        "videos": [{"url": url}],
        "storeInKVStore": False,
        "preferredQuality": "144p",
        "preferredFormat": "mp3",
    }
    run = client.actor("streamers/youtube-video-downloader").call(run_input=run_input)
    dataset_id = getattr(run, "default_dataset_id", None) or run.get("defaultDatasetId") or run.get("default_dataset_id")
    key_value_store_id = (
        getattr(run, "default_key_value_store_id", None)
        or run.get("defaultKeyValueStoreId")
        or run.get("default_key_value_store_id")
    )
    if not dataset_id:
        raise RuntimeError("Apify actor finished without a dataset id.")

    items = list(client.dataset(dataset_id).iterate_items())
    if not items:
        raise RuntimeError("Apify actor did not return any downloadable items.")

    first_item = items[0]
    if not isinstance(first_item, dict):
        raise RuntimeError("Apify actor returned an invalid item payload.")

    status = str(first_item.get("status") or "").lower()
    if "fail" in status or "error" in status:
        raise RuntimeError(str(first_item.get("error") or first_item.get("message") or "Apify download failed."))

    download_url = select_apify_download_url(first_item)
    if not download_url:
        download_url = select_any_non_youtube_url(first_item)
    if not download_url and key_value_store_id:
        download_url = select_apify_kv_media_url(str(key_value_store_id))
    if not download_url:
        raise RuntimeError(
            "Apify actor completed, but no downloadable file URL was returned from the dataset or key-value store."
        )

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    suffix = infer_download_suffix(download_url, fallback=".mp3")
    downloaded_path = output_dir_path / f"apify-download{suffix}"
    try:
        download_file_to_path(download_url, downloaded_path)
    except Exception as exc:
        raise RuntimeError(f"Apify returned a download URL, but the media fetch failed: {exc}") from exc

    if suffix in {".mp3", ".m4a", ".wav", ".aac", ".ogg", ".opus", ".webm"}:
        return str(downloaded_path)

    extracted_audio_path = output_dir_path / "apify-extracted-audio.wav"
    return extract_audio_from_video(str(downloaded_path), str(extracted_audio_path))


def normalize_yt_info_entry(info: dict[str, object]) -> dict[str, object]:
    entries = info.get("entries")
    if isinstance(entries, list):
        for entry in entries:
            if isinstance(entry, dict):
                return entry
    return info


def select_best_audio_format_id(info: dict[str, object]) -> Optional[str]:
    normalized_info = normalize_yt_info_entry(info)
    formats = normalized_info.get("formats")
    if not isinstance(formats, list):
        return None

    audio_only: list[dict[str, object]] = []
    muxed_with_audio: list[dict[str, object]] = []
    for fmt in formats:
        if not isinstance(fmt, dict):
            continue
        acodec = str(fmt.get("acodec") or "none")
        vcodec = str(fmt.get("vcodec") or "none")
        format_id = fmt.get("format_id")
        if not format_id or acodec == "none":
            continue
        if vcodec == "none":
            audio_only.append(fmt)
        else:
            muxed_with_audio.append(fmt)

    def format_rank(fmt: dict[str, object]) -> tuple[float, float, int]:
        abr = float(fmt.get("abr") or 0)
        tbr = float(fmt.get("tbr") or 0)
        ext = str(fmt.get("ext") or "")
        ext_preference = 1 if ext in {"m4a", "mp4", "webm"} else 0
        return (abr, tbr, ext_preference)

    if audio_only:
        best = max(audio_only, key=format_rank)
        return str(best.get("format_id"))
    if muxed_with_audio:
        best = max(muxed_with_audio, key=format_rank)
        return str(best.get("format_id"))
    return None


def download_audio(url: str, output_dir: str) -> str:
    if APIFY_TOKEN and is_youtube_url(url):
        try:
            return download_audio_via_apify(url, output_dir)
        except Exception as exc:
            raise RuntimeError(f"Apify YouTube download failed: {exc}") from exc

    yt_dlp = get_yt_dlp_module()
    ensure_ffmpeg()

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    cookies_file = maybe_write_youtube_cookies_file(output_dir_path)
    output_template = str(output_dir_path / "downloaded.%(ext)s")
    ydl_base_opts = {
        "outtmpl": output_template,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "socket_timeout": 30,
        "retries": 3,
        "fragment_retries": 3,
        "file_access_retries": 3,
        "extractor_retries": 3,
        "geo_bypass": True,
        "logger": QuietYtdlpLogger(),
        "http_headers": {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/127.0.0.0 Safari/537.36"
            )
        },
        "extractor_args": {
            "youtube": {
                "player_client": ["web_embedded", "web_creator", "web"]
            }
        },
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
    }
    if cookies_file:
        ydl_base_opts["cookiefile"] = str(cookies_file)
    proxy_url = get_proxy_url("https")
    if proxy_url:
        ydl_base_opts["proxy"] = proxy_url
    try:
        with yt_dlp.YoutubeDL({**ydl_base_opts, "skip_download": True}) as ydl:
            info = ydl.extract_info(url, download=False)
        selected_format_id = select_best_audio_format_id(info)
        ydl_opts = dict(ydl_base_opts)
        ydl_opts["format"] = selected_format_id or "bestaudio/best"

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except yt_dlp.utils.DownloadError as exc:
        error_text = str(exc)
        if "Video unavailable" in error_text:
            raise RuntimeError("This YouTube video is unavailable or private.") from exc
        if "Sign in to confirm your age" in error_text:
            raise RuntimeError("This YouTube video is age-restricted and cannot be downloaded without cookies.") from exc
        if "Please sign in" in error_text or "cookies" in error_text.lower():
            raise RuntimeError(
                "This YouTube video requires authenticated cookies. Add a Railway env var named "
                "YOUTUBE_COOKIES_B64 containing base64-encoded Netscape YouTube cookies, then redeploy."
            ) from exc
        if "Requested format is not available" in error_text:
            raise RuntimeError(
                "The video is available, but YouTube did not expose a compatible downloadable audio format to the server. "
                "Please retry once, try another video, or upload the media file directly."
            ) from exc
        if "HTTP Error 429" in error_text:
            raise RuntimeError("YouTube is rate-limiting the server right now. Please retry in a moment or upload the file directly.") from exc
        if "HTTP Error 403" in error_text or "PO Token" in error_text or "challenge" in error_text:
            raise RuntimeError(
                "YouTube blocked this server request. The app has been updated with stronger download support; please retry once. "
                "If the video still fails, upload the media file directly."
            ) from exc
        raise RuntimeError(
            f"Could not download audio from that YouTube URL. Details: {error_text}"
        ) from exc

    wav_files = sorted(output_dir_path.glob("downloaded*.wav"))
    if not wav_files:
        raise RuntimeError("Audio download completed, but no WAV file was produced.")
    return str(wav_files[0])


def extract_audio_from_video(video_path: str, output_audio_path: str) -> str:
    video_file_clip = get_video_file_clip_class()
    ensure_ffmpeg()

    with video_file_clip(video_path) as clip:
        if clip.audio is None:
            raise RuntimeError("The uploaded video does not contain an audio track.")
        clip.audio.write_audiofile(output_audio_path, logger=None)
    return output_audio_path


def transcribe_audio(audio_path: str) -> str:
    model = get_whisper_model()
    result = model.transcribe(audio_path, fp16=False, verbose=False)
    return result.get("text", "")


def chunk_text(text: str, max_chars: int = 600) -> list[str]:
    try:
        sentences = sent_tokenize(text)
    except LookupError:
        # Fall back when punkt data is unavailable in serverless environments.
        sentences = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", text) if segment.strip()]
    chunks, current = [], ""
    for sentence in sentences:
        if len(current) + len(sentence) <= max_chars:
            current += " " + sentence
        else:
            if current.strip():
                chunks.append(current.strip())
            current = sentence
    if current.strip():
        chunks.append(current.strip())
    return chunks or [text[:max_chars]]


def get_embeddings(chunks: list[str]) -> np.ndarray:
    embedder = get_embedder()
    return embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=False)


def build_faiss_index(embeddings: np.ndarray):
    faiss = get_faiss_module()
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype("float32"))
    return index


def retrieve_context(query: str, chunks: list[str], index, k: int = 3) -> str:
    embedder = get_embedder()
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    _, indices = index.search(query_embedding.astype("float32"), k)
    return "\n".join([chunks[i] for i in indices[0]])


def split_sentences(text: str) -> list[str]:
    normalized = " ".join((text or "").split())
    if not normalized:
        return []
    try:
        return [sentence.strip() for sentence in sent_tokenize(normalized) if sentence.strip()]
    except LookupError:
        return [sentence.strip() for sentence in re.split(r"(?<=[.!?])\s+", normalized) if sentence.strip()]


def extract_top_keywords(text: str, limit: int = 5) -> list[str]:
    tokens = re.findall(r"[A-Za-z][A-Za-z0-9'-]{3,}", text or "")
    counts = Counter(token.lower() for token in tokens if token.lower() not in COMMON_TOPIC_STOPWORDS)
    keywords: list[str] = []
    for word, _ in counts.most_common(limit * 3):
        cleaned = word.strip("-'")
        if not cleaned or cleaned in keywords:
            continue
        keywords.append(cleaned.title())
        if len(keywords) >= limit:
            break
    return keywords


def build_local_headline(query: str, transcript: str) -> str:
    points = build_local_summary_points(transcript, max_points=3)
    if len(points) > 1 and len(points[0].split()) < 10:
        points = points[1:]
    keywords = extract_top_keywords(" ".join(points), limit=3)
    if keywords:
        return f"{' | '.join(keywords)}: {query}".replace("Give ", "").strip()
    return f"Key Takeaways: {query}".strip()


def build_local_summary_points(transcript: str, max_points: int = 4) -> list[str]:
    sentences = split_sentences(transcript)
    if not sentences:
        return ["No transcript could be extracted from the provided source."]

    selected: list[str] = []
    for sentence in sentences:
        compact = " ".join(sentence.split())
        if len(compact) < 40:
            continue
        if compact in selected:
            continue
        selected.append(compact)
        if len(selected) >= max_points:
            break

    if not selected:
        selected = sentences[:max_points]
    return selected[:max_points]


def build_local_topic_details(summary_text: str) -> list[dict[str, str]]:
    points = [line[2:].strip() for line in summary_text.splitlines() if line.startswith("- ")]
    if len(points) > 1 and len(points[0].split()) < 10:
        points = points[1:]
    topic_source = " ".join(points) if points else summary_text
    keywords = extract_top_keywords(topic_source, limit=8)
    details: list[dict[str, str]] = []
    support_pool = points or split_sentences(summary_text)
    for index, keyword in enumerate(keywords[:8]):
        support_text = support_pool[min(index, len(support_pool) - 1)] if support_pool else summary_text
        details.append(
            {
                "title": keyword,
                "explanation": sanitize_inline_html(support_text[:220].strip()),
                "importance": sanitize_inline_html(f"This angle matters because it highlights how {keyword.lower()} shapes the broader message of the source material."),
            }
        )
    return details or [{
        "title": "Main Topic",
        "explanation": sanitize_inline_html("The source centers on one main issue that can be expanded into a professional article."),
        "importance": sanitize_inline_html("This topic is important because it captures the strongest idea presented in the source."),
    }]


def fallback_summary(transcript: str, query: str) -> dict[str, str]:
    points = build_local_summary_points(transcript)
    return {
        "headline": build_local_headline(query, transcript),
        "summary": "\n".join(f"- {point}" for point in points),
    }


def fallback_topics(summary_text: str) -> list[str]:
    points = [line[2:].strip() for line in summary_text.splitlines() if line.startswith("- ")]
    if len(points) > 1 and len(points[0].split()) < 10:
        points = points[1:]
    topic_source = " ".join(points) if points else summary_text
    return extract_top_keywords(topic_source, limit=5) or ["Main Topic"]


def get_topic_details_from_summary(summary_text: str) -> list[dict[str, str]]:
    prompt = f"""
You are analyzing a video summary for an article generation tool.

Create 5 to 8 article topic options based strictly on the summary below.
Each topic must include:
- title
- explanation
- importance

Return valid JSON only in this shape:
[
  {{"title": "Topic title", "explanation": "Short explanation", "importance": "Why it matters"}},
  {{"title": "Topic title", "explanation": "Short explanation", "importance": "Why it matters"}}
]

Summary:
{summary_text}
"""
    try:
        raw_text = gemini_generate_text(prompt)
        payload = json.loads(raw_text)
        if not isinstance(payload, list):
            raise ValueError("Topic details payload must be a list.")
        details: list[dict[str, str]] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title", "")).strip()
            explanation = str(item.get("explanation", "")).strip()
            importance = str(item.get("importance", "")).strip()
            if not title:
                continue
            details.append(
                {
                    "title": strip_html_tags(convert_asterisk_bold_to_html(title)),
                    "explanation": sanitize_inline_html(explanation),
                    "importance": sanitize_inline_html(importance),
                }
            )
        return details[:8] or build_local_topic_details(summary_text)
    except Exception:
        return build_local_topic_details(summary_text)


def fallback_article(headline_text: str, summary_text: str, topic: Optional[str] = None) -> str:
    topic_title = topic or "Main Topic"
    summary_points = [line[2:].strip() for line in summary_text.splitlines() if line.startswith("- ")]
    body_points = summary_points or ["No summary details were available from the source."]
    lead = body_points[0]
    support = body_points[1] if len(body_points) > 1 else lead
    detail = body_points[2] if len(body_points) > 2 else support
    closing_support = body_points[3] if len(body_points) > 3 else detail
    return sanitize_article_html(
        f"""
<h2>{headline_text}</h2>
<p>{lead} That opening point sets the tone for a larger discussion around <strong>{topic_title}</strong>, where speed, scale, and long-term ambition appear to converge in a way that reshapes how the wider story should be understood. Rather than standing alone as an isolated observation, it frames the subject as part of a broader shift with political, economic, or social weight.</p>
<p>{support} Read together, these details suggest that the subject is not simply about one event or one talking point. It is about a pattern that has been building over time, gathering momentum through visible examples and practical outcomes. That gives the article room to move beyond surface description and into a more serious explanation of why the subject matters.</p>
<h3>The Force Behind the Change</h3>
<p>{detail} The most compelling aspect of <strong>{topic_title}</strong> is the way it connects visible results with a deeper system of planning, coordination, and intent. What stands out is not only what happened, but how quickly or decisively it happened, and what that says about the institutions, priorities, or pressures operating beneath the surface.</p>
<p>That matters because readers rarely respond to raw facts alone. They respond to meaning. When a source presents a sequence of developments like this, the real task is to show how each example strengthens the central argument. In this case, the pattern points toward a subject that is larger than a single headline and more durable than a passing moment.</p>
<h3>What the Examples Reveal</h3>
<p>{lead} {support} These examples give the article substance. They show how the selected topic plays out in concrete terms, turning abstract discussion into something readers can picture and assess. That shift from claim to example is what gives the writing authority.</p>
<p>{detail} {closing_support} Taken together, these details suggest a story shaped by visible transformation, discipline, and a willingness to think in terms of outcomes rather than slogans. The selected topic becomes more persuasive because it is supported by examples that feel grounded rather than decorative.</p>
<h3>Why the Topic Carries Weight</h3>
<p>The importance of <strong>{topic_title}</strong> lies in how it opens a wider conversation. It helps explain not just what changed, but why the change deserves attention now. Topics like this often matter because they reveal how power works, how priorities are set, or how societies attempt to move from ambition to execution.</p>
<p>That is what gives the piece lasting value. A strong article does more than repeat highlights. It gives readers a coherent way to understand them. Here, the core material points toward a subject that rewards close attention because it sits at the intersection of visible change and the deeper structures that made that change possible.</p>
<h3>The Broader Meaning</h3>
<p>{closing_support} The final impression is not of a narrow issue, but of a subject with consequences that extend outward into public life, economic direction, and the language of progress itself. That is why <strong>{topic_title}</strong> works as a serious article subject: it contains both immediate interest and deeper analytical value.</p>
<p>The strongest closing point is that the selected topic does not need artificial drama to feel significant. Its significance comes from the weight of the ideas attached to it and from the examples that give those ideas shape. A well-written article can build on that foundation naturally, leaving the reader with a sharper understanding of the issue and a clearer sense of why it deserves continued attention.</p>
"""
    )


def build_fast_context_from_transcript(transcript: str, chunks: list[str]) -> str:
    if transcript.strip():
        normalized = " ".join(transcript.split())
        return normalized[:FAST_ANALYSIS_TRANSCRIPT_LIMIT]
    return " ".join(chunks[:FAST_ANALYSIS_CHUNK_LIMIT])


def build_article_image_url(topic: Optional[str]) -> str:
    topic_text = (topic or "news article").lower()
    image_map = [
        ({"technology", "ai", "software", "digital", "cyber"}, "https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&w=1200&q=80"),
        ({"finance", "business", "market", "economy", "trade"}, "https://images.unsplash.com/photo-1520607162513-77705c0f0d4a?auto=format&fit=crop&w=1200&q=80"),
        ({"media", "video", "film", "broadcast", "news"}, "https://images.unsplash.com/photo-1495020689067-958852a7765e?auto=format&fit=crop&w=1200&q=80"),
        ({"health", "medical", "hospital", "wellness"}, "https://images.unsplash.com/photo-1505751172876-fa1923c5c528?auto=format&fit=crop&w=1200&q=80"),
        ({"sports", "football", "cricket", "game", "team"}, "https://images.unsplash.com/photo-1517649763962-0c623066013b?auto=format&fit=crop&w=1200&q=80"),
        ({"travel", "tourism", "flight", "hotel"}, "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?auto=format&fit=crop&w=1200&q=80"),
    ]
    for keywords, url in image_map:
        if any(keyword in topic_text for keyword in keywords):
            return url
    return "https://images.unsplash.com/photo-1504711434969-e33886168f5c?auto=format&fit=crop&w=1200&q=80"


def gemini_rag(context: str, query: str) -> dict[str, str]:
    prompt = f"""
You are an expert video analyst and content strategist.

Study the source context carefully and extract only the article-worthy ideas that are actually present in it.

Context from media:
{context}

User Query:
{query}

Instructions:
- Create a strong, specific title for the source.
- Then write a concise and meaningful summary in 4 bullet points.
- Keep the wording natural, professional, and grounded in the source.
- Do not invent facts that are not supported by the source.
- If the source compares countries, systems, history, politics, economics, or culture, preserve that framing accurately.

Format strictly as:
Headline: <headline>
Summary:
- point 1
- point 2
- point 3
- point 4
"""
    response_text = gemini_generate_text(prompt)

    headline_text = ""
    summary_lines = []
    current_section = None
    for line in response_text.splitlines():
        if line.startswith("Headline:"):
            headline_text = line.replace("Headline:", "").strip()
            current_section = "headline"
        elif line.startswith("Summary:"):
            current_section = "summary"
        elif current_section == "summary" and line.strip():
            summary_lines.append(sanitize_inline_html(line.strip()))

    return {
        "headline": headline_text or "Media summary",
        "summary": "\n".join(summary_lines).strip() or "No summary generated.",
    }


def generate_news_article(headline_text: str, summary_text: str, topic: Optional[str] = None) -> str:
    if not headline_text.strip():
        headline_text = "Media summary"

    topic_instruction = f"Focus specifically on the topic '{topic}'." if topic else "Use the overall summary as the focus."
    prompt = f"""
You are an experienced editorial writer, video analyst, and professional article writer.

Headline:
{headline_text}

Summary:
{summary_text}

Instructions:
- Write a complete article only on the selected topic.
- Base the article strictly on the source material summarized above.
- Do not invent facts, quotes, names, or unsupported claims.
- Write in a natural, human, professional editorial style.
- Avoid robotic filler phrases and generic transitions.
- Do not write a topic brief, content plan, analysis note, or summary card.
- Do not use headings such as "Why This Topic Stands Out", "Key Developments", "What It Suggests", or similar template labels.
- Do not say "this article discusses", "the source material says", "the selected topic is", or any other meta-writing phrase.
- Write at least 900 words unless brevity is absolutely necessary for factual accuracy.
- Include:
  - one strong <h2> headline
  - a strong opening paragraph
  - 3 to 5 meaningful <h3> section headings
  - detailed body paragraphs in <p> tags
  - a thoughtful closing paragraph
- Use <strong> only where emphasis is genuinely useful.
- Return clean HTML only using: <h2>, <h3>, <p>, <ul>, <li>, <strong>.
- Do not return markdown.
- {topic_instruction}
"""
    try:
        article_text = gemini_generate_text(prompt)
        return sanitize_article_html(article_text) or fallback_article(headline_text, summary_text, topic)
    except Exception:
        return fallback_article(headline_text, summary_text, topic)


def get_topics_from_summary(summary_text: str) -> list[str]:
    return [item["title"] for item in get_topic_details_from_summary(summary_text)]


def enrich_analysis(result: dict[str, str], generate_article: bool = False, selected_topics: Optional[list[str]] = None, article_count: int = 1) -> dict[str, object]:
    headline = result.get("headline", "Media summary")
    summary = result.get("summary", "")
    topic_details = get_topic_details_from_summary(summary)
    topics = [item["title"] for item in topic_details]
    payload: dict[str, object] = {
        "headline": headline,
        "summary": summary,
        "topics": topics,
        "topic_details": topic_details,
        "articles": [],
    }

    if not generate_article:
        return payload

    topics_to_use = selected_topics or topics[:3] or ["Main topic"]
    articles: list[dict[str, str]] = []
    for topic in topics_to_use:
        for _ in range(max(article_count, 1)):
            articles.append({
                "topic": topic,
                "content": generate_news_article(headline, summary, topic),
                "image_url": build_article_image_url(topic),
            })
    payload["articles"] = articles
    return payload


def build_analysis_context(transcript: str, chunks: list[str], query: str) -> str:
    if FAST_ANALYSIS_MODE:
        return build_fast_context_from_transcript(transcript, chunks)

    embeddings = get_embeddings(chunks)
    index = build_faiss_index(embeddings)
    return retrieve_context(query, chunks, index)


def analyze_media(video_path: Optional[str] = None, local_audio_path: Optional[str] = None, query: str = "Summarize the content") -> dict[str, str]:
    audio_source_path = None
    if local_audio_path:
        audio_source_path = local_audio_path
    elif video_path:
        with tempfile.TemporaryDirectory(prefix="media-analyzer-extract-") as temp_dir:
            extracted_audio_path = str(Path(temp_dir) / "extracted_audio.wav")
            audio_source_path = extract_audio_from_video(video_path, extracted_audio_path)
            transcript = transcribe_audio(audio_source_path)
            if not transcript.strip():
                return fallback_summary("", query)
            chunks = chunk_text(transcript)
            try:
                context = build_analysis_context(transcript, chunks, query)
            except Exception:
                context = " ".join(chunks[:3])
            try:
                return gemini_rag(context, query)
            except Exception:
                return fallback_summary(transcript, query)

    if not audio_source_path:
        raise RuntimeError("No audio source provided or audio extraction failed")

    transcript = transcribe_audio(audio_source_path)
    if not transcript.strip():
        return fallback_summary("", query)

    chunks = chunk_text(transcript)
    try:
        context = build_analysis_context(transcript, chunks, query)
    except Exception:
        context = " ".join(chunks[:3])

    try:
        return gemini_rag(context, query)
    except Exception:
        return fallback_summary(transcript, query)


def analyze_text_content(transcript: str, query: str = "Summarize the content") -> dict[str, str]:
    if not transcript.strip():
        return fallback_summary("", query)

    chunks = chunk_text(transcript)
    try:
        context = build_analysis_context(transcript, chunks, query)
    except Exception:
        context = " ".join(chunks[:3])

    try:
        return gemini_rag(context, query)
    except Exception:
        return fallback_summary(transcript, query)


def analyze_url_source(url: str, query: str = "Summarize the content") -> dict[str, object]:
    transcript_override: Optional[str] = None
    temp_audio_path: Optional[str] = None

    with tempfile.TemporaryDirectory(prefix="media-analyzer-url-") as temp_dir:
        if is_youtube_url(url):
            transcript_error: Optional[Exception] = None
            try:
                transcript_override = fetch_youtube_transcript_text(url)
            except Exception as exc:
                transcript_error = exc
                try:
                    transcript_override = fetch_youtube_subtitles_text(url, temp_dir)
                except Exception as subtitle_exc:
                    try:
                        temp_audio_path = download_audio(url, temp_dir)
                    except Exception as download_exc:
                        raise RuntimeError(
                            f"YouTube transcript fallback failed: {transcript_error}. "
                            f"YouTube subtitle fallback failed: {subtitle_exc}. "
                            f"Media download fallback failed: {download_exc}"
                        ) from download_exc
                if transcript_override is None and temp_audio_path is None:
                    raise RuntimeError(
                        f"YouTube transcript fallback failed: {transcript_error}. "
                        f"YouTube subtitle fallback failed: {subtitle_exc}"
                    ) from subtitle_exc
        else:
            temp_audio_path = download_audio(url, temp_dir)

        if transcript_override is not None:
            result = analyze_text_content(
                transcript_override,
                query=query or "Summarize the content",
            )
        else:
            result = analyze_media(
                local_audio_path=temp_audio_path,
                query=query or "Summarize the content",
            )

    return enrich_analysis(result, generate_article=False)


def serialize_job_status(job: Job) -> dict[str, object]:
    status = job.get_status(refresh=True)
    if status == "finished":
        return {
            "success": True,
            "status": "completed",
            "result": job.result,
        }

    if status == "failed":
        last_line = ""
        if job.exc_info:
            last_line = job.exc_info.strip().splitlines()[-1]
        return {
            "success": False,
            "status": "failed",
            "error": last_line or "Background job failed.",
        }

    return {
        "success": True,
        "status": status,
        "queued": True,
    }


async def save_uploaded_file(upload: UploadFile, destination: Path) -> None:
    content = await upload.read()
    destination.write_bytes(content)


@app.get("/")
def index():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h1>ZENPEN is running</h1><p>Frontend assets are not available in this deployment bundle.</p>")


@app.get("/api/health")
def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "dependencies": dependency_status()})


@app.get("/api/config")
def config() -> JSONResponse:
    return JSONResponse(get_api_config())


@app.post("/api/auth/signup")
def signup(payload: SignupRequest) -> JSONResponse:
    if not supabase_is_configured():
        raise HTTPException(status_code=503, detail="Supabase authentication is not configured.")
    if len(payload.password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters long.")

    create_supabase_user(payload.name.strip(), payload.email.strip().lower(), payload.password)
    session = sign_in_supabase_user(payload.email.strip().lower(), payload.password)
    return JSONResponse({"success": True, "session": session})


@app.post("/api/auth/login")
def login(payload: LoginRequest) -> JSONResponse:
    if not supabase_is_configured():
        raise HTTPException(status_code=503, detail="Supabase authentication is not configured.")

    session = sign_in_supabase_user(payload.email.strip().lower(), payload.password)
    return JSONResponse({"success": True, "session": session})


@app.get("/api/history")
def history(request: Request) -> JSONResponse:
    user = resolve_supabase_user(request)
    if user:
        try:
            return JSONResponse(load_history_from_supabase(str(user["id"])))
        except Exception:
            pass
    return JSONResponse(load_history())


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> JSONResponse:
    job = fetch_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Analysis job not found.")
    payload = serialize_job_status(job)
    status_code = 200 if payload.get("success", True) else 500
    return JSONResponse(payload, status_code=status_code)


@app.post("/api/articles")
def generate_articles_endpoint(payload: ArticlesRequest) -> JSONResponse:
    selected_topics = [topic.strip() for topic in payload.selected_topics if topic.strip()]
    if not selected_topics:
        raise HTTPException(status_code=400, detail="Select at least one topic before generating articles.")

    base_result = {
        "headline": payload.headline.strip() or "Media summary",
        "summary": payload.summary.strip(),
    }
    enriched_result = enrich_analysis(
        base_result,
        generate_article=True,
        selected_topics=selected_topics,
        article_count=payload.article_count,
    )
    if payload.topics:
        enriched_result["topics"] = payload.topics
    return JSONResponse({"success": True, "result": enriched_result})


@app.post("/api/analyze")
async def analyze_endpoint(
    request: Request,
    url: Optional[str] = Form(None),
    query: Optional[str] = Form("Summarize the content"),
    file: Optional[UploadFile] = File(None),
    generate_article: bool = Form(False),
    article_count: int = Form(1),
    selected_topics: Optional[str] = Form(None),
) -> JSONResponse:
    if not file and not url:
        raise HTTPException(status_code=400, detail="Please provide a URL or upload an audio/video file")

    try:
        user = resolve_supabase_user(request)
        with tempfile.TemporaryDirectory(prefix="media-analyzer-") as temp_dir:
            temp_dir_path = Path(temp_dir)
            temp_audio_path: Optional[str] = None
            topic_list = [topic.strip() for topic in (selected_topics or "").split(",") if topic.strip()]

            if file:
                suffix = Path(file.filename or "upload.wav").suffix.lower() or ".wav"
                upload_path = temp_dir_path / f"upload{suffix}"
                await save_uploaded_file(file, upload_path)
                if suffix in {".mp4", ".mov", ".mkv", ".avi", ".webm"}:
                    temp_audio_path = extract_audio_from_video(
                        str(upload_path),
                        str(temp_dir_path / "extracted_audio.wav"),
                    )
                else:
                    temp_audio_path = str(upload_path)
                result = analyze_media(
                    local_audio_path=temp_audio_path,
                    query=query or "Summarize the content",
                )
                enriched_result = enrich_analysis(
                    result,
                    generate_article=generate_article,
                    selected_topics=topic_list or None,
                    article_count=article_count,
                )
            elif url:
                enriched_result = analyze_url_source(url, query or "Summarize the content")
                if generate_article:
                    existing_topics = list(enriched_result.get("topics") or [])
                    base_result = {
                        "headline": str(enriched_result.get("headline") or "Media summary"),
                        "summary": str(enriched_result.get("summary") or ""),
                    }
                    enriched_result = enrich_analysis(
                        base_result,
                        generate_article=True,
                        selected_topics=topic_list or None,
                        article_count=article_count,
                    )
                    enriched_result["topics"] = existing_topics
            source = "uploaded-file" if file else "youtube-url"
            source_type = "upload" if file else "url"
            result_for_history = {
                "headline": str(enriched_result.get("headline") or "Media summary"),
                "summary": str(enriched_result.get("summary") or ""),
            }
            add_history_entry(result_for_history, source)
            if user:
                try:
                    persist_analysis_to_supabase(
                        user_id=str(user["id"]),
                        source_type=source_type,
                        source_url=url,
                        source_file_name=file.filename if file else None,
                        source_mime_type=file.content_type if file else None,
                        query=query or "Summarize the content",
                        result=enriched_result,
                        selected_topics=topic_list,
                    )
                except Exception:
                    pass
            return JSONResponse({"success": True, "result": enriched_result})
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=500)


@app.get("/{full_path:path}")
def frontend_routes(full_path: str):
    if full_path.startswith(("api/", "static/")):
        raise HTTPException(status_code=404, detail="Not found")
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return HTMLResponse("<h1>ZENPEN is running</h1><p>Frontend assets are not available in this deployment bundle.</p>")
