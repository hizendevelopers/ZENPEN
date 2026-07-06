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
        "supabase_configured": supabase_is_configured(),
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


def supabase_is_configured() -> bool:
    return bool(SUPABASE_URL and SUPABASE_PUBLISHABLE_KEY and SUPABASE_SECRET_KEY)


def get_api_config() -> dict[str, object]:
    return {
        "supabase": {
            "enabled": bool(SUPABASE_URL and SUPABASE_PUBLISHABLE_KEY),
            "url": SUPABASE_URL or None,
            "publishableKey": SUPABASE_PUBLISHABLE_KEY or None,
        }
    }


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


def is_youtube_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return "youtube.com" in host or "youtu.be" in host


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


def infer_download_suffix(download_url: str, fallback: str = ".mp3") -> str:
    path = urlparse(download_url).path
    suffix = Path(path).suffix.lower()
    return suffix if suffix else fallback


def download_file_to_path(download_url: str, destination: Path) -> Path:
    with httpx.stream("GET", download_url, follow_redirects=True, timeout=120.0) as response:
        response.raise_for_status()
        with destination.open("wb") as file_handle:
            for chunk in response.iter_bytes():
                file_handle.write(chunk)
    return destination


def download_audio_via_apify(url: str, output_dir: str) -> str:
    if not APIFY_TOKEN:
        raise RuntimeError("Apify token is not configured")

    apify_client_class = get_apify_client_class()
    client = apify_client_class(APIFY_TOKEN)
    run_input = {
        "videos": [{"url": url}],
        "storeInKVStore": None,
        "preferredQuality": None,
        "preferredFormat": "mp3",
        "filenameTemplateParts": None,
        "s3AccessKeyId": None,
        "s3SecretAccessKey": None,
        "s3Bucket": None,
        "s3Region": None,
        "azureConnectionString": None,
        "azureContainerName": None,
        "googleCloudServiceKey": None,
        "googleCloudBucketName": None,
        "transcriptionAndSubtitle": None,
    }
    run = client.actor("streamers/youtube-video-downloader").call(run_input=run_input)
    dataset_id = getattr(run, "default_dataset_id", None) or run.get("defaultDatasetId") or run.get("default_dataset_id")
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
        raise RuntimeError("Apify actor completed, but no downloadable file URL was returned.")

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    suffix = infer_download_suffix(download_url, fallback=".mp3")
    downloaded_path = output_dir_path / f"apify-download{suffix}"
    download_file_to_path(download_url, downloaded_path)

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
    apify_error: Optional[Exception] = None
    if APIFY_TOKEN and is_youtube_url(url):
        try:
            return download_audio_via_apify(url, output_dir)
        except Exception as exc:
            apify_error = exc

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
        if apify_error is not None:
            raise RuntimeError(
                f"Could not download audio from that YouTube URL. Apify fallback failed with: {apify_error}. "
                f"yt-dlp details: {error_text}"
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


def fallback_summary(transcript: str, query: str) -> dict[str, str]:
    cleaned = " ".join(transcript.split())
    first_sentence = cleaned[:220].strip() if cleaned else "No transcript available."
    return {
        "headline": f"Key takeaways for: {query}",
        "summary": "\n".join([
            "- Main topic identified from the transcript.",
            f"- The speech content begins with: {first_sentence}",
            "- The system can expand this summary further when Gemini access is available.",
        ]),
    }


def fallback_topics(summary_text: str) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z0-9'-]{3,}", summary_text)
    seen: list[str] = []
    for word in words:
        normalized = word.lower()
        if normalized not in {item.lower() for item in seen}:
            seen.append(word)
        if len(seen) == 5:
            break
    return seen or ["Main topic"]


def fallback_article(headline_text: str, summary_text: str, topic: Optional[str] = None) -> str:
    focus = f" with a focus on {topic}" if topic else ""
    summary_points = [line[2:] for line in summary_text.splitlines() if line.startswith("- ")]
    body = " ".join(summary_points) or summary_text or "No summary details were available."
    return (
        f"{headline_text}{focus}\n\n"
        f"This generated article expands on the analysis in a readable news style. {body} "
        "The story highlights the central development, explains why it matters, and notes that more detail can be produced when Gemini is configured."
    )


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
    client = get_genai_client()
    prompt = f"""
You are an intelligent news and content analysis AI.
Context from media:
{context}
User Query:
{query}
Instructions:
- Provide a catchy headline on one line.
- Then provide a concise summary in bullet points.
- Keep it engaging and readable.
Format strictly as:
Headline: <headline>
Summary:
- point 1
- point 2
- point 3
"""
    time.sleep(3)
    response = client.models.generate_content(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
        contents=prompt,
    )
    response_text = getattr(response, "text", "")

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
            summary_lines.append(line.strip())

    return {
        "headline": headline_text or "Media summary",
        "summary": "\n".join(summary_lines).strip() or "No summary generated.",
    }


def generate_news_article(headline_text: str, summary_text: str, topic: Optional[str] = None) -> str:
    if not headline_text.strip():
        headline_text = "Media summary"

    topic_instruction = f"Focus specifically on the topic '{topic}'." if topic else "Use the overall summary as the focus."
    prompt = f"""
You are a professional journalist writing a clear and engaging news article.

Headline:
{headline_text}

Summary:
{summary_text}

Instructions:
- Write at least 300 words.
- Use a journalistic tone.
- Start directly with the article.
- Include context, implications, and balanced perspective.
- Keep it readable and human.
- {topic_instruction}
"""
    try:
        client = get_genai_client()
        response = client.models.generate_content(
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            contents=prompt,
        )
        article_text = getattr(response, "text", "").strip()
        return article_text or fallback_article(headline_text, summary_text, topic)
    except Exception:
        return fallback_article(headline_text, summary_text, topic)


def get_topics_from_summary(summary_text: str) -> list[str]:
    prompt = f"""Extract 3 to 5 distinct key topics from the summary below.
Return only a comma-separated list with no extra commentary.

Summary:
{summary_text}
"""
    try:
        client = get_genai_client()
        response = client.models.generate_content(
            model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"),
            contents=prompt,
        )
        raw_topics = getattr(response, "text", "")
        topics = [topic.strip(" -\n\r\t") for topic in raw_topics.split(",") if topic.strip()]
        return topics[:5] or fallback_topics(summary_text)
    except Exception:
        return fallback_topics(summary_text)


def enrich_analysis(result: dict[str, str], generate_article: bool = False, selected_topics: Optional[list[str]] = None, article_count: int = 1) -> dict[str, object]:
    headline = result.get("headline", "Media summary")
    summary = result.get("summary", "")
    topics = get_topics_from_summary(summary)
    payload: dict[str, object] = {
        "headline": headline,
        "summary": summary,
        "topics": topics,
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
                embeddings = get_embeddings(chunks)
                index = build_faiss_index(embeddings)
                context = retrieve_context(query, chunks, index)
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
        embeddings = get_embeddings(chunks)
        index = build_faiss_index(embeddings)
        context = retrieve_context(query, chunks, index)
    except Exception:
        context = " ".join(chunks[:3])

    try:
        return gemini_rag(context, query)
    except Exception:
        return fallback_summary(transcript, query)


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
        with tempfile.TemporaryDirectory(prefix="media-analyzer-") as temp_dir:
            temp_dir_path = Path(temp_dir)
            temp_audio_path: Optional[str] = None

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
            elif url:
                temp_audio_path = download_audio(url, temp_dir)

            result = analyze_media(
                local_audio_path=temp_audio_path,
                query=query or "Summarize the content",
            )
            topic_list = [topic.strip() for topic in (selected_topics or "").split(",") if topic.strip()]
            enriched_result = enrich_analysis(
                result,
                generate_article=generate_article,
                selected_topics=topic_list or None,
                article_count=article_count,
            )
            source = "uploaded-file" if file else "youtube-url"
            source_type = "upload" if file else "url"
            add_history_entry(result, source)
            user = resolve_supabase_user(request)
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
