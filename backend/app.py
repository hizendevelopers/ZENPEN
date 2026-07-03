from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    import faiss
except Exception:  # pragma: no cover - optional dependency
    faiss = None

import nltk
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.staticfiles import StaticFiles
from nltk.tokenize import sent_tokenize
from dotenv import load_dotenv

try:
    import yt_dlp
except Exception:  # pragma: no cover - optional dependency
    yt_dlp = None

try:
    from moviepy.editor import VideoFileClip
except Exception:  # pragma: no cover - optional dependency
    try:
        # moviepy 2.x exposes VideoFileClip at the top level.
        from moviepy import VideoFileClip
    except Exception:  # pragma: no cover - optional dependency
        VideoFileClip = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None

try:
    import whisper
except Exception:  # pragma: no cover - optional dependency
    whisper = None

try:
    from google import genai
except Exception:  # pragma: no cover - optional dependency
    genai = None

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
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

WHISPER_MODEL = None
EMBEDDER = None
GENAI_CLIENT = None


class QuietYtdlpLogger:
    def debug(self, msg: str) -> None:
        return None

    def warning(self, msg: str) -> None:
        return None

    def error(self, msg: str) -> None:
        return None


def dependency_status() -> dict[str, bool | str]:
    return {
        "python": sys.executable,
        "yt_dlp": yt_dlp is not None,
        "whisper": whisper is not None,
        "moviepy": VideoFileClip is not None,
        "faiss": faiss is not None,
        "sentence_transformers": SentenceTransformer is not None,
        "gemini": genai is not None,
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "gemini_api_key_configured": bool(GEMINI_API_KEY),
    }


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
    if genai is None:
        raise RuntimeError(build_missing_dependency_message("google-genai"))
    if not GEMINI_API_KEY:
        raise RuntimeError("Gemini is not configured")
    if GENAI_CLIENT is None:
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
    with HISTORY_FILE.open("w", encoding="utf-8") as handle:
        json.dump(items, handle, indent=2)


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
    if whisper is None:
        raise RuntimeError(build_missing_dependency_message("Whisper"))
    if WHISPER_MODEL is None:
        WHISPER_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        WHISPER_MODEL = whisper.load_model("tiny", download_root=str(WHISPER_CACHE_DIR))
    return WHISPER_MODEL


def get_embedder():
    global EMBEDDER
    if SentenceTransformer is None:
        raise RuntimeError(build_missing_dependency_message("sentence-transformers"))
    if EMBEDDER is None:
        EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2")
    return EMBEDDER


def download_audio(url: str, output_dir: str) -> str:
    if yt_dlp is None:
        raise RuntimeError(build_missing_dependency_message("yt-dlp"))
    ensure_ffmpeg()

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    output_template = str(output_dir_path / "downloaded.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_template,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "logger": QuietYtdlpLogger(),
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "wav",
                "preferredquality": "192",
            }
        ],
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except yt_dlp.utils.DownloadError as exc:
        raise RuntimeError(
            "Could not download audio from that YouTube URL. Please try another video or upload the media file directly."
        ) from exc

    wav_files = sorted(output_dir_path.glob("downloaded*.wav"))
    if not wav_files:
        raise RuntimeError("Audio download completed, but no WAV file was produced.")
    return str(wav_files[0])


def extract_audio_from_video(video_path: str, output_audio_path: str) -> str:
    if VideoFileClip is None:
        raise RuntimeError(build_missing_dependency_message("moviepy"))
    ensure_ffmpeg()

    with VideoFileClip(video_path) as clip:
        if clip.audio is None:
            raise RuntimeError("The uploaded video does not contain an audio track.")
        clip.audio.write_audiofile(output_audio_path, logger=None)
    return output_audio_path


def transcribe_audio(audio_path: str) -> str:
    model = get_whisper_model()
    result = model.transcribe(audio_path, fp16=False, verbose=False)
    return result.get("text", "")


def chunk_text(text: str, max_chars: int = 600) -> list[str]:
    sentences = sent_tokenize(text)
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
    if faiss is None:
        raise RuntimeError(build_missing_dependency_message("faiss-cpu"))
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
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/api/health")
def health() -> JSONResponse:
    return JSONResponse({"status": "ok", "dependencies": dependency_status()})


@app.get("/api/history")
def history() -> JSONResponse:
    return JSONResponse(load_history())


@app.post("/api/analyze")
async def analyze_endpoint(
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
            add_history_entry(result, source)
            return JSONResponse({"success": True, "result": enriched_result})
    except Exception as exc:
        return JSONResponse({"success": False, "error": str(exc)}, status_code=500)
