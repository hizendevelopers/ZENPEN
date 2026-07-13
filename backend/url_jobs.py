from __future__ import annotations

from typing import Optional

from rq import Retry
from rq.job import Job
from rq import get_current_job

from backend.queueing import get_url_analysis_queue


def update_job_progress(stage: str, message: str, progress: int) -> None:
    job = get_current_job()
    if job is None:
        return
    job.meta.setdefault("created_at_ts", time.time())
    job.meta["stage"] = stage
    job.meta["message"] = message
    job.meta["progress"] = max(0, min(progress, 100))
    job.meta["updated_at_ts"] = time.time()
    job.save_meta()


def process_url_analysis_job(
    *,
    url: str,
    query: str,
    user_id: Optional[str] = None,
) -> dict[str, object]:
    from backend import app as app_module

    if app_module.is_youtube_url(url):
        update_job_progress("audio_download", "Preparing video for Gemini transcription...", 8)
    else:
        update_job_progress("downloading_source", "Downloading source...", 8)
    enriched_result = app_module.analyze_url_source(
        url,
        query or "Summarize the content",
        progress_callback=update_job_progress,
    )
    update_job_progress("finalizing_output", "Finalizing output...", 98)
    result = {
        "headline": str(enriched_result.get("headline") or "Media summary"),
        "summary": str(enriched_result.get("summary") or ""),
    }
    app_module.add_history_entry(result, "youtube-url" if app_module.is_youtube_url(url) else "remote-url")

    if user_id:
        try:
            app_module.persist_analysis_to_supabase(
                user_id=user_id,
                source_type="url",
                source_url=url,
                source_file_name=None,
                source_mime_type=None,
                query=query or "Summarize the content",
                result=enriched_result,
                selected_topics=[],
            )
        except Exception:
            pass

    return enriched_result


def enqueue_url_analysis(
    *,
    url: str,
    query: str,
    user_id: Optional[str] = None,
):
    queue = get_url_analysis_queue()
    if queue is None:
        raise RuntimeError("Redis queue is not configured.")
    is_youtube = "youtube.com" in url or "youtu.be" in url

    job = queue.enqueue(
        process_url_analysis_job,
        kwargs={
            "url": url,
            "query": query,
            "user_id": user_id,
        },
        retry=Retry(max=3, interval=[10, 30, 60]),
        result_ttl=86400,
        failure_ttl=604800,
        job_timeout=480,
        description=f"Analyze URL: {url}",
    )
    now = time.time()
    job.meta["stage"] = "audio_download" if is_youtube else "downloading_source"
    job.meta["message"] = "Preparing video for Gemini transcription..." if is_youtube else "Preparing source..."
    job.meta["progress"] = 5
    job.meta["created_at_ts"] = now
    job.meta["updated_at_ts"] = now
    job.save_meta()
    return job
import time
