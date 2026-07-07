from __future__ import annotations

from typing import Optional

from rq import Retry

from backend.queueing import get_url_analysis_queue


def process_url_analysis_job(
    *,
    url: str,
    query: str,
    user_id: Optional[str] = None,
) -> dict[str, object]:
    from backend import app as app_module

    enriched_result = app_module.analyze_url_source(url, query or "Summarize the content")
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

    return queue.enqueue(
        process_url_analysis_job,
        kwargs={
            "url": url,
            "query": query,
            "user_id": user_id,
        },
        retry=Retry(max=3, interval=[10, 30, 60]),
        result_ttl=86400,
        failure_ttl=604800,
        job_timeout=1800,
        description=f"Analyze URL: {url}",
    )
