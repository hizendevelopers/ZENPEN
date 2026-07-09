from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from redis import Redis
from rq import Queue, Worker
from rq.job import Job

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"
load_dotenv(ENV_FILE)

REDIS_URL = os.getenv("REDIS_URL", "").strip()
URL_ANALYSIS_QUEUE_NAME = os.getenv("URL_ANALYSIS_QUEUE_NAME", "url-analysis")


@lru_cache(maxsize=1)
def get_redis_connection() -> Optional[Redis]:
    if not REDIS_URL:
        return None
    return Redis.from_url(REDIS_URL)


def queue_is_configured() -> bool:
    return bool(REDIS_URL)


def queue_is_available() -> bool:
    connection = get_redis_connection()
    if connection is None:
        return False
    try:
        connection.ping()
        return True
    except Exception:
        return False


def workers_available() -> bool:
    connection = get_redis_connection()
    if connection is None:
        return False
    try:
        return len(Worker.all(connection=connection)) > 0
    except Exception:
        return False


def get_url_analysis_queue() -> Optional[Queue]:
    connection = get_redis_connection()
    if connection is None:
        return None
    return Queue(URL_ANALYSIS_QUEUE_NAME, connection=connection, default_timeout=1800)


def fetch_job(job_id: str) -> Optional[Job]:
    connection = get_redis_connection()
    if connection is None:
        return None
    try:
        return Job.fetch(job_id, connection=connection)
    except Exception:
        return None
