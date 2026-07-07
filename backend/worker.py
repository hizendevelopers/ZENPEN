from __future__ import annotations

from rq import Worker

from backend.queueing import URL_ANALYSIS_QUEUE_NAME, get_redis_connection


def main() -> None:
    connection = get_redis_connection()
    if connection is None:
        raise RuntimeError("REDIS_URL is not configured for the worker.")

    worker = Worker([URL_ANALYSIS_QUEUE_NAME], connection=connection)
    worker.work()


if __name__ == "__main__":
    main()
