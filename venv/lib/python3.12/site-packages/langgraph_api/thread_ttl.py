"""Sweeping logic for cleaning up expired threads and checkpoints."""

import asyncio

import structlog

from langgraph_api.config import THREAD_TTL
from langgraph_storage.database import connect

logger = structlog.stdlib.get_logger(__name__)


async def thread_ttl_sweep_loop():
    """Periodically delete threads based on TTL configuration.

    Currently implements the 'delete' strategy, which deletes entire threads
    that have been inactive for longer than their configured TTL period.
    """
    # Use the same interval as store TTL sweep
    thread_ttl_config = THREAD_TTL or {}
    strategy = thread_ttl_config.get("strategy", "delete")
    if strategy != "delete":
        raise NotImplementedError(
            f"Unrecognized thread deletion strategy: {strategy}." " Expected 'delete'."
        )
    sweep_interval_minutes = thread_ttl_config.get("sweep_interval_minutes", 5)
    await logger.ainfo(
        "Starting thread TTL sweeper",
        interval_minutes=sweep_interval_minutes,
    )

    from langgraph_storage.ops import Threads

    while True:
        await asyncio.sleep(sweep_interval_minutes * 60)
        try:
            async with connect() as conn:
                threads_processed, threads_deleted = await Threads.sweep_ttl(conn)
                if threads_processed > 0:
                    await logger.ainfo(
                        "Thread TTL sweep completed",
                        threads_processed=threads_processed,
                        threads_deleted=threads_deleted,
                    )
        except Exception as exc:
            logger.exception("Thread TTL sweep iteration failed", exc_info=exc)
