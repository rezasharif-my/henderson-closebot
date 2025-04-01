import asyncio
from contextlib import asynccontextmanager

import structlog
from langchain_core.runnables.config import var_child_runnable_config
from langgraph.constants import CONF, CONFIG_KEY_STORE
from starlette.applications import Starlette

import langgraph_api.config as config
from langgraph_api.asyncio import SimpleTaskGroup, set_event_loop
from langgraph_api.cron_scheduler import cron_scheduler
from langgraph_api.graph import collect_graphs_from_env, stop_remote_graphs
from langgraph_api.http import start_http_client, stop_http_client
from langgraph_api.metadata import metadata_loop
from langgraph_api.thread_ttl import thread_ttl_sweep_loop
from langgraph_license.validation import get_license_status, plus_features_enabled
from langgraph_storage.database import start_pool, stop_pool
from langgraph_storage.queue import queue
from langgraph_storage.store import Store

logger = structlog.stdlib.get_logger(__name__)


@asynccontextmanager
async def lifespan(
    app: Starlette | None = None,
    with_cron_scheduler: bool = True,
    taskset: set[asyncio.Task] | None = None,
):
    try:
        current_loop = asyncio.get_running_loop()
        set_event_loop(current_loop)
    except RuntimeError:
        await logger.aerror("Failed to set loop")

    if not await get_license_status():
        raise ValueError(
            "License verification failed. Please ensure proper configuration:\n"
            "- For local development, set a valid LANGSMITH_API_KEY for an account with LangGraph Cloud access "
            "in the environment defined in your langgraph.json file.\n"
            "- For production, configure the LANGGRAPH_CLOUD_LICENSE_KEY environment variable "
            "with your LangGraph Cloud license key.\n"
            "Review your configuration settings and try again. If issues persist, "
            "contact support for assistance."
        )
    await start_http_client()
    await start_pool()
    await collect_graphs_from_env(True)
    try:
        async with SimpleTaskGroup(cancel=True, taskset=taskset) as tg:
            tg.create_task(metadata_loop())
            if config.N_JOBS_PER_WORKER > 0:
                tg.create_task(queue())
            if (
                with_cron_scheduler
                and config.FF_CRONS_ENABLED
                and plus_features_enabled()
            ):
                tg.create_task(cron_scheduler())
            store = Store()
            tg.create_task(Store().start_ttl_sweeper())
            tg.create_task(thread_ttl_sweep_loop())
            var_child_runnable_config.set({CONF: {CONFIG_KEY_STORE: store}})

            yield
    finally:
        await stop_remote_graphs()
        await stop_http_client()
        await stop_pool()
