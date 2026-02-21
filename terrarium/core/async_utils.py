from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, Callable, TypeVar

T = TypeVar("T")

_DEFAULT_MAX_WORKERS = int(os.getenv("TERRARIUM_BLOCKING_MAX_WORKERS", "32"))
_BLOCKING_EXECUTOR = ThreadPoolExecutor(max_workers=max(1, _DEFAULT_MAX_WORKERS))


async def run_blocking(func: Callable[..., T], /, *args: Any, **kwargs: Any) -> T:
    """
    Run a blocking function in a shared thread pool without blocking the asyncio loop.
    """
    fut = _BLOCKING_EXECUTOR.submit(partial(func, *args, **kwargs))
    poll_s = float(os.getenv("TERRARIUM_BLOCKING_POLL_S", "0.01"))
    try:
        while not fut.done():
            await asyncio.sleep(poll_s)
    except asyncio.CancelledError:
        fut.cancel()
        raise
    return fut.result()
