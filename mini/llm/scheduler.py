"""Priority scheduler with a global concurrency limit for LLM requests."""
from __future__ import annotations

import asyncio
import heapq
import itertools
from contextlib import asynccontextmanager

PRIORITY_USER = 0        # user-triggered jobs (write / review / advice / analysis)
PRIORITY_BACKGROUND = 1  # continuous transcript correction


class LLMScheduler:
    """At most `limit` requests run at once; waiting requests are served by priority, then FIFO.

    In-flight requests are never preempted.
    """

    def __init__(self, limit: int = 2):
        self.limit = max(1, int(limit))
        self.active = 0
        self._heap: list[tuple[int, int, asyncio.Future]] = []
        self._counter = itertools.count()

    @property
    def queued(self) -> int:
        return sum(1 for _, _, fut in self._heap if not fut.cancelled())

    def set_limit(self, limit: int):
        self.limit = max(1, int(limit))
        self._dispatch()

    def _dispatch(self):
        while self.active < self.limit and self._heap:
            _, _, fut = heapq.heappop(self._heap)
            if fut.cancelled():
                continue
            self.active += 1
            fut.set_result(None)

    def _release(self):
        self.active -= 1
        self._dispatch()

    @asynccontextmanager
    async def slot(self, priority: int = PRIORITY_USER):
        if self.active < self.limit and not self._heap:
            self.active += 1
        else:
            fut = asyncio.get_running_loop().create_future()
            heapq.heappush(self._heap, (priority, next(self._counter), fut))
            self._dispatch()
            try:
                await fut
            except asyncio.CancelledError:
                if fut.done() and not fut.cancelled():
                    self._release()  # the slot was granted just as we were cancelled
                raise
        try:
            yield
        finally:
            self._release()
