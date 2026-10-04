"""Single-slot job manager: only one of record / advice / analysis runs at a time."""
from __future__ import annotations

import asyncio
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Awaitable, Callable

from .llm import CallFailed

KIND_LABELS = {'record': '病歷書寫', 'advice': '問診建議', 'analysis': '整體分析'}


class JobFailed(RuntimeError):
    """A job failed in a reportable way (fail-closed; nothing was written)."""


class BusyError(RuntimeError):
    pass


@dataclass
class Job:
    id: str
    kind: str
    status: str = 'running'          # running | succeeded | failed | cancelled
    stage: str = ''
    message: str = ''                # result or failure text shown in the status bar
    started_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec='seconds'))
    task: asyncio.Task | None = None
    _on_change: Callable[[], None] | None = None

    @property
    def label(self) -> str:
        return KIND_LABELS.get(self.kind, self.kind)

    def set_stage(self, text: str):
        self.stage = text
        if self._on_change:
            self._on_change()


class JobManager:
    def __init__(self, session):
        self.session = session
        self.current: Job | None = None
        self.last: Job | None = None
        self._counter = 0

    @property
    def busy(self) -> bool:
        return self.current is not None and self.current.status == 'running'

    def start(self, kind: str, runner: Callable[[Job], Awaitable[Any]]) -> Job:
        if self.busy:
            raise BusyError('已有作業進行中，請等待完成或先取消。')
        self._counter += 1
        job = Job(f'J{self._counter:03d}', kind, stage='準備中', _on_change=self.session.on_change)
        self.current = job
        job.task = asyncio.create_task(self._run(job, runner))
        job.task.add_done_callback(lambda task, job=job: self._ensure_closed(job))
        self.session.on_change()
        return job

    def _ensure_closed(self, job: Job):
        """A task cancelled before its first step never runs `_run`; close the job here."""
        if job.status != 'running':
            return
        job.status, job.message = 'cancelled', '已取消'
        self.session.log.emit('job_cancelled', job_id=job.id, kind=job.kind, reason='使用者取消（尚未開始執行）')
        if self.current is job:
            self.last, self.current = job, None
        self.session.on_change()

    async def _run(self, job: Job, runner):
        log = self.session.log
        try:
            await runner(job)
            job.status = 'succeeded'
        except asyncio.CancelledError:
            job.status = 'cancelled'
            job.message = '已取消'
            log.emit('job_cancelled', job_id=job.id, kind=job.kind, reason='使用者取消')
        except JobFailed as exc:
            job.status, job.message = 'failed', str(exc)
            log.emit('job_failed', job_id=job.id, kind=job.kind, reason=str(exc))
        except CallFailed as exc:
            job.status, job.message = 'failed', str(exc)
            log.emit('job_failed', job_id=job.id, kind=job.kind, reason=str(exc), call_ids=exc.call_ids)
        except Exception as exc:                      # unexpected: keep the app alive, record everything
            job.status = 'failed'
            job.message = f'內部錯誤：{type(exc).__name__}: {exc}'
            log.emit('job_failed', job_id=job.id, kind=job.kind, reason=job.message,
                     traceback=traceback.format_exc())
        finally:
            self.last, self.current = job, None
            self.session.on_change()

    def cancel(self):
        if self.busy and self.current and self.current.task:
            self.current.task.cancel()

    async def cancel_and_wait(self):
        job = self.current
        if job and job.task and not job.task.done():
            job.task.cancel()
            try:
                await job.task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass


async def gather_all(*coros):
    """Run coroutines concurrently; on the first failure cancel the rest and re-raise it."""
    tasks = [asyncio.ensure_future(c) for c in coros]
    try:
        await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
        for task in tasks:
            if task.done() and not task.cancelled() and task.exception() is not None:
                raise task.exception()
        pending = [t for t in tasks if not t.done()]
        if pending:
            await asyncio.wait(pending)
        return [t.result() for t in tasks]
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
