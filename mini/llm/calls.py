"""One logical LLM call = scheduling + retries + validation + full audit logging."""
from __future__ import annotations

import asyncio
import itertools
from dataclasses import dataclass, field
from typing import Any, Callable

from ..config import Endpoint
from ..textutil import estimate_messages_tokens
from .client import LLMClient, LLMError
from .scheduler import LLMScheduler, PRIORITY_USER

CONTEXT_HINT = ('若伺服器實際可容納更多，請在「模型設定」調高此接口的「context 上限」；'
                '否則請縮短輸入（例如精簡患者匯入資料），或結束本次看診後再處理。')
_OVERFLOW_WORDS = ('exceed', 'too long', 'too large', 'overflow', 'n_ctx', 'maximum context', 'context length',
                   'context window', 'context size')


def prompt_too_long_message(endpoint: Endpoint, messages: list[dict], reserve: int | None) -> str:
    """Pre-flight check against the interface's context limit; '' when the prompt fits (or no limit is set)."""
    limit = endpoint.context_tokens
    if not limit:
        return ''
    estimate = estimate_messages_tokens(messages)
    reserve = reserve or 0
    if estimate + reserve <= limit:
        return ''
    return (f'提示詞估計約 {estimate} tokens，加上輸出預留 {reserve} tokens，超過此接口設定的 context 上限 {limit}。'
            + CONTEXT_HINT)


def is_context_overflow(text: str) -> bool:
    lowered = text.lower()
    return 'context' in lowered and any(word in lowered for word in _OVERFLOW_WORDS) or 'n_ctx' in lowered


class ValidationError(Exception):
    """Raised by a validator when an LLM output is unusable; triggers a retry with the reason."""


class CallFailed(RuntimeError):
    def __init__(self, agent: str, last_error: str, call_ids: list[str]):
        super().__init__(f'{agent} 呼叫失敗：{last_error}')
        self.agent = agent
        self.last_error = last_error
        self.call_ids = call_ids


@dataclass
class CallResult:
    text: str
    value: Any
    call_ids: list[str] = field(default_factory=list)
    attempts: int = 1


class LLMCaller:
    """Binds a client + scheduler to a log sink. `logger(record)` receives one dict per attempt."""

    def __init__(self, client: LLMClient, scheduler: LLMScheduler, *,
                 timeout: Callable[[], float], retries: Callable[[], int],
                 logger: Callable[[dict], None] | None = None, id_prefix: str = 'c',
                 backoff: float = 1.0):
        self.client = client
        self.scheduler = scheduler
        self._timeout = timeout
        self._retries = retries
        self.logger = logger
        self.id_prefix = id_prefix
        self.backoff = backoff
        self._ids = itertools.count(1)

    def next_id(self) -> str:
        return f'{self.id_prefix}{next(self._ids):04d}'

    async def call(self, *, agent: str, endpoint: Endpoint,
                   messages: list[dict] | Callable[[str | None], list[dict]],
                   job_id: str | None = None, priority: int = PRIORITY_USER,
                   validator: Callable[[str], Any] | None = None,
                   retries: int | None = None, max_tokens: int | None = None) -> CallResult:
        """Run the call, retrying up to `retries` times (default: the global setting).

        `messages` may be a callable receiving the previous attempt's error text so the
        retry can tell the model what was wrong.
        """
        limit = self._retries() if retries is None else retries
        last_error: str | None = None
        call_ids: list[str] = []
        for attempt in range(1, limit + 2):
            built = messages(last_error) if callable(messages) else messages
            call_id = self.next_id()
            call_ids.append(call_id)
            record = {'call_id': call_id, 'job_id': job_id, 'agent': agent, 'attempt': attempt,
                      'model': endpoint.model_name, 'url': endpoint.api_url,
                      'temperature': endpoint.temperature,
                      'max_tokens': endpoint.max_tokens if max_tokens is None else max_tokens,
                      'messages': built, 'response': None, 'finish_reason': None,
                      'latency_ms': None, 'usage': None, 'error': None}
            too_long = prompt_too_long_message(endpoint, built, record['max_tokens'])
            if too_long:                                    # retrying cannot help: fail now, clearly, with a log record
                record['error'] = too_long
                self._emit(record)
                raise CallFailed(agent, too_long, call_ids)
            try:
                async with self.scheduler.slot(priority):
                    result = await self.client.chat(endpoint, built, timeout=self._timeout(),
                                                    max_tokens=max_tokens)
                record.update(response=result.text, finish_reason=result.finish_reason,
                              latency_ms=result.latency_ms, usage=result.usage)
                value = validator(result.text) if validator else None
            except asyncio.CancelledError:
                record['error'] = '已取消'
                self._emit(record)
                raise
            except (LLMError, ValidationError, ValueError) as exc:
                last_error = str(exc) or type(exc).__name__
                record['error'] = last_error
                self._emit(record)
                if isinstance(exc, LLMError) and is_context_overflow(last_error):
                    raise CallFailed(agent, f'伺服器回報輸入超過模型 context：{last_error[:200]}。{CONTEXT_HINT}',
                                     call_ids) from exc
                if attempt <= limit and self.backoff:
                    await asyncio.sleep(self.backoff)
                continue
            self._emit(record)
            return CallResult(result.text, value, call_ids, attempt)
        raise CallFailed(agent, last_error or '未知錯誤', call_ids)

    def _emit(self, record: dict):
        if self.logger is not None:
            try:
                self.logger(record)
            except Exception:  # logging must never break a job
                pass
