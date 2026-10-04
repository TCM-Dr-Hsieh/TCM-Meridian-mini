"""Async OpenAI-compatible chat client (non-streaming)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import httpx

from ..config import Endpoint
from ..textutil import clean_think


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResult:
    text: str
    finish_reason: str = ''
    usage: dict = field(default_factory=dict)
    latency_ms: int = 0


def _headers(endpoint: Endpoint) -> dict:
    return {'Authorization': f'Bearer {endpoint.api_key}'} if endpoint.api_key else {}


def _mask(text: str, endpoint: Endpoint) -> str:
    return text.replace(endpoint.api_key, '***') if endpoint.api_key else text


class LLMClient:
    def __init__(self, transport=None):
        self._client = httpx.AsyncClient(timeout=None, transport=transport, trust_env=False)

    async def aclose(self):
        await self._client.aclose()

    async def chat(self, endpoint: Endpoint, messages: list[dict], *, timeout: float,
                   max_tokens: int | None = None) -> LLMResult:
        body = {'model': endpoint.model_name, 'messages': messages,
                'temperature': endpoint.temperature, 'stream': False}
        limit = endpoint.max_tokens if max_tokens is None else max_tokens
        if limit and limit > 0:
            body['max_tokens'] = limit
        started = time.monotonic()
        try:
            response = await self._client.post(endpoint.api_url + '/chat/completions',
                                               headers=_headers(endpoint), json=body,
                                               timeout=httpx.Timeout(timeout))
        except httpx.TimeoutException as exc:
            raise LLMError(f'LLM 逾時（{timeout:g} 秒）。') from exc
        except httpx.HTTPError as exc:
            raise LLMError(f'LLM 連線失敗：{_mask(str(exc) or type(exc).__name__, endpoint)}') from exc
        if response.is_error:
            detail = _mask(response.text[:500], endpoint)
            hint = ' 請確認模型名稱與伺服器設定。' if response.status_code in (400, 404, 415, 422) else ''
            raise LLMError(f'HTTP {response.status_code}：{detail}{hint}')
        try:
            data = response.json()
            choice = data['choices'][0]
            finish = choice.get('finish_reason') or ''
            content = choice['message']['content']
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise LLMError('模型回應格式錯誤，缺少 choices/message/content。') from exc
        if finish not in ('', 'stop'):
            raise LLMError(f'模型未完整輸出（finish_reason={finish}，可能達到 token 上限）。')
        if not isinstance(content, str):
            raise LLMError('模型沒有回傳文字。')
        text = clean_think(content)
        if not text:
            raise LLMError('模型回傳空白內容。')
        usage = data.get('usage') if isinstance(data.get('usage'), dict) else {}
        return LLMResult(text, finish, usage, int((time.monotonic() - started) * 1000))

    async def models(self, endpoint: Endpoint, timeout: float = 15.0) -> list[str]:
        try:
            response = await self._client.get(endpoint.api_url + '/models', headers=_headers(endpoint),
                                              timeout=httpx.Timeout(timeout))
        except httpx.HTTPError as exc:
            raise LLMError(f'連線失敗：{_mask(str(exc) or type(exc).__name__, endpoint)}') from exc
        if response.is_error:
            raise LLMError(f'HTTP {response.status_code}：{_mask(response.text[:300], endpoint)}')
        try:
            return [item['id'] for item in response.json()['data']]
        except (KeyError, TypeError, ValueError) as exc:
            raise LLMError('模型清單回應格式錯誤。') from exc
