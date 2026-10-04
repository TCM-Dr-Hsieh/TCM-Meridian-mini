"""Test doubles: scripted OpenAI-compatible server, fake ASR, fake microphone."""
from __future__ import annotations

import json
import unicodedata

import httpx
import numpy as np

from mini.config import Settings
from mini.voice.audio import AudioSource, RATE

ROLE_MARKERS = [
    ('corrector', '三段滾動校字員'),
    ('reviewer', '你是一位嚴謹且嚴格的「幻覺與過度聲稱審查員」'),
    ('writer', '你是「病歷書寫助理」'),
    ('advice', '問診建議助理'),
    ('arbitration', '擔任仲裁者'),
    ('cross', '評比對方的版本'),
    ('analysis', '獨立撰寫「分析與處置」'),
]

DEFAULT_WRITER_OPS = {'thinking': 't', 'operations': [
    {'op': 'insert', 'line': 1, 'content': '甲- 現病史：患者頭痛三天[語音#1]'}], 'summary': '寫入現病史'}
LONG_TEXT = '## 一- 西醫診斷：偏頭痛（待確認）\n## 二- 中醫診斷：頭痛'
DEFAULT_ARBITRATION = '===ARBITRATION===\n兩位教授意見大致一致。\n===FINAL_AT===\n## 一- 西醫診斷：偏頭痛（待確認）\n## 二- 中醫診斷：頭痛'
DEFAULT_CROSS = ('## 對方版本的優點\n- 診斷方向合理\n## 對方版本的缺點與風險\n- 無明顯安全疑慮\n'
                 '## 與我方版本的分歧\n- 實質一致\n## 建議採納與不建議採納\n- 建議採納')
DEFAULT_ADVICE = {'western_ddx': '- 偏頭痛', 'tcm_ddx': '- 肝陽上亢', 'next_questions': '- 疼痛位置？'}


def jdump(value) -> str:
    return json.dumps(value, ensure_ascii=False)


class FakeLLM:
    """httpx.MockTransport handler. `script[role]` is a list consumed in order, then `default`.

    An item is a str (reply text), a dict (JSON reply), an Exception/'HTTP500' marker (failure),
    or a callable(messages) returning one of those.
    """

    def __init__(self):
        self.script: dict[str, list] = {}
        self.calls: list[tuple[str, list]] = []
        self.active = 0
        self.max_active = 0
        self.delay = 0.0

    # -- script helpers --------------------------------------------------
    def queue(self, role: str, *items):
        self.script.setdefault(role, []).extend(items)

    def calls_for(self, role: str) -> list[list]:
        return [m for r, m in self.calls if r == role]

    @staticmethod
    def role_of(messages: list[dict]) -> str:
        system = messages[0]['content'] if messages else ''
        for role, marker in ROLE_MARKERS:
            if marker in system:
                return role
        return 'unknown'

    def default(self, role: str, messages: list[dict]):
        if role == 'corrector':
            payload = json.loads(messages[1]['content'])
            return {'segments': [{'index': r['index'], 'text': r['current']} for r in payload['segments']]}
        if role == 'writer':
            return DEFAULT_WRITER_OPS
        if role == 'reviewer':
            return {'pass': True, 'issues': [], 'comment': 'ok'}
        if role == 'advice':
            return DEFAULT_ADVICE
        if role == 'arbitration':
            return DEFAULT_ARBITRATION
        if role == 'cross':
            return DEFAULT_CROSS
        return LONG_TEXT

    # -- transport --------------------------------------------------------
    async def handler(self, request: httpx.Request) -> httpx.Response:
        import asyncio
        body = json.loads(request.content)
        messages = body['messages']
        role = self.role_of(messages)
        self.calls.append((role, messages))
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            if self.delay:
                await asyncio.sleep(self.delay)
            queue = self.script.get(role, [])
            item = queue.pop(0) if queue else self.default(role, messages)
            if callable(item):
                item = item(messages)
        finally:
            self.active -= 1
        if item == 'HTTP500':
            return httpx.Response(500, text='boom')
        if isinstance(item, Exception):
            raise item
        text = item if isinstance(item, str) else jdump(item)
        return httpx.Response(200, json={'choices': [{'message': {'content': text}, 'finish_reason': 'stop'}],
                                         'usage': {'total_tokens': 1}})

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handler)


def fake_items(text: str, duration: float) -> list[dict]:
    """One aligner item per alphanumeric character, evenly spread over `duration`."""
    chars = [c for c in text if unicodedata.normalize('NFKC', c).casefold().isalnum()]
    if not chars:
        return []
    step = duration / len(chars)
    return [{'text': c, 'start': i * step, 'end': (i + 1) * step} for i, c in enumerate(chars)]


class FakeASR:
    """Replaces LocalASR. `replies` maps window start (rounded seconds) -> text or Exception."""

    def __init__(self, replies: dict | None = None, default: str = ''):
        self.replies = replies or {}
        self.default = default
        self.calls: list[float] = []
        self.loaded = False
        self.info = {'type': 'ready', 'device': 'fake'}

    @property
    def ready(self) -> bool:
        return self.loaded

    async def load(self, settings):
        self.loaded = True
        return self.info

    async def recognize(self, settings, samples) -> dict:
        index = len(self.calls)
        self.calls.append(len(samples) / RATE)
        item = self.replies.get(index, self.default)
        if isinstance(item, Exception):
            raise item
        duration = len(samples) / RATE
        return {'text': item, 'items': fake_items(item, duration), 'language': 'Chinese'}

    async def close(self):
        self.loaded = False


def silent_source(settings: Settings, seconds: float, recording_path=None, block: float = 0.1):
    """A real AudioSource fed by a fast fake microphone producing `seconds` of low-level noise."""
    rng = np.random.default_rng(0)

    def make_iterator(stop):
        total = int(seconds * RATE)
        step = int(block * RATE)
        produced = 0
        while produced < total and not stop.is_set():
            yield (rng.standard_normal(step) * 0.01).astype(np.float32)
            produced += step

    return AudioSource(make_iterator, window_seconds=settings.asr.window_seconds,
                       overlap_seconds=settings.asr.overlap_seconds, recording_path=recording_path)


def make_settings(tmp_path, **overrides) -> Settings:
    settings = Settings()
    settings.visits_dir = str(tmp_path / 'visits')
    settings.llm.retries = 3
    for key, value in overrides.items():
        setattr(settings, key, value)
    settings.validate()
    return settings
