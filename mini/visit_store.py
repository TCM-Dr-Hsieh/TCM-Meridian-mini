"""On-disk layout of one visit: folder allocation, atomic writes, the event log, and log.md."""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable

from .fileio import atomic_write_json, atomic_write_text
from .textutil import format_clock


class EventLog:
    """Append-only chronological log. Every event carries wall-clock time and audio time."""

    def __init__(self, path: Path, t_provider: Callable[[], float | None] | None = None):
        self.path = path
        self._t = t_provider or (lambda: None)
        self.events: list[dict] = []
        self._seq = 0
        self._torn = False                          # a failed write may have left half a line in the file
        self._lock = threading.Lock()
        self._handle = open(path, 'a', encoding='utf-8', newline='\n')

    def emit(self, type: str, **fields) -> dict:
        """Record one event. It reaches `events` (and so log.md) only after it is on disk, so the two never
        disagree; when the write fails nothing is recorded, the sequence number is not consumed, and the
        exception propagates (the caller may emit again)."""
        with self._lock:
            seq = self._seq + 1
            t = self._t()
            event = {'seq': seq, 'ts': datetime.now().astimezone().isoformat(timespec='milliseconds'),
                     't': None if t is None else round(t, 2), 'type': type}
            event.update(fields)
            self._write(json.dumps(event, ensure_ascii=False) + '\n')
            self._seq = seq
            self.events.append(event)
            return event

    def _fragment_pending(self) -> bool:
        """True when the file does not end with a newline, i.e. a failed write left half a line behind."""
        try:
            with open(self.path, 'rb') as stream:
                if stream.seek(0, 2) == 0:
                    return False
                stream.seek(-1, 2)
                return stream.read(1) != b'\n'
        except OSError:
            return False

    def _write(self, line: str):
        try:
            if self._handle.closed:                 # e.g. finish() retried after a late failure
                self._handle = open(self.path, 'a', encoding='utf-8', newline='\n')
            # A failure that wrote nothing leaves no mark; only a real torn fragment is moved onto its own line.
            self._handle.write(('\n' + line) if self._torn and self._fragment_pending() else line)
            self._handle.flush()
            self._torn = False
        except BaseException:
            self._torn = True
            try:
                self._handle.close()                # drop buffered bytes; the next emit reopens the file
            except Exception:
                pass
            raise

    def close(self):
        with self._lock:
            if not self._handle.closed:
                self._handle.close()


class VisitStore:
    def __init__(self, folder: Path):
        self.folder = folder
        self.visit_id = folder.name
        self.audio_path = folder / 'audio.wav'
        self.log_path = folder / 'log.jsonl'
        self.transcript_llm_path = folder / 'transcript_llm.jsonl'

    @classmethod
    def allocate(cls, visits_dir: Path, now: datetime | None = None) -> 'VisitStore':
        """Create `日期-NNN` exclusively so concurrent starts can never collide."""
        visits_dir.mkdir(parents=True, exist_ok=True)
        day = (now or datetime.now()).strftime('%Y-%m-%d')
        number = 1
        while True:
            folder = visits_dir / f'{day}-{number:03d}'
            try:
                os.mkdir(folder)
            except FileExistsError:
                number += 1
                continue
            return cls(folder)

    # -- writers -------------------------------------------------------
    def write_json(self, relative: str, data) -> None:
        atomic_write_json(self.folder / relative, data)

    def write_text(self, relative: str, text: str) -> None:
        atomic_write_text(self.folder / relative, text)

    def append_transcript_llm(self, record: dict) -> None:
        with open(self.transcript_llm_path, 'a', encoding='utf-8', newline='\n') as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')

    def write_log_md(self, events: list[dict]) -> None:
        self.write_text('log.md', render_log_md(self.visit_id, events))


# ---------------------------------------------------------------------------
# log.md
# ---------------------------------------------------------------------------
def _clock(event: dict) -> str:
    stamp = event.get('ts', '')[11:23]
    t = event.get('t')
    return f'{stamp} (t+{format_clock(t)})' if t is not None else f'{stamp}'


def _fence(text, lang: str = 'text') -> str:
    return f'```{lang}\n{text}\n```'


def _details(summary: str, body: str) -> str:
    return f'<details><summary>{summary}</summary>\n\n{body}\n\n</details>'


def _llm_body(event: dict) -> str:
    parts = [f'模型 `{event.get("model")}` · temperature {event.get("temperature")} · '
             f'max_tokens {event.get("max_tokens")} · 耗時 {event.get("latency_ms")} ms · '
             f'finish_reason {event.get("finish_reason")}']
    for message in event.get('messages') or []:
        parts.append(f'**{message.get("role")}**\n\n{_fence(message.get("content", ""))}')
    if event.get('response') is not None:
        parts.append(f'**response**\n\n{_fence(event["response"])}')
    if event.get('error'):
        parts.append(f'**error**：{event["error"]}')
    return '\n\n'.join(parts)


def summarize_event(event: dict) -> tuple[str, str | None]:
    """Return (one-line summary, optional long body) for log.md."""
    kind = event.get('type', '')
    if kind == 'llm_call':
        status = '失敗：' + str(event['error']) if event.get('error') else '成功'
        title = (f'LLM {event.get("agent")} · {event.get("call_id")} · 第 {event.get("attempt")} 次 · '
                 f'{event.get("job_id") or "—"} · {status}')
        return title, _llm_body(event)
    if kind == 'transcript_segment':
        return (f'逐字稿 #{event["index"]} [{format_clock(event.get("new_start"))}–{format_clock(event.get("end"))}] '
                f'{str(event.get("corrected", ""))[:60]}',
                f'ASR 原稿：{event.get("raw_asr")}\n\n轉繁：{event.get("raw")}\n\n新增：{event.get("added")}\n\n'
                f'校稿後：{event.get("corrected")}')
    if kind == 'transcript_revised':
        rows = '\n'.join(f'- #{i}：{b} → {a}' for i, b, a in
                         zip(event.get('segments', []), event.get('before', []), event.get('after', [])))
        return f'逐字稿校稿修訂 #{",".join(map(str, event.get("segments", [])))}', rows
    if kind == 'transcript_gap':
        return (f'逐字稿缺漏 #{event.get("index")} {format_clock(event.get("start"))}–{format_clock(event.get("end"))}',
                f'錯誤：{event.get("error")}\n\nASR 原稿：{event.get("raw_asr")}')
    if kind in ('patient_input_set', 'patient_input_updated'):
        return f'患者匯入資料（{len(event.get("text", ""))} 字）', _fence(event.get('text', ''))
    if kind == 'job_started':
        return f'作業 {event.get("job_id")} 開始（{event.get("kind")}）', json.dumps(
            {k: v for k, v in event.items() if k not in ('seq', 'ts', 't', 'type')}, ensure_ascii=False, indent=2)
    if kind in ('job_failed', 'job_cancelled'):
        return f'作業 {event.get("job_id")} {"失敗" if kind == "job_failed" else "已取消"}：{event.get("reason", "")}', None
    if kind == 'review_result':
        return (f'審查第 {event.get("round")} 輪：agree={event.get("agree", "yes" if event.get("pass") else "no")}；累計通過 '
                f'{event.get("passes")}/{event.get("required")}',
                f'兩階段檢查：\n{event.get("thinking", "")}\n\n審查意見：\n{event.get("comment", "")}')
    if kind == 'review_skipped':
        return '⚠ 未審查（審查通過次數 n=0，對照組模式）', None
    if kind == 'writer_ops':
        return f'寫病歷操作：{event.get("summary", "")}', '\n'.join(event.get('logs', []))
    if kind == 'ops_rejected':
        return '寫病歷的行級操作不合法，整批退回', '\n'.join(event.get('problems', []))
    if kind == 'note_snapshot_pushed':
        return f'病歷新增版本 {event.get("index")}（來源：{event.get("source")}）', None
    if kind == 'note_truncate_redo':
        return f'回滾後修改：截斷 {len(event.get("snapshots", []))} 個後續版本（已於稽核保存）', json.dumps(
            event.get('snapshots', []), ensure_ascii=False, indent=2)
    if kind in ('note_undo', 'note_redo'):
        return f'病歷 {"回到" if kind == "note_undo" else "前進到"}版本 {event.get("index_after")}', None
    if kind == 'note_manual_edit':
        return f'醫師手動修改病歷 → 版本 {event.get("index")}', None
    if kind == 'advice_created':
        return f'問診建議第 {event.get("index")} 則完成', None
    if kind == 'analysis_stage':
        return f'整體分析階段：{event.get("stage")}', None
    if kind == 'analysis_created':
        return f'整體分析第 {event.get("index")} 則完成', None
    if kind == 'analysis_manual_edit':
        return f'醫師手動修改 A&T → 版本 {event.get("index")}（依據版本 {event.get("parent")}）', None
    if kind == 'script_mode_changed':
        return f'顯示字形切換：{event.get("mode")}', None
    if kind == 'visit_started':
        return f'看診開始（{event.get("visit_id")}）', None
    if kind == 'visit_finished':
        return f'看診結束並存檔（{event.get("visit_id")}）', None
    detail = {k: v for k, v in event.items() if k not in ('seq', 'ts', 't', 'type')}
    return f'{kind}', (json.dumps(detail, ensure_ascii=False, indent=2) if detail else None)


def render_log_md(visit_id: str, events: list[dict]) -> str:
    lines = [f'# 看診時間序 log · {visit_id}', '',
             '每筆格式：`時間 (t+音訊時間) [類型] 摘要`。t 為自看診開始的音訊秒數，可對應 audio.wav 位置。', '']
    for event in events:
        title, body = summarize_event(event)
        head = f'- `{_clock(event)}` **[{event.get("type")}]** {title}'
        lines.append(head)
        if body:
            lines.append('')
            lines.append(_details('展開', body))
            lines.append('')
    return '\n'.join(lines) + '\n'
