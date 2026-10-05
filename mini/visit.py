"""One consultation: audio + transcript + record versions + advice/analysis versions + audit files."""
from __future__ import annotations

import asyncio
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Callable

from . import __version__
from .agents.common import section
from .config import Settings
from .jobs import BusyError, JobManager
from .llm import LLMCaller, LLMClient, LLMScheduler
from .record.diff import build_diff_md
from .record.snapshots import SnapshotHistory
from .record.tags import tag_human_edits
from .visit_store import EventLog, VisitStore
from .voice.asr import LocalASR
from .voice.audio import make_source
from .voice.corrector import TranscriptCorrector
from .voice.pipeline import TranscriptPipeline

RECORD_AGENTS = ('record_writer', 'hallucination_corrector')


def _now() -> str:
    return datetime.now().isoformat(timespec='seconds')


@dataclass
class AdviceVersion:
    index: int
    timestamp: str
    t: float | None
    western_ddx: str
    tcm_ddx: str
    next_questions: str
    note_snapshot_id: str = ''
    call_ids: list = field(default_factory=list)
    job_id: str = ''
    patient_version: int = 0         # which version of the imported patient data this result was based on


@dataclass
class AnalysisVersion:
    index: int
    timestamp: str
    t: float | None
    source: str                      # 'llm' | '醫師手動'
    final_at: str
    arbitration_notes: str = ''
    parent: int | None = None
    note_snapshot_id: str = ''
    job_id: str = ''
    names: dict = field(default_factory=dict)
    professors: dict = field(default_factory=dict)
    calls: dict = field(default_factory=dict)
    patient_version: int = 0


@dataclass
class DeidVersion:
    """One 去識別化 result: the three blocks the model returned, plus what a scan of them found."""
    index: int
    timestamp: str
    t: float | None
    date_text: str
    patient_text: str
    note_text: str
    summary: str = ''                # the model's account of what it removed (for the physician, never copied)
    warnings: list = field(default_factory=list)       # residual identifiers found by deid_check
    extra_instruction: str = ''
    note_snapshot_id: str = ''
    call_ids: list = field(default_factory=list)
    job_id: str = ''
    patient_version: int = 0

    @property
    def text(self) -> str:
        """What the physician copies: the three blocks under plain headings, with no source tags or provenance notes."""
        return '\n\n'.join([section('今日看診日期', self.date_text), section('患者匯入資料', self.patient_text),
                           section('今日病歷', self.note_text)])


class VisitSession:
    def __init__(self, *, store: VisitStore, settings: Settings, patient_text: str, record_template: str,
                 analysis_template: str, client: LLMClient, scheduler: LLMScheduler, asr: LocalASR,
                 on_change: Callable[[], None] | None = None, source_factory: Callable | None = None,
                 llm_backoff: float = 1.0, today: date | None = None):
        self.store = store
        self.visit_id = store.visit_id
        # The visit's date, fixed when the visit starts (not per job, so it cannot change across midnight); every
        # prompt that carries the 今日病歷 is given it.
        self.visit_date: date = today or datetime.now().date()
        self.settings = deepcopy(settings)
        self.patient_text = patient_text
        self.record_template = record_template
        self.analysis_template = analysis_template
        self.asr = asr
        self._on_change = on_change or (lambda: None)
        self._source_factory = source_factory or (lambda path: make_source(self.settings.asr, path))
        self.source = None
        self.pipeline: TranscriptPipeline | None = None
        self.log = EventLog(store.log_path, self.audio_t)
        self.calls: dict[str, dict] = {}

        def timeout() -> float:
            return self.settings.llm.timeout_seconds

        def retries() -> int:
            return self.settings.llm.retries

        self.caller = LLMCaller(client, scheduler, timeout=timeout, retries=retries,
                                logger=self._log_llm_call, backoff=llm_backoff)
        self._transcript_caller = LLMCaller(client, scheduler, timeout=timeout, retries=retries,
                                            logger=self._log_transcript_llm, id_prefix='t', backoff=llm_backoff)
        self.corrector = TranscriptCorrector(self._transcript_caller,
                                             lambda: self.settings.agents['transcript_corrector'],
                                             lambda: self.settings.asr.correction_vocabulary)
        self.note = SnapshotHistory()
        self.note_audit: list[dict] = []
        self.patient_versions: list[dict] = []
        self.advice: list[AdviceVersion] = []
        self.advice_index = -1
        self.analysis: list[AnalysisVersion] = []
        self.analysis_index = -1
        self.deid: list[DeidVersion] = []
        self.deid_index = -1
        # The physician's extra instruction for 去識別化: kept for the whole visit and sent with every run until cleared.
        self.deid_extra = ''
        self.jobs = JobManager(self)
        self.phase = 'starting'          # starting | recording | finishing | finish_failed | finished
        self.finish_error = ''
        self._finished_logged = False
        self.started_at = _now()
        self.ended_at = ''
        self.writer_count = 0
        self.rev = 0
        self.transcript_llm_calls = 0
        self._tasks: list[asyncio.Task] = []
        self._mic_error_logged = ''
        self._last_flush = 0.0
        self._flushed_version = -1

    # -- basics --------------------------------------------------------
    def audio_t(self) -> float | None:
        return self.source.audio_time if self.source is not None else None

    def on_change(self):
        self.rev += 1
        self._on_change()

    def _log_llm_call(self, record: dict):
        self.calls[record['call_id']] = record
        self.log.emit('llm_call', **record)

    def _log_transcript_llm(self, record: dict):
        self.transcript_llm_calls += 1
        self.store.append_transcript_llm({'ts': _now(), 't': self.audio_t(), **record})

    def _require_idle(self):
        if self.jobs.busy:
            raise BusyError('作業進行中，暫時無法修改或切換病歷版本。')

    # -- lifecycle -----------------------------------------------------
    async def start(self):
        """Create the audio source and start the transcript pipeline. Raises on hardware failure."""
        self._write_meta('recording')
        self.patient_versions.append({'ts': _now(), 't': 0.0, 'text': self.patient_text})
        self._persist_patient()
        self.log.emit('visit_started', visit_id=self.visit_id, folder=str(self.store.folder),
                      app_version=__version__, settings=self.settings.public_dict())
        self.log.emit('patient_input_set', text=self.patient_text)
        self.note.push('', 'init', t=0.0)
        self.persist_note()
        self.source = self._source_factory(self.store.audio_path)
        self.source.start()
        remote = getattr(self.source, 'kind', 'local') == 'remote'
        self.log.emit('mic_started', device=self.source.label if remote else (self.settings.asr.microphone_id or '系統預設麥克風'),
                      audio_source='remote' if remote else 'local',
                      window_seconds=self.settings.asr.window_seconds,
                      overlap_seconds=self.settings.asr.overlap_seconds)
        self.pipeline = TranscriptPipeline(asr=self.asr, asr_settings=self.settings.asr, corrector=self.corrector,
                                           source=self.source, emit=self.log.emit)
        self._tasks = [asyncio.create_task(self.pipeline.run(), name='transcript-pipeline'),
                       asyncio.create_task(self._watch(), name='visit-watch')]
        self.phase = 'recording'
        self.on_change()

    async def _watch(self):
        """Persist the transcript periodically and surface microphone errors."""
        while True:
            await asyncio.sleep(1.0)
            if self.source is not None and self.source.error and self.source.error != self._mic_error_logged:
                self._mic_error_logged = self.source.error
                self.log.emit('mic_error', error=self.source.error)
                self.on_change()
            if (self.pipeline is not None and self.pipeline.version != self._flushed_version
                    and time.monotonic() - self._last_flush >= 5.0):
                self.flush_transcript()

    def flush_transcript(self):
        if self.pipeline is None:
            return
        self._flushed_version = self.pipeline.version
        self._last_flush = time.monotonic()
        self.store.write_json('transcript.json', self.pipeline.to_json())
        self.store.write_text('transcript.txt', self.pipeline.transcript_txt())

    async def finish(self) -> Path:
        """Stop recording, let the transcript drain, write every final file.

        Safe to call again after a failure: every step is idempotent. A failure leaves the visit in
        `finish_failed` (never stuck in `finishing`), with `finish_error` set, so the physician can retry.
        """
        if self.phase == 'finished':
            return self.store.folder
        self.phase = 'finishing'
        self.finish_error = ''
        self.on_change()
        try:
            await self._finish_steps()
        except BaseException as exc:
            self.phase = 'finish_failed'
            self.finish_error = f'{type(exc).__name__}: {exc}' if str(exc) else type(exc).__name__
            try:
                self.log.emit('visit_finish_failed', error=self.finish_error)
            except Exception:
                pass
            self.on_change()
            raise
        return self.store.folder

    async def _finish_steps(self):
        await self.jobs.cancel_and_wait()
        if self.source is not None:
            self.source.request_stop()
        if self._tasks:
            # The pipeline ends once the queue is drained and the source is done. If it crashed, keep going
            # with what was transcribed so far rather than making the visit impossible to close.
            outcome = (await asyncio.gather(self._tasks[0], return_exceptions=True))[0]
            if isinstance(outcome, Exception):
                self.log.emit('transcript_pipeline_error', error=f'{type(outcome).__name__}: {outcome}')
            for task in self._tasks[1:]:
                task.cancel()
            await asyncio.gather(*self._tasks[1:], return_exceptions=True)
        if self.source is not None:
            await self.source.close()
        self.flush_transcript()
        self.persist_note()
        self._persist_patient()
        current = self.note.current_note()
        self.store.write_text('今日病歷.md', current + ('\n' if current and not current.endswith('\n') else ''))
        shown = self.analysis[self.analysis_index] if 0 <= self.analysis_index < len(self.analysis) else None
        if shown is not None:
            self.store.write_text('分析與處置.md', shown.final_at.rstrip('\n') + '\n')
        self.store.write_json('analysis/index.json', self._analysis_index_doc())
        self.store.write_json('advice/index.json', {
            'displayed_index': self.advice_index + 1 if self.advice_index >= 0 else None,
            'versions': [{'index': v.index, 'timestamp': v.timestamp} for v in self.advice]})
        if self.deid:
            self.store.write_json('deidentified/index.json', self._deid_index_doc())
        if not self.ended_at:
            self.ended_at = _now()
        if not self._finished_logged:
            self.log.emit('visit_finished', visit_id=self.visit_id,
                          segments=len(self.pipeline.segments) if self.pipeline else 0,
                          note_versions=len(self.note.snapshots), advice_versions=len(self.advice),
                          analysis_versions=len(self.analysis), deid_versions=len(self.deid))
            self._finished_logged = True            # only once it is really in the log; a failed write is retried
        self._write_meta('finished')
        self.store.write_log_md(self.log.events)
        self.log.close()
        self.phase = 'finished'                       # only after every file is on disk
        self.on_change()

    async def abort(self):
        """Best-effort shutdown (app exit): keep what exists, mark the visit aborted."""
        await self.jobs.cancel_and_wait()
        if self.source is not None:
            self.source.cancelled.set()
            self.source.request_stop()
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        if self.source is not None:
            await self.source.close()
        try:
            self.flush_transcript()
            self.persist_note()
        finally:
            self.ended_at = _now()
            self.phase = 'finished'
            self.log.emit('visit_aborted', visit_id=self.visit_id)
            self._write_meta('aborted')
            self.log.close()
            self.store.write_log_md(self.log.events)

    def _write_meta(self, status: str):
        self.store.write_json('meta.json', {
            'visit_id': self.visit_id, 'folder': str(self.store.folder), 'status': status,
            'visit_date': self.visit_date.isoformat(),
            'started_at': self.started_at, 'ended_at': self.ended_at, 'app_version': __version__,
            'settings': self.settings.public_dict(),
            'counts': {'note_versions': len(self.note.snapshots), 'advice_versions': len(self.advice),
                       'analysis_versions': len(self.analysis), 'deid_versions': len(self.deid),
                       'segments': len(self.pipeline.segments) if self.pipeline else 0,
                       'transcript_llm_calls': self.transcript_llm_calls}})

    # -- patient input -------------------------------------------------
    @property
    def patient_version(self) -> int:
        """1-based number of the current version of the imported patient data."""
        return len(self.patient_versions)

    def set_patient_text(self, text: str):
        if text == self.patient_text:
            return
        self.patient_text = text
        self.patient_versions.append({'ts': _now(), 't': self.audio_t(), 'text': text})
        self._persist_patient()
        self.log.emit('patient_input_updated', text=text)
        self.on_change()

    def _persist_patient(self):
        self.store.write_json('patient_input.json', {'versions': self.patient_versions})
        self.store.write_text('patient_input.txt', self.patient_text + ('\n' if self.patient_text else ''))

    # -- daily record (NOTE) ------------------------------------------
    def next_writer_source(self, skipped: bool = False) -> str:
        self.writer_count += 1
        return f'病歷書寫 #{self.writer_count}' + ('（未審查）' if skipped else '')

    def push_note(self, note: str, source: str, meta: dict | None = None):
        before = self.note.current_index
        truncated = self.note.push(note, source, t=self.audio_t(), meta=meta)
        if truncated:
            self.note_audit.append({'event': 'truncate_redo', 'ts': _now(), 'source': source,
                                    'index_before': before + 1, 'index_after': self.note.current_index + 1,
                                    'snapshots': truncated})
            self.log.emit('note_truncate_redo', source=source, index_before=before + 1,
                          index_after=self.note.current_index + 1, snapshots=truncated)
        self.log.emit('note_snapshot_pushed', index=self.note.current_index + 1, source=source,
                      job_id=(meta or {}).get('job_id'), snapshot_id=self.note.get_current()['id'])
        self.persist_note()
        self.on_change()

    def edit_note_manual(self, new_text: str) -> bool:
        """Apply a physician edit; returns False when nothing changed."""
        self._require_idle()
        current = self.note.current_note()
        tagged = tag_human_edits(current, new_text)
        if tagged == current:
            return False
        self.push_note(tagged, '醫師手動')
        self.log.emit('note_manual_edit', index=self.note.current_index + 1)
        return True

    def undo_note(self) -> bool:
        self._require_idle()
        before = self.note.current_index
        if self.note.undo() is None:
            return False
        self.log.emit('note_undo', index_before=before + 1, index_after=self.note.current_index + 1)
        self.persist_note()
        self.on_change()
        return True

    def redo_note(self) -> bool:
        self._require_idle()
        before = self.note.current_index
        if self.note.redo() is None:
            return False
        self.log.emit('note_redo', index_before=before + 1, index_after=self.note.current_index + 1)
        self.persist_note()
        self.on_change()
        return True

    def persist_note(self):
        calls = [self.calls[c] for c in sorted(self.calls) if self.calls[c].get('agent') in RECORD_AGENTS]
        data = self.note.to_dict()
        data.update(truncated_audit=self.note_audit, llm_calls=calls)
        self.store.write_json('note/history.json', data)
        self.store.write_text('note/diff.md', build_diff_md(self.note.snapshots, self.note.current_index))

    # -- advice ----------------------------------------------------------
    def add_advice(self, parts: dict, note_snapshot_id: str, call_ids: list[str], job_id: str,
                   patient_version: int | None = None) -> AdviceVersion:
        version = AdviceVersion(len(self.advice) + 1, _now(), self.audio_t(), parts['western_ddx'],
                                parts['tcm_ddx'], parts['next_questions'], note_snapshot_id, list(call_ids), job_id,
                                patient_version or self.patient_version)
        self.advice.append(version)
        self.advice_index = len(self.advice) - 1
        name = f'advice/{version.index:03d}'
        self.store.write_json(name + '.json', {**asdict(version), 'llm_calls': [self.calls[c] for c in call_ids if c in self.calls]})
        self.store.write_text(name + '.md', (
            f'# 問診建議 第 {version.index} 則 · {version.timestamp}\n\n'
            f'## 建議西醫疾病鑑別診斷\n\n{version.western_ddx}\n\n'
            f'## 建議中醫證型鑑別診斷\n\n{version.tcm_ddx}\n\n'
            f'## 建議提出的問診問題\n\n{version.next_questions}\n'))
        self.log.emit('advice_created', index=version.index, job_id=job_id, call_ids=call_ids)
        self.on_change()
        return version

    def set_advice_index(self, index: int):
        if 0 <= index < len(self.advice):
            self.advice_index = index
            self.on_change()

    # -- analysis --------------------------------------------------------
    def _analysis_index_doc(self) -> dict:
        return {'displayed_index': self.analysis_index + 1 if self.analysis_index >= 0 else None,
                'versions': [{'index': v.index, 'source': v.source, 'parent': v.parent, 'timestamp': v.timestamp}
                             for v in self.analysis]}

    def _write_analysis(self, version: AnalysisVersion):
        name = f'analysis/{version.index:03d}'
        call_ids = [c for ids in version.calls.values() for c in ids]
        self.store.write_json(name + '.json', {**asdict(version),
                                               'llm_calls': [self.calls[c] for c in call_ids if c in self.calls]})
        names = version.names or {}
        lines = [f'# 整體分析 第 {version.index} 則 · {version.timestamp} · 來源：{version.source}', '']
        if version.parent:
            lines += [f'（醫師手動修改，依據第 {version.parent} 則）', '']
        lines += ['## 最終 A&T', '', version.final_at, '']
        if version.source == 'llm':
            lines += ['## 仲裁說明', '', version.arbitration_notes, '',
                      f'## {names.get("a", "教授甲")} 的 A&T', '', version.professors.get('a', ''), '',
                      f'## {names.get("b", "教授乙")} 的 A&T', '', version.professors.get('b', ''), '',
                      f'## {names.get("a", "教授甲")} 對 {names.get("b", "教授乙")} 的評比', '',
                      version.professors.get('review_a_on_b', ''), '',
                      f'## {names.get("b", "教授乙")} 對 {names.get("a", "教授甲")} 的評比', '',
                      version.professors.get('review_b_on_a', ''), '']
        self.store.write_text(name + '.md', '\n'.join(lines))
        self.store.write_json('analysis/index.json', self._analysis_index_doc())

    def add_analysis(self, *, final_at: str, arbitration_notes: str, note_snapshot_id: str, job_id: str,
                     names: dict, professors: dict, calls: dict,
                     patient_version: int | None = None) -> AnalysisVersion:
        version = AnalysisVersion(len(self.analysis) + 1, _now(), self.audio_t(), 'llm', final_at,
                                  arbitration_notes, None, note_snapshot_id, job_id, names, professors, calls,
                                  patient_version or self.patient_version)
        self.analysis.append(version)
        self.analysis_index = len(self.analysis) - 1
        self._write_analysis(version)
        self.log.emit('analysis_created', index=version.index, job_id=job_id, calls=calls)
        self.on_change()
        return version

    def edit_analysis_manual(self, text: str) -> AnalysisVersion | None:
        """Append a physician-edited version (never truncates or overwrites earlier versions)."""
        self._require_idle()
        shown = self.analysis[self.analysis_index] if 0 <= self.analysis_index < len(self.analysis) else None
        if shown is not None and shown.final_at == text:
            return None
        if shown is None and not text.strip():
            return None
        parent = shown.index if shown is not None else None
        version = AnalysisVersion(len(self.analysis) + 1, _now(), self.audio_t(), '醫師手動', text, '', parent,
                                  patient_version=self.patient_version)
        self.analysis.append(version)
        self.analysis_index = len(self.analysis) - 1
        self._write_analysis(version)
        self.log.emit('analysis_manual_edit', index=version.index, parent=parent)
        self.on_change()
        return version

    def set_analysis_index(self, index: int):
        if 0 <= index < len(self.analysis):
            self.analysis_index = index
            self.store.write_json('analysis/index.json', self._analysis_index_doc())
            self.on_change()

    # -- de-identification -------------------------------------------------
    def _deid_index_doc(self) -> dict:
        return {'displayed_index': self.deid_index + 1 if self.deid_index >= 0 else None,
                'versions': [{'index': v.index, 'timestamp': v.timestamp, 'warnings': len(v.warnings)}
                             for v in self.deid]}

    def add_deid(self, parts: dict, warnings: list[str], extra_instruction: str, note_snapshot_id: str,
                 call_ids: list[str], job_id: str, patient_version: int | None = None) -> DeidVersion:
        """Append a result (earlier ones stay; the physician may run it again and browse them).

        `deidentified/NNN.md` is exactly what 複製 copies: no heading, no timestamp (the real time of day is the date this
        feature hides). The LLM input (which is the identifiable original) is in `NNN.json` under `llm_calls` and in
        log.jsonl, like every other LLM call.
        """
        version = DeidVersion(len(self.deid) + 1, _now(), self.audio_t(), parts['date'], parts['patient'], parts['note'],
                              parts.get('summary', ''), list(warnings), extra_instruction, note_snapshot_id,
                              list(call_ids), job_id, patient_version or self.patient_version)
        self.deid.append(version)
        self.deid_index = len(self.deid) - 1
        name = f'deidentified/{version.index:03d}'
        self.store.write_json(name + '.json', {**asdict(version), 'text': version.text,
                                               'llm_calls': [self.calls[c] for c in call_ids if c in self.calls]})
        self.store.write_text(name + '.md', version.text + '\n')
        self.store.write_json('deidentified/index.json', self._deid_index_doc())
        self.log.emit('deid_created', index=version.index, job_id=job_id, call_ids=call_ids,
                      warnings=len(version.warnings))
        self.on_change()
        return version

    def set_deid_index(self, index: int):
        if 0 <= index < len(self.deid):
            self.deid_index = index
            self.on_change()

    # -- jobs ------------------------------------------------------------
    def start_job(self, kind: str, **options):
        """Start a job. `options` go to the job's runner (only 去識別化 takes one: `extra`, the physician's extra instruction)."""
        from .agents import advice_job, analysis_job, deid_job, record_job
        runner = {'record': record_job.run, 'advice': advice_job.run, 'analysis': analysis_job.run,
                  'deidentify': deid_job.run}[kind]
        if self.phase != 'recording':
            raise BusyError('目前不在看診中。')
        return self.jobs.start(kind, lambda job: runner(self, job, **options))
