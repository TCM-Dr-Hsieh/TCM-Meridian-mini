"""Process-wide application state and the none -> imported -> visiting state machine."""
from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import Callable

from .config import CONFIG_PATH, TEMPLATES_DIR, Settings
from .fileio import atomic_write_text
from .llm import LLMClient, LLMScheduler
from .visit import VisitSession
from .visit_store import VisitStore
from .voice.asr import ASRError, LocalASR, validate_model_dir
from .voice.remote import RemoteAudioSource

TEMPLATE_FILES = {'record': 'record_template.txt', 'analysis': 'analysis_template.txt'}
TEMPLATE_LABELS = {'record': '病歷模板', 'analysis': '分析模板'}
SCRIPT_MODES = ('original', 'simplified', 'traditional')


class StateError(RuntimeError):
    """A user-facing refusal (wrong state, invalid input)."""


class AppState:
    def __init__(self, *, config_path: Path = CONFIG_PATH, client: LLMClient | None = None,
                 asr: LocalASR | None = None, source_factory: Callable | None = None,
                 llm_backoff: float = 1.0, templates_dir: Path = TEMPLATES_DIR):
        self.config_path = config_path
        self.templates_dir = templates_dir
        self.settings = Settings.load(config_path)
        self.client = client or LLMClient()
        self.scheduler = LLMScheduler(self.settings.llm.max_concurrency)
        self.asr = asr or LocalASR()
        self._source_factory = source_factory
        self._llm_backoff = llm_backoff
        self.phase = 'none'                  # none | imported | visiting
        self.patient_revision = 0           # moved by every assignment to patient_text; see patient_stamp
        self._patient_text = ''
        self.visit: VisitSession | None = None
        self.script_mode = 'original'
        self.asr_status = ''
        self.last_visit_folder: Path | None = None
        self.rev = 0
        self.notices: list[tuple[str, str]] = []
        self._preload_task: asyncio.Task | None = None
        self._busy_transition = False
        self.ensure_templates()

    # -- change tracking -----------------------------------------------
    def bump(self):
        self.rev += 1

    def notify(self, message: str, level: str = 'info'):
        self.notices.append((level, message))

    # -- templates -----------------------------------------------------
    def ensure_templates(self):
        for kind, filename in TEMPLATE_FILES.items():
            path = self.templates_dir / filename
            default = self.templates_dir / 'defaults' / filename
            if not path.exists() and default.exists():
                shutil.copyfile(default, path)

    def get_template(self, kind: str) -> str:
        path = self.templates_dir / TEMPLATE_FILES[kind]
        return path.read_text(encoding='utf-8') if path.exists() else ''

    def default_template(self, kind: str) -> str:
        path = self.templates_dir / 'defaults' / TEMPLATE_FILES[kind]
        return path.read_text(encoding='utf-8') if path.exists() else ''

    def save_template(self, kind: str, text: str):
        self._require_not_visiting('模板')
        atomic_write_text(self.templates_dir / TEMPLATE_FILES[kind], text)
        self.bump()

    # -- settings --------------------------------------------------------
    def _require_not_visiting(self, what: str):
        if self.phase == 'visiting' or self._busy_transition:
            raise StateError(f'看診中不可變更{what}。')

    def update_settings(self, new: Settings):
        self._require_not_visiting('設定')
        new.validate()
        new.save(self.config_path)
        self.settings = new
        self.scheduler.set_limit(new.llm.max_concurrency)
        self.bump()

    # -- patient -----------------------------------------------------------
    @property
    def patient_text(self) -> str:
        """The imported text before a visit (and the text a visit started with)."""
        return self._patient_text

    @patient_text.setter
    def patient_text(self, value: str):
        # Every assignment -- an import, a clear, the clearing when a visit ends -- moves the revision, so no path can change
        # the text and leave the stamp as it was (a stamp that came back to an old value would let a stale window through).
        self._patient_text = value
        self.patient_revision += 1

    @property
    def current_patient_text(self) -> str:
        """The imported patient data as it stands now. During a visit that is the visit's copy: editing the data in a visit
        changes only that copy, so `patient_text` is still the text the visit started with."""
        return self.visit.patient_text if self.visit is not None else self.patient_text

    @property
    def patient_stamp(self) -> tuple:
        """Which patient data a window was looking at. During a visit: the visit (by identity) and its version of the data;
        before one: the revision counter, which every import and clear moves. The state is shared by every browser tab, so
        a window that edits the data compares the stamp it took when it opened with the current one when it saves, and
        refuses when they differ (another tab saved or cleared, or the visit started or ended)."""
        if self.visit is not None:
            return (self.visit, self.visit.patient_version)
        return (None, self.patient_revision)

    def import_patient(self, text: str):
        if self.phase == 'visiting':
            assert self.visit is not None
            if self._busy_transition or self.visit.phase != 'recording':
                # Not only the toolbar: a window another tab left open must not edit a visit that is being closed.
                raise StateError('看診正在結束或已經結束存檔，無法再修改患者資料。')
            if self.visit.jobs.busy:
                raise StateError('作業進行中，請等作業完成或取消後再修改患者資料（避免結果用到過期的資料）。')
            self.visit.set_patient_text(text.strip())
            self.bump()
            return
        text = text.strip()
        if not text:
            raise StateError('請輸入患者資料。')
        self.patient_text = text
        self.phase = 'imported'
        self.schedule_asr_preload()
        self.bump()

    def clear_patient(self):
        if self.phase != 'imported':
            raise StateError('只有尚未開始看診時才能清除患者。')
        self.patient_text = ''
        self.phase = 'none'
        self.bump()

    # -- ASR preload -------------------------------------------------------
    def schedule_asr_preload(self):
        if self._preload_task and not self._preload_task.done():
            return
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._preload_task = loop.create_task(self._preload_asr())

    async def _preload_asr(self):
        self.asr_status = '載入中…'
        self.bump()
        try:
            info = await self.asr.load(self.settings.asr)
            self.asr_status = f'就緒（{info.get("device", "?")}）'
        except Exception as exc:
            self.asr_status = f'載入失敗：{exc}'
        self.bump()

    # -- visit -------------------------------------------------------------
    @property
    def remote_token(self) -> str:
        """The current visit's remote-microphone token ('' for a local-microphone visit). Only the browser that
        started the visit is given it."""
        source = self.visit.source if self.visit is not None else None
        return source.token if isinstance(source, RemoteAudioSource) else ''

    async def start_visit(self, *, remote: bool = False, device_label: str = '') -> Path:
        """Start recording. `remote=True` takes the audio from the starting browser's microphone (streamed over a
        WebSocket, see `voice/remote.py`) instead of this computer's microphone; `device_label` is the name of the
        microphone the browser opened, kept in the audit trail."""
        if self.phase != 'imported':
            raise StateError('請先匯入患者資料。')
        if self._busy_transition:
            raise StateError('正在處理上一個動作。')
        self._busy_transition = True
        try:
            if self._source_factory is None:
                try:
                    validate_model_dir(self.settings.asr.asr_model_dir)
                    validate_model_dir(self.settings.asr.aligner_model_dir, aligner=True)
                except ASRError as exc:
                    raise StateError(f'ASR 設定有問題：{exc}') from exc
            source_factory = self._source_factory
            if remote:
                asr = self.settings.asr

                def source_factory(path):
                    return RemoteAudioSource(window_seconds=asr.window_seconds, overlap_seconds=asr.overlap_seconds,
                                             recording_path=path, device_label=device_label)
            store = VisitStore.allocate(self.settings.visits_path())
            visit = VisitSession(store=store, settings=self.settings, patient_text=self.patient_text,
                                 record_template=self.get_template('record'),
                                 analysis_template=self.get_template('analysis'), client=self.client,
                                 scheduler=self.scheduler, asr=self.asr, on_change=self.bump,
                                 source_factory=source_factory, llm_backoff=self._llm_backoff)
            try:
                await visit.start()
            except Exception as exc:
                await visit.abort()
                raise StateError(f'無法開始看診：{exc}') from exc
            self.visit = visit
            self.phase = 'visiting'
            self.schedule_asr_preload()
            self.bump()
            return store.folder
        finally:
            self._busy_transition = False

    async def finish_visit(self) -> Path:
        if self.phase != 'visiting' or self.visit is None:
            raise StateError('目前不在看診中。')
        if self._busy_transition:
            raise StateError('正在處理上一個動作。')
        self._busy_transition = True
        try:
            folder = await self.visit.finish()
        finally:
            self._busy_transition = False
        self.last_visit_folder = folder
        self.visit = None
        self.patient_text = ''
        self.phase = 'none'
        self.bump()
        return folder

    # -- display script ---------------------------------------------------
    def set_script_mode(self, mode: str):
        if mode not in SCRIPT_MODES:
            raise StateError('不支援的簡繁顯示模式。')
        if mode == self.script_mode:
            return
        self.script_mode = mode
        if self.visit is not None:
            self.visit.log.emit('script_mode_changed', mode=mode)
        self.bump()

    def display(self, text: str) -> str:
        """Display-layer conversion only; stored text and LLM inputs are never converted."""
        from .textutil import script
        if self.script_mode == 'simplified':
            return script.to_simplified(text)
        if self.script_mode == 'traditional':
            return script.to_traditional(text)
        return text

    # -- shutdown ----------------------------------------------------------
    async def shutdown(self):
        if self.visit is not None and self.visit.phase in ('starting', 'recording', 'finishing', 'finish_failed'):
            try:
                await self.visit.abort()
            except Exception:
                pass
        if self._preload_task and not self._preload_task.done():
            self._preload_task.cancel()
        try:
            await self.asr.close()
        finally:
            await self.client.aclose()
