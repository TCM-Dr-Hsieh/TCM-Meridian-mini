"""The single page: toolbar, status bar, and the three-column workspace."""
from __future__ import annotations

import asyncio
import html
import json
import time
from pathlib import Path
from types import SimpleNamespace

from nicegui import ui

from ..jobs import BusyError
from ..record.diff import diff_html, simple_md_render
from ..record.tags import strip_citations
from ..speaker import BACKGROUND, LABEL_NAMES, TEXT_SOURCES, UNKNOWN
from ..state import AppState, StateError
from ..textutil import format_clock
from . import dialogs
from .style import CSS, JS

REMOTE_JS = Path(__file__).with_name('remote_capture.js')     # served by app.py at REMOTE_JS_URL
REMOTE_JS_URL = '/mini-remote-capture.js'
EMPTY = '<span class="empty">（空白）</span>'
BACKLOG_WARNING_WINDOWS = 10        # ~30 s of audio waiting for ASR / correction (one window per 3 s stride)
WAIT_FOR_BROWSER_SECONDS = 15       # a remote-microphone visit whose browser has not connected after this long is flagged


def _placeholder(text: str) -> str:
    return f'<span class="empty">{html.escape(text)}</span>'


class MainPage:
    def __init__(self, app: AppState):
        self.app = app
        self.note_mode = 'browse'            # browse | diff | edit
        self.analysis_editing = False
        self.patient_open = False            # the floating 患者資訊 panel; a reloaded page starts with it closed
        self._sigs: dict[str, object] = {}
        self.w = SimpleNamespace()
        self.build()
        ui.timer(0.3, self.tick)

    # ------------------------------------------------------------------ build
    def build(self):
        ui.add_css(CSS)
        ui.add_head_html(f'<script>{JS}</script>')
        ui.add_head_html(f'<script src="{REMOTE_JS_URL}?v=1"></script>')
        ui.on('mini_remote', self.on_remote_event)
        ui.colors(primary='#2d6a4f', secondary='#52665d', accent='#c58d23', positive='#2e7d64')
        w = self.w
        with ui.element('div').classes('shell'):
            with ui.element('div').classes('toolbar'):
                w.btn_import = ui.button('患者匯入', icon='person_add', on_click=lambda: dialogs.open_patient_dialog(self.app))
                w.btn_start = ui.button('開始看診', icon='mic', on_click=self.start_visit)
                # Remote microphone: this browser's microphone instead of the one on the computer running the program.
                # The start button's click handler runs in the browser (a gesture is needed to open the microphone);
                # it reports back through the 'mini_remote' event, see on_remote_event.
                w.btn_start_remote = ui.button('開始看診', icon='mic').on(
                    'click', js_handler='() => window.miniRemote.begin()')
                w.src_toggle = ui.toggle({'local': '本機麥克風', 'remote': '此瀏覽器麥克風'}, value='local',
                                         on_change=self.on_source_change) \
                    .props('dense no-caps unelevated color=white text-color=primary toggle-color=primary '
                           'toggle-text-color=white').classes('source-toggle') \
                    .tooltip('本機＝執行程式的那台電腦的麥克風；此瀏覽器＝你現在使用的這台電腦的麥克風（例如透過 Cloudflare 連線時）')
                w.mic_select = ui.select({'': '預設麥克風'}, value='', on_change=self.on_mic_change) \
                    .props('dense outlined options-dense').classes('mic-select')
                w.btn_mic_refresh = ui.button(icon='refresh', on_click=self.refresh_mics).props('dense flat round') \
                    .tooltip('重新整理這台電腦的麥克風清單（瀏覽器會要求麥克風權限）')
                w.btn_end = ui.button('結束並存檔', icon='save', on_click=self.end_visit).props('color=secondary')
                ui.element('div').classes('spacer')
                w.btn_simplified = ui.button('轉簡體', on_click=lambda: self.toggle_script('simplified')).props('outline')
                w.btn_traditional = ui.button('轉繁體', on_click=lambda: self.toggle_script('traditional')).props('outline')
                w.btn_model = ui.button('模型設定', icon='tune', on_click=lambda: dialogs.open_settings_dialog(self.app)) \
                    .props('outline')
                w.btn_template = ui.button('模板設定', icon='description',
                                           on_click=lambda: dialogs.open_template_dialog(self.app)).props('outline')
            with ui.element('div').classes('toolbar'):
                w.btn_write = ui.button('病歷書寫', icon='edit_note', on_click=lambda: self.run_job('record'))
                w.btn_advice = ui.button('問診建議', icon='lightbulb', on_click=lambda: self.run_job('advice'))
                w.btn_analysis = ui.button('整體分析', icon='psychology', on_click=lambda: self.run_job('analysis'))
                # A read-only, draggable reference panel for the imported patient data. It never touches the visit, so it
                # is always available (no visit, a running job, note editing) and is not part of update_controls.
                w.btn_patient = ui.button('患者資訊', icon='assignment_ind', on_click=self.toggle_patient).props('outline')
                # Opens a window to browse and run 去識別化. It opens while another job runs too (to read earlier results);
                # the window's own 去識別化 button waits for the single job slot.
                w.btn_deid = ui.button('LLM 去識別化', icon='shield', on_click=self.open_deid).props('outline')
                # The voice groups of the speaker marking (who is the doctor, who spoke how much). Only in a visit that runs it.
                w.btn_speakers = ui.button('說話者群', icon='record_voice_over',
                                           on_click=lambda: dialogs.open_speaker_dialog(self.app)) \
                    .props('outline').mark('speakers-button')
                w.btn_cancel = ui.button('取消作業', icon='close', on_click=self.cancel_job).props('flat color=negative')
            with ui.element('div').classes('status-bar'):
                w.status = ui.label('').classes('grow')
                w.job = ui.label('').classes('job')
            with ui.element('div').classes('cols'):
                with ui.element('div').classes('work-col'):
                    self.build_note_panel()
                    self.build_analysis_panel()
                with ui.element('div').classes('work-col'):
                    self.build_advice_panel()
                with ui.element('div').classes('work-col'):
                    self.build_transcript_panel()
        self.build_patient_panel()
        self.render_all()
        self.update_status()
        self.update_controls()

    def build_note_panel(self):
        w = self.w
        with ui.element('div').classes('panel g3'):
            with ui.element('div').classes('panel-head'):
                ui.label('今日病歷').classes('panel-title')
                w.btn_browse = ui.button('瀏覽', on_click=lambda: self.set_note_mode('browse')).props('dense no-caps')
                w.btn_diff = ui.button('差異', on_click=lambda: self.set_note_mode('diff')).props('dense no-caps')
                w.btn_edit = ui.button('修改', icon='edit', on_click=self.enter_note_edit).props('dense no-caps')
                w.btn_edit_done = ui.button('完成', icon='check', on_click=lambda: self.leave_note_edit(True)) \
                    .props('dense no-caps color=positive')
                w.btn_edit_cancel = ui.button('放棄', icon='close', on_click=lambda: self.leave_note_edit(False)) \
                    .props('dense no-caps flat')
                w.btn_undo = ui.button(icon='undo', on_click=self.undo_note).props('dense flat round').tooltip('上一版（回滾）')
                w.btn_redo = ui.button(icon='redo', on_click=self.redo_note).props('dense flat round').tooltip('下一版')
                w.btn_copy_note = ui.button(icon='content_copy', on_click=self.copy_note).props('dense flat round') \
                    .tooltip('複製（去來源標籤）')
                w.note_version = ui.label('').classes('version-label')
            with ui.element('div').classes('panel-body') as w.note_body:
                w.note_html = ui.html('', sanitize=False)
            w.note_editor = ui.textarea().props('outlined').classes('editor w-full')
            w.note_editor.set_visibility(False)

    def build_analysis_panel(self):
        w = self.w
        with ui.element('div').classes('panel g2'):
            with ui.element('div').classes('panel-head'):
                ui.label('分析與處置').classes('panel-title')
                w.btn_at_prev = ui.button(icon='chevron_left', on_click=lambda: self.nav_analysis(-1)).props('dense flat round')
                w.at_label = ui.label('').classes('version-label')
                w.btn_at_next = ui.button(icon='chevron_right', on_click=lambda: self.nav_analysis(1)).props('dense flat round')
                w.btn_at_edit = ui.button('修改', icon='edit', on_click=self.enter_analysis_edit).props('dense no-caps')
                w.btn_at_save = ui.button('儲存', icon='check', on_click=lambda: self.leave_analysis_edit(True)) \
                    .props('dense no-caps color=positive')
                w.btn_at_cancel = ui.button('放棄', icon='close', on_click=lambda: self.leave_analysis_edit(False)) \
                    .props('dense no-caps flat')
                w.btn_process = ui.button('教授過程', icon='groups', on_click=lambda: dialogs.open_process_dialog(self.app)) \
                    .props('dense no-caps outline')
                w.btn_copy_at = ui.button(icon='content_copy', on_click=self.copy_analysis).props('dense flat round') \
                    .tooltip('複製')
            with ui.element('div').classes('panel-body') as w.at_body:
                w.at_md = ui.markdown('', extras=['fenced-code-blocks', 'tables'])
            w.at_editor = ui.textarea().props('outlined').classes('editor w-full')
            w.at_editor.set_visibility(False)

    def build_advice_panel(self):
        w = self.w
        with ui.element('div').classes('panel'):
            with ui.element('div').classes('panel-head').style('margin-bottom:0'):
                ui.label('問診建議').classes('panel-title')
                w.btn_ad_prev = ui.button(icon='chevron_left', on_click=lambda: self.nav_advice(-1)).props('dense flat round')
                w.ad_label = ui.label('').classes('version-label')
                w.btn_ad_next = ui.button(icon='chevron_right', on_click=lambda: self.nav_advice(1)).props('dense flat round')
        w.advice_md = {}
        for key, title in (('western_ddx', '建議西醫疾病鑑別診斷'), ('tcm_ddx', '建議中醫證型鑑別診斷'),
                           ('next_questions', '建議提出的問診問題')):
            with ui.element('div').classes('panel g1'):
                ui.label(title).classes('panel-title')
                with ui.element('div').classes('panel-body'):
                    w.advice_md[key] = ui.markdown('', extras=['fenced-code-blocks', 'tables'])

    def build_transcript_panel(self):
        w = self.w
        with ui.element('div').classes('panel grow'):
            with ui.element('div').classes('panel-head'):
                ui.label('語音文字辨識').classes('panel-title')
                w.level = ui.linear_progress(value=0, show_value=False).props('rounded size=6px color=primary') \
                    .classes('grow')
                w.tx_info = ui.label('').classes('version-label')
            with ui.element('div').classes('panel-body') as w.tx_scroll:
                w.tx_html = ui.html('', sanitize=False)

    def build_patient_panel(self):
        """A floating window with no backdrop (so the note behind it stays usable); dragging and resizing are done in the
        browser, see `window.miniFloat` in style.py."""
        w = self.w
        with ui.element('div').classes('float-panel') as w.patient_panel:
            with ui.element('div').classes('float-head').tooltip('拖曳標題列移動；連按兩下回到預設位置；右下角可調整大小'):
                ui.label('患者資訊').classes('float-title')
                ui.element('div').classes('spacer').style('flex:1 1 auto')
                w.btn_patient_copy = ui.button(icon='content_copy', on_click=self.copy_patient) \
                    .props('dense flat round').tooltip('複製（去來源標籤）').mark('patient-copy')
                ui.button(icon='close', on_click=self.close_patient).props('dense flat round').tooltip('關閉') \
                    .mark('patient-close')
            with ui.element('div').classes('float-body'):
                w.patient_text = ui.label('').classes('patient-text')
        w.patient_panel.set_visibility(False)

    # ------------------------------------------------------------------ helpers
    @property
    def visit(self):
        return self.app.visit

    def busy(self) -> bool:
        return bool(self.visit and self.visit.jobs.busy)

    def editing(self) -> bool:
        return self.note_mode == 'edit' or self.analysis_editing

    def safe(self, fn, *args):
        try:
            return fn(*args)
        except (BusyError, StateError) as exc:
            ui.notify(str(exc), type='warning')
        return None

    async def copy_text(self, text: str, what: str):
        await dialogs.copy_to_clipboard(text, what)

    # ------------------------------------------------------------------ actions
    async def start_visit(self):
        try:
            task = asyncio.create_task(self.app.start_visit())
            folder = await asyncio.shield(task)
        except StateError as exc:
            ui.notify(str(exc), type='negative', multi_line=True)
            return
        ui.notify(f'看診開始，資料夾：{folder.name}', type='positive')

    # -- remote microphone (this browser's microphone) -----------------------------------------
    @property
    def remote_mode(self) -> bool:
        return self.w.src_toggle.value == 'remote'

    async def on_source_change(self):
        if self.app.phase == 'visiting':               # only the display is synced to the running visit (see update_controls)
            self.update_controls()
            return
        if self.remote_mode:
            secure = await ui.run_javascript('window.isSecureContext && !!navigator.mediaDevices', timeout=5)
            if not secure:
                ui.notify('這個網址不是 HTTPS，瀏覽器不會開放麥克風。請改用 https:// 網址（例如 Cloudflare），'
                          '或在執行程式的電腦上用 localhost 開啟；否則請改選「本機麥克風」。',
                          type='warning', multi_line=True, close_button='關閉', timeout=0)
        self.update_controls()

    def on_mic_change(self):
        ui.run_javascript(f'window.miniRemote.deviceId = {json.dumps(self.w.mic_select.value or "")}')

    async def refresh_mics(self):
        try:
            devices = await ui.run_javascript('window.miniRemote.listDevices(true)', timeout=60)
        except Exception as exc:                      # permission refused, no microphone, timeout
            ui.notify(f'無法取得這台電腦的麥克風清單：{exc}', type='warning', multi_line=True)
            return
        options = {'': '預設麥克風', **{d['id']: d['label'] for d in devices or []}}
        current = self.w.mic_select.value
        self.w.mic_select.set_options(options, value=current if current in options else '')
        ui.notify(f'找到 {len(options) - 1} 支麥克風。', type='positive' if len(options) > 1 else 'warning')

    async def on_remote_event(self, event):
        """Reports from window.miniRemote (mini/ui/remote_capture.js)."""
        args = event.args
        payload = args if isinstance(args, dict) else (args[0] if args else {})
        kind, message = payload.get('type'), payload.get('message', '')
        if kind == 'prepared':                        # the microphone is open: start the visit and hand over the token
            if not self.remote_mode:                  # switched back to the local microphone while the prompt was open
                ui.run_javascript('window.miniRemote.abort()')
                ui.notify('已切回本機麥克風，取消這次遠端麥克風的看診。', type='info')
                return
            await self.start_remote_visit(str(payload.get('label') or ''))
        elif kind == 'error':
            ui.notify(message, type='negative', multi_line=True, close_button='關閉', timeout=0)
        elif kind == 'reconnecting':
            ui.notify('遠端麥克風連線暫時中斷，正在重新連線（30 秒內會自動補傳）…', type='warning')
        elif kind == 'reconnected':
            ui.notify('遠端麥克風已重新連線，中斷期間的聲音已補傳。', type='positive')
        elif kind == 'recording':
            ui.notify('遠端麥克風已連線，開始錄音。', type='positive')

    async def start_remote_visit(self, device_label: str = ''):
        try:
            task = asyncio.create_task(self.app.start_visit(remote=True, device_label=device_label))
            folder = await asyncio.shield(task)
        except StateError as exc:
            ui.run_javascript('window.miniRemote.abort()')
            ui.notify(str(exc), type='negative', multi_line=True)
            return
        except Exception as exc:
            ui.run_javascript('window.miniRemote.abort()')
            ui.notify(f'無法開始看診：{exc}', type='negative', multi_line=True)
            return
        ui.run_javascript(f'window.miniRemote.connect({json.dumps(self.app.remote_token)})')
        ui.notify(f'看診開始（此瀏覽器的麥克風），資料夾：{folder.name}', type='positive')

    async def end_visit(self):
        if self.editing():
            ui.notify('請先完成或放棄目前的修改，再結束看診。', type='warning')
            return
        if self.busy():
            confirmed = await self.confirm('目前有作業進行中，結束看診會取消該作業（已完成的 LLM 呼叫仍會保留在 log）。要繼續嗎？')
            if not confirmed:
                return
        try:
            task = asyncio.create_task(self.app.finish_visit())
            folder = await asyncio.shield(task)
        except StateError as exc:
            ui.notify(str(exc), type='negative')
            return
        except Exception as exc:                      # disk full, locked file, ...: the visit stays open and can be retried
            ui.notify(f'結束並存檔失敗：{exc}。看診資料夾仍保留，問題排除後可再按「結束並存檔」重試。',
                      type='negative', multi_line=True, close_button='關閉', timeout=0)
            return
        self.note_mode = 'browse'
        ui.notify(f'已結束並存檔：{folder}', type='positive', multi_line=True, close_button='關閉', timeout=20000)

    async def confirm(self, text: str) -> bool:
        with ui.dialog() as dialog, ui.card().classes('gap-3'):
            ui.label(text)
            with ui.row().classes('w-full justify-end'):
                ui.button('取消', on_click=lambda: dialog.submit(False)).props('flat')
                ui.button('確定', on_click=lambda: dialog.submit(True))
        return bool(await dialog)

    def run_job(self, kind: str):
        if self.editing():
            ui.notify('請先完成或放棄目前的修改。', type='warning')
            return
        if self.visit is None:
            return
        self.safe(self.visit.start_job, kind)

    def cancel_job(self):
        if self.visit:
            self.visit.jobs.cancel()

    # -- patient information (read-only reference panel) ----------------------
    def toggle_patient(self):
        self.set_patient_open(not self.patient_open)

    def close_patient(self):
        self.set_patient_open(False)

    def set_patient_open(self, opened: bool):
        w = self.w
        self.patient_open = opened
        w.patient_panel.set_visibility(opened)
        if opened:                                    # a filled button shows that the panel is open
            w.btn_patient.props(remove='outline')
            self.render_patient()
            ui.run_javascript(f"window.miniFloat.show('c{w.patient_panel.id}')")
        else:
            w.btn_patient.props('outline')

    def patient_source(self) -> str:
        return self.app.current_patient_text

    async def copy_patient(self):
        text = self.patient_source()
        if not text.strip():
            ui.notify('尚未匯入患者。', type='warning')
            return
        await self.copy_text(self.app.display(strip_citations(text)), '患者匯入資料（已去來源標籤）')

    def open_deid(self):
        if self.editing():
            ui.notify('請先完成或放棄目前的修改。', type='warning')
            return
        dialogs.open_deid_dialog(self.app)

    def toggle_script(self, mode: str):
        if self.editing():
            ui.notify('修改中不可切換簡繁顯示。', type='warning')
            return
        self.app.set_script_mode('original' if self.app.script_mode == mode else mode)

    # -- note ---------------------------------------------------------------
    def set_note_mode(self, mode: str):
        self.note_mode = mode
        self.render_all()
        self.update_controls()

    def enter_note_edit(self):
        if self.visit is None or self.busy():
            return
        if self.app.script_mode != 'original':
            self.app.set_script_mode('original')
            ui.notify('已還原為原文字形再進入修改。')
        self.w.note_editor.set_value(self.visit.note.current_note())
        self.note_mode = 'edit'
        self.render_all()
        self.update_controls()

    def leave_note_edit(self, save: bool):
        if save and self.visit:
            changed = self.safe(self.visit.edit_note_manual, self.w.note_editor.value or '')
            if changed is None:
                return
            ui.notify('修改已保存（新增行已標記 [醫師手動]）。' if changed else '未偵測到變更。',
                      type='positive' if changed else 'info')
        self.note_mode = 'browse'
        self.render_all()
        self.update_controls()

    def undo_note(self):
        if self.visit and self.safe(self.visit.undo_note):
            ui.notify(f'回到版本 {self.visit.note.current_index + 1}')

    def redo_note(self):
        if self.visit and self.safe(self.visit.redo_note):
            ui.notify(f'前進到版本 {self.visit.note.current_index + 1}')

    async def copy_note(self):
        if self.visit:
            await self.copy_text(self.app.display(strip_citations(self.visit.note.current_note())), '今日病歷（已去來源標籤）')

    # -- analysis -------------------------------------------------------------
    def nav_analysis(self, step: int):
        if self.visit and not self.analysis_editing:
            self.visit.set_analysis_index(self.visit.analysis_index + step)

    def enter_analysis_edit(self):
        v = self.visit
        if v is None or self.busy():
            return
        if self.app.script_mode != 'original':
            self.app.set_script_mode('original')
            ui.notify('已還原為原文字形再進入修改。')
        shown = v.analysis[v.analysis_index].final_at if 0 <= v.analysis_index < len(v.analysis) else ''
        self.w.at_editor.set_value(shown)
        self.analysis_editing = True
        self.render_all()
        self.update_controls()

    def leave_analysis_edit(self, save: bool):
        if save and self.visit:
            version = self.safe(self.visit.edit_analysis_manual, self.w.at_editor.value or '')
            ui.notify('已新增醫師手動修改版本。' if version else '未偵測到變更。',
                      type='positive' if version else 'info')
        self.analysis_editing = False
        self.render_all()
        self.update_controls()

    async def copy_analysis(self):
        v = self.visit
        if v and 0 <= v.analysis_index < len(v.analysis):
            await self.copy_text(self.app.display(v.analysis[v.analysis_index].final_at), '分析與處置')

    def nav_advice(self, step: int):
        if self.visit:
            self.visit.set_advice_index(self.visit.advice_index + step)

    # ------------------------------------------------------------------ rendering
    def render_all(self):
        self._sigs.clear()
        self.render_panels()

    def render_panels(self):
        self._maybe('note', self.note_signature(), self.render_note)
        self._maybe('analysis', self.analysis_signature(), self.render_analysis)
        self._maybe('advice', self.advice_signature(), self.render_advice)
        self._maybe('transcript', self.transcript_signature(), self.render_transcript)
        self._maybe('patient', self.patient_signature(), self.render_patient)

    def _maybe(self, name: str, signature, render):
        if self._sigs.get(name) != signature:
            self._sigs[name] = signature
            render()

    def note_signature(self):
        v = self.visit
        current = v.note.get_current() if v else None
        return (id(v), current['id'] if current else None, v.note.current_index if v else -1,
                len(v.note.snapshots) if v else 0, self.note_mode, self.app.script_mode)

    def analysis_signature(self):
        v = self.visit
        return (id(v), v.analysis_index if v else -1, len(v.analysis) if v else 0,
                self.analysis_editing, self.app.script_mode)

    def advice_signature(self):
        v = self.visit
        return (id(v), v.advice_index if v else -1, len(v.advice) if v else 0, self.app.script_mode)

    def patient_signature(self):
        return (id(self.visit), self.patient_source(), self.app.script_mode)

    def transcript_signature(self):
        v = self.visit
        pipeline = v.pipeline if v else None
        return (id(v), pipeline.version if pipeline else -1, pipeline.finished if pipeline else None,
                self.app.script_mode)

    def render_note(self):
        w, v, app = self.w, self.visit, self.app
        editing = self.note_mode == 'edit'
        w.note_body.set_visibility(not editing)
        w.note_editor.set_visibility(editing)
        if v is None:
            w.note_html.set_content(_placeholder('請先匯入患者並開始看診'))
            w.note_version.set_text('')
            return
        snapshot = v.note.get_current()
        w.note_version.set_text(
            f'版本 {v.note.current_index + 1}/{len(v.note.snapshots)} · 來源：{snapshot["source"]} · '
            f'{snapshot["timestamp"][11:19]}' + (f' · t+{format_clock(snapshot["t"])}' if snapshot.get('t') is not None else ''))
        if self.note_mode == 'diff':
            previous = v.note.get_previous()
            if previous is None:
                w.note_html.set_content(_placeholder('（無前一版可比較）'))
            else:
                index = v.note.current_index
                w.note_html.set_content(diff_html(app.display(previous['note']), app.display(snapshot['note']),
                                                  f'版本 {index} → 版本 {index + 1}'))
        elif self.note_mode == 'browse':
            text = app.display(strip_citations(snapshot['note']))
            w.note_html.set_content(simple_md_render(text) if text.strip() else EMPTY)

    def render_analysis(self):
        w, v, app = self.w, self.visit, self.app
        w.at_body.set_visibility(not self.analysis_editing)
        w.at_editor.set_visibility(self.analysis_editing)
        if v is None or not v.analysis:
            w.at_md.set_content('*（尚無分析；按「整體分析」產生，或按「修改」自行輸入）*' if v else '*（請先開始看診）*')
            w.at_label.set_text('0/0')
            return
        version = v.analysis[v.analysis_index]
        tag = '' if version.source == 'llm' else ' · 醫師手動' + (f'（依第 {version.parent} 則）' if version.parent else '')
        w.at_label.set_text(f'{v.analysis_index + 1}/{len(v.analysis)}{tag}')
        w.at_md.set_content(app.display(version.final_at) or '（空白）')

    def render_advice(self):
        w, v, app = self.w, self.visit, self.app
        if v is None or not v.advice:
            for md in w.advice_md.values():
                md.set_content('*（按「問診建議」產生）*' if v else '')
            w.ad_label.set_text('0/0')
            return
        version = v.advice[v.advice_index]
        w.ad_label.set_text(f'{v.advice_index + 1}/{len(v.advice)}')
        for key, md in w.advice_md.items():
            md.set_content(app.display(getattr(version, key)))

    def render_patient(self):
        w = self.w
        text = self.patient_source()
        empty = not text.strip()
        w.patient_text.set_text('（尚未匯入患者）' if empty else self.app.display(strip_citations(text)))
        w.patient_text.classes(replace='patient-text' + (' empty' if empty else ''))
        w.btn_patient_copy.set_enabled(not empty)

    def run_html(self, run) -> str:
        """One stretch of one speaker: `醫師：…`. A star marks a label read from the text, not heard (less reliable)."""
        if run.label == BACKGROUND:                     # grey, and not sent to the record: the physician can still read it
            return (f'<span class="who background" title="疑似隔壁房間或環境的人聲，不會送進病歷書寫">{LABEL_NAMES[BACKGROUND]}：</span>'
                    f'<span class="bg-text">{html.escape(self.app.display(run.text))}</span>')
        guessed = run.source in TEXT_SOURCES and run.label != UNKNOWN
        name = LABEL_NAMES[run.label] + ('*' if guessed else '')
        title = ' title="由上下文推測，沒有聲音佐證，可信度較低"' if guessed else ''
        return f'<span class="who {run.label}"{title}>{name}：</span>{html.escape(self.app.display(run.text))}'

    def render_transcript(self):
        w, v, app = self.w, self.visit, self.app
        if v is None or v.pipeline is None:
            w.tx_html.set_content(_placeholder('按「開始看診」後，語音辨識文字會顯示在這裡。'))
            return
        pipeline = v.pipeline
        unlocked = set(pipeline.unlocked_indexes())
        speaker = v.speaker if v.speaker is not None and v.speaker.marks_visible else None
        rows = []
        for seg in pipeline.segments:
            if not seg.visible:
                continue
            cls = 'seg gap' if seg.kind == 'gap' else ('seg unlocked' if seg.index in unlocked else 'seg')
            runs = speaker.runs(seg) if speaker is not None and seg.kind == 'speech' else None
            body = ' '.join(self.run_html(run) for run in runs) if runs else html.escape(app.display(seg.corrected))
            rows.append(f'<div class="{cls}"><span class="seg-no">#{seg.index} {format_clock(seg.new_start)}</span>{body}</div>')
        w.tx_html.set_content(''.join(rows) if rows else _placeholder('（等待語音…）'))
        ui.run_javascript(f"window.miniScrollToEnd('c{w.tx_scroll.id}')")

    # ------------------------------------------------------------------ periodic
    def tick(self):
        app = self.app
        while app.notices:
            level, message = app.notices.pop(0)
            ui.notify(message, type={'warn': 'warning', 'error': 'negative'}.get(level, level))
        self.update_status()
        self.update_controls()
        self.render_panels()

    def update_status(self):
        app, v, w = self.app, self.visit, self.w
        parts = []
        if app.phase == 'none':
            parts.append('尚未匯入患者')
        elif app.phase == 'imported':
            parts.append('患者已匯入，可以開始看診')
        elif v is not None:
            if v.phase == 'finishing':
                pending = v.pipeline.pending if v.pipeline else 0
                parts.append(f'收尾中：等待逐字稿處理完成（待處理 {pending}）並寫入檔案…')
            elif v.phase == 'finish_failed':
                parts.append(f'⚠ 結束並存檔失敗：{v.finish_error}（問題排除後可再按「結束並存檔」重試）')
            else:
                parts.append(f'看診中 · {format_clock(v.audio_t())} · {v.visit_id}')
            if v.pipeline is not None:
                behind = v.pipeline.pending if not v.pipeline.finished else 0
                parts.append(f'逐字稿：{v.pipeline.status}' + (f'（待處理 {behind}）' if behind else ''))
                if behind >= BACKLOG_WARNING_WINDOWS:
                    parts.append(f'⚠ 逐字稿已落後約 {behind * 3} 秒（錄音與音檔不受影響；結束時需等它處理完）')
            if v.speaker is not None:
                parts.append(v.speaker.status)
            elif v.speaker_note:
                parts.append(f'⚠ {v.speaker_note}')
            if v.source is not None and getattr(v.source, 'kind', '') == 'remote' and v.source.status_text:
                parts.append(f'遠端麥克風：{v.source.status_text}')
                if (v.source.status_text == '等待瀏覽器連線' and not v.source.done.is_set()
                        and time.monotonic() - v.source.created_at > WAIT_FOR_BROWSER_SECONDS):
                    parts.append('⚠ 瀏覽器一直沒有連上遠端音訊（尚未錄到任何聲音）：請按「結束並存檔」後改用本機麥克風，'
                                 '或確認網址是 HTTPS、代理沒有改寫 Host 後重新開始')
            if v.source is not None and v.source.error:
                parts.append(f'⚠ {v.source.error.splitlines()[0]}')
        if app.asr_status:
            parts.append(f'ASR：{app.asr_status}')
        sched = app.scheduler
        if sched.active or sched.queued:
            parts.append(f'LLM {sched.active}/{sched.limit}' + (f'（排隊 {sched.queued}）' if sched.queued else ''))
        w.status.set_text(' · '.join(parts))
        job = v.jobs.current or v.jobs.last if v else None
        if job is None:
            w.job.set_text('')
            return
        failed = job.status in ('failed',)
        text = f'{job.label}：{job.stage}' if job.status == 'running' else f'{job.label}：{job.message}'
        w.job.set_text(text)
        w.job.classes(replace='job failed' if failed else 'job')
        w.job.tooltip(job.message if job.status != 'running' else '')

    def update_controls(self):
        app, v, w = self.app, self.visit, self.w
        finishing = bool(v and v.phase == 'finishing') or app._busy_transition
        visiting = app.phase == 'visiting'
        busy = self.busy()
        editing = self.editing()
        recording = bool(v and v.phase == 'recording')
        finish_failed = bool(v and v.phase == 'finish_failed')
        can_act = visiting and recording and not finishing
        # Patient data may not change while an AI job runs (its result would rest on stale data).
        w.btn_import.set_enabled(not finishing and (app.phase in ('none', 'imported') or (can_act and not busy)))
        if visiting and v is not None and v.source is not None:
            # A page opened (or reloaded) during a visit shows the source that visit really uses, not its own default.
            actual = 'remote' if getattr(v.source, 'kind', 'local') == 'remote' else 'local'
            if w.src_toggle.value != actual:
                w.src_toggle.set_value(actual)
        remote = self.remote_mode
        can_start = app.phase == 'imported' and not finishing
        w.btn_start.set_visibility(not remote)
        w.btn_start.set_enabled(can_start)
        w.btn_start_remote.set_visibility(remote)
        w.btn_start_remote.set_enabled(can_start)
        w.src_toggle.set_enabled(not visiting and not finishing)       # the source is fixed for a whole visit
        w.mic_select.set_visibility(remote)
        w.mic_select.set_enabled(not visiting and not finishing)
        w.btn_mic_refresh.set_visibility(remote)
        w.btn_mic_refresh.set_enabled(not visiting and not finishing)
        w.btn_end.set_enabled(visiting and not finishing and not editing and (recording or finish_failed))
        for name in ('btn_write', 'btn_advice', 'btn_analysis'):
            getattr(w, name).set_enabled(can_act and not busy and not editing)
        w.btn_deid.set_enabled(can_act and not editing)
        has_speaker = bool(v and v.speaker is not None)
        w.btn_speakers.set_visibility(has_speaker)
        w.btn_speakers.set_enabled(has_speaker and can_act and v.speaker.tracker.group_count > 0)
        w.btn_cancel.set_visibility(busy)
        w.btn_model.set_enabled(not visiting and not finishing)
        w.btn_template.set_enabled(not visiting and not finishing)
        w.btn_simplified.set_enabled(not editing)
        w.btn_traditional.set_enabled(not editing)
        w.btn_simplified.set_text('✓ 轉簡體' if app.script_mode == 'simplified' else '轉簡體')
        w.btn_traditional.set_text('✓ 轉繁體' if app.script_mode == 'traditional' else '轉繁體')
        # note panel
        note_editing = self.note_mode == 'edit'
        w.btn_browse.set_enabled(visiting and not note_editing)
        w.btn_diff.set_enabled(visiting and not note_editing)
        w.btn_edit.set_enabled(can_act and not busy and not editing)
        w.btn_edit.set_visibility(not note_editing)
        w.btn_edit_done.set_visibility(note_editing)
        w.btn_edit_cancel.set_visibility(note_editing)
        w.btn_undo.set_enabled(can_act and not busy and not editing and v.note.can_undo())
        w.btn_redo.set_enabled(can_act and not busy and not editing and v.note.can_redo())
        w.btn_copy_note.set_enabled(visiting)
        # analysis panel
        count = len(v.analysis) if v else 0
        index = v.analysis_index if v else -1
        w.btn_at_prev.set_enabled(visiting and not self.analysis_editing and index > 0)
        w.btn_at_next.set_enabled(visiting and not self.analysis_editing and 0 <= index < count - 1)
        w.btn_at_edit.set_enabled(can_act and not busy and not editing)
        w.btn_at_edit.set_visibility(not self.analysis_editing)
        w.btn_at_save.set_visibility(self.analysis_editing)
        w.btn_at_cancel.set_visibility(self.analysis_editing)
        w.btn_process.set_enabled(visiting and count > 0 and not self.analysis_editing)
        w.btn_copy_at.set_enabled(visiting and count > 0)
        # advice panel
        a_count = len(v.advice) if v else 0
        a_index = v.advice_index if v else -1
        w.btn_ad_prev.set_enabled(visiting and a_index > 0)
        w.btn_ad_next.set_enabled(visiting and 0 <= a_index < a_count - 1)
        # transcript level meter
        w.level.set_value(v.source.level if v and v.source is not None and v.phase == 'recording' else 0)
        w.tx_info.set_text(f'{len(v.pipeline.segments)} 段' if v and v.pipeline else '')
