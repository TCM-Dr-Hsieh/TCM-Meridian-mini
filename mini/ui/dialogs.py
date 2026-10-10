"""Dialogs: patient import, model settings, template settings, professor process viewer, 去識別化."""
from __future__ import annotations

import asyncio
import json
import os
from copy import deepcopy
from types import SimpleNamespace

from nicegui import ui

from ..config import AGENT_KEYS, AGENT_LABELS, MAX_VOCABULARY_CHARS, Endpoint, Settings, resolve_path
from ..jobs import BusyError
from ..speaker import DOCTOR, OTHER, UNKNOWN
from ..speaker.embedder import availability
from ..state import AppState, StateError, TEMPLATE_LABELS
from ..voice.asr import validate_model_dir


async def copy_to_clipboard(text: str, what: str, *, caution: str = '') -> bool:
    """Copy through the browser (window.miniCopy) and say so; `caution` turns the confirmation into a warning."""
    ok = await ui.run_javascript(f'window.miniCopy({json.dumps(text, ensure_ascii=False)})', timeout=5)
    if not ok:
        ui.notify('複製失敗：瀏覽器未允許剪貼簿。', type='warning')
        return False
    ui.notify(f'已複製{what}。' + caution, type='warning' if caution else 'positive', multi_line=bool(caution))
    return True


# ---------------------------------------------------------------------------
# patient import
# ---------------------------------------------------------------------------
def open_patient_dialog(app: AppState):
    visiting = app.phase == 'visiting'
    stamp = app.patient_stamp                      # what this window is editing; see AppState.patient_stamp
    dialog = ui.dialog().props('persistent')
    with dialog, ui.card().classes('w-[760px] max-w-full gap-3'):
        ui.label('患者匯入').classes('text-xl font-semibold')
        ui.label('貼上患者基本資料；可包含上次就診的病歷（舊病歷的來源標籤只當歷史資料，不會被當成本次逐字稿）。'
                 + ('看診中修改會留下版本與 log。' if visiting else '')).classes('hint')
        area = ui.textarea(value=app.current_patient_text, placeholder='例：王○○，男，45歲。高血壓病史 5 年…') \
            .props('outlined input-style="height:340px"').classes('w-full')
        error = ui.label('').classes('text-red-700 text-sm')

        def stale() -> bool:
            """True (and says so) when the data changed since this window opened: saving or clearing now would overwrite
            a newer version, delete a patient another tab just saved, or act on the wrong patient."""
            if app.patient_stamp == stamp:
                return False
            error.set_text('患者資料在這個視窗開啟後已被改變（可能在另一個分頁修改，或看診已經開始／結束）。'
                           '請關閉後重新開啟，再操作。')
            return True

        def ok():
            if stale():
                return
            try:
                app.import_patient(area.value or '')
            except StateError as exc:
                error.set_text(str(exc))
                return
            dialog.close()
            ui.notify('已儲存患者資料。' if visiting else '患者資料已匯入，可以開始看診。', type='positive')

        def clear():
            if stale():
                return
            try:
                app.clear_patient()
            except StateError as exc:
                error.set_text(str(exc))
                return
            dialog.close()
            ui.notify('已清除患者。')

        with ui.row().classes('w-full justify-end'):
            ui.button('取消', on_click=dialog.close).props('flat')
            if app.phase == 'imported':
                ui.button('清除患者', on_click=clear).props('flat color=negative')
            ui.button('儲存修改' if visiting else '確定', on_click=ok)
    dialog.open()


# ---------------------------------------------------------------------------
# templates
# ---------------------------------------------------------------------------
def open_template_dialog(app: AppState):
    if app.phase == 'visiting':
        ui.notify('看診中不可變更模板。', type='warning')
        return
    areas: dict[str, ui.textarea] = {}
    dialog = ui.dialog().props('persistent')
    with dialog, ui.card().classes('w-[860px] max-w-full gap-3'):
        ui.label('模板設定').classes('text-xl font-semibold')
        ui.label('病歷模板決定寫病歷 agent 的輸出格式；分析模板決定整體分析的 A&T 格式。留空代表自由發揮。').classes('hint')
        with ui.tabs().classes('w-full') as tabs:
            tab_objs = {kind: ui.tab(label) for kind, label in TEMPLATE_LABELS.items()}
        with ui.tab_panels(tabs, value=tab_objs['record']).classes('w-full'):
            for kind in TEMPLATE_LABELS:
                with ui.tab_panel(tab_objs[kind]).classes('p-0 gap-2'):
                    areas[kind] = ui.textarea(value=app.get_template(kind)) \
                        .props('outlined input-style="height:420px; font-family:Consolas,monospace"').classes('w-full')

                    def restore(k=kind):
                        areas[k].set_value(app.default_template(k))
                        ui.notify('已載入預設內容，按「儲存」才會生效。')

                    ui.button('還原預設', on_click=restore).props('flat')
        error = ui.label('').classes('text-red-700 text-sm')

        def save():
            try:
                for kind, area in areas.items():
                    app.save_template(kind, area.value or '')
            except (StateError, OSError) as exc:
                error.set_text(str(exc))
                return
            dialog.close()
            ui.notify('模板已儲存。', type='positive')

        with ui.row().classes('w-full justify-end'):
            ui.button('取消', on_click=dialog.close).props('flat')
            ui.button('儲存', on_click=save)
    dialog.open()


# ---------------------------------------------------------------------------
# professor process viewer
# ---------------------------------------------------------------------------
def open_process_dialog(app: AppState):
    visit = app.visit
    if visit is None or not (0 <= visit.analysis_index < len(visit.analysis)):
        ui.notify('目前沒有可檢視的整體分析。', type='warning')
        return
    version = visit.analysis[visit.analysis_index]
    dialog = ui.dialog()
    with dialog, ui.card().classes('w-[900px] max-w-full h-[80vh] gap-2'):
        ui.label(f'教授過程 · 整體分析第 {version.index} 則').classes('text-xl font-semibold')
        if version.source != 'llm':
            ui.label('此版本為醫師手動修改' + (f'，依據第 {version.parent} 則。' if version.parent else '。')) \
                .classes('hint')
            if version.parent:
                def jump():
                    visit.set_analysis_index(version.parent - 1)
                    dialog.close()
                    open_process_dialog(app)
                ui.button(f'檢視第 {version.parent} 則的教授過程', on_click=jump).props('outline')
        else:
            n = version.names or {}
            pieces = [('仲裁說明', version.arbitration_notes),
                      (n.get('a', '教授甲'), version.professors.get('a', '')),
                      (n.get('b', '教授乙'), version.professors.get('b', '')),
                      (f'{n.get("a", "教授甲")}評{n.get("b", "教授乙")}', version.professors.get('review_a_on_b', '')),
                      (f'{n.get("b", "教授乙")}評{n.get("a", "教授甲")}', version.professors.get('review_b_on_a', ''))]
            with ui.tabs().classes('w-full') as tabs:
                tab_list = [ui.tab(title) for title, _ in pieces]
            with ui.tab_panels(tabs, value=tab_list[0]).classes('w-full grow overflow-auto'):
                for tab, (_, text) in zip(tab_list, pieces):
                    with ui.tab_panel(tab):
                        ui.markdown(app.display(text) or '（無內容）', extras=['fenced-code-blocks', 'tables'])
        with ui.row().classes('w-full justify-end'):
            ui.button('關閉', on_click=dialog.close)
    dialog.open()


# ---------------------------------------------------------------------------
# LLM de-identification
# ---------------------------------------------------------------------------
DEID_NOTICE = ('AI 去識別化不保證完整，對外使用前請人工逐行檢查，並另行確認可對外提供的依據，以及對方服務的資料留存與'
               '訓練政策。送進去的是可識別的原始資料，所以「模型設定」中的「去識別化」接口必須指向可信任的院內模型。')
DEID_EXTRA_HINT = ('額外指示（選填）：每次按「去識別化」都會帶上，直到你手動清除，看診結束後清空。'
                   '例如「連職業也刪掉」「年齡改成年齡區間」。只能讓處理更嚴格，不能要求保留姓名、證號、聯絡方式或確切日期。')


def open_deid_dialog(app: AppState):
    """Browse, run again and copy the de-identified versions of the visit (today's date + patient data + today's record)."""
    visit = app.visit
    if visit is None or visit.phase != 'recording':
        ui.notify('看診中才能使用 LLM 去識別化。', type='warning')
        return
    w = SimpleNamespace()
    seen: list = [None]
    dialog = ui.dialog()
    with dialog, ui.card().classes('w-[980px] max-w-full gap-2').style('max-height:94vh; overflow:auto'):
        with ui.row().classes('w-full items-center gap-2 no-wrap'):
            ui.label('LLM 去識別化').classes('text-xl font-semibold')
            w.prev = ui.button('上一則', icon='chevron_left', on_click=lambda: step(-1)) \
                .props('dense no-caps outline').mark('deid-prev')
            w.label = ui.label('').classes('version-label').mark('deid-label')
            w.next = ui.button('下一則', on_click=lambda: step(1)) \
                .props('dense no-caps outline icon-right=chevron_right').mark('deid-next')
            ui.element('div').classes('grow')
            w.copy = ui.button('複製', icon='content_copy', on_click=lambda: copy()).props('dense no-caps').mark('deid-copy')
        ui.label(DEID_NOTICE).classes('hint')
        w.warn = ui.element('div').classes('w-full text-sm text-amber-900 bg-amber-50 rounded p-2 gap-1') \
            .mark('deid-warn')
        w.text = ui.textarea(placeholder='（尚無結果；按「去識別化」產生）') \
            .props('outlined readonly input-style="height:38vh; font-family:Consolas,monospace"') \
            .classes('w-full').mark('deid-text')
        with ui.expansion('處理說明（模型自述，不會被複製）', icon='info').classes('w-full') as w.summary_box:
            w.summary = ui.markdown('')
        ui.label(DEID_EXTRA_HINT).classes('hint')
        w.extra = ui.textarea(label='額外指示', value=visit.deid_extra,
                              on_change=lambda e: setattr(visit, 'deid_extra', e.value or '')) \
            .props('outlined autogrow input-style="min-height:48px"') \
            .classes('w-full').mark('deid-extra')
        w.used = ui.label('').classes('hint')
        with ui.row().classes('w-full items-center justify-end gap-2'):
            w.status = ui.label('').classes('grow text-sm').mark('deid-status')
            w.cancel = ui.button('取消作業', icon='close', on_click=lambda: visit.jobs.cancel()) \
                .props('flat color=negative no-caps')
            w.run = ui.button('去識別化', icon='shield', on_click=lambda: run()).props('no-caps').mark('deid-run')
            ui.button('關閉', on_click=dialog.close).props('flat no-caps')

    def current():
        return visit.deid[visit.deid_index] if 0 <= visit.deid_index < len(visit.deid) else None

    def step(delta: int):
        visit.set_deid_index(visit.deid_index + delta)
        refresh()

    async def copy():
        version = current()
        if version is None:
            return
        caution = f'（注意：這則有 {len(version.warnings)} 項殘留提醒，請先檢查）' if version.warnings else ''
        await copy_to_clipboard(app.display(version.text), f'去識別化病歷第 {version.index} 則', caution=caution)

    def run():
        try:
            visit.start_job('deidentify', extra=w.extra.value or '')
        except (BusyError, StateError) as exc:
            ui.notify(str(exc), type='warning')
            return
        refresh()

    def refresh():
        if not dialog.value or app.visit is not visit:          # closed, or the visit ended meanwhile: stop polling
            timer.cancel()
            if dialog.value:
                dialog.close()
            return
        version, job, last, busy = current(), visit.jobs.current, visit.jobs.last, visit.jobs.busy
        failure = ''
        if not busy and last is not None and last.kind == 'deidentify' and last.status != 'succeeded':
            failure = f'去識別化{"失敗" if last.status == "failed" else "已取消"}：{last.message}'
        status = f'{job.label}：{job.stage}' if busy and job is not None else failure
        signature = (len(visit.deid), visit.deid_index, busy, status, app.script_mode, visit.phase)
        if seen[0] == signature:
            return
        seen[0] = signature
        count = len(visit.deid)
        w.label.set_text(f'{visit.deid_index + 1}/{count} · {version.timestamp[11:19]}' if version else '0/0')
        w.text.set_value(app.display(version.text) if version else '')
        w.warn.clear()
        with w.warn:
            for warning in version.warnings if version else []:
                ui.label('⚠ ' + warning)
        w.warn.set_visibility(bool(version and version.warnings))
        w.summary.set_content(app.display(version.summary) if version and version.summary else '（模型沒有提供處理說明）')
        w.summary_box.set_visibility(version is not None)
        w.used.set_text(f'這一則的額外指示：{version.extra_instruction}' if version and version.extra_instruction else '')
        w.used.set_visibility(bool(version and version.extra_instruction))
        w.status.set_text(status)
        w.status.classes(replace='grow text-sm ' + ('text-red-700' if failure else ''))
        w.prev.set_enabled(version is not None and visit.deid_index > 0)
        w.next.set_enabled(version is not None and visit.deid_index < count - 1)
        w.copy.set_enabled(version is not None)
        w.run.set_enabled(visit.phase == 'recording' and not busy)
        w.cancel.set_visibility(busy)

    timer = ui.timer(0.3, refresh, immediate=False)
    dialog.open()
    refresh()


# ---------------------------------------------------------------------------
# model settings
# ---------------------------------------------------------------------------
def open_settings_dialog(app: AppState):
    if app.phase == 'visiting':
        ui.notify('看診中不可變更設定。', type='warning')
        return
    draft: Settings = deepcopy(app.settings)
    dialog = ui.dialog().props('persistent')
    with dialog, ui.card().classes('w-[1100px] max-w-full gap-3'):
        ui.label('模型與設定').classes('text-xl font-semibold')
        with ui.tabs().classes('w-full') as tabs:
            t_asr, t_speaker, t_llm, t_review, t_prof, t_general = (ui.tab(x) for x in
                                                                    ('語音辨識', '說話者標記', 'LLM 接口', '審查', '教授', '一般'))
        with ui.tab_panels(tabs, value=t_asr).classes('w-full'):
            # ---- ASR -------------------------------------------------------
            with ui.tab_panel(t_asr).classes('gap-3 p-0'):
                asr_dir = ui.input('ASR 模型資料夾', value=draft.asr.asr_model_dir).classes('w-full')
                aligner_dir = ui.input('ForcedAligner 模型資料夾', value=draft.asr.aligner_model_dir).classes('w-full')
                python_path = ui.input('ASR Python 路徑（留空＝專案內 .venv-asr）', value=draft.asr.python_path) \
                    .classes('w-full')
                with ui.row().classes('w-full gap-3'):
                    device = ui.select({'auto': '自動（優先 CUDA）', 'cuda': 'NVIDIA GPU（CUDA）', 'cpu': 'CPU'},
                                       value=draft.asr.device, label='運算裝置').classes('w-56')
                    mic = ui.select({'': '系統預設麥克風'}, value=draft.asr.microphone_id, label='麥克風') \
                        .classes('grow min-w-64')
                with ui.row().classes('w-full gap-3'):
                    window = ui.number('音訊視窗長度 L（秒）', value=draft.asr.window_seconds, min=5, max=30, step=1) \
                        .classes('w-56')
                    overlap = ui.number('重疊 Z（秒）', value=draft.asr.overlap_seconds, min=0, max=27, step=0.5) \
                        .classes('w-56')
                    context = ui.number('校稿前文長度（字）', value=draft.asr.context_chars, min=100, max=20000,
                                        step=100).classes('w-56')
                ui.label('每隔 L−Z 秒產生一段辨識；Z ≤ L−3。').classes('hint')
                vocabulary = ui.textarea(f'ASR 專有詞（只給語音辨識當提示；最多 {MAX_VOCABULARY_CHARS:,} 字元）',
                                         value=draft.asr.vocabulary).classes('w-full')
                ui.label('校稿 LLM 專有詞只給「逐字稿校稿」的 LLM 當拼寫參考，不會送給語音辨識；校稿的每次 LLM 呼叫都會帶上它，'
                         '詞表越長越慢，建議只放真正容易被辨識錯的詞。').classes('hint')
                correction_vocabulary = ui.textarea(
                    f'校稿 LLM 專有詞（只給逐字稿校稿的 LLM；最多 {MAX_VOCABULARY_CHARS:,} 字元）',
                    value=draft.asr.correction_vocabulary).classes('w-full')
                asr_result = ui.label('').classes('text-sm')
                if os.environ.get('MINI_FAKE_AUDIO'):
                    ui.label(f'⚠ 開發模式：音訊來源為檔案 {os.environ["MINI_FAKE_AUDIO"]}（MINI_FAKE_AUDIO）').classes(
                        'text-amber-800 text-sm')

                async def load_mics():
                    try:
                        from ..voice.audio import list_microphones
                        options = await asyncio.to_thread(list_microphones)
                        mic.set_options({'': '系統預設麥克風', **options},
                                        value=draft.asr.microphone_id if draft.asr.microphone_id in options else '')
                    except Exception as exc:
                        asr_result.set_text(f'無法列出麥克風：{exc}')

                async def test_load():
                    try:
                        trial = deepcopy(draft.asr)
                        trial.asr_model_dir, trial.aligner_model_dir = asr_dir.value or '', aligner_dir.value or ''
                        trial.device, trial.python_path = device.value, python_path.value or ''
                        validate_model_dir(trial.asr_model_dir)
                        validate_model_dir(trial.aligner_model_dir, aligner=True)
                        asr_result.set_text('正在載入本機模型…首次載入可能需要數十秒。')
                        info = await app.asr.load(trial)
                        asr_result.set_text(f'ASR 與對齊模型載入成功 · {info.get("device")}')
                    except Exception as exc:
                        asr_result.set_text(f'載入失敗：{exc}')

                ui.button('測試載入 ASR', on_click=test_load).props('outline')
                ui.timer(0.2, load_mics, once=True)

            # ---- speaker marking -------------------------------------------------
            with ui.tab_panel(t_speaker).classes('gap-3 p-0'):
                ui.label('把逐字稿標成「醫師：／患者或家屬：／不明：」。程式用聲音分群，再請「逐字稿校稿」接口的 LLM 判斷哪一群是醫師；'
                         '標記少部分可能有誤，聲音太相近或太短的句子會標「不明」，不會硬猜。看診中不能更改這些設定。').classes('hint')
                speaker_on = ui.switch('啟用說話者標記', value=draft.speaker.enabled)
                speaker_model = ui.input('聲紋模型檔（相對路徑以專案目錄為基準）', value=draft.speaker.model_path).classes('w-full')
                speaker_jobs = ui.switch('把說話者標記（醫師／患者或家屬）一起給「病歷書寫」與「幻覺修正（審查）」的 prompt（判為背景人聲的句子不論此項都不會給）',
                                         value=draft.speaker.use_in_jobs)
                speaker_fill = ui.switch('用 LLM 依上下文補標「不明」的句子（只在與聲音傾向一致時採用，標示 *）',
                                         value=draft.speaker.text_fill)
                speaker_fill2 = ui.switch('二階補標：對一階補標後仍不明的句子再問 LLM 一次（不告訴它第一次的答案，但讓它看到鄰句的 * 標記），'
                                          '它答出醫師或患者或家屬就採用，標示 *（需先開啟上一項）',
                                          value=draft.speaker.text_fill2).mark('speaker-fill2')
                speaker_fill2.bind_enabled_from(speaker_fill, 'value')
                speaker_pct = ui.number('「不明」百分位（0–50，預設 15；越大越多「不明」、越少標錯）',
                                        value=draft.speaker.unknown_percentile, min=0, max=50, step=1).classes('w-96')
                speaker_result = ui.label('').classes('text-sm')

                def check_speaker():
                    reason = availability(resolve_path(speaker_model.value or ''))
                    speaker_result.set_text(reason or '聲紋模型與所需套件都可用。')

                ui.button('檢查聲紋模型', on_click=check_speaker).props('outline')
                ui.label('聲紋只存在記憶體，看診結束就消失；存檔只有角色對照表、各分鐘的統計與你的手動操作（不含聲紋）。').classes('hint')

            # ---- LLM ---------------------------------------------------------
            rows: dict[str, tuple] = {}
            with ui.tab_panel(t_llm).classes('gap-2 p-0'):
                ui.label('八個接口各自獨立；「全部套用同一組」只複製第一列的網址、金鑰與模型名稱。'
                         '「context 上限」是模型一次能處理的 tokens（輸入＋輸出），伺服器通常不會回報，請填實際值；'
                         '提示詞太長時會事先明確失敗，0＝不檢查。').classes('hint')
                with ui.grid(columns='120px 2fr 1.2fr 1.6fr 90px 100px 110px 70px').classes('w-full items-center gap-x-2 gap-y-1'):
                    for head in ('接口', 'API URL', 'API Key', '模型名稱', 'temperature', 'max_tokens', 'context 上限', ''):
                        ui.label(head).classes('text-xs font-semibold')
                    for key in AGENT_KEYS:
                        endpoint = draft.agents[key]
                        ui.label(AGENT_LABELS[key]).classes('text-sm')
                        url = ui.input(value=endpoint.api_url).props('dense outlined')
                        secret = ui.input(value=endpoint.api_key, password=True, password_toggle_button=True) \
                            .props('dense outlined')
                        model = ui.input(value=endpoint.model_name).props('dense outlined')
                        temp = ui.number(value=endpoint.temperature, min=0, max=2, step=0.1).props('dense outlined')
                        tokens = ui.number(value=endpoint.max_tokens, min=0, step=500).props('dense outlined')
                        context_limit = ui.number(value=endpoint.context_tokens, min=0, step=1024).props('dense outlined')
                        rows[key] = (url, secret, model, temp, tokens, context_limit)

                        async def test(k=key):
                            u, s, m, t, x, _c = rows[k]
                            try:
                                ep = Endpoint(u.value or '', s.value or '', m.value or '', t.value or 0, int(x.value or 0))
                                ep.validate(AGENT_LABELS[k])
                                names = await app.client.models(ep)
                                known = '，且找到此模型' if ep.model_name in names else f'，但清單中沒有「{ep.model_name}」'
                                llm_result.set_text(f'{AGENT_LABELS[k]}：連線成功，取得 {len(names)} 個模型{known}。')
                            except Exception as exc:
                                llm_result.set_text(f'{AGENT_LABELS[k]}：測試失敗：{exc}')

                        ui.button('測試', on_click=test).props('dense flat')
                llm_result = ui.label('').classes('text-sm')

                def apply_all():
                    first_url, first_key, first_model = (rows[AGENT_KEYS[0]][i].value for i in range(3))
                    for key in AGENT_KEYS[1:]:
                        rows[key][0].set_value(first_url)
                        rows[key][1].set_value(first_key)
                        rows[key][2].set_value(first_model)

                ui.button('全部套用同一組', on_click=apply_all).props('outline')
                with ui.row().classes('w-full gap-3'):
                    concurrency = ui.number('LLM 並行上限', value=draft.llm.max_concurrency, min=1, max=32, step=1) \
                        .classes('w-44')
                    timeout = ui.number('單次呼叫逾時（秒）', value=draft.llm.timeout_seconds, min=5, max=1800) \
                        .classes('w-44')
                    retries = ui.number('失敗重試次數', value=draft.llm.retries, min=0, max=10, step=1).classes('w-44')

            # ---- review --------------------------------------------------------
            with ui.tab_panel(t_review).classes('gap-3 p-0'):
                n_field = ui.number('審查通過次數 n（累積制）', value=draft.review.pass_required_n, min=0, max=50, step=1) \
                    .classes('w-72')
                rounds_field = ui.number('最大審查輪數', value=draft.review.max_review_rounds, min=1, max=50, step=1) \
                    .classes('w-72')
                ui.label('累積達 n 次通過才寫入病歷；超過最大輪數仍未通過則不寫入（fail-closed）。'
                         'n = 0 代表不審查（對照組模式）：寫病歷結果在行級操作合法的前提下直接寫入（不檢查內容與來源標籤），並明確標示「未審查」。').classes('hint')

            # ---- professors ----------------------------------------------------
            with ui.tab_panel(t_prof).classes('gap-3 p-0'):
                ui.label('教授甲、乙可設定名稱與風格（會注入 prompt 的 {name}、{role_style}）；教授丙為中立仲裁，只有名稱。') \
                    .classes('hint')
                prof_name, prof_style = {}, {}
                for key, title in (('a', '教授甲'), ('b', '教授乙')):
                    prof_name[key] = ui.input(f'{title} 名稱', value=draft.professors[key].name).classes('w-72')
                    prof_style[key] = ui.textarea(f'{title} 風格', value=draft.professors[key].role_style).classes('w-full')
                prof_name['c'] = ui.input('教授丙 名稱', value=draft.professors['c'].name).classes('w-72')

            # ---- general ---------------------------------------------------------
            with ui.tab_panel(t_general).classes('gap-3 p-0'):
                visits_dir = ui.input('看診資料夾（相對路徑以專案目錄為基準）', value=draft.visits_dir).classes('w-full')
                ui.label('設定保存於專案目錄的 config.json（含 API Key，已加入 .gitignore）。看診中此視窗鎖定。').classes('hint')

        error = ui.label('').classes('text-red-700 text-sm')

        def save():
            try:
                new = deepcopy(app.settings)
                new.asr.asr_model_dir, new.asr.aligner_model_dir = asr_dir.value or '', aligner_dir.value or ''
                new.asr.python_path, new.asr.device = python_path.value or '', device.value
                new.asr.microphone_id = mic.value or ''
                new.asr.window_seconds, new.asr.overlap_seconds = window.value, overlap.value
                new.asr.context_chars, new.asr.vocabulary = context.value, vocabulary.value or ''
                new.asr.correction_vocabulary = correction_vocabulary.value or ''
                new.speaker.enabled, new.speaker.model_path = bool(speaker_on.value), speaker_model.value or ''
                new.speaker.use_in_jobs, new.speaker.text_fill = bool(speaker_jobs.value), bool(speaker_fill.value)
                new.speaker.text_fill2 = bool(speaker_fill2.value) and bool(speaker_fill.value)
                new.speaker.unknown_percentile = speaker_pct.value if speaker_pct.value is not None else 15.0
                for key in AGENT_KEYS:
                    u, s, m, t, x, c = rows[key]
                    new.agents[key] = Endpoint(u.value or '', s.value or '', m.value or '', t.value, x.value, c.value)
                new.llm.max_concurrency, new.llm.timeout_seconds, new.llm.retries = (
                    concurrency.value, timeout.value, retries.value)
                new.review.pass_required_n, new.review.max_review_rounds = n_field.value, rounds_field.value
                for key in ('a', 'b', 'c'):
                    new.professors[key].name = prof_name[key].value or ''
                for key in ('a', 'b'):
                    new.professors[key].role_style = prof_style[key].value or ''
                new.visits_dir = visits_dir.value or ''
                app.update_settings(new)
            except (ValueError, TypeError, StateError, OSError) as exc:
                error.set_text(str(exc))
                return
            dialog.close()
            ui.notify('設定已儲存。', type='positive')

        with ui.row().classes('w-full justify-end'):
            ui.button('取消', on_click=dialog.close).props('flat')
            ui.button('儲存設定', on_click=save)
    dialog.open()


# ---------------------------------------------------------------------------
# speaker groups
# ---------------------------------------------------------------------------
ROLE_OPTIONS = {DOCTOR: '醫師', OTHER: '患者或家屬'}              # a group nobody named shows no value; it cannot be set back to that


def open_speaker_dialog(app: AppState):
    """The voice groups heard so far: who each one is (the physician can override the model), how much it spoke, two examples.

    The window is refreshed while it is open, but a group's row is only built once and then updated in place: rebuilding it
    would close a dropdown the physician is choosing from."""
    visit = app.visit
    service = visit.speaker if visit is not None else None
    if service is None or visit.phase != 'recording':
        ui.notify('看診中且啟用說話者標記時才能使用。', type='warning')
        return
    w = SimpleNamespace()
    rows: dict[int, SimpleNamespace] = {}
    shape: list = [None]
    programmatic = [False]                         # set while the window itself changes a widget (no click to answer)
    dialog = ui.dialog()
    with dialog, ui.card().classes('w-[760px] max-w-full gap-2').style('max-height:90vh; overflow:auto'):
        ui.label('說話者群').classes('text-xl font-semibold')
        ui.label('程式依聲音把說話的人分成幾群（S1、S2…），再請 LLM 判斷哪一群是醫師。判斷錯了可以在這裡改；'
                 '你改過之後以你的為準，逐字稿已標好的句子會跟著更新。患者與陪診家屬不必分開，都選「患者或家屬」。').classes('hint')
        w.status = ui.label('').classes('text-sm').mark('speaker-status')
        w.busy = ui.label('作業進行中，暫時不能修改對照（作業用的是它開始時的逐字稿）；作業結束後再改。') \
            .classes('text-sm text-amber-900').mark('speaker-busy')
        w.body = ui.element('div').classes('w-full gap-2')
        w.lock = ui.switch('鎖定目前的對照（不再請 LLM 重新判斷）', value=service.locked,
                           on_change=lambda e: toggle_lock(bool(e.value))).mark('speaker-lock')
        with ui.row().classes('w-full justify-end'):
            ui.button('關閉', on_click=dialog.close)

    def put_back(gid: int):
        """A change was refused: show what is really in force again."""
        programmatic[0] = True
        try:
            row = rows.get(gid)
            if row is not None:
                row.select.set_value(service.tracker.role_of(gid) if service.tracker.role_of(gid) != UNKNOWN else None)
        finally:
            programmatic[0] = False

    def choose(gid: int, role: str):
        if programmatic[0] or role not in (DOCTOR, OTHER):
            return
        try:
            visit.set_speaker_roles({gid: role})                # refused while a job runs
        except BusyError as exc:
            ui.notify(str(exc), type='warning')
            put_back(gid)

    def toggle_lock(locked: bool):
        if programmatic[0] or locked == service.locked:
            return
        try:
            visit.set_speaker_lock(locked)
        except BusyError as exc:
            ui.notify(str(exc), type='warning')
            programmatic[0] = True
            try:
                w.lock.set_value(service.locked)
            finally:
                programmatic[0] = False

    def build(groups):
        rows.clear()
        w.body.clear()
        with w.body:
            if not groups:
                ui.label('聲音群還沒有建立（看診開始後約 1 分鐘，且兩個人都要有說話）。').classes('hint')
            for g in groups:
                row = SimpleNamespace(examples=None, shown=None)
                with ui.element('div').classes('w-full border rounded p-2 gap-1').mark(f'speaker-group-{g["gid"]}'):
                    with ui.row().classes('w-full items-center gap-3 no-wrap'):
                        ui.label(g['name']).classes('font-semibold')
                        row.select = ui.select(ROLE_OPTIONS, value=g['role'] or None, label='這一群是',
                                               on_change=lambda e, gid=g['gid']: choose(gid, e.value)) \
                            .props('dense outlined').classes('w-44').mark(f'speaker-role-{g["gid"]}')
                        row.stats = ui.label('').classes('text-sm')
                    row.examples = ui.element('div').classes('w-full')
                rows[g['gid']] = row

    def refresh():
        if not dialog.value or app.visit is not visit or visit.phase != 'recording':
            timer.cancel()
            if dialog.value:
                dialog.close()
            return
        groups = service.groups()
        ids = tuple(g['gid'] for g in groups)
        if shape[0] != ids:
            build(groups)
            shape[0] = ids
        w.status.set_text(service.status)
        busy = visit.jobs.busy
        w.busy.set_visibility(busy)
        w.lock.set_enabled(not busy)
        programmatic[0] = True
        try:
            if w.lock.value != service.locked:
                w.lock.set_value(service.locked)
            for g in groups:
                row = rows[g['gid']]
                row.select.set_enabled(not busy)
                row.stats.set_text(f'說話約 {g["seconds"]:.0f} 秒（佔 {g["share"] * 100:.0f}%），{g["units"]} 個片段')
                if row.select.value != (g['role'] or None):
                    row.select.set_value(g['role'] or None)
                shown = (tuple(g['examples']), app.script_mode)
                if row.shown != shown:
                    row.shown = shown
                    row.examples.clear()
                    with row.examples:
                        for example in g['examples']:
                            ui.label('例：' + app.display(example)).classes('text-sm hint')
        finally:
            programmatic[0] = False

    timer = ui.timer(0.5, refresh, immediate=False)
    dialog.open()
    refresh()
