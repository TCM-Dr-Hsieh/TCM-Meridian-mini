"""UI wiring tests using NiceGUI's in-process user simulation (no browser needed)."""
import asyncio
import json
from types import SimpleNamespace

import pytest
from nicegui import ui
from nicegui.testing import User

from mini.config import Settings
from mini.llm import LLMClient
from mini.state import AppState, StateError
from mini.ui.page import MainPage
from mini.voice.remote import RemoteAudioSource
from tests.helpers import FakeASR, FakeLLM, deid_reply, silent_source


def build_state(tmp_path, fake: FakeLLM) -> AppState:
    settings = Settings()
    settings.visits_dir = str(tmp_path / 'visits')
    (tmp_path / 'config.json').write_text(json.dumps(settings.to_dict(), ensure_ascii=False), encoding='utf-8')
    templates = tmp_path / 'templates'
    (templates / 'defaults').mkdir(parents=True)
    (templates / 'defaults' / 'record_template.txt').write_text('甲- 現病史：\n乙- 過去病史：\n', encoding='utf-8')
    (templates / 'defaults' / 'analysis_template.txt').write_text('一- 西醫診斷：\n', encoding='utf-8')
    state: AppState = AppState(config_path=tmp_path / 'config.json', templates_dir=templates,
                               client=LLMClient(fake.transport()), asr=FakeASR(default='我頭痛三天了'),
                               llm_backoff=0, source_factory=lambda path: silent_source(state.settings, 7, recording_path=path))
    return state


def enabled(user: User, label: str) -> bool:
    """Enabled state of the toolbar button with this label (deterministic: buttons only)."""
    buttons = [e for e in user.find(label).elements if isinstance(e, ui.button)]
    assert len(buttons) == 1, f'{label}: expected exactly one button, found {len(buttons)}'
    return buttons[0].enabled


async def until(condition, timeout=15.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while not condition():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError('condition not reached in time')
        await asyncio.sleep(0.05)


async def test_full_consultation_through_the_ui(user: User, tmp_path):
    fake = FakeLLM()
    state = build_state(tmp_path, fake)

    @ui.page('/')
    def index():
        MainPage(state)

    await user.open('/')
    await user.should_see('尚未匯入患者', retries=40)
    await user.should_see('語音文字辨識', retries=40)

    # patient import
    user.find('患者匯入').click()
    await user.should_see('貼上患者基本資料', retries=40)
    user.find(kind=ui.textarea).type('王先生，45歲，高血壓')
    user.find('確定').click()
    await until(lambda: state.phase == 'imported')
    await user.should_see('患者已匯入', retries=40)

    # start the visit: transcript appears
    user.find('開始看診').click()
    await until(lambda: state.phase == 'visiting')
    await user.should_see('看診中', retries=40)
    await until(lambda: state.visit.pipeline.finished)
    await user.should_see('我頭痛三天了', retries=40)

    # 病歷書寫
    user.find('病歷書寫').click()
    await until(lambda: state.visit.note.current_index == 1)
    await user.should_see('甲- 現病史：患者頭痛三天', retries=40)
    assert state.visit.note.get_current()['source'] == '病歷書寫 #1'

    # 問診建議 (three panels)
    await until(lambda: not state.visit.jobs.busy)
    user.find('問診建議').click()
    await until(lambda: len(state.visit.advice) == 1)
    await user.should_see('偏頭痛', retries=40)
    await user.should_see('肝陽上亢', retries=40)
    await user.should_see('疼痛位置', retries=40)

    # 整體分析
    await until(lambda: not state.visit.jobs.busy)
    user.find('整體分析').click()
    await until(lambda: len(state.visit.analysis) == 1)
    await user.should_see('西醫診斷', retries=40)

    # simplified display is display-only
    user.find('轉簡體').click()
    await user.should_see('头痛', retries=40)
    assert '頭痛' in state.visit.note.current_note() and state.script_mode == 'simplified'
    user.find('轉簡體').click()
    assert state.script_mode == 'original'

    # end and save
    await until(lambda: not state.visit.jobs.busy)
    folder = state.visit.store.folder
    user.find('結束並存檔').click()
    await until(lambda: state.phase == 'none', timeout=20)
    assert (folder / 'log.md').exists() and (folder / '今日病歷.md').exists() and (folder / 'audio.wav').exists()
    await user.should_see('尚未匯入患者', retries=40)


class RecordedJavascript:
    """Stands in for ui.run_javascript (no browser here): records the code, and can be awaited for a canned answer."""

    def __init__(self, answers=None):
        self.calls: list[str] = []
        self.answers = answers or {}

    def __call__(self, code, **kwargs):
        self.calls.append(code)
        answer = next((value for key, value in self.answers.items() if key in code), None)

        class Answer:
            def __await__(self_inner):
                if False:
                    yield
                return answer
        return Answer()


async def test_remote_microphone_controls_follow_the_selected_source(user: User, tmp_path, monkeypatch):
    js = RecordedJavascript({'isSecureContext': True})
    monkeypatch.setattr(ui, 'run_javascript', js)
    state = build_state(tmp_path, FakeLLM())
    holder = {}

    @ui.page('/')
    def index():
        holder['page'] = MainPage(state)

    state.import_patient('x')
    await user.open('/')
    w = holder['page'].w
    await until(lambda: w.btn_start.visible and w.btn_start.enabled)
    assert not w.btn_start_remote.visible and not w.mic_select.visible and not w.btn_mic_refresh.visible

    w.src_toggle.set_value('remote')                      # this browser's microphone
    await until(lambda: w.btn_start_remote.visible)
    assert not w.btn_start.visible and w.mic_select.visible and w.btn_mic_refresh.visible
    assert w.btn_start_remote.enabled and w.src_toggle.enabled
    assert any('isSecureContext' in code for code in js.calls)            # the page checked the browser can use a mic

    w.mic_select.set_value('')                            # choosing a device is passed on to the browser script
    w.src_toggle.set_value('local')
    await until(lambda: w.btn_start.visible and not w.btn_start_remote.visible)


async def test_a_page_that_is_not_https_warns_before_the_remote_microphone_is_used(user: User, tmp_path, monkeypatch):
    monkeypatch.setattr(ui, 'run_javascript', RecordedJavascript({'isSecureContext': False}))
    state = build_state(tmp_path, FakeLLM())
    holder = {}

    @ui.page('/')
    def index():
        holder['page'] = MainPage(state)

    await user.open('/')
    holder['page'].w.src_toggle.set_value('remote')
    await user.should_see('不是 HTTPS', retries=40)


async def test_the_source_cannot_be_changed_during_a_visit(user: User, tmp_path, monkeypatch):
    monkeypatch.setattr(ui, 'run_javascript', RecordedJavascript({'isSecureContext': True}))
    state = build_state(tmp_path, FakeLLM())
    holder = {}

    @ui.page('/')
    def index():
        holder['page'] = MainPage(state)

    state.import_patient('x')
    await state.start_visit()
    await user.open('/')
    w = holder['page'].w
    await until(lambda: not w.src_toggle.enabled)
    await state.finish_visit()
    await until(lambda: w.src_toggle.enabled and not state.visit)


async def test_the_browser_reporting_an_open_microphone_starts_a_remote_visit(user: User, tmp_path, monkeypatch):
    js = RecordedJavascript({'isSecureContext': True})
    monkeypatch.setattr(ui, 'run_javascript', js)
    state = build_state(tmp_path, FakeLLM())
    holder = {}

    @ui.page('/')
    def index():
        holder['page'] = MainPage(state)

    state.import_patient('x')
    await user.open('/')
    page = holder['page']
    page.w.src_toggle.set_value('remote')
    await until(lambda: page.remote_mode)
    with user.client:
        await page.on_remote_event(SimpleNamespace(args={'type': 'prepared', 'sample_rate': 48000}))
    assert state.phase == 'visiting' and isinstance(state.visit.source, RemoteAudioSource)
    assert state.remote_token
    assert f'window.miniRemote.connect({json.dumps(state.remote_token)})' in js.calls   # the browser gets the token
    await state.finish_visit()


async def test_the_microphone_name_the_browser_reports_reaches_the_visit(user: User, tmp_path, monkeypatch):
    monkeypatch.setattr(ui, 'run_javascript', RecordedJavascript({'isSecureContext': True}))
    state = build_state(tmp_path, FakeLLM())
    holder = {}

    @ui.page('/')
    def index():
        holder['page'] = MainPage(state)

    state.import_patient('x')
    await user.open('/')
    holder['page'].w.src_toggle.set_value('remote')
    await until(lambda: holder['page'].remote_mode)
    with user.client:
        await holder['page'].on_remote_event(
            SimpleNamespace(args={'type': 'prepared', 'sample_rate': 48000, 'label': 'USB 麥克風'}))
    assert state.visit.source.label == '遠端瀏覽器麥克風 · USB 麥克風'
    await state.finish_visit()


async def test_a_late_prepared_event_is_ignored_after_switching_back_to_the_local_microphone(user: User, tmp_path,
                                                                                               monkeypatch):
    """The permission prompt can stay open while the physician changes their mind: when the browser finally reports
    'prepared', no remote visit may start, and the browser must let go of the microphone."""
    js = RecordedJavascript({'isSecureContext': True})
    monkeypatch.setattr(ui, 'run_javascript', js)
    state = build_state(tmp_path, FakeLLM())
    holder = {}

    @ui.page('/')
    def index():
        holder['page'] = MainPage(state)

    state.import_patient('x')
    await user.open('/')
    page = holder['page']
    assert not page.remote_mode                                           # still the local microphone
    with user.client:
        await page.on_remote_event(SimpleNamespace(args={'type': 'prepared', 'sample_rate': 48000}))
    assert state.phase == 'imported' and state.visit is None
    assert 'window.miniRemote.abort()' in js.calls and not any('miniRemote.connect' in c for c in js.calls)
    await user.should_see('已切回本機麥克風', retries=40)


async def test_a_page_opened_during_a_visit_shows_the_source_the_visit_really_uses(user: User, tmp_path, monkeypatch):
    monkeypatch.setattr(ui, 'run_javascript', RecordedJavascript({'isSecureContext': True}))
    state = build_state(tmp_path, FakeLLM())
    holder = {}

    @ui.page('/')
    def index():
        holder['page'] = MainPage(state)

    state.import_patient('x')
    await state.start_visit(remote=True)                                  # started from another browser
    await user.open('/')
    w = holder['page'].w
    await until(lambda: w.src_toggle.value == 'remote')                   # not its own default of 'local'
    assert not w.src_toggle.enabled and w.mic_select.visible              # shown for information; cannot be changed
    await state.finish_visit()

    state.import_patient('y')
    await state.start_visit()                                             # a local visit
    await user.open('/')
    w = holder['page'].w
    assert w.src_toggle.value == 'local' and not w.src_toggle.enabled
    await state.finish_visit()


async def test_if_the_visit_cannot_start_the_browser_is_told_to_release_the_microphone(user: User, tmp_path,
                                                                                         monkeypatch):
    js = RecordedJavascript({'isSecureContext': True})
    monkeypatch.setattr(ui, 'run_javascript', js)
    state = build_state(tmp_path, FakeLLM())                                 # no patient imported
    holder = {}

    @ui.page('/')
    def index():
        holder['page'] = MainPage(state)

    await user.open('/')
    holder['page'].w.src_toggle.set_value('remote')                          # the remote mode is on: the START fails
    await until(lambda: holder['page'].remote_mode)
    with user.client:
        await holder['page'].on_remote_event(SimpleNamespace(args={'type': 'prepared'}))
    assert state.phase == 'none' and 'window.miniRemote.abort()' in js.calls
    assert not any('miniRemote.connect' in code for code in js.calls)
    await user.should_see('請先匯入患者資料', retries=40)                      # the real reason is shown, not the mode guard


async def test_a_remote_visit_whose_browser_never_connects_is_flagged_in_the_status_bar(user: User, tmp_path,
                                                                                         monkeypatch):
    monkeypatch.setattr(ui, 'run_javascript', RecordedJavascript({'isSecureContext': True}))
    state = build_state(tmp_path, FakeLLM())

    @ui.page('/')
    def index():
        MainPage(state)

    state.import_patient('x')
    await state.start_visit(remote=True)
    await user.open('/')
    await user.should_see('遠端麥克風：等待瀏覽器連線', retries=40)
    await asyncio.sleep(0.5)
    assert '一直沒有連上' not in user.find('遠端麥克風：').elements.pop().text        # not yet: only flagged after a while
    state.visit.source.created_at -= 60                                          # ... as if a minute had passed
    await user.should_see('瀏覽器一直沒有連上遠端音訊', retries=40)
    await state.finish_visit()


async def test_remote_microphone_errors_are_shown_to_the_physician(user: User, tmp_path):
    state = build_state(tmp_path, FakeLLM())
    holder = {}

    @ui.page('/')
    def index():
        holder['page'] = MainPage(state)

    await user.open('/')
    with user.client:
        await holder['page'].on_remote_event(SimpleNamespace(args={'type': 'error', 'message': '麥克風被拒絕（測試）'}))
    await user.should_see('麥克風被拒絕（測試）', retries=40)


async def test_the_settings_dialog_keeps_the_two_vocabularies_separate(user: User, tmp_path):
    state = build_state(tmp_path, FakeLLM())
    state.settings.asr.vocabulary, state.settings.asr.correction_vocabulary = '舊的ASR詞', '舊的校稿詞'

    @ui.page('/')
    def index():
        MainPage(state)

    await user.open('/')
    user.find('模型設定').click()
    await user.should_see('ASR 專有詞', retries=40)
    await user.should_see('校稿 LLM 專有詞', retries=40)
    boxes = {str(e.props.get('label', '')): e for e in user.find(kind=ui.textarea).elements}
    asr_box = next(box for label, box in boxes.items() if label.startswith('ASR 專有詞'))
    llm_box = next(box for label, box in boxes.items() if label.startswith('校稿 LLM 專有詞'))
    assert (asr_box.value, llm_box.value) == ('舊的ASR詞', '舊的校稿詞')          # each box shows its own list

    llm_box.set_value('新的校稿詞')                                              # change only the correction list
    user.find('儲存設定').click()
    await until(lambda: state.settings.asr.correction_vocabulary == '新的校稿詞')
    assert state.settings.asr.vocabulary == '舊的ASR詞'                          # the ASR list is untouched
    saved = json.loads(state.config_path.read_text(encoding='utf-8'))['asr']
    assert (saved['vocabulary'], saved['correction_vocabulary']) == ('舊的ASR詞', '新的校稿詞')


async def test_agent_buttons_are_unavailable_until_a_visit_starts(user: User, tmp_path):
    state = build_state(tmp_path, FakeLLM())

    @ui.page('/')
    def index():
        MainPage(state)

    await user.open('/')
    for label in ('病歷書寫', '問診建議', '整體分析', 'LLM 去識別化', '開始看診', '結束並存檔'):
        assert not enabled(user, label), label
    assert enabled(user, '患者匯入')
    assert enabled(user, '模型設定') and enabled(user, '模板設定')


async def test_settings_and_templates_are_locked_while_visiting(user: User, tmp_path):
    state = build_state(tmp_path, FakeLLM())

    @ui.page('/')
    def index():
        MainPage(state)

    state.import_patient('x')
    await state.start_visit()
    await user.open('/')
    await until(lambda: not enabled(user, '模型設定'))
    assert not enabled(user, '模板設定')
    assert not enabled(user, '開始看診')
    assert enabled(user, '結束並存檔')
    await state.finish_visit()


async def test_after_a_failed_finish_the_end_button_stays_usable_and_ai_buttons_stay_off(user: User, tmp_path):
    state = build_state(tmp_path, FakeLLM())

    @ui.page('/')
    def index():
        MainPage(state)

    state.import_patient('x')
    await state.start_visit()
    visit = state.visit
    real = visit.store.write_json
    flag = {'failed': False}

    def flaky(rel, data):
        if rel == 'advice/index.json' and not flag['failed']:
            flag['failed'] = True
            raise OSError('disk full (simulated)')
        return real(rel, data)

    visit.store.write_json = flaky
    with pytest.raises(OSError):
        await state.finish_visit()
    await user.open('/')
    await user.should_see('結束並存檔失敗', retries=40)
    assert enabled(user, '結束並存檔')                                        # retry is possible from the screen
    for label in ('病歷書寫', '問診建議', '整體分析'):
        assert not enabled(user, label), label
    user.find('結束並存檔').click()
    await until(lambda: state.phase == 'none', timeout=20)


async def test_patient_import_button_is_locked_while_a_job_runs(user: User, tmp_path):
    fake = FakeLLM()
    fake.delay = 0.8
    state = build_state(tmp_path, fake)

    @ui.page('/')
    def index():
        MainPage(state)

    state.import_patient('x')
    await state.start_visit()
    await until(lambda: state.visit.pipeline.finished)
    await user.open('/')
    assert enabled(user, '患者匯入')
    job = state.visit.start_job('record')
    await until(lambda: not enabled(user, '患者匯入'), timeout=5)
    await job.task
    await until(lambda: enabled(user, '患者匯入'), timeout=5)
    await state.finish_visit()


# ============================ LLM 去識別化 window =================================
def marked(user: User, marker: str):
    elements = list(user.find(marker=marker).elements)
    assert len(elements) == 1, f'{marker}: {len(elements)} elements'
    return elements[0]


def deid_text(user: User) -> str:
    return marked(user, 'deid-text').value


async def open_deid_window(user: User, state):
    await until(lambda: state.visit.pipeline.finished)
    user.find(kind=ui.button, content='LLM 去識別化').click()
    await user.should_see(marker='deid-run', retries=40)


async def test_the_deidentification_window_runs_browses_and_copies_the_shown_version(user: User, tmp_path, monkeypatch):
    js = RecordedJavascript({'miniCopy': True})
    monkeypatch.setattr(ui, 'run_javascript', js)
    fake = FakeLLM()
    fake.queue('deid', deid_reply(note='甲- 現病史：第一版'), deid_reply(note='甲- 現病史：第二版'))
    state = build_state(tmp_path, fake)

    @ui.page('/')
    def index():
        MainPage(state)

    state.import_patient('王大明，男，45歲')
    await state.start_visit()
    await user.open('/')
    await open_deid_window(user, state)
    assert deid_text(user) == '' and marked(user, 'deid-label').text == '0/0'
    for marker in ('deid-prev', 'deid-next', 'deid-copy'):                       # nothing to browse or copy yet
        assert not marked(user, marker).enabled, marker
    assert marked(user, 'deid-run').enabled

    marked(user, 'deid-extra').set_value('連職業也刪掉')
    user.find(marker='deid-run').click()
    await until(lambda: len(state.visit.deid) == 1)
    await until(lambda: '第一版' in deid_text(user))
    assert '## 【今日看診日期】\nD日' in deid_text(user) and marked(user, 'deid-label').text.startswith('1/1')
    assert state.visit.deid[0].extra_instruction == '連職業也刪掉'
    assert '## 【醫師額外指示（本次）】\n連職業也刪掉' in fake.calls_for('deid')[0][1]['content']

    await until(lambda: marked(user, 'deid-run').enabled)                          # the instruction keeps applying until cleared
    user.find(marker='deid-run').click()
    await until(lambda: len(state.visit.deid) == 2)
    await until(lambda: '第二版' in deid_text(user))
    assert marked(user, 'deid-label').text.startswith('2/2') and marked(user, 'deid-prev').enabled
    assert not marked(user, 'deid-next').enabled
    assert state.visit.deid[1].extra_instruction == '連職業也刪掉'

    user.find(marker='deid-prev').click()                                           # 上一則 shows the earlier run
    await until(lambda: '第一版' in deid_text(user))
    assert marked(user, 'deid-label').text.startswith('1/2')
    assert not marked(user, 'deid-prev').enabled and marked(user, 'deid-next').enabled

    user.find(marker='deid-copy').click()                                           # 複製 copies the version on screen
    await until(lambda: any('miniCopy' in code for code in js.calls))
    copied = [code for code in js.calls if 'miniCopy' in code]
    assert copied == [f'window.miniCopy({json.dumps(state.visit.deid[0].text, ensure_ascii=False)})']
    assert state.visit.deid_index == 0
    await state.finish_visit()


async def test_the_extra_instruction_applies_to_every_run_until_the_physician_clears_it(user: User, tmp_path):
    fake = FakeLLM()
    state = build_state(tmp_path, fake)

    @ui.page('/')
    def index():
        MainPage(state)

    state.import_patient('x')
    await state.start_visit()
    await user.open('/')
    await open_deid_window(user, state)
    assert '直到你手動清除' in next(iter(user.find(content='直到你手動清除').elements)).text      # what the window promises
    marked(user, 'deid-extra').set_value('連職業也刪掉')
    assert state.visit.deid_extra == '連職業也刪掉'                                    # kept on the visit, not only in the box
    user.find(marker='deid-run').click()
    await until(lambda: len(state.visit.deid) == 1)

    await user.open('/')                                                             # a reloaded page: a brand-new window
    await open_deid_window(user, state)
    assert marked(user, 'deid-extra').value == '連職業也刪掉'                         # still there, no retyping
    await until(lambda: marked(user, 'deid-run').enabled)
    user.find(marker='deid-run').click()
    await until(lambda: len(state.visit.deid) == 2)
    assert [v.extra_instruction for v in state.visit.deid] == ['連職業也刪掉', '連職業也刪掉']
    assert all('## 【醫師額外指示（本次）】\n連職業也刪掉' in call[1]['content'] for call in fake.calls_for('deid'))

    marked(user, 'deid-extra').set_value('')                                         # cleared by hand: no longer sent
    assert state.visit.deid_extra == ''
    await until(lambda: marked(user, 'deid-run').enabled)
    user.find(marker='deid-run').click()
    await until(lambda: len(state.visit.deid) == 3)
    assert state.visit.deid[2].extra_instruction == '' and '醫師額外指示' not in fake.calls_for('deid')[2][1]['content']
    await state.finish_visit()


async def test_the_copy_follows_the_simplified_display_like_the_other_panels(user: User, tmp_path, monkeypatch):
    js = RecordedJavascript({'miniCopy': True})
    monkeypatch.setattr(ui, 'run_javascript', js)
    state = build_state(tmp_path, FakeLLM())

    @ui.page('/')
    def index():
        MainPage(state)

    state.import_patient('x')
    await state.start_visit()
    await user.open('/')
    await open_deid_window(user, state)
    user.find(marker='deid-run').click()
    await until(lambda: len(state.visit.deid) == 1)
    state.set_script_mode('simplified')
    await until(lambda: '头痛' in deid_text(user))
    user.find(marker='deid-copy').click()
    await until(lambda: any('miniCopy' in code for code in js.calls))
    assert '头痛' in js.calls[-1] and '頭痛' in state.visit.deid[0].note_text        # shown and copied converted; stored as is
    await state.finish_visit()


async def test_residual_warnings_and_the_models_summary_are_shown_but_not_copied(user: User, tmp_path, monkeypatch):
    js = RecordedJavascript({'miniCopy': True})
    monkeypatch.setattr(ui, 'run_javascript', js)
    fake = FakeLLM()
    fake.queue('deid', deid_reply(patient='患者，王先生，男，45歲', summary='- 已刪除電話'))
    state = build_state(tmp_path, fake)

    @ui.page('/')
    def index():
        MainPage(state)

    state.import_patient('王先生，男，45歲')
    await state.start_visit()
    await user.open('/')
    await open_deid_window(user, state)
    user.find(marker='deid-run').click()
    await until(lambda: len(state.visit.deid) == 1)
    await user.should_see('姓氏＋稱謂', retries=40)
    await user.should_see('已刪除電話', retries=40)
    user.find(marker='deid-copy').click()
    await until(lambda: any('miniCopy' in code for code in js.calls))
    assert '已刪除電話' not in js.calls[-1] and '姓氏＋稱謂' not in js.calls[-1]
    await user.should_see('這則有 1 項殘留提醒', retries=40)                          # copying a flagged version says so
    await state.finish_visit()


async def test_copying_a_clean_version_does_not_warn(user: User, tmp_path, monkeypatch):
    js = RecordedJavascript({'miniCopy': True})
    monkeypatch.setattr(ui, 'run_javascript', js)
    state = build_state(tmp_path, FakeLLM())

    @ui.page('/')
    def index():
        MainPage(state)

    state.import_patient('x')
    await state.start_visit()
    await user.open('/')
    await open_deid_window(user, state)
    user.find(marker='deid-run').click()
    await until(lambda: len(state.visit.deid) == 1)
    await until(lambda: '頭痛' in deid_text(user))
    user.find(marker='deid-copy').click()
    await user.should_see('已複製去識別化病歷第 1 則', retries=40)
    assert not user.notify.contains('殘留提醒')
    await state.finish_visit()


async def test_a_failed_run_says_so_in_the_window_and_the_next_run_works(user: User, tmp_path):
    fake = FakeLLM()
    fake.queue('deid', *['缺標記'] * 4)
    state = build_state(tmp_path, fake)

    @ui.page('/')
    def index():
        MainPage(state)

    state.import_patient('x')
    await state.start_visit()
    await user.open('/')
    await open_deid_window(user, state)
    user.find(marker='deid-run').click()
    await until(lambda: '去識別化失敗' in marked(user, 'deid-status').text, timeout=20)
    status = marked(user, 'deid-status')
    assert state.visit.deid == [] and marked(user, 'deid-run').enabled and 'text-red-700' in status.classes
    fake.delay = 0.6
    user.find(marker='deid-run').click()
    await until(lambda: state.visit.jobs.busy)
    await until(lambda: status.text.startswith('去識別化：'))                          # running: progress, not the old failure
    assert 'text-red-700' not in status.classes
    await until(lambda: len(state.visit.deid) == 1)
    await until(lambda: status.text == '')                                            # the old failure is gone
    await state.finish_visit()


async def test_the_window_opens_while_another_job_runs_and_its_run_button_waits_for_the_slot(user: User, tmp_path):
    fake = FakeLLM()
    fake.delay = 0.8
    state = build_state(tmp_path, fake)

    @ui.page('/')
    def index():
        MainPage(state)

    state.import_patient('x')
    await state.start_visit()
    await until(lambda: state.visit.pipeline.finished)
    await user.open('/')
    await until(lambda: enabled(user, 'LLM 去識別化'))
    job = state.visit.start_job('advice')
    await open_deid_window(user, state)                                              # opening is allowed while busy
    assert not marked(user, 'deid-run').enabled and marked(user, 'deid-status').text.startswith('問診建議')
    await job.task
    await until(lambda: marked(user, 'deid-run').enabled)                           # the slot is free again
    await state.finish_visit()


async def test_a_running_deidentification_can_be_cancelled_from_its_window(user: User, tmp_path):
    fake = FakeLLM()
    fake.delay = 5
    state = build_state(tmp_path, fake)

    @ui.page('/')
    def index():
        MainPage(state)

    state.import_patient('x')
    await state.start_visit()
    await user.open('/')
    await open_deid_window(user, state)
    user.find(marker='deid-run').click()
    await until(lambda: state.visit.jobs.busy)
    await until(lambda: marked(user, 'deid-status').text.startswith('去識別化：'))
    assert not marked(user, 'deid-run').enabled
    user.find(kind=ui.button, content='取消作業').click()
    await until(lambda: not state.visit.jobs.busy, timeout=10)
    assert state.visit.deid == [] and state.visit.jobs.last.status == 'cancelled'
    await until(lambda: marked(user, 'deid-run').enabled)
    await state.finish_visit()


async def test_the_window_closes_itself_when_the_visit_ends(user: User, tmp_path):
    state = build_state(tmp_path, FakeLLM())

    @ui.page('/')
    def index():
        MainPage(state)

    state.import_patient('x')
    await state.start_visit()
    await user.open('/')
    await open_deid_window(user, state)
    dialog = next(a for a in marked(user, 'deid-run').ancestors() if isinstance(a, ui.dialog))
    assert dialog.value                                                              # open
    await state.finish_visit()
    await until(lambda: not dialog.value)                                            # nothing left to run against


async def test_the_settings_dialog_lists_the_deidentification_interface(user: User, tmp_path):
    state = build_state(tmp_path, FakeLLM())

    @ui.page('/')
    def index():
        MainPage(state)

    await user.open('/')
    user.find('模型設定').click()
    await user.should_see('LLM 接口', retries=40)
    user.find('LLM 接口').click()
    await user.should_see('八個接口各自獨立', retries=40)
    await user.should_see('去識別化', retries=40)
    user.find('儲存設定').click()
    await until(lambda: 'deidentifier' in json.loads(state.config_path.read_text(encoding='utf-8'))['agents'])


# ============================ 患者資訊 floating panel =============================
PATIENT_WITH_TAGS = '王大明，男，45歲。\n上次就診：頭痛三天 [語音#3]，高血壓 [歷史]'
PATIENT_SHOWN = '王大明，男，45歲。\n上次就診：頭痛三天，高血壓'                        # the same, without the source tags


def patient_page(state):
    holder = {}

    @ui.page('/')
    def index():
        holder['page'] = MainPage(state)

    return holder


async def test_the_patient_button_is_always_available_and_the_panel_starts_closed(user: User, tmp_path):
    fake = FakeLLM()
    state = build_state(tmp_path, fake)
    holder = patient_page(state)
    await user.open('/')
    page = holder['page']
    assert enabled(user, '患者資訊') and not page.w.patient_panel.visible                   # no patient at all
    assert 'patient_panel' not in dir(state)                                                   # purely a page matter

    state.import_patient(PATIENT_WITH_TAGS)
    await until(lambda: enabled(user, '患者資訊'))                                              # imported, not started
    await state.start_visit()
    await until(lambda: state.visit.pipeline.finished)
    assert enabled(user, '患者資訊')                                                            # visiting

    fake.delay = 0.6
    job = state.visit.start_job('advice')                                                      # a job is running
    await until(lambda: state.visit.jobs.busy)
    await until(lambda: not enabled(user, '問診建議'))                                          # the page has caught up
    assert enabled(user, '患者資訊')
    await job.task

    with user.client:
        page.enter_note_edit()                                                                     # the note is being edited
    assert page.note_mode == 'edit'
    await until(lambda: not enabled(user, '問診建議'))
    assert enabled(user, '患者資訊')
    with user.client:
        page.leave_note_edit(False)
    await state.finish_visit()
    assert enabled(user, '患者資訊')


async def test_the_button_opens_and_closes_the_panel_which_shows_the_patient_data_without_source_tags(
        user: User, tmp_path, monkeypatch):
    js = RecordedJavascript()
    monkeypatch.setattr(ui, 'run_javascript', js)
    state = build_state(tmp_path, FakeLLM())
    holder = patient_page(state)
    state.import_patient(PATIENT_WITH_TAGS)
    await user.open('/')
    w = holder['page'].w
    assert not w.patient_panel.visible

    user.find(kind=ui.button, content='患者資訊').click()
    await until(lambda: w.patient_panel.visible)
    assert w.patient_text.text == PATIENT_SHOWN and 'empty' not in w.patient_text.classes
    assert f"window.miniFloat.show('c{w.patient_panel.id}')" in js.calls                      # the browser places it
    assert state.patient_text == PATIENT_WITH_TAGS                                             # display only: nothing changed

    user.find(kind=ui.button, content='患者資訊').click()                                       # the same button closes it
    await until(lambda: not w.patient_panel.visible)
    user.find(kind=ui.button, content='患者資訊').click()
    await until(lambda: w.patient_panel.visible)
    user.find(marker='patient-close').click()                                                  # so does the ✕
    await until(lambda: not w.patient_panel.visible)


async def test_the_panel_follows_the_patient_data_during_a_visit_and_the_display_script(user: User, tmp_path):
    state = build_state(tmp_path, FakeLLM())
    holder = patient_page(state)
    state.import_patient(PATIENT_WITH_TAGS)
    await state.start_visit()
    await user.open('/')
    w = holder['page'].w
    with user.client:
        holder['page'].set_patient_open(True)
    await until(lambda: w.patient_text.text == PATIENT_SHOWN)

    state.import_patient('李小美，女，30歲。頭痛兩週。')                                          # edited during the visit
    await until(lambda: w.patient_text.text == '李小美，女，30歲。頭痛兩週。')

    state.set_script_mode('simplified')                                                        # display layer only
    await until(lambda: '头痛两周' in w.patient_text.text)
    assert state.visit.patient_text == '李小美，女，30歲。頭痛兩週。'
    state.set_script_mode('original')
    await until(lambda: '頭痛兩週' in w.patient_text.text)
    await state.finish_visit()


async def test_without_a_patient_the_panel_says_so_and_cannot_copy(user: User, tmp_path, monkeypatch):
    js = RecordedJavascript({'miniCopy': True})
    monkeypatch.setattr(ui, 'run_javascript', js)
    state = build_state(tmp_path, FakeLLM())
    holder = patient_page(state)
    await user.open('/')
    page = holder['page']
    with user.client:
        page.set_patient_open(True)
    assert page.w.patient_text.text == '（尚未匯入患者）' and 'empty' in page.w.patient_text.classes
    assert not page.w.btn_patient_copy.enabled

    state.import_patient('x，女')                                                               # a patient arrives
    await until(lambda: page.w.patient_text.text == 'x，女' and page.w.btn_patient_copy.enabled)
    state.import_patient('y')
    await state.start_visit()
    await state.finish_visit()                                                                  # and leaves: the open panel empties
    await until(lambda: page.w.patient_text.text == '（尚未匯入患者）' and not page.w.btn_patient_copy.enabled)
    assert page.w.patient_panel.visible                                                         # still open: the physician decides


async def test_copy_takes_the_displayed_text_without_source_tags(user: User, tmp_path, monkeypatch):
    js = RecordedJavascript({'miniCopy': True})
    monkeypatch.setattr(ui, 'run_javascript', js)
    state = build_state(tmp_path, FakeLLM())
    holder = patient_page(state)
    state.import_patient(PATIENT_WITH_TAGS)
    await user.open('/')
    with user.client:
        holder['page'].set_patient_open(True)
    user.find(marker='patient-copy').click()
    await until(lambda: any('miniCopy' in code for code in js.calls))
    assert [code for code in js.calls if 'miniCopy' in code] == [
        f'window.miniCopy({json.dumps(PATIENT_SHOWN, ensure_ascii=False)})']
    await user.should_see('已複製患者匯入資料', retries=40)

    state.set_script_mode('simplified')                                                         # copy follows the display, like the others
    await until(lambda: '头痛' in holder['page'].w.patient_text.text)
    user.find(marker='patient-copy').click()
    await until(lambda: sum('miniCopy' in code for code in js.calls) == 2)
    assert '头痛三天' in js.calls[-1]


async def test_an_open_panel_does_not_get_in_the_way_of_a_job(user: User, tmp_path):
    state = build_state(tmp_path, FakeLLM())
    holder = patient_page(state)
    state.import_patient(PATIENT_WITH_TAGS)
    await state.start_visit()
    await user.open('/')
    page = holder['page']
    await until(lambda: state.visit.pipeline.finished)
    with user.client:
        page.set_patient_open(True)
    job = state.visit.start_job('record')
    await asyncio.wait_for(job.task, 20)
    assert job.status == 'succeeded' and state.visit.note.current_index == 1                  # the note was written as usual
    assert page.w.patient_panel.visible and page.w.patient_text.text == PATIENT_SHOWN
    await state.finish_visit()


def test_the_browser_side_drag_code_and_its_styles_are_wired_to_the_same_class_names():
    """The drag itself runs in the browser and is checked there by hand (SPEC §20.5); this only guards the contract
    between the Python widgets, the CSS and the script."""
    from mini.ui.style import CSS, JS
    for name in ('.float-panel', '.float-head', '.float-body', '.patient-text'):
        assert name in CSS, name
    assert 'window.miniFloat' in JS and "closest('.float-head')" in JS and "querySelectorAll('.float-panel')" in JS
    assert 'z-index:3000' in CSS                                    # above the page, below Quasar dialogs (6000)
    # The header's buttons are at its right end and are not draggable, so the left edge must never leave the window (an
    # earlier rule let a panel hide all but its right 80 px, leaving only the buttons to grab).
    assert 'Math.max(left, 0)' in JS and 'MIN_VISIBLE - ' not in JS
    # opening and shrinking the window pull the panel back into view on both axes (the content, not only the header)
    assert JS.count('window.innerHeight - el.offsetHeight') == 1 and 'window.innerHeight - h)' in JS
    # With storage blocked, reopening keeps this page's place but must still be fitted, so show() has no early exit other
    # than "the element is missing" (it once returned early when the element already had a place, which skipped the fitting;
    # a pattern match on that exact form is too easy to dodge by renaming a variable, so count the exits instead).
    show = JS.split('function show(id) {')[1].split('function reset')[0]
    assert show.count('return') == 1 and 'if (!el) return;' in show and 'parseFloat(el.style.left)' in show


# ============================ editing the patient data during a visit ===============
async def test_the_import_dialog_during_a_visit_opens_on_the_latest_text_and_a_second_save_builds_on_it(
        user: User, tmp_path):
    """Editing the data in a visit changes only the visit's copy; the dialog used to load the text the visit started with,
    so a second edit began from the first version and silently replaced the first edit."""
    state = build_state(tmp_path, FakeLLM())
    patient_page(state)
    state.import_patient('第一版：王先生，高血壓')
    await state.start_visit()
    state.import_patient('第二版：王先生，高血壓，新增糖尿病')                                # the first edit, during the visit
    assert state.patient_text == '第一版：王先生，高血壓'                                       # the text the visit started with

    await user.open('/')
    await until(lambda: enabled(user, '患者匯入'))
    user.find('患者匯入').click()
    await user.should_see('看診中修改會留下版本與 log', retries=40)
    area = next(iter(user.find(kind=ui.textarea).elements))
    assert area.value == '第二版：王先生，高血壓，新增糖尿病'                                  # not the first version

    area.set_value(area.value + '，近日胸悶')                                                  # the second edit builds on it
    user.find('儲存修改').click()
    await until(lambda: state.visit.patient_version == 3)
    assert [v['text'] for v in state.visit.patient_versions] == [
        '第一版：王先生，高血壓', '第二版：王先生，高血壓，新增糖尿病', '第二版：王先生，高血壓，新增糖尿病，近日胸悶']
    assert [e['type'] for e in state.visit.log.events if e['type'].startswith('patient_input')] == [
        'patient_input_set', 'patient_input_updated', 'patient_input_updated']

    await user.open('/')                                                                       # and again after a reload
    await until(lambda: enabled(user, '患者匯入'))
    user.find('患者匯入').click()
    await user.should_see('看診中修改會留下版本與 log', retries=40)
    assert next(iter(user.find(kind=ui.textarea).elements)).value.endswith('近日胸悶')
    await state.finish_visit()


async def test_the_current_patient_text_follows_the_visit_and_goes_with_it(tmp_path):
    state = build_state(tmp_path, FakeLLM())
    assert state.current_patient_text == ''
    state.import_patient('甲')
    assert state.current_patient_text == '甲'                                                  # imported, not started
    await state.start_visit()
    assert state.current_patient_text == '甲'
    state.import_patient('乙')                                                                 # edited in the visit
    assert state.current_patient_text == '乙' and state.patient_text == '甲'
    await state.finish_visit()
    assert state.current_patient_text == ''                                                    # the visit is over, so is the patient


# ============================ a window left open in another tab ====================
async def open_import_dialog(user: User, state, button: str):
    await user.open('/')
    await until(lambda: enabled(user, '患者匯入'))
    user.find('患者匯入').click()
    await user.should_see(button, retries=40)


async def test_a_stale_import_window_cannot_overwrite_a_version_another_tab_saved(user: User, tmp_path):
    state = build_state(tmp_path, FakeLLM())
    patient_page(state)
    state.import_patient('第一版')
    await state.start_visit()
    await open_import_dialog(user, state, '儲存修改')                                        # "tab B" opens the window ...
    area = next(iter(user.find(kind=ui.textarea).elements))
    assert area.value == '第一版'
    state.import_patient('第二版（另一個分頁存的）')                                            # ... "tab A" saves a newer version
    area.set_value('第一版，加上 B 分頁的修改')
    user.find('儲存修改').click()
    await user.should_see('已被改變', retries=40)                                             # refused, with the reason
    assert state.visit.patient_text == '第二版（另一個分頁存的）' and state.visit.patient_version == 2
    await state.finish_visit()


async def test_a_stale_import_window_cannot_turn_the_old_visits_text_into_the_next_patient(user: User, tmp_path):
    state = build_state(tmp_path, FakeLLM())
    patient_page(state)
    state.import_patient('王先生')
    await state.start_visit()
    await open_import_dialog(user, state, '儲存修改')
    await state.finish_visit()                                                                # the visit ended in another tab
    assert state.phase == 'none'
    user.find('儲存修改').click()
    await user.should_see('已被改變', retries=40)
    assert state.phase == 'none' and state.patient_text == ''                                 # nobody was imported


async def test_a_stale_import_window_cannot_edit_the_next_visit_even_when_its_version_number_matches(user: User, tmp_path):
    """Visit 1 and visit 2 both stand at version 1 of their data, so the version alone cannot tell them apart."""
    state = build_state(tmp_path, FakeLLM())
    patient_page(state)
    state.import_patient('第一位患者')
    await state.start_visit()
    await open_import_dialog(user, state, '儲存修改')
    next(iter(user.find(kind=ui.textarea).elements)).set_value('第一位患者，過期視窗的修改')
    await state.finish_visit()
    state.import_patient('第二位患者')
    await state.start_visit()                                                                 # the next visit, also version 1
    assert state.visit.patient_version == 1
    user.find('儲存修改').click()
    await user.should_see('已被改變', retries=40)
    assert state.visit.patient_text == '第二位患者' and state.visit.patient_version == 1
    await state.finish_visit()


async def test_a_window_opened_before_the_visit_cannot_edit_the_visit_with_stale_text(user: User, tmp_path):
    state = build_state(tmp_path, FakeLLM())
    patient_page(state)
    state.import_patient('開看診前的文字')
    await open_import_dialog(user, state, '確定')
    await state.start_visit()                                                                 # started in another tab
    user.find('確定').click()
    await user.should_see('已被改變', retries=40)
    assert state.visit.patient_text == '開看診前的文字' and state.visit.patient_version == 1
    await state.finish_visit()


async def test_an_import_window_that_is_still_current_saves_as_before(user: User, tmp_path):
    state = build_state(tmp_path, FakeLLM())
    patient_page(state)
    state.import_patient('第一版')
    await state.start_visit()
    await open_import_dialog(user, state, '儲存修改')
    next(iter(user.find(kind=ui.textarea).elements)).set_value('第二版')
    user.find('儲存修改').click()
    await until(lambda: state.visit.patient_version == 2)
    assert state.visit.patient_text == '第二版'
    await state.finish_visit()


async def test_the_backend_refuses_edits_to_a_visit_that_is_closing_or_closed(tmp_path):
    """Not only the toolbar: a window another tab left open reaches import_patient directly."""
    state = build_state(tmp_path, FakeLLM())
    state.import_patient('第一版')
    await state.start_visit()
    visit = state.visit
    for phase in ('starting', 'finishing', 'finish_failed', 'finished'):
        visit.phase = phase
        with pytest.raises(StateError, match='無法再修改'):
            state.import_patient('不該存進去')
    visit.phase = 'recording'
    state._busy_transition = True                                                             # an end-visit in progress
    with pytest.raises(StateError, match='無法再修改'):
        state.import_patient('不該存進去')
    state._busy_transition = False
    assert visit.patient_text == '第一版' and visit.patient_version == 1
    state.import_patient('第二版')                                                            # recording again: allowed
    assert visit.patient_text == '第二版'
    await state.finish_visit()


# ============================ two windows before the visit ==========================
async def test_before_a_visit_a_stale_import_window_cannot_overwrite_what_another_tab_imported(user: User, tmp_path):
    state = build_state(tmp_path, FakeLLM())
    patient_page(state)
    state.import_patient('X')
    await open_import_dialog(user, state, '確定')                                            # "tab B" opens its window
    state.import_patient('A 分頁改的')                                                        # "tab A" imports other text
    next(iter(user.find(kind=ui.textarea).elements)).set_value('B 分頁改的')
    user.find('確定').click()
    await user.should_see('已被改變', retries=40)
    assert state.patient_text == 'A 分頁改的' and state.phase == 'imported'


async def test_a_patient_cleared_in_another_tab_is_not_brought_back_by_a_stale_window(user: User, tmp_path):
    state = build_state(tmp_path, FakeLLM())
    patient_page(state)
    state.import_patient('X')
    await open_import_dialog(user, state, '確定')
    state.clear_patient()                                                                     # cleared in "tab A"
    assert state.phase == 'none' and state.patient_text == ''
    user.find('確定').click()
    await user.should_see('已被改變', retries=40)
    assert state.phase == 'none' and state.patient_text == ''                                 # still nobody


async def test_a_window_opened_with_no_patient_cannot_overwrite_one_imported_meanwhile(user: User, tmp_path):
    state = build_state(tmp_path, FakeLLM())
    patient_page(state)
    await open_import_dialog(user, state, '確定')
    state.import_patient('A 分頁匯入的')
    next(iter(user.find(kind=ui.textarea).elements)).set_value('B 分頁打的')
    user.find('確定').click()
    await user.should_see('已被改變', retries=40)
    assert state.patient_text == 'A 分頁匯入的'


async def test_before_a_visit_a_window_that_is_still_current_imports_as_before(user: User, tmp_path):
    state = build_state(tmp_path, FakeLLM())
    patient_page(state)
    state.import_patient('X')
    await open_import_dialog(user, state, '確定')
    next(iter(user.find(kind=ui.textarea).elements)).set_value('Y')
    user.find('確定').click()
    await until(lambda: state.patient_text == 'Y')
    assert state.phase == 'imported'


async def test_the_patient_stamp_moves_with_every_change_and_only_then(tmp_path):
    state = build_state(tmp_path, FakeLLM())
    empty = state.patient_stamp
    assert state.patient_stamp == empty                                                       # reading it changes nothing
    state.import_patient('X')
    imported = state.patient_stamp
    assert imported != empty
    state.import_patient('X')                                                                 # the same text again is still an import
    assert state.patient_stamp != imported
    again = state.patient_stamp
    state.clear_patient()
    assert state.patient_stamp not in (empty, imported, again)
    state.import_patient('Y')
    await state.start_visit()
    in_visit = state.patient_stamp
    assert in_visit[0] is state.visit and in_visit[1] == 1
    state.import_patient('Z')
    assert state.patient_stamp != in_visit and state.patient_stamp[1] == 2
    await state.finish_visit()


# ============================ the stamp must never come back to an old value ========
async def test_a_window_opened_before_a_whole_visit_cannot_bring_the_old_patient_back(user: User, tmp_path):
    """The stamp used to return to `(None, revision)` once a visit that another tab started and finished had cleared the
    patient without moving the revision, so the stale window passed its check and imported the previous patient again."""
    state = build_state(tmp_path, FakeLLM())
    patient_page(state)
    state.import_patient('上一位患者')
    await open_import_dialog(user, state, '確定')                                            # "tab B" opens its window
    await state.start_visit()                                                                 # "tab A" runs a whole visit
    await state.finish_visit()
    assert state.phase == 'none' and state.patient_text == ''
    user.find('確定').click()
    await user.should_see('已被改變', retries=40)
    assert state.phase == 'none' and state.patient_text == ''                                 # the old patient stays gone


async def test_the_stamp_never_returns_to_an_earlier_value_over_a_whole_lifecycle(tmp_path):
    state = build_state(tmp_path, FakeLLM())
    stamps = [state.patient_stamp]
    state.import_patient('甲')
    stamps.append(state.patient_stamp)
    state.import_patient('甲')                                                                # even the same text again
    stamps.append(state.patient_stamp)
    state.clear_patient()
    stamps.append(state.patient_stamp)
    state.import_patient('乙')
    stamps.append(state.patient_stamp)
    await state.start_visit()
    stamps.append(state.patient_stamp)
    state.import_patient('乙，改過')
    stamps.append(state.patient_stamp)
    await state.finish_visit()
    stamps.append(state.patient_stamp)                                                        # a finished visit is not the start
    state.import_patient('乙')                                                                # the very same text as before
    stamps.append(state.patient_stamp)
    await state.start_visit()
    stamps.append(state.patient_stamp)
    await state.finish_visit()
    stamps.append(state.patient_stamp)
    assert len(set(stamps)) == len(stamps), stamps


async def test_every_assignment_to_the_patient_text_moves_the_revision(tmp_path):
    """Whatever path writes the text -- import, clear, the end of a visit, or one added later -- leaves a trace."""
    state = build_state(tmp_path, FakeLLM())
    before = state.patient_revision
    state.patient_text = 'x'
    assert state.patient_revision == before + 1
    state.patient_text = 'x'
    assert state.patient_revision == before + 2 and state.patient_text == 'x'


# ============================ the clear button gets the same guard ==================
async def test_a_stale_window_cannot_clear_a_patient_another_tab_just_saved(user: User, tmp_path):
    state = build_state(tmp_path, FakeLLM())
    patient_page(state)
    state.import_patient('X')
    await open_import_dialog(user, state, '清除患者')                                        # "tab B" opens its window
    state.import_patient('A 分頁存的新版')                                                    # "tab A" saves a newer text
    user.find('清除患者').click()
    await user.should_see('已被改變', retries=40)
    assert state.phase == 'imported' and state.patient_text == 'A 分頁存的新版'                # nothing was deleted


async def test_a_window_opened_before_the_visit_cannot_clear_once_the_visit_started(user: User, tmp_path):
    state = build_state(tmp_path, FakeLLM())
    patient_page(state)
    state.import_patient('X')
    await open_import_dialog(user, state, '清除患者')
    await state.start_visit()                                                                 # started in another tab
    user.find('清除患者').click()
    await user.should_see('已被改變', retries=40)
    assert state.phase == 'visiting' and state.visit.patient_text == 'X'
    await state.finish_visit()


async def test_a_window_that_is_still_current_clears_as_before(user: User, tmp_path):
    state = build_state(tmp_path, FakeLLM())
    patient_page(state)
    state.import_patient('X')
    await open_import_dialog(user, state, '清除患者')
    user.find('清除患者').click()
    await until(lambda: state.phase == 'none')
    assert state.patient_text == ''
    await user.should_see('已清除患者', retries=40)
