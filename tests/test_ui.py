"""UI wiring tests using NiceGUI's in-process user simulation (no browser needed)."""
import asyncio
import json

import pytest
from nicegui import ui
from nicegui.testing import User

from mini.config import Settings
from mini.llm import LLMClient
from mini.state import AppState
from mini.ui.page import MainPage
from tests.helpers import FakeASR, FakeLLM, silent_source


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


async def test_agent_buttons_are_unavailable_until_a_visit_starts(user: User, tmp_path):
    state = build_state(tmp_path, FakeLLM())

    @ui.page('/')
    def index():
        MainPage(state)

    await user.open('/')
    for label in ('病歷書寫', '問診建議', '整體分析', '開始看診', '結束並存檔'):
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
