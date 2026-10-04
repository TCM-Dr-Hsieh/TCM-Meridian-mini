import json

import pytest

from mini.config import Settings
from mini.llm import LLMClient
from mini.state import AppState, StateError
from tests.helpers import FakeASR, FakeLLM, silent_source


@pytest.fixture
def app(tmp_path):
    settings = Settings()
    settings.visits_dir = str(tmp_path / 'visits')
    (tmp_path / 'config.json').write_text(json.dumps(settings.to_dict(), ensure_ascii=False), encoding='utf-8')
    templates = tmp_path / 'templates'
    (templates / 'defaults').mkdir(parents=True)
    (templates / 'defaults' / 'record_template.txt').write_text('甲- 現病史：\n', encoding='utf-8')
    (templates / 'defaults' / 'analysis_template.txt').write_text('一- 西醫診斷：\n', encoding='utf-8')
    fake = FakeLLM()
    state = AppState(config_path=tmp_path / 'config.json', templates_dir=templates,
                     client=LLMClient(fake.transport()), asr=FakeASR(default='頭痛'), llm_backoff=0,
                     source_factory=lambda path: silent_source(state.settings, 1, recording_path=path))
    return state


async def test_state_machine_happy_path(app):
    assert app.phase == 'none'
    with pytest.raises(StateError):
        await app.start_visit()                      # no patient yet
    app.import_patient('  王先生 45歲  ')
    assert app.phase == 'imported' and app.patient_text == '王先生 45歲'
    folder = await app.start_visit()
    assert app.phase == 'visiting' and folder.name.endswith('-001') and app.visit.phase == 'recording'
    app.import_patient('王先生 45歲，補充資料')                 # allowed during the visit, versioned
    assert len(app.visit.patient_versions) == 2
    with pytest.raises(StateError):
        await app.start_visit()
    done = await app.finish_visit()
    assert done == folder and app.phase == 'none' and app.visit is None and app.patient_text == ''
    assert app.last_visit_folder == folder
    app.import_patient('下一位')
    assert (await app.start_visit()).name.endswith('-002')
    await app.finish_visit()


async def test_empty_import_is_refused_and_clear_patient_only_before_the_visit(app):
    with pytest.raises(StateError):
        app.import_patient('   ')
    app.import_patient('x')
    app.clear_patient()
    assert app.phase == 'none'
    app.import_patient('x')
    await app.start_visit()
    with pytest.raises(StateError):
        app.clear_patient()
    await app.finish_visit()


async def test_global_settings_and_templates_are_locked_during_a_visit(app):
    app.import_patient('x')
    new = Settings.from_dict(app.settings.to_dict())
    new.llm.max_concurrency = 4
    app.update_settings(new)                          # fine before the visit
    assert app.scheduler.limit == 4 and Settings.load(app.config_path).llm.max_concurrency == 4
    app.save_template('analysis', '一- 診斷：\n')
    assert app.get_template('analysis') == '一- 診斷：\n'
    assert app.default_template('analysis') == '一- 西醫診斷：\n'
    await app.start_visit()
    with pytest.raises(StateError):
        app.update_settings(new)
    with pytest.raises(StateError):
        app.save_template('record', 'x')
    await app.finish_visit()


async def test_invalid_settings_are_rejected_and_not_saved(app):
    bad = Settings.from_dict(app.settings.to_dict())
    bad.review.pass_required_n = 99
    with pytest.raises(ValueError):
        app.update_settings(bad)
    assert Settings.load(app.config_path).review.pass_required_n == 2


async def test_start_failure_cleans_up_and_stays_imported(app):
    def boom(path):
        raise RuntimeError('沒有麥克風')

    app._source_factory = boom
    app.import_patient('x')
    with pytest.raises(StateError, match='沒有麥克風'):
        await app.start_visit()
    assert app.phase == 'imported' and app.visit is None


async def test_display_conversion_never_touches_stored_text(app):
    app.import_patient('x')
    await app.start_visit()
    s = app.visit
    s.push_note('甲- 現病史：頭痛，咳嗽', '醫師手動')
    app.set_script_mode('simplified')
    assert app.display(s.note.current_note()) == '甲- 现病史：头痛，咳嗽'
    assert s.note.current_note() == '甲- 現病史：頭痛，咳嗽'
    app.set_script_mode('traditional')
    assert app.display('头痛、咳嗽') == '頭痛、咳嗽'
    app.set_script_mode('original')
    assert app.display('头痛') == '头痛'
    assert [e['mode'] for e in s.log.events if e['type'] == 'script_mode_changed'] == [
        'simplified', 'traditional', 'original']
    with pytest.raises(StateError):
        app.set_script_mode('klingon')
    await app.finish_visit()


async def test_app_level_finish_failure_is_recoverable(app, monkeypatch):
    app.import_patient('x')
    await app.start_visit()
    visit = app.visit
    real = visit.store.write_json
    state = {'failed': False}

    def flaky(rel, data):
        if rel == 'advice/index.json' and not state['failed']:
            state['failed'] = True
            raise OSError('disk full (simulated)')
        return real(rel, data)

    monkeypatch.setattr(visit.store, 'write_json', flaky)
    with pytest.raises(OSError):
        await app.finish_visit()
    assert app.phase == 'visiting' and visit.phase == 'finish_failed' and not app._busy_transition
    folder = await app.finish_visit()                                        # the same call again succeeds
    assert app.phase == 'none' and app.visit is None and (folder / 'log.md').exists()
