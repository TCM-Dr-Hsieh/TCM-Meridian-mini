"""Speaker marking on the screen: labels in the transcript, the status line, the speaker-groups window, the settings tab."""
import asyncio
import json

from nicegui import ui
from nicegui.testing import User

from mini.config import Settings
from mini.llm import LLMClient
from mini.state import AppState
from mini.ui.page import MainPage
from tests.helpers import FakeASR, FakeLLM, roles_reply
from tests.speaker_fakes import DOCTOR_LINE, PATIENT_LINE, FakeEmbedder, bases, conversation, scripted_source


def speaker_state(tmp_path, fake: FakeLLM, *, enabled=True, turns=40) -> AppState:
    settings = Settings()
    settings.visits_dir = str(tmp_path / 'visits')
    settings.speaker.enabled = enabled
    (tmp_path / 'config.json').write_text(json.dumps(settings.to_dict(), ensure_ascii=False), encoding='utf-8')
    templates = tmp_path / 'templates'
    (templates / 'defaults').mkdir(parents=True)
    (templates / 'defaults' / 'record_template.txt').write_text('甲- 現病史：\n', encoding='utf-8')
    (templates / 'defaults' / 'analysis_template.txt').write_text('一- 西醫診斷：\n', encoding='utf-8')
    audio, replies, _ = conversation(turns)
    state: AppState = AppState(
        config_path=tmp_path / 'config.json', templates_dir=templates, client=LLMClient(fake.transport()),
        asr=FakeASR(replies), llm_backoff=0,
        source_factory=lambda path: scripted_source(state.settings, audio, recording_path=path),
        speaker_embedder=lambda path: FakeEmbedder(bases(2), seed=3))
    return state


async def until(condition, timeout=20.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while not condition():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError('condition not reached in time')
        await asyncio.sleep(0.05)


def marked(user: User, marker: str):
    elements = list(user.find(marker=marker).elements)
    assert len(elements) == 1, f'{marker}: {len(elements)} elements'
    return elements[0]


def button(user: User, label: str):
    buttons = [e for e in user.find(label).elements if isinstance(e, ui.button)]
    assert len(buttons) == 1, f'{label}: {len(buttons)} buttons'
    return buttons[0]


async def started(user: User, tmp_path, fake=None, **kwargs):
    kwargs.setdefault('turns', 70)                                  # long enough for the role map to be confirmed (trusted)
    state = speaker_state(tmp_path, fake or FakeLLM(), **kwargs)

    @ui.page('/')
    def index():
        MainPage(state)

    state.import_patient('45歲男性')
    await state.start_visit()
    await user.open('/')
    await until(lambda: state.visit.pipeline.finished)
    return state


async def test_the_transcript_shows_who_speaks_and_the_status_line_says_marking_is_on(user: User, tmp_path):
    state = await started(user, tmp_path)
    await until(lambda: state.visit.speaker.trusted)
    await user.should_see('醫師：', retries=60)
    await user.should_see('患者或家屬：', retries=60)
    await user.should_see('說話者標記：已啟用', retries=60)
    html_text = marked_html(state)
    assert f'醫師：</span>{DOCTOR_LINE}' in html_text and f'患者或家屬：</span>{PATIENT_LINE}' in html_text
    await state.finish_visit()


def marked_html(state) -> str:
    """The transcript panel's HTML as the page builds it (rendered through the real page code)."""
    from mini.ui.page import MainPage as Page

    class Probe:
        app = state

        run_html = Page.run_html

    visit = state.visit
    return ''.join(Probe().run_html(run) for seg in visit.pipeline.segments if seg.kind == 'speech'
                   for run in (visit.speaker.runs(seg) or []))


async def test_the_speaker_groups_window_lists_the_groups_and_a_change_relabels_the_transcript(user: User, tmp_path):
    state = await started(user, tmp_path)
    await until(lambda: state.visit.speaker.voter.usable)
    await until(lambda: button(user, '說話者群').enabled)
    user.find('說話者群').click()
    await user.should_see(marker='speaker-group-0', retries=60)
    await user.should_see(marker='speaker-group-1', retries=60)
    service = state.visit.speaker
    doctor = next(g for g, r in service.tracker.roles.items() if r == 'doctor')
    other = next(g for g, r in service.tracker.roles.items() if r == 'other')
    before = [s for s in service.tracker.stats().items()]
    marked(user, f'speaker-role-{doctor}').set_value('other')                   # the physician disagrees with the model
    await until(lambda: service.tracker.roles.get(doctor) == 'other')
    assert service.tracker.roles[doctor] == 'other' and service.locked
    assert service.manual_events[-1]['event'] == 'roles'
    assert state.visit.pipeline.version > 0 and before
    await state.finish_visit()
    assert other in service.tracker.roles


async def test_there_is_no_speaker_button_when_marking_is_off(user: User, tmp_path):
    state = await started(user, tmp_path, enabled=False)
    await user.should_not_see(marker='speakers-button')
    await user.should_not_see('說話者標記')
    await state.finish_visit()


async def test_a_missing_model_is_reported_on_the_status_line_and_the_visit_goes_on(user: User, tmp_path):
    def missing(path):
        raise RuntimeError('找不到說話者聲紋模型')

    state = speaker_state(tmp_path, FakeLLM())
    state._speaker_embedder = missing

    @ui.page('/')
    def index():
        MainPage(state)

    state.import_patient('x')
    await state.start_visit()
    await user.open('/')
    await user.should_see('說話者標記未啟用：找不到說話者聲紋模型', retries=60)
    await user.should_not_see(marker='speakers-button')
    await until(lambda: state.visit.pipeline.finished)
    await user.should_see(DOCTOR_LINE, retries=60)                   # plain transcript, as before
    await state.finish_visit()


async def test_the_settings_tab_saves_the_speaker_options(user: User, tmp_path):
    state = speaker_state(tmp_path, FakeLLM(), enabled=False)

    @ui.page('/')
    def index():
        MainPage(state)

    await user.open('/')
    user.find('模型設定').click()
    await user.should_see('說話者標記', retries=60)
    user.find('說話者標記').click()                                         # the tab
    await user.should_see('啟用說話者標記', retries=60)
    switches = {str(e.text): e for e in user.find(kind=ui.switch).elements}
    switches['啟用說話者標記'].set_value(True)
    next(e for label, e in switches.items() if label.startswith('用 LLM 依上下文補標')).set_value(False)
    pct = next(e for e in user.find(kind=ui.number).elements if str(e.props.get('label', '')).startswith('「不明」百分位'))
    pct.set_value(25)
    user.find('儲存設定').click()
    await until(lambda: state.settings.speaker.enabled)
    assert state.settings.speaker.text_fill is False and state.settings.speaker.unknown_percentile == 25
    saved = json.loads(state.config_path.read_text(encoding='utf-8'))['speaker']
    assert saved['enabled'] is True and saved['text_fill'] is False and saved['use_in_jobs'] is True


async def test_a_bad_percentile_is_refused_with_a_message(user: User, tmp_path):
    state = speaker_state(tmp_path, FakeLLM(), enabled=False)

    @ui.page('/')
    def index():
        MainPage(state)

    await user.open('/')
    user.find('模型設定').click()
    user.find('說話者標記').click()
    await user.should_see('啟用說話者標記', retries=60)
    pct = next(e for e in user.find(kind=ui.number).elements if str(e.props.get('label', '')).startswith('「不明」百分位'))
    pct.set_value(90)
    user.find('儲存設定').click()
    await user.should_see('不可大於 50', retries=60)
    assert state.settings.speaker.unknown_percentile == 15.0


async def test_an_open_dropdown_is_not_rebuilt_while_the_numbers_behind_it_keep_changing(user: User, tmp_path):
    """The window refreshes itself; if it rebuilt the rows each time, a dropdown the physician is choosing from would close."""
    state = await started(user, tmp_path)
    await until(lambda: state.visit.speaker.voter.usable)
    await until(lambda: button(user, '說話者群').enabled)
    user.find('說話者群').click()
    await user.should_see(marker='speaker-role-0', retries=60)
    select = marked(user, 'speaker-role-0')
    before = state.visit.speaker.groups()[0]['units']
    await asyncio.sleep(1.5)                                                     # several refreshes while the numbers move
    assert not select.is_deleted and marked(user, 'speaker-role-0') is select
    assert state.visit.speaker.groups()[0]['units'] >= before
    await state.finish_visit()


async def test_a_role_changed_by_the_model_shows_in_the_dropdown_without_counting_as_a_click(user: User, tmp_path):
    state = await started(user, tmp_path)
    await until(lambda: state.visit.speaker.voter.usable)
    await until(lambda: button(user, '說話者群').enabled)
    user.find('說話者群').click()
    await user.should_see(marker='speaker-role-0', retries=60)
    service = state.visit.speaker
    doctor = next(g for g, r in service.tracker.roles.items() if r == 'doctor')
    service.tracker.set_roles({g: ('other' if g == doctor else 'doctor') for g in range(2)})   # as if the model had changed its mind
    await until(lambda: marked(user, f'speaker-role-{doctor}').value == 'other')
    assert not service.locked and service.manual_events == []                    # refreshing the widget did not count as the physician
    await state.finish_visit()


async def test_the_speaker_window_shows_but_does_not_change_the_roles_while_a_job_runs(user: User, tmp_path):
    fake = FakeLLM()
    state = await started(user, tmp_path, fake=fake)
    await until(lambda: state.visit.speaker.voter.usable)
    await until(lambda: button(user, '說話者群').enabled)
    fake.delay = 1.5
    job = state.visit.start_job('record')
    user.find('說話者群').click()
    await user.should_see(marker='speaker-role-0', retries=60)
    await until(lambda: not marked(user, 'speaker-role-0').enabled)             # the window is read-only while the job runs
    assert not marked(user, 'speaker-lock').enabled
    await user.should_see(marker='speaker-busy', retries=60)
    service = state.visit.speaker
    doctor = next(g for g, r in service.tracker.roles.items() if r == 'doctor')
    before = dict(service.tracker.roles)
    marked(user, f'speaker-role-{doctor}').set_value('other')                    # even a forced change is refused by the backend
    assert service.tracker.roles == before and service.manual_events == []
    await until(lambda: marked(user, f'speaker-role-{doctor}').value == 'doctor')   # and the window shows what is really in force
    fake.delay = 0.0
    await job.task
    await until(lambda: marked(user, 'speaker-role-0').enabled)
    await state.finish_visit()


async def test_a_group_nobody_named_shows_no_role_and_the_dropdown_offers_only_the_two_real_choices(user: User, tmp_path):
    fake = FakeLLM()

    def only_the_doctor(messages):
        answer = roles_reply(json.loads(messages[1]['content']))
        return {**answer, 'roles': {g: (r if r == '醫師' else '不明') for g, r in answer['roles'].items()}}

    fake.script['speaker_roles'] = [only_the_doctor] * 60
    state = await started(user, tmp_path, fake)
    service = state.visit.speaker
    await until(lambda: service.tracker.doctor_known and service.tracker.group_count == 2)
    await until(lambda: button(user, '說話者群').enabled)
    user.find('說話者群').click()
    await user.should_see(marker='speaker-group-0', retries=60)
    unnamed = next(g for g in range(2) if service.tracker.role_of(g) == 'unknown')
    select = marked(user, f'speaker-role-{unnamed}')
    await until(lambda: select.value is None)
    assert set(select.options) == {'doctor', 'other'}
    await state.finish_visit()


async def test_the_second_text_fill_switch_follows_the_first_one_and_is_saved(user: User, tmp_path):
    state = speaker_state(tmp_path, FakeLLM(), enabled=False)

    @ui.page('/')
    def index():
        MainPage(state)

    await user.open('/')
    user.find('模型設定').click()
    await user.should_see('說話者標記', retries=60)
    user.find('說話者標記').click()
    await user.should_see('啟用說話者標記', retries=60)
    second = marked(user, 'speaker-fill2')
    assert second.value is False and second.enabled                              # off by default; usable while the first fill is on
    first = next(e for e in user.find(kind=ui.switch).elements if str(e.text).startswith('用 LLM 依上下文補標'))
    second.set_value(True)
    user.find('儲存設定').click()
    await until(lambda: state.settings.speaker.text_fill2)
    assert json.loads(state.config_path.read_text(encoding='utf-8'))['speaker']['text_fill2'] is True
    first.set_value(False)                                                        # the second one cannot work without the first
    await until(lambda: not second.enabled)
    user.find('儲存設定').click()
    await until(lambda: not state.settings.speaker.text_fill)
    assert state.settings.speaker.text_fill2 is False
