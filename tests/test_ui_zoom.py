"""The enlarged panels (今日病歷, 分析與處置, 問診建議): one at a time, the small panels paused behind them, the enlarged ones
carry every function of the small ones (the advice one is read-only). NiceGUI's in-process user simulation, no browser."""
import asyncio
import inspect
import re

import pytest
from nicegui import ui
from nicegui.testing import User

from mini.record.tags import editable_text
from mini.ui import dialogs, style
from mini.ui.page import MainPage
from tests.helpers import FakeLLM
from tests.test_ui import build_state, until

KINDS = ('note', 'analysis', 'advice')
SMALL_BUTTONS = ('btn_browse', 'btn_diff', 'btn_edit', 'btn_undo', 'btn_redo', 'btn_copy_note',
                 'btn_at_prev', 'btn_at_next', 'btn_at_edit', 'btn_process', 'btn_copy_at',
                 'btn_ad_prev', 'btn_ad_next', 'btn_zoom_note', 'btn_zoom_analysis', 'btn_zoom_advice')


async def visiting_page(user: User, tmp_path, *, jobs=('record', 'analysis', 'advice', 'advice'), delay=0.0):
    """A page on a running visit that already has a note, an analysis and two advice versions."""
    fake = FakeLLM()
    state = build_state(tmp_path, fake)
    holder = {}

    @ui.page('/')
    def index():
        holder['page'] = MainPage(state)

    state.import_patient('王先生，45歲，高血壓')
    await state.start_visit()
    await until(lambda: state.visit.pipeline.finished)
    for kind in jobs:
        await state.visit.start_job(kind).task
    fake.delay = delay
    await user.open('/')
    page = holder['page']
    await until(lambda: page.w.btn_zoom_note.enabled)                                 # the page has caught up with the visit
    return state, page, fake


def shown(page, kind):
    z = page.z[kind]
    return z.root.visible


def paused(page):
    return [name for name in SMALL_BUTTONS if getattr(page.w, name).enabled]


async def test_each_panel_enlarges_alone_and_the_small_ones_are_paused_behind_it(user: User, tmp_path):
    state, page, _ = await visiting_page(user, tmp_path)
    assert page.zoom is None and not any(shown(page, k) for k in KINDS)
    assert page.w.btn_zoom_note.enabled and page.w.btn_zoom_analysis.enabled and page.w.btn_zoom_advice.enabled
    assert page.w.btn_browse.enabled and page.w.btn_at_edit.enabled                    # nothing is paused yet

    for kind in KINDS:
        with user.client:
            page.open_zoom(kind)
        assert page.zoom == kind
        assert [k for k in KINDS if shown(page, k)] == [kind]                          # one at a time
        await until(lambda: paused(page) == [])                                       # every button of the small panels waits
        assert page.z[kind].btn_zoom.enabled                                          # the way back
        with user.client:
            page.close_zoom()
        assert page.zoom is None and not shown(page, kind)
        await until(lambda: page.w.btn_zoom_note.enabled and page.w.btn_browse.enabled)
    await state.finish_visit()


async def test_a_second_enlarge_is_refused_while_one_is_open(user: User, tmp_path):
    state, page, _ = await visiting_page(user, tmp_path)
    with user.client:
        page.open_zoom('note')
        page.open_zoom('analysis')
        page.open_zoom('advice')
    assert page.zoom == 'note' and [k for k in KINDS if shown(page, k)] == ['note']
    with user.client:
        page.close_zoom()
    await state.finish_visit()


async def test_there_is_nothing_to_enlarge_without_a_visit_and_the_enlarged_panel_goes_when_the_visit_ends(
        user: User, tmp_path):
    state = build_state(tmp_path, FakeLLM())
    holder = {}

    @ui.page('/')
    def index():
        holder['page'] = MainPage(state)

    await user.open('/')
    page = holder['page']
    assert not (page.w.btn_zoom_note.enabled or page.w.btn_zoom_analysis.enabled or page.w.btn_zoom_advice.enabled)
    with user.client:
        page.open_zoom('note')
    assert page.zoom is None and not shown(page, 'note')                                # no patient, no visit

    state.import_patient('x')
    await state.start_visit()
    await until(lambda: page.w.btn_zoom_note.enabled)
    with user.client:
        page.open_zoom('advice')
    assert shown(page, 'advice')
    await state.finish_visit()
    await until(lambda: page.zoom is None)                                              # the visit is over
    assert not shown(page, 'advice')
    await until(lambda: not page.w.btn_zoom_advice.enabled)


async def test_the_enlarged_panels_show_the_same_content_and_follow_new_results(user: User, tmp_path):
    state, page, _ = await visiting_page(user, tmp_path, jobs=('record', 'analysis', 'advice'))
    with user.client:
        page.open_zoom('note')
    z = page.z['note']
    assert z.note_html.content == page.w.note_html.content and '患者頭痛三天' in z.note_html.content
    assert z.note_version.text == page.w.note_version.text
    with user.client:
        page.set_note_mode('diff')                                                      # the mode is shared: both follow
    assert z.note_html.content == page.w.note_html.content and 'diff-box' in z.note_html.content
    with user.client:
        page.set_note_mode('browse')
    version = len(state.visit.note.snapshots)
    await state.visit.start_job('record').task                                           # a job result appears in the open panel
    await until(lambda: z.note_version.text.startswith(f'版本 {state.visit.note.current_index + 1}/'))
    assert len(state.visit.note.snapshots) > version and z.note_html.content == page.w.note_html.content
    with user.client:
        page.close_zoom()

    with user.client:
        page.open_zoom('analysis')
    a = page.z['analysis']
    assert a.at_md.content == page.w.at_md.content and '偏頭痛' in a.at_md.content and a.at_label.text == '1/1'
    assert a.btn_process.enabled and a.btn_copy_at.enabled and a.btn_at_edit.enabled
    with user.client:
        page.close_zoom()
    await state.finish_visit()


async def test_the_enlarged_advice_panel_is_read_only_and_pages_through_the_versions(user: User, tmp_path):
    state, page, _ = await visiting_page(user, tmp_path)                                 # two advice versions
    visit = state.visit
    assert len(visit.advice) == 2 and visit.advice_index == 1
    with user.client:
        page.open_zoom('advice')
    z = page.z['advice']
    assert z.ad_label.text == '2/2' and not z.btn_ad_next.enabled and z.btn_ad_prev.enabled
    assert set(z.advice_md) == {'western_ddx', 'tcm_ddx', 'next_questions'}
    for key, md in z.advice_md.items():
        assert md.content == page.w.advice_md[key].content
    assert not hasattr(z, 'note_editor') and not hasattr(z, 'at_editor') and not hasattr(z, 'btn_edit')
    with user.client:
        page.nav_advice(-1)
    assert visit.advice_index == 0
    await until(lambda: z.ad_label.text == '1/2' and page.w.ad_label.text == '1/2')
    await until(lambda: z.btn_ad_next.enabled and not z.btn_ad_prev.enabled)
    with user.client:
        page.nav_advice(1)
        page.close_zoom()
    await state.finish_visit()


async def test_the_note_is_edited_in_the_enlarged_panel_which_cannot_close_meanwhile(user: User, tmp_path):
    state, page, _ = await visiting_page(user, tmp_path)
    visit = state.visit
    before = len(visit.note.snapshots)
    with user.client:
        page.open_zoom('note')
        page.enter_note_edit()
    z = page.z['note']
    assert page.note_mode == 'edit' and z.note_editor.visible and not z.note_body.visible
    assert z.note_editor.value == editable_text(visit.note.current_note())
    assert not page.w.note_editor.visible and page.w.note_body.visible                  # behind it: the note as it stands
    assert page.w.btn_edit.visible and not page.w.btn_edit_done.visible and not page.w.btn_edit_cancel.visible
    assert z.btn_edit_done.visible and z.btn_edit_cancel.visible and not z.btn_edit.visible
    assert page.zoom_locked()
    await until(lambda: not z.btn_zoom.enabled)                                        # 縮小 waits for 完成 / 放棄
    with user.client:
        page.close_zoom()
        page.open_zoom('analysis')
    assert page.zoom == 'note'
    z.note_editor.set_value(editable_text(visit.note.current_note()) + '\n丙- 備註：高血壓服藥中')
    with user.client:
        page.leave_note_edit(True)
    assert len(visit.note.snapshots) == before + 1 and '高血壓服藥中' in visit.note.current_note()
    assert page.note_mode == 'browse' and not page.zoom_locked()
    await until(lambda: z.btn_zoom.enabled and '高血壓服藥中' in z.note_html.content)
    assert '高血壓服藥中' in page.w.note_html.content
    with user.client:
        page.close_zoom()
    assert page.zoom is None
    await state.finish_visit()


async def test_discarding_an_edit_in_the_enlarged_panel_keeps_the_note_and_frees_the_panel(user: User, tmp_path):
    state, page, _ = await visiting_page(user, tmp_path)
    visit = state.visit
    before = visit.note.current_note()
    with user.client:
        page.open_zoom('note')
        page.enter_note_edit()
    page.z['note'].note_editor.set_value('完全不同的內容')
    with user.client:
        page.leave_note_edit(False)
    assert visit.note.current_note() == before and not page.zoom_locked()
    with user.client:
        page.close_zoom()
    assert page.zoom is None
    await state.finish_visit()


async def test_an_edit_still_open_in_the_enlarged_panel_is_dropped_when_the_visit_ends(user: User, tmp_path):
    state, page, _ = await visiting_page(user, tmp_path)
    with user.client:
        page.open_zoom('note')
        page.enter_note_edit()
    assert page.zoom_locked()
    await state.finish_visit()                                                          # (the end button itself waits for the edit)
    await until(lambda: page.zoom is None)
    assert page.note_mode == 'browse' and not page.zoom_locked() and not shown(page, 'note')


async def test_an_analysis_edit_still_open_in_the_enlarged_panel_is_dropped_when_the_visit_ends(user: User, tmp_path):
    state, page, _ = await visiting_page(user, tmp_path)
    with user.client:
        page.open_zoom('analysis')
        page.enter_analysis_edit()
    assert page.zoom_locked()
    await state.finish_visit()
    await until(lambda: page.zoom is None)
    assert not page.analysis_editing and not page.zoom_locked() and not shown(page, 'analysis')


def press_escape(user: User):
    """What the browser does on Esc (the page's JS emits 'mini_zoom_escape'), through the page's own listener."""
    layout = user.client.layout
    ids = [listener.id for listener in layout._event_listeners.values() if listener.type == 'mini_zoom_escape']
    assert len(ids) == 1, 'the page listens for the Esc event'
    with user.client:
        layout._handle_event({'listener_id': ids[0], 'args': {}})


async def test_escape_puts_the_enlarged_panel_back_unless_it_is_being_edited(user: User, tmp_path):
    state, page, _ = await visiting_page(user, tmp_path)
    press_escape(user)                                                                  # nothing enlarged: nothing happens
    assert page.zoom is None
    with user.client:
        page.open_zoom('analysis')
    press_escape(user)
    assert page.zoom is None and not shown(page, 'analysis')
    with user.client:
        page.open_zoom('note')
        page.enter_note_edit()
    press_escape(user)
    assert page.zoom == 'note' and shown(page, 'note')                                  # editing: 完成 / 放棄 first
    with user.client:
        page.leave_note_edit(False)
    press_escape(user)
    assert page.zoom is None
    await state.finish_visit()


async def test_the_analysis_is_edited_in_the_enlarged_panel(user: User, tmp_path):
    state, page, _ = await visiting_page(user, tmp_path)
    visit = state.visit
    with user.client:
        page.open_zoom('analysis')
        page.enter_analysis_edit()
    z = page.z['analysis']
    assert page.analysis_editing and z.at_editor.visible and not z.at_body.visible
    assert not page.w.at_editor.visible and page.w.at_body.visible
    assert page.w.btn_at_edit.visible and not page.w.btn_at_save.visible and not page.w.btn_at_cancel.visible
    assert z.btn_at_save.visible and z.btn_at_cancel.visible and not z.btn_at_edit.visible
    assert page.zoom_locked()
    with user.client:
        page.close_zoom()
    assert page.zoom == 'analysis'
    z.at_editor.set_value('一- 西醫診斷：緊縮型頭痛')
    with user.client:
        page.leave_analysis_edit(True)
    assert len(visit.analysis) == 2 and visit.analysis[visit.analysis_index].source != 'llm'
    assert '緊縮型頭痛' in visit.analysis[visit.analysis_index].final_at
    await until(lambda: '緊縮型頭痛' in z.at_md.content and '緊縮型頭痛' in page.w.at_md.content)
    with user.client:
        page.close_zoom()
    await state.finish_visit()


async def test_an_edit_in_the_small_panel_keeps_every_enlarge_button_off(user: User, tmp_path):
    state, page, _ = await visiting_page(user, tmp_path)
    with user.client:
        page.enter_note_edit()
    await until(lambda: not page.w.btn_zoom_note.enabled)
    assert not page.w.btn_zoom_analysis.enabled and not page.w.btn_zoom_advice.enabled
    with user.client:
        page.open_zoom('note')
    assert page.zoom is None
    with user.client:
        page.leave_note_edit(False)
    await until(lambda: page.w.btn_zoom_note.enabled)
    await state.finish_visit()


async def test_the_toolbar_keeps_working_and_a_running_job_does_not_pause_the_way_back(user: User, tmp_path):
    state, page, fake = await visiting_page(user, tmp_path, delay=0.5)
    with user.client:
        page.open_zoom('note')
    await until(lambda: page.w.btn_write.enabled)                                      # 病歷書寫 stays on the toolbar
    job = state.visit.start_job('record')
    await until(lambda: not page.w.btn_write.enabled)
    assert page.z['note'].btn_zoom.enabled                                             # a running job does not lock it
    with user.client:
        page.close_zoom()
    assert page.zoom is None
    await job.task
    await state.finish_visit()


async def test_a_new_page_starts_without_an_enlarged_panel(user: User, tmp_path):
    state, page, _ = await visiting_page(user, tmp_path)
    with user.client:
        page.open_zoom('advice')
    other = {}

    @ui.page('/second')
    def second():
        other['page'] = MainPage(state)

    await user.open('/second')                                                          # another tab / a reload
    assert other['page'].zoom is None and not any(shown(other['page'], k) for k in KINDS)
    assert page.zoom == 'advice'                                                        # the first one is its own
    with user.client:
        page.close_zoom()
    await state.finish_visit()


class Recorder:
    """Stands in for the clipboard and for the 教授過程 dialog: what was copied, how often the dialog was opened."""

    def __init__(self, monkeypatch):
        self.copied, self.opened = [], 0

        async def copy(text, what):
            self.copied.append(what)

        def open_process(app):
            self.opened += 1

        monkeypatch.setattr(dialogs, 'copy_to_clipboard', copy)
        monkeypatch.setattr(dialogs, 'open_process_dialog', open_process)


def panel_actions(page):
    """Every function of each small panel's buttons that a click can reach (the enlarged panel shares them)."""
    return {
        'note': [lambda: page.set_note_mode('diff'), page.enter_note_edit, page.undo_note, page.redo_note, page.copy_note,
                 lambda: page.leave_note_edit(False)],
        'analysis': [lambda: page.nav_analysis(-1), page.enter_analysis_edit, page.copy_analysis, page.open_process,
                     lambda: page.leave_analysis_edit(False)],
        'advice': [lambda: page.nav_advice(-1)],
    }


async def call(action):
    result = action()
    if inspect.isawaitable(result):
        await result


def snapshot(page, visit, recorder):
    return (page.note_mode, page.analysis_editing, page.zoom, visit.note.current_index, len(visit.note.snapshots),
            visit.analysis_index, len(visit.analysis), visit.advice_index, len(visit.advice),
            tuple(recorder.copied), recorder.opened)


async def two_of_everything(user, tmp_path):
    """A visit with three note versions (the middle one shown: both 上一版 and 下一版 move), two analyses and two advices."""
    state, page, _ = await visiting_page(user, tmp_path, jobs=('record', 'record', 'analysis', 'analysis', 'advice', 'advice'))
    state.visit.undo_note()
    await until(lambda: page.w.btn_redo.enabled)
    visit = state.visit
    assert (visit.note.current_index, len(visit.note.snapshots), visit.analysis_index, visit.advice_index) == (1, 3, 1, 1)
    return state, page


@pytest.mark.parametrize('enlarged', KINDS)
async def test_clicks_on_their_way_to_a_paused_panel_do_nothing_whatever_they_do(user: User, tmp_path, monkeypatch, enlarged):
    """The handlers themselves refuse an action of a panel that is paused behind another panel's enlarged window, whatever
    it does: nothing changes, nothing is copied, no dialog opens (the second line; the clicks of a disabled button never get
    this far, see the test through the real click path below)."""
    recorder = Recorder(monkeypatch)
    state, page = await two_of_everything(user, tmp_path)
    with user.client:
        page.open_zoom(enlarged)
    before = snapshot(page, state.visit, recorder)
    for kind, actions in panel_actions(page).items():
        if kind == enlarged:
            continue
        for action in actions:
            with user.client:
                await call(action)
    assert snapshot(page, state.visit, recorder) == before
    assert page.zoom == enlarged
    with user.client:
        page.close_zoom()
    await state.finish_visit()


@pytest.mark.parametrize('kind,enlarged', [('note', 'analysis'), ('note', 'advice'), ('analysis', 'note'), ('analysis', 'advice')])
async def test_an_edit_cannot_be_finished_or_discarded_from_behind_another_enlarged_panel(user: User, tmp_path, kind, enlarged):
    """完成 / 放棄 / 儲存 are actions too: with the edit open and another panel enlarged (forced here: the buttons cannot get
    into this state), they change nothing; once the guard no longer applies the same call saves."""
    state, page = await two_of_everything(user, tmp_path)
    visit = state.visit
    saved = ((lambda: '被改過的版本' in visit.note.current_note()) if kind == 'note'
             else (lambda: '被改過的版本' in visit.analysis[visit.analysis_index].final_at))     # (a note edit after a rollback replaces later versions)
    editing = (lambda: page.note_mode == 'edit') if kind == 'note' else (lambda: page.analysis_editing)
    leave = page.leave_note_edit if kind == 'note' else page.leave_analysis_edit
    with user.client:
        (page.enter_note_edit if kind == 'note' else page.enter_analysis_edit)()
    editor = page.w.note_editor if kind == 'note' else page.w.at_editor
    editor.set_value('甲- 現病史：一個被改過的版本')
    assert not saved()
    page.zoom = enlarged                                                                # forced: open_zoom would refuse
    with user.client:
        leave(True)
        leave(False)
    assert editing() and not saved() and page.zoom == enlarged
    page.zoom = None
    with user.client:
        leave(True)
    assert not editing() and saved()
    await state.finish_visit()


BUTTONS = {'note': ('btn_browse', 'btn_diff', 'btn_edit', 'btn_undo', 'btn_redo', 'btn_copy_note'),
           'analysis': ('btn_at_prev', 'btn_at_next', 'btn_at_edit', 'btn_process', 'btn_copy_at'),
           'advice': ('btn_ad_prev', 'btn_ad_next')}


async def click(button):
    """A click as the browser sends it: through the client's own event dispatch, which drops the events of a disabled element."""
    listener = next(item for item in button._event_listeners.values() if item.type == 'click')
    button.client.handle_event({'id': button.id, 'listener_id': listener.id, 'args': []})
    await asyncio.sleep(0.05)                                                           # (the copy handlers are coroutines)


@pytest.mark.parametrize('enlarged', KINDS)
async def test_a_click_on_a_button_paused_behind_an_enlarged_panel_does_nothing(user: User, tmp_path, monkeypatch, enlarged):
    """The first line, through the real click path: the server disables the buttons the moment a panel is enlarged (before it
    handles the next event), and the client's event dispatch drops the clicks of a disabled button. Checked once with the
    handlers' guard switched off: this test still passes, only the direct-call test above fails."""
    recorder = Recorder(monkeypatch)
    state, page = await two_of_everything(user, tmp_path)
    with user.client:
        page.open_zoom(enlarged)
    before = snapshot(page, state.visit, recorder)
    for names in BUTTONS.values():
        for name in names:
            await click(getattr(page.w, name))                                         # the small panels, all three
    for name in ('btn_zoom_note', 'btn_zoom_analysis', 'btn_zoom_advice'):
        await click(getattr(page.w, name))
    assert snapshot(page, state.visit, recorder) == before and page.zoom == enlarged
    with user.client:
        page.close_zoom()
    await state.finish_visit()


async def test_the_buttons_of_the_enlarged_panel_reach_the_actions(user: User, tmp_path, monkeypatch):
    recorder = Recorder(monkeypatch)
    state, page = await two_of_everything(user, tmp_path)
    visit = state.visit
    with user.client:
        page.open_zoom('analysis')
    z = page.z['analysis']
    await click(z.btn_at_prev)
    assert visit.analysis_index == 0
    await click(z.btn_process)
    await click(z.btn_copy_at)
    assert recorder.opened == 1 and recorder.copied == ['分析與處置']
    await click(z.btn_at_edit)
    assert page.analysis_editing and page.zoom_locked()
    await click(z.btn_zoom)                                                             # 縮小 is disabled while editing
    assert page.zoom == 'analysis'
    await click(z.btn_at_cancel)
    assert not page.analysis_editing
    await click(z.btn_zoom)
    assert page.zoom is None
    await state.finish_visit()


@pytest.mark.parametrize('kind', KINDS)
@pytest.mark.parametrize('enlarged', [False, True])
async def test_the_actions_of_a_panel_work_in_the_small_panel_and_in_its_own_enlarged_panel(
        user: User, tmp_path, monkeypatch, kind, enlarged):
    recorder = Recorder(monkeypatch)
    state, page = await two_of_everything(user, tmp_path)
    visit = state.visit
    if enlarged:
        with user.client:
            page.open_zoom(kind)
        assert page.zoom == kind
    with user.client:
        if kind == 'note':
            page.set_note_mode('diff')
            assert page.note_mode == 'diff'
            page.set_note_mode('browse')
            page.undo_note()
            assert visit.note.current_index == 0
            page.redo_note()
            page.redo_note()
            assert visit.note.current_index == 2
            await page.copy_note()
            assert recorder.copied == ['今日病歷（已去來源標籤）']
            page.enter_note_edit()
            assert page.note_mode == 'edit'
            page.leave_note_edit(False)
            assert page.note_mode == 'browse'
        elif kind == 'analysis':
            page.nav_analysis(-1)
            assert visit.analysis_index == 0
            page.nav_analysis(1)
            assert visit.analysis_index == 1
            await page.copy_analysis()
            assert recorder.copied == ['分析與處置']
            page.open_process()
            assert recorder.opened == 1
            page.enter_analysis_edit()
            assert page.analysis_editing
            page.leave_analysis_edit(False)
            assert not page.analysis_editing
        else:
            page.nav_advice(-1)
            assert visit.advice_index == 0
            page.nav_advice(1)
            assert visit.advice_index == 1
        if enlarged:
            page.close_zoom()
    await state.finish_visit()


def css_value(selector: str, prop: str) -> int:
    block = re.search(re.escape(selector) + r'\s*\{([^}]*)\}', style.CSS).group(1)
    return int(re.search(prop + r':\s*(\d+)', block).group(1))


def test_the_layers_are_the_patient_panel_above_the_enlarged_panels_above_the_columns():
    zoom, patient = css_value('.zoom-panel', 'z-index'), css_value('.float-panel', 'z-index')
    assert 0 < zoom < patient < 6000                                                   # (Quasar dialogs and messages are above)
    assert 'mini_zoom_escape' in style.JS and "Escape" in style.JS
    assert '.q-dialog' in style.JS                                                      # Esc belongs to an open dialog first


@pytest.mark.parametrize('kind,side', [('note', 'left'), ('advice', 'left'), ('analysis', 'right')])
async def test_the_enlarged_panels_cover_the_columns_the_user_named(user: User, tmp_path, kind, side):
    """今日病歷 and 問診建議 lie over the left and middle column, 分析與處置 over the middle and right column."""
    state, page, _ = await visiting_page(user, tmp_path)
    classes = page.z[kind].root.classes
    assert 'zoom-panel' in classes and side in classes
    assert re.search(r'\.zoom-panel\.left\s*\{grid-column:1 / span 2', style.CSS)
    assert re.search(r'\.zoom-panel\.right\s*\{grid-column:2 / span 2', style.CSS)
    await state.finish_visit()
