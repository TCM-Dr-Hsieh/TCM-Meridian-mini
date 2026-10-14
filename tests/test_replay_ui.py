"""The replay window's wiring in the page: rows carry the visit's token, and the player is closed when the visit shown changes."""
from nicegui import ui
from nicegui.testing import User

from mini.ui.page import MainPage
from tests.helpers import FakeLLM
from tests.test_ui import RecordedJavascript, build_state, until

CLOSE = 'window.miniReplay && window.miniReplay.close()'


async def open_visit(user, tmp_path, monkeypatch):
    js = RecordedJavascript()
    monkeypatch.setattr(ui, 'run_javascript', js)
    state = build_state(tmp_path, FakeLLM())
    holder = {}

    @ui.page('/')
    def index():
        holder['page'] = MainPage(state)

    state.import_patient('x')
    await state.start_visit()
    await user.open('/')
    page = holder['page']
    await until(lambda: state.visit.pipeline.finished and 'data-token' in page.w.tx_html.content)
    return state, page, js


async def test_transcript_rows_are_clickable_and_carry_the_visit_token(user: User, tmp_path, monkeypatch):
    state, page, js = await open_visit(user, tmp_path, monkeypatch)
    content = page.w.tx_html.content
    token = state.visit.replay_token
    assert f'data-token="{token}"' in content
    assert 'class="seg-no"' in content and 'data-start="' in content and 'data-idx="1"' in content
    assert page._transcript_visit == token
    await state.finish_visit()


async def test_the_player_closes_when_the_visit_ends(user: User, tmp_path, monkeypatch):
    state, page, js = await open_visit(user, tmp_path, monkeypatch)
    assert CLOSE not in js.calls
    await state.finish_visit()
    await until(lambda: CLOSE in js.calls)
    assert page._transcript_visit is None


async def test_the_player_closes_when_another_visit_replaces_the_shown_one_without_an_empty_state_between(
        user: User, tmp_path, monkeypatch):
    """A background tab whose timer was throttled never renders the 'no visit' state between two visits: the page last drew
    visit A and the next render finds visit B."""
    state, page, js = await open_visit(user, tmp_path, monkeypatch)
    page._transcript_visit = 'token-of-the-visit-drawn-before'
    js.calls.clear()
    page.render_transcript()
    assert CLOSE in js.calls
    assert page._transcript_visit == state.visit.replay_token
    js.calls.clear()
    page.render_transcript()                                   # the same visit again: nothing to close
    assert CLOSE not in js.calls
    await state.finish_visit()
