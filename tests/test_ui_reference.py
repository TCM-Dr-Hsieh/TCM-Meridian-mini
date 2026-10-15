"""The note edit: the box without tags, the read-only 參考版本 window beside it, and the line that says which tags the save
would drop. NiceGUI's in-process user simulation, no browser."""
from nicegui.testing import User

from mini.record.tags import MANUAL_TAG, editable_text
from mini.ui.page import reference_html
from tests.test_ui import until
from tests.test_ui_zoom import visiting_page


def test_the_reference_html_shows_every_tag_in_light_colour_and_escapes_the_text():
    html = reference_html('甲- 現病史：<頭痛>三天[語音#1]\n乙- 加的[醫師手動]')
    assert '&lt;頭痛&gt;' in html and '<頭痛>' not in html
    assert '<span class="src-tag">[語音#1]</span>' in html
    assert '<span class="src-tag manual">[醫師手動]</span>' in html
    assert '\n' in html                                                     # the line breaks stay (pre-wrap)
    assert 'empty' in reference_html('  ')                                  # an empty note says so


async def test_the_reference_window_opens_with_the_edit_closes_with_it_and_the_box_has_no_tags(user: User, tmp_path):
    state, page, _ = await visiting_page(user, tmp_path, jobs=('record',))
    note = state.visit.note.current_note()
    assert '[語音#1]' in note and not page.reference_open
    with user.client:
        page.enter_note_edit()
    assert page.reference_open and page.w.ref_panel.visible
    assert page.w.note_editor.value == editable_text(note) and '[語音#' not in page.w.note_editor.value
    assert '[語音#1]' in page.w.ref_html.content and 'src-tag' in page.w.ref_html.content
    assert page.w.ref_title.text.startswith('參考：版本') and '修改前' in page.w.ref_title.text
    assert page.w.note_preview.visible
    with user.client:
        page.leave_note_edit(False)
    assert not page.reference_open and not page.w.ref_panel.visible and not page.w.note_preview.visible
    await state.finish_visit()


async def test_a_reference_window_closed_by_the_physician_stays_closed_for_that_edit_only(user: User, tmp_path):
    state, page, _ = await visiting_page(user, tmp_path, jobs=('record',))
    with user.client:
        page.enter_note_edit()
        page.close_reference()
    assert not page.reference_open and page.reference_dismissed
    with user.client:
        page.sync_reference()                                               # tick: it must not pop up again
    assert not page.reference_open
    with user.client:
        page.leave_note_edit(False)
    assert not page.reference_dismissed
    with user.client:
        page.enter_note_edit()
    assert page.reference_open                                              # the next edit opens it again
    with user.client:
        page.leave_note_edit(False)
    await state.finish_visit()


async def test_the_reference_shows_the_version_as_it_was_when_the_edit_began(user: User, tmp_path):
    state, page, _ = await visiting_page(user, tmp_path, jobs=('record',))
    with user.client:
        page.enter_note_edit()
    shown = page.w.ref_html.content
    page.w.note_editor.set_value('完全不同的內容')
    assert page._preview_dirty                                              # a change of the box marks the preview stale
    with user.client:
        page.update_edit_preview()
    assert page.w.ref_html.content == shown                                 # it is the reference, not a mirror of the box
    with user.client:
        page.leave_note_edit(False)
    await state.finish_visit()


async def test_the_preview_lists_the_tags_the_save_would_drop_and_save_drops_exactly_those(user: User, tmp_path):
    state, page, _ = await visiting_page(user, tmp_path, jobs=('record',))
    visit = state.visit
    with user.client:
        page.enter_note_edit()
    assert '不會移除' in page.w.note_preview.text and 'warn' not in page.w.note_preview.classes
    page.w.note_editor.set_value('乙- 完全不同的句子')
    assert page._preview_dirty
    with user.client:
        page.update_edit_preview()                                          # tick does this
    assert '[語音#1]' in page.w.note_preview.text and 'warn' in page.w.note_preview.classes
    assert [r.tag for r in visit.preview_note_edit('乙- 完全不同的句子').removed] == ['[語音#1]']
    with user.client:
        page.leave_note_edit(True)
    assert visit.note.current_note() == '乙- 完全不同的句子' + MANUAL_TAG
    assert not page.reference_open
    await state.finish_visit()


async def test_the_reference_window_goes_when_the_visit_ends_during_an_edit(user: User, tmp_path):
    state, page, _ = await visiting_page(user, tmp_path, jobs=('record',))
    with user.client:
        page.open_zoom('note')
        page.enter_note_edit()                                              # an edit in the enlarged panel is dropped with the visit
    assert page.reference_open
    await state.finish_visit()
    await until(lambda: page.note_mode == 'browse')
    await until(lambda: not page.reference_open)
    assert not page.w.ref_panel.visible


async def test_a_save_over_a_newer_version_and_a_typed_tag_are_refused_on_the_page(user: User, tmp_path):
    state, page, _ = await visiting_page(user, tmp_path, jobs=('record',))
    visit = state.visit
    with user.client:
        page.enter_note_edit()
    base_text = visit.note.current_note()
    page.w.note_editor.set_value(editable_text(base_text) + '[語音#9]')                 # a tag pasted from the reference window
    with user.client:
        page.update_edit_preview()
    assert '[語音#9]' in page.w.note_preview.text and 'warn' in page.w.note_preview.classes
    with user.client:
        page.leave_note_edit(True)
    assert page.note_mode == 'edit' and visit.note.current_note() == base_text             # refused, the box stays open
    page.w.note_editor.set_value(editable_text(base_text) + '，加重')
    visit.push_note(base_text + '[醫師手動]', '醫師手動')                                  # the note moved on (another tab) meanwhile
    newer = visit.note.current_note()
    with user.client:
        page.update_edit_preview()
    assert '已被更新' in page.w.note_preview.text
    with user.client:
        page.leave_note_edit(True)
    assert page.note_mode == 'edit' and visit.note.current_note() == newer                 # not overwritten
    with user.client:
        page.leave_note_edit(False)                                                        # 放棄, then edit again from the new version
        page.enter_note_edit()
    page.w.note_editor.set_value(editable_text(newer) + '，加重')
    with user.client:
        page.leave_note_edit(True)
    assert visit.note.current_note() != newer and page.note_mode == 'browse'
    await state.finish_visit()
