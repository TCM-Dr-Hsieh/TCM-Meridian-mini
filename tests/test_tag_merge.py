"""The note edit box shows no source tags; on save the program puts them back and takes away those whose words are gone
(`merge_human_edit`). Pure text logic first, then what a physician edit leaves in the version history."""
import pytest

from mini.record.diff import build_diff_md
from mini.record.tags import (MANUAL_TAG, NoteEditError, editable_text, merge_human_edit, tokenize,
                              typed_tags)

EXAMPLE = '胃口佳[語音#1]。輕微脹氣[語音#2]。大便1天2次，偏軟。[語音#3]'


def tags_of(merge):
    return [r.tag for r in merge.removed]


def test_the_edit_box_text_has_no_tags_and_an_unchanged_save_changes_nothing():
    clean = editable_text(EXAMPLE)
    assert clean == '胃口佳。輕微脹氣。大便1天2次，偏軟。'
    merge = merge_human_edit(EXAMPLE, clean)
    assert merge.text == EXAMPLE and merge.removed == ()


def test_the_example_of_the_physician_the_tag_of_the_deleted_words_goes_the_others_stay():
    edited = editable_text(EXAMPLE).replace('。大便1天2次，偏軟', '，胃食道逆流，大便偏硬')
    assert edited == '胃口佳。輕微脹氣，胃食道逆流，大便偏硬。'
    merge = merge_human_edit(EXAMPLE, edited)
    # "大便" was typed again, but two letters are chance, not the old words surviving
    assert merge.text == '胃口佳[語音#1]。輕微脹氣[語音#2]，胃食道逆流，大便偏硬。' + MANUAL_TAG
    assert tags_of(merge) == ['[語音#3]'] and merge.removed[0].text == '。大便1天2次，偏軟。'


def test_unchanged_lines_keep_their_tags_exactly_and_only_the_changed_line_is_marked():
    old = '甲- 現病史：頭痛三天 [語音#1]\n高血壓[歷史]\n丙- 個人史：不抽菸[語音#4]'
    edited = editable_text(old).replace('高血壓', '糖尿病')
    merge = merge_human_edit(old, edited)
    lines = merge.text.split('\n')
    assert lines[0] == '甲- 現病史：頭痛三天 [語音#1]' and lines[2] == '丙- 個人史：不抽菸[語音#4]'      # even the blank before the tag
    assert lines[1] == '糖尿病' + MANUAL_TAG
    assert tags_of(merge) == ['[歷史]']


def test_no_field_name_is_assumed_a_word_left_standing_keeps_the_tag_even_if_it_was_a_label():
    # templates differ: whatever stays counts as words. The safe direction: the tag is kept too long, never lost.
    old = '辛.9- 大便：一日三次[語音#8]'
    left = merge_human_edit(old, '辛.9- 大便：偏乾')
    assert left.removed == () and '[語音#8]' in left.text
    # a line with no label at all: everything rewritten means the tag goes
    assert tags_of(merge_human_edit('大便一日三次[語音#8]', '偏乾')) == ['[語音#8]']
    # a colon inside the sentence is nothing special either
    kept = merge_human_edit('患者說：頭痛三天[語音#2]', '患者說：頭痛四天')
    assert kept.removed == () and kept.text == '患者說：頭痛四天[語音#2]' + MANUAL_TAG


def test_words_that_survive_keep_the_tag_even_when_a_part_is_rewritten():
    old = '大便1天2次，偏軟[語音#3]'
    merge = merge_human_edit(old, '大便1天3次，偏軟')
    assert merge.text == '大便1天3次，偏軟[語音#3]' + MANUAL_TAG and merge.removed == ()


def test_a_short_span_that_was_only_extended_keeps_its_tag():
    merge = merge_human_edit('口乾[語音#5]', '口乾多飲')
    assert merge.text == '口乾[語音#5]多飲' + MANUAL_TAG and merge.removed == ()          # the new words come after the tag


def test_text_typed_right_after_a_sentence_is_not_covered_by_that_sentences_tag():
    merge = merge_human_edit('胃口佳[語音#1]。', '胃口佳，嘔吐。')
    assert merge.text == '胃口佳[語音#1]，嘔吐。' + MANUAL_TAG and merge.removed == ()


def test_tags_written_one_after_another_cover_the_same_words_and_go_together():
    old = '大便偏乾[語音#1][語音#2]。睡眠差[語音#3]'
    untouched = merge_human_edit(old, '大便偏乾。睡眠好')                      # words of the first group survive
    assert untouched.text.startswith('大便偏乾[語音#1][語音#2]。') and tags_of(untouched) == ['[語音#3]']
    gone = merge_human_edit(old, '食慾差。睡眠差')                            # the words of the first group are gone
    assert tags_of(gone) == ['[語音#1]', '[語音#2]'] and '[語音#3]' in gone.text


def test_tags_with_no_words_between_them_to_begin_with_are_not_removed():
    old = '胃口佳[語音#1]。[語音#2]'
    merge = merge_human_edit(old, '胃口差。')
    assert tags_of(merge) == ['[語音#1]']                                    # "胃口佳" is gone
    assert '[語音#2]' in merge.text                                           # its span was only a full stop


def test_a_deleted_line_takes_its_tags_along_and_they_are_reported():
    old = '甲- 現病史：頭痛三天[語音#1]\n乙- 個人史：不抽菸[語音#4]'
    merge = merge_human_edit(old, '甲- 現病史：頭痛三天')
    assert merge.text == '甲- 現病史：頭痛三天[語音#1]'
    assert [(r.tag, r.text) for r in merge.removed] == [('[語音#4]', '乙- 個人史：不抽菸')]


def test_a_new_line_gets_the_manual_tag_and_blank_lines_stay_blank():
    merge = merge_human_edit('甲- 現病史：頭痛三天[語音#1]', '甲- 現病史：頭痛三天\n\n乙- 醫師加的')
    assert merge.text == '甲- 現病史：頭痛三天[語音#1]\n\n乙- 醫師加的' + MANUAL_TAG


def test_the_manual_tag_is_not_listed_as_removed_nor_stacked():
    old = '大便偏乾[醫師手動]'
    merge = merge_human_edit(old, '大便偏硬，三天一次')
    assert merge.text == '大便偏硬，三天一次' + MANUAL_TAG and merge.removed == ()
    again = merge_human_edit(merge.text, editable_text(merge.text).replace('三天', '四天'))
    assert again.text.count(MANUAL_TAG) == 1


def test_a_tag_typed_or_pasted_into_the_edit_box_is_refused_not_stored_as_a_real_one():
    for typed in ('[語音#9]', '[歷史]', '[醫師手動]', '[語音#3-5]', '[語音#2, #4]'):
        for text in ('胃口佳' + typed, typed + '胃口佳', '胃口佳 ' + typed + '。'):
            with pytest.raises(NoteEditError) as error:
                merge_human_edit('胃口佳[語音#1]', text)
            assert typed in str(error.value)
    with pytest.raises(NoteEditError):
        merge_human_edit('', '新的一行[語音#1]')                                # also into an empty note
    assert merge_human_edit('胃口佳[語音#1]', '胃口佳[Lab]').text == '胃口佳[語音#1][Lab]' + MANUAL_TAG    # other brackets are plain text
    assert typed_tags('甲[語音#1]乙[語音#1][歷史]') == ['[語音#1]', '[歷史]']


def test_every_manual_tag_stays_even_when_the_words_before_them_are_gone():
    old = '舊內容[醫師手動][語音#8][醫師手動]'
    merge = merge_human_edit(old, '新內容')
    assert merge.text == '新內容[醫師手動][醫師手動]'                               # nothing merged, nothing de-duplicated
    assert tags_of(merge) == ['[語音#8]']                                       # only the source tag goes with its words
    both = merge_human_edit('甲[醫師手動]乙[醫師手動]', '甲丙')
    assert both.text.count(MANUAL_TAG) >= 2 and both.removed == ()


def test_windows_line_breaks_from_the_browser_do_not_change_the_lines():
    old = '甲- 現病史：頭痛三天[語音#1]\n乙- 個人史：不抽菸[語音#4]'
    assert merge_human_edit(old, editable_text(old).replace('\n', '\r\n')).text == old


def test_the_whole_note_rewritten_loses_every_tag_and_empty_to_text_has_none_to_lose():
    old = '甲- 現病史：頭痛三天[語音#1]\n乙- 個人史：不抽菸[語音#4]'
    merge = merge_human_edit(old, '完全不同的內容')
    assert tags_of(merge) == ['[語音#1]', '[語音#4]'] and merge.text == '完全不同的內容' + MANUAL_TAG
    assert merge_human_edit('', '第一行').text == '第一行' + MANUAL_TAG


def test_tokenize_splits_text_and_tags():
    assert tokenize('甲[語音#1]乙[歷史][醫師手動]') == [('甲', False), ('[語音#1]', True), ('乙', False), ('[歷史]', True),
                                                       ('[醫師手動]', True)]
    assert tokenize('[Lab] 血壓') == [('[Lab] 血壓', False)]


async def test_a_physician_edit_records_the_removed_tags_in_the_version_history(make_session):
    h = await make_session()
    s = h.session
    s.push_note('甲- 現病史：患者頭痛三天[語音#1]。無發燒[語音#2]', 'init')
    assert s.preview_note_edit('甲- 現病史：患者頭痛三天。胃食道逆流').removed[0].tag == '[語音#2]'      # preview, nothing saved
    assert s.note.current_note().endswith('[語音#2]')
    assert s.edit_note_manual('甲- 現病史：患者頭痛三天。胃食道逆流') is True
    assert s.note.current_note() == '甲- 現病史：患者頭痛三天[語音#1]。胃食道逆流' + MANUAL_TAG
    snapshot = s.note.get_current()
    assert snapshot['meta']['removed_tags'] == [{'tag': '[語音#2]', 'text': '。無發燒'}]
    event = [e for e in s.log.events if e['type'] == 'note_manual_edit'][-1]
    assert event['removed_tags'] == snapshot['meta']['removed_tags']
    md = build_diff_md(s.note.snapshots, s.note.current_index)
    assert '移除來源標籤：[語音#2]（原涵蓋：。無發燒）' in md
    # an edit that removes nothing leaves no such entry
    assert s.edit_note_manual('甲- 現病史：患者頭痛三天。胃食道逆流，加重') is True
    assert 'removed_tags' not in s.note.get_current()['meta']


def test_the_threshold_decides_how_many_left_over_letters_keep_a_tag():
    old = '大便偏乾，三天一次[語音#1]'
    edited = '大便偏乾，一日兩回'                                                   # "大便偏乾" (4 letters) stays, the rest is rewritten
    assert merge_human_edit(old, edited).removed == ()                               # default 3: four left over keep the tag
    assert merge_human_edit(old, edited, 4).removed == ()
    assert tags_of(merge_human_edit(old, edited, 5)) == ['[語音#1]']                  # stricter: four are not enough
    # the physician's example: "大便" typed again; with 1 even two left over letters keep [語音#3]
    again = editable_text(EXAMPLE).replace('。大便1天2次，偏軟', '，胃食道逆流，大便偏硬')
    assert tags_of(merge_human_edit(EXAMPLE, again)) == ['[語音#3]']
    assert merge_human_edit(EXAMPLE, again, 1).removed == ()


async def test_the_visit_uses_the_threshold_of_its_settings_for_the_preview_and_the_save(make_session):
    text = '甲- 現病史：大便偏乾，三天一次[語音#1]'
    edited = '甲- 現病史：大便偏乾，一日兩回'
    strict = await make_session(note_edit__min_surviving_run=9)
    strict.session.push_note(text, 'init')
    assert strict.session.preview_note_edit(edited).removed[0].tag == '[語音#1]'
    assert strict.session.edit_note_manual(edited) is True
    assert strict.session.note.get_current()['meta']['removed_tags'][0]['tag'] == '[語音#1]'
    easy = await make_session(note_edit__min_surviving_run=1)
    easy.session.push_note(text, 'init')
    assert easy.session.preview_note_edit(edited).removed == ()
    assert easy.session.edit_note_manual(edited) is True
    assert '[語音#1]' in easy.session.note.current_note()


async def test_a_typed_tag_is_refused_by_the_preview_and_the_save_and_nothing_is_written(make_session):
    h = await make_session()
    s = h.session
    s.push_note('甲- 現病史：患者頭痛三天[語音#1]', 'init')
    before = (s.note.current_note(), s.note.current_index)
    for call in (s.preview_note_edit, s.edit_note_manual):
        with pytest.raises(NoteEditError, match=r'\[語音#9\]'):
            call('甲- 現病史：患者頭痛三天[語音#9]')
    assert (s.note.current_note(), s.note.current_index) == before
    assert not [e for e in s.log.events if e['type'] == 'note_manual_edit']


async def test_an_edit_that_began_on_an_older_version_cannot_overwrite_a_newer_one(make_session):
    h = await make_session()
    s = h.session
    s.push_note('甲- 現病史：頭痛三天[語音#1]', 'init')
    base = s.note_version_id()
    assert base and s.edit_note_manual('甲- 現病史：頭痛三天，加重', base) is True       # the version is still the one the edit began on
    stale = base                                                                          # another tab saved: the version moved on
    newer = s.note.current_note()
    for call in (s.preview_note_edit, s.edit_note_manual):
        with pytest.raises(NoteEditError, match='已被更新'):
            call('完全不同的內容', stale)
    assert s.note.current_note() == newer                                                # nothing was overwritten
    s.undo_note()                                                                          # an undo moves the version as well
    with pytest.raises(NoteEditError):
        s.edit_note_manual('又一個內容', s.note.get_current()['id'] + 'x')
    assert s.edit_note_manual('甲- 現病史：頭痛三天，三天後好轉', s.note_version_id()) is True
    assert s.edit_note_manual('甲- 現病史：頭痛三天，三天後好轉') is False               # no base: not pinned (old callers), no change
