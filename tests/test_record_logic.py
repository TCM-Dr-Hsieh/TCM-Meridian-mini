import pytest

from mini.record.diff import build_diff_context, build_diff_md, diff_html, unified_diff
from mini.record.line_ops import OperationsError, apply_operations, number_lines, parse_operations
from mini.record.snapshots import SnapshotHistory
from mini.record.tags import MANUAL_TAG, strip_citations, tag_human_edits

BASE = 'A\nB\nC'


# --- line operations -------------------------------------------------------
def test_operations_use_original_line_numbers():
    ok, text, _, failures = apply_operations(BASE, [
        {'op': 'delete', 'line': 1}, {'op': 'replace', 'line': 2, 'content': 'B2'},
        {'op': 'insert', 'line': 3, 'content': 'X'}, {'op': 'insert', 'line': 3, 'content': 'Y'}])
    assert ok and not failures
    assert text == 'B2\nX\nY\nC'


def test_insert_past_the_end_is_an_append():
    ok, text, logs, _ = apply_operations(BASE, [{'op': 'insert', 'line': 99, 'content': 'Z'}])
    assert ok and text == 'A\nB\nC\nZ'
    assert any('NORMALIZE' in line for line in logs)


def test_consecutive_inserts_past_the_end_keep_the_order_they_were_listed_in():
    """What the writer prompt promises: line 2,3,5 on a 1-line note is tolerated and stays in order, while an
    out-of-range replace/delete rejects the whole batch (the prompt no longer claims inserts are rejected)."""
    ok, text, _, failures = apply_operations('一', [{'op': 'insert', 'line': 2, 'content': 'a'},
                                                   {'op': 'insert', 'line': 3, 'content': 'b'},
                                                   {'op': 'insert', 'line': 5, 'content': 'c'}])
    assert ok and not failures and text == '一\na\nb\nc'
    ok, text, _, failures = apply_operations('一', [{'op': 'replace', 'line': 3, 'content': 'x'}])
    assert not ok and text == '一' and '超出範圍' in failures[0]


@pytest.mark.parametrize('ops', [
    [{'op': 'delete', 'line': 9}],
    [{'op': 'replace', 'line': 0, 'content': 'x'}],
    [{'op': 'frob', 'line': 1}],
    [{'op': 'delete', 'line': 1}, {'op': 'replace', 'line': 1, 'content': 'x'}],
    [{'op': 'insert', 'line': 'a', 'content': 'x'}],
])
def test_invalid_batch_is_rejected_as_a_whole(ops):
    ok, text, _, failures = apply_operations(BASE, ops + [{'op': 'insert', 'line': 1, 'content': 'ok'}])
    assert not ok and failures and text == BASE


def test_empty_document_fills_instead_of_leaving_a_blank_line():
    ok, text, _, _ = apply_operations('', [{'op': 'insert', 'line': 1, 'content': 'one'},
                                           {'op': 'insert', 'line': 1, 'content': 'two'}])
    assert ok and text == 'one\ntwo'
    ok, text, _, _ = apply_operations('', [{'op': 'insert', 'line': 2, 'content': 'tail'}])
    assert ok and text == 'tail'


def test_parse_operations_shape_checks():
    assert parse_operations({'operations': []}) == []
    for bad in (None, [], {'operations': 'x'}, {'operations': [1]}, {'operations': [{'op': 'insert', 'line': 1}]},
                {'operations': [{'op': 'insert', 'line': 1, 'content': 'a\nb'}]}):
        with pytest.raises(OperationsError):
            parse_operations(bad)


def test_number_lines():
    assert number_lines('a\nb').splitlines()[1].strip().startswith('2 |')
    assert '空白' in number_lines('')


# --- tags --------------------------------------------------------------------
def test_manual_tag_is_added_once_and_only_to_changed_lines():
    old = '甲- 現病史：頭痛[語音#1]\n乙- 過去病史：'
    new = '甲- 現病史：頭痛[語音#1]\n乙- 過去病史：高血壓\n丙- 個人史：'
    tagged = tag_human_edits(old, new)
    lines = tagged.split('\n')
    assert lines[0].endswith('[語音#1]') and MANUAL_TAG not in lines[0]
    assert lines[1].endswith(MANUAL_TAG) and lines[2].endswith(MANUAL_TAG)
    assert tag_human_edits(tagged, tagged.replace('高血壓', '糖尿病')).count(MANUAL_TAG) == 2   # not stacked


def test_a_repeated_manual_tag_records_that_the_line_was_edited_again():
    """As in the original project: only a tag already at the END is not repeated. A line the AI updated reads
    `…[醫師手動][語音#8]` (tag in the middle), so when the physician edits it again the tag repeats on purpose --
    the repetition marks the re-edit, and the 病歷修改 diff 過程 shows what changed each time."""
    updated = '辛.9- 大便：偏乾，兩天一次[醫師手動][語音#8]'
    edited = tag_human_edits(updated, '辛.9- 大便：偏乾，三天一次[醫師手動][語音#8]')     # the edit box shows raw tags
    assert edited == '辛.9- 大便：偏乾，三天一次[醫師手動][語音#8][醫師手動]'
    again = tag_human_edits(edited, edited.replace('三天', '四天'))                    # tag now last: not stacked further
    assert again.count(MANUAL_TAG) == 2 and again.endswith('[語音#8][醫師手動]')
    stripped = tag_human_edits(updated, '辛.9- 大便：偏乾，三天一次')                  # the physician deleted the tags
    assert stripped == '辛.9- 大便：偏乾，三天一次[醫師手動]'
    assert tag_human_edits(edited, edited + '\n壬- 脈象：弦') == edited + '\n壬- 脈象：弦[醫師手動]'   # unchanged lines untouched


def test_consecutive_manual_edits_keep_one_tag_so_the_tag_count_is_not_the_edit_count():
    note = ''
    for text in ('辛.9- 大便：一日三次', '辛.9- 大便：一日兩次', '辛.9- 大便：一日一次'):
        note = tag_human_edits(note, text + ('[醫師手動]' if note else ''))      # the edit box shows the raw tag
    assert note == '辛.9- 大便：一日一次[醫師手動]'                                 # three edits, still one tag
    ai_updated = note.replace('一日一次', '一日一次[語音#8]').replace('[語音#8][醫師手動]', '[醫師手動][語音#8]')
    assert tag_human_edits(ai_updated, ai_updated.replace('一日一次', '兩日一次')).count(MANUAL_TAG) == 2


def test_the_history_diff_shows_each_edit_of_a_repeatedly_edited_line_and_browsing_hides_every_tag():
    first = '辛.9- 大便：一日三次[醫師手動]'
    second = '辛.9- 大便：偏乾，兩天一次[醫師手動][語音#8]'                       # AI update, tag kept + new source
    third = tag_human_edits(second, '辛.9- 大便：偏乾，三天一次[醫師手動][語音#8]')     # physician edits again
    history = [snap('', 'init'), snap(first, '醫師手動'), snap(second, '病歷書寫 #1'), snap(third, '醫師手動')]
    text = build_diff_context(history)
    assert '+' + first in text and '-' + first in text and '+' + second in text and '-' + second in text
    assert '+' + third in text and third.count(MANUAL_TAG) == 2
    assert text.index('版本 3（病歷書寫 #1') < text.index('版本 4（醫師手動')                # the order of the edits is visible
    assert strip_citations(third) == '辛.9- 大便：偏乾，三天一次'                       # all repeated tags are hidden


def test_blank_lines_are_not_tagged():
    assert MANUAL_TAG not in tag_human_edits('a', 'a\n\n')


def test_strip_citations_only_removes_known_tags():
    text = '頭痛 [語音#12-14][歷史] 與 [Lab] 血壓[醫師手動]'
    assert strip_citations(text) == '頭痛 與 [Lab] 血壓'


# --- history -------------------------------------------------------------------
def test_history_undo_redo_and_truncation():
    h = SnapshotHistory()
    for i, text in enumerate(['', 'v2', 'v3']):
        h.push(text, 'init' if i == 0 else f's{i}')
    assert h.current_index == 2 and h.can_undo() and not h.can_redo()
    assert h.undo()['note'] == 'v2' and h.can_redo()
    truncated = h.push('v2b', '醫師手動')
    assert [s['note'] for s in truncated] == ['v3']
    assert [s['note'] for s in h.snapshots] == ['', 'v2', 'v2b'] and not h.can_redo()


def test_history_serialization_roundtrip():
    h = SnapshotHistory()
    h.push('a', 'init', t=0.0)
    h.push('b', 's', t=2.5, meta={'job_id': 'J1'})
    h.undo()
    other = SnapshotHistory()
    other.restore(h.to_dict())
    assert other.current_index == 0 and other.snapshots[1]['meta'] == {'job_id': 'J1'}


# --- diff ---------------------------------------------------------------------
def test_diff_helpers():
    assert 'diff-add' in diff_html('a', 'a\nb') and 'diff-none' in diff_html('a', 'a')
    assert '+b' in unified_diff('a', 'a\nb')
    md = build_diff_md([{'note': 'a', 'source': 'init', 'timestamp': 't0', 't': 0.0, 'meta': {}},
                        {'note': 'a\nb', 'source': '病歷書寫 #1', 'timestamp': 't1', 't': 5.0,
                         'meta': {'job_id': 'J001', 'review': {'rounds': 2, 'passes': 2}, 'call_ids': ['c0001']}}], 1)
    assert '版本 2' in md and '```diff' in md and 'J001' in md and '通過 2 次' in md


def snap(note, source, stamp='2026-10-05T09:00:00'):
    return {'note': note, 'source': source, 'timestamp': stamp, 't': None, 'meta': {}}


def test_diff_context_for_the_prompts_matches_the_original_projects_format():
    assert build_diff_context([]) == '（無病歷版本歷史）' and build_diff_context(None) == '（無病歷版本歷史）'
    assert '只有一個版本' in build_diff_context([snap('', 'init')])
    history = [snap('', 'init'), snap('甲：頭痛', '病歷書寫 #1', '2026-10-05T09:01:00'),
               snap('甲：頭痛\n乙：高血壓[醫師手動]', '醫師手動', '2026-10-05T09:02:00')]
    text = build_diff_context(history)
    assert '### 版本 1 -> 版本 2：NOTE' in text and '### 版本 2 -> 版本 3：NOTE' in text
    assert '```diff' in text and '+甲：頭痛' in text and '+乙：高血壓[醫師手動]' in text
    assert '版本 2（病歷書寫 #1, 2026-10-05T09:01:00）' in text and '版本 3（醫師手動, 2026-10-05T09:02:00）' in text
    assert text.index('版本 1 -> 版本 2') < text.index('版本 2 -> 版本 3')            # chronological


def test_diff_context_stops_at_the_version_being_edited_and_skips_identical_versions():
    history = [snap('', 'init'), snap('a', 's1'), snap('a\nb', 's2')]
    undone = build_diff_context(history, current_index=1)                          # the physician undid version 3
    assert '版本 1 -> 版本 2' in undone and '版本 2 -> 版本 3' not in undone and '+b' not in undone
    assert '沒有內容差異' in build_diff_context([snap('a', 'init'), snap('a', 'same')])
    assert build_diff_context(history, current_index=99) == build_diff_context(history)   # clamped, never an error


def test_a_repeated_tag_appears_whenever_anything_follows_the_tag_not_only_an_ai_source():
    """The physician typing after the tag (no AI source involved) repeats it too, so a repeat only means that the
    tag was not last when the line was edited; it is not a count of edits."""
    assert tag_human_edits('A[醫師手動]', 'A[醫師手動]補充') == 'A[醫師手動]補充[醫師手動]'
    assert tag_human_edits('A[醫師手動]', 'B[醫師手動]') == 'B[醫師手動]'            # tag stayed last: still one
