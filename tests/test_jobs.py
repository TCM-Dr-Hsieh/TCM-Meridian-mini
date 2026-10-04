import asyncio

import pytest

from mini.jobs import BusyError

PASS = {'pass': True, 'issues': [], 'comment': 'ok'}
FAIL = {'pass': False, 'issues': [{'line': 1, 'category': 'B-1', 'quote': '頭痛三天', 'problem': '逐字稿 #1 沒有',
                                    'fix_hint': '改成未知'}], 'comment': '有問題'}


def ops(*items, summary='s'):
    return {'thinking': 't', 'operations': list(items), 'summary': summary}


def insert(content, line=1):
    return {'op': 'insert', 'line': line, 'content': content}


def replace(line, content):
    return {'op': 'replace', 'line': line, 'content': content}


# ============================ 病歷書寫 =========================================
async def test_record_job_writes_after_n_passes(make_session):
    h = await make_session()
    job = await h.run_job('record')
    s = h.session
    assert job.status == 'succeeded', job.message
    assert s.note.current_index == 1 and s.note.current_note().startswith('甲- 現病史')
    snap = s.note.get_current()
    assert snap['source'] == '病歷書寫 #1' and snap['meta']['review'] == {'skipped': False, 'rounds': 2,
                                                                          'passes': 2, 'required': 2}
    assert len(h.fake.calls_for('writer')) == 1 and len(h.fake.calls_for('reviewer')) == 2
    writer_user = h.fake.calls_for('writer')[0][1]['content']
    assert '[#1 ' in writer_user and '45歲男性' in writer_user and '甲- 現病史：' in writer_user   # transcript+patient+template
    assert [e['type'] for e in s.log.events].count('review_result') == 2


async def test_review_failure_sends_issues_back_and_passes_accumulate_across_rewrites(make_session):
    h = await make_session()
    h.fake.queue('reviewer', PASS, FAIL, PASS)          # pass, fail -> rewrite, pass => 2 cumulative passes
    h.fake.queue('writer', ops(insert('甲- 現病史：頭痛[語音#1]')), ops(replace(1, '甲- 現病史：頭痛三天[語音#1]')))
    job = await h.run_job('record')
    assert job.status == 'succeeded', job.message
    assert len(h.fake.calls_for('writer')) == 2 and len(h.fake.calls_for('reviewer')) == 3
    second_writer = h.fake.calls_for('writer')[1][1]['content']
    assert '歷次審查意見與被退件內容' in second_writer and '逐字稿 #1 沒有' in second_writer
    assert h.session.note.current_note() == '甲- 現病史：頭痛三天[語音#1]'
    assert h.session.note.get_current()['meta']['review']['rounds'] == 3


async def test_a_rewrite_edits_the_version_the_reviewer_rejected_not_the_base(make_session):
    """Original behaviour: the rewrite's 今日病歷（附行號） is the rejected version, so line numbers in the reviewer's
    opinion match what the writer sees and an earlier fix can never be undone by the next rewrite."""
    h = await make_session()
    h.fake.queue('reviewer', FAIL, PASS, PASS)
    h.fake.queue('writer', ops(insert('甲- 現病史：頭痛[語音#1]'), insert('乙- 過去病史：高血壓[歷史]', line=2)),
                 ops(replace(1, '甲- 現病史：頭痛三天[語音#1]')))        # touches line 1 only; line 2 must survive
    job = await h.run_job('record')
    assert job.status == 'succeeded', job.message
    first, second = (c[1]['content'] for c in h.fake.calls_for('writer'))
    assert '（空白，尚無任何內容' in first                                      # first write: the empty base
    note_section = second[second.index('## 【今日病歷（附行號）】'):second.index('## 【病歷模板】')]
    assert '1 | 甲- 現病史：頭痛[語音#1]' in note_section and '2 | 乙- 過去病史：高血壓[歷史]' in note_section
    assert h.session.note.current_note() == '甲- 現病史：頭痛三天[語音#1]\n乙- 過去病史：高血壓[歷史]'
    writer_events = [e for e in h.session.log.events if e['type'] == 'writer_ops']
    assert [e['rewrite_of_rejected'] for e in writer_events] == [0, 1]       # logged: which write was a rewrite


async def test_every_round_of_opinions_and_every_rejected_version_goes_back_to_the_writer(make_session):
    """The failure that motivated this: with only the latest opinion the writer fixed A, re-broke B, fixed B,
    re-broke A... Now round 3's prompt carries rounds 1 and 2 as well (opinion + the version they rejected)."""
    h = await make_session()

    def complaint(quote, problem):
        return {'pass': False, 'issues': [{'line': 1, 'category': 'D-1', 'quote': quote, 'problem': problem,
                                           'fix_hint': 'x'}], 'comment': 'c'}

    h.fake.queue('reviewer', complaint('無藥物/食物過敏', '意見甲：沒有區分過敏種類'), complaint('感冒3天', '意見乙：沒有診斷名稱'),
                 PASS, PASS)
    h.fake.queue('writer', ops(insert('甲- 現病史：無藥物/食物過敏，感冒3天[語音#1]')),
                 ops(replace(1, '甲- 現病史：否認過敏，感冒3天[語音#1]')), ops(replace(1, '甲- 現病史：否認過敏，頭痛3天[語音#1]')))
    job = await h.run_job('record')
    assert job.status == 'succeeded', job.message
    third = h.fake.calls_for('writer')[2][1]['content']
    assert '意見甲：沒有區分過敏種類' in third and '意見乙：沒有診斷名稱' in third        # both opinions, not just the last
    assert '[第 1 次被退件的病歷內容' in third and '[第 2 次被退件的病歷內容' in third
    history = third[third.index('## 【歷次審查意見與被退件內容】'):]
    assert history.index('無藥物/食物過敏，感冒3天') < history.index('否認過敏，感冒3天')   # oldest first, full content each
    assert h.session.note.current_note() == '甲- 現病史：否認過敏，頭痛3天[語音#1]'      # both fixes survive
    reviewer_prompt = h.fake.calls_for('reviewer')[-1][1]['content']
    assert '1 | 甲- 現病史：否認過敏，頭痛3天[語音#1]' in reviewer_prompt              # the whole candidate, numbered
    assert '本次修改 diff' not in reviewer_prompt                                  # ... and no "what changed" hint
    histories = {c[1]['content'][c[1]['content'].index('## 【病歷修改 diff 過程】'):
                                 c[1]['content'].index('## 【即將登載的今日病歷')] for c in h.fake.calls_for('reviewer')}
    assert len(histories) == 1                                                     # same history every round, as the original


async def test_exceeding_max_review_rounds_fails_closed(make_session):
    h = await make_session(review__max_review_rounds=3)
    h.fake.queue('reviewer', FAIL, FAIL, FAIL, FAIL)
    job = await h.run_job('record')
    assert job.status == 'failed' and '審查未通過' in job.message and '已達最大審查輪數 3' in job.message
    assert h.session.note.current_index == 0 and h.session.note.current_note() == ''    # nothing written
    assert len(h.fake.calls_for('reviewer')) == 3                                       # no wasted extra review
    assert any(e['type'] == 'job_failed' for e in h.session.log.events)


async def test_n_zero_skips_review_and_says_so(make_session):
    h = await make_session(review__pass_required_n=0)
    job = await h.run_job('record')
    assert job.status == 'succeeded' and '未審查' in job.message
    assert h.fake.calls_for('reviewer') == []
    snap = h.session.note.get_current()
    assert snap['source'] == '病歷書寫 #1（未審查）' and snap['meta']['review']['skipped'] is True
    assert any(e['type'] == 'review_skipped' for e in h.session.log.events)


async def test_persistent_invalid_output_exhausts_retries_and_writes_nothing(make_session):
    h = await make_session()
    h.fake.queue('writer', *[ops({'op': 'delete', 'line': 99})] * 6)     # line 99 does not exist
    job = await h.run_job('record')
    assert job.status == 'failed' and h.session.note.current_index == 0
    writer_calls = [e for e in h.session.log.events if e['type'] == 'llm_call' and e['agent'] == 'record_writer']
    assert len(writer_calls) == 4 and all(c['error'] for c in writer_calls)           # 1 try + 3 retries, all logged


async def test_reviewer_outage_is_fail_closed(make_session):
    h = await make_session()
    h.fake.queue('reviewer', *['not json'] * 6)
    job = await h.run_job('record')
    assert job.status == 'failed' and h.session.note.current_index == 0
    assert len(h.fake.calls_for('writer')) == 1                    # no rewrite storm against a broken reviewer


async def test_transient_http_errors_are_retried(make_session):
    h = await make_session()
    h.fake.queue('writer', 'HTTP500', 'HTTP500')
    job = await h.run_job('record')
    assert job.status == 'succeeded' and len(h.fake.calls_for('writer')) == 3


async def test_an_empty_operation_list_skips_the_reviewer_but_a_valid_noop_is_still_reviewed(make_session):
    """Original: `if not operations` skips review. A valid operation whose result equals the note is NOT empty, so
    the reviewer still runs; once it passes nothing new is written (the original never pushes an identical note)."""
    h = await make_session()
    await h.run_job('record')                                                   # version 2 exists
    note = h.session.note.current_note()
    h.fake.calls.clear()
    h.fake.queue('writer', ops(replace(1, note.splitlines()[0])))              # replace line 1 with itself
    job = await h.run_job('record')
    assert job.status == 'succeeded', job.message
    assert len(h.fake.calls_for('reviewer')) == 2                               # reviewed n=2 times despite no change
    assert h.session.note.current_index == 1 and h.session.note.current_note() == note   # no duplicate version
    assert '內容與目前版本相同' in job.message and '未建立新版本' in job.message


async def test_a_valid_noop_with_n_zero_is_neither_reviewed_nor_pushed(make_session):
    h = await make_session(review__pass_required_n=0)
    await h.run_job('record')
    note = h.session.note.current_note()
    h.fake.queue('writer', ops(replace(1, note.splitlines()[0])))
    job = await h.run_job('record')
    assert job.status == 'succeeded' and '內容與目前版本相同' in job.message and '未審查' in job.message
    assert h.fake.calls_for('reviewer') == [] and h.session.note.current_index == 1


async def test_a_no_change_answer_creates_no_version(make_session):
    h = await make_session()
    h.fake.queue('writer', ops())
    job = await h.run_job('record')
    assert job.status == 'succeeded' and '無需更新' in job.message and '沒有經過審查員的遺漏檢查' in job.message
    assert h.session.note.current_index == 0 and h.fake.calls_for('reviewer') == []


async def test_incremental_write_edits_lines_in_place(make_session):
    h = await make_session()
    await h.run_job('record')                                       # line 1 written
    h.fake.queue('writer', ops({'op': 'replace', 'line': 1, 'content': '甲- 現病史：頭痛三天，無發燒[語音#1-2]'},
                               insert('乙- 過去病史：高血壓[歷史]', line=2)))
    job = await h.run_job('record')
    assert job.status == 'succeeded', job.message
    assert h.session.note.current_note().split('\n') == ['甲- 現病史：頭痛三天，無發燒[語音#1-2]',
                                                         '乙- 過去病史：高血壓[歷史]']
    assert h.session.note.get_current()['source'] == '病歷書寫 #2'
    numbered = h.fake.calls_for('writer')[1][1]['content']
    assert '  1 | 甲- 現病史：患者頭痛三天[語音#1]' in numbered          # second round sees the current note with line numbers


async def test_undo_then_ai_write_truncates_redo_and_audits_it(make_session):
    h = await make_session()
    await h.run_job('record')
    h.session.edit_note_manual('甲- 現病史：頭痛三天[語音#1]\n乙- 醫師加的')
    assert h.session.note.current_index == 2
    h.session.undo_note()
    h.session.undo_note()
    assert h.session.note.current_index == 0
    h.fake.queue('writer', ops(insert('甲- 現病史：由回滾後重寫[語音#1]')))
    await h.run_job('record')
    s = h.session
    assert [x['source'] for x in s.note.snapshots] == ['init', '病歷書寫 #2']
    assert s.note_audit and len(s.note_audit[0]['snapshots']) == 2                 # both discarded versions are preserved
    assert any(e['type'] == 'note_truncate_redo' for e in s.log.events)


async def test_manual_edit_is_tagged_and_blocked_while_a_job_runs(make_session):
    h = await make_session()
    await h.run_job('record')
    assert h.session.edit_note_manual(h.session.note.current_note() + '\n丙- 個人史：不抽菸') is True
    lines = h.session.note.current_note().split('\n')
    assert lines[-1].endswith('[醫師手動]') and not lines[0].endswith('[醫師手動]')
    assert h.session.edit_note_manual(h.session.note.current_note()) is False
    h.fake.delay = 0.3
    job = h.session.start_job('record')
    await asyncio.sleep(0.05)
    with pytest.raises(BusyError):
        h.session.edit_note_manual('x')
    with pytest.raises(BusyError):
        h.session.undo_note()
    with pytest.raises(BusyError):
        h.session.start_job('advice')
    await job.task


async def test_job_can_be_cancelled_without_leaving_anything_behind(make_session):
    h = await make_session()
    h.fake.delay = 5
    job = h.session.start_job('record')
    await asyncio.sleep(0.1)
    h.session.jobs.cancel()
    await asyncio.wait_for(asyncio.shield(job.task), 5)
    assert job.status == 'cancelled' and h.session.note.current_index == 0
    assert any(e['type'] == 'job_cancelled' for e in h.session.log.events)
    assert not h.session.jobs.busy


# ============================ 問診建議 =========================================
async def test_advice_job_creates_versions_with_navigation(make_session):
    h = await make_session()
    await h.run_job('record')
    h.fake.queue('advice', {'western_ddx': ['偏頭痛', '緊張型頭痛'], 'tcm_ddx': '肝陽上亢', 'next_questions': '疼痛位置？'})
    job = await h.run_job('advice')
    assert job.status == 'succeeded'
    v = h.session.advice[0]
    assert v.western_ddx.startswith('- 偏頭痛') and v.tcm_ddx == '肝陽上亢'
    prompt = h.fake.calls_for('advice')[0][1]['content']
    assert '45歲男性' in prompt and '頭痛三天' in prompt and '[語音#' not in prompt        # patient + note, tags stripped
    assert '[#1 ' not in prompt                                                          # the transcript is NOT an input
    await h.run_job('advice')
    s = h.session
    assert len(s.advice) == 2 and s.advice_index == 1
    s.set_advice_index(0)
    assert s.advice_index == 0
    assert (s.store.folder / 'advice' / '001.json').exists() and (s.store.folder / 'advice' / '002.md').exists()


async def test_advice_failure_after_retries_creates_no_version(make_session):
    h = await make_session()
    h.fake.queue('advice', *[{'western_ddx': 'x'}] * 6)
    job = await h.run_job('advice')
    assert job.status == 'failed' and h.session.advice == []


# ============================ 整體分析 =========================================
async def test_analysis_runs_five_calls_in_three_stages(make_session):
    h = await make_session()
    h.fake.delay = 0.05
    await h.run_job('record')
    job = await h.run_job('analysis')
    assert job.status == 'succeeded', job.message
    roles = [r for r, _ in h.fake.calls if r in ('analysis', 'cross', 'arbitration')]
    assert roles == ['analysis', 'analysis', 'cross', 'cross', 'arbitration']
    assert h.fake.max_active == 2                                                         # A and B truly ran in parallel
    s = h.session
    v = s.analysis[0]
    assert v.source == 'llm' and '偏頭痛' in v.final_at and v.arbitration_notes.startswith('兩位教授')
    assert set(v.calls) == {'a', 'b', 'review_a_on_b', 'review_b_on_a', 'c'}
    a_prompt, b_prompt = [m for r, m in h.fake.calls if r == 'analysis']
    assert '嚴謹、保守' in a_prompt[0]['content'] and '溫和' in b_prompt[0]['content']      # distinct styles
    assert '教授甲' in a_prompt[0]['content'] and '教授乙' in b_prompt[0]['content']
    assert '[#1 ' not in a_prompt[1]['content'] and '一- 西醫診斷' in a_prompt[1]['content']   # no transcript; template present
    cross = [m for r, m in h.fake.calls if r == 'cross']
    assert '教授乙' in cross[0][0]['content'] and '你的版本' in cross[0][1]['content']
    arb = h.fake.calls_for('arbitration')[0][1]['content']
    for heading in ('教授甲 的 A&T', '教授乙 的 A&T', '教授甲 對 教授乙 的評比', '教授乙 對 教授甲 的評比'):
        assert heading in arb


async def test_analysis_concurrency_limit_is_respected(make_session):
    h = await make_session(limit=1)
    h.fake.delay = 0.03
    await h.run_job('analysis')
    assert h.fake.max_active == 1 and h.session.analysis


async def test_arbitration_without_markers_is_retried(make_session):
    h = await make_session()
    h.fake.queue('arbitration', '沒有標記的一大段文字，沒有標記的一大段文字')
    job = await h.run_job('analysis')
    assert job.status == 'succeeded' and len(h.fake.calls_for('arbitration')) == 2
    second = h.fake.calls_for('arbitration')[1][1]['content']
    assert '上一次輸出無效的原因' in second and 'FINAL_AT' in second


async def test_professor_output_missing_template_sections_is_retried_with_the_reason(make_session):
    h = await make_session()
    h.fake.queue('analysis', '## 一- 西醫診斷：偏頭痛（待確認）\n（只寫了一項，缺少其餘項目的內容）')
    job = await h.run_job('analysis')
    assert job.status == 'succeeded', job.message
    prompts = [m for r, m in h.fake.calls if r == 'analysis']
    assert len(prompts) == 3                                                    # A once more than B
    retry = next(m for m in prompts if '上一次輸出無效的原因' in m[1]['content'])
    assert '缺少分析模板項目' in retry[1]['content'] and '中醫診斷' in retry[1]['content']


async def test_cross_review_without_the_required_headings_is_retried(make_session):
    h = await make_session()
    h.fake.queue('cross', '我覺得對方寫得不錯，但是處方有點問題，建議再討論一下細節，謝謝。')
    job = await h.run_job('analysis')
    assert job.status == 'succeeded', job.message
    prompts = [m for r, m in h.fake.calls if r == 'cross']
    assert len(prompts) == 3
    assert any('缺少評比標題' in m[1]['content'] for m in prompts)


async def test_final_at_missing_template_sections_is_retried(make_session):
    h = await make_session()
    h.fake.queue('arbitration', '===ARBITRATION===\n意見一致。\n===FINAL_AT===\n## 一- 西醫診斷：偏頭痛（待確認）\n（缺中醫診斷）')
    job = await h.run_job('analysis')
    assert job.status == 'succeeded' and len(h.fake.calls_for('arbitration')) == 2
    assert '中醫診斷' in h.fake.calls_for('arbitration')[1][1]['content']
    assert '中醫診斷' in h.session.analysis[0].final_at


async def test_persistently_incomplete_at_fails_without_creating_a_version(make_session):
    h = await make_session()
    h.fake.queue('arbitration', *['===ARBITRATION===\n意見一致。\n===FINAL_AT===\n## 一- 西醫診斷：偏頭痛（待確認）'] * 8)
    job = await h.run_job('analysis')
    assert job.status == 'failed' and h.session.analysis == []
    assert len(h.fake.calls_for('arbitration')) == 4                           # 1 try + 3 retries


async def test_free_form_analysis_template_only_needs_non_trivial_output(make_session):
    h = await make_session()
    h.session.analysis_template = ''
    h.fake.queue('analysis', '這是一段夠長的自由格式分析內容，沒有套用任何模板。', '另一段夠長的自由格式分析內容，同樣沒有模板。')
    h.fake.queue('arbitration', '===ARBITRATION===\n意見一致。\n===FINAL_AT===\n最終的自由格式分析內容，足夠長度沒有任何模板項目。')
    job = await h.run_job('analysis')
    assert job.status == 'succeeded', job.message
    assert h.session.analysis[0].final_at.startswith('最終的自由格式')


async def test_one_professor_failing_cancels_the_job_and_creates_no_version(make_session):
    h = await make_session()

    def fail_b(messages):
        if '教授乙' in messages[0]['content']:
            return 'HTTP500'
        return '## 一- 西醫診斷：偏頭痛（待確認）'

    h.fake.queue('analysis', fail_b, fail_b)
    h.fake.queue('analysis', *['HTTP500'] * 6)
    job = await h.run_job('analysis')
    assert job.status == 'failed' and h.session.analysis == []
    assert h.fake.calls_for('cross') == [] and h.fake.calls_for('arbitration') == []     # never reaches later stages


async def test_manual_analysis_edit_appends_a_version_and_keeps_history(make_session):
    h = await make_session()
    await h.run_job('analysis')
    await h.run_job('analysis')
    s = h.session
    s.set_analysis_index(0)
    v = s.edit_analysis_manual('醫師改寫的 A&T')
    assert v.index == 3 and v.source == '醫師手動' and v.parent == 1 and s.analysis_index == 2
    assert [a.index for a in s.analysis] == [1, 2, 3] and s.analysis[0].source == 'llm'
    assert s.edit_analysis_manual('醫師改寫的 A&T') is None                               # unchanged -> no version
    assert (s.store.folder / 'analysis' / '003.md').read_text(encoding='utf-8').count('醫師改寫的 A&T') >= 1


# ===================== 醫師手動行：可依新資料更新，但必須保留來源 =====================
async def seed_manual_line(h, text='辛.9- 大便：一日三次'):
    """A physician-typed line (tagged 醫師手動 by the program) on top of an empty record."""
    assert h.session.edit_note_manual(text)
    assert h.session.note.current_note().endswith('[醫師手動]')


async def test_ai_may_update_a_manual_line_from_newer_evidence_keeping_the_tag_and_adding_the_source(make_session):
    h = await make_session()
    await seed_manual_line(h)
    h.fake.queue('writer', ops({'op': 'replace', 'line': 1, 'content': '辛.9- 大便：偏乾，兩天一次[醫師手動][語音#2]'}))
    job = await h.run_job('record')
    assert job.status == 'succeeded', job.message
    assert h.session.note.current_note() == '辛.9- 大便：偏乾，兩天一次[醫師手動][語音#2]'
    assert h.session.note.get_current()['source'].startswith('病歷書寫')


async def test_writer_and_reviewer_both_receive_the_note_modification_history(make_session):
    """Like the original project: both agents get 【病歷修改 diff 過程】 (initial version -> current version)."""
    h = await make_session()
    await h.run_job('record')                                                   # version 2: 病歷書寫 #1
    assert h.session.edit_note_manual('甲- 現病史：患者頭痛三天[語音#1]\n辛.9- 大便：一日三次')     # version 3: physician
    h.fake.calls.clear()
    job = await h.run_job('record')
    assert job.status == 'succeeded', job.message
    for prompt in (h.fake.calls_for('writer')[0][1]['content'], h.fake.calls_for('reviewer')[0][1]['content']):
        assert '## 【病歷修改 diff 過程】' in prompt
        assert '### 版本 1 -> 版本 2：NOTE' in prompt and '### 版本 2 -> 版本 3：NOTE' in prompt
        assert '版本 2（病歷書寫 #1' in prompt and '版本 3（醫師手動' in prompt           # who made each version
        assert '+辛.9- 大便：一日三次[醫師手動]' in prompt                              # what the physician typed
    reviewer = h.fake.calls_for('reviewer')[0][1]['content']
    assert reviewer.index('病歷修改 diff 過程') < reviewer.index('即將登載的今日病歷')      # history, then the candidate
    assert '本次修改 diff' not in reviewer                                          # the original gives the reviewer no such diff


async def test_writer_and_reviewer_system_prompts_share_identical_status_and_category_rules(make_session):
    """Like the original project (same text in both prompts): the 七大醫療狀態 and 八大類 definitions the writer
    follows are exactly what the reviewer judges by, and no placeholder is left unfilled."""
    from mini.agents.common import load_prompt
    h = await make_session()
    await h.run_job('record')
    writer = h.fake.calls_for('writer')[0][0]['content']
    reviewer = h.fake.calls_for('reviewer')[0][0]['content']
    status = load_prompt('shared_clinical_status.txt').strip()
    categories = load_prompt('shared_hallucination_categories.txt').strip()
    for system in (writer, reviewer):
        assert status in system and categories in system
        assert '{clinical_status}' not in system and '{hallucination_categories}' not in system
        assert 'reported negative' in system and '(a) 尚未取得' in system and 'H-2. 重大診斷' in system
    assert '[問診紀錄' not in writer + reviewer and '[Forum' not in writer + reviewer     # nothing left over from the original


async def test_segments_still_being_proofread_are_explained_as_valid_sources_to_both_agents(make_session):
    """Real visit 2026-10-05-001: the header said 「尚未鎖定：#61、#62」 and neither prompt said what that meant, so the
    reviewer invented 'do not cite unlocked segments' while also demanding the same facts (H-1) -- ten rounds of
    contradictions on the newest, most important lines. Both agents must be told they are ordinary, citable sources."""
    h = await make_session()
    h.session.pipeline.finished = False                      # as if the button were pressed during the visit
    job = await h.run_job('record')
    assert job.status == 'succeeded', job.message
    writer_call = h.fake.calls_for('writer')[0]
    reviewer_call = h.fake.calls_for('reviewer')[0]
    for call in (writer_call, reviewer_call):
        user, system = call[1]['content'], call[0]['content']
        assert '仍在校稿中（文字日後可能微調，仍是有效來源，可以引用）' in user and '尚未鎖定' not in user
        assert '與其他段落一樣是有效來源' in system
    assert '不要因此寫成「未知」或「待確認」' in writer_call[0]['content']
    assert '不得要求降級為「未知」或「待確認」' in reviewer_call[0]['content']


async def test_prompts_carry_the_rules_that_shorten_review_loops(make_session):
    """Found in the user's real visit and a replay of it: the reviewer listed only some of the lines that repeated
    one problem; a fact split across 3-second segments was tagged with one segment; a compound question answered
    with one vague reply flip-flopped for three rounds. Each now has an explicit rule for the agent concerned."""
    h = await make_session()
    await h.run_job('record')
    writer = h.fake.calls_for('writer')[0][0]['content']
    reviewer = h.fake.calls_for('reviewer')[0][0]['content']
    # writer: keep every occurrence of a fact consistent; tag all segments a fact spans (existing range/list forms)
    assert '同步檢查整份病歷中該事實的**所有位置**' in writer
    assert '標籤要涵蓋所有包含它的段落' in writer and '[語音#47-48]' in writer and '[語音#47,48]' in writer
    # reviewer: one issue per occurrence, list everything at once; a range that includes the question is correct
    assert '每一行都各列一筆 issue' in reviewer and '不要只列一部分' in reviewer
    assert '不得因為範圍包含提問段或相鄰段就判 G-2' in reviewer
    # both (shared definition): a compound question with one vague answer is recorded as that answer
    for system in (writer, reviewer):
        assert '一問多症、一句籠統回答' in system and '照原話記錄該回答並標來源' in system
        assert '也不可寫成「尚未詢問」（已經問過了）' in system


async def test_the_prompts_hold_no_numbered_label_examples_so_the_template_alone_decides_the_line_format(make_session):
    """Real visit 2026-10-05-002: the physician removed 甲乙丙… from the template, but the writer still opened every
    line with 甲-/乙-/乙.1- because the system prompt itself said 「如『甲- 現病史：』」 and showed 甲-/辛.9-/壬- examples.
    The model copies examples over the template, so no prompt may carry such labels."""
    import re
    h = await make_session()
    await h.run_job('record')
    label = re.compile(r'[甲乙丙丁戊己庚辛壬癸](?:\.\d+)?-')
    writer = h.fake.calls_for('writer')[0][0]['content']
    reviewer = h.fake.calls_for('reviewer')[0][0]['content']
    assert label.findall(writer) == [] and label.findall(reviewer) == []
    assert '模板沒有編號，你就不要自己加編號' in writer
    assert '欄位名稱與每一行的開頭一律以【病歷模板】為準' in writer


async def test_the_first_write_has_no_history_yet_and_the_history_follows_undo(make_session):
    h = await make_session()
    await h.run_job('record')
    assert '尚無版本間 diff' in h.fake.calls_for('writer')[0][1]['content']        # only the empty initial version
    await h.run_job('record')                                                   # version 3
    h.session.undo_note()                                                       # back to version 2
    h.fake.calls.clear()
    await h.run_job('record')
    writer = h.fake.calls_for('writer')[0][1]['content']
    assert '版本 1 -> 版本 2' in writer and '版本 2 -> 版本 3' not in writer         # the undone version is not history


async def test_the_program_does_not_check_source_tags_the_reviewer_decides(make_session):
    """As in the original TCM-Meridian: only the line operations are validated by code. Whether a tag is right
    (a made-up segment, a forged [醫師手動], a missing tag) is for the reviewer LLM, and with n=0 for nobody."""
    h = await make_session()
    h.fake.queue('writer', ops(insert('甲- 現病史：頭痛[語音#99][醫師手動]'), insert('乙- 過去病史：高血壓', line=2)))
    job = await h.run_job('record')
    assert job.status == 'succeeded', job.message
    assert h.session.note.current_note() == '甲- 現病史：頭痛[語音#99][醫師手動]\n乙- 過去病史：高血壓'
    reviewed = h.fake.calls_for('reviewer')[0][1]['content']
    assert '頭痛[語音#99][醫師手動]' in reviewed                      # the reviewer sees exactly what the writer wrote


async def test_the_reviewer_can_reject_a_wrong_tag_and_the_writer_is_asked_to_fix_it(make_session):
    h = await make_session()
    complaint = {'pass': False, 'issues': [{'line': 1, 'category': 'G-2', 'quote': '頭痛[語音#99]',
                                            'problem': '逐字稿沒有第 99 段', 'fix_hint': '改成 [語音#1]'}], 'comment': 'x'}
    h.fake.queue('writer', ops(insert('甲- 現病史：頭痛[語音#99]')), ops(replace(1, '甲- 現病史：頭痛[語音#1]')))
    h.fake.queue('reviewer', complaint)
    job = await h.run_job('record')
    assert job.status == 'succeeded', job.message
    assert '逐字稿沒有第 99 段' in h.fake.calls_for('writer')[1][1]['content']
    assert h.session.note.current_note() == '甲- 現病史：頭痛[語音#1]'


async def test_with_n_zero_nothing_checks_tags_at_all(make_session):
    h = await make_session(review__pass_required_n=0)
    h.fake.queue('writer', ops(insert('乙- 過去病史：患者有胸痛三天')))                 # no source tag
    job = await h.run_job('record')
    assert job.status == 'succeeded' and h.fake.calls_for('reviewer') == []
    assert h.session.note.current_note() == '乙- 過去病史：患者有胸痛三天'


# ===================== 患者資料版本、作業中不可修改 =====================
async def test_job_results_record_which_patient_data_version_they_used(make_session):
    h = await make_session()
    s = h.session
    assert s.patient_version == 1
    await h.run_job('record')
    await h.run_job('advice')
    await h.run_job('analysis')
    assert s.note.get_current()['meta']['patient_version'] == 1
    assert s.advice[0].patient_version == 1 and s.analysis[0].patient_version == 1
    s.set_patient_text('45歲男性，高血壓病史；補充：糖尿病')
    assert s.patient_version == 2
    await h.run_job('advice')
    assert s.advice[1].patient_version == 2
    started = [e for e in s.log.events if e['type'] == 'job_started']
    assert [e['patient_version'] for e in started] == [1, 1, 1, 2]


async def test_patient_data_cannot_be_changed_while_a_job_runs(make_session):
    from mini.state import AppState, StateError                       # the same guard the import button relies on
    h = await make_session()
    state = AppState.__new__(AppState)
    state.phase, state.visit = 'visiting', h.session
    state.bump = lambda: None
    h.fake.delay = 0.4
    job = h.session.start_job('record')
    await asyncio.sleep(0.05)
    with pytest.raises(StateError, match='作業進行中'):
        AppState.import_patient(state, '另一份資料')
    await job.task
    assert h.session.patient_version == 1
    AppState.import_patient(state, '另一份資料')                       # fine once the job is done
    assert h.session.patient_version == 2
