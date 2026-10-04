import json
import wave
from datetime import datetime

from mini.visit_store import EventLog, VisitStore, render_log_md


def test_folder_numbering_is_per_day_and_never_collides(tmp_path):
    day = datetime(2026, 10, 4, 9, 0)
    names = [VisitStore.allocate(tmp_path, day).visit_id for _ in range(3)]
    assert names == ['2026-10-04-001', '2026-10-04-002', '2026-10-04-003']
    assert VisitStore.allocate(tmp_path, datetime(2026, 10, 5)).visit_id == '2026-10-05-001'


def test_event_log_is_append_only_jsonl_with_wall_and_audio_time(tmp_path):
    clock = iter([None, 12.345])
    log = EventLog(tmp_path / 'log.jsonl', lambda: next(clock))
    log.emit('a', x=1)
    log.emit('b', y='中文')
    log.close()
    rows = [json.loads(line) for line in (tmp_path / 'log.jsonl').read_text(encoding='utf-8').splitlines()]
    assert [r['seq'] for r in rows] == [1, 2] and rows[0]['t'] is None and rows[1]['t'] == 12.35
    assert rows[1]['y'] == '中文' and 'T' in rows[0]['ts']


class FailingHandle:
    """A log file handle whose write fails (optionally after writing part of the line, like a full disk)."""

    def __init__(self, partial: int = 0):
        self.partial, self.closed, self.written = partial, False, ''

    def write(self, text):
        self.written += text[:self.partial]
        raise OSError('磁碟已滿（模擬）')

    def flush(self):
        pass

    def close(self):
        self.closed = True


def disk_rows(path):
    """Every complete JSON line on disk (blank lines and torn fragments skipped)."""
    rows = []
    for line in path.read_text(encoding='utf-8').splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass
    return rows


def test_a_failed_log_write_records_nothing_and_memory_still_matches_the_disk(tmp_path):
    import pytest
    log = EventLog(tmp_path / 'log.jsonl')
    log.emit('a')
    real, log._handle = log._handle, FailingHandle()
    with pytest.raises(OSError):
        log.emit('visit_finished')
    assert [e['type'] for e in log.events] == ['a']                         # not in memory (so not in log.md either)
    real.flush()
    again = log.emit('visit_finished')                                      # the retry succeeds ...
    assert again['seq'] == 2                                                # ... and reuses the unconsumed number
    log.close()
    assert [(r['seq'], r['type']) for r in disk_rows(tmp_path / 'log.jsonl')] == [(1, 'a'), (2, 'visit_finished')]
    assert [(e['seq'], e['type']) for e in log.events] == [(1, 'a'), (2, 'visit_finished')]


def test_a_failed_write_that_wrote_nothing_leaves_no_blank_line_behind(tmp_path):
    """Strict readers (json.loads on every line) must keep working: only a real torn fragment is separated."""
    import pytest
    log = EventLog(tmp_path / 'log.jsonl')
    log.emit('a')
    log._handle = FailingHandle()                                           # fails without writing a single character
    with pytest.raises(OSError):
        log.emit('b')
    log.emit('c')
    log.close()
    text = (tmp_path / 'log.jsonl').read_text(encoding='utf-8')
    assert text.endswith('\n') and '\n\n' not in text
    assert [json.loads(line)['type'] for line in text.splitlines()] == ['a', 'c']       # no line is rejected


def test_a_torn_line_from_a_failed_write_cannot_corrupt_the_next_event(tmp_path):
    import pytest
    log = EventLog(tmp_path / 'log.jsonl')
    log.emit('a')
    real = log._handle
    real.write('{"seq": 2, "type": "half')                                  # what a full disk leaves behind
    real.flush()
    log._handle = FailingHandle()
    with pytest.raises(OSError):
        log.emit('b')
    log.emit('c')
    log.close()
    lines = (tmp_path / 'log.jsonl').read_text(encoding='utf-8').split('\n')
    assert [r['type'] for r in disk_rows(tmp_path / 'log.jsonl')] == ['a', 'c']   # 'c' is intact, on its own line
    assert any(line.startswith('{"seq": 2, "type": "half') for line in lines)     # the fragment stays isolated


def test_log_md_is_chronological_and_folds_llm_prompts():
    events = [
        {'seq': 1, 'ts': '2026-10-04T09:00:00.000+08:00', 't': None, 'type': 'visit_started', 'visit_id': 'v'},
        {'seq': 2, 'ts': '2026-10-04T09:00:05.000+08:00', 't': 5.0, 'type': 'llm_call', 'agent': 'record_writer',
         'call_id': 'c0001', 'attempt': 1, 'job_id': 'J001', 'model': 'm', 'temperature': 0.7, 'max_tokens': 10,
         'latency_ms': 3, 'finish_reason': 'stop', 'error': None,
         'messages': [{'role': 'system', 'content': 'SYS'}, {'role': 'user', 'content': 'USER PROMPT'}],
         'response': 'REPLY'},
        {'seq': 3, 'ts': '2026-10-04T09:00:06.000+08:00', 't': 6.0, 'type': 'note_snapshot_pushed', 'index': 2,
         'source': '病歷書寫 #1'},
    ]
    md = render_log_md('v', events)
    assert md.index('visit_started') < md.index('record_writer') < md.index('病歷新增版本 2')
    assert '<details>' in md and 'USER PROMPT' in md and 'REPLY' in md and 't+00:05' in md


async def test_finishing_a_visit_writes_every_audit_file(make_session):
    h = await make_session(seconds=7)
    s = h.session
    await h.run_job('record')
    s.edit_note_manual(s.note.current_note() + '\n乙- 過去病史：高血壓')
    await h.run_job('advice')
    await h.run_job('analysis')
    s.edit_analysis_manual('醫師修改後的 A&T')
    s.set_patient_text('45歲男性，高血壓病史；補充：糖尿病')
    folder = await s.finish()
    names = {p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}
    expected = {'meta.json', 'audio.wav', 'transcript.json', 'transcript.txt', 'transcript_llm.jsonl',
                'patient_input.json', 'patient_input.txt', 'note/history.json', 'note/diff.md',
                'advice/001.json', 'advice/001.md', 'advice/index.json',
                'analysis/001.json', 'analysis/001.md', 'analysis/002.json', 'analysis/002.md', 'analysis/index.json',
                '今日病歷.md', '分析與處置.md', 'log.jsonl', 'log.md'}
    assert expected <= names, expected - names

    meta = json.loads((folder / 'meta.json').read_text(encoding='utf-8'))
    assert meta['status'] == 'finished' and meta['counts']['note_versions'] == 3
    assert 'api_key' not in json.dumps(meta)

    with wave.open(str(folder / 'audio.wav')) as wav:
        assert wav.getnframes() == 7 * 16000

    transcript = json.loads((folder / 'transcript.json').read_text(encoding='utf-8'))
    assert transcript['finished'] and len(transcript['segments']) == 2
    assert 'raw_asr' in transcript['segments'][0] and 'alignment' in transcript['segments'][0]
    assert (folder / 'transcript.txt').read_text(encoding='utf-8').startswith('[#1 00:00–00:06]')

    history = json.loads((folder / 'note/history.json').read_text(encoding='utf-8'))
    assert [x['source'] for x in history['snapshots']] == ['init', '病歷書寫 #1', '醫師手動']
    agents = {c['agent'] for c in history['llm_calls']}
    assert agents == {'record_writer', 'hallucination_corrector'}
    assert all(c['messages'] for c in history['llm_calls'])                      # every LLM input is preserved
    assert '```diff' in (folder / 'note/diff.md').read_text(encoding='utf-8')

    advice = json.loads((folder / 'advice/001.json').read_text(encoding='utf-8'))
    assert advice['llm_calls'][0]['messages'] and advice['western_ddx']
    analysis = json.loads((folder / 'analysis/001.json').read_text(encoding='utf-8'))
    assert len(analysis['llm_calls']) == 5 and all(c['messages'] for c in analysis['llm_calls'])
    index = json.loads((folder / 'analysis/index.json').read_text(encoding='utf-8'))
    assert index['displayed_index'] == 2 and index['versions'][1]['source'] == '醫師手動'
    assert (folder / '分析與處置.md').read_text(encoding='utf-8').strip() == '醫師修改後的 A&T'

    patient = json.loads((folder / 'patient_input.json').read_text(encoding='utf-8'))
    assert len(patient['versions']) == 2 and '糖尿病' in (folder / 'patient_input.txt').read_text(encoding='utf-8')

    events = [json.loads(line) for line in (folder / 'log.jsonl').read_text(encoding='utf-8').splitlines()]
    kinds = [e['type'] for e in events]
    assert kinds[0] == 'visit_started' and kinds[-1] == 'visit_finished'
    assert [e['seq'] for e in events] == list(range(1, len(events) + 1))
    assert all(e['ts'] for e in events) and any(e['t'] is not None for e in events)
    for needed in ('transcript_segment', 'llm_call', 'note_snapshot_pushed', 'advice_created', 'analysis_created',
                   'analysis_manual_edit', 'patient_input_updated', 'note_manual_edit', 'job_started'):
        assert needed in kinds, needed
    llm_events = [e for e in events if e['type'] == 'llm_call']
    assert all(e['messages'] for e in llm_events)
    corrector_lines = (folder / 'transcript_llm.jsonl').read_text(encoding='utf-8').splitlines()
    assert corrector_lines and json.loads(corrector_lines[0])['agent'] == 'transcript_corrector'
    assert (folder / '今日病歷.md').read_text(encoding='utf-8').startswith('甲- 現病史：')
    assert (folder / 'log.md').read_text(encoding='utf-8').startswith('# 看診時間序 log')


async def test_finishing_cancels_a_running_job_and_still_saves(make_session):
    h = await make_session()
    h.fake.delay = 5
    job = h.session.start_job('record')
    folder = await h.session.finish()
    assert job.status == 'cancelled' and (folder / 'log.md').exists()
    assert any(e['type'] == 'job_cancelled' for e in h.session.log.events)


async def test_aborting_marks_the_visit_aborted_but_keeps_files(make_session):
    h = await make_session()
    await h.session.abort()
    meta = json.loads((h.session.store.folder / 'meta.json').read_text(encoding='utf-8'))
    assert meta['status'] == 'aborted' and (h.session.store.folder / 'log.md').exists()


# ===================== 結束失敗：可重試，不會卡在 finishing =====================
def break_once(obj, name, when):
    """Make obj.name raise OSError the first time `when(*args)` is true, then behave normally."""
    real, state = getattr(obj, name), {'failed': False}

    def wrapper(*args, **kwargs):
        if not state['failed'] and when(*args):
            state['failed'] = True
            raise OSError('磁碟已滿（模擬）')
        return real(*args, **kwargs)

    setattr(obj, name, wrapper)


async def test_a_failed_finish_leaves_a_retryable_state_and_the_retry_completes(make_session):
    import pytest
    from mini.jobs import BusyError
    h = await make_session()
    s = h.session
    await h.run_job('record')
    break_once(s.store, 'write_json', lambda rel, data=None: rel == 'analysis/index.json')
    with pytest.raises(OSError):
        await s.finish()
    assert s.phase == 'finish_failed' and '磁碟已滿' in s.finish_error      # NOT stuck in "finishing"
    with pytest.raises(BusyError):
        s.start_job('record')                                                # AI work stays off until the visit is closed
    folder = await s.finish()                                                # retry
    assert s.phase == 'finished' and (folder / 'log.md').exists() and (folder / '今日病歷.md').exists()
    kinds = [e['type'] for e in s.log.events]
    assert kinds.count('visit_finished') == 1 and 'visit_finish_failed' in kinds
    assert json.loads((folder / 'meta.json').read_text(encoding='utf-8'))['status'] == 'finished'


async def test_a_failure_after_the_log_was_closed_can_still_be_retried(make_session):
    h = await make_session()
    s = h.session
    break_once(s.store, 'write_log_md', lambda events: True)
    import pytest
    with pytest.raises(OSError):
        await s.finish()
    assert s.phase == 'finish_failed'
    folder = await s.finish()
    assert (folder / 'log.md').exists() and s.phase == 'finished'
    lines = (folder / 'log.jsonl').read_text(encoding='utf-8').splitlines()
    assert all(json.loads(line) for line in lines)                           # the reopened log stayed valid JSONL


async def test_a_failed_visit_finished_event_is_written_again_on_retry(make_session):
    import pytest
    h = await make_session()
    s = h.session
    break_once(s.log, 'emit', lambda kind, *rest: kind == 'visit_finished')   # the event's own write fails once
    with pytest.raises(OSError):
        await s.finish()
    assert s.phase == 'finish_failed'
    folder = await s.finish()                                                 # retry
    assert s.phase == 'finished'
    on_disk = [r['type'] for r in disk_rows(folder / 'log.jsonl')]
    assert on_disk.count('visit_finished') == 1                               # the ending is in log.jsonl ...
    assert [e['type'] for e in s.log.events].count('visit_finished') == 1
    assert (folder / 'log.md').read_text(encoding='utf-8').count('visit_finished') >= 1   # ... and in log.md


async def test_a_crashed_transcript_pipeline_does_not_make_the_visit_impossible_to_close(make_session):
    import asyncio

    async def boom():
        raise RuntimeError('pipeline exploded')

    h = await make_session()
    s = h.session
    s._tasks[0] = asyncio.create_task(boom())
    folder = await s.finish()
    assert s.phase == 'finished' and (folder / 'log.md').exists()
    assert any(e['type'] == 'transcript_pipeline_error' and 'exploded' in e['error'] for e in s.log.events)
