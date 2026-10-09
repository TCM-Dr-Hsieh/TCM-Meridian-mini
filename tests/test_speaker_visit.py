"""Speaker marking inside a whole visit: fake microphone with two voices, fake ASR, fake voiceprint model, fake LLM."""
import json

from mini.llm import LLMClient, LLMScheduler
from mini.speaker.service import MARKS_NOTE
from mini.speaker.tracker import group_name
from mini.visit import VisitSession
from mini.visit_store import VisitStore
from tests.helpers import FakeASR, FakeLLM, make_settings, roles_reply
from tests.speaker_fakes import DOCTOR_LINE, PATIENT_LINE, FakeEmbedder, bases, conversation, scripted_source


def noisy_embedder(path):
    """Voices that overlap a little, so that some sentences come out too weak to decide."""
    return FakeEmbedder(bases(2, 0.3), noise=0.35, seed=3)


async def run_visit(tmp_path, fake: FakeLLM, *, turns=40, embedder=None, enabled=True, **speaker):
    settings = make_settings(tmp_path)
    settings.speaker.enabled = enabled
    for key, value in speaker.items():
        setattr(settings.speaker, key, value)
    settings.validate()
    audio, replies, truth = conversation(turns)
    store = VisitStore.allocate(settings.visits_path())
    session = VisitSession(
        store=store, settings=settings, patient_text='45歲男性，高血壓病史', record_template='甲- 現病史：\n乙- 過去病史：',
        analysis_template='一- 西醫診斷：\n二- 中醫診斷：', client=LLMClient(fake.transport()), scheduler=LLMScheduler(2),
        asr=FakeASR(replies), source_factory=lambda path: scripted_source(settings, audio, recording_path=path),
        llm_backoff=0, speaker_embedder=embedder or (lambda path: FakeEmbedder(bases(2), seed=3)))
    await session.start()
    await session.pipeline.wait_finished(timeout=30)
    return session, truth


def segment_labels(session):
    """{segment number: the one label its tokens carry} for the segments that have a single label."""
    out = {}
    for seg in range(1, len(session.pipeline.segments) + 1):
        labels = session.speaker.tracker.labels_for(seg)
        if labels and len({(label, source) for _, _, label, source in labels}) == 1:
            out[seg] = labels[0][2:]
    return out


async def test_the_stored_transcript_carries_the_marks_and_they_are_right(tmp_path):
    fake = FakeLLM()
    session, truth = await run_visit(tmp_path, fake)
    await session.finish()
    folder = session.store.folder
    text = (folder / 'transcript.txt').read_text(encoding='utf-8')
    assert f'醫師: {DOCTOR_LINE}' in text and f'患者或家屬: {PATIENT_LINE}' in text
    labels = segment_labels(session)
    late = [s for s in range(20, 39) if s in labels]
    assert len(late) >= 15
    wrong = [s for s in late if labels[s][0] != ('doctor' if truth[s] == 'doctor' else 'other') and labels[s][0] != 'unknown']
    assert wrong == []
    assert sum(labels[s][0] != 'unknown' for s in late) >= 0.75 * len(late)
    data = json.loads((folder / 'transcript.json').read_text(encoding='utf-8'))
    assert data['speaker']['labels'] and 'status' in data['speaker']


async def test_the_first_minute_says_unknown_and_is_filled_in_when_the_groups_exist(tmp_path):
    session, _ = await run_visit(tmp_path, FakeLLM())
    await session.finish()
    first = session.speaker.tracker.labels_for(1)
    later = session.speaker.tracker.labels_for(8)                # about 24 s in: before the first refit, labelled only afterwards
    assert first is not None and later is not None
    assert {label for _, _, label, _ in later} <= {'doctor', 'other', 'unknown'}
    events = [e['type'] for e in session.log.events]
    assert events.index('speaker_started') < events.index('speaker_groups_built') < events.index('speaker_map_set')
    assert 'speaker_finished' in events


async def test_the_job_snapshot_has_the_marks_in_the_agreed_format(tmp_path):
    session, _ = await run_visit(tmp_path, FakeLLM(), turns=70)                  # long enough for a second, confirming answer
    assert session.speaker.trusted
    snap = session.pipeline.snapshot()
    assert MARKS_NOTE in snap.text.splitlines()[0]
    first = next(line for line in snap.lines.splitlines() if line.startswith('語音#1 '))
    assert first == f'語音#1 00:00–00:06 醫師: {DOCTOR_LINE} -> 患者或家屬: {PATIENT_LINE}' or '不明: ' in first
    assert '語音#30 ' in snap.lines
    await session.finish()


async def test_use_in_jobs_off_keeps_the_prompt_plain_but_the_file_marked(tmp_path):
    session, _ = await run_visit(tmp_path, FakeLLM(), use_in_jobs=False)
    snap = session.pipeline.snapshot()
    assert '醫師:' not in snap.text and '患者或家屬:' not in snap.text and MARKS_NOTE not in snap.text
    await session.finish()
    assert '醫師:' in (session.store.folder / 'transcript.txt').read_text(encoding='utf-8')


async def test_marking_is_off_by_default_and_the_transcript_is_the_old_one(tmp_path):
    session, _ = await run_visit(tmp_path, FakeLLM(), enabled=False)
    assert session.speaker is None and session.pipeline.speaker is None
    assert '醫師:' not in session.pipeline.snapshot().text and MARKS_NOTE not in session.pipeline.snapshot().text
    await session.finish()
    assert not (session.store.folder / 'speaker').exists()
    assert 'speaker' not in json.loads((session.store.folder / 'transcript.json').read_text(encoding='utf-8'))


async def test_a_model_that_cannot_load_leaves_the_visit_working_without_marks(tmp_path):
    def broken(path):
        raise RuntimeError('找不到說話者聲紋模型')

    session, _ = await run_visit(tmp_path, FakeLLM(), embedder=broken)
    assert session.speaker is None and '找不到說話者聲紋模型' in session.speaker_note
    assert 'speaker_unavailable' in [e['type'] for e in session.log.events]
    assert session.pipeline.segments and '醫師:' not in session.pipeline.snapshot().text
    await session.finish()


async def test_a_voiceprint_error_in_the_middle_turns_marking_off_but_not_the_transcript(tmp_path):
    inner = FakeEmbedder(bases(2), seed=3)

    def flaky(samples):
        if inner.calls >= 40:
            raise RuntimeError('onnx exploded')
        return inner(samples)

    session, _ = await run_visit(tmp_path, FakeLLM(), embedder=lambda path: flaky)
    assert session.speaker.failed and 'onnx exploded' in session.speaker.failed
    assert len(session.pipeline.segments) >= 38                  # every window still became a segment
    assert 'speaker_error' in [e['type'] for e in session.log.events]
    snapshot = session.pipeline.snapshot().text                   # a failed service is not printed as if it were running
    assert MARKS_NOTE not in snapshot and '不明:' not in snapshot and '醫師:' not in snapshot
    await session.finish()
    assert (session.store.folder / 'transcript.txt').exists()


async def test_the_role_map_is_a_version_list_without_voiceprints(tmp_path):
    session, _ = await run_visit(tmp_path, FakeLLM())
    await session.finish()
    raw = (session.store.folder / 'speaker' / 'map.json').read_text(encoding='utf-8')
    doc = json.loads(raw)
    assert doc['role_versions'][0]['source'] == 'llm'
    assert set(doc['role_versions'][0]['roles'].values()) == {'醫師', '患者或家屬'}
    assert doc['minutes'] and {'minute', 'units', 'groups'} <= set(doc['minutes'][0])
    assert not any(word in raw.lower() for word in ('centre', 'center', 'centroid', 'embedding', 'voiceprint'))


async def test_three_agreeing_answers_lock_the_roles_and_the_model_is_asked_no_more(tmp_path):
    fake = FakeLLM()
    session, _ = await run_visit(tmp_path, fake, turns=70)
    await session.finish()
    asked = len(fake.calls_for('speaker_roles'))
    assert 3 <= asked <= 4                                       # 3 agreeing answers, and possibly one more at the final pass


async def test_the_model_is_asked_about_the_roles_once_per_minute_not_on_every_round(tmp_path):
    fake = FakeLLM()
    session, _ = await run_visit(tmp_path, fake)
    asked = len(fake.calls_for('speaker_roles'))
    assert asked >= 1 and not session.speaker.voter.locked
    await session.speaker.settle()
    await session.speaker.settle()
    assert len(fake.calls_for('speaker_roles')) == asked
    await session.finish()


async def test_the_physician_can_swap_the_roles_and_the_model_is_then_left_out(tmp_path):
    fake = FakeLLM()
    session, _ = await run_visit(tmp_path, fake)
    tracker = session.speaker.tracker
    before = segment_labels(session)
    doctor = next(g for g, r in tracker.roles.items() if r == 'doctor')
    other = next(g for g, r in tracker.roles.items() if r == 'other')
    session.speaker.set_group_roles({doctor: 'other', other: 'doctor'})
    after = segment_labels(session)
    flip = {'doctor': 'other', 'other': 'doctor', 'unknown': 'unknown'}
    assert all(flip[before[s][0]] == after[s][0] for s in before if s in after)
    asked = len(fake.calls_for('speaker_roles'))
    await session.finish()
    assert len(fake.calls_for('speaker_roles')) == asked         # manual wins: no further questions, not even at the end
    assert [e['source'] for e in session.speaker.role_versions][-1] == 'manual'
    assert session.speaker.manual_events and session.speaker.manual_events[-1]['event'] == 'roles'


async def test_text_fill_marks_only_sentences_where_the_answer_agrees_with_the_voice(tmp_path):
    fake = FakeLLM()

    def truthful(messages):
        payload = json.loads(messages[1]['content'])
        rows = {r['id']: r for r in payload['dialogue']}
        return {'answers': [{'id': i, 'role': '醫師' if rows[i]['text'].startswith('請問') else '患者家屬', 'basis': '問答'}
                            for i in payload['ask_ids']]}

    fake.script['speaker_fill'] = [truthful] * 20
    session, truth = await run_visit(tmp_path, fake, unknown_percentile=30.0, embedder=noisy_embedder, turns=60)
    await session.finish()
    stats = session.speaker.tracker.stats()
    assert stats['text_filled'] > 0 and session.speaker.fill_stats['accepted'] == stats['text_filled']
    text = session.pipeline.snapshot().text
    assert '醫師*:' in text or '患者或家屬*:' in text
    tracker = session.speaker.tracker
    for i, s in enumerate(tracker.sentences):
        if s.source == 'text':
            said = ''.join(session.speaker._sentence_text(i))
            assert tracker.role_of(s.gid) == ('doctor' if said.startswith('請問') else 'other')

    fake2 = FakeLLM()
    fake2.script['speaker_fill'] = [lambda m: {'answers': [{'id': i, 'role': '醫師', 'basis': '其他'}
                                                           for i in json.loads(m[1]['content'])['ask_ids']]}] * 20
    session2, _ = await run_visit(tmp_path / 'again', fake2, unknown_percentile=30.0, embedder=noisy_embedder, turns=60)
    await session2.finish()
    tracker2 = session2.speaker.tracker
    for i, s in enumerate(tracker2.sentences):
        if s.source == 'text':
            assert session2.speaker._sentence_text(i).startswith('請問')         # a "醫師" answer is refused for a patient sentence
    assert session2.speaker.fill_stats['rejected'] > 0                     # the patient sentences it called 醫師 were turned down


async def test_text_fill_can_be_switched_off(tmp_path):
    fake = FakeLLM()
    session, _ = await run_visit(tmp_path, fake, text_fill=False, embedder=noisy_embedder, turns=60, unknown_percentile=30.0)
    await session.finish()
    assert session.speaker.tracker.stats()['unknown'] > 5          # there is plenty the fill would have been asked about
    assert fake.calls_for('speaker_fill') == [] and session.speaker.tracker.stats()['text_filled'] == 0


async def test_a_failing_llm_leaves_everything_unknown_and_the_prompts_unmarked(tmp_path):
    fake = FakeLLM()
    fake.script['speaker_roles'] = ['HTTP500'] * 50
    session, _ = await run_visit(tmp_path, fake)
    assert session.speaker.tracker.stats()['labelled'] == 0
    snapshot = session.pipeline.snapshot().text
    assert MARKS_NOTE not in snapshot and '不明:' not in snapshot and '醫師:' not in snapshot   # no doctor named: no marks at all
    assert session.speaker.runs(session.pipeline.segments[3])[0].label == 'unknown'             # the screen still says 不明
    await session.finish()
    assert 'speaker_roles_failed' in [e['type'] for e in session.log.events]
    assert '醫師:' not in (session.store.folder / 'transcript.txt').read_text(encoding='utf-8')


async def test_the_physician_can_name_the_doctor_when_the_model_cannot(tmp_path):
    fake = FakeLLM()
    fake.script['speaker_roles'] = ['HTTP500'] * 50
    session, truth = await run_visit(tmp_path, fake)
    service, tracker = session.speaker, session.speaker.tracker
    assert tracker.group_count == 2 and not tracker.doctor_known
    doctor = 0                                                        # either group will do for this test
    service.set_group_roles({doctor: 'doctor'})                       # naming one of two voices names the other
    assert tracker.roles == {doctor: 'doctor', 1 - doctor: 'other'}
    assert tracker.stats()['labelled'] > 0 and MARKS_NOTE in session.pipeline.snapshot().text
    await session.finish()


async def test_text_fill_waits_for_a_batch_instead_of_calling_for_every_sentence(tmp_path):
    fake = FakeLLM()
    session, _ = await run_visit(tmp_path, fake, turns=150, embedder=noisy_embedder, unknown_percentile=30.0)
    await session.finish()
    asked = session.speaker.fill_stats['asked']
    calls = len(fake.calls_for('speaker_fill'))
    assert asked >= 20 and calls >= 1
    assert calls <= asked / 4, (calls, asked)                    # at least ~4 sentences per call on average, not one call each


async def test_a_retried_end_of_visit_does_not_run_the_final_pass_twice(tmp_path):
    session, _ = await run_visit(tmp_path, FakeLLM())
    await session.speaker.finish()
    await session.speaker.finish()
    assert [e['type'] for e in session.log.events].count('speaker_finished') == 1
    await session.finish()
    assert [e['type'] for e in session.log.events].count('speaker_finished') == 1


async def test_a_fill_call_cancelled_by_the_end_of_the_visit_is_asked_again_in_the_final_pass(tmp_path):
    import asyncio
    fake = FakeLLM()
    session, _ = await run_visit(tmp_path, fake, turns=60, embedder=noisy_embedder, unknown_percentile=30.0, text_fill=True)
    service, tracker = session.speaker, session.speaker.tracker
    await service.stop()
    tracker.finish()
    waiting = tracker.fill_candidates(final=True)
    assert waiting
    fake.delay = 1.5
    calls_before = len(fake.calls_for('speaker_fill'))
    task = asyncio.create_task(service.settle(final=True))
    while len(fake.calls_for('speaker_fill')) == calls_before:
        await asyncio.sleep(0.02)                                  # the call is on its way (and will be slow)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert tracker.fill_candidates(final=True) == waiting          # nothing was lost: they are still to be asked
    fake.delay = 0.0
    await session.finish()


def first_answer_only(fake: FakeLLM):
    """The role layer answers once (correctly) and then fails: the map is never confirmed."""
    fake.script['speaker_roles'] = [lambda m: roles_reply(json.loads(m[1]['content']))] + ['HTTP500'] * 60


async def test_marks_wait_for_a_second_agreeing_answer_before_they_reach_the_prompts_or_the_file(tmp_path):
    fake = FakeLLM()
    first_answer_only(fake)
    session, _ = await run_visit(tmp_path, fake)
    service = session.speaker
    assert service.tracker.doctor_known and not service.trusted and service.tracker.stats()['labelled'] > 0
    assert any(run.label != 'unknown' for seg in session.pipeline.segments for run in (service.runs(seg) or []))   # the screen has them
    assert '暫定' in service.status
    snapshot = session.pipeline.snapshot()
    assert not snapshot.marked and MARKS_NOTE not in snapshot.text and '醫師:' not in snapshot.text
    await session.finish()
    assert not service.trusted
    assert '醫師:' not in (session.store.folder / 'transcript.txt').read_text(encoding='utf-8')
    assert json.loads((session.store.folder / 'transcript.json').read_text(encoding='utf-8'))['speaker']['roles_trusted'] is False


async def test_the_final_pass_asks_once_more_so_a_short_visit_can_still_be_confirmed(tmp_path):
    fake = FakeLLM()
    session, _ = await run_visit(tmp_path, fake, turns=30)                    # 90 s: at most one role answer so far
    service = session.speaker
    await service.stop()
    service.tracker.finish()
    service.voter.confirmed = min(service.voter.confirmed, 1)
    service.voter.trusted = False
    await service.settle(final=True)
    assert service.trusted and service.marks_in_text
    await session.finish()


async def test_a_group_the_model_will_not_name_stays_unknown_on_screen_and_the_prompts_stay_plain(tmp_path):
    fake = FakeLLM()

    def only_the_doctor(messages):
        answer = roles_reply(json.loads(messages[1]['content']))
        return {**answer, 'roles': {g: (r if r == '醫師' else '不明') for g, r in answer['roles'].items()}}

    fake.script['speaker_roles'] = [only_the_doctor] * 60
    session, truth = await run_visit(tmp_path, fake)
    service = session.speaker
    assert service.tracker.doctor_known and not service.voter.usable and not service.trusted
    labels = {run.label for seg in session.pipeline.segments for run in (service.runs(seg) or [])}
    assert 'other' not in labels and 'doctor' in labels                       # never promoted to patient by elimination
    assert MARKS_NOTE not in session.pipeline.snapshot().text
    await session.finish()


async def test_role_changes_are_refused_while_a_job_runs_and_allowed_afterwards(tmp_path):
    import pytest
    from mini.jobs import BusyError
    fake = FakeLLM()
    session, _ = await run_visit(tmp_path, fake)
    fake.delay = 0.6
    job = session.start_job('record')
    assert session.jobs.busy
    roles_before = dict(session.speaker.tracker.roles)
    with pytest.raises(BusyError):
        session.set_speaker_roles({0: 'doctor', 1: 'other'})
    with pytest.raises(BusyError):
        session.set_speaker_lock(True)
    assert session.speaker.tracker.roles == roles_before and not session.speaker.locked
    fake.delay = 0.0
    await job.task
    session.set_speaker_roles({0: 'doctor', 1: 'other'})                       # idle again: the physician may change it
    assert session.speaker.locked
    await session.finish()


async def test_a_record_job_logs_the_role_map_its_transcript_was_built_with(tmp_path):
    fake = FakeLLM()
    session, _ = await run_visit(tmp_path, fake, turns=70)
    assert session.speaker.trusted
    await (session.start_job('record')).task
    started = next(e for e in session.log.events if e['type'] == 'job_started' and e['kind'] == 'record')
    assert started['speaker_marks'] is True
    assert started['speaker']['trusted'] is True and started['speaker']['in_text'] is True
    assert set(started['speaker']['roles'].values()) == {'醫師', '患者或家屬'}
    await session.finish()


async def test_the_stored_labels_say_which_voice_group_each_one_came_from_and_rebuild_from_the_role_map(tmp_path):
    session, _ = await run_visit(tmp_path, FakeLLM(), turns=70, embedder=noisy_embedder, unknown_percentile=30.0)
    await session.finish()
    folder = session.store.folder
    doc = json.loads((folder / 'transcript.json').read_text(encoding='utf-8'))['speaker']
    roles_by_group = json.loads((folder / 'speaker' / 'map.json').read_text(encoding='utf-8'))['role_versions'][-1]['roles']
    names = {'doctor': '醫師', 'other': '患者或家屬'}
    labelled = undecided = 0
    for seg, items in doc['labels'].items():
        for c0, c1, label, source, gid in items:
            if label == 'unknown':
                undecided += gid is None
            else:
                labelled += 1
                assert roles_by_group[f'S{gid + 1}'] == names[label], (seg, c0, label, gid)   # rebuilt from the group and the map
    assert labelled > 100 and undecided > 0


async def test_a_group_with_one_answer_behind_it_is_unknown_on_the_screen_in_the_counts_in_the_files_and_in_the_prompts(tmp_path):
    session, _ = await run_visit(tmp_path, FakeLLM(), turns=70)
    service = session.speaker
    assert service.trusted and service.marks_in_text
    patient = next(g for g, r in service.voter.current.items() if r == 'other')
    assert '患者或家屬: ' in session.pipeline.snapshot().text
    unknown_before = service.tracker.stats()['unknown']
    stored_before = service.labels_doc()
    assert any(item[2] == 'other' for items in stored_before.values() for item in items)
    service.voter.sure.discard(patient)                                       # as if one answer had just named the patient's group
    service._sync_confirmed()
    snapshot = session.pipeline.snapshot()
    assert '患者或家屬' not in snapshot.text and '醫師: ' in snapshot.text and '不明: ' in snapshot.text
    screen = {run.label for seg in session.pipeline.segments for run in (service.runs(seg) or [])}
    assert 'other' not in screen and 'doctor' in screen                       # the screen waits for the confirmation too
    stored = service.labels_doc()
    assert not any(item[2] == 'other' for items in stored.values() for item in items)       # ... and so does the saved transcript.json
    assert any(item[4] == patient for items in stored.values() for item in items)           # which still says which voice group it was
    assert service.tracker.stats()['unknown'] > unknown_before and f'不明 {service.tracker.stats()["unknown"] * 100 // service.tracker.stats()["decided"]}%' in service.status
    assert '待確認' in service.status
    assert service.role_summary()['unconfirmed_groups'] == [group_name(patient)]
    service.voter.sure.add(patient)
    service._sync_confirmed()
    assert '患者或家屬: ' in session.pipeline.snapshot().text and '待確認' not in service.status
    assert any(run.label == 'other' for seg in session.pipeline.segments for run in (service.runs(seg) or []))
    assert service.labels_doc() == stored_before
    await session.finish()


async def test_the_second_try_of_a_fill_call_is_told_the_fill_format_not_the_role_format(tmp_path):
    fake = FakeLLM()
    fake.queue('speaker_fill', 'this is not json')
    session, _ = await run_visit(tmp_path, fake, turns=70, embedder=noisy_embedder, unknown_percentile=30.0)
    await session.finish()
    fills = [messages for role, messages in fake.calls if role == 'speaker_fill']
    assert len(fills) >= 2
    first, retry = fills[0][0]['content'], fills[1][0]['content']
    assert retry.startswith(first) and retry != first
    note = retry[len(first):]
    assert 'answers' in note and 'ask_ids' in note and 'roles' not in note


async def test_the_final_pass_asks_again_while_a_group_is_still_waiting_for_its_second_answer(tmp_path):
    session, _ = await run_visit(tmp_path, FakeLLM(), turns=70)
    service = session.speaker
    await service.stop()
    tracker = service.tracker
    service._asked_roles_at = (len(tracker.minutes), tracker.groups_version)     # nothing is due by the clock
    assert service.voter.locked and not service._roles_due(True)
    patient = next(g for g, r in service.voter.current.items() if r == 'other')
    service.voter.sure.discard(patient)
    assert service.trusted and not service.voter.locked
    assert not service._roles_due(False) and service._roles_due(True)
    service.voter.sure.add(patient)
    assert not service._roles_due(True)


async def test_the_answer_that_confirms_a_role_is_saved_as_a_new_version_and_refreshes_the_page(tmp_path):
    session, _ = await run_visit(tmp_path, FakeLLM(), turns=70)
    service = session.speaker
    await service.stop()
    patient = next(g for g, r in service.voter.current.items() if r == 'other')
    service.voter.sure.discard(patient)
    service._sync_confirmed()
    assert not any(label == 'other' for labels in service.tracker.all_labels().values() for _, _, label, _ in labels)
    versions, refreshed = len(service.role_versions), []
    service._on_change = lambda: refreshed.append(1)
    await service._ask_roles()                                                  # the model agrees once more
    assert patient in service.voter.confirmed_roles
    assert any(label == 'other' for labels in service.tracker.all_labels().values() for _, _, label, _ in labels)   # shown again at once
    assert len(service.role_versions) == versions + 1 and group_name(patient) in service.role_versions[-1]['confirmed_groups']
    assert refreshed
    await session.finish()


async def test_a_voice_group_no_answer_has_named_is_listed_as_waiting_too(tmp_path):
    session, _ = await run_visit(tmp_path, FakeLLM(), turns=70)
    service = session.speaker
    assert service.role_summary()['unconfirmed_groups'] == [] and '待確認' not in service.status
    service.voter.groups = 3                                                  # a third voice group that no answer has named (not in `current`)
    assert service.role_summary()['unconfirmed_groups'] == ['S3'] and '1 群待確認' in service.status
    await session.finish()


async def test_the_physician_locking_the_map_confirms_every_group_on_the_screen(tmp_path):
    session, _ = await run_visit(tmp_path, FakeLLM(), turns=70)
    service = session.speaker
    await service.stop()
    patient = next(g for g, r in service.voter.current.items() if r == 'other')
    service.voter.sure.discard(patient)
    service._sync_confirmed()
    shown = lambda: any(label == 'other' for labels in service.tracker.all_labels().values() for _, _, label, _ in labels)   # noqa: E731
    assert not shown()
    service.set_lock(True)
    assert shown() and service.voter.confirmed_roles.keys() >= {patient}


def says(role):
    return lambda m: {'answers': [{'id': i, 'role': role, 'basis': '其他'} for i in json.loads(m[1]['content'])['ask_ids']]}


async def test_the_second_text_fill_does_nothing_unless_it_is_switched_on(tmp_path):
    fake = FakeLLM()
    fake.script['speaker_fill'] = [says('醫師')] * 40
    session, _ = await run_visit(tmp_path, fake, unknown_percentile=30.0, embedder=noisy_embedder, turns=60)
    await session.finish()
    assert fake.calls_for('speaker_fill') and fake.calls_for('speaker_fill2') == []
    assert session.speaker.tracker.stats()['text_filled2'] == 0 and session.speaker.fill2_stats['asked'] == 0
    assert session.speaker.map_doc()['settings']['text_fill2'] is False                      # the saved record says the switch was off


async def test_the_second_text_fill_takes_an_answer_only_when_it_equals_the_first_one_and_is_asked_blind(tmp_path):
    fake = FakeLLM()
    fake.script['speaker_fill'] = [says('醫師')] * 40                      # the first fill answers 醫師 for everything
    fake.script['speaker_fill2'] = [says('醫師')] * 40                     # ... and the second one says the same
    session, _ = await run_visit(tmp_path, fake, unknown_percentile=30.0, embedder=noisy_embedder, turns=60, text_fill2=True)
    await session.finish()
    service, tracker = session.speaker, session.speaker.tracker
    stats = tracker.stats()
    assert fake.calls_for('speaker_fill2') and stats['text_filled2'] > 0 and service.fill2_stats['accepted'] == stats['text_filled2']
    assert service.map_doc()['settings']['text_fill2'] is True and service.map_doc()['fill2']['asked'] > 0
    for i, s in enumerate(tracker.sentences):
        if s.source == 'text2':
            assert s.said == 'doctor' and tracker.role_of(s.gid) == 'doctor'
            assert s.lean is None or tracker.role_of(s.lean) == 'other'          # it overrode a voice that leaned the other way
    assert '醫師*:' in session.pipeline.snapshot().text                              # shown with the same star as the first fill
    stars = False
    for messages in fake.calls_for('speaker_fill2'):
        assert '說話者二階補標員' in messages[0]['content']
        payload = json.loads(messages[1]['content'])
        assert set(payload) == {'ask_ids', 'dialogue'}                             # no first answer anywhere
        assert all(set(row) == {'id', 't', 'gap', 'who', 'text'} for row in payload['dialogue'])
        stars = stars or any(row['who'].endswith('*') for row in payload['dialogue'])
    assert stars                                                                    # the neighbours' text-filled labels are in view
    events = [e for e in session.log.events if e['type'] == 'speaker_fill2']
    assert events and sum(e['accepted'] for e in events) == service.fill2_stats['accepted']


async def test_a_second_answer_that_differs_from_the_first_leaves_the_sentence_unknown(tmp_path):
    fake = FakeLLM()
    fake.script['speaker_fill'] = [says('醫師')] * 40
    fake.script['speaker_fill2'] = [says('患者家屬')] * 40
    session, _ = await run_visit(tmp_path, fake, unknown_percentile=30.0, embedder=noisy_embedder, turns=60, text_fill2=True)
    await session.finish()
    service = session.speaker
    assert service.fill2_stats['asked'] > 0 and service.fill2_stats['rejected'] > 0
    assert service.tracker.stats()['text_filled2'] == 0


async def test_a_failing_second_fill_changes_nothing_and_does_not_stop_the_visit(tmp_path):
    fake = FakeLLM()
    fake.script['speaker_fill'] = [says('醫師')] * 40
    fake.script['speaker_fill2'] = ['HTTP500'] * 60
    session, _ = await run_visit(tmp_path, fake, unknown_percentile=30.0, embedder=noisy_embedder, turns=60, text_fill2=True)
    await session.finish()
    assert session.speaker.fill2_stats['failed_calls'] > 0 and session.speaker.tracker.stats()['text_filled2'] == 0
    assert not session.speaker.failed
