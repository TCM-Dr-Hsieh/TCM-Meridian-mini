"""Background voices (another room, SPEC 4.3): a quiet voice that fits no voice group is marked as background, shown grey, and kept out of
the record writer's transcript, the role question, the text fill and the unknown count; quiet units are kept out of the group fit."""
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np

from mini.llm import LLMClient, LLMScheduler
from mini.speaker import BACKGROUND, BACKGROUND_GID, DOCTOR, OTHER, UNKNOWN
from mini.speaker.render import Run, prompt_body, segment_runs
from mini.speaker.roles import dialogue_lines
from mini.speaker.service import MARKS_NOTE, SpeakerService
from mini.speaker.tracker import Params, SpeakerTracker, Unit
from mini.visit import VisitSession
from mini.visit_store import VisitStore
from tests.helpers import FakeASR, FakeLLM, make_settings
from tests.speaker_fakes import DIM, DOCTOR_LINE, PATIENT_LINE, RATE, FakeEmbedder, build, feed, make_turns, scripted_source

# The audio level of speaker n is 0.1 * (n + 1): the doctor (4) is loud, the patient (3) a little quieter, the quiet voice (0) is 14 dB below the doctor.
LOUD, MIDDLE, QUIET, STRANGER = 4, 3, 0, 5
BACKGROUND_LINE = '隔壁房間在叫號。'
OFF = float('inf')


def voices(*, alike=False):
    out = {n: np.eye(DIM)[n] for n in range(6)}
    if alike:                                     # doctor and patient share a big component (cosine 0.69); the quiet voice shares nothing
        common = np.eye(DIM)[15]
        for n in (MIDDLE, LOUD):
            v = out[n] + 1.5 * common
            out[n] = v / np.linalg.norm(v)
    return out


class Scattered(FakeEmbedder):
    """Voices in `scattered` get a random voiceprint every time: somebody the model has never heard before and cannot group."""

    def __init__(self, bases, scattered=(), **kwargs):
        super().__init__(bases, **kwargs)
        self.scattered = tuple(scattered)

    def __call__(self, samples):
        level = float(np.mean(samples))
        if abs(level) >= 0.05 and int(round(level / 0.1)) - 1 in self.scattered:
            self.calls += 1
            return self.rng.normal(0, 1, DIM)
        return super().__call__(samples)


def scenario(pattern, *, scattered=(), alike=False, params=None, seed=1):
    turns = make_turns(pattern, seed=seed, lo=1.8, hi=4.0)
    audio, tokens, truth = build(turns)
    vs = voices(alike=alike)
    return SpeakerTracker(Scattered(vs, scattered, noise=0.15, seed=seed), params), audio, tokens, truth, vs


def run_through(tracker, audio, tokens, vs):
    feed(tracker, audio, tokens)
    doctor = int(np.argmax(tracker._centres @ vs[LOUD]))
    tracker.set_roles({g: (DOCTOR if g == doctor else OTHER) for g in range(tracker.group_count)})
    tracker.finish()


def shown(tracker, tokens):
    got = {}
    for seg in {t.seg for t in tokens}:
        for c0, c1, label, source in tracker.labels_for(seg) or ():
            got[(seg, c0)] = label
    return [got.get((t.seg, t.c0), UNKNOWN) for t in tokens]


def rate(labels, truth, speaker, label):
    mine = [got for got, s in zip(labels, truth) if s == speaker]
    return sum(got == label for got in mine) / len(mine)


def with_strangers(speaker, at=(30, 50, 62, 70), length=80):
    pattern = [LOUD, MIDDLE] * (length // 2)
    for k in at:
        pattern[k] = speaker
    return pattern


# --- the rule ------------------------------------------------------------------------------------------------------------
def test_a_quiet_voice_that_fits_no_group_is_background_and_the_people_in_the_room_are_not():
    tracker, audio, tokens, truth, vs = scenario(with_strangers(QUIET), scattered=(QUIET,))
    run_through(tracker, audio, tokens, vs)
    labels = shown(tracker, tokens)
    assert tracker.group_count == 2
    assert rate(labels, truth, QUIET, BACKGROUND) >= 0.8
    assert rate(labels, truth, LOUD, BACKGROUND) <= 0.02 and rate(labels, truth, MIDDLE, BACKGROUND) <= 0.02
    assert rate(labels, truth, LOUD, DOCTOR) >= 0.8 and rate(labels, truth, MIDDLE, OTHER) >= 0.8
    stats = tracker.stats()
    assert stats['background'] >= 3 and stats['labelled'] + stats['unknown'] + stats['background'] == stats['decided']


def test_a_loud_stranger_is_not_background_and_neither_is_a_quiet_voice_a_group_knows():
    loud, audio, tokens, truth, vs = scenario(with_strangers(STRANGER), scattered=(STRANGER,))
    run_through(loud, audio, tokens, vs)
    assert rate(shown(loud, tokens), truth, STRANGER, BACKGROUND) <= 0.1               # not quieter than the doctor: one cue is not enough
    familiar, audio, tokens, truth, vs = scenario([LOUD, QUIET] * 70, params=Params(quiet_gate_db=20.0))      # quiet, but the same voice every time: a weak patient
    run_through(familiar, audio, tokens, vs)
    assert familiar.group_count == 2
    labels = shown(familiar, tokens)
    assert rate(labels, truth, QUIET, BACKGROUND) <= 0.02 and rate(labels, truth, QUIET, OTHER) >= 0.7


def test_the_rule_can_be_switched_off_and_needs_a_named_doctor_group():
    tracker, audio, tokens, truth, vs = scenario(with_strangers(QUIET), scattered=(QUIET,), params=Params(background_rel_db=-OFF))
    run_through(tracker, audio, tokens, vs)
    assert tracker.stats()['background'] == 0 and BACKGROUND not in shown(tracker, tokens)
    tracker, audio, tokens, truth, vs = scenario(with_strangers(QUIET), scattered=(QUIET,))
    feed(tracker, audio, tokens)
    assert tracker._ref_level is None and tracker.stats()['background'] == 0           # nobody is the doctor yet: there is no yardstick for "quieter"


def test_background_sentences_are_kept_out_of_the_role_question_the_text_fill_and_the_unknown_count():
    tracker, audio, tokens, truth, vs = scenario(with_strangers(QUIET), scattered=(QUIET,))
    run_through(tracker, audio, tokens, vs)
    flagged = [i for i, s in enumerate(tracker.sentences) if s.background]
    assert flagged
    assert not set(flagged) & set(tracker.fill_candidates(final=True))
    quiet_token = next(i for i, (t, s) in enumerate(zip(tokens, truth)) if s == QUIET and shown(tracker, [t])[0] == BACKGROUND)
    pieces = tracker.voice_groups_for(tokens[quiet_token].seg)
    assert BACKGROUND_GID in [gid for _, _, gid in pieces]
    assert tracker.stats()['unknown'] == sum(1 for s in tracker.sentences[:tracker._cursor] if not s.background and tracker.shown_role(s.gid) == UNKNOWN)
    unlabelled = [s for s in tracker.sentences if s.background and tracker.shown_role(s.gid) != UNKNOWN]
    assert all(tracker.labels_for(tracker.tokens[s.first].seg) for s in unlabelled)   # a background sentence keeps its group number for the record


def test_a_sentence_is_judged_again_when_the_roles_change():
    tracker, audio, tokens, truth, vs = scenario(with_strangers(QUIET), scattered=(QUIET,))
    run_through(tracker, audio, tokens, vs)
    assert tracker.stats()['background']
    level = tracker._ref_level
    doctor = next(g for g, r in tracker.roles.items() if r == DOCTOR)
    tracker.set_roles({g: (OTHER if g == doctor else DOCTOR) for g in range(tracker.group_count)})      # the physician swaps them
    assert tracker._ref_level is not None and abs(tracker._ref_level - level) > 1.0     # the yardstick is now the other voice's level (2 dB lower)
    tracker.set_roles({})
    assert tracker.stats()['background'] == 0 and tracker._ref_level is None            # no doctor, no yardstick: nothing is background


def test_a_background_sentence_is_not_unknown_and_is_never_asked_about_by_either_text_fill():
    tracker, audio, tokens, truth, vs = scenario(with_strangers(QUIET), scattered=(QUIET,))
    run_through(tracker, audio, tokens, vs)
    index = next(i for i, s in enumerate(tracker.sentences) if s.background)
    sentence = tracker.sentences[index]
    sentence.gid, sentence.said = None, DOCTOR
    assert not tracker._is_unknown(sentence) and index not in tracker.fill2_candidates() and index not in tracker.fill_candidates(final=True)
    sentence.background = False
    assert tracker._is_unknown(sentence) and index in tracker.fill2_candidates() and index in tracker.fill_candidates(final=True)


def test_sentences_are_judged_as_they_are_decided_not_only_at_the_next_refit_or_the_end():
    tracker, audio, tokens, truth, vs = scenario(with_strangers(QUIET), scattered=(QUIET,))
    feed(tracker, audio, tokens)
    doctor = int(np.argmax(tracker._centres @ vs[LOUD]))
    tracker.set_roles({g: (DOCTOR if g == doctor else OTHER) for g in range(tracker.group_count)})        # no refit, no finish after this
    assert tracker.stats()['background'] >= 3
    assert rate(shown(tracker, tokens), truth, QUIET, BACKGROUND) >= 0.8


def test_a_refit_judges_the_decided_sentences_again():
    tracker, audio, tokens, truth, vs = scenario(with_strangers(QUIET), scattered=(QUIET,))
    feed(tracker, audio, tokens)
    doctor = int(np.argmax(tracker._centres @ vs[LOUD]))
    tracker.set_roles({g: (DOCTOR if g == doctor else OTHER) for g in range(tracker.group_count)})
    assert tracker.stats()['background']
    tracker.params = replace(tracker.params, background_max_cos=-1.0)                    # nothing is "unlike every group" any more
    tracker._next_horizon = tracker.units[-1].t1 - 1.0                                   # ... and a refit is due
    tracker._advance(set())
    assert tracker.stats()['background'] == 0 and BACKGROUND not in shown(tracker, tokens)


def test_an_unknown_background_sentence_is_not_promoted_by_the_rescoring():
    tracker, audio, tokens, truth, vs = scenario(with_strangers(QUIET), scattered=(QUIET,))
    run_through(tracker, audio, tokens, vs)
    chosen = next(s for s in tracker.sentences if s.background)
    chosen.gid = None
    tracker._history = [0.0] * 20                                                        # a threshold that any score clears
    before = tracker.rescored
    tracker._rescore()
    assert chosen.gid is None and chosen.background
    others = [s for s in tracker.sentences if not s.background and s.gid is None]
    assert tracker.rescored == before + sum(1 for s in others if tracker._score(s, [g for g, r in tracker.roles.items() if r == DOCTOR], [g for g, r in tracker.roles.items() if r == OTHER]))


def test_the_speaker_groups_window_does_not_offer_a_background_sentence_as_an_example_of_a_voice():
    tracker, audio, tokens, truth, vs = scenario(with_strangers(QUIET), scattered=(QUIET,))
    run_through(tracker, audio, tokens, vs)
    index = max(i for i in range(tracker._cursor) if tracker.sentences[i].last - tracker.sentences[i].first >= 2)
    sentence = tracker.sentences[index]
    sentence.gid, sentence.source, sentence.background = 0, 'voice', True
    assert index not in tracker.group_info()[0].examples
    sentence.background = False
    assert tracker.group_info()[0].examples[0] == index


def test_the_doctors_level_is_the_median_of_the_units_nearest_to_a_doctor_group():
    tracker = SpeakerTracker(lambda samples: None)
    tracker._centres = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0]])
    tracker._roles = {0: DOCTOR, 1: OTHER}
    tracker._model_horizon = 100.0

    def unit(t, vec, level):
        return Unit(0, 0, t, t + 0.8, 0, np.array(vec), level)

    doctor = [unit(i, [1.0, 0, 0, 0], -10.0) for i in range(10)] + [unit(20 + i, [1.0, 0, 0, 0], -2.0) for i in range(2)]
    patient = [unit(30 + i, [0, 1.0, 0, 0], -30.0) for i in range(12)]
    tracker.units = doctor + patient
    tracker._update_reference()
    assert tracker._ref_level == -10.0                                                   # the patient's units and the loud outliers do not count
    tracker.units = doctor[:5] + patient
    tracker._update_reference()
    assert tracker._ref_level is None                                                    # too few units to know the doctor's level


# --- the quiet gate -------------------------------------------------------------------------------------------------------
def alike_pattern():
    return [LOUD, MIDDLE, QUIET] * 30


def test_a_quiet_voice_that_is_the_most_distinct_does_not_take_a_group_from_two_alike_voices():
    tracker, audio, tokens, truth, vs = scenario(alike_pattern(), alike=True)
    run_through(tracker, audio, tokens, vs)
    labels = shown(tracker, tokens)
    assert tracker.group_count == 2
    assert rate(labels, truth, LOUD, DOCTOR) >= 0.8 and rate(labels, truth, MIDDLE, OTHER) >= 0.8
    assert rate(labels, truth, MIDDLE, DOCTOR) <= 0.05
    assert rate(labels, truth, QUIET, BACKGROUND) >= 0.7
    assert any(m['left_out_of_fit'] for m in tracker.minutes)
    plain, audio, tokens, truth, vs = scenario(alike_pattern(), alike=True, params=Params(quiet_gate_db=OFF))
    run_through(plain, audio, tokens, vs)
    assert rate(shown(plain, tokens), truth, MIDDLE, DOCTOR) >= 0.5                     # without the gate the quiet voice takes a group and the patient is taken for the doctor


def test_a_second_voice_that_is_quiet_all_the_time_makes_no_group_while_the_gate_is_on():
    # the known price of the gate (SPEC 4.3): the loud units are one voice, so no group is built, nothing is labelled, and the status line says so.
    # A weak patient and a steady voice from the next room are the same audio here, so there is deliberately no "use all units after a while" fallback.
    tracker, audio, tokens, truth, vs = scenario([LOUD, QUIET] * 70)
    feed(tracker, audio, tokens)
    assert tracker.group_count == 0 and tracker.stats()['labelled'] == 0
    assert all(m['left_out_of_fit'] > 0 for m in tracker.minutes[1:])
    plain = scenario([LOUD, QUIET] * 70, params=Params(quiet_gate_db=OFF))
    feed(plain[0], plain[1], plain[2])
    assert plain[0].group_count == 2 and [e['at'] for e in plain[0].events if e['event'] == 'speaker_groups_built'] == [60.0]


def test_the_groups_window_does_not_count_background_voices_as_the_speech_of_a_group():
    tracker, audio, tokens, truth, vs = scenario(with_strangers(QUIET), scattered=(QUIET,))
    run_through(tracker, audio, tokens, vs)
    inside = sum(1 for u in tracker.units if u.vec is not None and tracker.sentences[u.sentence].background)
    assert inside > 0
    measured = sum(1 for u in tracker.units if u.vec is not None)
    infos = tracker.group_info()
    assert sum(g.units for g in infos) == measured - inside
    assert abs(sum(g.seconds for g in infos) - sum(u.seconds for u in tracker.units if u.vec is not None and not tracker.sentences[u.sentence].background)) < 1e-6
    assert abs(sum(g.share for g in infos) - 1.0) < 1e-9


# --- what is shown and what is sent ----------------------------------------------------------------------------------------
def test_the_prompt_leaves_background_out_and_the_saved_file_keeps_it_marked():
    runs = [Run(DOCTOR, 'voice', '請問哪裡不舒服'), Run(BACKGROUND, 'voice', '隔壁在叫號'), Run(OTHER, 'voice', '胃痛')]
    assert prompt_body(runs) == '醫師: 請問哪裡不舒服 -> 患者或家屬: 胃痛'
    assert prompt_body(runs, keep_background=True) == '醫師: 請問哪裡不舒服 -> 背景: 隔壁在叫號 -> 患者或家屬: 胃痛'
    assert prompt_body([Run(BACKGROUND, 'voice', '叫號')]) == ''                          # nothing else left: the caller drops the line
    marks = ((0, 2, DOCTOR, 'voice'), (2, 3, BACKGROUND, 'voice'), (3, 4, BACKGROUND, 'voice'), (4, 6, OTHER, 'voice'))
    assert [(r.label, r.text) for r in segment_runs('請問叫號胃痛', '請問叫號胃痛', marks)] == [(DOCTOR, '請問'), (BACKGROUND, '叫號'), (OTHER, '胃痛')]


def test_the_role_question_does_not_show_the_model_background_voices():
    segment = SimpleNamespace(index=3, kind='speech', visible=True, added='請問叫號胃痛', corrected='請問叫號胃痛')
    pieces = [(0, 2, 0), (2, 4, BACKGROUND_GID), (4, 6, 1)]
    tracker = SimpleNamespace(voice_groups_for=lambda seg: pieces)
    assert dialogue_lines([segment], tracker) == ['#3 [S1] 請問 [S2] 胃痛']
    only = SimpleNamespace(voice_groups_for=lambda seg: [(0, 6, BACKGROUND_GID)])
    assert dialogue_lines([segment], only) == []


def test_the_screen_shows_background_voices_grey_with_an_explanation():
    from mini.ui.page import MainPage

    class Probe:
        app = SimpleNamespace(display=lambda text: text)
        run_html = MainPage.run_html

    html = Probe().run_html(Run(BACKGROUND, 'voice', '叫號<b>'))
    assert html.startswith('<span class="who background"') and '背景：</span>' in html and '不會送進病歷書寫' in html
    assert '<span class="bg-text">叫號&lt;b&gt;</span>' in html
    assert Probe().run_html(Run(DOCTOR, 'voice', '好')) == '<span class="who doctor">醫師：</span>好'


# --- a whole visit -------------------------------------------------------------------------------------------------------
def background_conversation(turns=70, background=(40, 48, 56)):
    speakers = [LOUD if i % 2 == 0 else MIDDLE for i in range(turns)]
    for i in background:
        speakers[i] = QUIET
    audio = np.concatenate([np.full(int(3.0 * RATE), 0.1 * (s + 1), dtype=np.float32) for s in speakers])
    lines = [BACKGROUND_LINE if s == QUIET else (DOCTOR_LINE if s == LOUD else PATIENT_LINE) for s in speakers]
    replies = {k: lines[k] + lines[k + 1] for k in range(turns - 1)}
    replies[turns - 1] = lines[turns - 1]
    return audio, replies


async def run_visit(tmp_path, fake=None, noise=0.15, background=(40, 48, 56), **speaker):
    settings = make_settings(tmp_path)
    settings.speaker.enabled = True
    for key, value in speaker.items():
        setattr(settings.speaker, key, value)
    settings.validate()
    audio, replies = background_conversation(background=background)
    session = VisitSession(
        store=VisitStore.allocate(settings.visits_path()), settings=settings, patient_text='45歲男性，高血壓病史',
        record_template='甲- 現病史：\n乙- 過去病史：', analysis_template='一- 西醫診斷：\n二- 中醫診斷：', client=LLMClient((fake or FakeLLM()).transport()),
        scheduler=LLMScheduler(2), asr=FakeASR(replies), source_factory=lambda path: scripted_source(settings, audio, recording_path=path),
        llm_backoff=0, speaker_embedder=lambda path: Scattered(voices(), (QUIET,), noise=noise, seed=3))
    await session.start()
    await session.pipeline.wait_finished(timeout=30)
    return session


async def test_the_record_writers_transcript_has_no_background_line_but_the_file_and_the_screen_keep_it(tmp_path):
    session = await run_visit(tmp_path)
    assert session.speaker.trusted
    assert session.speaker.tracker.stats()['background'] >= 2
    snap = session.pipeline.snapshot()
    assert BACKGROUND_LINE not in snap.text and '背景' not in snap.text
    assert all(snap.lines.splitlines())                                                  # no empty line is left where background voices were
    assert f'醫師: {DOCTOR_LINE}' in snap.text or DOCTOR_LINE in snap.text
    flagged = [seg for seg in session.pipeline.segments if BACKGROUND_LINE in seg.corrected
               and all(label == BACKGROUND for _, _, label, _ in (session.speaker.tracker.labels_for(seg.index) or ()))]
    assert flagged
    for seg in flagged:
        assert f'語音#{seg.index} ' not in snap.lines                                    # a line with nothing but background voices is not sent at all
        assert seg.index not in snap.unlocked
        runs = session.speaker.runs(seg)
        assert [r.label for r in runs] == [BACKGROUND]                                  # the screen still has it (grey)
    await session.finish()
    text = (session.store.folder / 'transcript.txt').read_text(encoding='utf-8')
    assert f'背景: {BACKGROUND_LINE}' in text
    status = session.speaker.status
    assert '背景' in status and '句' in status
    stored = (session.store.folder / 'transcript.json').read_text(encoding='utf-8')
    assert '"background"' in stored


async def test_use_in_jobs_off_sends_no_marks_but_still_no_background(tmp_path):
    session = await run_visit(tmp_path, use_in_jobs=False)
    assert session.speaker.tracker.stats()['background'] >= 2
    text = session.pipeline.snapshot().text
    assert BACKGROUND_LINE not in text and MARKS_NOTE not in text and '醫師:' not in text and '背景' not in text     # the switch is about the marks only
    assert DOCTOR_LINE in text and PATIENT_LINE in text
    await session.finish()
    assert f'背景: {BACKGROUND_LINE}' in (session.store.folder / 'transcript.txt').read_text(encoding='utf-8')


async def test_background_is_kept_out_before_the_role_map_is_confirmed_too(tmp_path):
    session = await run_visit(tmp_path)
    session.speaker.voter.trusted = False                                               # between the first and the second role answer
    assert not session.speaker.marks_in_text
    assert any(r.label == BACKGROUND for seg in session.pipeline.segments if (runs := session.speaker.runs(seg)) for r in runs)     # the screen says "background" ...
    snap = session.pipeline.snapshot()
    assert BACKGROUND_LINE not in snap.text and not snap.marked and '醫師:' not in snap.text                          # ... so the record writer must not get it
    assert DOCTOR_LINE in snap.text
    session.speaker.voter.trusted = True
    await session.finish()


def test_a_segment_without_marks_loses_only_its_background_part():
    def service(labels, failed=''):
        return SimpleNamespace(failed=failed, tracker=SimpleNamespace(labels_for=lambda index: labels))

    segment = SimpleNamespace(index=4, added='請問叫號胃痛', corrected='請問叫號胃痛')
    mixed = ((0, 2, DOCTOR, 'voice'), (2, 4, BACKGROUND, 'voice'), (4, 6, OTHER, 'voice'))
    assert SpeakerService.body(service(mixed), segment, marks=False) == '請問 胃痛'
    assert SpeakerService.body(service(mixed), segment, marks=True) == '醫師: 請問 -> 患者或家屬: 胃痛'
    assert SpeakerService.body(service(mixed), segment, marks=False, keep_background=True) is None       # the file: as it is
    only = ((0, 6, BACKGROUND, 'voice'),)
    assert SpeakerService.body(service(only), segment, marks=False) == '' and SpeakerService.body(service(only), segment, marks=True) == ''
    plain = ((0, 3, DOCTOR, 'voice'), (3, 6, OTHER, 'voice'))
    assert SpeakerService.body(service(plain), segment, marks=False) is None            # nothing to take out: the text is used as it is
    assert SpeakerService.body(service(mixed, failed='x'), segment, marks=False) is None and SpeakerService.body(service(None), segment, marks=False) is None


async def test_the_text_fill_is_not_shown_background_voices_as_part_of_the_dialogue(tmp_path):
    fake = FakeLLM()
    session = await run_visit(tmp_path, fake, noise=0.9, unknown_percentile=40.0)
    await session.finish()
    tracker = session.speaker.tracker
    background = [i for i, s in enumerate(tracker.sentences) if s.background]
    assert background
    calls = [json.loads(m[1]['content']) for m in fake.calls_for('speaker_fill')]
    spanning = [c for c in calls if any(min(r['id'] for r in c['dialogue']) < i < max(r['id'] for r in c['dialogue']) for i in background)]
    assert spanning                                                                      # a call whose window really covers a background sentence
    for call in calls:
        shown = {r['id'] for r in call['dialogue']}
        assert not shown & set(background) and not set(call['ask_ids']) & set(background)
        assert all(BACKGROUND_LINE not in r['text'] for r in call['dialogue'])


async def test_a_segment_of_nothing_but_background_voices_is_not_named_as_still_being_corrected(tmp_path):
    session = await run_visit(tmp_path, background=(40, 48, 68))
    session.pipeline.finished = False                                                    # as during the visit: the last segments are "still being corrected"
    last = [seg.index for seg in session.pipeline.segments if BACKGROUND_LINE in seg.corrected][-1]
    assert last in session.pipeline.unlocked_indexes()
    snap = session.pipeline.snapshot()
    assert last not in snap.unlocked and f'#{last} ' not in snap.text
    assert snap.unlocked                                                                 # the other recent segments are still named
    session.pipeline.finished = True
    await session.finish()
