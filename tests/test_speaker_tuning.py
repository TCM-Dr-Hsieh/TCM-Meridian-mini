"""The tunable numbers of speaker marking (SPEC 4.3): the registry, the checks, that every knob reaches the code that uses it, and the pop-up."""
import asyncio
import json
import math

import numpy as np
import pytest
from nicegui import ui
from nicegui.testing import User

from mini.config import Settings, SpeakerSettings
from mini.speaker import DOCTOR, OTHER, fill, roles, tuning
from mini.speaker.service import SpeakerService, build_params
from mini.speaker.tracker import Params, Sentence, SpeakerTracker, Unit
from mini.speaker.units import Token, UnitParams, UnitStream, build_units
from mini.ui.page import MainPage
from tests.helpers import FakeLLM
from tests.speaker_fakes import FakeEmbedder, bases, build, feed, make_turns
from tests.test_speaker_ui import marked, speaker_state, until
from tests.test_speaker_visit import noisy_embedder, run_visit


def a_service(**settings) -> SpeakerService:
    return SpeakerService(settings=SpeakerSettings(**settings), embed=lambda samples: None, caller=None, endpoint=lambda: None,
                          segments=lambda: [], emit=lambda *args, **kwargs: None, on_change=lambda: None)


def other_value(knob):
    """A valid value that is not the default (two steps up, or two steps down at the top of the range)."""
    value = knob.default + 2 * knob.step
    if value > knob.high:
        value = knob.default - 2 * knob.step
    return int(round(value)) if knob.integer else round(value, 6)


# --- the registry ------------------------------------------------------------------------------------------------------------
def test_every_knob_is_described_in_range_and_in_the_order_of_the_pipeline():
    assert len({k.key for k in tuning.KNOBS}) == len(tuning.KNOBS)
    stage_order = [tuning.STAGES.index(k.stage) for k in tuning.KNOBS]
    assert stage_order == sorted(stage_order) and set(stage_order) == set(range(len(tuning.STAGES)))     # every stage has knobs, in order
    for k in tuning.KNOBS:
        assert k.low <= k.default <= k.high and k.step > 0, k.key
        assert all(text.strip() for text in (k.label, k.what, k.higher, k.lower)), k.key
        assert k.integer == (isinstance(k.default, int) and not isinstance(k.default, bool)), k.key
    starred = {k.key for k in tuning.KNOBS if k.star}
    assert starred == {'quiet_gate_db', 'first_split_max_cos', 'new_group_max_cos', 'unknown_percentile', 'background_max_cos'}


def test_the_defaults_are_the_numbers_the_pipeline_was_built_with():
    values = tuning.effective({}, tuning.DEFAULTS['unknown_percentile'])
    assert build_params(values) == Params()                                              # tracker, background, gate, fills and units
    assert UnitParams() == build_params(values).units
    assert roles.RoleVoter().lock_after == tuning.DEFAULTS['lock_after'] and roles.MAX_LINES == tuning.DEFAULTS['max_lines']
    assert (fill.CHUNK, fill.BEFORE, fill.AFTER) == (tuning.DEFAULTS['fill_chunk'], tuning.DEFAULTS['fill_before'], tuning.DEFAULTS['fill_after'])
    assert a_service().effective_tuning() == tuning.DEFAULTS                             # and the running objects hold exactly the defaults


@pytest.mark.parametrize('knob', tuning.KNOBS, ids=lambda k: k.key)
def test_every_knob_reaches_the_code_that_uses_it(knob):
    value = other_value(knob)
    assert value != knob.default
    settings = SpeakerSettings(unknown_percentile=value) if knob.top_level else SpeakerSettings(advanced={knob.key: value})
    settings.validate()
    held = a_service(**{f: getattr(settings, f) for f in ('unknown_percentile', 'advanced')}).effective_tuning()
    assert held[knob.key] == value and isinstance(held[knob.key], int) == knob.integer
    assert {k: v for k, v in held.items() if k != knob.key} == {k: v for k, v in tuning.DEFAULTS.items() if k != knob.key}   # and nothing else moved


# --- checking and saving ---------------------------------------------------------------------------------------------------------
def test_only_the_values_that_differ_from_the_default_are_kept_and_unknown_keys_are_ignored():
    s = SpeakerSettings(advanced={'far_cos': 0.25, 'max_groups': 5, 'from_the_future': 7})
    s.validate()
    assert s.advanced == {'max_groups': 5}
    assert tuning.changed(tuning.effective(s.advanced, 20.0)) == {'max_groups': 5, 'unknown_percentile': 20.0}


@pytest.mark.parametrize('advanced, words', [
    ({'far_cos': 0.9}, '不可大於'), ({'refit_seconds': 5}, '不可小於'), ({'max_groups': 4.5}, '整數'), ({'clip': True}, '數值'),
    ({'clip': 'six'}, '數值'), ({'min_unit': 1.0, 'target': 0.6}, '單位長度'), ({'target': 2.0}, '單位長度'), (['far_cos'], '物件')])
def test_a_bad_value_is_refused_with_its_name(advanced, words):
    with pytest.raises(ValueError, match=words):
        SpeakerSettings(advanced=advanced).validate()


def test_the_settings_file_keeps_only_the_overrides_and_an_older_file_still_loads(tmp_path):
    path = tmp_path / 'config.json'
    settings = Settings()
    settings.speaker.advanced = {'background_max_cos': 0.5, 'fill_chunk': 10}
    settings.save(path)
    saved = json.loads(path.read_text(encoding='utf-8'))['speaker']
    assert saved['advanced'] == {'background_max_cos': 0.5, 'fill_chunk': 10} and saved['unknown_percentile'] == 15.0
    assert Settings.load(path).speaker.advanced == {'background_max_cos': 0.5, 'fill_chunk': 10}
    older = json.loads(path.read_text(encoding='utf-8'))
    del older['speaker']['advanced']
    path.write_text(json.dumps(older), encoding='utf-8')
    assert Settings.load(path).speaker.advanced == {}                                    # a file from before the pop-up: all defaults


# --- the numbers really change the behaviour -------------------------------------------------------------------------------------
def test_unit_numbers_change_how_the_text_is_cut():
    tokens = [Token(1, i, i + 1, i * 0.25, i * 0.25 + 0.25) for i in range(12)]          # continuous speech, 3 s
    assert len(build_units(tokens)) == 3
    assert len(build_units(tokens, UnitParams(target=0.5))) == 6 and len(build_units(tokens, UnitParams(target=1.5, max_merged=2.0))) == 2
    assert [u for u in UnitStream(UnitParams(target=0.5)).add(tokens)] != [u for u in UnitStream().add(tokens)]
    gap = [Token(1, 0, 1, 0.0, 0.5), Token(1, 1, 2, 1.1, 1.6)]                           # a 0.6 s pause
    for sent_pause, sentences in ((0.5, 2), (1.0, 1)):
        tracker = SpeakerTracker(lambda samples: None, Params(units=UnitParams(sent_pause=sent_pause)))
        tracker.add_tokens(gap)
        assert len(tracker.sentences) == sentences


def test_the_refit_period_changes_how_often_the_groups_are_refitted():
    turns = make_turns([0, 1] * 25, seed=2)
    audio, tokens, _ = build(turns)
    counts = {}
    for seconds in (60.0, 30.0):
        tracker = SpeakerTracker(FakeEmbedder(bases(2), noise=0.15, seed=2), Params(refit_seconds=seconds))
        feed(tracker, audio, tokens)
        counts[seconds] = len(tracker.minutes)
    assert counts[30.0] >= 2 * counts[60.0] - 1 and counts[60.0] >= 2


def test_the_role_and_fill_numbers_change_what_is_asked():
    quick, default = roles.RoleVoter(lock_after=2), roles.RoleVoter()
    for _ in range(2):
        quick.feed({0: DOCTOR, 1: OTHER})
        default.feed({0: DOCTOR, 1: OTHER})
    assert quick.locked and not default.locked                                           # two agreeing answers lock the first, not the second
    assert fill.chunks([3, 5, 20, 27, 28, 60], 10) == [[3, 5], [20, 27, 28], [60]]
    assert list(fill.window([100], 500, 10, 5, 2)) == list(range(95, 112)) and list(fill.window([100], 105, 10, 5, 2)) == list(range(95, 105))
    tracker = SpeakerTracker(lambda samples: None, Params(fill_batch=2, fill_max_age=1000.0, fill2_batch=3))
    assert tracker.fill2_due([1, 2, 3]) and not tracker.fill2_due([1, 2])


def fabricated(nodes, params=None, history=()):
    """A tracker with two voice groups (doctor, other), a calibration with slope 2 and middle 0, and one closed one-unit sentence per value of
    `nodes`: that value is the sentence's vote (positive = the doctor's side), so a test can say exactly how strong each sentence is."""
    tracker = SpeakerTracker(lambda samples: None, params)
    tracker._centres = np.array([[1.0, 0.0], [0.0, 1.0]])
    tracker._roles = {0: DOCTOR, 1: OTHER}
    tracker._calibration = (-1.0, 1.0, 1.0)
    tracker._history = list(history)
    for i, node in enumerate(nodes):
        d = node / 2.0
        x = (d + math.sqrt(2 - d * d)) / 2.0
        tracker.units.append(Unit(i, i, i * 2.0, i * 2.0 + 1.0, i, np.array([x, x - d])))
        tracker.sentences.append(Sentence(i, i, i * 2.0, i * 2.0 + 1.0, closed=True, units=[i]))
        tracker.stream.tokens.append(Token(1, i, i + 1, i * 2.0, i * 2.0 + 1.0))
    tracker.stream.settled = len(nodes) + 5
    return tracker


def decided(tracker):
    for s in tracker.sentences:
        s.decided = True
    tracker._cursor = len(tracker.sentences)
    return tracker


def test_how_many_past_sentences_are_needed_decides_when_sentences_are_judged_and_against_what():
    waiting = fabricated([2.0, 1.0, 0.5, 0.2], Params(min_history=6))
    waiting._decide()
    assert waiting._cursor == 0                                                          # 0 past + 4 waiting < 6: too little to judge "weak" against
    ready = fabricated([2.0, 1.0, 0.5, 0.2], Params(min_history=3))
    ready._decide()
    assert ready._cursor == 4
    nodes = [0.6, 0.7, 1.5, 2.0]
    history = [0.5] * 5                                                                  # five past sentences, every one 0.5 strong
    from_the_past = fabricated(nodes, Params(min_history=3), history)                    # enough past: the threshold is theirs (0.5)
    from_the_past._decide()
    assert [s.gid for s in from_the_past.sentences] == [0, 0, 0, 0]
    seeded = fabricated(nodes, Params(min_history=8), history)                           # not enough past: the batch is judged against itself (0.645)
    seeded._decide()
    assert [s.gid for s in seeded.sentences] == [None, 0, 0, 0]


def test_the_rescoring_waits_for_as_many_past_sentences_as_the_settings_say():
    for needed, promoted in ((30, 0), (8, 2)):
        tracker = decided(fabricated([0.1, 2.5], Params(min_history=needed), [0.0] * 20))     # 20 past sentences, any vote clears a threshold of 0
        tracker._rescore()
        assert tracker.rescored == promoted and sum(s.gid is not None for s in tracker.sentences) == promoted


def test_the_text_fill_numbers_decide_which_sentences_are_asked_and_when():
    nodes = [0.0] * 12                                                                   # twelve unknown sentences (a vote of 0 is no vote)
    assert decided(fabricated(nodes)).fill_candidates() == [0, 1]
    assert len(decided(fabricated(nodes, Params(fill_after=3))).fill_candidates()) == 9
    assert len(decided(fabricated(nodes, Params(fill_after=3))).fill_candidates(final=True)) == 12
    two = decided(fabricated(nodes, Params(fill_batch=2, fill_max_age=1000.0)))
    assert two.fill_due([0, 1]) and not two.fill_due([0])                                # two waiting are enough; one has not waited 1000 s
    old = decided(fabricated(nodes, Params(fill_batch=50, fill_max_age=10.0)))
    assert old.fill_due([0]) and not decided(fabricated(nodes)).fill_due([0])            # the oldest has waited 22 s, more than 10 s (default: 120 s)
    for refits, wanted in ((1, [0]), (5, [])):
        tracker = decided(fabricated([0.0], Params(fill2_refits=refits)))
        tracker.sentences[0].said, tracker.minutes = DOCTOR, [{}, {}, {}]                # the first answer came back before the first refit; three since
        assert tracker.fill2_candidates() == wanted


def test_a_background_limit_of_zero_switches_the_filter_off_even_for_a_voice_pointing_away_from_every_group():
    for limit, background in ((0.425, True), (0.0, False)):
        tracker = SpeakerTracker(lambda samples: None, Params(background_max_cos=limit))
        tracker._centres = np.array([[1.0, 0.0], [0.0, 1.0]])
        tracker._ref_level = -20.0                                                       # the doctor's usual level
        tracker.units.append(Unit(0, 0, 0.0, 1.0, 0, np.array([-0.7071, -0.7071]), -40.0))     # 20 dB quieter, best cosine -0.71 (a cosine can be negative)
        tracker.sentences.append(Sentence(0, 0, 0.0, 1.0, closed=True, units=[0]))
        assert tracker._is_background(tracker.sentences[0]) is background


def test_the_unit_numbers_are_used_wherever_text_is_cut():
    quiet = Params(units=UnitParams(target=0.5))
    for params, count in ((Params(), 3), (quiet, 6)):
        tracker = SpeakerTracker(lambda samples: None, params)
        tracker.add_tokens([Token(1, i, i + 1, i * 0.25, i * 0.25 + 0.25) for i in range(12)])
        tracker.finish()
        assert len(tracker.units) == count                                               # the tracker's own stream follows its parameters
    tokens = [Token(1, 0, 1, 0.0, 0.5), Token(1, 1, 2, 0.75, 0.875)]                    # a 0.5 s word, a 0.25 s pause, a 0.125 s fragment
    for unit_params, units in ((UnitParams(pause=0.4), 1),                               # the pause is shorter than 0.4: one unit
                               (UnitParams(pause=0.25), 1),                              # exactly the pause: cut, then the fragment joins the previous unit
                               (UnitParams(pause=0.25, max_merged=0.8), 1),              # (the merged unit would be 0.875 s: too long, so the fragment is dropped)
                               (UnitParams(pause=0.25, min_unit=0.1), 2)):               # a 0.125 s fragment is a unit of its own when 0.1 s is enough
        assert len(build_units(tokens, unit_params)) == units, unit_params
    assert build_units(tokens, UnitParams(pause=0.25)) == [[0, 1]] and build_units(tokens, UnitParams(pause=0.25, max_merged=0.8)) == [[0]]
    assert build_units(tokens, UnitParams(pause=0.25, min_unit=0.1)) == [[0], [1]]
    fragment_first = [Token(1, 0, 1, 0.0, 0.125), Token(1, 1, 2, 0.375, 0.875)]          # a 0.125 s fragment, a 0.25 s pause, a 0.5 s word
    assert build_units(fragment_first, UnitParams(pause=0.25)) == [[0, 1]]               # the fragment joins the NEXT unit (pause 0.25 is allowed) ...
    assert build_units(fragment_first, UnitParams(pause=0.125)) == [[1]]                 # ... but not when 0.125 s is the longest pause allowed
    for tokens_, unit_params in ((tokens, UnitParams(pause=0.25, min_unit=0.1)), (fragment_first, UnitParams(pause=0.25, min_unit=0.1))):
        stream = UnitStream(unit_params)
        streamed = stream.add(tokens_) + stream.finish()
        assert streamed == build_units(tokens_, unit_params)                             # the incremental stream follows the same numbers


# --- the whole visit ---------------------------------------------------------------------------------------------------------------
async def test_a_visit_runs_with_the_numbers_of_the_settings_and_says_so(tmp_path):
    session, _ = await run_visit(tmp_path, FakeLLM(), advanced={'refit_seconds': 30.0, 'max_lines': 20}, unknown_percentile=25.0)
    held = session.speaker.effective_tuning()
    assert held['refit_seconds'] == 30.0 and held['max_lines'] == 20 and held['unknown_percentile'] == 25.0
    await session.finish()
    started = next(e for e in session.log.events if e['type'] == 'speaker_started')
    assert started['tuning'] == {'refit_seconds': 30.0, 'max_lines': 20}
    saved = json.loads((session.store.folder / 'speaker' / 'map.json').read_text(encoding='utf-8'))['settings']
    assert saved['tuning'] == {'refit_seconds': 30.0, 'max_lines': 20, 'unknown_percentile': 25.0}
    assert len(session.speaker.tracker.minutes) >= 2


# --- the pop-up -------------------------------------------------------------------------------------------------------------------
async def open_speaker_tab(user: User, tmp_path):
    state = speaker_state(tmp_path, FakeLLM(), enabled=False)

    @ui.page('/')
    def index():
        MainPage(state)

    await user.open('/')
    user.find('模型設定').click()
    await user.should_see('說話者標記', retries=60)
    user.find('說話者標記').click()
    await user.should_see('啟用說話者標記', retries=60)
    await user.should_see('目前全部使用預設值。', retries=60)
    return state


async def open_pop_up(user: User, tmp_path):
    state = await open_speaker_tab(user, tmp_path)
    user.find(marker='speaker-tuning-button').click()
    await user.should_see(marker='tuning-apply', retries=60)
    return state


async def test_the_summary_under_the_button_follows_the_unknown_percentile_field_of_the_page(user: User, tmp_path):
    await open_speaker_tab(user, tmp_path)
    field = marked(user, 'speaker-unknown-percentile')
    field.set_value(20)
    await user.should_see('目前有 1 項參數不是預設值。', retries=60)
    field.set_value(15)
    await user.should_see('目前全部使用預設值。', retries=60)


async def test_the_pop_up_lists_every_number_with_its_default_and_what_raising_or_lowering_does(user: User, tmp_path):
    await open_pop_up(user, tmp_path)
    for knob in tuning.KNOBS:
        field = marked(user, f'tuning-{knob.key}')
        assert field.value == knob.default, knob.key
        await user.should_see(('★ ' if knob.star else '') + knob.label)
        await user.should_see(knob.higher)
        await user.should_see(knob.lower)
    await user.should_see('預設 0.425（0–1）')
    await user.should_see('說話者標記的進階參數')


async def test_changes_in_the_pop_up_are_saved_with_the_settings_and_only_when_saved(user: User, tmp_path):
    state = await open_pop_up(user, tmp_path)
    marked(user, 'tuning-background_max_cos').set_value(0.5)
    marked(user, 'tuning-unknown_percentile').set_value(20)
    marked(user, 'tuning-fill_chunk').set_value(10)
    user.find(marker='tuning-apply').click()
    await user.should_see('目前有 3 項參數不是預設值。', retries=60)
    assert state.settings.speaker.advanced == {}                                         # applied to the dialog only
    user.find('儲存設定').click()
    await until(lambda: state.settings.speaker.advanced)
    assert state.settings.speaker.advanced == {'background_max_cos': 0.5, 'fill_chunk': 10} and state.settings.speaker.unknown_percentile == 20.0
    saved = json.loads(state.config_path.read_text(encoding='utf-8'))['speaker']
    assert saved['advanced'] == {'background_max_cos': 0.5, 'fill_chunk': 10} and saved['unknown_percentile'] == 20.0


async def test_a_bad_value_keeps_the_pop_up_open_and_cancel_and_reset_work(user: User, tmp_path):
    state = await open_pop_up(user, tmp_path)
    marked(user, 'tuning-target').set_value(3.0)                                         # longer than the merged-unit cap
    user.find(marker='tuning-apply').click()
    await user.should_see('單位長度須符合', retries=60)
    marked(user, 'tuning-quiet_gate_db').set_value(12.0)
    user.find(marker='tuning-reset').click()
    await until(lambda: marked(user, 'tuning-target').value == 0.8 and marked(user, 'tuning-quiet_gate_db').value == 10.0)
    marked(user, 'tuning-far_cos').set_value(0.3)
    user.find(marker='tuning-cancel').click()
    await user.should_see('目前全部使用預設值。', retries=60)
    user.find('儲存設定').click()
    await asyncio.sleep(0.5)
    assert state.settings.speaker.advanced == {} and state.settings.speaker.unknown_percentile == 15.0      # nothing was applied


def truthful(messages):
    payload = json.loads(messages[1]['content'])
    rows = {r['id']: r for r in payload['dialogue']}
    return {'answers': [{'id': i, 'role': '醫師' if rows[i]['text'].startswith('請問') else '患者或家屬', 'basis': '問答'} for i in payload['ask_ids']]}


async def test_the_role_and_fill_numbers_limit_what_the_model_is_shown(tmp_path):
    seen = {}
    for name, advanced in (('default', {}), ('small', {'max_lines': 10, 'fill_chunk': 5, 'fill_before': 2, 'fill_after': 1, 'fill_wait': 2, 'fill_batch': 2})):
        fake = FakeLLM()
        fake.script['speaker_fill'] = [truthful] * 60
        session, _ = await run_visit(tmp_path / name, fake, unknown_percentile=40.0, embedder=noisy_embedder, turns=60, advanced=advanced)
        await session.finish()
        asked = [json.loads(m[1]['content']) for m in fake.calls_for('speaker_fill')]
        seen[name] = (max(len(json.loads(m[1]['content'])['transcript_by_voice_cluster']) for m in fake.calls_for('speaker_roles')),
                      max(len(c['ask_ids']) for c in asked), max(len(c['dialogue']) for c in asked), len(asked))
    assert seen['small'][0] <= 10 < seen['default'][0]                             # lines of the dialogue the role question shows
    assert seen['small'][1] <= 5 < seen['default'][1]                                    # sentences asked per fill call
    assert seen['small'][2] <= 5 + 2 + 1 < seen['default'][2]                            # and the context shown with them
    assert seen['small'][3] >= 2
