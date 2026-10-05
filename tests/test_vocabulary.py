"""The two vocabularies: one hint for the speech recognizer, one spelling reference for the correction LLM.

They used to be a single shared field. Now each consumer receives only its own list, and a config written before the
split keeps working: the corrector starts from the list it used to receive."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from mini.config import MAX_VOCABULARY_CHARS, ASRSettings, Settings
from mini.voice.asr import LocalASR

ASR_ONLY = '只給語音辨識的詞'
CORRECTOR_ONLY = '只給逐字稿校稿的詞'


# ============================ settings ======================================================
def test_both_vocabularies_default_to_empty_and_are_separate_fields():
    s = Settings()
    assert s.asr.vocabulary == '' and s.asr.correction_vocabulary == ''
    s.asr.vocabulary = 'a'
    assert s.asr.correction_vocabulary == ''


def test_the_two_vocabularies_roundtrip_independently(tmp_path):
    path = tmp_path / 'config.json'
    s = Settings()
    s.asr.vocabulary, s.asr.correction_vocabulary = ASR_ONLY, CORRECTOR_ONLY
    s.save(path)
    saved = json.loads(path.read_text(encoding='utf-8'))['asr']
    assert saved['vocabulary'] == ASR_ONLY and saved['correction_vocabulary'] == CORRECTOR_ONLY
    again = Settings.load(path)
    assert (again.asr.vocabulary, again.asr.correction_vocabulary) == (ASR_ONLY, CORRECTOR_ONLY)
    again.asr.vocabulary = '改過的ASR詞'                                        # editing one never touches the other
    again.save(path)
    assert Settings.load(path).asr.correction_vocabulary == CORRECTOR_ONLY


def test_an_old_config_with_one_shared_vocabulary_keeps_it_for_the_corrector(tmp_path):
    """Before the split the corrector received `asr.vocabulary`. An old config.json has no `correction_vocabulary`, so
    the corrector starts from that list instead of silently losing it."""
    path = tmp_path / 'config.json'
    path.write_text(json.dumps({'asr': {'vocabulary': '右寸。右關。右尺。'}}), encoding='utf-8')
    s = Settings.load(path)
    assert s.asr.vocabulary == '右寸。右關。右尺。'
    assert s.asr.correction_vocabulary == '右寸。右關。右尺。'
    s.asr.correction_vocabulary = '只給校稿'                                    # after saving they are independent
    s.save(path)
    again = Settings.load(path)
    assert (again.asr.vocabulary, again.asr.correction_vocabulary) == ('右寸。右關。右尺。', '只給校稿')


def test_an_explicitly_empty_correction_vocabulary_is_not_refilled_from_the_asr_one(tmp_path):
    path = tmp_path / 'config.json'
    path.write_text(json.dumps({'asr': {'vocabulary': '只給ASR', 'correction_vocabulary': ''}}), encoding='utf-8')
    s = Settings.load(path)
    assert s.asr.vocabulary == '只給ASR' and s.asr.correction_vocabulary == ''


def test_a_config_without_any_asr_section_still_loads():
    s = Settings.from_dict({})
    assert (s.asr.vocabulary, s.asr.correction_vocabulary) == ('', '')
    s = Settings.from_dict({'asr': 'not a dict'})
    assert (s.asr.vocabulary, s.asr.correction_vocabulary) == ('', '')


@pytest.mark.parametrize('field, label', [('vocabulary', 'ASR 專有詞'), ('correction_vocabulary', '校稿 LLM 專有詞')])
def test_each_vocabulary_has_its_own_length_limit_and_message(field, label):
    s = Settings()
    setattr(s.asr, field, 'x' * (MAX_VOCABULARY_CHARS + 1))
    with pytest.raises(ValueError, match=label):
        s.validate()
    setattr(s.asr, field, 'x' * MAX_VOCABULARY_CHARS)
    s.validate()                                                                  # exactly at the limit is fine


# ============================ who receives which list ===========================================
async def test_the_correction_llm_gets_only_its_own_vocabulary(make_session):
    h = await make_session(asr__vocabulary=ASR_ONLY, asr__correction_vocabulary=CORRECTOR_ONLY)
    calls = h.fake.calls_for('corrector')
    assert calls, 'the transcript was never sent to the correction LLM'
    for system, user in calls:
        assert CORRECTOR_ONLY in system['content'] and '領域常見詞彙' in system['content']
        assert json.loads(user['content'])['vocabulary'] == CORRECTOR_ONLY
        assert ASR_ONLY not in system['content'] and ASR_ONLY not in user['content']


async def test_without_a_correction_vocabulary_nothing_is_sent_even_if_the_asr_one_is_set(make_session):
    h = await make_session(asr__vocabulary=ASR_ONLY, asr__correction_vocabulary='')
    calls = h.fake.calls_for('corrector')
    assert calls
    for system, user in calls:
        assert '領域常見詞彙' not in system['content'] and ASR_ONLY not in system['content']
        assert json.loads(user['content'])['vocabulary'] == ''


async def test_the_speech_recognizer_gets_only_the_asr_vocabulary():
    """What LocalASR writes to its worker: `context` is the ASR list, and the correction list never reaches it."""
    asr = LocalASR()
    written = []

    class Stdin:
        def write(self, data):
            written.append(data)

        async def drain(self):
            pass

    async def ensure(settings):
        return {}

    async def read():
        return {'text': '好', 'items': []}

    asr.process, asr._ensure, asr._read = SimpleNamespace(stdin=Stdin()), ensure, read
    settings = ASRSettings(vocabulary=ASR_ONLY, correction_vocabulary=CORRECTOR_ONLY)
    await asr.recognize(settings, np.zeros(1600, dtype=np.float32))
    request = json.loads(written[0])
    assert request['context'] == ASR_ONLY
    assert CORRECTOR_ONLY not in written[0].decode('utf-8')
