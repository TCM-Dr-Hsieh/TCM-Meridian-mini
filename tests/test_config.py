import json

import pytest

from mini.config import AGENT_KEYS, DEFAULT_MODEL, DEFAULT_URL, Settings


def test_defaults_match_the_spec():
    s = Settings()
    assert set(s.agents) == set(AGENT_KEYS)
    assert all(e.api_url == DEFAULT_URL and e.model_name == DEFAULT_MODEL and e.api_key == 'no-key'
               for e in s.agents.values())
    assert s.agents['transcript_corrector'].temperature == 0.2
    assert s.agents['professor_c'].temperature == 0.3
    assert s.llm.max_concurrency == 2 and s.llm.retries == 3
    assert (s.asr.window_seconds, s.asr.overlap_seconds, s.asr.context_chars) == (6.0, 3.0, 1000)
    assert (s.review.pass_required_n, s.review.max_review_rounds) == (2, 6)
    assert s.professors['a'].name == '教授甲' and '嚴謹' in s.professors['a'].role_style
    assert '溫和' in s.professors['b'].role_style and s.professors['c'].role_style == ''


def test_first_load_writes_defaults_and_roundtrips(tmp_path):
    path = tmp_path / 'config.json'
    s = Settings.load(path)
    assert path.exists()
    s.agents['record_writer'].temperature = 0.5
    s.professors['a'].role_style = '自訂風格'
    s.save(path)
    again = Settings.load(path)
    assert again.agents['record_writer'].temperature == 0.5
    assert again.professors['a'].role_style == '自訂風格'


def test_load_tolerates_unknown_and_missing_fields(tmp_path):
    path = tmp_path / 'config.json'
    path.write_text(json.dumps({'llm': {'max_concurrency': 3, 'bogus': 1}, 'agents': {'nope': {}, 'ai_advice': {
        'model_name': 'm2'}}, 'extra': 1}), encoding='utf-8')
    s = Settings.load(path)
    assert s.llm.max_concurrency == 3
    assert s.agents['ai_advice'].model_name == 'm2' and s.agents['ai_advice'].api_url == DEFAULT_URL


def test_public_dict_never_contains_keys():
    s = Settings()
    s.agents['record_writer'].api_key = 'secret-123'
    assert 'secret-123' not in json.dumps(s.public_dict())
    assert 'api_key' not in s.public_dict()['agents']['record_writer']


@pytest.mark.parametrize('mutate, message', [
    (lambda s: setattr(s.review, 'pass_required_n', 9), 'n'),
    (lambda s: setattr(s.asr, 'overlap_seconds', 5.0), '重疊'),
    (lambda s: setattr(s.asr, 'window_seconds', 3), '視窗'),
    (lambda s: setattr(s.agents['ai_advice'], 'api_url', 'ftp://x'), 'URL'),
    (lambda s: setattr(s.agents['ai_advice'], 'model_name', ' '), '模型'),
    (lambda s: setattr(s.llm, 'max_concurrency', 0), '並行'),
])
def test_validation_rejects_bad_values(mutate, message):
    s = Settings()
    mutate(s)
    with pytest.raises(ValueError) as error:
        s.validate()
    assert message in str(error.value)


def test_n_zero_means_no_review_and_is_valid():
    s = Settings()
    s.review.pass_required_n = 0
    s.validate()
    assert s.review.pass_required_n == 0


def test_context_tokens_defaults_roundtrips_and_validates(tmp_path):
    from mini.config import DEFAULT_CONTEXT_TOKENS
    s = Settings()
    assert all(e.context_tokens == DEFAULT_CONTEXT_TOKENS for e in s.agents.values())
    s.agents['record_writer'].context_tokens = 131072
    s.agents['ai_advice'].context_tokens = 0                          # 0 = no check
    path = tmp_path / 'config.json'
    s.save(path)
    again = Settings.load(path)
    assert again.agents['record_writer'].context_tokens == 131072 and again.agents['ai_advice'].context_tokens == 0
    for bad in (-1, 5_000_000, 1.5, 'x', True):
        s.agents['record_writer'].context_tokens = bad
        with pytest.raises(ValueError, match='context'):
            s.validate()


def test_old_config_without_context_tokens_gets_the_default(tmp_path):
    from mini.config import DEFAULT_CONTEXT_TOKENS
    path = tmp_path / 'config.json'
    path.write_text(json.dumps({'agents': {'ai_advice': {'model_name': 'm'}}}), encoding='utf-8')
    assert Settings.load(path).agents['ai_advice'].context_tokens == DEFAULT_CONTEXT_TOKENS


def test_the_web_ui_listens_on_all_interfaces_by_default_and_can_be_restricted():
    from mini.config import DEFAULT_HOST, DEFAULT_PORT, server_binding
    assert (DEFAULT_HOST, DEFAULT_PORT) == ('0.0.0.0', 5050)
    assert server_binding({}) == ('0.0.0.0', 5050)                                   # LAN / tunnel reachable
    assert server_binding({'MINI_HOST': '127.0.0.1'}) == ('127.0.0.1', 5050)         # back to this computer only
    assert server_binding({'MINI_HOST': ' 192.168.1.20 ', 'MINI_PORT': ' 8080 '}) == ('192.168.1.20', 8080)
    assert server_binding({'MINI_HOST': '', 'MINI_PORT': ''}) == ('0.0.0.0', 5050)    # empty values mean "default"


@pytest.mark.parametrize('port', ['abc', '0', '70000', '-1', '50.5'])
def test_a_bad_port_is_reported_clearly_instead_of_crashing_later(port):
    from mini.config import server_binding
    with pytest.raises(ValueError, match='MINI_PORT'):
        server_binding({'MINI_PORT': port})
