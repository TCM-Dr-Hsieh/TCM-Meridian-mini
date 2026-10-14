"""Replay: a WAV clip cut out of the recording in progress, and the route that serves it."""
import json
import struct
import wave
from types import SimpleNamespace

import numpy as np
import pytest

from mini.config import Settings
from mini.llm import LLMClient
from mini.state import AppState
from mini.voice.audio import RATE
from mini.voice.replay import CLIP_SECONDS, REPLAY_PATH, ReplayError, clip_wav, register_replay_routes, replay_response
from tests.helpers import FakeASR, FakeLLM, silent_source


def write_ramp(path, seconds: float, *, patch_header: bool = True):
    """A recording whose sample value tells its position: sample k holds k % 30000."""
    total = int(seconds * RATE)
    samples = (np.arange(total) % 30000).astype('<i2')
    with wave.open(str(path), 'wb') as recording:
        recording.setnchannels(1)
        recording.setsampwidth(2)
        recording.setframerate(RATE)
        recording.writeframes(samples.tobytes())
    if not patch_header:                  # what a file being recorded looks like before the first header patch
        raw = bytearray(path.read_bytes())
        raw[4:8] = struct.pack('<I', 36)
        raw[40:44] = struct.pack('<I', 0)
        path.write_bytes(bytes(raw))


def read_clip(data: bytes, tmp_path):
    path = tmp_path / 'clip.wav'
    path.write_bytes(data)
    with wave.open(str(path), 'rb') as clip:
        assert (clip.getnchannels(), clip.getsampwidth(), clip.getframerate()) == (1, 2, RATE)
        return np.frombuffer(clip.readframes(clip.getnframes()), dtype='<i2')


def test_clip_starts_at_the_requested_second_and_lasts_30_seconds(tmp_path):
    write_ramp(tmp_path / 'audio.wav', 100)
    samples = read_clip(clip_wav(tmp_path / 'audio.wav', 15.0), tmp_path)
    assert len(samples) == int(CLIP_SECONDS * RATE)
    assert samples[0] == (15 * RATE) % 30000
    assert samples[-1] == (45 * RATE - 1) % 30000


def test_clip_near_the_end_is_shorter_and_not_padded(tmp_path):
    write_ramp(tmp_path / 'audio.wav', 20)
    samples = read_clip(clip_wav(tmp_path / 'audio.wav', 15.0), tmp_path)
    assert len(samples) == 5 * RATE


def test_fractional_start(tmp_path):
    write_ramp(tmp_path / 'audio.wav', 10)
    samples = read_clip(clip_wav(tmp_path / 'audio.wav', 1.5), tmp_path)
    assert samples[0] == int(1.5 * RATE) % 30000


def test_a_recording_with_an_unpatched_header_is_still_cut_by_file_size(tmp_path):
    write_ramp(tmp_path / 'audio.wav', 12, patch_header=False)
    samples = read_clip(clip_wav(tmp_path / 'audio.wav', 10.0), tmp_path)
    assert len(samples) == 2 * RATE


def test_a_half_written_sample_is_ignored(tmp_path):
    write_ramp(tmp_path / 'audio.wav', 3)
    with open(tmp_path / 'audio.wav', 'ab') as handle:
        handle.write(b'\x01')
    assert len(read_clip(clip_wav(tmp_path / 'audio.wav', 2.0), tmp_path)) == RATE


@pytest.mark.parametrize('start', [3.0, 99.0])
def test_start_past_what_is_recorded_is_refused(tmp_path, start):
    write_ramp(tmp_path / 'audio.wav', 3)
    with pytest.raises(ReplayError) as error:
        clip_wav(tmp_path / 'audio.wav', start)
    assert error.value.status == 404


@pytest.mark.parametrize('start', [-1.0, float('nan'), float('inf')])
def test_bad_start_is_refused(tmp_path, start):
    write_ramp(tmp_path / 'audio.wav', 3)
    with pytest.raises(ReplayError) as error:
        clip_wav(tmp_path / 'audio.wav', start)
    assert error.value.status == 400


def test_missing_or_empty_file_is_refused(tmp_path):
    with pytest.raises(ReplayError):
        clip_wav(tmp_path / 'audio.wav', 0.0)
    (tmp_path / 'audio.wav').write_bytes(b'')
    with pytest.raises(ReplayError):
        clip_wav(tmp_path / 'audio.wav', 0.0)


# -- the route ----------------------------------------------------------------------------------------------

class FakeApp:
    """Collects what register_replay_routes mounts, like FastAPI's `app.get(path)` decorator."""

    def __init__(self):
        self.routes = {}

    def get(self, path):
        def decorate(fn):
            self.routes[path] = fn
            return fn
        return decorate


class Request:
    def __init__(self, **headers):
        self.headers = {key.replace('_', '-').lower(): value for key, value in headers.items()}


TOKEN = 'token-of-visit-A'


def make_state(tmp_path, seconds=60, token=TOKEN):
    write_ramp(tmp_path / 'audio.wav', seconds)
    return SimpleNamespace(visit=SimpleNamespace(replay_token=token, store=SimpleNamespace(audio_path=tmp_path / 'audio.wav')))


def ask(state, start=12.0, token=TOKEN, **headers):
    return replay_response(state, Request(**headers), start, token)


def test_route_is_mounted_and_serves_the_current_visit_clip(tmp_path):
    app, state = FakeApp(), make_state(tmp_path)
    register_replay_routes(app, state)
    response = app.routes[REPLAY_PATH](Request(), 12.0, TOKEN)
    assert response.status_code == 200
    assert response.media_type == 'audio/wav'
    assert response.headers['cache-control'] == 'no-store'
    assert len(read_clip(response.body, tmp_path)) == int(CLIP_SECONDS * RATE)


def test_route_without_a_visit():
    assert ask(SimpleNamespace(visit=None)).status_code == 409


def test_route_reports_why_a_clip_is_not_there(tmp_path):
    response = ask(make_state(tmp_path, seconds=3), start=30.0)
    assert response.status_code == 404
    assert '還沒有錄到' in response.body.decode()


@pytest.mark.parametrize('token', ['', 'token-of-visit-A', 'token-of-visit-Bx'])
def test_route_refuses_a_request_of_another_visit(tmp_path, token):
    """A page left over from visit A must not be handed visit B's audio once B is the current visit."""
    state = make_state(tmp_path, token='token-of-visit-B')
    response = ask(state, token=token)
    assert response.status_code == 409
    assert '另一場看診' in response.body.decode()


def test_route_refuses_when_the_visit_has_no_token(tmp_path):
    assert ask(make_state(tmp_path, token=''), token='').status_code == 409


def test_route_refuses_a_cross_site_request(tmp_path):
    state = make_state(tmp_path, seconds=10)
    assert ask(state, start=0.0, sec_fetch_site='cross-site').status_code == 403
    assert ask(state, start=0.0, sec_fetch_site='same-origin').status_code == 200


@pytest.fixture
def app_state(tmp_path):
    settings = Settings()
    settings.visits_dir = str(tmp_path / 'visits')
    (tmp_path / 'config.json').write_text(json.dumps(settings.to_dict(), ensure_ascii=False), encoding='utf-8')
    templates = tmp_path / 'templates'
    (templates / 'defaults').mkdir(parents=True)
    (templates / 'defaults' / 'record_template.txt').write_text('甲- 現病史：\n', encoding='utf-8')
    (templates / 'defaults' / 'analysis_template.txt').write_text('一- 西醫診斷：\n', encoding='utf-8')
    fake = FakeLLM()
    state = AppState(config_path=tmp_path / 'config.json', templates_dir=templates,
                     client=LLMClient(fake.transport()), asr=FakeASR(default='頭痛'), llm_backoff=0,
                     source_factory=lambda path: silent_source(state.settings, 1, recording_path=path))
    return state


async def test_consecutive_real_visits_have_different_tokens_and_do_not_answer_each_other(app_state):
    state = app_state
    state.import_patient('甲')
    await state.start_visit()
    first = state.visit.replay_token
    assert len(first) >= 16
    assert ask(state, token=first, start=0.0).status_code != 409
    await state.finish_visit()
    state.import_patient('乙')
    await state.start_visit()
    second = state.visit.replay_token
    assert second != first
    assert ask(state, token=first, start=0.0).status_code == 409        # the old page asking while the next visit runs
    assert ask(state, token=second, start=0.0).status_code != 409
    await state.finish_visit()
