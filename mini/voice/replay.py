"""Replay of the visit that is being recorded: a short WAV clip cut out of `audio.wav` while it is still growing.

The physician clicks `#n mm:ss` in the transcript to listen again to what was said. The browser asks for the clip that
starts at that audio time (`start`, seconds since the visit began, the same clock the transcript shows) and plays it
in a floating window.

The clip is cut here, not streamed from the file, for two reasons: `audio.wav` is still being written (its header
only holds the length as of the last patch, so a browser given the file itself would show a wrong, stale duration),
and a clip of its own length gives the player's time bar the right range. The sample count is therefore taken from
the file size, never from the header, and a clip that reaches past what has been recorded so far is simply shorter
(no silence is added).

Only the current visit's audio can be asked for, by time only: there is no path in the request. The request must also carry
the visit's random `replay_token` (the transcript rows of that visit hold it). The server answers from whatever visit is
current when the request arrives, so without the token a page left over from patient A, while patient B's visit has
started, would be handed B's recording at the time it clicked in A's transcript. A stale page is refused instead.
"""
from __future__ import annotations

import io
import math
import secrets
import struct
import wave
from pathlib import Path

from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response

from .audio import RATE

REPLAY_PATH = '/mini-replay.wav'
CLIP_SECONDS = 30.0
SAMPLE_BYTES = 2                    # 16-bit mono, as AudioSource and RemoteAudioSource write it


class ReplayError(Exception):
    """A request that cannot be answered; `status` is the HTTP status to send, the message is shown to the physician."""

    def __init__(self, message: str, status: int = 404):
        super().__init__(message)
        self.status = status


def _data_offset(handle) -> tuple[int, int]:
    """(offset of the first sample, sample rate) found by walking the RIFF chunks. The sizes in a header that is still
    being written are not trusted: only where the chunks start."""
    head = handle.read(12)
    if len(head) < 12 or head[:4] != b'RIFF' or head[8:12] != b'WAVE':
        raise ReplayError('錄音檔還沒有可用的內容，請稍後再試。')
    rate = RATE
    position = 12
    while True:
        handle.seek(position)
        chunk = handle.read(8)
        if len(chunk) < 8:
            raise ReplayError('錄音檔還沒有可用的內容，請稍後再試。')
        kind, size = chunk[:4], struct.unpack('<I', chunk[4:])[0]
        if kind == b'fmt ':
            body = handle.read(min(size, 16))
            if len(body) < 16:
                raise ReplayError('錄音檔格式不完整。', 500)
            tag, channels, rate, _, _, bits = struct.unpack('<HHIIHH', body)
            if tag != 1 or channels != 1 or bits != 16:
                raise ReplayError('錄音檔格式不是 16 位元單聲道，無法回放。', 500)
        elif kind == b'data':
            return position + 8, rate
        position += 8 + size + (size & 1)


def clip_wav(path: Path, start: float, seconds: float = CLIP_SECONDS) -> bytes:
    """A complete WAV of up to `seconds` of the recording from `start` (seconds into the visit)."""
    if not math.isfinite(start) or start < 0:
        raise ReplayError('起點時間不合法。', 400)
    try:
        with open(path, 'rb') as handle:
            offset, rate = _data_offset(handle)
            handle.seek(0, 2)
            available = max(0, handle.tell() - offset) // SAMPLE_BYTES
            first = int(start * rate)
            if first >= available:
                raise ReplayError('這個時間點的聲音還沒有錄到。')
            count = min(int(seconds * rate), available - first)
            handle.seek(offset + first * SAMPLE_BYTES)
            pcm = handle.read(count * SAMPLE_BYTES)
    except FileNotFoundError:
        raise ReplayError('錄音檔還不存在。') from None
    out = io.BytesIO()
    with wave.open(out, 'wb') as clip:
        clip.setnchannels(1)
        clip.setsampwidth(SAMPLE_BYTES)
        clip.setframerate(rate)
        clip.writeframes(pcm[:len(pcm) // SAMPLE_BYTES * SAMPLE_BYTES])
    return out.getvalue()


def replay_response(state, request: Request, start: float, token: str = '') -> Response:
    """The clip of the visit in progress for `state` (an AppState), as an HTTP response. `token` must be that visit's replay
    token. The `Sec-Fetch-Site` check only rejects what the browser itself labels cross-site; the token is what keeps
    other pages (and stale ones) out."""
    if request.headers.get('sec-fetch-site') == 'cross-site':
        return PlainTextResponse('不允許跨網站讀取。', status_code=403)
    visit = state.visit
    if visit is None:
        return PlainTextResponse('目前沒有進行中的看診。', status_code=409, headers={'Cache-Control': 'no-store'})
    expected = getattr(visit, 'replay_token', '')
    if not expected or not secrets.compare_digest(str(token).encode(), expected.encode()):
        return PlainTextResponse('這個畫面屬於另一場看診（或已過期），無法回聽；請重新整理頁面。', status_code=409,
                                 headers={'Cache-Control': 'no-store'})
    try:
        body = clip_wav(visit.store.audio_path, start)
    except ReplayError as exc:
        return PlainTextResponse(str(exc), status_code=exc.status, headers={'Cache-Control': 'no-store'})
    return Response(body, media_type='audio/wav', headers={'Cache-Control': 'no-store'})


def register_replay_routes(app, state) -> None:
    """Mount the clip route on a FastAPI-style app (NiceGUI's `app`, or a plain FastAPI app in tests)."""
    @app.get(REPLAY_PATH)
    def replay(request: Request, start: float = 0.0, token: str = ''):
        return replay_response(state, request, start, token)
