/* Browser-side protocol smoke test for mini/ui/remote_capture.js.
 *
 * Run with `node tests/remote_capture_smoke.js` (tests/test_remote_capture_js.py does it when Node.js is installed).
 * No microphone, browser or server is needed: everything the script touches is faked, and the test decides when each
 * timer fires. Ported from voice_to_text's tests/remote_capture_smoke.js and extended for this project's flow
 * (begin -> prepared -> connect -> ready -> stream -> stop). */
const assert = require('node:assert/strict');

function define(name, value) {
  Object.defineProperty(globalThis, name, {value, configurable: true, writable: true});
}

// ---- the page's globals ---------------------------------------------------------------------------
const events = [];                                   // what the page is told: emitEvent('mini_remote', {type, ...})
define('emitEvent', (name, payload) => { if (name === 'mini_remote') events.push(payload); });
define('window', {isSecureContext: true});
define('document', {documentElement: {dataset: {}}});
define('location', {protocol: 'https:', host: 'mini.example'});
let uuids = 0;
define('crypto', {randomUUID: () => `session-${String(++uuids).padStart(8, '0')}`});
define('Blob', class { constructor(parts, options) { this.parts = parts; this.options = options; } });
define('URL', {createObjectURL: () => 'blob:worklet', revokeObjectURL() {}});

// ---- manual timers: nothing fires unless the test says so ---------------------------------------------
const timers = new Map();
let nextTimer = 1;
define('setTimeout', (fn, ms) => { const id = nextTimer++; timers.set(id, {fn, ms}); return id; });
define('clearTimeout', id => { timers.delete(id); });
define('setInterval', (fn, ms) => { const id = nextTimer++; timers.set(id, {fn, ms, repeat: true}); return id; });
define('clearInterval', id => { timers.delete(id); });
function timerWith(ms) {
  for (const [id, timer] of timers) if (timer.ms === ms && !timer.repeat) return id;
  return null;
}
function fire(ms) {
  const id = timerWith(ms);
  assert.ok(id !== null, `no ${ms} ms timer is pending`);
  const {fn} = timers.get(id);
  timers.delete(id);
  return fn();
}

// ---- the WebSocket to the server ------------------------------------------------------------------------
class FakeWebSocket {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;
  static sockets = [];
  constructor(url) {
    this.url = url;
    this.readyState = 0;
    this.bufferedAmount = 0;
    this.sent = [];
    FakeWebSocket.sockets.push(this);
  }
  open() { this.readyState = FakeWebSocket.OPEN; this.onopen(); }
  send(value) { this.sent.push(value); }
  async message(value) { await this.onmessage({data: JSON.stringify(value)}); }
  close() {
    if (this.readyState === FakeWebSocket.CLOSED) return;
    this.readyState = FakeWebSocket.CLOSED;
    if (this.onclose) this.onclose();
  }
  texts() { return this.sent.filter(v => typeof v === 'string').map(v => JSON.parse(v)); }
  packets() { return this.sent.filter(v => typeof v !== 'string').map(v => new DataView(v).getUint32(0, true)); }
}
define('WebSocket', FakeWebSocket);

// ---- the microphone and the audio graph ---------------------------------------------------------------------
const userMediaCalls = [];
let releaseUserMedia = null;                         // answers the pending permission prompt
let nextUserMedia = 'grant';
let lastTrack = null;
define('navigator', {mediaDevices: {
  getUserMedia(constraints) {
    userMediaCalls.push(constraints);
    return new Promise((resolve, reject) => {
      releaseUserMedia = () => {
        if (nextUserMedia === 'deny') {
          const error = new Error('Permission denied');
          error.name = 'NotAllowedError';
          reject(error);
          return;
        }
        const track = {label: '測試麥克風', stopped: false, listeners: {},
          addEventListener(name, fn) { this.listeners[name] = fn; }, stop() { this.stopped = true; }};
        lastTrack = track;
        resolve({getAudioTracks: () => [track], getTracks: () => [track]});
      };
    });
  },
}});
const contexts = [];
define('AudioContext', class {
  constructor() {
    this.sampleRate = 48000;
    this.closed = false;
    this.sources = [];
    this.destination = {};
    this.audioWorklet = {addModule: async () => {}};
    contexts.push(this);
  }
  async resume() {}
  createMediaStreamSource(stream) {
    const source = {stream, connectedTo: null, connect(node) { this.connectedTo = node; }};
    this.sources.push(source);
    return source;
  }
  close() { this.closed = true; }
});
define('AudioWorkletNode', class {
  constructor(context, name) {
    this.name = name;
    this.disconnected = false;
    this.posted = [];
    const node = this;
    this.port = {onmessage: null, postMessage(message) { node.posted.push(message); }};
  }
  connect() {}
  disconnect() { this.disconnected = true; }
});

require('../mini/ui/remote_capture.js');

const kinds = type => events.filter(event => event.type === type);
const worklet = (capture, value) => capture.node.port.onmessage({data: new Float32Array([value, value])});

async function main() {
  const m = window.miniRemote;

  // 1. Two clicks while the permission prompt is still open must open ONE capture.
  const first = m.begin();
  const second = m.begin();
  assert.equal(userMediaCalls.length, 1, 'the second click must not open a second capture');
  releaseUserMedia();
  await first;
  await second;
  assert.equal(userMediaCalls.length, 1);
  assert.equal(kinds('prepared').length, 1);
  assert.equal(kinds('prepared')[0].sample_rate, 48000);
  assert.equal(kinds('prepared')[0].label, '測試麥克風');
  assert.equal(contexts.length, 1);
  await m.begin();                                   // a click once prepared does nothing either
  assert.equal(userMediaCalls.length, 1);

  // 2. connect(): the first hello is 'start'; sound only flows once the server says 'ready'.
  m.connect('tok-123');
  const s1 = FakeWebSocket.sockets[0];
  assert.equal(s1.url, 'wss://mini.example/mini-remote-audio/tok-123');
  s1.open();
  assert.deepEqual(s1.texts()[0], {type: 'start', session_id: m.sessionId});
  assert.equal(m.streaming, false);
  await s1.message({type: 'ready', next_sequence: 0});
  assert.equal(m.streaming, true);
  assert.equal(contexts[0].sources[0].connectedTo, m.node);
  assert.deepEqual(s1.texts()[1], {type: 'format', sample_rate: 48000});
  assert.equal(kinds('recording').length, 1);

  // 3. Packets are numbered in order; an ACK drops them from the buffer.
  for (const value of [.1, .2, .3]) worklet(m, value);
  assert.deepEqual(s1.packets(), [0, 1, 2]);
  assert.equal(m.frames.length, 3);
  await s1.message({type: 'ack', sequence: 1});
  assert.equal(m.frames.length, 1);

  // 4. A lost connection: the page is told, capture continues, and the retry RESUMES the same recording and replays
  //    only what the server has not stored. Here the server DID store packet 2 but its ACK was lost with the
  //    connection, so it asks for 3 onwards: packet 2 must not be sent (and written) a second time.
  s1.close();
  assert.equal(kinds('reconnecting').length, 1);
  assert.equal(m.socket, null);
  worklet(m, .4);                                    // still capturing while offline
  assert.equal(m.frames.length, 2);                  // packet 2 (never acknowledged) and packet 3 (new)
  assert.ok(timerWith(30000) !== null, 'the 30 s reconnect deadline is running');
  fire(700);                                         // the retry timer
  const s2 = FakeWebSocket.sockets[1];
  s2.open();
  assert.deepEqual(s2.texts()[0], {type: 'resume', session_id: m.sessionId});
  await s2.message({type: 'resumed', next_sequence: 3});
  assert.deepEqual(s2.packets(), [3]);
  assert.equal(m.frames.length, 1, 'packet 2 is now known to be stored and leaves the buffer');
  assert.equal(kinds('reconnected').length, 1);
  assert.equal(kinds('recording').length, 1, 'a reconnect is not a new recording');
  assert.equal(timerWith(30000), null, 'the deadline is cancelled once reconnected');

  // 5. The visit ends: the server asks for the tail. The browser flushes the worklet, releases the microphone, waits
  //    for the outstanding ACKs and only THEN says 'stop' (otherwise the end of the recording would be cut off).
  const stopping = s2.message({type: 'stop'});
  assert.ok(m.node.posted.includes('finish'), 'the worklet is told to flush its last buffer');
  m.node.port.onmessage({data: 'finished'});
  await stopping;
  assert.equal(lastTrack.stopped, true, 'the microphone is released');
  assert.equal(s2.texts().some(text => text.type === 'stop'), false, 'must wait for the unacknowledged packets');
  await s2.message({type: 'ack', sequence: 3});
  assert.equal(s2.texts().at(-1).type, 'stop');
  await s2.message({type: 'finished'});
  assert.equal(m.sessionId, null);
  assert.equal(kinds('finished').length, 1);
  assert.equal(contexts[0].closed, true);

  // 6. A refused microphone: a clear message, nothing left open, and a later attempt is not blocked.
  nextUserMedia = 'deny';
  const denied = m.begin();
  releaseUserMedia();
  await denied;
  assert.match(kinds('error').at(-1).message, /麥克風權限/);
  assert.equal(m.prepared, false);
  assert.equal(m.preparing, false);
  nextUserMedia = 'grant';

  // 7. Not a secure context (plain http): refused before the microphone is even asked for.
  const asked = userMediaCalls.length;
  window.isSecureContext = false;
  await m.begin();
  assert.equal(userMediaCalls.length, asked);
  assert.match(kinds('error').at(-1).message, /HTTPS/);
  window.isSecureContext = true;

  // 8. The server could not start the visit: abort() lets go of the microphone.
  const aborting = m.begin();
  releaseUserMedia();
  await aborting;
  const abortedTrack = lastTrack;
  const abortedContext = contexts.at(-1);
  m.abort();
  assert.equal(abortedTrack.stopped, true);
  assert.equal(abortedContext.closed, true);
  assert.equal(m.prepared, false);

  // 9. Never connected at all (a proxy rewrote the Host header, WebSockets are blocked): a quick, specific error, and
  //    no "reconnecting" announcement for a recording that never started.
  const announced = kinds('reconnecting').length;
  const early = m.begin();
  releaseUserMedia();
  await early;
  m.connect('tok-456');
  const dead = FakeWebSocket.sockets.at(-1);
  dead.open();
  dead.close();                                      // the server refuses before 'ready'
  assert.equal(kinds('reconnecting').length, announced);
  assert.ok(timerWith(10000) !== null && timerWith(30000) === null);
  fire(10000);
  assert.match(kinds('error').at(-1).message, /WebSocket/);
  assert.equal(m.sessionId, null);

  // 10. An established recording that cannot reconnect for 30 s ends with the other message.
  const established = m.begin();
  releaseUserMedia();
  await established;
  m.connect('tok-789');
  const live = FakeWebSocket.sockets.at(-1);
  live.open();
  await live.message({type: 'ready', next_sequence: 0});
  live.close();
  assert.ok(timerWith(30000) !== null);
  fire(30000);
  assert.match(kinds('error').at(-1).message, /30 秒/);
  assert.equal(m.sessionId, null);

  console.log('remote_capture smoke test: ok');
}

main().catch(error => { console.error(error); process.exitCode = 1; });
