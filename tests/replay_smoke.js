/* Browser-side smoke test for mini/ui/replay.js (the replay window).
 *
 * Run with `node tests/replay_smoke.js` (tests/test_replay_js.py does it when Node.js is installed). No browser or server:
 * the DOM, Audio, fetch and timers are faked and the test decides when each fetch answers and when the timer fires.
 * Covers what the Python tests cannot see: the clip request carries the clicked row's start and the visit token, a late
 * answer of an earlier click never replaces a newer one, an error from the server is shown and nothing plays, the player
 * closes by itself when the transcript on screen belongs to another visit (or is gone), the start field, the time bar. */
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function define(name, value) {
  Object.defineProperty(globalThis, name, {value, configurable: true, writable: true});
}

// ---- a tiny DOM -----------------------------------------------------------------------------------------
const byTestId = {};
class Element {
  constructor(tag) {
    this.tag = tag; this.children = []; this.style = {}; this.dataset = {}; this.attrs = {}; this.listeners = {};
    this.className = ''; this._text = ''; this.value = ''; this.disabled = false;
  }
  set textContent(v) { this._text = String(v); this.children = []; }
  get textContent() { return this._text + this.children.map(c => c.textContent).join(''); }
  get firstChild() { return this.children[0]; }
  appendChild(c) { this.children.push(c); return c; }
  append(...cs) { cs.forEach(c => this.children.push(c)); }
  setAttribute(k, v) { this.attrs[k] = v; if (k === 'data-testid') byTestId[v] = this; }
  addEventListener(name, fn) { (this.listeners[name] ||= []).push(fn); }
  fire(name, event = {}) { (this.listeners[name] || []).forEach(fn => fn(event)); }
}

let marks = [];                                        // the `#n mm:ss` spans of the transcript on screen
let rows = [];                                         // their rows
function setTranscript(token, starts = [12, 15, 18]) {
  marks = []; rows = [];
  starts.forEach((start, i) => {
    const classes = new Set(['seg']);
    const mark = {dataset: {start: start.toFixed(2), idx: String(i + 1), token}, textContent: `#${i + 1} 00:${start}`,
                  closest() { return mark; }};
    rows.push({dataset: {start: start.toFixed(2), idx: String(i + 1)}, classList: {add: c => classes.add(c), remove: c => classes.delete(c),
               has: c => classes.has(c)}, querySelector: () => mark,
               cloneNode: () => ({querySelector: () => ({remove() {}}), textContent: ` 不明：第 ${i + 1} 句 `})});
    marks.push(mark);
  });
}
const body = new Element('body');
const documentListeners = {};
define('document', {
  body,
  createElement: tag => new Element(tag),
  addEventListener: (name, fn) => { (documentListeners[name] ||= []).push(fn); },
  querySelector: sel => (sel === '.seg-no[data-token]' ? marks[0] || null : null),
  querySelectorAll: sel => (sel === '.seg[data-start]' ? rows : sel === '.seg.replaying' ? rows.filter(r => r.classList.has('replaying')) : []),
});
define('window', {miniFloat: {show() {}}});

// ---- audio, fetch, URL, timers ----------------------------------------------------------------------------
class FakeAudio {
  constructor() { this.listeners = {}; this.paused = true; this.ended = false; this.currentTime = 0; this.duration = NaN; this.src = ''; this.plays = 0; }
  addEventListener(name, fn) { (this.listeners[name] ||= []).push(fn); }
  fire(name) { (this.listeners[name] || []).forEach(fn => fn()); }
  play() { this.plays++; this.paused = false; return Promise.resolve(); }
  pause() { this.paused = true; }
  load() {}
  removeAttribute(name) { if (name === 'src') this.src = ''; }
}
let audio = null;
define('Audio', class extends FakeAudio { constructor() { super(); audio = this; } });
const pending = [];                                    // fetches waiting for the test to answer them
define('fetch', url => new Promise(resolve => pending.push({url, resolve})));
let blobs = 0;
define('URL', {createObjectURL: () => `blob:${++blobs}`, revokeObjectURL() {}});
const timers = new Map();
let nextTimer = 1;
define('setInterval', (fn, ms) => { const id = nextTimer++; timers.set(id, {fn, ms}); return id; });
define('clearInterval', id => { timers.delete(id); });
function tick() { for (const {fn} of [...timers.values()]) fn(); }
const flush = () => new Promise(resolve => setImmediate(resolve));
const okResponse = () => ({ok: true, blob: async () => ({}), text: async () => ''});
const errorResponse = text => ({ok: false, blob: async () => ({}), text: async () => text});

vm.runInThisContext(fs.readFileSync(path.join(__dirname, '..', 'mini', 'ui', 'replay.js'), 'utf8'), {filename: 'replay.js'});
const replay = window.miniReplay;
const click = i => documentListeners.click.forEach(fn => fn({target: marks[i]}));
const panel = () => body.children.find(c => c.id === 'mini-replay');
const shown = () => panel() && panel().style.display !== 'none';

(async () => {
  // parsing of the start field
  assert.equal(replay.parseStart('01:20'), 80);
  assert.equal(replay.parseStart('1:02:03'), 3723);
  assert.equal(replay.parseStart('12.5'), 12.5);
  assert.equal(replay.parseStart('abc'), null);
  assert.equal(replay.clock(75), '01:15');

  // a click on a click target outside the transcript does nothing
  documentListeners.click.forEach(fn => fn({target: {closest: () => null}}));
  assert.equal(panel(), undefined);

  // 1. a click opens the window and asks for the clip at the row's exact start with the visit's token
  setTranscript('TOKEN-A');
  click(1);
  assert.ok(shown());
  assert.equal(pending.length, 1);
  assert.match(pending[0].url, /^\/mini-replay\.wav\?start=15\.00&token=TOKEN-A$/);
  assert.equal(byTestId['replay-msg'].textContent, '載入中…');
  assert.match(panel().children[0].children[0].textContent, /從 00:15 起/);
  assert.equal(byTestId['replay-start'].value, '00:15');
  assert.ok(panel().children[1].children.some(c => /請使用耳機/.test(c.textContent)), 'earphone reminder is on the window');
  pending[0].resolve(okResponse());
  await flush();
  assert.equal(audio.src, 'blob:1');
  assert.equal(audio.plays, 1);
  assert.equal(byTestId['replay-msg'].textContent, '');

  // the time bar follows the clip and a drag seeks
  audio.duration = 30; audio.fire('loadedmetadata');
  assert.equal(byTestId['replay-bar'].max, '30');
  byTestId['replay-bar'].value = '12'; byTestId['replay-bar'].fire('input');
  assert.equal(audio.currentTime, 12);
  assert.match(panel().children[1].children[2].children[2].textContent, /0:12 \/ 0:30/);

  // the row being played is highlighted (15 s + 12 s = 27 s: the last row that starts before it is #3 at 18 s)
  tick();
  assert.ok(rows[2].classList.has('replaying') && !rows[1].classList.has('replaying'));

  // pause / resume
  byTestId['replay-toggle'].fire('click');
  assert.equal(audio.paused, true);
  byTestId['replay-toggle'].fire('click');
  assert.equal(audio.paused, false);

  // 2. a click on another row while open: same window, new request; an earlier, slower answer is dropped
  pending.length = 0; blobs = 0;
  click(0);                                            // slow one (start 12)
  click(2);                                            // the one that counts (start 18)
  assert.equal(pending.length, 2);
  assert.match(pending[1].url, /start=18\.00/);
  pending[1].resolve(okResponse()); await flush();
  pending[0].resolve(okResponse()); await flush();
  assert.equal(blobs, 1, 'the late answer of the first click must not become a clip');
  assert.equal(audio.src, 'blob:1');
  assert.match(panel().children[0].children[0].textContent, /從 00:18 起/);

  // 2b. a slow ERROR of an earlier click must not overwrite what the newer click shows
  pending.length = 0; blobs = 0;
  click(0);
  click(2);
  pending[1].resolve(okResponse()); await flush();
  pending[0].resolve(errorResponse('STALE-ERROR')); await flush();
  assert.equal(byTestId['replay-msg'].textContent, '', 'the stale error must not be shown');
  assert.equal(blobs, 1);

  // 2c. the body of an earlier click's clip is still downloading when another click comes: that clip is dropped
  pending.length = 0; blobs = 0;
  let finishBody;
  const slowBody = new Promise(resolve => { finishBody = resolve; });
  click(0);
  pending[0].resolve({ok: true, blob: () => slowBody, text: async () => ''}); await flush();   // headers are in, the body is not
  click(2);
  finishBody({}); await flush();
  assert.equal(blobs, 0, 'the clip that finished downloading after another click must not be used');
  pending[1].resolve(okResponse()); await flush();
  assert.equal(blobs, 1);
  assert.equal(audio.src, 'blob:1');

  // 2d. the HEADERS of an error are in but its body is still arriving when a newer click loads and plays: when the old
  // body finally comes, it must not replace what the newer clip shows
  pending.length = 0; blobs = 0;
  let finishErrorBody;
  const slowErrorBody = new Promise(resolve => { finishErrorBody = resolve; });
  click(0);
  pending[0].resolve({ok: false, blob: async () => ({}), text: () => slowErrorBody}); await flush();
  click(2);
  pending[1].resolve(okResponse()); await flush();
  assert.equal(audio.src, 'blob:1');
  finishErrorBody('SLOW-STALE-ERROR'); await flush();
  assert.equal(byTestId['replay-msg'].textContent, '', 'the old error body must not be shown over the newer clip');
  assert.equal(audio.src, 'blob:1', 'and the newer clip is still the one loaded');

  // 3. the server refuses (a page of another visit): the reason is shown and nothing plays
  pending.length = 0; const playsBefore = audio.plays;
  click(1);
  pending[0].resolve(errorResponse('這個畫面屬於另一場看診（或已過期），無法回聽；請重新整理頁面。')); await flush();
  assert.match(byTestId['replay-msg'].textContent, /另一場看診/);
  assert.equal(audio.src, '', 'the previous clip is gone after a failed load');
  assert.equal(audio.plays, playsBefore);
  assert.equal(byTestId['replay-bar'].disabled, true);

  // 4. the start field: a bad value is reported without a request, a good one asks for that second
  pending.length = 0;
  byTestId['replay-start'].value = 'abc'; byTestId['replay-go'].fire('click');
  assert.match(byTestId['replay-msg'].textContent, /起點格式不對/);
  assert.equal(pending.length, 0);
  byTestId['replay-start'].value = '05:00'; byTestId['replay-start'].fire('keydown', {key: 'Enter'});
  assert.match(pending[0].url, /start=300\.00&token=TOKEN-A/);
  pending[0].resolve(okResponse()); await flush();
  const clipsBeforeClose = blobs;

  // 5. the transcript on screen becomes another visit's (no empty state in between): the player closes by itself
  assert.ok(shown());
  setTranscript('TOKEN-B');
  tick();
  assert.equal(shown(), false);
  assert.equal(audio.paused, true);
  assert.equal(audio.src, '');
  assert.equal(timers.size, 0, 'the refresh timer stops with the window');
  assert.equal(rows.filter(r => r.classList.has('replaying')).length, 0);
  assert.ok(clipsBeforeClose >= 1);

  // a click on the new visit's row opens it again with the new token
  pending.length = 0;
  click(0);
  assert.ok(shown());
  assert.match(pending[0].url, /token=TOKEN-B/);
  pending[0].resolve(okResponse()); await flush();

  // 6. the transcript goes away (visit over): closes too
  marks = []; rows = [];
  tick();
  assert.equal(shown(), false);

  // 7. the close button, and a clip that arrives after closing is dropped
  setTranscript('TOKEN-B');
  pending.length = 0; blobs = 0;
  click(1);
  byTestId['replay-close'].fire('click');
  assert.equal(shown(), false);
  pending[0].resolve(okResponse()); await flush();
  assert.equal(blobs, 0, 'a clip that arrives after the window was closed is dropped');
  assert.equal(audio.src, '');

  console.log('ok');
})().catch(error => { console.error(error); process.exit(1); });
