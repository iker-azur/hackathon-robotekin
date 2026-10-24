import { readFile } from 'node:fs/promises';
import test from 'node:test';
import assert from 'node:assert/strict';

const code = await readFile(new URL('../voice.js', import.meta.url), 'utf8');
const { initVoiceInterface } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);

class Element {
  constructor() {
    this.textContent = ''; this.value = ''; this.hidden = false; this.disabled = false;
    this.children = []; this.style = {}; this.attributes = {}; this.events = {};
    this.classList = { toggle() {} };
  }
  addEventListener(name, callback) { this.events[name] = callback; }
  setAttribute(name, value) { this.attributes[name] = value; }
  replaceChildren(...children) { this.children = children; }
  append(...children) { this.children.push(...children); }
  pause() {}
}
function fixture({ permissionError = null, installed = true, apiError = false } = {}) {
  const elements = new Map();
  const events = [];
  const root = {
    getElementById(id) { if (!elements.has(id)) elements.set(id, new Element()); return elements.get(id); },
    querySelectorAll() { return []; }, createElement() { return new Element(); },
    dispatchEvent(event) { events.push(event); return true; },
  };
  const requests = [];
  let stopped = 0;
  const track = { stop() { stopped++; }, addEventListener() {} };
  const stream = { getTracks: () => [track], getAudioTracks: () => [track] };
  const instances = [];
  class Recorder {
    static isTypeSupported(type) { return type.startsWith('audio/webm'); }
    constructor() { this.mimeType = 'audio/webm'; this.state = 'inactive'; instances.push(this); }
    start() { this.state = 'recording'; }
    stop() {
      this.state = 'inactive';
      this.ondataavailable({ data: new Blob(['audio']) });
      this.done = this.onstop();
    }
  }
  let timer = 0;
  const timers = new Map();
  const browser = {
    navigator: { mediaDevices: { async getUserMedia() { if (permissionError) throw permissionError; return stream; } } },
    MediaRecorder: Recorder, Blob, AbortController,
    CustomEvent: class { constructor(type, options) { this.type = type; this.detail = options.detail; } },
    URL: { createObjectURL: () => 'blob:recording', revokeObjectURL() {} },
    addEventListener() {}, cancelAnimationFrame() {},
    setInterval(callback) { timers.set(++timer, callback); return timer; },
    setTimeout(callback) { timers.set(++timer, callback); return timer; },
    clearTimeout(id) { timers.delete(id); }, clearInterval(id) { timers.delete(id); },
    async fetch(url, options = {}) {
      requests.push({ url, ...options });
      if (url === '/api/voice/status') return { ok: true, json: async () => ({ available: installed, install_command: 'install whisper' }) };
      if (url === '/api/voice/transcribe') return { ok: true, json: async () => ({ text: 'Pausa el proceso' }) };
      if (url === '/api/command') return {
        ok: !apiError, status: apiError ? 400 : 200,
        json: async () => apiError ? { detail: 'Orden no válida' } : {
          status: 'proposed', classifier: 'Reglas de demo', source: JSON.parse(options.body).source, execution: 'preview_only',
          actions: [{ label: 'Pausar temporalmente', intent: 'PAUSE_PROCESS', parameters: {} }],
          steps: [{ number: 1, text: 'Guardar el paso actual' }], issues: [],
        },
      };
      throw new Error(`Unexpected request ${url}`);
    },
  };
  const controller = initVoiceInterface(root, browser);
  return { controller, root, browser, requests, instances, timers, events, stopped: () => stopped };
}
const settle = () => new Promise(resolve => setImmediate(resolve));

test('captures actual browser audio, releases microphone, transcribes and shows the proposed plan', async () => {
  const f = fixture();
  await settle();
  await f.controller.startRecording();
  assert.equal(f.root.getElementById('microphone').attributes['aria-pressed'], 'true');
  assert.equal(f.root.getElementById('analyze-command').disabled, true);
  f.controller.stopRecording();
  await f.instances[0].done;
  assert.equal(f.stopped(), 1);
  assert.equal(f.root.getElementById('microphone').attributes['aria-pressed'], 'false');
  assert.equal(f.root.getElementById('recorded-audio').hidden, false);
  assert.equal(f.root.getElementById('command-text').value, 'Pausa el proceso');
  assert.equal(f.root.getElementById('plan-steps').children.length, 1);
  const audio = f.requests.find(request => request.url === '/api/voice/transcribe');
  assert.equal(await audio.body.text(), 'audio');
  assert.equal(JSON.parse(f.requests.find(request => request.url === '/api/command').body).source, 'voice');
  assert.equal(f.timers.size, 0);
  assert.ok(f.events.some(event => event.type === 'hmi:phase' && event.detail === 'listening'));
  assert.ok(f.events.some(event => event.type === 'hmi:plan' && event.detail?.execution === 'preview_only'));
  assert.equal(f.root.getElementById('plan-panel').hidden, false);
  f.controller.destroy();
});

test('permission denial restores controls and provides a specific explanation', async () => {
  const f = fixture({ permissionError: Object.assign(new Error('denied'), { name: 'NotAllowedError' }) });
  await settle();
  await f.controller.startRecording();
  assert.match(f.root.getElementById('voice-help').textContent, /denegado/);
  assert.equal(f.root.getElementById('microphone').disabled, false);
  assert.equal(f.root.getElementById('command-text').readOnly, false);
  assert.equal(f.instances.length, 0);
  f.controller.destroy();
});

test('without Whisper the captured recording remains playable and typed commands still work', async () => {
  const f = fixture({ installed: false });
  await settle();
  await f.controller.startRecording();
  f.controller.stopRecording();
  await f.instances[0].done;
  assert.equal(f.root.getElementById('recorded-audio').hidden, false);
  assert.equal(f.root.getElementById('voice-setup').hidden, false);
  assert.match(f.root.getElementById('voice-help').textContent, /Falta instalar Whisper/);
  assert.equal(f.requests.some(request => request.url === '/api/voice/transcribe'), false);
  f.root.getElementById('command-text').value = 'Pausa el proceso';
  await f.controller.analyze();
  assert.equal(f.root.getElementById('plan-badge').textContent, 'PLAN PROPUESTO');
  f.controller.destroy();
});

test('editing the transcript clears a stale plan', async () => {
  const f = fixture();
  f.root.getElementById('command-text').value = 'Pausa el proceso';
  await f.controller.analyze();
  assert.equal(f.root.getElementById('plan-steps').children.length, 1);
  f.root.getElementById('command-text').events.input();
  assert.equal(f.root.getElementById('plan-badge').textContent, 'SIN ANALIZAR');
  assert.equal(f.root.getElementById('plan-steps').children.length, 0);
  assert.equal(f.events.filter(event => event.type === 'hmi:plan').at(-1).detail, null);
  f.controller.destroy();
});

test('server errors never leave an old plan on screen or controls blocked', async () => {
  const f = fixture({ apiError: true });
  f.root.getElementById('command-text').value = 'Pausa el proceso';
  await f.controller.analyze();
  assert.equal(f.root.getElementById('plan-badge').textContent, 'ERROR DE ANÁLISIS');
  assert.equal(f.root.getElementById('plan-steps').children.length, 0);
  assert.equal(f.root.getElementById('analyze-command').disabled, false);
  f.controller.destroy();
});

test('leaving the page stops capture without transcribing or classifying', async () => {
  const f = fixture();
  await settle();
  await f.controller.startRecording();
  f.controller.destroy();
  await f.instances[0].done;
  assert.equal(f.stopped(), 1);
  assert.equal(f.requests.some(request => request.url === '/api/voice/transcribe'), false);
  assert.equal(f.requests.some(request => request.url === '/api/command'), false);
  assert.equal(f.timers.size, 0);
});
