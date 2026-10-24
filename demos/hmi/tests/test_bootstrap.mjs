import { readFile } from 'node:fs/promises';
import test from 'node:test';
import assert from 'node:assert/strict';

const html = await readFile(new URL('../index.html', import.meta.url), 'utf8');
const code = html.match(/<script type="module" id="scene-bootstrap">([\s\S]*?)<\/script>/)[1];
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
const boot = new AsyncFunction('document', 'fetch', 'loadSceneModule', 'console', code.replace("import('./app.js')", 'loadSceneModule()'));
function fixture() {
  const elements = new Map();
  return { elements, getElementById(id) {
    if (!elements.has(id)) elements.set(id, {textContent: '', hidden: true, dataset: {}, setAttribute(name, value) { this[name] = value; }});
    return elements.get(id);
  }};
}
const consoleStub = {error() {}};
test('a missing model module on an old server shows concrete restart instructions', async () => {
  const document = fixture();
  await boot(document, async () => ({status:404}), async () => { throw new TypeError('Failed to fetch dynamically imported module'); }, consoleStub);
  const notice = document.getElementById('scene-notice');
  assert.equal(notice.hidden, false);
  assert.equal(notice.role, 'alert');
  assert.match(notice.textContent, /Ctrl\+C/);
  assert.match(notice.textContent, /server.py --open/);
  assert.equal(document.getElementById('connection').textContent, 'ERROR DE ARRANQUE');
});
test('other startup failures display their actual cause', async () => {
  const document = fixture();
  await boot(document, async () => ({status:200}), async () => { throw new Error('WebGL no disponible'); }, consoleStub);
  assert.match(document.getElementById('scene-notice').textContent, /WebGL no disponible/);
});
test('successful module startup leaves normal scene status intact', async () => {
  const document = fixture();
  await boot(document, async () => { assert.fail('no diagnostic request expected'); }, async () => {}, consoleStub);
  assert.equal(document.elements.size, 0);
});
test('an unreachable server gives local launch instructions', async () => {
  const document = fixture();
  await boot(document, async () => { throw new TypeError('Failed to fetch'); }, async () => { throw new Error('Import failed'); }, consoleStub);
  assert.match(document.getElementById('scene-notice').textContent, /No se puede acceder al servidor local/);
});
