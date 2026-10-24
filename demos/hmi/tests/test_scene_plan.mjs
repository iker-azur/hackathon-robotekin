import { readFile } from 'node:fs/promises';
import test from 'node:test';
import assert from 'node:assert/strict';
const code = await readFile(new URL('../scene-plan.js', import.meta.url), 'utf8');
const { sceneProposal } = await import(`data:text/javascript;base64,${Buffer.from(code).toString('base64')}`);
const objects = [{ id: 'blue1', color: 'blue', shape: 'cube' }, { id: 'red1', color: 'red', shape: 'cube' }, { id: 'green1', color: 'green', shape: 'cylinder' }];
function plan(parameters, extra = {}) { return { status: 'proposed', execution: 'preview_only', actions: [{ intent: 'PLACE_OBJECT', parameters, ...extra }] }; }

test('a plan highlights the matching visible piece and names its proposed destination', () => {
  const result = sceneProposal(plan({ object_color: 'azul', quantity: 1, destination_color: 'rojo' }), objects);
  assert.deepEqual(result.targetIds, ['blue1']);
  assert.equal(result.destination.color, 'red');
  assert.match(result.destination.label, /propuesta/);
});

test('missing pieces never receive a fabricated route', () => {
  const result = sceneProposal(plan({ object_color: 'azul', quantity: 2, destination_color: 'rojo' }), objects);
  assert.equal(result.destination, null);
  assert.deepEqual(result.targetIds, ['blue1']);
  assert.match(result.note, /Ruta pendiente/);
});

test('incomplete, rejected and negated plans do not produce spatial actions', () => {
  for (const value of [null, { status: 'needs_clarification' }, plan({}, { negated: true }), { ...plan({}), execution: 'live' }]) {
    assert.deepEqual(sceneProposal(value, objects).targetIds, []);
    assert.equal(sceneProposal(value, objects).destination, null);
  }
});

test('inspection highlights visible objects without inventing a destination', () => {
  const value = plan({ object_type: 'pieza' });
  value.actions[0].intent = 'INSPECT_BATCH';
  const result = sceneProposal(value, objects);
  assert.equal(result.targetIds.length, 3);
  assert.equal(result.destination, null);
});

test('absent or unsupported colors are reported without selecting an unrelated piece', () => {
  for (const color of ['amarillo', 'morado']) {
    const result = sceneProposal(plan({ object_color: color, destination_color: 'rojo' }), objects);
    assert.deepEqual(result.targetIds, []);
    assert.equal(result.destination, null);
    assert.ok(result.note);
  }
});

test('object type and shape must match, not only color', () => {
  assert.deepEqual(sceneProposal(plan({ object_type: 'cubo', object_color: 'verde' }), objects).targetIds, []);
  assert.deepEqual(sceneProposal(plan({ object_type: 'tornillo', object_color: 'azul' }), objects).targetIds, []);
  assert.deepEqual(sceneProposal(plan({ object_type: 'cilindro', object_color: 'verde' }), objects).targetIds, ['green1']);
});

test('trained pin features select only objects with a known matching pin state', () => {
  const parts = [{ id: 'pin', with_pin: true }, { id: 'no-pin', with_pin: false }, { id: 'unknown' }];
  assert.deepEqual(sceneProposal(plan({ object_type: 'pieza', object_feature: 'con perno' }), parts).targetIds, ['pin']);
  assert.deepEqual(sceneProposal(plan({ object_type: 'pieza', object_feature: 'sin perno' }), parts).targetIds, ['no-pin']);
});

test('quality predictions never invent quality labels for demo objects', () => {
  const result = sceneProposal(plan({ object_type: 'pieza', object_quality: 'defectuosa' }), objects);
  assert.deepEqual(result.targetIds, []);
  assert.match(result.note, /calidad/);
});
