import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
const repo = fileURLToPath(new URL('../../../', import.meta.url));
const threeURL = new URL('../../08_recovery_kit/server/static/vendor/three.module.min.js', import.meta.url);
const THREE = await import(threeURL.href);
const source = readFileSync(new URL('../robot-model.js', import.meta.url), 'utf8').replace("from 'three'", `from '${threeURL.href}'`);
const {createSO101, createWorkcellBoxes} = await import('data:text/javascript;base64,' + Buffer.from(source).toString('base64'));
const result = spawnSync(repo + '.venv/bin/python', ['-c', `
import sys,json
sys.path.insert(0,'demos/hmi')
import server
ref={'links':server.joint_poses(dict.fromkeys(server.JOINTS,0.0))}
poses=[{'links':server.joint_poses(server.demo_joints(i,server.JOINTS))} for i in range(12)]
jaw={'links':server.joint_poses({**dict.fromkeys(server.JOINTS,0.0),'gripper':0.8})}
print(json.dumps([ref,poses,jaw]))
`], {cwd:repo,encoding:'utf8'});
assert.equal(result.status,0,result.stderr);
const [reference, poses, jawPose] = JSON.parse(result.stdout);
const model = createSO101(reference);
const camera = new THREE.PerspectiveCamera();
for(const pose of poses) {
  model.update(pose);model.updateCamera(camera);
  for(const [name,[pos]] of Object.entries(pose.links)) {
    assert.ok(model.links[name].getWorldPosition(new THREE.Vector3()).distanceTo(new THREE.Vector3(...pos)) < 1e-9);
  }
  const bearing = model.root.getObjectByName('jaw-bearing');
  const movingMount = model.root.getObjectByName('moving-finger-mount');
  assert.ok(bearing.getWorldPosition(new THREE.Vector3()).distanceTo(movingMount.getWorldPosition(new THREE.Vector3())) < 1e-7,
    'la mordaza móvil debe permanecer unida al soporte de su eje');
  model.root.traverse(o=>assert.ok(o.matrixWorld.elements.every(Number.isFinite)));
  assert.ok(camera.position.distanceTo(model.lens.getWorldPosition(new THREE.Vector3())) < 1e-9);
  const lensDirection=new THREE.Vector3(0,0,1).applyQuaternion(model.lens.getWorldQuaternion(new THREE.Quaternion()));
  assert.ok(camera.getWorldDirection(new THREE.Vector3()).dot(lensDirection) > .999999);
  const toolForward=new THREE.Vector3(0,-1,0).applyQuaternion(model.links.gripper.getWorldQuaternion(new THREE.Quaternion()));
  assert.ok(lensDirection.dot(toolForward) > .999999, 'la lente debe mirar al frente de la pinza');
}
model.update(reference);
const hand = model.links.gripper.matrixWorld.clone();
const jaw = model.links.jaw.matrixWorld.clone();
model.update(jawPose);
assert.deepEqual(model.links.gripper.matrixWorld.elements,hand.elements);
assert.notDeepEqual(model.links.jaw.matrixWorld.elements,jaw.elements);
console.log('OK: 12 poses SO-101, lente/cámara alineadas y mandíbula móvil independiente.');

const boxes = createWorkcellBoxes();
boxes.updateMatrixWorld(true);
assert.equal(boxes.children.length, 3);
const bounds = boxes.children.map(box => new THREE.Box3().setFromObject(box));
for (const bound of bounds) {
  assert.ok(bound.min.z >= -.0011, 'las cajas deben apoyar en la mesa');
  assert.ok(bound.min.x > -.46 && bound.max.x < .46 && bound.min.y > -.58 && bound.max.y < .25,
    'las cajas deben caber sobre el tablero');
}
for(let i=0;i<bounds.length;i++)for(let j=i+1;j<bounds.length;j++)assert.ok(!bounds[i].intersectsBox(bounds[j]),
  'las tres cajas deben estar separadas');
boxes.traverse(o => {
  if(o.geometry)assert.ok(Array.from(o.geometry.attributes.position.array).every(Number.isFinite));
});
assert.ok(boxes.getObjectByName('white-wire-basket').children.some(o => o.isLineSegments));
console.log('OK: tres cajas sobre la mesa, separadas y con cesta de rejilla.');
