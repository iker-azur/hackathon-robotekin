import * as THREE from 'three';
import { OrbitControls } from '/vendor/OrbitControls.js';
import { sceneProposal } from './scene-plan.js';
import { createSO101, createWorkcellBoxes } from './robot-model.js';
import { createBuddy } from './buddy.js';

const $ = id => document.getElementById(id);
const jointNames = {
  shoulder_pan: 'Giro de base', shoulder_lift: 'Hombro', elbow_flex: 'Codo',
  wrist_flex: 'Flexión de muñeca', wrist_roll: 'Giro de muñeca', gripper: 'Pinza',
};
$('joints').innerHTML = Object.entries(jointNames).map(([key, name], index) =>
  `<div class="joint-row"><span class="joint-index">0${index + 1}</span><span class="joint-name">${name}</span><span class="joint-value" id="joint-${key}">—</span></div>`).join('');

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xeeeae2);
scene.fog = new THREE.Fog(0xeeeae2, 2.5, 6);
const canvas = $('scene');
let renderer;
let cameraRenderer;
try {
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  cameraRenderer = new THREE.WebGLRenderer({ canvas: $('virtual-camera'), antialias: true });
} catch (error) {
  $('scene-notice').textContent = 'No se puede iniciar WebGL. Activa la aceleración gráfica del navegador.';
  throw error;
}
for (const render of [renderer, cameraRenderer]) {
  render.setPixelRatio(Math.min(devicePixelRatio, 2));
  render.outputColorSpace = THREE.SRGBColorSpace;
}
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
const camera = new THREE.PerspectiveCamera(36, 1, .005, 20);
camera.up.set(0, 0, 1);
const virtualCamera = new THREE.PerspectiveCamera(48, 4 / 3, .005, 20);
virtualCamera.up.set(0, 0, 1);
const wristCamera = new THREE.PerspectiveCamera(72, 1, .005, 20);
const traces = new THREE.Group();
scene.add(traces);
const referenceLine = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineDashedMaterial({ color: 0xa08c65, dashSize: .012, gapSize: .009, depthTest: false }));
const robotLine = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ color: 0xc54c27, depthTest: false }));
for (const line of [referenceLine, robotLine]) { line.layers.set(1); line.renderOrder = 5; traces.add(line); }
camera.layers.enable(1);
virtualCamera.layers.enable(1);
const robotTrace = [];
let previousPhase = null;
let ghostsVisible = true;
let attentionEnabled = false;
const heatCanvas = $('attention-overlay');
const heatContext = heatCanvas.getContext('2d');
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = .09;
controls.minDistance = .15;
controls.maxDistance = 4;
controls.maxPolarAngle = Math.PI * .49;
controls.zoomToCursor = true;
controls.screenSpacePanning = true;
scene.add(new THREE.HemisphereLight(0xfffcf2, 0x9ca28c, 2));
const keyLight = new THREE.DirectionalLight(0xfff9e7, 2.8);
keyLight.position.set(-.4, -.5, 1.5);
keyLight.castShadow = true;
keyLight.shadow.mapSize.set(1024, 1024);
Object.assign(keyLight.shadow.camera, { left: -.8, right: .8, top: .8, bottom: -.8, near: .1, far: 3 });
keyLight.shadow.bias = -.0003;
keyLight.shadow.camera.updateProjectionMatrix();
scene.add(keyLight);
const fillLight = new THREE.DirectionalLight(0xd9e2ce, 1.2);
fillLight.position.set(-1, .5, .8);
scene.add(fillLight);

// All model transforms use the repository's Z-up, metre, wxyz convention.
const floor = new THREE.Group();
scene.add(floor);
const plane = new THREE.Mesh(new THREE.PlaneGeometry(8, 8),
  new THREE.MeshStandardMaterial({ color: 0xeeeae2, roughness: 1 }));
plane.position.z = -.032;
plane.receiveShadow = true;
floor.add(plane);
const tabletopShape = new THREE.Shape();
const x0 = -.46, x1 = .46, y0 = -.58, y1 = .25, radius = .055;
tabletopShape.moveTo(x0 + radius, y0);
tabletopShape.lineTo(x1 - radius, y0); tabletopShape.quadraticCurveTo(x1, y0, x1, y0 + radius);
tabletopShape.lineTo(x1, y1 - radius); tabletopShape.quadraticCurveTo(x1, y1, x1 - radius, y1);
tabletopShape.lineTo(x0 + radius, y1); tabletopShape.quadraticCurveTo(x0, y1, x0, y1 - radius);
tabletopShape.lineTo(x0, y0 + radius); tabletopShape.quadraticCurveTo(x0, y0, x0 + radius, y0);
const tabletop = new THREE.Mesh(new THREE.ExtrudeGeometry(tabletopShape, { depth: .018, bevelEnabled: false, curveSegments: 16 }), new THREE.MeshStandardMaterial({ color: 0xe0e5d7, roughness: .9 }));
tabletop.position.z = -.018;
tabletop.castShadow = true;
tabletop.receiveShadow = true;
floor.add(tabletop);
for (const [size, divisions, color, opacity] of [[.7, 14, 0x87957e, .19], [.7, 2, 0x87957e, .13]]) {
  const grid = new THREE.GridHelper(size, divisions, color, color);
  grid.rotation.x = Math.PI / 2;
  grid.position.z = .0002;
  grid.position.y = -.16;
  grid.material.transparent = true;
  grid.material.opacity = opacity;
  floor.add(grid);
}
const axes = new THREE.AxesHelper(.06);
axes.position.x = -.36;
axes.position.y = .14;
axes.position.z = .002;
floor.add(axes);
const inspectionSpot = new THREE.Group();
inspectionSpot.name = 'puesto-cenital-a-la-izquierda';
inspectionSpot.position.set(.17, -.16, .205);
inspectionSpot.visible = false;
inspectionSpot.add(new THREE.Mesh(new THREE.RingGeometry(.037, .04, 48),
  new THREE.MeshBasicMaterial({ color: 0xc54c27, transparent: true, opacity: .72, side: THREE.DoubleSide })));
inspectionSpot.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints([
  new THREE.Vector3(-.012, 0, 0), new THREE.Vector3(.012, 0, 0),
  new THREE.Vector3(0, -.012, 0), new THREE.Vector3(0, .012, 0),
]), new THREE.LineBasicMaterial({ color: 0xc54c27, transparent: true, opacity: .85 })));
floor.add(inspectionSpot);
const workcellBoxes = createWorkcellBoxes();
floor.add(workcellBoxes);
const buddy = createBuddy();
floor.add(buddy.root);
floor.add(buddy.shadow);
const buddyCameraTarget = new THREE.WebGLRenderTarget(256, 144, { depthBuffer: true, stencilBuffer: false });
buddyCameraTarget.texture.colorSpace = THREE.SRGBColorSpace;
buddy.setScreenTexture(buddyCameraTarget.texture);
document.addEventListener('hmi:level', event => buddy.setAudioLevel(event.detail));
let buddyJournalVersion = -1;
let lastBuddyCameraFrame = 0;
function updateBuddyJournal() {
  const data = buddy.getJournal();
  if (data.version === buddyJournalVersion) return;
  buddyJournalVersion = data.version;
  $('buddy-journal-count').textContent = String(data.observedDrops);
  $('buddy-journal-summary').textContent = data.observedDrops ? `${data.summary} · resumen de lo observado` : 'Aún no hay eventos observados.';
  const list = $('buddy-journal-entries');
  list.replaceChildren();
  for (const entry of data.entries.slice(-12).reverse()) {
    const item = document.createElement('li');
    if (entry.kind === 'error') item.classList.add('error');
    const stamp = document.createElement('time'); stamp.textContent = entry.time;
    const text = document.createElement('span'); text.textContent = entry.text;
    item.append(stamp, text); list.append(item);
  }
}
const journalPanel = $('buddy-journal');
$('buddy-journal-toggle').addEventListener('click', () => {
  journalPanel.hidden = !journalPanel.hidden;
  $('buddy-journal-toggle').setAttribute('aria-expanded', String(!journalPanel.hidden));
  updateBuddyJournal();
});
$('buddy-journal-close').addEventListener('click', () => {
  journalPanel.hidden = true; $('buddy-journal-toggle').setAttribute('aria-expanded', 'false');
});
let robotModel;
const pieces = new Map();
let base = new THREE.Vector3();
let floorHeight = .0301;
let centeredLive = false;
let demo = false;
let healthy = false;
let lastPoseSignature = null;
let frameId = -1;
let frameURL = null;
let framePending = false;
let frameFresh = false;
let reference = null;
let view = 'perspective';
let activePlan = null;
let selectedPiece = null;
const pieceLabels = new Map();
const boxLabels = [
  [$('source-bin-label'), new THREE.Vector3(.16, -.37, .082)],
  [$('pin-bin-label'), new THREE.Vector3(-.20, -.20, .082)],
  [$('no-pin-bin-label'), new THREE.Vector3(-.12, -.43, .082)],
];
const planRoute = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineDashedMaterial({ color: 0xc54c27, dashSize: .012, gapSize: .006, depthTest: false }));
planRoute.visible = false;
planRoute.renderOrder = 6;
scene.add(planRoute);
const destination = new THREE.Mesh(new THREE.RingGeometry(.043, .046, 64), new THREE.MeshBasicMaterial({ color: 0xc54c27, transparent: true, opacity: .8, side: THREE.DoubleSide }));
destination.visible = false;
scene.add(destination);
const destinationLabel = document.createElement('span');
destinationLabel.className = 'piece-label destination-label';
destinationLabel.hidden = true;
$('object-labels').append(destinationLabel);

function setView(name = 'perspective') {
  view = name;
  document.body.dataset.view = name;
  controls.enabled = name !== 'pilot';
  for (const id of ['zoom-in', 'zoom-out']) $(id).disabled = name === 'pilot';
  $('camera').hidden = name !== 'pilot' || demo || !frameFresh;
  canvas.style.visibility = !$('camera').hidden ? 'hidden' : 'visible';
  const target = base.clone().add(new THREE.Vector3(-.07, -.13, .11));
  controls.target.copy(target);
  if (name === 'top') camera.position.copy(target).add(new THREE.Vector3(.001, 0, 1.05));
  else if (name === 'front') camera.position.copy(target).add(new THREE.Vector3(0, -1.05, .06));
  else camera.position.copy(target).add(new THREE.Vector3(.68, -.87, .78));
  controls.update();
  document.querySelectorAll('[data-view]').forEach(button => {
    const selected = button.dataset.view === name;
    button.classList.toggle('selected', selected);
    button.setAttribute('aria-pressed', String(selected));
  });
}
function setPose(object, pos, quat) {
  object.position.fromArray(pos);
  object.quaternion.set(quat[1], quat[2], quat[3], quat[0]).normalize();
}
function highlightPiece(object, highlighted) {
  object.traverse(child => {
    if (!child.isMesh || !child.material?.emissive) return;
    child.material.emissive.setHex(highlighted ? 0xc54c27 : 0);
    child.material.emissiveIntensity = .18;
  });
}
function castingGeometry(data, name) {
  const part = new THREE.Group();
  part.name = name;
  const profile = new THREE.Shape();
  profile.moveTo(-.026, -.030); profile.lineTo(-.020, -.034); profile.lineTo(-.009, -.031);
  profile.lineTo(.004, -.026); profile.lineTo(.023, -.030); profile.lineTo(.027, -.022);
  profile.lineTo(.023, -.004); profile.lineTo(.027, .020); profile.lineTo(.021, .030);
  profile.lineTo(.006, .027); profile.lineTo(-.006, .032); profile.lineTo(-.024, .025);
  profile.lineTo(-.021, .008); profile.lineTo(-.027, -.010); profile.closePath();
  const bore = new THREE.Path();
  bore.absarc(-.014, -.020, .0035, 0, Math.PI * 2, true);
  profile.holes.push(bore);
  const gray = new THREE.MeshStandardMaterial({ color: 0x89908d, metalness: .56, roughness: .65 });
  const body = new THREE.Mesh(new THREE.ExtrudeGeometry(profile, {
    depth: .012, bevelEnabled: true, bevelSegments: 2, bevelSize: .0012, bevelThickness: .0012,
  }), gray);
  body.position.z = -.006;
  body.name = name;
  body.castShadow = true;
  body.receiveShadow = true;
  part.add(body);
  if (data.with_pin) {
    const pin = new THREE.Mesh(new THREE.CylinderGeometry(.004, .0045, .020, 16).rotateX(Math.PI / 2),
      new THREE.MeshStandardMaterial({ color: 0x69716e, metalness: .72, roughness: .42 }));
    pin.position.set(.010, .008, .016);
    pin.name = name;
    pin.castShadow = true;
    part.add(pin);
    const seat = new THREE.Mesh(new THREE.CylinderGeometry(.008, .008, .0025, 20).rotateX(Math.PI / 2), gray);
    seat.position.set(.010, .008, .007);
    seat.name = name;
    part.add(seat);
  }
  part.userData = { color: data.color, shape: data.shape, with_pin: data.with_pin };
  return part;
}
function selectPiece(name) {
  const mesh = pieces.get(name);
  if (!mesh?.visible) return;
  selectedPiece = name;
  $('plan-panel').hidden = true;
  $('selection').hidden = false;
  const color = { red: 'roja', blue: 'azul', green: 'verde', yellow: 'amarilla' }[mesh.userData.color];
  $('selection-title').textContent = mesh.userData.shape === 'casting'
    ? 'Pieza de fundición' : `Pieza ${color || ''}`.trim();
  $('selection-description').textContent = mesh.userData.shape === 'casting'
    ? 'Origen · Cesta roja' : 'Pieza de ejemplo';
  for (const [id, object] of pieces) {
    highlightPiece(object, id === name);
    pieceLabels.get(id)?.classList.toggle('selected', id === name);
  }
}
function updatePlanProjection() {
  const proposal = sceneProposal(activePlan, [...pieces].filter(([, mesh]) => mesh.visible).map(([id, mesh]) => ({ id, color: mesh.userData.color, shape: mesh.userData.shape })));
  planRoute.visible = false;
  destination.visible = false;
  destinationLabel.hidden = true;
  for (const [id, mesh] of pieces) {
    const selected = !$('selection').hidden ? id === selectedPiece : proposal.targetIds.includes(id);
    highlightPiece(mesh, selected);
    pieceLabels.get(id)?.classList.toggle('selected', selected);
  }
  if (!demo || !proposal.destination || !proposal.targetIds.length) return;
  destination.position.copy(base).add(new THREE.Vector3(.26, -.13, floorHeight - base.z + .002));
  destination.material.color.setHex({ red: 0xc16b4c, blue: 0x6687a0, green: 0x8a9b65, yellow: 0xd9be75 }[proposal.destination.color]);
  destination.visible = true;
  const start = pieces.get(proposal.targetIds[0]).position.clone();
  start.z = floorHeight + .003;
  const end = destination.position.clone();
  const midpoint = start.clone().lerp(end, .5).add(new THREE.Vector3(.015, -.075, 0));
  planRoute.geometry.dispose();
  planRoute.geometry = new THREE.BufferGeometry().setFromPoints(new THREE.QuadraticBezierCurve3(start, midpoint, end).getPoints(64));
  planRoute.computeLineDistances();
  planRoute.visible = true;
  destinationLabel.textContent = proposal.destination.label;
}
function updateLabels(activeCamera) {
  const viewport = $('viewport');
  function positionLabel(element, point) {
    const projected = point.clone().project(activeCamera);
    const visible = healthy && demo && view !== 'pilot' && projected.z > -1 && projected.z < 1 && Math.abs(projected.x) < .94 && Math.abs(projected.y) < .9;
    element.hidden = !visible;
    if (visible) {
      element.style.left = `${(projected.x + 1) * viewport.clientWidth / 2}px`;
      element.style.top = `${(1 - projected.y) * viewport.clientHeight / 2 - 28}px`;
    }
  }
  for (const [id, label] of pieceLabels) {
    const mesh = pieces.get(id);
    if (!mesh.visible) label.hidden = true;
    else positionLabel(label, mesh.position.clone().add(new THREE.Vector3(0, 0, .022)));
  }
  for (const [label, offset] of boxLabels) positionLabel(label, base.clone().add(offset));
  if (destination.visible) positionLabel(destinationLabel, destination.position);
  else destinationLabel.hidden = true;
}
function setPhase(phase) {
  document.body.dataset.phase = phase;
  document.querySelectorAll('[data-phase-step]').forEach(element => element.classList.toggle('active', element.dataset.phaseStep === phase));
}
document.addEventListener('hmi:phase', event => setPhase(event.detail));
document.addEventListener('hmi:plan', event => {
  activePlan = event.detail;
  $('selection').hidden = true;
  updatePlanProjection();
  if (activePlan) setPhase(activePlan.status === 'proposed' ? 'planning' : 'understanding');
  const pinSort = activePlan?.actions?.some(action =>
    ['CLASSIFY_OBJECT', 'SORT_OBJECTS'].includes(action.intent)) && /perno/i.test(activePlan?.text || '');
  $('classifier-state').textContent = pinSort ? 'Orden lista · Clasificador sin conectar' : 'Clasificador sin conectar';
});
const raycaster = new THREE.Raycaster();
let pointerStart = null;
canvas.addEventListener('pointerdown', event => { pointerStart = [event.clientX, event.clientY]; });
canvas.addEventListener('pointerup', event => {
  if (view === 'pilot' || !pointerStart || Math.hypot(event.clientX - pointerStart[0], event.clientY - pointerStart[1]) > 5) return;
  const rect = canvas.getBoundingClientRect();
  const mouse = new THREE.Vector2((event.clientX - rect.left) / rect.width * 2 - 1, 1 - (event.clientY - rect.top) / rect.height * 2);
  raycaster.setFromCamera(mouse, camera);
  const match = raycaster.intersectObjects([...pieces.values()].filter(mesh => mesh.visible))[0];
  if (match) selectPiece(match.object.name);
});
$('selection-close').addEventListener('click', () => { $('selection').hidden = true; selectedPiece = null; updatePlanProjection(); });
$('plan-close').addEventListener('click', () => { $('plan-panel').hidden = true; });
$('show-plan').addEventListener('click', () => { $('plan-panel').hidden = !$('plan-panel').hidden; $('selection').hidden = true; });
function applyPoses(poses, live = false, sourceError = null) {
  const workflow = poses.workflow;
  robotModel.update(poses);
  robotModel.updateCamera(wristCamera);
  if (poses.links?.base) {
    base.fromArray(poses.links.base[0]);
    // Existing mesh base bottom is 30.1 mm above its link origin.
    floorHeight = base.z + .0301;
    floor.position.set(base.x, base.y, floorHeight);
    if (!centeredLive) {
      setView(view);
      centeredLive = live;
    }
    virtualCamera.position.copy(base).add(new THREE.Vector3(.17, -.16, 1.05));
    virtualCamera.up.set(0, 1, 0);
    virtualCamera.lookAt(base.clone().add(new THREE.Vector3(.17, -.16, .235)));
  }
  floor.updateMatrixWorld(true);
  const armPoints = Object.entries(poses.links || {}).filter(([name]) => name !== 'base').map(([, pose]) => {
    if (!Array.isArray(pose?.[0]) || pose[0].length < 3) return null;
    const point = floor.worldToLocal(new THREE.Vector3(pose[0][0], pose[0][1], pose[0][2]));
    return { x: point.x, y: point.y, z: point.z };
  }).filter(Boolean);
  buddy.setTelemetry({ workflow, joints: poses.joints, armPoints, error: sourceError });
  for (const [name, data] of Object.entries(poses.pieces || {})) {
    let mesh = pieces.get(name);
    if (!mesh) {
      const geometry = data.shape === 'casting'
        ? null
        : data.shape === 'cylinder'
        ? new THREE.CylinderGeometry(data.size / 2, data.size / 2, data.height, 24).rotateX(Math.PI / 2)
        : new THREE.BoxGeometry(data.size, data.size, data.height);
      if (data.shape === 'casting') mesh = castingGeometry(data, name);
      else {
        const color = { red: 0xc16b4c, blue: 0x6687a0, green: 0x8a9b65, yellow: 0xd9be75 }[data.color] || 0xa0b6be;
        mesh = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ color, roughness: .7 }));
        mesh.name = name;
        mesh.userData.color = data.color;
        mesh.userData.shape = data.shape;
        mesh.castShadow = true;
        mesh.receiveShadow = true;
      }
      scene.add(mesh);
      pieces.set(name, mesh);
      const label = document.createElement('button');
      label.className = 'piece-label';
      label.type = 'button';
      label.textContent = data.shape === 'casting'
        ? `${String(pieceLabels.size + 1).padStart(2, '0')} / PIEZA DE FUNDICIÓN`
        : `${String(pieceLabels.size + 1).padStart(2, '0')} / ${{ red: 'ROJA', blue: 'AZUL', green: 'VERDE', yellow: 'AMARILLA' }[data.color] || name}`;
      label.style.setProperty('--piece-color', data.shape === 'casting' ? '#89908d' : `#${mesh.material.color.getHexString()}`);
      label.addEventListener('click', () => selectPiece(name));
      $('object-labels').append(label);
      pieceLabels.set(name, label);
    }
    setPose(mesh, data.pos, data.quat);
  }
  for (const [name, mesh] of pieces) mesh.visible = Boolean(poses.pieces?.[name]);
  if (selectedPiece && !pieces.get(selectedPiece)?.visible) { selectedPiece = null; $('selection').hidden = true; }
  if (activePlan) updatePlanProjection();
  for (const [name] of Object.entries(jointNames)) {
    const angle = poses.joints?.[name];
    $(`joint-${name}`).textContent = Number.isFinite(angle) ? `${(angle * 180 / Math.PI).toFixed(1)}°` : '—';
  }
  const comparison = poses.comparison;
  if (demo && comparison?.kind === 'illustrative') {
    if (previousPhase !== null && comparison.phase < previousPhase) robotTrace.length = 0;
    previousPhase = comparison.phase;
    robotTrace.push(new THREE.Vector3(...comparison.robot_tcp));
    if (robotTrace.length > 480) robotTrace.shift();
    robotLine.geometry.dispose();
    robotLine.geometry = new THREE.BufferGeometry().setFromPoints(robotTrace);
    $('trace-distance').textContent = `${comparison.distance_mm.toFixed(1)} mm`;
  } else {
    robotTrace.length = 0;
    previousPhase = null;
    $('trace-distance').textContent = '—';
  }
  traces.visible = demo && ghostsVisible;
  workcellBoxes.visible = true;
  inspectionSpot.visible = demo;
  if (workflow) {
    $('classifier-state').textContent = workflow.stage === 'clasificando'
      ? `Ejemplo: ${workflow.with_pin ? 'perno → verde' : 'sin perno → blanco'} · Sin clasificador`
      : 'Clasificador sin conectar';
    for (const item of document.querySelectorAll('.inspection-flow li')) {
      const activeStage = workflow.stage === 'soltando' ? 'depositando' : workflow.stage;
      item.classList.toggle('active', item.dataset.stage === activeStage);
    }
  }
}

async function loadScene() {
  const responses = await Promise.all([fetch('/api/reference'), fetch('/api/demo-reference')]);
  if (responses.some(response => !response.ok)) throw new Error('No se puede cargar el modelo del SO-101');
  const [pose, path] = await Promise.all(responses.map(response => response.json()));
  referenceLine.geometry.setFromPoints(path.points.map(point => new THREE.Vector3(...point)));
  referenceLine.computeLineDistances();
  reference = pose;
  robotModel = createSO101(reference);
  scene.add(robotModel.root);
  applyPoses(reference);
}
function resize(render, camera, element) {
  const width = Math.max(element.clientWidth, 1);
  const height = Math.max(element.clientHeight, 1);
  const pixelRatio = render.getPixelRatio();
  if (render.domElement.width !== Math.round(width * pixelRatio) || render.domElement.height !== Math.round(height * pixelRatio)) {
    render.setSize(width, height, false);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  }
}
function animate() {
  const activeCamera = view === 'pilot' ? wristCamera : camera;
  resize(renderer, activeCamera, $('viewport'));
  // Both cameras share one canvas, so changing views must also refresh aspect.
  activeCamera.aspect = Math.max($('viewport').clientWidth, 1) / Math.max($('viewport').clientHeight, 1);
  activeCamera.updateProjectionMatrix();
  controls.update();
  floor.updateMatrixWorld(true);
  buddy.animate(performance.now(), floor.worldToLocal(activeCamera.position.clone()));
  if (robotModel && performance.now() - lastBuddyCameraFrame > 140) {
    // The monitor is briefly hidden while its own camera frame is rendered to avoid feedback.
    buddy.setScreenVisible(false);
    renderer.setRenderTarget(buddyCameraTarget);
    renderer.clear();
    renderer.render(scene, wristCamera);
    renderer.setRenderTarget(null);
    buddy.setScreenVisible(true);
    lastBuddyCameraFrame = performance.now();
  }
  renderer.render(scene, activeCamera);
  if (healthy) {
    const secondaryCamera = virtualCamera;
    const frame = document.querySelector('.camera-view');
    resize(cameraRenderer, secondaryCamera, frame);
    secondaryCamera.aspect = Math.max(frame.clientWidth, 1) / Math.max(frame.clientHeight, 1);
    secondaryCamera.updateProjectionMatrix();
    cameraRenderer.render(scene, secondaryCamera);
  }
  updateLabels(activeCamera);
  updateBuddyComment(activeCamera);
  updateBuddyJournal();
  drawAttention();
  requestAnimationFrame(animate);
}
function updateBuddyComment(activeCamera) {
  const bubble = $('buddy-comment'), comment = buddy.getComment();
  if (!healthy || performance.now() > comment.until || view === 'pilot') { bubble.hidden = true; return; }
  const point = buddy.root.getWorldPosition(new THREE.Vector3()).add(new THREE.Vector3(0, 0, .105)).project(activeCamera);
  if (point.z < -1 || point.z > 1 || Math.abs(point.x) > .94 || Math.abs(point.y) > .92) { bubble.hidden = true; return; }
  bubble.textContent = comment.text;
  bubble.hidden = false;
  bubble.style.left = `${(point.x + 1) * $('viewport').clientWidth / 2}px`;
  bubble.style.top = `${(1 - point.y) * $('viewport').clientHeight / 2}px`;
}
function drawAttention() {
  const visible = attentionEnabled && demo && healthy;
  heatCanvas.hidden = !visible;
  if (!visible || !heatContext) return;
  const container = view === 'pilot' ? $('viewport') : document.querySelector('.camera-view');
  if (heatCanvas.parentElement !== container) container.append(heatCanvas);
  const width = Math.max(container.clientWidth, 1), height = Math.max(container.clientHeight, 1);
  if (heatCanvas.width !== width || heatCanvas.height !== height) { heatCanvas.width = width; heatCanvas.height = height; }
  heatContext.clearRect(0, 0, width, height);
  const point = pieces.get('pieza_01')?.position.clone().project(wristCamera);
  if (!point || point.z < -1 || point.z > 1 || Math.abs(point.x) > 1.2 || Math.abs(point.y) > 1.2) {
    $('attention-note').textContent = 'Mapa ilustrativo · La pieza de ejemplo está fuera del encuadre.';
    return;
  }
  const x = (point.x + 1) * width / 2, y = (1 - point.y) * height / 2;
  const radius = Math.max(45, width * .15);
  const gradient = heatContext.createRadialGradient(x, y, 0, x, y, radius);
  gradient.addColorStop(0, 'rgba(255,65,35,.95)');
  gradient.addColorStop(.24, 'rgba(255,185,35,.8)');
  gradient.addColorStop(.5, 'rgba(165,210,55,.55)');
  gradient.addColorStop(.77, 'rgba(45,140,220,.25)');
  gradient.addColorStop(1, 'rgba(45,140,220,0)');
  heatContext.fillStyle = gradient;
  heatContext.fillRect(0, 0, width, height);
  heatCanvas.style.opacity = Number($('attention-opacity').value) / 100;
  $('attention-note').textContent = 'Halo sintético sobre la pieza azul de ejemplo. No procede de una política.';
}
function showDisconnected(message) {
  healthy = false;
  buddy.reportFault(message);
  $('connection').textContent = demo ? 'DEMO SIN CONEXIÓN' : 'SIN TELEMETRÍA';
  $('connection').classList.remove('live');
  $('live-dot').classList.add('neutral');
  $('live-state').textContent = 'Sin datos recientes';
  $('live-detail').textContent = message;
  $('scene-notice').hidden = false;
  $('scene-notice').textContent = lastPoseSignature === null ? 'Geometría de referencia · Sin posición medida' : 'Señal interrumpida · Última posición recibida';
  $('pose-status').textContent = 'Sin datos recientes';
  $('pause-demo').disabled = true;
}
function cameraUnavailable() {
  frameFresh = false;
  $('camera-status').textContent = 'SIN SEÑAL';
  $('camera-status').classList.remove('live');
  $('camera').hidden = true;
  $('virtual-camera').hidden = true;
  $('camera-empty').hidden = false;
  $('camera-age').textContent = '—';
  canvas.style.visibility = 'visible';
  $('trace-distance').textContent = '—';
}
async function updateFrame(id) {
  if (framePending || frameId === id) return;
  framePending = true;
  try {
    const response = await fetch(`/api/frame.jpg?id=${id}`, { cache: 'no-store' });
    if (!response.ok) throw new Error('Sin imagen');
    const url = URL.createObjectURL(await response.blob());
    const image = new Image();
    image.src = url;
    try { await image.decode(); } catch (error) { URL.revokeObjectURL(url); throw error; }
    if (frameURL) URL.revokeObjectURL(frameURL);
    frameURL = url;
    $('camera').src = url;
    frameId = id;
  } catch (error) {
    cameraUnavailable();
  } finally {
    framePending = false;
  }
}
async function poll() {
  try {
    const response = await fetch('/api/state', { signal: AbortSignal.timeout(1800), cache: 'no-store' });
    if (!response.ok) throw new Error(`Servidor: ${response.status}`);
    const state = await response.json();
    frameFresh = state.mode !== 'demo' && state.frame_age !== null && state.frame_age < 2;
    demo = state.mode === 'demo';
    $('joint-heading').textContent = demo ? 'POSICIÓN SIMULADA' : 'POSICIÓN MEDIDA';
    $('source').textContent = state.source;
    $('pause-demo').hidden = !demo;
    $('pause-demo').disabled = false;
    $('pause-demo').textContent = state.paused ? 'Reanudar demo' : 'Pausar demo';
    $('camera-title').textContent = 'CENITAL';
    $('attention-toggle').disabled = !demo;
    $('ghost-toggle').disabled = !demo;
    if (!demo) $('attention-note').textContent = 'Sin mapas de política conectados. Esta capa solo está disponible como ejemplo en la demo.';
    else if (!attentionEnabled) $('attention-note').textContent = 'Mapa sintético. No procede de una política ni explica sus decisiones.';
    if (state.connected && state.poses) {
      healthy = true;
      const signature = JSON.stringify(state.poses);
      if (signature !== lastPoseSignature) { applyPoses(state.poses, true, state.error); lastPoseSignature = signature; }
      $('connection').textContent = demo ? 'DEMO EN DIRECTO' : 'TELEMETRÍA EN DIRECTO';
      $('connection').classList.add('live');
      $('live-dot').classList.remove('neutral');
      $('scene-notice').hidden = true;
      $('pose-status').textContent = `${demo ? 'Demo' : 'Poses recibidas'} · ${(state.pose_age * 1000).toFixed(0)} ms desde la última muestra`;
      $('live-state').textContent = demo ? (state.paused ? 'Demo en pausa' : 'Demo en movimiento') : 'Siguiendo el robot';
      $('live-detail').textContent = demo ? 'Referencia, robot y cámara virtual de ejemplo · Sin hardware ni política conectados' : 'Poses recibidas de la fuente · Navegación independiente';
      $('joint-note').textContent = demo ? 'Ángulos de la demo · No son mediciones del robot físico.' : 'Ángulos en grados; «—» si la fuente solo proporciona poses.';
    } else showDisconnected(state.error || 'No llegan poses recientes de la fuente.');
    if (demo && healthy) {
      $('camera').hidden = true;
      $('virtual-camera').hidden = false;
      $('camera-empty').hidden = true;
      $('camera-status').textContent = state.paused ? 'EN PAUSA' : 'EN DIRECTO';
      $('camera-status').classList.add('live');
      $('camera-source').textContent = '';
      $('camera-age').textContent = 'SIMULADA';
      canvas.style.visibility = 'visible';
    } else if (!demo && state.frame_age !== null && state.frame_age < 2) {
      await updateFrame(state.frame_id);
      const available = frameId === state.frame_id;
      $('camera').hidden = !available || view !== 'pilot';
      canvas.style.visibility = !$('camera').hidden ? 'hidden' : 'visible';
      $('virtual-camera').hidden = !healthy;
      $('camera-empty').hidden = healthy;
      $('camera-status').textContent = available ? 'EN DIRECTO' : 'CARGANDO';
      $('camera-status').classList.toggle('live', available);
      $('camera-age').textContent = `${Math.round(state.frame_age * 1000)} ms`;
      $('camera-source').textContent = '';
    } else if (healthy) {
      $('camera').hidden = true;
      canvas.style.visibility = 'visible';
      $('virtual-camera').hidden = false;
      $('camera-empty').hidden = true;
      $('camera-source').textContent = '';
    } else cameraUnavailable();
  } catch (error) {
    showDisconnected(`No se puede acceder a la fuente: ${error.message}`);
    $('scene-notice').dataset.error = 'true';
    $('scene-notice').hidden = false;
    $('scene-notice').textContent = 'Sin conexión con la fuente';
    cameraUnavailable();
  }
}
let polling = false;
async function requestState() {
  if (polling) return;
  polling = true;
  try { await poll(); } finally { polling = false; }
}
document.querySelectorAll('[data-view]').forEach(button => button.addEventListener('click', () => setView(button.dataset.view)));
$('reset').addEventListener('click', () => setView('perspective'));
$('ghost-toggle').addEventListener('click', () => {
  ghostsVisible = !ghostsVisible;
  traces.visible = demo && ghostsVisible;
  $('ghost-toggle').setAttribute('aria-pressed', String(ghostsVisible));
  $('ghost-toggle').textContent = ghostsVisible ? 'Traza ✓' : 'Traza';
});
$('attention-toggle').addEventListener('click', () => {
  attentionEnabled = !attentionEnabled;
  $('attention-toggle').setAttribute('aria-pressed', String(attentionEnabled));
  $('attention-toggle').textContent = attentionEnabled ? 'Ocultar mapa de ejemplo' : 'Mostrar mapa de ejemplo';
});
function zoom(factor) { camera.position.sub(controls.target).multiplyScalar(factor).add(controls.target); controls.update(); }
$('zoom-in').addEventListener('click', () => zoom(.82));
$('zoom-out').addEventListener('click', () => zoom(1.22));
$('fullscreen').addEventListener('click', async () => {
  try {
    if (document.fullscreenElement) await document.exitFullscreen();
    else await document.querySelector('.studio').requestFullscreen();
  } catch (error) {
    $('scene-notice').dataset.error = 'true';
    $('scene-notice').hidden = false;
    $('scene-notice').textContent = 'No se pudo ampliar la escena';
  }
});
$('reconnect').addEventListener('click', requestState);
$('pause-demo').addEventListener('click', async () => {
  try {
    const response = await fetch('/api/demo/toggle', { method: 'POST' });
    if (!response.ok) throw new Error('No se pudo pausar la demo');
    await requestState();
  } catch (error) { $('live-detail').textContent = error.message; }
});
function tick() { $('clock').textContent = new Date().toLocaleTimeString('es-ES'); }
tick();
setInterval(tick, 1000);
try {
  await loadScene();
  animate();
  await requestState();
  setInterval(requestState, 50);
} catch (error) {
  $('scene-notice').dataset.error = 'true';
  $('scene-notice').hidden = false;
  $('scene-notice').textContent = 'No se pudo cargar la escena';
}
