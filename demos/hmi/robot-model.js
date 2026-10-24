import * as THREE from 'three';

// Detalles visuales de so101-model.html; poses métricas del HMI, sin cinemática paralela.
const orange = new THREE.MeshStandardMaterial({color:0xf47719, roughness:.84, metalness:0});
const black = new THREE.MeshStandardMaterial({color:0x202220, roughness:.72});
const rubber = new THREE.MeshStandardMaterial({color:0x101311, roughness:.98});
const metal = new THREE.MeshStandardMaterial({color:0x9b9d99, metalness:.8, roughness:.35});
const red = new THREE.MeshStandardMaterial({color:0xb32b22, roughness:.65});
const white = new THREE.MeshStandardMaterial({color:0xece7d8, roughness:.65});
const pcb = new THREE.MeshStandardMaterial({color:0x203229, roughness:.75});
const glass = new THREE.MeshStandardMaterial({color:0x122830, metalness:.6, roughness:.12});
function group(parent, x=0,y=0,z=0) {const g=new THREE.Group();g.position.set(x,y,z);parent.add(g);return g;}
function mesh(parent,geometry,material,x=0,y=0,z=0) {
  const m=new THREE.Mesh(geometry,material);m.position.set(x,y,z);m.castShadow=true;m.receiveShadow=true;parent.add(m);return m;
}
function box(parent,w,h,d,x,y,z,material=orange) {return mesh(parent,new THREE.BoxGeometry(w,h,d),material,x,y,z);}
function cylinder(parent,r,h,x,y,z,material=orange,axis='z') {
  const m=mesh(parent,new THREE.CylinderGeometry(r,r,h,32),material,x,y,z);
  if(axis==='z')m.rotation.x=Math.PI/2;
  if(axis==='x')m.rotation.z=Math.PI/2;
  return m;
}
function screw(parent,x,y,z,axis='z') {
  cylinder(parent,.0025,.0015,x,y,z,metal,axis);
  const slot=box(parent,.003,.0006,.0017,x,y,z,black);
  if(axis==='y')slot.rotation.x=Math.PI/2;
}
// Perfil de horquilla con ventana pasante: dos costillas impresas, no un bloque macizo.
function cheek(parent,length,z,width=.031) {
  const r=width/2,s=new THREE.Shape();
  s.moveTo(-r,0);s.absarc(0,0,r,Math.PI,Math.PI*2,false);
  s.lineTo(r,length);s.absarc(0,length,r,0,Math.PI,false);s.closePath();
  const hole=new THREE.Path();
  hole.moveTo(-r*.45,.027);hole.lineTo(-r*.45,length-.027);
  hole.lineTo(r*.45,length-.027);hole.lineTo(r*.45,.027);hole.closePath();
  if(length>.07)s.holes.push(hole);
  const g=new THREE.ExtrudeGeometry(s,{depth:.006,bevelEnabled:true,bevelSegments:2,steps:1,bevelSize:.001,bevelThickness:.001,curveSegments:20});
  mesh(parent,g,orange,0,0,z-.003);
  for(const y of [0,length]) {cylinder(parent,.009,.002,0,y,z+Math.sign(z)*.004,orange);screw(parent,0,y,z+Math.sign(z)*.0055);}
}
function fork(parent,length,half=.022) {
  cheek(parent,length,-half);cheek(parent,length,half);
  box(parent,.027,.012,half*2,0,length*.53,0);
}
// Carcasa aproximada STS3215 (45 × 24 × 35 mm), orejas, tapa y tornillos.
function servo(parent,x=0,y=0,z=0) {
  const g=group(parent,x,y,z);
  box(g,.045,.024,.035,-.012,-.009,0,black);
  box(g,.049,.003,.039,-.012,-.020,0,black);
  for(const side of [-1,1]) {
    box(g,.009,.006,.041,-.012+side*.026,-.006,0,black);
    for(const face of [-1,1])screw(g,-.012+side*.019,-.014,face*.018);
  }
  cylinder(g,.010,.003,0,0,.020,metal);
  cylinder(g,.007,.004,0,0,.023,orange);
  screw(g,0,0,.026);
  return g;
}
function wire(parent,points,material,r=.00085) {
  return mesh(parent,new THREE.TubeGeometry(new THREE.CatmullRomCurve3(points.map(p=>new THREE.Vector3(...p))),24,r,6,false),material);
}
function localWires(parent,length,z=.029) {
  for(let i=0;i<2;i++)wire(parent,[[-.023,.005,z+i*.002],[-.041,length*.3,z+.005+i*.002],[-.038,length*.72,z+.006+i*.002],[-.015,length-.01,z+i*.002]],i?red:white);
}

export function createSO101(reference) {
  const root = new THREE.Group();
  const links = {};
  const frames = {};
  for (const [name, [pos, quat]] of Object.entries(reference.links)) {
    const frame = new THREE.Matrix4().compose(new THREE.Vector3(...pos),
      new THREE.Quaternion(quat[1], quat[2], quat[3], quat[0]), new THREE.Vector3(1, 1, 1));
    frames[name] = frame;
    links[name] = group(root);
    links[name].name = name;
  }
  // Las piezas usan +Y longitudinal y +Z para el eje del servo.
  // Se orientan UNA VEZ en el marco local de cada eslabón del SO-101.
  function between(name, next) {
    const start = new THREE.Vector3().setFromMatrixPosition(frames[name]);
    const end = new THREE.Vector3().setFromMatrixPosition(frames[next]);
    const y = end.clone().applyMatrix4(frames[name].clone().invert());
    const length = start.distanceTo(end);
    y.normalize();
    const z = new THREE.Vector3(0, 0, 1).addScaledVector(y, -y.z).normalize();
    const x = new THREE.Vector3().crossVectors(y, z).normalize();
    const body = group(links[name]);
    body.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(x, y, z));
    return {body, length};
  }
  const base = group(links.base);
  base.rotation.x = Math.PI / 2;
  // Mantener la superficie de apoyo del modelo anterior (z = 30.1 mm).
  base.position.set(reference.links.shoulder[0][0] - reference.links.base[0][0],
    reference.links.shoulder[0][1] - reference.links.base[0][1], .0301);
  box(base,.104,.008,.082,0,.004,0);
  for(const x of [-.043,.043])for(const z of [-.032,.032])screw(base,x,.009,z,'y');
  box(base,.054,.022,.047,0,.020,0);
  const baseServo=group(base,0,.043,0);baseServo.rotation.x=-Math.PI/2;servo(baseServo);
  cylinder(base,.027,.012,0,.058,0,orange,'y');
  const shoulder = between('shoulder','upper_arm');
  fork(shoulder.body,shoulder.length,.023);
  const arm = between('upper_arm','lower_arm');
  servo(arm.body);fork(arm.body,arm.length);localWires(arm.body,arm.length);
  const forearm = between('lower_arm','wrist');
  servo(forearm.body);fork(forearm.body,forearm.length,.025);localWires(forearm.body,forearm.length,.032);
  servo(forearm.body,0,forearm.length,0);
  const wrist = between('wrist','gripper');
  cheek(wrist.body,wrist.length,-.025,.037);cheek(wrist.body,wrist.length,.025,.037);
  box(wrist.body,.040,.013,.050,0,wrist.length*.5,0);
  const rollServo=group(wrist.body,0,wrist.length-.016,0);rollServo.rotation.x=Math.PI/2;servo(rollServo);
  cylinder(wrist.body,.022,.010,0,wrist.length-.005,0,orange,'y');
  const hand = group(links.gripper);
  hand.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(
    new THREE.Vector3(-1,0,0),new THREE.Vector3(0,0,-1),new THREE.Vector3(0,-1,0)));
  box(hand,.063,.017,.051,0,.0085,0);
  servo(hand,.011,.025,0);
  // Dedos esbeltos: nacen en su soporte, con almohadillas compactas.
  function finger(parent,side,length) {
    const g=group(parent);g.rotation.z=side*.12;
    const profile=new THREE.Shape();
    profile.moveTo(-.0055,-.004);profile.lineTo(.0055,-.004);
    profile.lineTo(.004,length-.010);profile.lineTo(-.004,length-.010);profile.closePath();
    mesh(g,new THREE.ExtrudeGeometry(profile,{depth:.014,bevelEnabled:true,
      bevelSize:.0007,bevelThickness:.0007,bevelSegments:2,steps:1}),orange,0,0,-.007);
    box(g,.013,.019,.017,0,length-.009,0,rubber);
    for(let i=0;i<4;i++)box(g,.001,.001,.016,-side*.0065,length-.015+i*.0035,0,black);
    return g;
  }
  const fixedMount=group(hand,.025,.014,0);
  fixedMount.name='fixed-finger-mount';
  box(fixedMount,.015,.014,.022,0,0,0);
  finger(fixedMount,1,.073);
  const moving=group(links.jaw);
  moving.name='moving-finger-mount';
  // Orientación de reposo de la mano; la mandíbula sigue su propio sexto eje.
  const handWorld=frames.gripper.clone().multiply(new THREE.Matrix4().makeRotationFromQuaternion(hand.quaternion));
  moving.quaternion.setFromRotationMatrix(frames.jaw.clone().invert().multiply(handWorld));
  // Soporte fijo hasta el eje real de apertura, sin desplazar la articulación.
  const bearingPosition=new THREE.Vector3().setFromMatrixPosition(frames.jaw).applyMatrix4(handWorld.clone().invert());
  const bearing=group(hand,bearingPosition.x,bearingPosition.y,bearingPosition.z);
  bearing.name='jaw-bearing';
  box(bearing,.018,.020,.018,0,-.005,0);
  cylinder(bearing,.009,.026,0,0,0,orange);
  finger(moving,-1,.065);
  cylinder(moving,.007,.028,0,0,0,orange);screw(moving,0,0,.015);
  const cam=group(hand,0,.006,-.030);
  box(cam,.012,.035,.012,0,.010,-.006);
  const plate=group(cam,0,.033,-.016);plate.rotation.x=-55*Math.PI/180;
  box(plate,.063,.045,.004,0,0,0);
  box(plate,.050,.034,.002,0,0,.0035,pcb);
  for(const x of [-.022,.022])for(const y of [-.013,.013])screw(plate,x,y,.006);
  box(plate,.020,.020,.006,0,0,.007,black);
  cylinder(plate,.010,.012,0,0,.015,black);
  cylinder(plate,.007,.001,0,0,.0215,glass);
  const lens=group(plate,0,0,.023);
  wire(cam,[[.022,.027,-.008],[.038,.020,-.025],[.028,-.010,-.025],[0,-.008,0]],black,.0018);
  const bridges=[];
  for(const [a,b,length] of [[arm.body,forearm.body,arm.length],[forearm.body,wrist.body,forearm.length],[wrist.body,hand,wrist.length]]) {
    const end=group(a,-.020,length-.012,.029),start=group(b,-.025,.004,.028);
    for(let i=0;i<2;i++)bridges.push({end,start,i,line:wire(root,[[0,0,0],[0,.01,0],[0,.02,0]],i?red:white)});
  }
  function update(poses) {
    for(const [name,[pos,quat]] of Object.entries(poses.links || {}))if(links[name]) {
      links[name].position.fromArray(pos);
      links[name].quaternion.set(quat[1],quat[2],quat[3],quat[0]);
    }
    root.updateMatrixWorld(true);
    for(const {end,start,i,line} of bridges) {
      const a=root.worldToLocal(end.getWorldPosition(new THREE.Vector3()));
      const b=root.worldToLocal(start.getWorldPosition(new THREE.Vector3()));
      a.x+=i*.002;b.x+=i*.002;
      const middle=a.clone().lerp(b,.5);middle.x+=.018;
      line.geometry.dispose();
      line.geometry=new THREE.TubeGeometry(new THREE.CatmullRomCurve3([a,middle,b]),16,.0008,6,false);
    }
  }
  function updateCamera(camera) {
    lens.getWorldPosition(camera.position);
    const orientation=lens.getWorldQuaternion(new THREE.Quaternion());
    camera.up.copy(new THREE.Vector3(0,1,0).applyQuaternion(orientation));
    camera.lookAt(camera.position.clone().add(new THREE.Vector3(0,0,1).applyQuaternion(orientation)));
    camera.updateMatrixWorld(true);
  }
  update(reference);
  return {root, links, lens, update, updateCamera};
}

// Cajas de la foto: medidas aproximadas en metros, con Z vertical como la mesa.
export function createWorkcellBoxes() {
  const root = new THREE.Group();
  root.name = 'photo-workcell-boxes';
  const green = new THREE.MeshStandardMaterial({color:0x315d45,roughness:.94});
  const crateOrange = new THREE.MeshStandardMaterial({color:0xc75b2e,roughness:.95});
  const wornGreen = new THREE.MeshStandardMaterial({color:0x72816b,roughness:1});
  const wornOrange = new THREE.MeshStandardMaterial({color:0x93543b,roughness:1});
  const basketWhite = new THREE.MeshStandardMaterial({color:0xe4e1d7,roughness:.85});
  const castMetal = new THREE.MeshStandardMaterial({color:0x79817b,metalness:.5,roughness:.72});
  function block(parent,w,d,h,x,y,z,material) {
    return box(parent,w,d,h,x,y,z,material);
  }
  function rectangle(w,d,r) {
    const shape = new THREE.Shape();
    shape.moveTo(-w/2+r,-d/2);shape.lineTo(w/2-r,-d/2);
    shape.quadraticCurveTo(w/2,-d/2,w/2,-d/2+r);shape.lineTo(w/2,d/2-r);
    shape.quadraticCurveTo(w/2,d/2,w/2-r,d/2);shape.lineTo(-w/2+r,d/2);
    shape.quadraticCurveTo(-w/2,d/2,-w/2,d/2-r);shape.lineTo(-w/2,-d/2+r);
    shape.quadraticCurveTo(-w/2,-d/2,-w/2+r,-d/2);
    return shape;
  }
  function rim(parent,w,d,t,h,z,material) {
    const outline=rectangle(w,d,.008);
    outline.holes.push(new THREE.Path(rectangle(w-2*t,d-2*t,.004).getPoints()));
    return mesh(parent,new THREE.ExtrudeGeometry(outline,{depth:h,bevelEnabled:false,curveSegments:8}),material,0,0,z);
  }
  function tray(name,x,y,w,d,h,material,worn) {
    const tray=group(root,x,y);tray.name=name;
    tray.userData={width:w,depth:d,height:h,bottom:.006};
    mesh(tray,new THREE.ExtrudeGeometry(rectangle(w,d,.008),{depth:.006,bevelEnabled:false,curveSegments:8}),material);
    rim(tray,w,d,.005,h-.006,.006,material);
    rim(tray,w+.006,d+.006,.009,.007,h-.007,material);
    // Nervios exteriores y pequeños desgastes en las esquinas del reborde.
    for(const side of [-1,1]) {
      for(const fraction of [-.34,-.12,.12,.34])block(tray,.009,.008,h-.014,w*side/2,d*fraction,h/2,material);
      for(const fraction of [-.30,0,.30])block(tray,.009,.010,h-.014,w*fraction,d*side/2,h/2,material);
      for(const corner of [-1,1])block(tray,.019,.009,.001,w*corner*.40,d*side/2,h+.0006,worn);
    }
    return tray;
  }
  const greenTray=tray('green-parts-tray',-.20,-.20,.24,.32,.074,green,wornGreen);
  tray('orange-empty-crate',.16,-.37,.32,.36,.065,crateOrange,wornOrange);
  // Tres piezas fundidas aproximadas, con placas, orejas y nervios metálicos.
  for(const [x,y,angle] of [[-.033,.055,.4],[.028,-.02,-.65],[-.025,-.085,.8]]) {
    const casting=group(greenTray,x,y,.009);casting.rotation.z=angle;
    const profile=new THREE.Shape();
    profile.moveTo(-.031,-.019);profile.lineTo(-.017,-.031);profile.lineTo(.023,-.031);
    profile.lineTo(.033,-.013);profile.lineTo(.023,.025);profile.lineTo(-.019,.025);profile.closePath();
    const hole=new THREE.Path();hole.absarc(.012,-.015,.0035,0,Math.PI*2,true);profile.holes.push(hole);
    mesh(casting,new THREE.ExtrudeGeometry(profile,{depth:.006,bevelEnabled:true,bevelSize:.001,bevelThickness:.001,bevelSegments:1}),castMetal);
    for(const side of [-1,1])block(casting,.005,.042,.025,.019*side,-.002,.018,castMetal);
    block(casting,.035,.006,.021,0,.018,.016,castMetal);
  }
  const basket=group(root,-.12,-.43);basket.name='white-wire-basket';
  const w=.12,d=.10,h=.042;
  basket.userData={width:w,depth:d,height:h,bottom:.0015};
  rim(basket,w,d,.003,.003,h-.003,basketWhite);
  rim(basket,w,d,.002,.002,0,basketWhite);
  const wirePoints=[];
  function segment(a,b){wirePoints.push(...a,...b);}
  const step=.005;
  for(let x=-w/2+.004;x<w/2;x+=step) {
    segment([x,-d/2,.002],[x,d/2,.002]);
    for(const side of [-1,1])segment([x,side*d/2,.002],[x,side*d/2,h]);
  }
  for(let y=-d/2+.004;y<d/2;y+=step) {
    segment([-w/2,y,.002],[w/2,y,.002]);
    for(const side of [-1,1])segment([side*w/2,y,.002],[side*w/2,y,h]);
  }
  for(let z=.007;z<h;z+=step) {
    segment([-w/2,-d/2,z],[w/2,-d/2,z]);segment([-w/2,d/2,z],[w/2,d/2,z]);
    segment([-w/2,-d/2,z],[-w/2,d/2,z]);segment([w/2,-d/2,z],[w/2,d/2,z]);
  }
  const lattice=new THREE.LineSegments(new THREE.BufferGeometry().setAttribute('position',new THREE.Float32BufferAttribute(wirePoints,3)),new THREE.LineBasicMaterial({color:0xe9e6df}));
  basket.add(lattice);
  for(const x of [-w/2,w/2])for(const y of [-d/2,d/2])block(basket,.003,.003,h,x,y,h/2,basketWhite);
  return root;
}
