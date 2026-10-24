import * as THREE from 'three';

export const BUDDY_CONFIG={armAvoidDistance:.115,peekDistance:.145,peekHoldSeconds:1.8,audioListenThreshold:.035,audioEchoThreshold:.20,audioSpikeDelta:.22};

// Compañero autónomo situado en la mesa, separado del modelo y la cinemática del brazo.
export function createBuddy() {
  const root = new THREE.Group();
  root.name = 'buddy-de-mesa';
  root.position.set(-.37, .075, 0);
  const shadow = new THREE.Mesh(new THREE.CircleGeometry(.024,32),
    new THREE.MeshBasicMaterial({color:0x26352f,transparent:true,opacity:.17,depthWrite:false}));
  shadow.name='buddy-shadow';shadow.position.set(root.position.x,root.position.y,.0005);
  shadow.scale.y=.46;

  const shell = new THREE.MeshStandardMaterial({color:0xe97432, roughness:.72});
  const cream = new THREE.MeshStandardMaterial({color:0xf5eedc, roughness:.8});
  const dark = new THREE.MeshStandardMaterial({color:0x26352f, roughness:.55});
  const metal = new THREE.MeshStandardMaterial({color:0x929a94, metalness:.55, roughness:.38});
  const white = new THREE.MeshStandardMaterial({color:0xfffdf4, roughness:.38});
  const pupilMat = new THREE.MeshStandardMaterial({color:0x17241f, roughness:.4});
  const gold = new THREE.MeshStandardMaterial({color:0xf4b94b, emissive:0x4a2600, roughness:.35});
  const ledMaterial = new THREE.MeshStandardMaterial({color:0xa7b092, emissive:0x20291b, roughness:.32});
  const screenMaterial = new THREE.MeshBasicMaterial({color:0x20312d, side:THREE.DoubleSide, toneMapped:false});
  function ball(parent, material, position, scale) {
    const item = new THREE.Mesh(new THREE.SphereGeometry(1,20,16),material);
    item.position.set(...position); item.scale.set(...scale);
    item.castShadow=true; item.receiveShadow=true; parent.add(item); return item;
  }
  function rod(parent, radius, length, material, position) {
    const item = new THREE.Mesh(new THREE.CylinderGeometry(radius,radius,length,12),material);
    item.position.set(...position); item.rotation.x=Math.PI/2; item.castShadow=true; parent.add(item); return item;
  }

  // El torso actúa como centro cinemático de hombros, cuello y cabeza.
  const torso=new THREE.Group();torso.position.set(0,0,.027);root.add(torso);
  ball(torso,shell,[0,0,0],[.025,.020,.021]);
  ball(torso,dark,[0,-.018,-.006],[.014,.005,.010]);
  const monitorFrame=new THREE.Mesh(new THREE.BoxGeometry(.023,.002,.015),dark);
  monitorFrame.position.set(0,-.021,-.006);monitorFrame.castShadow=true;torso.add(monitorFrame);
  const monitor=new THREE.Mesh(new THREE.PlaneGeometry(.020,.012),screenMaterial);
  monitor.position.set(0,-.0225,-.006);monitor.rotation.x=Math.PI/2;torso.add(monitor);
  const chestLight=ball(torso,ledMaterial,[0,-.022,-.016],[.003,.0015,.003]);
  const evolutionBadge=ball(torso,gold,[.017,-.014,.012],[.004,.002,.004]);evolutionBadge.visible=false;
  const backpack=new THREE.Mesh(new THREE.BoxGeometry(.021,.012,.017),shell);
  backpack.position.set(0,.018,-.003);backpack.castShadow=true;backpack.visible=false;torso.add(backpack);

  const head=new THREE.Group();head.position.set(0,0,.029);torso.add(head);
  ball(head,cream,[0,0,0],[.021,.019,.021]);
  ball(head,dark,[0,-.014,-.001],[.015,.007,.011]);
  const eyes=[];
  for(const side of [-1,1]){
    const sclera=ball(head,white,[side*.007,-.019,.004],[.0057,.003,.007]);
    const pupil=ball(head,pupilMat,[side*.007,-.022,.004],[.0025,.002,.0038]);
    eyes.push({sclera,pupil,side});
  }
  const mouthIdle=new THREE.Mesh(new THREE.TorusGeometry(.0038,.0007,6,16,Math.PI),dark);
  mouthIdle.position.set(0,-.020,-.009);mouthIdle.rotation.x=Math.PI/2;head.add(mouthIdle);
  const mouthHappy=new THREE.Mesh(new THREE.TorusGeometry(.0048,.0008,6,18,Math.PI),dark);
  mouthHappy.position.set(0,-.020,-.009);mouthHappy.rotation.x=Math.PI/2;mouthHappy.visible=false;head.add(mouthHappy);
  const mouthOpen=ball(head,dark,[0,-.021,-.009],[.0035,.0015,.001]);mouthOpen.visible=false;

  // Antena articulada con un muelle amortiguado en el movimiento.
  const antenna=new THREE.Group();antenna.position.set(0,0,.018);head.add(antenna);
  rod(antenna,.0018,.010,dark,[0,0,.005]);
  ball(antenna,gold,[0,0,.012],[.0045,.0045,.0045]);

  // Brazo real: hombro, húmero, codo, antebrazo, muñeca y pinza.
  const arms=[];
  for(const side of [-1,1]){
    const shoulder=new THREE.Group();shoulder.position.set(side*.021,0,.009);torso.add(shoulder);
    ball(shoulder,dark,[0,0,0],[.007,.007,.007]);
    ball(shoulder,shell,[0,-.004,0],[.0045,.004,.0045]);
    rod(shoulder,.0042,.014,shell,[0,0,-.007]);
    const elbow=new THREE.Group();elbow.position.set(0,0,-.014);shoulder.add(elbow);
    ball(elbow,dark,[0,0,0],[.0055,.0055,.0055]);
    ball(elbow,metal,[0,-.004,0],[.003,.0015,.003]);
    rod(elbow,.0036,.013,cream,[0,0,-.0065]);
    const wrist=new THREE.Group();wrist.position.set(0,0,-.013);elbow.add(wrist);
    ball(wrist,shell,[0,0,0],[.0045,.0045,.0045]);
    const hand=new THREE.Group();hand.position.set(0,-.002,-.003);wrist.add(hand);
    ball(hand,dark,[0,0,0],[.0055,.006,.004]);
    const fingers=[];
    for(const fingerSide of [-1,1]){
      const finger=new THREE.Group();finger.position.set(fingerSide*.0035,-.004,0);hand.add(finger);
      rod(finger,.0015,.006,cream,[0,-.003,0]);
      ball(finger,shell,[0,-.006,0],[.002,.002,.002]);
      fingers.push(finger);
    }
    arms.push({side,shoulder,elbow,wrist,fingers});
  }

  // Piernas unidas a las caderas, con rodillas, tobillos y pies articulados.
  const legs=[];
  for(const side of [-1,1]){
    const hip=new THREE.Group();hip.position.set(side*.011,0,-.009);torso.add(hip);
    ball(hip,dark,[0,0,0],[.006,.006,.006]);
    rod(hip,.0048,.010,cream,[0,0,-.005]);
    const knee=new THREE.Group();knee.position.set(0,0,-.010);hip.add(knee);
    ball(knee,shell,[0,0,0],[.005,.005,.005]);
    ball(knee,dark,[0,-.004,0],[.0024,.0015,.0024]);
    rod(knee,.0039,.007,dark,[0,0,-.0035]);
    const ankle=new THREE.Group();ankle.position.set(0,0,-.007);knee.add(ankle);
    ball(ankle,metal,[0,0,0],[.0035,.0035,.0035]);
    const foot=new THREE.Group();ankle.add(foot);
    ball(foot,dark,[0,-.003,.003],[.008,.013,.004]);
    ball(foot,shell,[0,-.011,.004],[.006,.004,.003]);
    legs.push({side,hip,knee,ankle,foot});
  }
  const antennaVelocity={x:0,y:0};
  let gaitPhase=0;
  const echoRings=[0,1].map(index=>{
    const ring=new THREE.Mesh(new THREE.TorusGeometry(.012+index*.006,.0007,6,28),
      new THREE.MeshBasicMaterial({color:0x78947c,transparent:true,opacity:.4,depthWrite:false}));
    ring.position.set(0,0,.073);ring.rotation.x=Math.PI/2;ring.visible=false;root.add(ring);return ring;
  });
  const startZ=0;
  let mood='idle';
  let hasPin=false;
  let lastStage=null, visit=null, pendingBin=null, previousTime=null;
  let roamPath=[],roamIndex=0,roamPause=0;
  let armPoints=[],jointSample=null,prediction=null,resultIsExample=false;
  let holding=false,armWasNear=false,evadePath=null,evadeReturn=null;
  let audioInput=0,audioLevel=0,audioAlertUntil=0,audioMood='silent';
  let commentary='Te acompaño desde la mesa.',commentaryUntil=0,lastFault='';
  const journal=[],deposits={green:0,white:0};
  let journalVersion=0;
  const home=new THREE.Vector3(-.37,.075,.003);
  const roamStops=[new THREE.Vector2(-.40,.14),new THREE.Vector2(.37,.14),new THREE.Vector2(-.40,-.52)];
  const bins={
    'green-parts-tray':{x:-.20,y:-.20,width:.24,depth:.32,wall:.074},
    'orange-empty-crate':{x:.16,y:-.37,width:.32,depth:.36,wall:.065},
    'white-wire-basket':{x:-.12,y:-.43,width:.12,depth:.10,wall:.042},
  };
  const footprint=.025, bounds={left:-.435,right:.435,top:-.555,bottom:.225};
  function rects(extra=.004){return Object.values(bins).map(b=>({
    left:b.x-b.width/2-footprint-extra,right:b.x+b.width/2+footprint+extra,
    top:b.y-b.depth/2-footprint-extra,bottom:b.y+b.depth/2+footprint+extra,
  }));}
  function validPoint(p){return p.x>bounds.left&&p.x<bounds.right&&p.y>bounds.top&&p.y<bounds.bottom&&
    !rects().some(r=>p.x>=r.left&&p.x<=r.right&&p.y>=r.top&&p.y<=r.bottom);}
  function hitsRect(a,b,r){
    let lo=0,hi=1;
    for(const [axis,min,max] of [['x',r.left,r.right],['y',r.top,r.bottom]]){
      const d=b[axis]-a[axis];
      if(Math.abs(d)<1e-9){if(a[axis]<min||a[axis]>max)return false;}
      else{let u=(min-a[axis])/d,v=(max-a[axis])/d;if(u>v)[u,v]=[v,u];lo=Math.max(lo,u);hi=Math.min(hi,v);if(lo>hi)return false;}
    }
    return true;
  }
  function clearPath(a,b){return validPoint(a)&&validPoint(b)&&!rects().some(r=>hitsRect(a,b,r));}
  function pathTo(start,goal){
    if(!validPoint(start)||!validPoint(goal))return null;
    const nodes=[start,goal],pad=footprint+.014;
    for(const b of Object.values(bins)){
      const x0=b.x-b.width/2-pad,x1=b.x+b.width/2+pad,y0=b.y-b.depth/2-pad,y1=b.y+b.depth/2+pad;
      for(const p of [{x:x0,y:y0},{x:x0,y:y1},{x:x1,y:y0},{x:x1,y:y1}])if(validPoint(p))nodes.push(p);
    }
    const cost=nodes.map(()=>Infinity),prev=nodes.map(()=>-1),used=new Set();cost[0]=0;
    while(used.size<nodes.length){
      let u=-1;for(let i=0;i<nodes.length;i++)if(!used.has(i)&&(u<0||cost[i]<cost[u]))u=i;
      if(u<0||!Number.isFinite(cost[u]))break;
      if(u===1){const path=[];for(let i=1;i!==0;i=prev[i])path.unshift(nodes[i]);return path;}
      used.add(u);
      for(let v=0;v<nodes.length;v++)if(!used.has(v)&&clearPath(nodes[u],nodes[v])){
        const d=Math.hypot(nodes[u].x-nodes[v].x,nodes[u].y-nodes[v].y);
        if(cost[u]+d<cost[v]){cost[v]=cost[u]+d;prev[v]=u;}
      }
    }
    return null;
  }
  function nextRoam(){
    roamIndex=(roamIndex+1)%roamStops.length;
    const goal=roamStops[roamIndex];
    roamPath=pathTo({x:root.position.x,y:root.position.y},{x:goal.x,y:goal.y})||[];
  }
  roamPath=pathTo({x:root.position.x,y:root.position.y},{x:roamStops[0].x,y:roamStops[0].y})||[];
  function boxApproach(id){
    const b=bins[id],pad=footprint+.018;
    const candidates=[
      {side:'left',p:{x:b.x-b.width/2-pad,y:b.y}},
      {side:'right',p:{x:b.x+b.width/2+pad,y:b.y}},
      {side:'back',p:{x:b.x,y:b.y-b.depth/2-pad}},
      {side:'front',p:{x:b.x,y:b.y+b.depth/2+pad}},
    ];
    let best=null;
    for(const c of candidates){
      const path=pathTo({x:root.position.x,y:root.position.y},c.p);
      if(path){const length=path.reduce((sum,p,i)=>sum+Math.hypot(p.x-(i?path[i-1].x:root.position.x),p.y-(i?path[i-1].y:root.position.y)),0);
        if(!best||length<best.length)best={...c,path,length};
      }
    }
    if(!best)return null;
    const inside={x:b.x,y:b.y};
    if(best.side==='left')inside.x=b.x-b.width/2+b.wall*.7;
    if(best.side==='right')inside.x=b.x+b.width/2-b.wall*.7;
    if(best.side==='back')inside.y=b.y-b.depth/2+b.wall*.7;
    if(best.side==='front')inside.y=b.y+b.depth/2-b.wall*.7;
    return {...best,inside,exit:{x:best.p.x,y:best.p.y}};
  }
  function startVisit(id){
    const approach=boxApproach(id);
    if(!approach)return false;
    visit={id,approach,path:approach.path.map(p=>new THREE.Vector3(p.x,p.y,root.position.z)),mode:'walk-in',time:0};
    roamPath=[];mood='happy';return true;
  }
  function finishVisit(){
    const next=pendingBin;pendingBin=null;visit=null;roamPause=.45;
    if(next)startVisit(next);else nextRoam();
  }
  function walkRoute(points,dt,speed){
    while(points.length&&Math.hypot(root.position.x-points[0].x,root.position.y-points[0].y)<.004)points.shift();
    if(!points.length)return true;
    const p=points[0],dx=p.x-root.position.x,dy=p.y-root.position.y,d=Math.hypot(dx,dy),step=Math.min(d,speed*dt);
    root.position.x+=dx/d*step;root.position.y+=dy/d*step;
    return points.length===1&&d<=step;
  }

  function beginHop(mode){
    visit.mode=mode;visit.time=0;
    visit.from=new THREE.Vector2(root.position.x,root.position.y);
    const p=['hop-in','peek-in'].includes(mode)?visit.approach.inside:visit.approach.exit;
    visit.to=new THREE.Vector2(p.x,p.y);
  }

  function nearestArm(){
    let best=Infinity,point=null;
    for(const p of armPoints){const d=Math.hypot(p.x-root.position.x,p.y-root.position.y);if(d<best){best=d;point=p;}}
    return {distance:best,point};
  }
  function planEvade(){
    const {point}=nearestArm();if(!point)return false;
    const dx=root.position.x-point.x,dy=root.position.y-point.y,base=Math.atan2(dy,dx);
    const candidates=[];
    for(let i=0;i<12;i++){
      const a=base+(i-5.5)*Math.PI/12,d=.17;
      const p={x:root.position.x+Math.cos(a)*d,y:root.position.y+Math.sin(a)*d};
      if(validPoint(p)){const path=pathTo({x:root.position.x,y:root.position.y},p);if(path)candidates.push({p,path,rank:path.reduce((sum,q,index)=>sum+Math.hypot(q.x-(index?path[index-1].x:root.position.x),q.y-(index?path[index-1].y:root.position.y)),0)});}
    }
    candidates.sort((a,b)=>a.rank-b.rank);
    if(!candidates.length)return false;
    evadePath=candidates[0].path.map(p=>new THREE.Vector3(p.x,p.y,root.position.z));
    const route=visit?.path||roamPath;
    evadeReturn=route.length?{x:route.at(-1).x,y:route.at(-1).y}:null;
    if(visit){visit.evadeResume=visit.mode;visit.mode='evade';visit.path=evadePath;}
    else roamPath=evadePath;
    return true;
  }

  function setState(stage, withPin=false) {
    hasPin=Boolean(withPin);
    const newDeposit=stage==='soltando'&&lastStage!=='soltando';
    lastStage=stage||null;
    mood=stage==='inspeccionando'?'curious'
      :stage==='clasificando'?(hasPin?'pin':'clear')
      :stage==='depositando'||stage==='soltando'||stage==='siguiente'?'happy'
      :'idle';
    ledMaterial.color.set(mood==='pin'?0x80ac55:mood==='clear'?0xf0dfb4:mood==='happy'?0xf0bd54:0xa7b092);
    ledMaterial.emissive.set(mood==='pin'?0x253d10:mood==='clear'?0x493d20:mood==='happy'?0x70420b:0x20291b);
    chestLight.material.color.set(mood==='pin'?0x80ac55:mood==='clear'?0xf0dfb4:mood==='happy'?0xf0bd54:0xa7b092);
    mouthIdle.visible=mood!=='happy';
    mouthHappy.visible=mood==='happy';
    if(newDeposit){
      const id=hasPin?'green-parts-tray':'white-wire-basket';
      if(visit){pendingBin=id;if(['inside','peek','peek-out','peek-in'].includes(visit.mode))beginHop('hop-out');}
      else if(!startVisit(id))mood='idle';
    }
  }
  let lastJournalStage=null;
  function setTelemetry(data={}){
    const workflow=data.workflow||{},stage=workflow.stage||null;
    armPoints=Array.isArray(data.armPoints)?data.armPoints.filter(p=>p&&Number.isFinite(p.x)&&Number.isFinite(p.y)&&Number.isFinite(p.z)):[];
    jointSample=data.joints&&typeof data.joints==='object'?data.joints:null;
    const explicit=workflow.prediction?.destination??workflow.predicted_bin??workflow.destination_prediction??null;
    prediction=explicit?String(explicit).toLowerCase():null;
    resultIsExample=Boolean(workflow.result_is_example);
    const closed=Number.isFinite(jointSample?.gripper)&&jointSample.gripper<.22;
    holding=Boolean(workflow.grasped??(stage==='recogiendo'&&closed));
    setState(stage,workflow.with_pin);
    if(stage&&stage!==lastJournalStage){
      const labels={recogiendo:'Contengo la respiración: la pinza está recogiendo.',elevando:'Pieza en el aire; sigo el movimiento.',inspeccionando:'Ojos abiertos: la pieza está bajo inspección.',clasificando:'La decisión se está preparando.',depositando:'Acompaño la pieza hasta su destino.',soltando:'¡Pieza soltada! Entrega observada.',siguiente:'Listo para observar la siguiente pieza.'};
      if(labels[stage])note(labels[stage]);
      if(stage==='clasificando'&&resultIsExample)note(`Señalo ${hasPin?'verde':'blanco'} según la etiqueta ilustrativa de la demo; no hay política conectada.`,'example');
      else if(stage==='clasificando'&&prediction)note(`La predicción recibida apunta a ${prediction}.`,'prediction');
      if(stage==='soltando'){
        const droppedTo=String(workflow.destination??workflow.dropped_in??'').toLowerCase();
        const green=droppedTo?/(verde|green|pin|perno)/.test(droppedTo):Boolean(workflow.with_pin);
        deposits[green?'green':'white']++;
        note(green?'Entrega observada en verde.':'Entrega observada en blanco.','drop');
      }
      if(stage==='siguiente'&&(deposits.green+deposits.white)>0)note(`Bitácora: ${deposits.green+deposits.white} entregas observadas; el HMI conserva sus propios contadores.`);
      lastJournalStage=stage;
    }
    if(data.error&&data.error!==lastFault){lastFault=data.error;reportFault(data.error);}
    if(!data.error)lastFault='';
  }
  function note(text,kind='info'){
    commentary=text;commentaryUntil=performance.now()+4200;
    if(journal.at(-1)?.text!==text){journal.push({time:new Date().toLocaleTimeString('es-ES',{hour:'2-digit',minute:'2-digit'}),text,kind});if(journal.length>28)journal.shift();journalVersion++;}
  }
  function reportFault(message){if(message)note(`Aviso de la fuente: ${message}`,'error');}
  function setAudioLevel(level){
    const next=THREE.MathUtils.clamp(Number(level)||0,0,1),spike=next-audioInput>BUDDY_CONFIG.audioSpikeDelta;
    audioInput=next;audioLevel+=(next-audioLevel)*.24;
    if(spike){audioAlertUntil=performance.now()+1250;audioMood='detective';note('¡He oído un cambio brusco! Escuchando.','audio');}
    else if(audioLevel>BUDDY_CONFIG.audioEchoThreshold)audioMood='echo';
    else if(audioLevel>BUDDY_CONFIG.audioListenThreshold)audioMood='sing';
    else if(performance.now()>audioAlertUntil)audioMood='silent';
  }
  function setScreenTexture(texture){screenMaterial.map=texture;screenMaterial.color.set(texture?0xffffff:0x20312d);screenMaterial.needsUpdate=true;}
  function setScreenVisible(visible){monitor.visible=Boolean(visible);monitorFrame.visible=Boolean(visible);}
  function setPrediction(destination){prediction=destination?String(destination).toLowerCase():null;}
  function getJournal(){return {version:journalVersion,entries:journal.slice(),summary:`${deposits.green} a verde · ${deposits.white} a blanco · ${journal.filter(e=>e.kind==='error').length} avisos`,observedDrops:deposits.green+deposits.white};}
  let evadeCooldown=0;

  function animate(time,cameraPosition=null) {
    const t=time/1000;
    const dt=previousTime===null?0:Math.min((time-previousTime)/1000,.04);previousTime=time;
    const oldX=root.position.x,oldY=root.position.y;
    let walking=false;
    const arm=nearestArm(),armNear=arm.distance<BUDDY_CONFIG.armAvoidDistance;
    if(visit){
      if(visit.mode==='inside'&&arm.distance<BUDDY_CONFIG.peekDistance&&!visit.peeking){visit.peeking=true;beginHop('peek-out');note('Asomo por encima de la caja: el brazo está cerca.');}
      if(!holding&&armNear&&!armWasNear&&['walk-in','walk-home'].includes(visit.mode)&&performance.now()>evadeCooldown&&planEvade()){
        evadeCooldown=performance.now()+1400;note('Me aparto un poco para dejar pasar el brazo.');
      }
      armWasNear=armNear;
      if(holding)visit.time=Math.max(0,visit.time-dt);
      else {
      visit.time+=dt;
      if(visit.mode==='evade'){
        walking=true;
        if(walkRoute(visit.path,dt,.085)){
          const resume=visit.evadeResume||'walk-home',goal=evadeReturn;evadeReturn=null;evadePath=null;
          visit.mode=resume;visit.path=goal?pathTo({x:root.position.x,y:root.position.y},goal)||[]:[];
        }
      }else if(visit.mode==='walk-in'){
        walking=true;
        if(walkRoute(visit.path,dt,.14)){
          root.position.x=visit.approach.exit.x;root.position.y=visit.approach.exit.y;
          visit.mode='cheer';visit.time=0;
        }
      }else if(visit.mode==='cheer'&&visit.time>1.35){beginHop('hop-in');}
      else if(['hop-in','hop-out','peek-out','peek-in'].includes(visit.mode)){
        const hopMode=visit.mode,entering=hopMode==='hop-in'||hopMode==='peek-in',p=Math.min(visit.time/1.05,1),ease=p*p*(3-2*p);
        root.position.x=THREE.MathUtils.lerp(visit.from.x,visit.to.x,ease);
        root.position.y=THREE.MathUtils.lerp(visit.from.y,visit.to.y,ease);
        root.position.z=startZ+Math.sin(Math.PI*p)*(bins[visit.id].wall+.035);
        if(p>=1){
          root.position.set(visit.to.x,visit.to.y,startZ);visit.time=0;
          if(entering){visit.mode='inside';visit.time=0;if(hopMode==='peek-in')visit.peeking=false;}
          else if(hopMode==='peek-out'){visit.mode='peek';visit.time=0;}
          else{visit.mode='walk-home';visit.path=pathTo({x:root.position.x,y:root.position.y},{x:home.x,y:home.y})||[];}
        }
      }else if(visit.mode==='peek'&&visit.time>BUDDY_CONFIG.peekHoldSeconds&&arm.distance>=BUDDY_CONFIG.peekDistance*.8)beginHop('peek-in');
      else if(visit.mode==='walk-home'){
        walking=true;
        if(walkRoute(visit.path,dt,.14))finishVisit();
      }
      const mode=visit?.mode;
      const towardCamera=mode==='cheer'||mode==='hop-in'||mode==='inside'||mode==='hop-out'||mode==='peek-out'||mode==='peek';
      const waypoint=mode==='walk-in'?visit?.path[0]:mode==='walk-home'?visit?.path[0]:null;
      if(towardCamera&&cameraPosition){
        const dx=cameraPosition.x-root.position.x,dy=cameraPosition.y-root.position.y;
        const target=Math.atan2(dx,-dy),delta=Math.atan2(Math.sin(target-root.rotation.z),Math.cos(target-root.rotation.z));
        root.rotation.z+=delta*(1-Math.exp(-5*dt));
      }else if(waypoint){
        const target=Math.atan2(waypoint.x-root.position.x,-(waypoint.y-root.position.y));
        const delta=Math.atan2(Math.sin(target-root.rotation.z),Math.cos(target-root.rotation.z));
        root.rotation.z+=delta*(1-Math.exp(-6*dt));
      }
      if(visit&&['hop-in','hop-out','peek-out','peek-in'].includes(visit.mode))
        root.position.z=startZ+Math.sin(Math.PI*Math.min(visit.time/1.05,1))*(bins[visit.id].wall+.035);
      else root.position.z=startZ;
      shadow.position.set(root.position.x,root.position.y,.0005);
      shadow.material.opacity=.17*Math.max(.18,1-Math.max(0,root.position.z-startZ)*5);
      shadow.scale.set(1+Math.max(0,root.position.z-startZ)*8,.46+Math.max(0,root.position.z-startZ)*3,1);
      }
    }else{
      if(!holding){
        if(armNear&&!armWasNear&&performance.now()>evadeCooldown&&planEvade()){evadeCooldown=performance.now()+1400;note('Me aparto un poco para dejar pasar el brazo.');}
        armWasNear=armNear;
        if(roamPause>0)roamPause=Math.max(0,roamPause-dt);
        else {
          walking=roamPath.length>0;
          if(walkRoute(roamPath,dt,evadePath ? .085 : .065)){
            if(evadePath){evadePath=null;roamPath=[];if(evadeReturn){roamPath=pathTo({x:root.position.x,y:root.position.y},evadeReturn)||[];evadeReturn=null;}else nextRoam();roamPause=.18;}
            else{roamPause=.8;nextRoam();}
          }
        }
      }
      root.position.z=startZ;
      const waypoint=roamPath[0];
      if(waypoint){
        const dx=waypoint.x-root.position.x,dy=waypoint.y-root.position.y,target=Math.atan2(dx,-dy);
        const delta=Math.atan2(Math.sin(target-root.rotation.z),Math.cos(target-root.rotation.z));
        root.rotation.z+=delta*(1-Math.exp(-4*dt));
      }else if(cameraPosition){
        const dx=cameraPosition.x-root.position.x,dy=cameraPosition.y-root.position.y,target=Math.atan2(dx,-dy);
        const delta=Math.atan2(Math.sin(target-root.rotation.z),Math.cos(target-root.rotation.z));
        root.rotation.z+=delta*(1-Math.exp(-3*dt));
      }
      shadow.position.set(root.position.x,root.position.y,.0005);shadow.material.opacity=.17;
      shadow.scale.set(1,.46,1);
    }
    const traveled=Math.hypot(root.position.x-oldX,root.position.y-oldY);
    if(walking&&dt>0)gaitPhase+=traveled/.042*2*Math.PI;
    const gait=Math.sin(gaitPhase),party=!holding&&visit&&['cheer','hop-in','inside','hop-out','peek-out','peek','peek-in'].includes(visit.mode);
    const hopping=!holding&&visit&&['hop-in','hop-out','peek-out','peek-in'].includes(visit.mode);
    const bounce=walking?.0015*Math.abs(Math.sin(gaitPhase*2)):
      hopping?0:party?.0015*Math.sin(t*8):mood==='happy'?.0025*Math.abs(Math.sin(t*7)):.0008*Math.sin(t*2.2);
    torso.position.z=.027+bounce;
    torso.rotation.x=holding?0:walking?.055*Math.cos(gaitPhase):party?.09*Math.sin(t*5):audioMood==='sing'?.08*Math.sin(t*4):mood==='curious'?.12+.045*Math.sin(t*2.1):.025*Math.sin(t*1.7);
    torso.rotation.y=holding?0:walking?.045*Math.sin(gaitPhase*2):party?.08*Math.sin(t*6):.035*Math.sin(t*1.4);
    head.rotation.x=holding?0:party?.10*Math.sin(t*6):mood==='curious'?.16+.08*Math.sin(t*2.4):mood==='happy'?.10*Math.sin(t*5):.025*Math.sin(t*1.3);
    head.rotation.y=holding?0:audioMood==='detective'?.5*Math.sin(t*7):.09*Math.sin(t*(mood==='curious'?2.7:1.4));
    head.rotation.z=holding?0:party?.09*Math.sin(t*5):mood==='curious'?.16*Math.sin(t*1.8):0;
    if(jointSample&&!holding){
      const pan=THREE.MathUtils.clamp(jointSample.shoulder_pan||0,-Math.PI,Math.PI);
      torso.rotation.z+=(pan*.24-torso.rotation.z)*Math.min(1,dt*3.5);
      torso.rotation.x=THREE.MathUtils.lerp(torso.rotation.x,THREE.MathUtils.clamp(jointSample.shoulder_lift||0,-1.4,1.4)*.22,.72);
    }

    const destinationHint=prediction||((resultIsExample&&jointSample&&lastStage==='clasificando')?(hasPin?'verde':'blanco'):null);
    const pointsToDestination=/verde|green|pin|perno|blanco|white|clear/.test(destinationHint||'');
    for(const {side,shoulder,elbow,wrist,fingers} of arms){
      const phase=gaitPhase+(side<0?0:Math.PI),step=Math.sin(phase);
      if(walking){
        shoulder.rotation.x=-.42*step;shoulder.rotation.y=-side*.08;
        elbow.rotation.x=.22+.24*Math.max(0,-step);wrist.rotation.x=-.10*step;
      }else if(party){
        shoulder.rotation.x=.16*Math.sin(t*7+side);shoulder.rotation.y=-side*(visit.mode==='inside'?.44:.72+.12*Math.sin(t*8));
        elbow.rotation.x=-.56+.12*Math.sin(t*8+side);wrist.rotation.x=.18*Math.sin(t*9+side);
      }else{
        shoulder.rotation.x=.045*Math.sin(t*1.6+side);shoulder.rotation.y=-side*.06;
        elbow.rotation.x=.28+.05*Math.sin(t*2+side);wrist.rotation.x=.04*Math.sin(t*2.5+side);
      }
      if(holding){shoulder.rotation.x*=.03;elbow.rotation.x*=.03;wrist.rotation.x*=.03;}
      if(pointsToDestination&&side===1){shoulder.rotation.x=-.78;shoulder.rotation.y=-side*.2;elbow.rotation.x=-.35;wrist.rotation.x=.12;}
      if(audioMood==='detective'&&side===1){shoulder.rotation.x=-.65;elbow.rotation.x=-.9;}
      if(jointSample&&!holding){
        shoulder.rotation.x=THREE.MathUtils.lerp(shoulder.rotation.x,THREE.MathUtils.clamp(jointSample.elbow_flex||0,-2,2)*.24,.68);
        elbow.rotation.x=THREE.MathUtils.lerp(elbow.rotation.x,THREE.MathUtils.clamp(jointSample.wrist_flex||0,-2,2)*.32,.68);
        wrist.rotation.x=THREE.MathUtils.lerp(wrist.rotation.x,THREE.MathUtils.clamp(jointSample.wrist_roll||0,-Math.PI,Math.PI)*.2,.68);
        for(const finger of fingers)finger.rotation.x=THREE.MathUtils.lerp(finger.rotation.x,THREE.MathUtils.clamp(jointSample.gripper||0,-1,1)*.24,.68);
      }
      shoulder.rotation.z=walking?.045*Math.sin(gaitPhase+side):party?.12*Math.sin(t*8+side):.025*Math.sin(t*1.8+side);
      for(let i=0;i<fingers.length;i++)fingers[i].rotation.x=party?.22*Math.sin(t*9+i+side):.03*Math.sin(t*2+i);
    }
    for(const {side,hip,knee,ankle,foot} of legs){
      const phase=gaitPhase+(side<0?Math.PI:0),step=Math.sin(phase);
      hip.rotation.x=walking?.48*step:party?.16*Math.sin(t*8+side):.025*Math.sin(t*1.6+side);
      knee.rotation.x=walking?.58*Math.max(0,-step):hopping?.82:party?.20*Math.max(0,Math.sin(t*8+side)):.06;
      ankle.rotation.x=walking?-.30*step-knee.rotation.x*.35:party?-.12*Math.sin(t*8+side):0;
      foot.rotation.x=walking?.12*step:0;
      if(holding){hip.rotation.x=0;knee.rotation.x=0;ankle.rotation.x=0;foot.rotation.x=0;}
    }
    const blink=(t%4.6)<.15?Math.max(.12,Math.abs(Math.sin((t%4.6)/.15*Math.PI))):1;
    for(const {sclera,pupil,side} of eyes){
      sclera.scale.y=.003*blink;pupil.scale.y=.002*blink;
      pupil.position.x=side*.007+.001*Math.sin(t*1.8+side);
    }
    const antennaTarget=holding?0:(walking?.16*Math.sin(gaitPhase+1):party?.30*Math.sin(t*9):audioMood==='sing'?.2*Math.sin(t*6):.035*Math.sin(t*2.1))+(hopping?.12*Math.sin(t*15):0);
    antennaVelocity.x+=((antennaTarget-antenna.rotation.x)*58-antennaVelocity.x*10)*dt;
    antenna.rotation.x+=antennaVelocity.x*dt;
    antenna.rotation.y=.12*Math.sin(t*2.8+(party?1:0));
    mouthOpen.visible=audioMood==='sing'&&audioLevel>BUDDY_CONFIG.audioListenThreshold;
    mouthOpen.scale.y=.0015+audioLevel*.003;
    for(let i=0;i<echoRings.length;i++){
      const active=audioMood==='echo',pulse=(t*.7+i*.38)%1;
      echoRings[i].visible=active;echoRings[i].scale.setScalar(active?.55+pulse*1.1:1);echoRings[i].material.opacity=active?.45*(1-pulse):0;
    }
    const evolved=deposits.green+deposits.white;
    evolutionBadge.visible=evolved>=3;backpack.visible=evolved>=6;
    if(evolved>=3)evolutionBadge.material.color.set(evolved>=8?0x91ad62:0xf4b94b);
  }
  return {root,shadow,setState,setTelemetry,setAudioLevel,setScreenTexture,setScreenVisible,setPrediction,reportFault,getJournal,getComment:()=>({text:commentary,until:commentaryUntil}),animate};
}
