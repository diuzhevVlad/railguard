const $ = id => document.getElementById(id);
const ui = {bag:$('bag'),topic:$('topic'),seek:$('seek'),seekPosition:$('seek-position'),seekEnd:$('seek-end'),notice:$('notice'),status:$('frame-status'),time:$('time-display')};
let detectionParameters={train_width_m:2.1,train_height_m:3,max_forward_m:150};
let simulationParameters={initial_speed_m_s:0};
let detectionGeneration=0,resumeAfterDetection=false;
async function applyDetectionParameters(){
  for(const input of $('detection-parameters').querySelectorAll('input'))if(!input.reportValidity())return;
  const next={train_width_m:Number($('train-width').value),train_height_m:Number($('train-height').value),max_forward_m:Number($('detection-distance').value)};
  if(!Object.values(next).every(Number.isFinite))return;
  if(Object.keys(next).every(key=>next[key]===detectionParameters[key]))return;
  const wasVisible=detectionVisible||(bagReady&&$('detect').disabled),wasPlaying=resumeAfterDetection||Boolean(socket&&!ended&&!paused);
  detectionParameters=next;detectionGeneration++;
  stopPlayback();activeDetectionId=null;detectionVisible=false;
  $('detection-legend').hidden=true;$('detection-job').hidden=true;$('detect').disabled=!bagReady;showDetectionButton();
  if(lastFrame&&lastBasePoints&&!mapEditor.active)renderFrame(lastBasePoints,{...lastFrame,detected_points:0});
  $('detection-parameter-status').textContent=`Active envelope · ${next.train_width_m} × ${next.train_height_m} m · detection to ${next.max_forward_m} m`;
  if(wasVisible&&!mapEditor.active){resumeAfterDetection=wasPlaying;await detect();}
  else if(wasPlaying&&!mapEditor.active)play(false);
}
function applySimulationParameters(){
  if(!$('initial-speed').reportValidity())return false;
  const speed=Number($('initial-speed').value);
  if(!Number.isFinite(speed)||speed<0)return false;
  if(speed!==simulationParameters.initial_speed_m_s){
    simulationParameters={initial_speed_m_s:speed};
    mapEditor.reset({clearViewer:mapEditor.active});
    notice('Simulation speed changed. Reopen the merged map to prepare its poses.');
  }
  $('simulation-parameter-status').textContent=`Active initial speed · ${speed} m/s`;
  return true;
}
let models=[],obstacles=[],obstacleSequence=0;
let socket=null, pendingFrame=null, ended=true, finished=false, paused=false, bagDuration=0, seekMax=0, scrubbing=false, seekTimer=null;
let bagReady=false,bagLoadVersion=0;
let activeDetectionId=null,detectionVisible=false,lastFrame=null,lastBasePoints=null;
function showDetectionButton(){
  renderObstacleIndicator(lastFrame);
  $('detect').textContent=detectionVisible?'Detect obstacles: On':'Detect obstacles: Off';
  $('detect').setAttribute('aria-pressed',String(detectionVisible));
  $('detect').classList.toggle('primary',detectionVisible);
}

function formatTime(seconds){const s=Math.max(0,Number(seconds)||0);return `${Math.floor(s/60)}:${(s%60).toFixed(1).padStart(4,'0')}`;}
function showSeek(seconds,moveHandle=true){const s=Math.max(0,Math.min(seekMax,Number(seconds)||0));if(moveHandle)ui.seek.value=String(s);ui.seekPosition.textContent=formatTime(s);}

function notice(message,kind=''){ui.notice.textContent=message;ui.notice.className='notice '+kind;}
async function api(url,options){const res=await fetch(url,options);const data=await res.json();if(!res.ok)throw Error(Array.isArray(data.detail)?data.detail.map(e=>e.msg).join('; '):data.detail||res.statusText);return data;}
function renderModelPicker(container,entries,selected,onChange){
  container.classList.add('model-picker');container.setAttribute('role','group');container.setAttribute('aria-label','Obstacle model');container.value=selected;container.replaceChildren();
  for(const entry of entries){
    const button=document.createElement('button');button.type='button';button.className='model-tile';button.dataset.modelId=entry.id;button.title=entry.name;button.setAttribute('aria-pressed',String(entry.id===selected));
    const image=document.createElement('img');image.src=entry.thumbnail;image.alt='';image.loading='lazy';
    const label=document.createElement('span');label.textContent=entry.name;button.append(image,label);
    button.onclick=()=>{container.value=entry.id;for(const tile of container.children)tile.setAttribute('aria-pressed',String(tile===button));onChange(entry.id);};container.append(button);
  }
}
async function loadModels(){
  models=await api('/api/models');mapEditor.setModels(models);
  if(!models.length){notice('No obstacle meshes are available.','error');return;}
  $('add-obstacle').disabled=bagDuration<=4;
}
function addObstacle(){
  if(!models.length){notice('Obstacle models are still loading.','error');return;}
  if(bagDuration<=4){notice('Choose a recording longer than 4 seconds to add obstacles.','error');return;}
  obstacles.push({id:`obstacle-${Date.now()}-${++obstacleSequence}`,model_id:models[0].id,encounter_seconds:Math.max(2,Math.floor(Math.min(5,bagDuration-2)*10)/10),offset:0,height_above_track_m:0});
  drawObstacles();
}
function drawObstacles(){
  const list=$('obstacle-list');list.replaceChildren();
  for(const [index,item] of obstacles.entries()){
    const card=document.createElement('div');card.className='obstacle-card';
    card.innerHTML=`<div class="obstacle-heading"><strong>Obstacle ${index+1}</strong><button type="button" class="remove">Remove</button></div>
      <div class="model"></div>
      <label>Train reaches object at · seconds<input class="encounter" type="number" min="2" step="0.1" required></label>
      <label>Distance from track center · m<input class="offset" type="number" min="-1" max="1" step="0.1"></label>
      <p class="hint">Negative = left · 0 = center · positive = right. Limit: 1 m each way.</p>
      <label>Height above track · m<input class="height" type="number" min="0" max="4" step="0.1"></label><p class="hint">0 puts the object's feet at rail level. Increase for a raised surface.</p>`;
    const model=card.querySelector('.model');
    renderModelPicker(model,models,item.model_id,id=>item.model_id=id);
    const time=card.querySelector('.encounter');time.value=item.encounter_seconds;time.max=Math.floor(Math.max(2,bagDuration-2)*10)/10;
    const offset=card.querySelector('.offset'),height=card.querySelector('.height');offset.value=item.offset??0;height.value=item.height_above_track_m;
    model.onchange=()=>item.model_id=model.value;
    time.oninput=()=>item.encounter_seconds=Number(time.value);
    offset.oninput=()=>item.offset=Number(offset.value);
    height.oninput=()=>item.height_above_track_m=Number(height.value);
    card.querySelector('.remove').onclick=()=>{obstacles=obstacles.filter(o=>o.id!==item.id);drawObstacles();};
    list.append(card);
  }
}
async function refreshBags(selectId){
  const bags=await api('/api/bags');ui.bag.replaceChildren();
  for(const bag of bags){const opt=document.createElement('option');opt.value=bag.id;opt.textContent=(bag.scope==='output'?'Export · ':'Source · ')+bag.name;ui.bag.append(opt);}
  if(selectId)ui.bag.value=selectId;
  if(!bags.length){notice('No bags found. Mount bag directories under /data.','error');return;}
  await loadBag();
}
async function loadBag(){
  const loadVersion=++bagLoadVersion,bagId=ui.bag.value;bagReady=false;bagDuration=0;
  for(const id of ['play','step','detect'])$(id).disabled=true;
  mapEditor.reset();
  clearTimeout(seekTimer);scrubbing=false;
  stopPlayback();
  detectionGeneration++;resumeAfterDetection=false;activeDetectionId=null;detectionVisible=false;lastFrame=null;lastBasePoints=null;
  $('detection-legend').hidden=true;showDetectionButton();$('detection-job').hidden=true;
  renderer.setPrediction(null);renderer.setPoints(new Float32Array(0));
  try{
    const info=await api('/api/info?bag_id='+encodeURIComponent(bagId));
    if(loadVersion!==bagLoadVersion)return;
    simulationParameters.initial_speed_m_s=info.initial_speed_m_s??0;$('initial-speed').value=simulationParameters.initial_speed_m_s;
    $('simulation-parameter-status').textContent=`Active initial speed · ${simulationParameters.initial_speed_m_s} m/s`;
    ui.topic.replaceChildren();
    for(const topic of info.topics.filter(t=>t.type==='sensor_msgs/msg/PointCloud2')){
      const opt=document.createElement('option');opt.value=topic.name;opt.textContent=topic.name;ui.topic.append(opt);
    }
    $('bag-info').textContent=`${info.duration_seconds.toFixed(1)} s · ${info.topics.length} topic(s)`;
    bagDuration=info.duration_seconds;seekMax=Math.max(0,bagDuration-.001);ui.seek.max=String(seekMax);ui.seekEnd.textContent=formatTime(bagDuration);showSeek(0);
    $('add-obstacle').disabled=!models.length||bagDuration<=4;
    drawObstacles();
    ui.time.textContent='0.0 s';bagReady=true;
    notice('Ready. Use the timeline to seek; export processes the whole bag.');
  }catch(e){if(loadVersion===bagLoadVersion)notice(e.message,'error');}
  finally{if(loadVersion===bagLoadVersion)for(const id of ['play','step','detect'])$(id).disabled=!bagReady;}
}
function stopPlayback(){if(socket){socket.close();socket=null;}ended=true;finished=false;paused=false;pendingFrame=null;}
function sendMessage(payload){
  if(!socket||ended)return;
  const target=socket,send=()=>{if(socket===target&&target.readyState===WebSocket.OPEN)target.send(JSON.stringify(payload));};
  if(target.readyState===WebSocket.CONNECTING)target.addEventListener('open',send,{once:true});else send();
}
function sendCommand(action){sendMessage({action});}
function renderObstacleIndicator(frame=null){
  const box=$('obstacle-indicator');
  let state='off',title='Detection off',distance='',detail='Turn on Detect obstacles to check the current scan.';
  if(detectionVisible){
    const prediction=frame?.rail_prediction;
    state='unknown';title='Waiting for detection';detail='No result for the current scan yet.';
    if(prediction?.status==='error'||prediction?.status==='unavailable'){
      title='Detection unavailable';detail=prediction.status==='error'?'RailGuard inference failed.':'No valid track prediction for this scan.';
    }else if(prediction?.status==='ready'){
      const checked=Number(prediction.checked_forward_m)||0;
      if(frame.detected&&Number.isFinite(frame.distance_m)){
        state='obstacle';title='Obstacle ahead';distance=`${frame.distance_m.toFixed(1)} m`;
        detail='Nearest retained point · distance along predicted track';
      }else if(!frame.detected){
        state='clear';title='No obstacle detected';detail=`Checked envelope · ${prediction.checked_min_forward_m??0}–${checked.toFixed(0)} m ahead`;
      }
    }
  }
  box.dataset.state=state;$('obstacle-indicator-title').textContent=title;
  $('obstacle-indicator-distance').textContent=distance;$('obstacle-indicator-detail').textContent=detail;
}
function renderStatus(frame,pointCount){
  renderObstacleIndicator(frame);
  const result=detectionVisible&&activeDetectionId?(frame.rail_prediction?.status==='unavailable'?' · Track unsupported — obstacle detection unavailable':frame.rail_prediction?.status==='error'?' · RailGuard inference failed':frame.detected?` · ${frame.candidate_points} filtered obstacle points · nearest ${frame.distance_m.toFixed(1)} m`:' · No obstacle clusters after filtering'):'';
  const simulation=frame.obstacle_points||frame.removed_points?` · ${frame.obstacle_points||0} obstacle points · ${frame.removed_points||0} hidden background points removed`:'';
  const modelInfo=detectionVisible&&frame.rail_prediction?.status==='ready'?` · RailGuard ${frame.rail_prediction.inference_ms} ms · checked ${frame.rail_prediction.checked_forward_m.toFixed(0)} m · vote horizon ${frame.rail_prediction.horizon_m.toFixed(0)} m${frame.rail_prediction.sigma_max_checked_m!=null?` · max lateral σ ${frame.rail_prediction.sigma_max_checked_m.toFixed(2)} m`:''}`:'';
  ui.status.textContent=`Frame ${frame.frame} · ${pointCount.toLocaleString()} preview points · ${frame.frame_id}${simulation}${result}${detectionVisible?` · ${frame.rejected_points||0} points rejected by distance/height/cluster filters · ${frame.excluded_rail_points||0} rail points excluded`:""}${modelInfo}`;
  ui.status.title=ui.status.textContent;
}
function renderFrame(points,frame){
  if(mapEditor.active)return;
  renderer.setPoints(points,frame.obstacle_points||0,frame.detected_points||0);
  renderer.setPrediction(detectionVisible?frame.rail_center:null,detectionVisible?frame.rail_points:null,frame.rail_prediction?.lookahead_limit_m,frame.rail_prediction?.checked_min_forward_m||0,frame.rail_prediction?.train_width_m??2.1,frame.rail_prediction?.train_height_m??3);
  renderer.setRejected(detectionVisible?frame.rejected_xyz:null);
  renderStatus(frame,points.length/3);
  ui.time.textContent=frame.time_seconds.toFixed(2)+' s';
  if(!scrubbing)showSeek(frame.time_seconds);
}
function play(startPaused=false){
  if(!bagReady)return;
  if(socket&&!ended){sendCommand('resume');paused=false;return;}
  if(!ui.bag.value||!ui.topic.value)return;
  if(finished){showSeek(0);finished=false;}
  const qs=new URLSearchParams({bag_id:ui.bag.value,topic:ui.topic.value,start:ui.seek.value,paused:startPaused?'1':'0'});
  if(activeDetectionId&&detectionVisible)qs.set('detection_id',activeDetectionId);
  socket=new WebSocket(`${location.protocol==='https:'?'wss':'ws'}://${location.host}/ws/playback?${qs}`);
  const thisSocket=socket;
  socket.binaryType='arraybuffer';ended=false;paused=startPaused;pendingFrame=null;
  if(startPaused)socket.onopen=()=>thisSocket.send(JSON.stringify({action:'step'}));
  ui.status.textContent='Opening bag…';
  socket.onmessage=event=>{
    if(socket!==thisSocket)return;
    if(typeof event.data==='string'){
      const message=JSON.parse(event.data);
      if(message.type==='frame'){pendingFrame=message;}
      else if(message.type==='end'){ended=true;finished=true;ui.status.textContent=`Finished · ${message.frames} frames`;}
      else if(message.type==='error'){ended=true;notice(message.message,'error');ui.status.textContent='Playback error';}
      else if(message.type==='overlay_error'){notice(message.message,'error');}
    }else if(pendingFrame){
      let points=new Float32Array(event.data),frame=pendingFrame;
      lastBasePoints=points.slice(0,points.length-(frame.detected_points||0)*3);
      if(!detectionVisible&&frame.detected_points){points=lastBasePoints;frame={...frame,detected_points:0,rail_center:null,detected:false};}
      lastFrame=frame;
      renderFrame(points,frame);
      pendingFrame=null;
    }
  };
  socket.onerror=()=>{if(socket===thisSocket)notice('Playback connection failed.','error');};
  socket.onclose=()=>{if(socket===thisSocket){ended=true;socket=null;}};
}
function pause(){if(socket&&!ended){sendCommand('pause');paused=true;}}
function step(){if(!socket||ended){play(true);return;}sendCommand('step');paused=true;}
function commitSeek(){
  scrubbing=true;
  const target=Math.max(0,Math.min(seekMax,Number(ui.seek.value)||0));
  const wasPlaying=Boolean(socket&&!ended&&!paused);
  clearTimeout(seekTimer);
  seekTimer=setTimeout(()=>{
    stopPlayback();showSeek(target);scrubbing=false;
    // A paused seek previews one frame. An active seek continues playback.
    play(!wasPlaying);
  },180);
}
async function simulate(){
  if(!applySimulationParameters())return;
  let request;
  if(mapEditor.active){try{request=mapEditor.request();}catch(error){notice(error.message,'error');return;}}else{
  if(bagDuration<=4){notice('Recording must be longer than 4 seconds to add obstacles.','error');return;}
  if(!obstacles.length){notice('Add at least one obstacle.','error');return;}
  for(const input of $('obstacle-list').querySelectorAll('input')){if(!input.reportValidity())return;}
  request={bag_id:ui.bag.value,topic:ui.topic.value,obstacles:obstacles.map(item=>({id:item.id,model_id:item.model_id,encounter_seconds:item.encounter_seconds,lateral_offset_m:item.offset??0,height_above_track_m:item.height_above_track_m}))};
  }
  if(!$('initial-speed').reportValidity())return;
  request.initial_speed_m_s=simulationParameters.initial_speed_m_s;
  $('simulate').disabled=true;$('job').hidden=false;$('progress-fill').style.width='0%';$('job-text').textContent='Starting export…';
  try{
    const job=await api('/api/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(request)});
    const timer=setInterval(async()=>{
      try{
        const current=await api('/api/jobs/'+job.id);
        $('progress-fill').style.width=Math.round(current.progress*100)+'%';$('job-text').textContent=current.message;
        if(current.state==='done'||current.state==='failed'){
          clearInterval(timer);$('simulate').disabled=false;
          if(current.state==='done'){if(mapEditor.active){$('placement-mode').value='timed';mapEditor.setActive(false);}await refreshBags(current.output_bag_id);notice('Export complete. The new bag is ready to play.'+(current.warnings?.length?' '+current.warnings.join(' '):''),current.warnings?.length?'':'success');}
          else notice(current.message,'error');
        }
      }catch(e){clearInterval(timer);$('simulate').disabled=false;notice(e.message,'error');}
    },600);
  }catch(e){$('simulate').disabled=false;notice(e.message,'error');}
}

async function refreshCurrentOverlay(){
  if(!detectionVisible||!activeDetectionId||!lastFrame||!lastBasePoints)return;
  const timestamp=lastFrame.timestamp_ns,jobId=activeDetectionId;
  try{
    const result=await api(`/api/detections/${jobId}/frame?timestamp_ns=${encodeURIComponent(timestamp)}`);
    if(!detectionVisible||activeDetectionId!==jobId||lastFrame?.timestamp_ns!==timestamp)return;
    const red=new Float32Array(result.detected_xyz.length*3);
    for(let i=0;i<result.detected_xyz.length;i++)red.set(result.detected_xyz[i],i*3);
    const points=new Float32Array(lastBasePoints.length+red.length);
    points.set(lastBasePoints);points.set(red,lastBasePoints.length);
    lastFrame={...lastFrame,...result,detected_points:red.length/3};
    renderFrame(points,lastFrame);
  }catch(e){notice(e.message,'error');}
}

async function setDetectionVisible(visible){
  if(!activeDetectionId)return;
  detectionVisible=visible;
  $('detection-legend').hidden=!visible;
  showDetectionButton();
  if(socket&&!ended){
    if(visible)sendMessage({action:'show_detection',detection_id:activeDetectionId});
    else sendCommand('hide_detection');
    if(!visible&&lastFrame&&lastBasePoints)renderFrame(lastBasePoints,{...lastFrame,detected_points:0});
    return;
  }
  if(visible)await refreshCurrentOverlay();
  else if(lastFrame&&lastBasePoints)renderFrame(lastBasePoints,{...lastFrame,detected_points:0});
}

async function detect(){
  if(!bagReady)return;
  if(activeDetectionId){await setDetectionVisible(!detectionVisible);return;}
  const bagId=ui.bag.value,topic=ui.topic.value,generation=detectionGeneration;
  if(!bagId||!topic)return;
  $('detect').disabled=true;$('detect').textContent='Loading model…';$('detection-job').hidden=false;$('detection-progress-fill').style.width='0%';$('detection-job-text').textContent='Loading RailGuard weights and refiner…';
  try{
    const job=await api('/api/detections',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({bag_id:bagId,topic,...detectionParameters})});
    if(generation!==detectionGeneration)return;
    const timer=setInterval(async()=>{
      try{
        if(generation!==detectionGeneration){clearInterval(timer);return;}
        const current=await api('/api/detections/'+job.id);
        if(generation!==detectionGeneration){clearInterval(timer);return;}
        $('detection-progress-fill').style.width=Math.round(current.progress*100)+'%';$('detection-job-text').textContent=current.message;
        if(current.state==='done'||current.state==='failed'){
          clearInterval(timer);$('detection-job').hidden=true;$('detect').disabled=false;showDetectionButton();
          if(current.state==='done'&&ui.bag.value===bagId&&ui.topic.value===topic){
            activeDetectionId=job.id;
            $('detection-job-text').textContent='RailGuard + refiner ready · checking each displayed scan';
            notice(`RailGuard ready on ${current.device}. Detection follows the displayed scan; obstacle detection checks 1.5–${detectionParameters.max_forward_m} m ahead; the full predicted centerline stays visible.`,'success');
            await setDetectionVisible(true);
            // A fresh page has no frame to decorate. Show one paused preview
            // without changing the position or state of an existing player.
            if(resumeAfterDetection){resumeAfterDetection=false;play(false);}
            else if(!lastFrame&&(!socket||ended))play(true);
          }else if(current.state==='failed')notice(current.message,'error');
        }
      }catch(e){clearInterval(timer);$('detection-job').hidden=true;$('detect').disabled=false;showDetectionButton();notice(e.message,'error');}
    },600);
  }catch(e){$('detection-job').hidden=true;$('detect').disabled=false;showDetectionButton();notice(e.message,'error');}
}

class CloudRenderer{
  constructor(canvas){
    this.canvas=canvas;this.gl=canvas.getContext('webgl2',{antialias:true});this.yaw=0;this.pitch=.25;this.distance=34;this.target=[0,-10,0];this.count=0;this.obstacleCount=0;this.detectedCount=0;this.frameRequested=false;
    if(!this.gl){notice('This browser needs WebGL 2 for the point-cloud view.','error');return;}
    const gl=this.gl;
    // With Z up and the default camera looking along -Y, screen-right is -X.
    const vs=`#version 300 es
      in vec3 aPosition,aColor;in vec2 aUV;out vec3 vColor,vWorld;out vec2 vUV;uniform bool uMesh;uniform vec3 uMeshOffset,uMeshPivot;uniform float uMeshScale,uMeshAngle;uniform float uYaw,uPitch,uDistance,uAspect,uSize,uPixelRatio;uniform vec3 uTarget;uniform vec2 uPixelOffset;
      void main(){
        vColor=aColor;vUV=aUV;vec3 local=(aPosition-uMeshPivot)*uMeshScale;
        float mc=cos(uMeshAngle),ms=sin(uMeshAngle);
        vWorld=uMesh?uMeshPivot+vec3(local.x*mc-local.y*ms,local.x*ms+local.y*mc,local.z)+uMeshOffset:aPosition;
        vec3 p=vWorld-uTarget;float c=cos(uYaw),s=sin(uYaw);
        float h=-(p.x*c+p.y*s);float d=-p.x*s+p.y*c;
        float cp=cos(uPitch),sp=sin(uPitch);float v=p.z*cp-d*sp;
        float depth=uDistance-d*cp-p.z*sp;
        // Homogeneous perspective lets WebGL clip points and lines behind the
        // camera. Clamping negative depth projected those points into the view.
        float nearPlane=0.1,farPlane=20000.0;
        float z=((farPlane+nearPlane)/(farPlane-nearPlane))*depth
                -2.0*farPlane*nearPlane/(farPlane-nearPlane);
        gl_Position=vec4(h*1.35/uAspect,v*1.35,z,depth);
        gl_Position.xy+=uPixelOffset*depth;
        gl_PointSize=clamp(uSize*22.0/max(nearPlane,depth),1.0,7.0)*uPixelRatio;
      }`;
    const fs=`#version 300 es
      precision highp float;uniform vec4 uColor;uniform bool uMesh,uCardboard;
      in vec3 vColor,vWorld;in vec2 vUV;out vec4 color;
      float noise(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453);}
      void main(){
        if(!uMesh){color=uColor;return;}
        vec3 base=vColor;
        if(uCardboard){
          float grain=noise(floor(vUV*900.));
          base=vec3(.63,.43,.23)*( .91+.16*grain+.025*sin(vUV.y*1400.));
          float tape=1.-smoothstep(.065,.07,abs(vUV.x-.5));
          base=mix(base,vec3(.76,.57,.32)*(.95+.05*grain),tape);
          float seam=1.-smoothstep(.001,.004,abs(vUV.x-.5));
          base*=1.-.28*seam;
          float edge=min(min(vUV.x,1.-vUV.x),min(vUV.y,1.-vUV.y));
          base*=.75+.25*smoothstep(0.,.018,edge);
          if(vUV.x>.69&&vUV.x<.93&&vUV.y>.2&&vUV.y<.42){
            base=vec3(.88,.86,.79);
            if(vUV.y<.34&&vUV.y>.23&&vUV.x>.71&&vUV.x<.91&&noise(vec2(floor(vUV.x*210.),0.))>.45)base=vec3(.12);
          }
        }
        vec3 normal=normalize(cross(dFdx(vWorld),dFdy(vWorld)));
        float light=.48+.52*abs(dot(normal,normalize(vec3(.4,-.6,1.))));
        color=vec4(base*light,1.);
      }`;
    const shader=(type,source)=>{const sh=gl.createShader(type);gl.shaderSource(sh,source);gl.compileShader(sh);if(!gl.getShaderParameter(sh,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(sh));return sh;};
    this.program=gl.createProgram();gl.attachShader(this.program,shader(gl.VERTEX_SHADER,vs));gl.attachShader(this.program,shader(gl.FRAGMENT_SHADER,fs));gl.linkProgram(this.program);
    this.buffer=gl.createBuffer();this.grid=gl.createBuffer();this.leftRail=gl.createBuffer();this.rightRail=gl.createBuffer();this.train=gl.createBuffer();this.rejectedBuffer=gl.createBuffer();this.rejectedCount=0;this.railPointsBuffer=gl.createBuffer();this.railPointsCount=0;this.centerlineBuffer=gl.createBuffer();this.centerlineCount=0;this.leftRailCount=0;this.rightRailCount=0;this.trainCount=0;
    const lines=[];for(let i=-20;i<=20;i+=2){lines.push(-20,i,-2,20,i,-2,i,-20,-2,i,20,-2);}
    lines.push(0,0,0,10,0,0,0,0,0,0,-20,0,0,0,0,0,0,5);
    this.gridCount=lines.length/3;gl.bindBuffer(gl.ARRAY_BUFFER,this.grid);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(lines),gl.STATIC_DRAW);
    this.position=gl.getAttribLocation(this.program,'aPosition');
    this.uniforms=Object.fromEntries(['uPixelRatio','uPixelOffset','uYaw','uPitch','uDistance','uAspect','uSize','uTarget','uColor','uMesh','uCardboard','uMeshOffset','uMeshPivot','uMeshScale','uMeshAngle'].map(x=>[x,gl.getUniformLocation(this.program,x)]));
    this.meshColorBuffer=gl.createBuffer();this.meshUVBuffer=gl.createBuffer();this.colorAttribute=gl.getAttribLocation(this.program,'aColor');this.uvAttribute=gl.getAttribLocation(this.program,'aUV');this.meshOffset=[0,0,0];this.meshScale=1;this.meshAngle=0;this.meshPivot=[0,0,0];this.controls=new MeshControls(this);
    this.meshBuffer=gl.createBuffer();this.meshCount=0;this.points=new Float32Array(0);
    let drag=null;
    canvas.addEventListener('contextmenu',e=>e.preventDefault());
    canvas.addEventListener('pointerdown',e=>{
      if(e.button!==0&&e.button!==2)return;
      e.preventDefault();
      canvas.focus({preventScroll:true});
      drag={startX:e.clientX,startY:e.clientY,moved:false,x:e.clientX,y:e.clientY,id:e.pointerId,mode:e.button===2?'pan':'orbit'};
      const rect=canvas.getBoundingClientRect();
      if(e.button===0&&this.mapMode&&this.controls.begin(e.clientX-rect.left,e.clientY-rect.top))drag.mode='object';
      canvas.style.cursor=drag.mode==='pan'?'move':'grabbing';
      canvas.setPointerCapture(e.pointerId);
    });
    canvas.addEventListener('pointermove',e=>{
      if(!drag||e.pointerId!==drag.id)return;
      if(Math.hypot(e.clientX-drag.startX,e.clientY-drag.startY)>4)drag.moved=true;
      if(!drag.moved)return;
      if(drag.mode==='object'){const rect=canvas.getBoundingClientRect();this.controls.move(e.clientX-rect.left,e.clientY-rect.top);return;}
      const dx=e.clientX-drag.x,dy=e.clientY-drag.y;
      if(drag.mode==='orbit'){
        // Preserve the original drag feel after correcting the mirrored view.
        this.yaw-=dx*.006;
        this.pitch=Math.max(-1.2,Math.min(1.2,this.pitch+dy*.006));
      }else{
        // Move the target in the camera's screen plane, so the scene follows the drag.
        const scale=2*this.distance/(Math.max(1,canvas.clientHeight)*1.35);
        const c=Math.cos(this.yaw),s=Math.sin(this.yaw),sp=Math.sin(this.pitch),cp=Math.cos(this.pitch);
        this.target[0]+=scale*(dx*c+dy*s*sp);
        this.target[1]+=scale*(dx*s-dy*c*sp);
        this.target[2]+=scale*dy*cp;
      }
      drag.x=e.clientX;drag.y=e.clientY;this.requestDraw();
    });
    const endDrag=e=>{if(drag&&e.pointerId===drag.id){if(drag.mode==='object')this.controls.end(e.type==='pointercancel');const picked=drag.mode==='orbit'&&!drag.moved&&e.type==='pointerup';drag=null;canvas.style.cursor=this.onPick?'crosshair':'grab';if(picked&&this.onPick){const rect=canvas.getBoundingClientRect();this.onPick(e.clientX-rect.left,e.clientY-rect.top);}if(canvas.hasPointerCapture(e.pointerId))canvas.releasePointerCapture(e.pointerId);}};
    canvas.addEventListener('pointerup',endDrag);canvas.addEventListener('pointercancel',endDrag);
    canvas.addEventListener('wheel',e=>{e.preventDefault();this.distance=Math.max(3,Math.min(this.maxDistance||150,this.distance*Math.exp(e.deltaY*.001)));this.requestDraw();},{passive:false});
    this.installKeyboardNavigation();
    new ResizeObserver(()=>this.requestDraw()).observe(canvas);window.addEventListener('resize',()=>this.requestDraw());this.requestDraw();
  }
  installKeyboardNavigation(){
    const canvas=this.canvas,keys=new Set();
    canvas.tabIndex=0;
    canvas.setAttribute('aria-label','3D point cloud. W/S move forward/backward, A/D move left/right. Hold Shift to move faster.');
    let animation=null,previous=null,fast=false;
    const stop=()=>{keys.clear();fast=false;previous=null;if(animation!==null)cancelAnimationFrame(animation);animation=null;};
    const tick=now=>{
      animation=null;
      if(document.activeElement!==canvas||document.hidden||!keys.size){stop();return;}
      if(previous!==null){
        this.moveCamera(Number(keys.has('KeyW'))-Number(keys.has('KeyS')),
                        Number(keys.has('KeyD'))-Number(keys.has('KeyA')),
                        Math.min((now-previous)/1000,.05),fast);
      }
      previous=now;animation=requestAnimationFrame(tick);
    };
    canvas.addEventListener('keydown',event=>{
      if(event.ctrlKey||event.metaKey||event.altKey){stop();return;}
      fast=event.shiftKey;
      if(!['KeyW','KeyA','KeyS','KeyD'].includes(event.code))return;
      event.preventDefault();keys.add(event.code);
      if(animation===null){previous=null;animation=requestAnimationFrame(tick);}
    });
    window.addEventListener('keyup',event=>{keys.delete(event.code);fast=event.shiftKey;if(!keys.size)stop();});
    canvas.addEventListener('blur',stop);window.addEventListener('blur',stop);
    document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
  }
  moveCamera(forward,right,seconds,fast=false){
    const norm=Math.hypot(forward,right);if(!norm)return;
    const step=seconds*Math.max(1,Math.min(100,this.distance*.4))*(fast?3:1)/norm;
    const c=Math.cos(this.yaw),s=Math.sin(this.yaw),cp=Math.cos(this.pitch),sp=Math.sin(this.pitch);
    // Translate the orbit target and camera together in camera-relative axes.
    // Screen-right is -X at the default yaw; forward follows the view's pitch.
    this.target[0]+=step*(forward*s*cp-right*c);
    this.target[1]+=step*(-forward*c*cp-right*s);
    this.target[2]-=step*forward*sp;
    this.requestDraw();
  }
  requestDraw(){if(this.frameRequested)return;this.frameRequested=true;requestAnimationFrame(()=>{this.frameRequested=false;this.draw();});}
  setPoints(points,obstacleCount=0,detectedCount=0){this.points=points;if(!this.gl)return;const gl=this.gl;this.count=points.length/3;this.obstacleCount=obstacleCount;this.detectedCount=detectedCount;gl.bindBuffer(gl.ARRAY_BUFFER,this.buffer);gl.bufferData(gl.ARRAY_BUFFER,points,gl.DYNAMIC_DRAW);this.requestDraw();}
  setRejected(points){
    if(!this.gl)return;const values=new Float32Array((points||[]).flat());this.rejectedCount=values.length/3;
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER,this.rejectedBuffer);this.gl.bufferData(this.gl.ARRAY_BUFFER,values,this.gl.DYNAMIC_DRAW);this.requestDraw();
  }
  transformMeshPoint(point){
    const p=point.map((v,i)=>(v-this.meshPivot[i])*this.meshScale),c=Math.cos(this.meshAngle),s=Math.sin(this.meshAngle);
    return [p[0]*c-p[1]*s,p[0]*s+p[1]*c,p[2]].map((v,i)=>v+this.meshPivot[i]+this.meshOffset[i]);
  }
  uploadMesh(mesh,layer){
    const data=[],colors=[],uvs=[];
    if(mesh)for(const [f,face] of mesh.triangles.entries()){
      const local=face.map(i=>mesh.local_vertices?.[i]||[0,0,0]);
      const span=[0,1,2].map(a=>Math.max(...local.map(p=>p[a]))-Math.min(...local.map(p=>p[a])));
      const flat=span.indexOf(Math.min(...span)),axes=[0,1,2].filter(a=>a!==flat);
      for(const [j,index] of face.entries()){data.push(...mesh.vertices[index]);colors.push(...(mesh.colors?.[f]||[.7,.5,.3]));uvs.push(local[j][axes[0]],local[j][axes[1]]);}
    }
    layer.count=data.length/3;
    for(const [buffer,values] of [[layer.position,data],[layer.color,colors],[layer.uv,uvs]]){
      this.gl.bindBuffer(this.gl.ARRAY_BUFFER,buffer);this.gl.bufferData(this.gl.ARRAY_BUFFER,new Float32Array(values),this.gl.DYNAMIC_DRAW);
    }
  }
  setMesh(mesh){
    if(!this.gl)return;this.mesh=mesh;this.meshOffset=[0,0,0];this.meshScale=1;this.meshAngle=0;
    const layer={position:this.meshBuffer,color:this.meshColorBuffer,uv:this.meshUVBuffer};this.uploadMesh(mesh,layer);this.meshCount=layer.count;
    this.controls.update();this.requestDraw();
  }
  setFixedMeshes(meshes){
    const gl=this.gl;if(!gl)return;
    for(const layer of this.fixedMeshes||[])for(const key of ['position','color','uv'])gl.deleteBuffer(layer[key]);
    this.fixedMeshes=meshes.map(mesh=>{const layer={mesh,position:gl.createBuffer(),color:gl.createBuffer(),uv:gl.createBuffer()};this.uploadMesh(mesh,layer);return layer;});this.requestDraw();
  }
  setMapLines(center,trajectory){
    const gl=this.gl;this.trainCount=0;this.centerlineCount=0;this.railPointsCount=0;this.rejectedCount=0;
    for(const [points,buffer,count] of [[center,this.leftRail,'leftRailCount'],[trajectory,this.rightRail,'rightRailCount']]){
      const lines=[];for(let i=1;i<points.length;i++)lines.push(...points[i-1],...points[i]);
      gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(lines),gl.DYNAMIC_DRAW);this[count]=lines.length/3;
    }this.requestDraw();
  }
  projectPoint(point){
    const [x,y,z]=point.map((v,i)=>v-this.target[i]),c=Math.cos(this.yaw),s=Math.sin(this.yaw),cp=Math.cos(this.pitch),sp=Math.sin(this.pitch);
    const horizontal=-(x*c+y*s),d=-x*s+y*c,vertical=z*cp-d*sp,depth=this.distance-d*cp-z*sp;
    const w=this.canvas.clientWidth,h=this.canvas.clientHeight;
    return [(horizontal*1.35/(w/h)/depth+1)*w/2,(1-vertical*1.35/depth)*h/2,depth];
  }
  pickPoint(x,y){
    let selected=null,nearest=Infinity;
    for(let i=0;i<this.points.length;i+=3){
      const point=[this.points[i],this.points[i+1],this.points[i+2]],p=this.projectPoint(point);
      if(p[2]<.1||p[2]>20000||p[0]<0||p[0]>this.canvas.clientWidth||p[1]<0||p[1]>this.canvas.clientHeight)continue;
      if((p[0]-x)**2+(p[1]-y)**2<=36&&p[2]<nearest){selected=point;nearest=p[2];}
    }return selected;
  }
  setPrediction(center,railPoints=null,maxForward=null,minForward=0,width=2.1,height=3){
    if(!this.gl)return;
    const gl=this.gl,left=[],right=[],train=[],middle=[];
    const rails=new Float32Array((railPoints||[]).flat());
    gl.bindBuffer(gl.ARRAY_BUFFER,this.railPointsBuffer);gl.bufferData(gl.ARRAY_BUFFER,rails,gl.DYNAMIC_DRAW);this.railPointsCount=rails.length/3;
    if(center&&center.length>=2){
      const corners=[];
      for(let i=0;i<center.length;i++){
        const p=center[i],before=center[Math.max(0,i-1)],after=center[Math.min(center.length-1,i+1)];
        const dx=after[0]-before[0],dy=after[1]-before[1],norm=Math.hypot(dx,dy)||1,nx=dy/norm,ny=-dx/norm;
        const l=[p[0]+nx*.75,p[1]+ny*.75,p[2]],r=[p[0]-nx*.75,p[1]-ny*.75,p[2]];
        const lb=[p[0]+nx*width/2,p[1]+ny*width/2,p[2]],rb=[p[0]-nx*width/2,p[1]-ny*width/2,p[2]];
        const lt=[lb[0],lb[1],lb[2]+height],rt=[rb[0],rb[1],rb[2]+height];
        corners.push({l,r,lb,rb,lt,rt});
        if(i){middle.push(...center[i-1],...p);const q=corners[i-1];for(const key of ['lb','rb','lt','rt']){
          const a=q[key],b=corners[i][key];
          const da=-a[1],db=-b[1],far=maxForward??Infinity;
          if(db>=minForward&&da<=far){
            const lo=da<minForward?(minForward-da)/(db-da):0,hi=db>far?(far-da)/(db-da):1;
            train.push(...a.map((v,j)=>v+lo*(b[j]-v)),...a.map((v,j)=>v+hi*(b[j]-v)));
          }
        }}
        if((-p[1]>=minForward)&&(maxForward==null||-p[1]<=maxForward)&&(i%80===0||i===center.length-1)){train.push(...lb,...lt,...lt,...rt,...rt,...rb,...rb,...lb);}
      }
    }
    for(const [buffer,data,count] of [[this.leftRail,left,'leftRailCount'],[this.rightRail,right,'rightRailCount'],[this.train,train,'trainCount'],[this.centerlineBuffer,middle,'centerlineCount']]){
      gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(data),gl.DYNAMIC_DRAW);this[count]=data.length/3;
    }
    this.requestDraw();
  }
  draw(){
    if(!this.gl)return;const gl=this.gl,canvas=this.canvas,w=canvas.clientWidth,h=canvas.clientHeight;
    if(!w||!h)return;const scale=window.devicePixelRatio||1,width=Math.floor(w*scale),height=Math.floor(h*scale);
    if(canvas.width!==width||canvas.height!==height){canvas.width=width;canvas.height=height;}
    gl.viewport(0,0,canvas.width,canvas.height);
    gl.clearColor(.106,.110,.125,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);if(this.mapMode)gl.enable(gl.DEPTH_TEST);else gl.disable(gl.DEPTH_TEST);gl.useProgram(this.program);
    gl.uniform1f(this.uniforms.uPixelRatio,scale);gl.uniform1f(this.uniforms.uYaw,this.yaw);gl.uniform1f(this.uniforms.uPitch,this.pitch);gl.uniform1f(this.uniforms.uDistance,this.distance);gl.uniform1f(this.uniforms.uAspect,w/h);gl.uniform3fv(this.uniforms.uTarget,this.target);
    gl.uniform1i(this.uniforms.uMesh,0);gl.disableVertexAttribArray(this.colorAttribute);gl.disableVertexAttribArray(this.uvAttribute);
    gl.uniform2f(this.uniforms.uPixelOffset,0,0);gl.lineWidth(1);
    gl.enableVertexAttribArray(this.position);
    gl.bindBuffer(gl.ARRAY_BUFFER,this.grid);gl.vertexAttribPointer(this.position,3,gl.FLOAT,false,0,0);gl.uniform1f(this.uniforms.uSize,1);gl.uniform4f(this.uniforms.uColor,.23,.24,.28,.6);gl.drawArrays(gl.LINES,0,this.gridCount);
    if(this.count){gl.bindBuffer(gl.ARRAY_BUFFER,this.buffer);gl.vertexAttribPointer(this.position,3,gl.FLOAT,false,0,0);gl.uniform1f(this.uniforms.uSize,3);gl.uniform4f(this.uniforms.uColor,.68,.70,.75,1);gl.drawArrays(gl.POINTS,0,this.count-this.obstacleCount-this.detectedCount);if(this.obstacleCount){gl.uniform1f(this.uniforms.uSize,4);gl.uniform4f(this.uniforms.uColor,1,.67,.35,1);gl.drawArrays(gl.POINTS,this.count-this.obstacleCount-this.detectedCount,this.obstacleCount);}}
    for(const [buffer,count,color] of [[this.leftRail,this.leftRailCount,[.34,.65,1,1]],[this.rightRail,this.rightRailCount,[.34,.65,1,1]],[this.train,this.trainCount,[.55,.46,.27,1]],[this.centerlineBuffer,this.centerlineCount,[.2,1,.3,1]]]){
      if(count){gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.vertexAttribPointer(this.position,3,gl.FLOAT,false,0,0);gl.uniform4fv(this.uniforms.uColor,color);
        // WebGL implementations often clamp lineWidth to one pixel. Offset
        // passes keep the centerline three CSS pixels wide on those devices.
        if(buffer===this.centerlineBuffer){
          for(let x=-1;x<=1;x++)for(let y=-1;y<=1;y++){
            gl.uniform2f(this.uniforms.uPixelOffset,2*x/w,2*y/h);gl.drawArrays(gl.LINES,0,count);
          }
          gl.uniform2f(this.uniforms.uPixelOffset,0,0);
        }else gl.drawArrays(gl.LINES,0,count);}
    }
    if(this.mapMode){
      const layers=[...(this.fixedMeshes||[])];
      if(this.meshCount)layers.push({mesh:this.mesh,count:this.meshCount,position:this.meshBuffer,color:this.meshColorBuffer,uv:this.meshUVBuffer,active:true});
      for(const layer of layers){
        gl.bindBuffer(gl.ARRAY_BUFFER,layer.position);gl.vertexAttribPointer(this.position,3,gl.FLOAT,false,0,0);
        gl.uniform1i(this.uniforms.uMesh,1);gl.uniform1i(this.uniforms.uCardboard,layer.mesh.cardboard?1:0);
        gl.uniform3fv(this.uniforms.uMeshOffset,layer.active?this.meshOffset:[0,0,0]);gl.uniform3fv(this.uniforms.uMeshPivot,layer.active?this.meshPivot:[0,0,0]);
        gl.uniform1f(this.uniforms.uMeshScale,layer.active?this.meshScale:1);gl.uniform1f(this.uniforms.uMeshAngle,layer.active?this.meshAngle:0);
        gl.bindBuffer(gl.ARRAY_BUFFER,layer.color);gl.enableVertexAttribArray(this.colorAttribute);gl.vertexAttribPointer(this.colorAttribute,3,gl.FLOAT,false,0,0);
        gl.bindBuffer(gl.ARRAY_BUFFER,layer.uv);gl.enableVertexAttribArray(this.uvAttribute);gl.vertexAttribPointer(this.uvAttribute,2,gl.FLOAT,false,0,0);
        gl.drawArrays(gl.TRIANGLES,0,layer.count);
      }
      gl.uniform1i(this.uniforms.uMesh,0);gl.disableVertexAttribArray(this.colorAttribute);gl.disableVertexAttribArray(this.uvAttribute);
    }
    if(this.mapMode)this.controls.draw();
    if(this.detectedCount){gl.bindBuffer(gl.ARRAY_BUFFER,this.buffer);gl.vertexAttribPointer(this.position,3,gl.FLOAT,false,0,0);gl.uniform1f(this.uniforms.uSize,5);gl.uniform4f(this.uniforms.uColor,1,.2,.18,1);gl.drawArrays(gl.POINTS,this.count-this.detectedCount,this.detectedCount);}
    if(!this.mapMode&&detectionVisible&&$('show-rejected').checked&&this.rejectedCount){
      gl.bindBuffer(gl.ARRAY_BUFFER,this.rejectedBuffer);gl.vertexAttribPointer(this.position,3,gl.FLOAT,false,0,0);
      gl.uniform1f(this.uniforms.uSize,5);gl.uniform4f(this.uniforms.uColor,.76,.54,1,1);gl.drawArrays(gl.POINTS,0,this.rejectedCount);
    }
    if(this.railPointsCount){
      gl.bindBuffer(gl.ARRAY_BUFFER,this.railPointsBuffer);gl.vertexAttribPointer(this.position,3,gl.FLOAT,false,0,0);
      gl.uniform1f(this.uniforms.uSize,5);gl.uniform4f(this.uniforms.uColor,.34,.65,1,1);gl.drawArrays(gl.POINTS,0,this.railPointsCount);
    }
  }
}
const renderer=new CloudRenderer($('cloud'));
const mapEditor=new MapEditor(renderer);
$('show-rejected').addEventListener('change',()=>renderer.requestDraw());
ui.bag.addEventListener('change',loadBag);
$('add-obstacle').addEventListener('click',addObstacle);
ui.topic.addEventListener('change',()=>{detectionGeneration++;resumeAfterDetection=false;mapEditor.reset();stopPlayback();activeDetectionId=null;detectionVisible=false;lastFrame=null;lastBasePoints=null;$('detection-legend').hidden=true;$('detection-job').hidden=true;showDetectionButton();renderer.setPrediction(null);renderer.setPoints(new Float32Array(0));});
ui.seek.addEventListener('input',()=>{scrubbing=true;showSeek(ui.seek.value,false);});
ui.seek.addEventListener('change',commitSeek);
$('play').addEventListener('click',()=>play());$('pause').addEventListener('click',pause);$('step').addEventListener('click',step);$('simulate').addEventListener('click',simulate);$('detect').addEventListener('click',detect);
showDetectionButton();
// Model controls must not wait for bag metadata or playback initialization.
const modelReady=loadModels().catch(e=>notice('Cannot load obstacle models: '+e.message,'error'));
const bagsReady=refreshBags().catch(e=>notice(e.message,'error'));
Promise.all([modelReady,bagsReady]).then(()=>{if(models.length&&!obstacles.length)addObstacle();});

$('apply-detection-parameters').onclick=applyDetectionParameters;
$('apply-simulation-parameters').onclick=applySimulationParameters;
for(const id of ['train-width','train-height','detection-distance']){
  $(id).addEventListener('change',applyDetectionParameters);
}
$('detection-distance').oninput=()=>{$('detection-distance-value').textContent=$('detection-distance').value+' m';};

$('fullscreen').onclick=async()=>{
  try{
    if(document.fullscreenElement===$('viewer-panel'))await document.exitFullscreen();
    else await $('viewer-panel').requestFullscreen();
  }catch(error){notice('Cannot open fullscreen: '+error.message,'error');}
};
document.addEventListener('fullscreenchange',()=>{
  const active=document.fullscreenElement===$('viewer-panel');
  $('fullscreen').textContent=active?'Exit fullscreen':'Fullscreen';
  $('fullscreen').setAttribute('aria-pressed',String(active));
  renderer.requestDraw();
});
