const $ = id => document.getElementById(id);
const ui = {bag:$('bag'),topic:$('topic'),seek:$('seek'),seekPosition:$('seek-position'),seekEnd:$('seek-end'),notice:$('notice'),status:$('frame-status'),time:$('time-display')};
let detectionParameters={initial_speed_m_s:0,train_width_m:2.1,train_height_m:3,max_forward_m:150};
let simulationParameters={initial_speed_m_s:0};
let topicModels={};
function updateAutomaticModel(){
  const model=topicModels[ui.topic.value];if(!model)return;
  $('detection-model').textContent=`Automatic model · ${model.model_id}`;
  $('detection-distance').max=String(model.model_range_m);
  detectionParameters.max_forward_m=Math.min(detectionParameters.max_forward_m,model.model_range_m);
  $('detection-distance').value=String(detectionParameters.max_forward_m);
  $('detection-distance-value').textContent=detectionParameters.max_forward_m+' m';
  $('detection-parameter-status').textContent=`Active envelope · ${detectionParameters.train_width_m} × ${detectionParameters.train_height_m} m · detection to ${detectionParameters.max_forward_m} m`;
}
let detectionGeneration=0,resumeAfterDetection=false;
async function applyDetectionParameters(){
  for(const input of $('detection-parameters').querySelectorAll('input'))if(!input.reportValidity())return;
  const next={initial_speed_m_s:Number($('detection-speed').value),train_width_m:Number($('train-width').value),train_height_m:Number($('train-height').value),max_forward_m:Number($('detection-distance').value)};
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
  renderer.setCandidates(null);renderer.setPrediction(null);renderer.setPoints(new Float32Array(0));
  try{
    const info=await api('/api/info?bag_id='+encodeURIComponent(bagId));
    if(loadVersion!==bagLoadVersion)return;
    detectionParameters.initial_speed_m_s=info.initial_speed_m_s??0;$('detection-speed').value=detectionParameters.initial_speed_m_s;
    simulationParameters.initial_speed_m_s=info.initial_speed_m_s??0;$('initial-speed').value=simulationParameters.initial_speed_m_s;
    $('simulation-parameter-status').textContent=`Active initial speed · ${simulationParameters.initial_speed_m_s} m/s`;
    topicModels=Object.fromEntries(info.topics.map(topic=>[topic.name,topic]));
    ui.topic.replaceChildren();
    for(const topic of info.topics.filter(t=>t.type==='sensor_msgs/msg/PointCloud2')){
      const opt=document.createElement('option');opt.value=topic.name;opt.textContent=topic.name;ui.topic.append(opt);
    }
    updateAutomaticModel();
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
  const result=detectionVisible&&activeDetectionId?(frame.rail_prediction?.status==='unavailable'?' · Track unsupported — obstacle detection unavailable':frame.rail_prediction?.status==='error'?' · RailGuard inference failed':frame.detected?` · ${frame.candidate_points} confirmed obstacle points · nearest ${frame.distance_m.toFixed(1)} m`:` · ${frame.raw_candidate_points||0} candidate points · awaiting confirmation or clear`):'';
  const simulation=frame.obstacle_points||frame.removed_points?` · ${frame.obstacle_points||0} obstacle points · ${frame.removed_points||0} hidden background points removed`:'';
  const modelInfo=detectionVisible&&frame.rail_prediction?.status==='ready'?` · RailGuard ${frame.rail_prediction.model_range_m?`${frame.rail_prediction.model_range_m} m · `:""}${frame.rail_prediction.inference_ms} ms · checked ${frame.rail_prediction.checked_forward_m.toFixed(0)} m · vote horizon ${frame.rail_prediction.horizon_m.toFixed(0)} m${frame.rail_prediction.sigma_max_checked_m!=null?` · max lateral σ ${frame.rail_prediction.sigma_max_checked_m.toFixed(2)} m`:''}`:'';
  ui.status.textContent=`Frame ${frame.frame} · ${pointCount.toLocaleString()} preview points · ${frame.frame_id}${simulation}${result}${detectionVisible?` · ${frame.rejected_points||0} points rejected by distance/height/cluster filters · ${frame.excluded_rail_points||0} rail points excluded`:""}${modelInfo}`;
  ui.status.title=ui.status.textContent;
}
function renderFrame(points,frame){
  if(mapEditor.active)return;
  renderer.setPoints(points,frame.obstacle_points||0,frame.detected_points||0);
  renderer.setPrediction(detectionVisible?frame.rail_center:null,detectionVisible?frame.rail_points:null,frame.rail_prediction?.lookahead_limit_m,frame.rail_prediction?.checked_min_forward_m||0,frame.rail_prediction?.train_width_m??2.1,frame.rail_prediction?.train_height_m??3,frame.rail_prediction);
  renderer.setCandidates(detectionVisible?frame.unconfirmed_xyz:null);
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
            notice(`RailGuard${current.model_range_m?` ${current.model_range_m} m`:""} ready on ${current.device}. Detection follows the displayed scan; obstacle detection checks 1.5–${detectionParameters.max_forward_m} m ahead; the full predicted centerline stays visible.`,'success');
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

const renderer=new CloudRenderer($('cloud'));
const mapEditor=new MapEditor(renderer);
$('show-rejected').addEventListener('change',()=>renderer.requestDraw());
ui.bag.addEventListener('change',loadBag);
$('add-obstacle').addEventListener('click',addObstacle);
ui.topic.addEventListener('change',()=>{updateAutomaticModel();detectionGeneration++;resumeAfterDetection=false;mapEditor.reset();stopPlayback();activeDetectionId=null;detectionVisible=false;lastFrame=null;lastBasePoints=null;$('detection-legend').hidden=true;$('detection-job').hidden=true;showDetectionButton();renderer.setCandidates(null);renderer.setPrediction(null);renderer.setPoints(new Float32Array(0));});
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
for(const id of ['train-width','train-height','detection-distance','detection-speed']){
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
