/* Editable world-space objects, independent of the encounter-time list. */
class MapEditor {
  constructor(renderer) {
    this.renderer=renderer;this.active=false;this.generation=0;this.previewVersion=0;
    this.mapId=null;this.placement=null;this.placing=false;this.items=[];this.selectedEntry=null;this.sequence=0;this.confirming=false;
    renderer.controls.onStart=()=>{this.previewVersion++;clearTimeout(this.previewTimer);this.moveStart=null;this.scaleStart=this.placement?.scale??1;this.yawStart=this.placement?.yaw_deg??0;renderer.meshPivot=this.placement?.position.slice()||[0,0,0];};
    renderer.controls.onRotate=angle=>{
      if(!this.placement)return 0;
      this.placement.yaw_deg=this.yawStart+angle*180/Math.PI;
      $('map-yaw').value=this.placement.yaw_deg;
      return angle;
    };
    renderer.controls.onScale=factor=>{
      if(!this.placement)return 1;
      this.placement.scale=Math.max(.001,this.scaleStart*factor);
      $('map-scale').value=this.placement.scale;
      return this.placement.scale/this.scaleStart;
    };
    renderer.controls.onMove=delta=>{
      if(!this.placement)return;
      if(!this.moveStart)this.moveStart=this.placement.position.slice();
      this.placement.position=this.moveStart.map((v,i)=>v+delta[i]);
      ['x','y','z'].forEach((name,i)=>$('map-'+name).value=this.placement.position[i]);
    };
    renderer.controls.onEnd=()=>{this.moveStart=null;this.preview();};
    $('placement-mode').onchange=()=>this.setActive($('placement-mode').value==='map');
    $('prepare-map').onclick=()=>this.prepare();
    $('place-map').onclick=async()=>{
      if(this.confirming)return;
      if(this.placing){this.placing=false;this.updatePicking();return;}
      if(this.placement&&!await this.confirm())return;
      if(this.items.length>=50){notice('Maximum 50 map objects per export.');return;}
      this.placing=true;this.updatePicking();
    };
    $('confirm-map').onclick=()=>this.confirm();
    $('remove-map').onclick=()=>this.remove();
    $('map-model').onchange=()=>{this.anchorHint();if(this.placement){this.placement.model_id=$('map-model').value;this.preview();}};
    for(const name of ['x','y','z','yaw','scale'])$('map-'+name).oninput=()=>this.edit();
  }
  setModels(models) {
    renderModelPicker($('map-model'),models,models[0]?.id,id=>$('map-model').dispatchEvent(new Event('change')));
    this.anchorHint();
  }
  anchorHint(){ $('map-anchor').textContent=$('map-model').value==='wire_hanging'?'Click the ceiling to attach the top of the wire.':'Click a map point to place the model’s base.'; }
  setActive(active) {
    stopPlayback();this.active=active;this.renderer.mapMode=active;
    $('timed-controls').hidden=active;$('map-controls').hidden=!active;
    document.querySelector('.timeline').hidden=active;document.querySelector('.transport').hidden=active;
    $('detect').hidden=active;$('detection-legend').hidden=true;$('obstacle-indicator').hidden=active;renderObstacleIndicator();
    document.querySelector('.view-label').textContent=active?'Merged world map · metres':'LiDAR frame · metres';
    this.renderer.setPrediction(null);this.renderer.setMesh(null);this.renderer.setFixedMeshes([]);
    if(active){
      this.renderer.maxDistance=5000;
      if(this.points){this.show();this.renderer.setMesh(this.selectedEntry?.mesh||null);this.renderer.controls.select(!!this.selectedEntry);this.refreshLayers();this.preview();}else{this.renderer.setPoints(new Float32Array(0));this.prepare();}
    }else{
      this.renderer.maxDistance=150;this.renderer.target=[0,-10,0];this.renderer.distance=34;this.renderer.pitch=.25;
      if(lastFrame&&lastBasePoints)renderFrame(lastBasePoints,lastFrame);else this.renderer.setPoints(new Float32Array(0));
    }
    this.updatePicking();this.renderer.requestDraw();
  }
  reset({clearViewer=true}={}) {
    this.generation++;this.previewVersion++;clearTimeout(this.previewTimer);
    this.mapId=null;this.points=null;this.geometry=null;this.clearItems();
    $('prepare-map').disabled=false;$('place-map').disabled=true;$('map-job').hidden=true;
    if(clearViewer){this.renderer.setPoints(new Float32Array(0));this.renderer.setPrediction(null);}
  }
  clearItems(){
    this.items=[];this.confirming=false;this.selectEntry(null);this.renderer.setFixedMeshes([]);
  }
  refreshLayers(){
    this.renderer.setFixedMeshes(this.items.filter(item=>item!==this.selectedEntry&&item.mesh).map(item=>item.mesh));
  }
  drawItems(){
    $('map-objects').replaceChildren(...this.items.map((entry,index)=>{
      const button=document.createElement('button');button.type='button';button.textContent=`${index+1}. ${models.find(m=>m.id===entry.spec.model_id)?.name||entry.spec.model_id}`;
      button.setAttribute('aria-pressed',String(entry===this.selectedEntry));button.onclick=()=>this.selectEntry(entry);return button;
    }));
    $('confirm-map').disabled=!this.placement||this.confirming;
    $('remove-map').disabled=!this.placement;
  }
  selectEntry(entry){
    this.selectedEntry=entry;this.placement=entry?.spec||null;this.placing=false;
    this.renderer.setMesh(entry?.mesh||null);this.renderer.controls.select(!!entry);this.refreshLayers();
    $('map-transform').disabled=!entry;
    if(entry){
      const spec=entry.spec;$('map-model').value=spec.model_id;
      for(const button of $('map-model').children)button.setAttribute('aria-pressed',String(button.dataset.modelId===spec.model_id));
      ['x','y','z'].forEach((name,i)=>$('map-'+name).value=spec.position[i]);$('map-yaw').value=spec.yaw_deg;$('map-scale').value=spec.scale;
      this.anchorHint();if(!entry.mesh)this.preview();
    }
    this.drawItems();this.updatePicking();
  }
  async confirm(){
    if(!this.selectedEntry)return true;
    for(const input of $('map-transform').querySelectorAll('input'))if(!input.reportValidity())return false;
    const entry=this.selectedEntry,generation=this.generation;this.confirming=true;this.drawItems();
    try{
      await this.renderEntry(entry);
      if(generation!==this.generation||!this.items.includes(entry))return false;
      if(entry.meshKey!==JSON.stringify(entry.spec)){notice('Object changed while confirming. Press Confirm placement again.');return false;}
      if(this.selectedEntry===entry)this.selectEntry(null);
      notice('Placement confirmed. Select a model and press Add obstacle to place another.');return true;
    }catch(error){notice(error.message,'error');return false;}
    finally{this.confirming=false;this.drawItems();}
  }
  remove(){
    if(this.selectedEntry)this.items=this.items.filter(item=>item!==this.selectedEntry);
    this.selectEntry(null);
  }
  updatePicking(){
    this.renderer.onPick=this.active?(x,y)=>{
      if(this.placing){this.pick(x,y);return;}
      const hits=this.items.filter(entry=>entry.mesh&&this.renderer.controls.hit(x,y,entry===this.selectedEntry?this.renderer.mesh:entry.mesh));
      hits.sort((a,b)=>this.renderer.projectPoint(a.spec.position)[2]-this.renderer.projectPoint(b.spec.position)[2]);
      this.selectEntry(hits[0]||null);
    }:null;
    $('place-map').setAttribute('aria-pressed',String(this.placing));
    $('place-map').textContent=this.placing?'Click map to place':'Add obstacle';
    this.renderer.canvas.style.cursor=this.placing?'crosshair':'grab';
  }
  async prepare() {
    if(!applySimulationParameters())return;
    if(!this.active||!ui.bag.value||!ui.topic.value)return;
    const generation=++this.generation,bagId=ui.bag.value,topic=ui.topic.value;
    this.mapId=null;this.points=null;this.clearItems();this.renderer.setPoints(new Float32Array(0));this.renderer.setPrediction(null);
    $('prepare-map').disabled=true;$('place-map').disabled=true;$('map-job').hidden=false;
    try{
      let job=await api('/api/maps',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({bag_id:bagId,topic,initial_speed_m_s:simulationParameters.initial_speed_m_s})});
      while(generation===this.generation){
        $('map-status').textContent=job.message;$('map-progress').style.width=(job.progress*100)+'%';
        if(job.state==='failed')throw Error(job.message);
        if(job.state==='done')break;
        await new Promise(resolve=>setTimeout(resolve,500));
        if(generation!==this.generation)return;
        job=await api('/api/maps/jobs/'+job.id);
      }
      if(generation!==this.generation)return;
      const params=new URLSearchParams({bag_id:bagId,topic}),base=`/api/maps/${job.map_id}/`;
      const response=await fetch(base+'points?'+params);if(!response.ok)throw Error((await response.json()).detail);
      const points=new Float32Array(await response.arrayBuffer()),geometry=await api(base+'geometry?'+params);
      if(generation!==this.generation)return;
      this.mapId=job.map_id;this.points=points;this.geometry=geometry;
      $('place-map').disabled=false;$('map-status').textContent=`Map ready · ${geometry.point_count.toLocaleString()} points`;
      if(this.active)this.show();
    }catch(error){if(generation===this.generation)notice(error.message,'error');}
    finally{if(generation===this.generation)$('prepare-map').disabled=false;}
  }
  show(){
    this.renderer.setPoints(this.points);this.renderer.setMapLines(this.geometry.centerline,this.geometry.trajectory);
    const route=this.geometry.trajectory;
    this.renderer.target=route[Math.floor(route.length/2)]?.slice()||[0,0,0];
    let radius=10;for(const point of route)radius=Math.max(radius,Math.hypot(...point.map((v,i)=>v-this.renderer.target[i])));
    this.renderer.distance=Math.min(4500,radius*1.8+15);this.renderer.pitch=1.;this.renderer.yaw=0;this.renderer.requestDraw();
    ui.status.textContent='Merged map · add objects, then confirm each placement';
  }
  pick(x,y){
    const point=this.renderer.pickPoint(x,y);
    if(!point){notice('No visible map point here. Click a point or move the camera.');return;}
    const entry={id:++this.sequence,spec:{model_id:$('map-model').value,position:point,yaw_deg:0,scale:1},mesh:null};
    this.items.push(entry);this.selectEntry(entry);
    notice('Adjust the object, then Confirm placement to finish and add another.');
  }
  edit(){
    if(!this.placement)return;
    for(const input of $('map-transform').querySelectorAll('input'))if(!input.checkValidity())return;
    this.placement.position=['x','y','z'].map(name=>Number($('map-'+name).value));
    this.placement.yaw_deg=Number($('map-yaw').value);this.placement.scale=Number($('map-scale').value);this.preview();
  }
  async renderEntry(entry){
    const key=JSON.stringify(entry.spec),generation=this.generation;
    if(entry.mesh&&entry.meshKey===key)return;
    const mesh=await api('/api/world-preview',{method:'POST',headers:{'Content-Type':'application/json'},body:key});
    if(generation!==this.generation||!this.items.includes(entry)||JSON.stringify(entry.spec)!==key)return;
    entry.mesh=mesh;entry.meshKey=key;
    if(this.active){if(this.selectedEntry===entry){this.renderer.setMesh(mesh);this.renderer.controls.select(true);}this.refreshLayers();}
    this.drawItems();
  }
  preview(){
    this.previewVersion++;clearTimeout(this.previewTimer);
    if(!this.selectedEntry||!this.active)return;
    const entry=this.selectedEntry;
    this.previewTimer=setTimeout(()=>this.renderEntry(entry).catch(error=>notice(error.message,'error')),100);
  }
  request(){
    if(!this.mapId||!this.items.length)throw Error('Prepare a map and add an obstacle first.');
    if(this.placement)for(const input of $('map-transform').querySelectorAll('input'))if(!input.reportValidity())throw Error('Correct the object coordinates and scale.');
    return {bag_id:ui.bag.value,topic:ui.topic.value,map_id:this.mapId,world_obstacles:this.items.map(entry=>entry.spec)};
  }
}
