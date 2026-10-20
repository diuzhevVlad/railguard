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
    this.buffer=gl.createBuffer();this.grid=gl.createBuffer();this.leftRail=gl.createBuffer();this.rightRail=gl.createBuffer();this.train=gl.createBuffer();this.candidateBuffer=gl.createBuffer();this.candidateCount=0;this.rejectedBuffer=gl.createBuffer();this.rejectedCount=0;this.railPointsBuffer=gl.createBuffer();this.railPointsCount=0;this.centerlineBuffer=gl.createBuffer();this.centerlineCount=0;this.leftRailCount=0;this.rightRailCount=0;this.trainCount=0;
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
  setCandidates(points){
    if(!this.gl)return;const values=new Float32Array((points||[]).flat());this.candidateCount=values.length/3;
    this.gl.bindBuffer(this.gl.ARRAY_BUFFER,this.candidateBuffer);this.gl.bufferData(this.gl.ARRAY_BUFFER,values,this.gl.DYNAMIC_DRAW);this.requestDraw();
  }
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
  setPrediction(center,railPoints=null,maxForward=null,minForward=0,width=2.1,height=3,clearance=null){
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
        const offset=clearance?(-p[1]<=clearance.near_envelope_depth_m?clearance.near_envelope_offset_m:clearance.envelope_offset_m)||0:0;
        const lb=[p[0]+nx*width/2,p[1]+ny*width/2,p[2]+offset],rb=[p[0]-nx*width/2,p[1]-ny*width/2,p[2]+offset];
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
    if(this.count){gl.bindBuffer(gl.ARRAY_BUFFER,this.buffer);gl.vertexAttribPointer(this.position,3,gl.FLOAT,false,0,0);gl.uniform1f(this.uniforms.uSize,this.pointSize??3);gl.uniform4f(this.uniforms.uColor,.68,.70,.75,1);gl.drawArrays(gl.POINTS,0,this.count-this.obstacleCount-this.detectedCount);if(this.obstacleCount){gl.uniform1f(this.uniforms.uSize,4);gl.uniform4f(this.uniforms.uColor,1,.67,.35,1);gl.drawArrays(gl.POINTS,this.count-this.obstacleCount-this.detectedCount,this.obstacleCount);}}
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
    if(!this.mapMode&&this.candidateCount){gl.bindBuffer(gl.ARRAY_BUFFER,this.candidateBuffer);gl.vertexAttribPointer(this.position,3,gl.FLOAT,false,0,0);gl.uniform1f(this.uniforms.uSize,4);gl.uniform4f(this.uniforms.uColor,1,.67,.35,1);gl.drawArrays(gl.POINTS,0,this.candidateCount);}
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
