/* Translation, uniform scale and Z rotation use the same anchor as export. */
class MeshControls {
  constructor(renderer){this.r=renderer;this.selected=false;this.buffer=renderer.gl.createBuffer();}
  update(){
    const mesh=this.r.mesh;if(!mesh){this.selected=false;this.bounds=null;return;}
    this.bounds=[0,1].map(side=>[0,1,2].map(axis=>mesh.vertices.reduce((v,p)=>side?Math.max(v,p[axis]):Math.min(v,p[axis]),side?-Infinity:Infinity)));
  }
  geometry(){
    if(!this.bounds)return null;
    const r=this.r,corners=Array.from({length:8},(_,i)=>r.transformMeshPoint([0,1,2].map(a=>this.bounds[(i>>a)&1][a])));
    const lo=[0,1,2].map(a=>Math.min(...corners.map(p=>p[a]))),hi=[0,1,2].map(a=>Math.max(...corners.map(p=>p[a])));
    const center=lo.map((v,i)=>(v+hi[i])/2),depth=r.projectPoint(center)[2];
    if(depth<=.1)return null;
    const length=Math.max(.15,depth*140/(r.canvas.clientHeight*1.35));
    const c=Math.cos(r.yaw),s=Math.sin(r.yaw),sp=Math.sin(r.pitch),cp=Math.cos(r.pitch);
    const diagonal=[-c+s*sp,-s-c*sp,cp];
    const scaleHandle=center.map((v,i)=>v+length*diagonal[i]*1.15);
    const ring=Array.from({length:73},(_,i)=>[center[0]+length*1.4*Math.cos(i*Math.PI/36),center[1]+length*1.4*Math.sin(i*Math.PI/36),center[2]]);
    return {lo,hi,center,length,scaleHandle,ring,ends:[0,1,2].map(axis=>center.map((v,i)=>v+(i===axis?length:0)))};
  }
  hit(x,y,mesh=this.r.mesh){
    if(!mesh)return false;
    const projected=mesh.vertices.map(p=>this.r.projectPoint(mesh===this.r.mesh?this.r.transformMeshPoint(p):p));
    for(const face of mesh.triangles){
      const [a,b,c]=face.map(i=>projected[i]);if(Math.min(a[2],b[2],c[2])<=.1)continue;
      const det=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1]);if(Math.abs(det)<1e-8)continue;
      const u=((b[1]-c[1])*(x-c[0])+(c[0]-b[0])*(y-c[1]))/det;
      const v=((c[1]-a[1])*(x-c[0])+(a[0]-c[0])*(y-c[1]))/det;
      if(u>=0&&v>=0&&u+v<=1)return true;
    }
    // A thin wire can be smaller than one screen pixel.
    return projected.some(p=>p[2]>.1&&Math.hypot(p[0]-x,p[1]-y)<5);
  }
  select(value){this.selected=value;this.r.requestDraw();}
  rotationAngle(x,y,center){
    const r=this.r,c=Math.cos(r.yaw),s=Math.sin(r.yaw),cp=Math.cos(r.pitch),sp=Math.sin(r.pitch);
    const forward=[s*cp,-c*cp,-sp],right=[-c,-s,0],up=[s*sp,-c*sp,cp];
    const nx=(2*x/r.canvas.clientWidth-1)*(r.canvas.clientWidth/r.canvas.clientHeight)/1.35,ny=(1-2*y/r.canvas.clientHeight)/1.35;
    const origin=r.target.map((v,i)=>v-forward[i]*r.distance),direction=forward.map((v,i)=>v+nx*right[i]+ny*up[i]);
    if(Math.abs(direction[2])<1e-4)return null;
    const t=(center[2]-origin[2])/direction[2];if(t<=0)return null;
    return Math.atan2(origin[1]+t*direction[1]-center[1],origin[0]+t*direction[0]-center[0]);
  }
  begin(x,y){
    const g=this.geometry();if(!this.selected||!g||!this.onMove)return false;
    const start=this.r.projectPoint(g.center);let candidate=null,best=10;
    for(let axis=0;axis<3;axis++){
      const end=this.r.projectPoint(g.ends[axis]),dx=end[0]-start[0],dy=end[1]-start[1],norm=dx*dx+dy*dy;
      if(norm<100||end[2]<=.1)continue;
      const t=((x-start[0])*dx+(y-start[1])*dy)/norm;
      const distance=Math.hypot(x-start[0]-Math.max(.2,Math.min(1,t))*dx,y-start[1]-Math.max(.2,Math.min(1,t))*dy);
      if(distance<best){best=distance;candidate={axis,x,y,dx,dy,norm,length:g.length};}
    }
    const scalePoint=this.r.projectPoint(g.scaleHandle);
    if(scalePoint[2]>.1&&Math.hypot(x-scalePoint[0],y-scalePoint[1])<12)candidate={mode:"scale",x,y};
    if(!candidate){
      const projected=g.ring.map(p=>this.r.projectPoint(p));
      for(let i=1;i<projected.length;i++){
        const a=projected[i-1],b=projected[i];if(Math.min(a[2],b[2])<=.1)continue;
        const dx=b[0]-a[0],dy=b[1]-a[1],norm=dx*dx+dy*dy;
        const t=Math.max(0,Math.min(1,((x-a[0])*dx+(y-a[1])*dy)/(norm||1)));
        if(Math.hypot(x-a[0]-t*dx,y-a[1]-t*dy)<8){
          const angle=this.rotationAngle(x,y,g.center),screen=this.r.projectPoint(g.center);
          const fallback=angle===null||Math.abs(Math.sin(this.r.pitch))<.1;
          candidate={mode:'rotate',center:g.center,screen,fallback,last:fallback?Math.atan2(y-screen[1],x-screen[0]):angle,total:0};break;
        }
      }
    }
    this.drag=candidate;if(candidate)this.onStart?.();return !!candidate;
  }
  move(x,y){
    if(!this.drag)return;const d=this.drag;
    if(d.mode==='rotate'){
      const angle=d.fallback?Math.atan2(y-d.screen[1],x-d.screen[0]):this.rotationAngle(x,y,d.center);
      if(angle===null)return;
      d.total+=Math.atan2(Math.sin(angle-d.last),Math.cos(angle-d.last));d.last=angle;
      this.r.meshAngle=this.onRotate(d.total);this.r.requestDraw();return;
    }
    if(d.mode==='scale'){this.r.meshScale=this.onScale(Math.exp(Math.max(-12,Math.min(12,((x-d.x)-(y-d.y))/140))));this.r.requestDraw();return;}
    const delta=[0,0,0];delta[d.axis]=((x-d.x)*d.dx+(y-d.y)*d.dy)/d.norm*d.length;
    this.r.meshOffset=delta;this.onMove(delta);this.r.requestDraw();
  }
  end(cancel){
    if(!this.drag)return;const mode=this.drag.mode;this.drag=null;
    if(cancel){this.r.meshOffset=[0,0,0];this.r.meshScale=1;this.r.meshAngle=0;if(mode==='scale')this.onScale(1);else if(mode==='rotate')this.onRotate(0);else this.onMove([0,0,0]);}
    if(this.r.mesh){this.r.mesh.vertices=this.r.mesh.vertices.map(p=>this.r.transformMeshPoint(p));this.r.setMesh(this.r.mesh);}
    this.onEnd?.();
  }
  draw(){
    if(!this.selected)return;const g=this.geometry();if(!g)return;
    const r=this.r,gl=r.gl,corners=[];
    for(let i=0;i<8;i++)corners.push([0,1,2].map(a=>(i&(1<<a))?g.hi[a]:g.lo[a]));
    const lines=[];for(let i=0;i<8;i++)for(let a=0;a<3;a++)if(!(i&(1<<a)))lines.push(...corners[i],...corners[i|(1<<a)]);
    const draw=(data,color,mode=gl.LINES)=>{gl.bindBuffer(gl.ARRAY_BUFFER,this.buffer);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(data),gl.DYNAMIC_DRAW);gl.vertexAttribPointer(r.position,3,gl.FLOAT,false,0,0);gl.uniform4fv(r.uniforms.uColor,color);gl.drawArrays(mode,0,data.length/3);};
    gl.disable(gl.DEPTH_TEST);draw(lines,[1,.85,.3,1]);draw(g.ring.flat(),[.25,.95,1,1],gl.LINE_STRIP);
    [...g.ends,g.scaleHandle].forEach((end,axis)=>{
      const color=[[1,.25,.23,1],[.3,1,.4,1],[.3,.6,1,1],[1,.45,1,1]][axis];draw([...g.center,...end],color);
      const h=g.length*.055,verts=[];
      for(let i=0;i<8;i++)verts.push(end.map((v,a)=>v+((i&(1<<a))?h:-h)));
      const triangles=[0,1,3,0,3,2,4,6,7,4,7,5,0,4,5,0,5,1,2,3,7,2,7,6,0,2,6,0,6,4,1,5,7,1,7,3];
      draw(triangles.flatMap(i=>verts[i]),color,gl.TRIANGLES);
    });gl.enable(gl.DEPTH_TEST);
  }
}
