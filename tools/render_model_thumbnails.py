"""Capture the web renderer's actual meshes; run against the local app."""
from pathlib import Path
from playwright.sync_api import sync_playwright
ROOT=Path(__file__).resolve().parents[1]
output=ROOT/'obstacle_app/static/model-thumbnails'
output.mkdir(parents=True,exist_ok=True)
with sync_playwright() as p:
    browser=p.chromium.launch(headless=True,args=['--no-sandbox','--use-angle=swiftshader','--enable-unsafe-swiftshader'])
    page=browser.new_page(viewport={'width':400,'height':350},device_scale_factor=2)
    page.goto('http://localhost:8000')
    page.wait_for_function('models.length>0')
    page.evaluate("""()=>{document.body.append(renderer.canvas);renderer.canvas.style='position:fixed;left:0;top:0;width:400px;height:350px;z-index:999';renderer.gridCount=0;renderer.mapMode=true;renderer.setPoints(new Float32Array());renderer.setPrediction(null);renderer.controls.selected=false;const clear=renderer.gl.clearColor.bind(renderer.gl);renderer.gl.clearColor=()=>clear(.70,.75,.77,1);}""")
    for model in page.evaluate('models'):
        page.evaluate("""async id=>{
          const mesh=await api('/api/world-preview',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({model_id:id,position:[0,0,0],yaw_deg:0,scale:1})});
          renderer.setMesh(mesh);const bounds=renderer.controls.bounds;
          renderer.target=bounds[0].map((v,i)=>(v+bounds[1][i])/2);
          const extent=bounds[0].map((v,i)=>bounds[1][i]-v);
          renderer.distance=Math.max(...extent)*.35;renderer.pitch=.25;renderer.yaw=-1.0;
          for(let attempt=0;attempt<40;attempt++){
            const fits=mesh.vertices.every(v=>{const q=renderer.projectPoint(v);return q[2]>.12&&q[0]>30&&q[0]<370&&q[1]>30&&q[1]<320;});
            if(fits)break;renderer.distance*=1.12;
          }
          renderer.requestDraw();
        }""",model['id'])
        page.wait_for_timeout(180)
        page.locator('#cloud').screenshot(path=str(output/(model['id']+'.png')))
        print(model['id'],flush=True)
    browser.close()
