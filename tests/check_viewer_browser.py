"""Exercise the actual viewer shaders in headless Chrome via Playwright.

Run manually: python3 tests/check_viewer_browser.py
Requires Playwright and a Chrome executable, not ROS or a running app.
"""
from pathlib import Path
import json

from playwright.sync_api import sync_playwright


source = (Path(__file__).resolve().parents[1] / "obstacle_app/static/app.js").read_text()
renderer = source[source.index("class CloudRenderer{"):source.index("\nconst renderer=new CloudRenderer")]
with sync_playwright() as playwright:
    browser = playwright.chromium.launch(executable_path="/usr/bin/google-chrome", headless=True,
        args=["--no-sandbox", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"])
    page = browser.new_page()
    page.set_content('<canvas id="cloud" style="width:400px;height:400px"></canvas>')
    page.add_script_tag(content=(Path(__file__).resolve().parents[1] / "obstacle_app/static/mesh-controls.js").read_text())
    page.add_script_tag(content="const detectionVisible=false; const $=id=>document.getElementById(id); function notice(message){throw Error(message);}\n"
                         + renderer + "\nwindow.viewer=new CloudRenderer($('cloud'));")
    result = page.evaluate("""() => {
      const r=window.viewer,gl=r.gl;
      if(!gl.getProgramParameter(r.program,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(r.program));
      r.gridCount=0;r.yaw=0;r.pitch=0;r.distance=10;r.target=[0,0,0];
      const sample=(point,yaw=0)=>{
        r.yaw=yaw;
        r.setPoints(new Float32Array(point),1,0);r.draw();
        const pixel=new Uint8Array(4);
        gl.readPixels(gl.drawingBufferWidth/2,gl.drawingBufferHeight/2,1,1,gl.RGBA,gl.UNSIGNED_BYTE,pixel);
        if(gl.getError()!==gl.NO_ERROR)throw Error('WebGL error');
        return [...pixel];
      };
      return {
        ahead:sample([0,-5,0]),
        recordedRear:sample([0,5,0]),
        behindCamera:sample([0,20,0]),
        lookingBackward:sample([0,20,0],Math.PI),
        nearClip:sample([0,9.95,0])
      };
    }""")
    orange = lambda pixel: pixel[0] > 200 and pixel[1] > 100
    assert orange(result["ahead"]), result
    assert orange(result["recordedRear"]), result
    assert orange(result["lookingBackward"]), result
    for key in ("behindCamera", "nearClip"):
        assert not orange(result[key]), (key, result)
    print(json.dumps(result, indent=2))
    print("PASS: no display masking; camera clipping, near plane, and backward orbit.")
    page.evaluate("""() => {
      const r=window.viewer;
      const check=(forward,right,yaw,pitch,fast)=>{
        r.target=[0,0,0];r.yaw=yaw;r.pitch=pitch;r.distance=10;
        r.moveCamera(forward,right,.25,fast);return [...r.target];
      };
      const close=(a,b)=>a.every((v,i)=>Math.abs(v-b[i])<1e-8);
      if(!close(check(1,0,0,0,false),[0,-1,0]))throw Error('Forward direction');
      if(!close(check(-1,0,0,0,false),[0,1,0]))throw Error('Backward direction');
      if(!close(check(0,1,0,0,false),[-1,0,0]))throw Error('Right direction');
      if(!close(check(0,-1,0,0,false),[1,0,0]))throw Error('Left direction');
      if(!close(check(1,0,Math.PI/2,0,false),[1,0,0]))throw Error('Rotated direction');
      if(Math.abs(Math.hypot(...check(1,1,0,.4,false))-1)>1e-8)throw Error('Diagonal speed');
      if(!close(check(1,0,0,0,true),[0,-3,0]))throw Error('Shift speed');
      r.target=[0,0,0];r.yaw=0;r.pitch=0;
    }""")
    page.focus('#cloud')
    page.keyboard.down('w')
    page.wait_for_function('window.viewer.target[1]<-.1')
    # Losing focus while a key is held must stop movement, including in fields.
    page.evaluate("""() => {
      const field=document.createElement('input');field.id='editing';document.body.append(field);field.focus();
    }""")
    stopped = page.evaluate('[...window.viewer.target]')
    page.keyboard.up('w')
    page.keyboard.type('wasd')
    page.evaluate('() => new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)))')
    assert page.evaluate('window.viewer.target') == stopped
    assert page.input_value('#editing') == 'wasd'
    # Keyboard navigation also runs in merged-map mode.
    page.evaluate('window.viewer.mapMode=true')
    page.focus('#cloud')
    page.keyboard.down('d')
    page.wait_for_function('window.viewer.target[0]<-.1')
    page.keyboard.up('d')
    print('PASS: camera-relative WASD, normalized diagonal speed, Shift, focus safety, and map mode.')
    browser.close()
