"""Check stable viewer layout with real HTML/CSS; no server or ROS required."""
from pathlib import Path
import re

from playwright.sync_api import sync_playwright


root = Path(__file__).resolve().parents[1] / "obstacle_app/static"
html = re.sub(r"<script\b[^>]*>.*?</script>", "", (root / "index.html").read_text(), flags=re.S)
html = re.sub(r'<link\b[^>]*rel="stylesheet"[^>]*>', "", html)
with sync_playwright() as p:
    browser = p.chromium.launch(executable_path="/usr/bin/google-chrome", headless=True,
                                args=["--no-sandbox"])
    page = browser.new_page()
    page.set_content(html)
    page.add_style_tag(content=(root / "style.css").read_text())
    for width, height in [(1440, 900), (1200, 800), (1000, 768), (390, 844)]:
        page.set_viewport_size({"width": width, "height": height})
        # Detection controls stay visible as frames update.
        page.evaluate("""() => {
          document.getElementById('detection-job').hidden=false;
          document.getElementById('detection-legend').hidden=false;
        }""")
        layouts = []
        for text in ["Opening bag…", "Frame 9 · No obstacle clusters after filtering",
                     "Frame 100 · 20,000 preview points · hesai_lidar · 125000 filtered obstacle points · nearest 1.5 m · 400 points rejected by distance/height/cluster filters · 1500 rail points excluded · RailGuard 130 ms · checked 150 m · vote horizon 120 m · max lateral σ 0.50 m"]:
            layouts.append(page.evaluate("""text => {
              document.getElementById('frame-status').textContent=text;
              document.getElementById('notice').textContent=text;
              document.getElementById('detection-job-text').textContent=text;
              const canvas=document.getElementById('cloud');
              // Drawing-buffer size must not feed back into CSS layout.
              canvas.width=2400;canvas.height=1800;
              const r=canvas.getBoundingClientRect();
              return {x:r.x,y:r.y,w:r.width,h:r.height,scroll:scrollY,
                      pageHeight:document.documentElement.scrollHeight};
            }""", text))
        assert all(layout == layouts[0] for layout in layouts), (width, layouts)
        assert layouts[0]["h"] > 100, (width, layouts)
        if width > 900:
            assert layouts[0]["pageHeight"] == height, (width, layouts)
        print(f"PASS: stable viewer at {width}×{height}")
    browser.close()
