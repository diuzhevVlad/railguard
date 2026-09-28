# Obstacle Bag Studio

This local web app opens a ROS 2 SQLite bag, plays its `PointCloud2` topic in a
3D browser view, and writes a new bag with ray-rendered obstacles in every LiDAR
frame. The source bag is mounted read-only. It does **not** run
Gazebo; the `SimulationBackend` boundary in `obstacle_app/core.py` is where a
Gazebo-based producer can later be added.

From the `railguard/` repository root (after placing model files in `weights/` and bags in `data/`):

```bash
mkdir -p data outputs
docker compose up -d --build web
```

Open <http://127.0.0.1:8000>. Choose a source bag and point-cloud topic, then
press **Play**. The timeline slider follows playback. Drag it
to a new bag time and release to seek; an active playback continues from that
position, while a paused playback shows one frame and stays paused. The slider
seeks on release, so dragging does not repeatedly read the bag. Only the
current downsampled frame is sent to the browser; the bag is not preloaded.
In the 3D view, left-drag to orbit, right-drag to pan, and scroll to zoom.
Click the view (or focus it with Tab), then hold **W/S** to move forward/backward
and **A/D** to move sideways relative to the camera. **Shift** moves three times
faster. This works in playback and merged-map views; leaving the view, editing
a field, or switching browser tabs stops movement. The neutral dark palette
uses gray scan points while preserving the colored detection overlays.
Augmented obstacle returns are restricted to points ahead of the LiDAR (−Y).
Once an obstacle is entirely behind the LiDAR, no points or background shadow
from that obstacle are written to new exports. The rule applies to every model;
it is enforced in the bag backend, not by hiding points in the browser.
Known synthetic tails from an input app export are also stripped of rear
obstacle points, with retained synthetic labels carried into the new export.
Original recorded background remains intact. Old exports should be regenerated
from their original source bags to apply the new rule to every inserted object.
Points behind the camera are clipped correctly by the viewer.
Press **Detect obstacles: Off** to load RailGuard's vote model and refiner once,
then run them on the current full-resolution scan and subsequent displayed
frames. This uses `weights/path/best.pt`, `refine/last.pt`, and
`refine/calib.json`, mounted read-only at `/weights`. It does not build a merged
map or estimate odometry for detection. Pausing and seeking work with the same
button; turning detection off leaves the player in place. Recent scan results
are cached (12 per recording/topic session). Failed model loading is reported;
there is no silent fallback to merged GT rails.

The model expects Hesai Pandar128 scans with forward −Y and Z up. Ring IDs are
preserved when present, otherwise inferred from elevation; intensity is ignored
by the deployed weights. Restart the app after replacing weight files to reload them. RailGuard predicts the centerline and lateral sigma
from one scan. Detection uses the finite predicted centerline, including the refiner
continuation, up to the selected maximum forward distance (150 m by default, adjustable to 200 m). Points beyond that limit
are excluded. Detection also excludes points less than 1.5 m forward of the
LiDAR, before clustering (`ObstacleFilterConfig.min_forward_m`). This is an
intentional near-sensor blind zone, including for real objects. The full
prediction remains visible up to 200 m.
The model’s segmented rail points are blue and the full predicted centerline is
a thick green line. The train envelope uses thin, subdued lines. The continuation
beyond the vote-supported horizon is also used for detection; its greater
uncertainty can affect obstacle labels. The status reports checked
range, vote horizon, inference time, and maximum lateral sigma in the checked
range. Sigma is uncertainty of the centerline, not obstacle confidence; it does
not automatically widen the physical train envelope.

The train envelope defaults to 2.1 m wide and 3 m high; the Parameters panel changes both dimensions. Every full-resolution point
strictly above predicted track height and inside the envelope becomes a candidate.
Points classified as rails by RailGuard are explicitly excluded before
cluster filtering. Matching uses the model's returned XYZ with a 0.01 mm
numerical tolerance, not an exclusion zone around the rail. Nearby non-rail
points remain eligible. Excluded rails remain blue; the purple debug overlay
contains only points rejected by height or clustering. This depends on the model's
segmentation: rail points it fails to label can still become candidates.
Non-rail points must be more than 20 cm above track height inside the first 10 m, and more than 10 cm above track height farther away.
This tolerance is configurable as `min_height_above_track_m` in
`ObstacleFilterConfig`; objects whose returns are all below this threshold can
be missed. Remaining candidates pass a per-scan 3D connected-component filter: only clusters of
at least 5 points are marked red. The neighbor radius interpolates between
0.2 m at 0–30 m ahead, 0.35 m at 80 m, and 0.6 m at 150 m. Both points must
satisfy their radius for a connection. Configure these values through
`ObstacleFilterConfig` in `railpath/obstacle_filter.py`, optionally passed to
`ScanDetector(filter_config=...)`. Up to 20,000 retained and 20,000 rejected
points are sent as display overlays; filtering uses full-resolution candidates.
Enable **Show rejected points** to inspect points discarded by distance, height or clustering in purple, including
while paused. This is an experimental noise filter: sparse wires and distant
objects may be rejected. It does not fix errors in the predicted track height. This is a
geometric test, not object classification or tracking. A missing/invalid
centerline is explicitly shown as detection unavailable, not as a clear track.
Rail height errors can still cause false positives. Distance readouts use arc
length calculated from the predicted curve. The centerline is resampled densely
before the envelope check to avoid gaps between the model's 1 m samples.

The Docker image includes PyTorch 2.6 with CUDA 12.4 and the upstream `railpath`
package. Compose requests the NVIDIA GPU and uses `OBSTACLE_RAIL_DEVICE=cuda`.
The NVIDIA Container Toolkit and a compatible driver are required for this
configuration. For a CPU-only machine, remove the Compose `gpus` entry, set
`OBSTACLE_RAIL_DEVICE=cpu`, and optionally build with
`--build-arg TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu`. CPU latency
is substantially higher. The app installs Python dependencies directly in the
container; it does not require ROS installation, Pixi, or Conda for inference.

The merged map editor and world-placement exports still use their separate
odometry/rail cache. Opening the app no longer precomputes a map for the fake
object bag. A map is prepared only when requested by the editor.

The map builder seeds the second KISS-ICP scan with the agreed initial speeds:
10 m/s for `roundT_doubleT`, `roundT_squareT_pressureGate_squareT`,
`squareT_platform_squareT_switch`, and `new_data`; 5 m/s for
`roundT_pressureGate_roundT`; 3 m/s for `cloud_with_fake_obj`; and 0 m/s for
`doubleT_obstacle`. Other bags default to 0 m/s. This is only the initial
motion guess, not a fixed train speed for the run.

### Place objects on the merged map

Choose a bag and topic, then switch **Placement mode** to **Merged map**.
Preparation shows progress while estimating/caching poses and building the map.
The preview contains at most 500,000 points, a red rail centerline, and a blue
train trajectory. The first preparation can take time; subsequent opens reuse
`outputs/.world_maps/` and the detector's pose cache. Point-cloud merging reads
one scan at a time and bounds its temporary point pool.

Click a mesh thumbnail to select a model and press **Add obstacle**, then click a visible map point.
The placed object is selected with a yellow bounding box and colored translation
handles. Drag red **X**, green **Y**, or blue **Z** to move along that world axis.
The **pink handle** scales the whole object: drag outward to enlarge or inward
to shrink. Scaling keeps its placement anchor fixed (the base for ordinary
objects and top attachment for wires). The numeric **Scale** field allows exact
values. Drag the **cyan ring** to rotate about the vertical Z axis, or enter
an exact angle in **Yaw**. Rotation keeps the placement anchor fixed.
Press **Confirm placement** to finish editing it. Then choose another model and
press **Add obstacle** to place the next object. Confirmed objects stay visible;
up to 50 can share one export. **Add obstacle** also confirms the current object
before starting a new one. Select any existing object by clicking its mesh or
its entry in the placed-object list to move, rotate, scale or remove it. Click
empty space to deselect. Dragging elsewhere still orbits/pans the camera.
**Remove** deletes only the selected object.
Edit **World X/Y/Z**, **Yaw**, and **Scale** to refine placement, including
positions between observed surfaces. These are unrestricted world coordinates,
not rail offsets. Yaw starts at 0 degrees around world Z and scale starts at 1.

Eight models are available as clickable mesh thumbnails: Niva car, two boxes,
standing person, person from photo, alien, stylized Donald Trump, and hanging wire.
The Niva is a simplified original mesh based on the reference photograph, with
approximate dimensions; its windows are opaque to the current LiDAR renderer. Ordinary models use their
bottom-center anchor. A wire attaches by its top and extends downward; click a
ceiling point to hang it. The preview shows the models’ material colors with lighting. Boxes have a
procedural cardboard appearance with grain, taped seams and labels. These
appearance details are browser-only; the preview geometry uses the exact
transform used by the LiDAR export renderer.

**Simulate & export bag** creates a new full recording with all placed
world objects, including the currently selected object. They share one ray-casting
scene, so objects can occlude one another. It reuses the editor's exact timestamp/pose sequence without
estimating odometry again. Ray casting, tunnel occlusion, background shadows,
forward visibility and per-scan contact protection still apply. The completed
bag opens in the player. Switching source bags or topics clears the placements;
stale map IDs, changed source files, or changed cached poses are rejected.
World alignment still depends on the quality of the original estimated poses.

**Encounter time** remains a separate mode with its existing obstacle list and
placement limits. Each export uses only the active mode; the training sampler
is separate from this editor.

### Place objects by encounter time

Choose a bag and its point-cloud topic under **Recording**. Use **Add obstacle**
to create a list of objects. Each object has:

- A model selected by clicking its mesh thumbnail.
- **Train reaches object at · seconds**: time from the source recording's start.
  For example, 12 s places the object at the train's estimated route position
  at 12 s. The object exists from the beginning and stays fixed in the world;
  this is an encounter time, not a spawn time.
- **Distance from track center**: −1 to +1 m; negative is left and positive is right.
- **Height above track**: 0 places the object's feet at rail level.
  Increase it to place the object on a platform or raised surface. This does
  not automatically detect platform height.

Press **Simulate & export bag** to process the full source recording. All objects
share one ray-casting scene, including occlusion between objects. The completed
bag is stored under `outputs/` and selected for playback. The object list remains unchanged. Simulation uses the currently selected
bag and topic; select the original source again to generate another independent
export. The playback slider does not change encounter times. You can also replay:

```bash
ros2 bag play outputs/<export-name>
```

`data/` may contain multiple bag directories, each with `metadata.yaml`.
`outputs/` is created on the host before Compose starts so the container can
write to it as user 1000. If your host UID or GID differs, run Compose with
`LOCAL_UID=$(id -u) LOCAL_GID=$(id -g)` in its environment.

The first export pass estimates KISS-ICP poses and merges RailGuard rail
observations. For each encounter time, the backend interpolates the train pose
and projects its position onto the rail centerline. If the merged route is
truncated, nearby paired rails at that time establish the track base instead.
If neither can support placement, the job fails with a clear error rather than
moving the object to the shortened route's endpoint. Times must lie within the
recorded LiDAR scans. Timing and placement depend on odometry accuracy; if the
train is stationary, different encounter times can select the same location.
Placements overlapping a conservative 0.3 m clearance around the starting
LiDAR are rejected before writing. Use a moving recording for timed encounters,
and a later encounter time that clears the sensor.

Each obstacle faces the rail direction and stands vertically in the map frame.
Its world position stays fixed; every scan transforms rays using the train's
full estimated pose. After passage, rear obstacle returns are omitted from the
exported bag. No distance-along-rails, rail-end policy, or raw Z controls are
needed in the menu. `ObstacleSpec` and `Scenario.obstacles` in `core.py` carry
these settings independently of the UI and simulation backend.

`merged_rails.npz` stores the centerline, arc lengths, poses, and timestamps.
`obstacle_scenario.json` records each object's ID, model, encounter timestamp,
world anchor, sideways offset and height, plus per-frame added/removed counts.
The renderer loads each distinct model once and uses separate angular bounds
for objects to limit ray-casting work.

The renderer caches an Open3D triangle acceleration structure for the stationary
obstacles. It renders first intersections on an approximate 128-channel sensor
grid and removes recorded returns whose own directions intersect a closer
obstacle surface. Closer recorded foreground suppresses synthetic returns using
the nearest angular cell's minimum range. Genuine gaps in the mesh remain open;
there is no rectangular bounding-box shadow. Rays without an original return
can produce a new obstacle return.

When all 128 rings have enough valid returns, the grid uses estimated median
channel elevations, horizontal 0.1°/0.2° steps, and phases from the first scan.
Otherwise it uses a nominal 128-channel approximation spanning −25° to +15°,
with a denser central band and 0.1° horizontal steps. This fallback is not an
exact manufacturer calibration. The output records the inferred/fallback
model, per-frame added/removed counts, and totals in `obstacle_scenario.json`.
Playback highlights the varying synthetic tail and reports removed counts;
older fixed-sampling exports remain replayable.

New points receive their simulated channel's ring ID, median retained-frame
intensity and timestamp, and zero for other fields. Geometry models opaque
first-hit returns, with a 0.3–200 m synthetic return range. It does not model
material reflectance/dropout, beam divergence, multiple echoes, or motion
during one scan. Visibility remains independent of return range: a blocking
opaque surface does not become transparent when its synthetic echo is rejected.
Estimated pose drift and approximate angular-cell matching can affect fidelity.

The app checks an estimated space requirement before writing. `data/new_data`
is 84 GB, so exporting all of it would require much more free disk space than
is currently available. Failed exports remove
their temporary output directory.

Run focused tests on the host after installing `requirements-web.txt`
and the test client dependency (`httpx2` for the current Starlette version):

```bash
python3 -m unittest discover -s tests -p 'test_obstacle_app.py'
python3 -m unittest discover -s tests -p 'test_ray_rendering.py'
python3 -m unittest discover -s tests -p 'test_world_map.py'
python3 -m unittest discover -s tests -p 'test_learned_detection.py'
```

### Tunnel occlusion during simulation

Simulate now builds a temporary mesh from each current LiDAR scan, connecting nearby angular samples. This mesh hides synthetic obstacle returns behind observed tunnel surfaces and prevents hidden objects from casting new background shadows. Existing recorded background points remain governed by the obstacle shadow logic; tunnel geometry never replaces them. The mesh does not use merged odometry, although stationary obstacle placement still does.

Triangles are rejected across large range discontinuities, edges over 1.5 m, and vertical gaps over 0.75 degrees. A surface must be at least 0.3 m closer than an obstacle to hide it. Mesh construction is limited to obstacle viewing cones with a 2 degree margin; meshes are discarded after each frame. Missing surfaces retain the existing angular-depth fallback. Small openings narrower than the scan sampling resolution and reconstruction holes can still produce visibility errors. Previously generated synthetic tails are excluded from tunnel reconstruction.

Export metadata records `tunnel_visibility`, plus per-frame `tunnel_hidden_points` and `tunnel_mesh_triangles`. No separate UI control is needed; this applies to all obstacle models when Simulate is pressed.


Obstacle encounter times must leave **2 seconds at the beginning and end** of the selected recording. Recordings of 4 seconds or less cannot accept obstacles. The browser restricts the time field and the job API validates the same bounds. Existing obstacle times remain visible when switching recordings, so invalid values must be corrected before export.

Sideways obstacle placement is limited to −1 to +1 m in both browser and API. Height above track is adjustable from 0 to 4 m.

### Export speed and timing

The export progress includes all processing, not just disk writing. The browser shows a progress bar and message/frame counts. Exported `obstacle_scenario.json` includes a `performance` summary with odometry, rail detection, decoding, rendering, point packing, serialization, writing, and writer-finalization timings. The stage sums are partial: source reads/decoding during the first pass and miscellaneous setup appear in the overall phase totals.

Tunnel meshing now uses only vertices within the distance of relevant obstacle hits plus the triangle edge limit. More distant surfaces cannot occlude these hits. Empty-hit scans skip meshing, angular neighbor lookup visits occupied grid cells, synthetic records are packed with NumPy, and scans after all objects pass skip point-coordinate work. These changes preserve the sensor firing pattern and original recorded point resolution.

### Stops and sensor contact

If the LiDAR is inside an obstacle mesh, or within its 0.3 m minimum range of the mesh surface, that object is temporarily excluded from ray casting. Its synthetic points and background shadows are both omitted; other objects continue rendering. The contact test uses signed distance to the closed mesh, with bounding boxes used only to select candidates. Per-frame `near_contact_obstacles` and `near_contact_policy` are saved in export metadata.

Encounter times separated by at least 2 seconds that place objects within 0.5 m of one location during a nearly stationary trajectory produce a placement warning. Completion warnings are shown in the app and saved in `obstacle_scenario.json`. Moving the encounter time outside the stop avoids stacking objects at one position. This guard does not model physical train collisions.

To recover previously erased background points, generate a new export from the original source recording. Existing augmented bags are unchanged.

Model thumbnails are screenshots of the same WebGL mesh preview, including box
materials. Regenerate them against the running app with
`python tools/render_model_thumbnails.py` (Playwright and Chromium required),
then rebuild the app image to include the updated PNG files.


### Parameters

The **Parameters** panel controls initial odometry speed (m/s), detection-envelope
width and height (m), and maximum detection distance (5–200 m slider). Click
**Apply parameters** to refresh the current detection without seeking elsewhere.
Train width and height also apply automatically when leaving the edited field;
the detection-distance slider applies on release. Initial-speed changes still
require **Apply parameters** because they clear map placements.
Initial speed defaults to the selected recording's preset. Changing speed clears
map placements; reopen the merged map before placing/exporting objects. It does
not change playback speed or the per-scan neural prediction. Map caches include
initial speed; detection caches include width, height, and range. Defaults remain
2.1 m wide, 3 m high, and 150 m detection range.

### Shared code and storage

Both the ROS node and web app use `railpath/` for neural inference and
`railpath/obstacle_filter.py` for envelope selection, rail exclusion,
height thresholds, and connected-component filtering. The app's original
filter imports are compatibility wrappers, not a second implementation.

The browser reads SQLite bags directly using `rosbags`; it does not subscribe
to ROS topics and does not require `ros2 bag play`. The ROS node remains a
separate Docker image/workflow described in the root README.

Compose mounts `data/` read-only at `/data`, `weights/` read-only at `/weights`,
and `outputs/` writable at `/output`. Existing exports and map/pose caches live
under `outputs/`; do not delete it when rebuilding. On Linux with a different
user ID, start with `LOCAL_UID=$(id -u) LOCAL_GID=$(id -g) docker compose up -d web`.
Use `WEB_PORT=8001 docker compose up -d web` if port 8000 is already occupied.

### Verification

From the repository root, test the packaged source in the built image:

```bash
docker compose build web
docker run --rm -e PYTHONPATH=/work:/work/railguard \
  -v "$PWD:/work:ro" -w /work railguard-web \
  python -m unittest discover -s tests -p 'test_*.py'
python3 tests/check_viewer_browser.py
python3 tests/check_viewer_layout_browser.py
```

The last two checks require Playwright and `/usr/bin/google-chrome` on the host.
Training experiments, historical metric reports, and evaluation bag-generation
scripts remain in the original research workspace; they are not runtime dependencies.
