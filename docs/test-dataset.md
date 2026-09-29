# Generate a six-bag obstacle test dataset

Run `tools/generate_test_dataset.py` from a RailGuard checkout. It uses the app's
`RayCastingBackend`, `timed_anchor` and `world_mesh` functions directly. There is
no separate LiDAR renderer or detection-model inference in this tool.

Default source bags in `data/`:

- `doubleT_platform`
- `new_data`
- `roundT_doubleT`
- `roundT_pressureGate_roundT`
- `roundT_squareT_pressureGate_squareT`
- `squareT_platform_squareT_switch`

`doubleT_obstacle` and `cloud_with_fake_obj` are excluded.

## Scene sampling

Edit `config/test_dataset.json` to change the seed, schedule and sampling ranges.
The defaults are:

- One encounter at 5, 20, 35, … seconds in **every** bag, including `new_data`.
  Keep a two-second end margin.
- Balance standing people, square boxes and long boxes within each bag; no wires.
- Lateral offset sampled uniformly from −1 to +1 m relative to the reference
  centerline. Positive is right along the train's travel direction.
- Bottom anchor at track height. Relative yaw sampled from −45° to +45°.
- Person scale 0.9–1.1; square-box scale 0.5–3; long-box scale 0.5–1.5.
  Uniform scaling changes all dimensions. A square box is therefore 0.25–1.5 m
  per side; a long box is 1–3 m long and 0.1–0.3 m wide/high.

Objects remain fixed in world coordinates and may be visible before their
encounter time. The encounter time is when the train reaches their location,
not a spawn time. The app handles ray casting, mutual and per-scan tunnel
occlusion, removal of background points, forward visibility and near contact.

The reference/“GT” centerline is the same merged rail estimate used by the app,
not surveyed ground truth. Positions are projected onto it through the app's
placement function. Missing reliable coverage is reported in
`placement_failures.json`; the tool stops before exporting rather than silently
moving or dropping an obstacle. This command creates bags and placement metadata;
it does not calculate detection metrics or add a point-label field.

## Run with Docker

Build the existing web image and choose a writable output location. The external
output mount keeps large bags off the system disk. Run from the repository root:

```sh
docker compose build web
# Set this to your chosen dataset directory, possibly on an external disk:
DATASET_OUTPUT=/path/to/test_dataset
mkdir -p "$DATASET_OUTPUT"
docker run --rm --user "$(id -u):$(id -g)" \
  -e PYTHONPATH=/work -v "$PWD:/work:ro" \
  --mount "type=bind,src=$DATASET_OUTPUT,dst=/dataset" \
  -w /work railguard-web \
  python tools/generate_test_dataset.py --output /dataset
```

No GPU is required. Add `--plan-only` for schedule/storage estimates, or
`--prepare-only` to build/validate reference maps and placements without exporting.
`--bags roundT_doubleT` runs only that source; the default runs all six.

By default, maps are built with the app's merged-map code and per-bag initial
speeds. Existing reference maps can be imported with `--reference-root /references`
and an additional read-only Docker mount for that directory. It must contain
`<bag>/reference.npz` and `reference.json` with the exact source fingerprint,
topic, timestamps, speed and NPZ hash. Import rejects mismatches. The dataset's
own `.maps/` cache is reused on subsequent runs.

## Output and playback

Each `<source>_obstacles/` is a complete SQLite ROS 2 bag, retaining original
message timestamps/topics and the full recording. Non-cloud messages are copied.
The source recording is never modified. Output is losslessly Zstd-compressed at
the message level by default; use `--compression none` for uncompressed bags.

```sh
# In a ROS 2 container with the dataset mounted at /dataset:
ros2 bag info /dataset/roundT_doubleT_obstacles
ros2 bag play /dataset/roundT_doubleT_obstacles
```

`dataset.json` records source identities, code/model hashes, seed and sampled
parameters. `<source>_placements.json` records actual positions, yaw and dimensions.
Each bag has the app's `obstacle_scenario.json` (synthetic/removed counts and
near-contact omissions), `merged_rails.npz`, and `dataset_complete.json`.

The generator checks space before exporting, verifies exported cloud timestamps,
and marks the dataset complete only after all selected bags finish. Completed
bags are checked and skipped on a rerun of the same command. A changed source,
configuration or implementation requires a new output directory. Interrupted
bags stay in `.partial/`; remove only the identified incomplete bag directory
before retrying. Resume is at bag boundaries, not individual frames.

Tests (in the web image with the repository mounted at /work):

```sh
python -m unittest tests.test_dataset_generation tests.test_world_map
```
