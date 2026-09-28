# View the LiDAR bag in Rerun

## Use `ros2 bag play` and subscribe live

Run these commands from this repository root after sourcing your local ROS 2
installation. Install `requirements-rerun.txt` into the Python environment used
by the viewer. The direct-bag workflow below does not require ROS.

Terminal 1:

```bash
ros2 bag play data/new_data
```

Terminal 2, from the repository root:

```bash
python3 tools/view_ros_topic_rerun.py --topic /lidar_points
```

The app subscribes to the ROS 2 topic and opens Rerun. If playback started
before the app, it shows new frames from the moment it connects. To see the
start of the bag, launch the viewer before playback or replay the bag. For a
short trial, add `--max-frames 100` to the viewer command. The viewer uses
best-effort, volatile QoS and accepts this bag's reliable publisher. It uses
arrival time in Rerun because the point cloud's own header stamp differs from
the bag timestamp. The full `new_data` bag has 221 SQLite segments, so `ros2
bag play` may spend time indexing them before publishing.

The smaller obstacle bag has a different topic. To view it, use:

```bash
ros2 bag play data/doubleT_obstacle
```

```bash
python3 tools/view_ros_topic_rerun.py \
  --topic /sensing/lidar/hesai128/pointcloud
```

## Read the bag directly, without ROS

The bag at `data/new_data/` contains `/lidar_points` (`sensor_msgs/msg/PointCloud2`).
`tools/view_bag_rerun.py` reads the SQLite bag directly, replays its timestamps,
and streams each scan to Rerun. ROS 2 does not need to be installed.

Install dependencies if needed:

```bash
python3 -m pip install -r requirements-rerun.txt
```

From the repository root, preview the first 30 seconds:

```bash
python3 tools/view_bag_rerun.py data/new_data --duration 30
```

The Rerun viewer opens automatically. The default point cap is 80,000 per
frame; increase it for detail or lower it if rendering is slow:

```bash
python3 tools/view_bag_rerun.py data/new_data --start 120 --duration 30 --max-points 150000
```

To produce a portable Rerun recording without opening the viewer:

```bash
python3 tools/view_bag_rerun.py data/new_data --duration 10 --speed 0 --output lidar_preview.rrd
rerun lidar_preview.rrd
```

The data is displayed in its original `hesai_lidar` sensor coordinates. The bag
contains no TF or odometry, so scans are not accumulated into a world map.
Many source points are `(0,0,0)` placeholders; the reader filters these using
`--min-range 0.3`. The whole bag lasts about 20 minutes and is large; use
`--start`, `--duration`, and `--max-frames` to inspect shorter sections.

## View the merged map and segmented rails

The obstacle app's exports include `merged_rails.npz`, containing the KISS-ICP
poses and rail centerline used for obstacle placement. The following command
uses the matching original bag to accumulate a downsampled world point cloud
and overlay the detected rails in Rerun:

```bash
python3 tools/view_merged_rails_rerun.py \
  data/doubleT_platform \
  outputs/person_standing_20260924_115124_c281a5/merged_rails.npz \
  --output outputs/rail_visualizations/doubleT_platform_merged_rails.rrd
rerun outputs/rail_visualizations/doubleT_platform_merged_rails.rrd
```

Gray shows accumulated LiDAR returns; red and blue show left and right rail
detections and their paths; yellow is the merged centerline used for placement;
green shows the estimated LiDAR trajectory. The `.rrd` is a static, downsampled
map (500,000 background points by default), so opening it does not replay the
full bag. Use a matching source bag and export map; timestamps are checked.

For another source bag, use the helper below from the repository root. It
generates a matching merged map in Docker, produces a compact `.rrd` with the
locally installed Rerun SDK, and opens it. It reuses the files on subsequent
runs.

```bash
./tools/visualize_merged_bag.sh roundT_doubleT
./tools/visualize_merged_bag.sh roundT_pressureGate_roundT
./tools/visualize_merged_bag.sh roundT_squareT_pressureGate_squareT
./tools/visualize_merged_bag.sh squareT_platform_squareT_switch
./tools/visualize_merged_bag.sh doubleT_obstacle
./tools/visualize_merged_bag.sh new_data
```

Run one command at a time and close its viewer before starting the next.
`new_data` is much larger (84 GB, 11,271 LiDAR frames), so generating its map
will take substantially longer. `doubleT_obstacle` uses its Hesai topic
automatically. Recordings are saved as
`outputs/rail_visualizations/<bag_name>_merged_rails.rrd`.

To seed KISS-ICP with an initial forward speed of 5 m/s, pass a second
argument:

```bash
./tools/visualize_merged_bag.sh doubleT_platform 5
```

With the default 0.1 s scan interval, this seeds a 0.5 m step along -Y for
the second scan only; KISS-ICP estimates subsequent motion. The command saves
`doubleT_platform_speed_5ms_merged_rails.rrd` separately from the zero-speed
recording. The same second argument works for other bags.
