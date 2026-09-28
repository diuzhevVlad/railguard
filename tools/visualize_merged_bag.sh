#!/usr/bin/env bash
# Build a merged rail map, create a compact Rerun recording, and open it.
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 BAG_NAME [INITIAL_SPEED_M_S] (for example: doubleT_platform 5)" >&2
  exit 2
fi

repo_dir="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
bag_name="$1"
initial_speed="${2:-0}"
speed_label="$(python3 -c 'import math, sys; speed=float(sys.argv[1]); assert math.isfinite(speed); print(f"{speed:g}")' "$initial_speed")"
speed_suffix=""
if [[ "$speed_label" != "0" ]]; then
  speed_suffix="_speed_${speed_label}ms"
fi
bag_dir="$repo_dir/data/$bag_name"
viz_dir="$repo_dir/outputs/rail_visualizations"
map_path="$viz_dir/${bag_name}${speed_suffix}_merged_rails.npz"
recording_path="$viz_dir/${bag_name}${speed_suffix}_merged_rails.rrd"

if [[ ! -f "$bag_dir/metadata.yaml" ]]; then
  echo "No ROS 2 bag found at $bag_dir" >&2
  exit 2
fi
mkdir -p "$viz_dir"
cd "$repo_dir"

if [[ ! -f "$map_path" ]]; then
  docker compose build web
  docker run --rm --user "$(id -u):$(id -g)" \
    -v "$repo_dir/data:/data:ro" \
    -v "$repo_dir/tools:/app/tools:ro" \
    -v "$viz_dir:/viz" \
    railguard-web \
    python -m tools.build_merged_map "/data/$bag_name" "/viz/$(basename "$map_path")" \
    --initial-speed "$initial_speed"
fi

if [[ ! -f "$map_path" ]]; then
  echo "No merged rail map was produced for $bag_name" >&2
  exit 1
fi

if [[ ! -f "$recording_path" || "$map_path" -nt "$recording_path" ]]; then
  topic_args=()
  if [[ "$bag_name" == "doubleT_obstacle" ]]; then
    topic_args=(--topic /sensing/lidar/hesai128/pointcloud)
  fi
  python3 tools/view_merged_rails_rerun.py "$bag_dir" "$map_path" \
    --output "$recording_path" "${topic_args[@]}"
fi
rerun "$recording_path"
