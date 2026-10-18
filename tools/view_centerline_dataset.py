"""View exported centerline scenes in 2D.

Run from the repository root, for example:
    python3 -m tools.view_centerline_dataset eval_new_data
    python3 -m tools.view_centerline_dataset eval_new_data --frame 100 --save /tmp/scene.png

The left panel uses the current sensor's X / forward (-Y) coordinates. The
right panel places the same scene and sensor trajectory in the first frame's
coordinates.
Use the slider or left/right arrow keys to change frames; Shift+arrows move 10.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


DEFAULT_DATASET_DIR = Path("outputs/centerline_dataset_all_generated")
REQUIRED_KEYS = (
    "timestamps_ns", "sensor_pose_start", "sensor_position_start_m",
    "detected_xyz_sensor_m", "detected_offsets",
    "gt_xyz_sensor_m", "gt_offsets",
)


class CenterlineScene:
    """One bag's centerlines and sensor trajectory from an exported NPZ."""

    def __init__(self, path: Path):
        self.path = path
        with np.load(path, allow_pickle=False) as saved:
            missing = set(REQUIRED_KEYS) - set(saved.files)
            if missing:
                raise ValueError(f"{path} is missing {', '.join(sorted(missing))}")
            self.arrays = {key: saved[key] for key in REQUIRED_KEYS}

        count = len(self.arrays["timestamps_ns"])
        pose = self.arrays["sensor_pose_start"]
        position = self.arrays["sensor_position_start_m"]
        if (not count or pose.shape != (count, 4, 4)
                or position.shape != (count, 3)):
            raise ValueError(f"Invalid frame or pose arrays in {path}")
        for kind in ("detected", "gt"):
            xyz = self.arrays[f"{kind}_xyz_sensor_m"]
            offsets = self.arrays[f"{kind}_offsets"]
            if (xyz.ndim != 2 or xyz.shape[1] != 3
                    or offsets.shape != (count + 1,)
                    or offsets[0] != 0 or offsets[-1] != len(xyz)
                    or np.any(np.diff(offsets) < 0)):
                raise ValueError(f"Invalid {kind} centerline slices in {path}")

    @property
    def frame_count(self) -> int:
        return len(self.arrays["timestamps_ns"])

    def centerline(self, kind: str, frame: int) -> np.ndarray:
        if kind not in ("detected", "gt"):
            raise ValueError("kind must be 'detected' or 'gt'")
        if not 0 <= frame < self.frame_count:
            raise IndexError(frame)
        offsets = self.arrays[f"{kind}_offsets"]
        return self.arrays[f"{kind}_xyz_sensor_m"][offsets[frame]:offsets[frame + 1]]


def scene_path(name: str, dataset_dir: Path) -> Path:
    """Accept either a bag stem or a direct NPZ path."""
    path = Path(name)
    if path.suffix == ".npz" or path.is_absolute() or len(path.parts) > 1:
        return path
    return dataset_dir / f"{name}.npz"


def view(scene: CenterlineScene, initial_frame: int, forward_m: float,
         save: Path | None = None) -> None:
    if not 0 <= initial_frame < scene.frame_count:
        raise ValueError(f"Frame must be between 0 and {scene.frame_count - 1}")
    if not np.isfinite(forward_m) or forward_m <= 0:
        raise ValueError("forward_m must be positive")

    if save is not None:
        import matplotlib
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider

    fig, (local_ax, route_ax) = plt.subplots(1, 2, figsize=(13, 6))
    fig.subplots_adjust(bottom=0.18, wspace=0.28)

    local_ax.set_title("Centerlines in current sensor frame")
    local_ax.set_xlabel("Sensor X (m)")
    local_ax.set_ylabel("Forward / -Y (m)")
    local_ax.grid(alpha=0.3)
    local_ax.set_aspect("equal", adjustable="box")
    local_ax.set_ylim(-3, forward_m)
    local_ax.plot(0, 0, marker="^", color="black", markersize=8, label="Sensor")
    gt_line, = local_ax.plot([], [], color="tab:green", lw=2, label="GT (merged rails)")
    detected_line, = local_ax.plot([], [], color="tab:orange", lw=1.5,
                                   label="Detected")
    local_ax.legend(loc="upper right")

    lateral = 8.0
    for kind in ("gt", "detected"):
        xyz = scene.arrays[f"{kind}_xyz_sensor_m"]
        visible = (xyz[:, 1] <= 0) & (xyz[:, 1] >= -forward_m)
        if np.any(visible):
            lateral = max(lateral, float(np.max(np.abs(xyz[visible, 0]))) + 2)
    local_ax.set_xlim(-lateral, lateral)

    positions = scene.arrays["sensor_position_start_m"]
    route_x, route_forward = positions[:, 0], -positions[:, 1]
    route_ax.plot(route_x, route_forward, color="0.75", lw=1, label="Whole trajectory")
    past_line, = route_ax.plot([], [], color="tab:blue", lw=2, label="Travelled")
    route_gt_line, = route_ax.plot([], [], color="tab:green", lw=2,
                                   label="GT (merged rails)")
    route_detected_line, = route_ax.plot([], [], color="tab:orange", lw=1.5,
                                         label="Detected")
    sensor_dot, = route_ax.plot([], [], "o", color="tab:red", markersize=7,
                                label="Current sensor")
    heading_line, = route_ax.plot([], [], color="tab:red", lw=2)
    route_ax.set_title("Scene relative to bag start")
    route_ax.set_xlabel("Start-frame X (m)")
    route_ax.set_ylabel("Start-frame forward / -Y (m)")
    route_ax.set_aspect("equal", adjustable="box")
    route_ax.grid(alpha=0.3)
    route_ax.legend(loc="upper right")

    def draw_frame(index: int) -> None:
        pose = scene.arrays["sensor_pose_start"][index]
        scene_x = []
        scene_forward = []
        for kind, artist in (("gt", gt_line), ("detected", detected_line)):
            xyz = scene.centerline(kind, index)
            visible = (xyz[:, 1] <= 0) & (xyz[:, 1] >= -forward_m)
            artist.set_data(xyz[visible, 0], -xyz[visible, 1])
            in_start = xyz[visible] @ pose[:3, :3].T + pose[:3, 3]
            route_artist = route_gt_line if kind == "gt" else route_detected_line
            route_artist.set_data(in_start[:, 0], -in_start[:, 1])
            scene_x.extend(in_start[:, 0])
            scene_forward.extend(-in_start[:, 1])
        past_line.set_data(route_x[:index + 1], route_forward[:index + 1])
        x, y = route_x[index], route_forward[index]
        sensor_dot.set_data([x], [y])
        heading = pose[:3, :3] @ np.array([0.0, -1.0, 0.0])
        heading_line.set_data([x, x + 8 * heading[0]],
                              [y, y - 8 * heading[1]])
        x_radius = max(0.4 * forward_m, lateral,
                       max((abs(value - x) for value in scene_x), default=0) + 2)
        route_ax.set_xlim(x - x_radius, x + x_radius)
        route_ax.set_ylim(min(y - 3, min(scene_forward, default=y) - 2),
                          max(y + forward_m, max(scene_forward, default=y) + 2))
        elapsed_s = (int(scene.arrays["timestamps_ns"][index])
                     - int(scene.arrays["timestamps_ns"][0])) / 1e9
        fig.suptitle(f"{scene.path.stem}  |  frame {index + 1}/{scene.frame_count}"
                     f"  |  t = {elapsed_s:.2f} s"
                     f"  |  position = ({positions[index, 0]:.1f},"
                     f" {positions[index, 1]:.1f}, {positions[index, 2]:.1f}) m")
        fig.canvas.draw_idle()

    draw_frame(initial_frame)
    if save is not None:
        # if save.exists():
        #     raise FileExistsError(f"Image already exists: {save}")
        fig.savefig(save, dpi=150)
        plt.close(fig)
        return

    if scene.frame_count > 1:
        slider_ax = fig.add_axes((0.13, 0.07, 0.74, 0.035))
        slider = Slider(slider_ax, "Frame", 0, scene.frame_count - 1,
                        valinit=initial_frame, valstep=1)
        slider.on_changed(lambda value: draw_frame(int(value)))

        def on_key(event) -> None:
            step = {"left": -1, "right": 1, "shift+left": -10,
                    "shift+right": 10}.get(event.key)
            if step is not None:
                slider.set_val(max(0, min(scene.frame_count - 1,
                                          int(slider.val) + step)))
            elif event.key == "home":
                slider.set_val(0)
            elif event.key == "end":
                slider.set_val(scene.frame_count - 1)

        fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", nargs="?", default="eval_new_data",
                        help="Bag name in dataset dir or direct NPZ path")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--forward-m", type=float, default=150.0)
    parser.add_argument("--save", type=Path, help="Save this frame to a PNG instead of opening a window")
    args = parser.parse_args()
    view(CenterlineScene(scene_path(args.scene, args.dataset_dir)),
         args.frame, args.forward_m, args.save)


if __name__ == "__main__":
    main()
