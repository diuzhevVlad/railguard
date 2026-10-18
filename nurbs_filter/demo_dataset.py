"""Run the rolling NURBS filter on exported centerline scenes and score it.

Examples:
    python demo_dataset.py eval_new_data
    python demo_dataset.py eval_new_data --interactive
    python demo_dataset.py all --output-dir dataset_demo_output

Each scene writes a summary JSON, frame CSV, and diagnostic PNG. Ground truth
is used only for scoring; the filter receives detections and saved poses.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

if __package__:
    from .forward_filter import (detector_lateral_change,
                                 enforce_forward_reach,
                                 initialize_forward_window,
                                 sample_forward_curve)
    from .rolling_nurbs import Noise, RollingNURBSKF
else:
    from forward_filter import (detector_lateral_change,
                                enforce_forward_reach,
                                initialize_forward_window,
                                sample_forward_curve)
    from rolling_nurbs import Noise, RollingNURBSKF


DEFAULT_DATASET = Path("centerline_dataset_all_generated")


def closest_polyline(points: np.ndarray, line: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project points to an ordered 2-D polyline; flag projections at its ends."""
    if len(line) < 2:
        return np.full_like(points, np.nan), np.zeros(len(points), bool)
    segment = np.diff(line, axis=0)
    _, vertex = cKDTree(line).query(points)
    candidates = np.stack([np.clip(vertex - 1, 0, len(segment) - 1),
                           np.clip(vertex, 0, len(segment) - 1)], axis=1)
    start = line[candidates]
    direction = segment[candidates]
    length2 = np.sum(direction**2, axis=2)
    t = np.clip(np.sum((points[:, None] - start)*direction, axis=2) /
                np.maximum(length2, 1e-12), 0., 1.)
    projected = start + t[:, :, None]*direction
    choice = np.argmin(np.sum((points[:, None] - projected)**2, axis=2), axis=1)
    row = np.arange(len(points))
    indices, fractions = candidates[row, choice], t[row, choice]
    interior = (length2[row, choice] > 1e-12) & ~(
        ((indices == 0) & (fractions < 1e-6)) |
        ((indices == len(segment) - 1) & (fractions > 1 - 1e-6)))
    return projected[row, choice], interior


def truth_anchors(xyz: np.ndarray, distance: np.ndarray,
                  max_range: float, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """Evenly spaced GT locations and their 2-D normals."""
    if len(xyz) < 3 or len(distance) != len(xyz):
        return np.empty((0, 2)), np.empty((0, 2))
    good = np.isfinite(distance) & np.isfinite(xyz[:, :2]).all(axis=1)
    xyz, distance = xyz[good], distance[good]
    if len(xyz) < 3 or np.any(np.diff(distance) <= 0):
        return np.empty((0, 2)), np.empty((0, 2))
    targets = np.arange(spacing, min(max_range, distance[-1] - spacing) + 1e-9, spacing)
    if not len(targets):
        return np.empty((0, 2)), np.empty((0, 2))
    xy = np.column_stack([np.interp(targets, distance, xyz[:, axis]) for axis in range(2)])
    seg = np.clip(np.searchsorted(distance, targets, side="right") - 1, 0, len(xyz) - 2)
    tangent = xyz[seg + 1, :2] - xyz[seg, :2]
    length = np.linalg.norm(tangent, axis=1)
    valid = length > 1e-8
    tangent = tangent[valid] / length[valid, None]
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    return xy[valid], normal


def score_frame(gt: np.ndarray, normal: np.ndarray, detected: np.ndarray,
                filtered: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, int]:
    """Signed lateral errors on common GT anchors; return total GT count too."""
    if len(gt) == 0 or len(detected) < 2 or filtered is None or len(filtered) < 2:
        return np.empty(0), np.empty(0), len(gt)
    det_closest, det_ok = closest_polyline(gt, detected)
    filt_closest, filt_ok = closest_polyline(gt, filtered)
    common = det_ok & filt_ok
    detector_error = np.sum((det_closest[common] - gt[common])*normal[common], axis=1)
    filter_error = np.sum((filt_closest[common] - gt[common])*normal[common], axis=1)
    return detector_error, filter_error, len(gt)


def planar_increment(previous: np.ndarray, current: np.ndarray) -> tuple[np.ndarray, float]:
    """Map the saved 3-D relative pose to the filter's planar odometry input."""
    relative_rotation = previous[:3, :3].T @ current[:3, :3]
    translation = previous[:3, :3].T @ (current[:3, 3] - previous[:3, 3])
    yaw = float(np.arctan2(relative_rotation[1, 0], relative_rotation[0, 0]))
    return translation[:2], yaw


def plot_scene(path: Path, rows: list[dict], last: dict) -> None:
    import matplotlib.pyplot as plt

    fig, (scene_ax, metric_ax) = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for label, value, color in (("Ground truth", last["gt"], "tab:green"),
                                ("Detection", last["detected"], "tab:orange"),
                                ("Filtered", last["filtered"], "tab:blue")):
        if value is not None and len(value):
            scene_ax.plot(value[:, 0], -value[:, 1], label=label, color=color, lw=1.5)
    if last["controls"] is not None:
        cp = last["controls"]
        scene_ax.plot(cp[:, 0], -cp[:, 1], "o--", ms=3, lw=.8,
                      color="tab:purple", label="Control points")
    scene_ax.plot(0, 0, "k^", label="Sensor")
    scene_ax.set(xlabel="Sensor X [m]", ylabel="Forward / -Y [m]",
                 title=f"{last['name']}, frame {last['frame']}")
    # Forward span is much longer than the lateral errors of interest.
    # Exaggerate lateral scale so curve differences remain visible.
    scene_ax.set_aspect("auto")
    reference_x = last["gt"][:, 0]
    if last["filtered"] is not None:
        reference_x = np.r_[reference_x, last["filtered"][:, 0]]
    if len(reference_x):
        midpoint = .5*(np.min(reference_x) + np.max(reference_x))
        half_width = max(8., .5*np.ptp(reference_x) + 2.)
        scene_ax.set_xlim(midpoint - half_width, midpoint + half_width)
    scene_ax.set_ylim(-15, last["max_forward_m"])
    scene_ax.grid(alpha=.3)
    scene_ax.legend(fontsize=8)
    x = [row["frame"] for row in rows]
    metric_ax.plot(x, [row["detector_rmse_m"] for row in rows],
                   label="Detection", color="tab:orange")
    metric_ax.plot(x, [row["filter_rmse_m"] for row in rows],
                   label="Filtered", color="tab:blue")
    metric_ax.set(xlabel="Frame", ylabel="Lateral RMSE [m]",
                  title="Errors on common ground truth anchors")
    metric_ax.grid(alpha=.3)
    metric_ax.legend()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def show_interactive(path: Path, arrays: dict, rows: list[dict],
                     filtered_history: list[np.ndarray | None], max_forward_m: float,
                     control_history: list[np.ndarray | None],
                     initial_frame: int = 0, show: bool = True):
    """Browse filtered scenes with a slider, keyboard, and playback controls."""
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, Slider

    if not 0 <= initial_frame < len(rows):
        raise ValueError(f"Frame must be between 0 and {len(rows) - 1}")
    det = arrays["detected_xyz_sensor_m"]
    det_o = arrays["detected_offsets"]
    gt = arrays["gt_xyz_sensor_m"]
    gt_o = arrays["gt_offsets"]
    timestamps = arrays["timestamps_ns"]

    fig, (scene_ax, metric_ax) = plt.subplots(1, 2, figsize=(14, 6))
    fig.subplots_adjust(bottom=.20, top=.88, wspace=.28)
    gt_line, = scene_ax.plot([], [], lw=2, color="tab:green", label="Ground truth")
    det_line, = scene_ax.plot([], [], lw=1.4, color="tab:orange", label="Detection")
    filt_line, = scene_ax.plot([], [], lw=2, color="tab:blue", label="Filtered")
    cp_line, = scene_ax.plot([], [], "o--", ms=3, lw=.8,
                             color="tab:purple", label="Control points")
    scene_ax.plot(0, 0, "k^", ms=7, label="Sensor")
    scene_ax.set(xlabel="Sensor X [m]", ylabel="Forward / -Y [m]",
                 ylim=(-15, max_forward_m), title="Current sensor frame")
    scene_ax.grid(alpha=.3)
    scene_ax.legend(loc="upper right", fontsize=8)
    frames = np.arange(len(rows))
    metric_ax.plot(frames, [row["detector_rmse_m"] for row in rows],
                   lw=1.5, color="tab:orange", label="Detection")
    metric_ax.plot(frames, [row["filter_rmse_m"] for row in rows],
                   lw=1.5, color="tab:blue", label="Filtered")
    cursor = metric_ax.axvline(initial_frame, color="black", lw=1, alpha=.7)
    metric_ax.set(xlabel="Frame (click to jump)", ylabel="Lateral RMSE [m]",
                  title="Error on common ground truth anchors")
    metric_ax.grid(alpha=.3)
    metric_ax.legend(fontsize=8)

    slider_ax = fig.add_axes((.16, .09, .62, .035))
    slider = Slider(slider_ax, "Frame", 0, len(rows) - 1,
                    valinit=initial_frame, valstep=1)
    back_ax = fig.add_axes((.16, .025, .075, .045))
    play_ax = fig.add_axes((.245, .025, .075, .045))
    next_ax = fig.add_axes((.33, .025, .075, .045))
    zoom_ax = fig.add_axes((.415, .025, .075, .045))
    back_button = Button(back_ax, "Prev")
    play_button = Button(play_ax, "Play")
    next_button = Button(next_ax, "Next")
    zoom_button = Button(zoom_ax, "Near")
    timer = fig.canvas.new_timer(interval=120)
    playing = False
    near_view = False

    def draw(index: int) -> None:
        da, db = det_o[index:index + 2]
        ga, gb = gt_o[index:index + 2]
        full_detected = det[da:db, :2]
        detected = full_detected[-full_detected[:, 1] <= max_forward_m]
        truth = gt[ga:gb, :2]
        filtered = filtered_history[index]
        controls = control_history[index]
        for artist, value in ((gt_line, truth), (det_line, detected),
                              (filt_line, filtered), (cp_line, controls)):
            if value is None or not len(value):
                artist.set_data([], [])
            else:
                artist.set_data(value[:, 0], -value[:, 1])
        visible_forward = min(50., max_forward_m) if near_view else max_forward_m
        scene_ax.set_ylim(-15, visible_forward)
        reference_x = np.empty(0)
        for value in (truth, filtered, controls):
            if value is not None and len(value):
                visible = (-value[:, 1] >= -15) & (-value[:, 1] <= visible_forward)
                reference_x = np.r_[reference_x, value[visible, 0]]
        if len(reference_x):
            midpoint = .5*(float(np.min(reference_x)) + float(np.max(reference_x)))
            half_width = max(8., .5*float(np.ptp(reference_x)) + 2.)
            scene_ax.set_xlim(midpoint - half_width, midpoint + half_width)
        cursor.set_xdata([index, index])
        row = rows[index]
        elapsed = (int(timestamps[index]) - int(timestamps[0])) / 1e9
        def metric(value: float) -> str:
            return f"{value:.3f} m" if np.isfinite(value) else "n/a"
        coverage = f"{row['coverage']:.0%}" if np.isfinite(row["coverage"]) else "n/a"
        fig.suptitle(f"{path.stem}  |  frame {index + 1}/{len(rows)}  |  t={elapsed:.1f} s\n"
                     f"detector {metric(row['detector_rmse_m'])}  ·  "
                     f"filtered {metric(row['filter_rmse_m'])}  ·  "
                     f"reach {metric(row['forward_reach_m'])}  ·  "
                     f"coverage {coverage}  ·  accepted {row['accepted']}/{row['measurement_points']}"
                     f"  ·  path {row['path_accepted']}"
                     f"  ·  {row['status']}")
        fig.canvas.draw_idle()

    slider.on_changed(lambda value: draw(int(value)))
    back_button.on_clicked(lambda _event: slider.set_val(max(0, int(slider.val) - 1)))
    next_button.on_clicked(lambda _event: slider.set_val(min(len(rows) - 1, int(slider.val) + 1)))

    def toggle_play(_event=None) -> None:
        nonlocal playing
        playing = not playing
        play_button.label.set_text("Pause" if playing else "Play")
        if playing:
            timer.start()
        else:
            timer.stop()
        fig.canvas.draw_idle()

    def advance() -> None:
        slider.set_val((int(slider.val) + 1) % len(rows))

    timer.add_callback(advance)
    play_button.on_clicked(toggle_play)

    def toggle_zoom(_event=None) -> None:
        nonlocal near_view
        near_view = not near_view
        zoom_button.label.set_text("Full" if near_view else "Near")
        draw(int(slider.val))

    zoom_button.on_clicked(toggle_zoom)

    def on_key(event) -> None:
        step = {"left": -1, "right": 1, "shift+left": -10,
                "shift+right": 10}.get(event.key)
        if step is not None:
            slider.set_val(max(0, min(len(rows) - 1, int(slider.val) + step)))
        elif event.key == "home":
            slider.set_val(0)
        elif event.key == "end":
            slider.set_val(len(rows) - 1)
        elif event.key in (" ", "space"):
            toggle_play()

    def on_click(event) -> None:
        if event.inaxes == metric_ax and event.xdata is not None:
            slider.set_val(max(0, min(len(rows) - 1, round(event.xdata))))

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("close_event", lambda _event: timer.stop())
    fig._demo_controls = dict(slider=slider, back=back_button, play=play_button,
                              next=next_button, zoom=zoom_button, timer=timer)
    draw(initial_frame)
    if show:
        plt.show()
    return fig


def run_scene(path: Path, noise: Noise, args: argparse.Namespace) -> dict:
    with np.load(path, allow_pickle=False) as data:
        required = ("timestamps_ns", "sensor_pose_start", "detected_xyz_sensor_m",
                    "detected_offsets", "gt_xyz_sensor_m",
                    "gt_distance_along_m", "gt_offsets")
        missing = set(required) - set(data.files)
        if missing:
            raise ValueError(f"{path}: missing {', '.join(sorted(missing))}")
        arrays = {key: data[key] for key in required}
    timestamps, poses = arrays["timestamps_ns"], arrays["sensor_pose_start"]
    det, det_o = (arrays[key] for key in ("detected_xyz_sensor_m", "detected_offsets"))
    gt, gt_s, gt_o = (arrays[key] for key in
                      ("gt_xyz_sensor_m", "gt_distance_along_m", "gt_offsets"))
    if (poses.shape != (len(timestamps), 4, 4) or len(det_o) != len(timestamps) + 1 or
            len(gt_o) != len(timestamps) + 1 or det_o[-1] != len(det) or gt_o[-1] != len(gt)):
        raise ValueError(f"{path}: inconsistent frame arrays")
    frame_count = min(len(timestamps), args.max_frames) if args.max_frames else len(timestamps)
    if args.interactive and not 0 <= args.frame < frame_count:
        raise ValueError(f"--frame must be between 0 and {frame_count - 1}")
    measurement_targets = np.arange(args.measurement_spacing_m,
                                    args.max_forward_m + 1e-9, args.measurement_spacing_m)
    recovery_targets = np.arange(10., args.max_forward_m-5., 5.)
    rows: list[dict] = []
    filtered_history = [] if args.interactive else None
    control_history = [] if args.interactive else None
    detector_errors, filter_errors = [], []
    filter_state = None
    last = None
    resets = 0
    low_accept_streak = 0
    for frame in range(frame_count):
        da, db = det_o[frame:frame + 2]
        ga, gb = gt_o[frame:frame + 2]
        visible = -det[da:db, 1] <= args.max_forward_m
        detected = np.asarray(det[da:db, :2][visible], float)
        measured = sample_forward_curve(det[da:db], measurement_targets)
        measured = measured[np.isfinite(measured).all(axis=1)]
        truth = np.asarray(gt[ga:gb, :2], float)
        anchor, normal = truth_anchors(gt[ga:gb], gt_s[ga:gb],
                                       args.max_forward_m, args.metric_spacing_m)
        accepted, rolled, reset, status = 0, 0, False, "no_measurement"
        endpoint_correction = 0.
        path_accepted, path_residual = 0, np.nan
        if len(measured) >= args.controls - args.behind_spans:
            if filter_state is None:
                filter_state = initialize_forward_window(measured, args.controls,
                                                         args.behind_spans, noise)
                status = "initialized"
            else:
                translation, yaw = planar_increment(poses[frame - 1], poses[frame])
                dt = max(0., (int(timestamps[frame]) - int(timestamps[frame - 1])) / 1e9)
                filter_state.predict(translation, yaw, dt=dt)
                try:
                    rolled = filter_state.recenter(anchor_fraction=args.anchor_fraction)
                    endpoint_correction += enforce_forward_reach(
                        filter_state, args.max_forward_m + .01)
                    update = filter_state.update_points(measured, max_distance=args.max_distance_m,
                                                        gate=args.gate,
                                                        robust_threshold=(args.robust_threshold_m or None))
                    accepted = update["accepted"]
                    status = "updated" if accepted else "no_update"
                    if accepted < args.reinit_accept_fraction*len(measured):
                        low_accept_streak += 1
                    else:
                        low_accept_streak = 0
                    stable_detector = False
                    if (args.reinit_accept_fraction > 0 and
                            low_accept_streak >= args.reinit_patience and frame >= 5):
                        stable_detector = True
                        for lag, limit in ((1, .5), (3, 2.5), (5, 3.)):
                            old_a, old_b = det_o[frame-lag:frame-lag+2]
                            change = detector_lateral_change(
                                det[da:db], det[old_a:old_b],
                                poses[frame], poses[frame-lag], recovery_targets)
                            if change >= limit:
                                stable_detector = False
                                break
                    if stable_detector:
                        filter_state = initialize_forward_window(measured, args.controls,
                                                                 args.behind_spans, noise)
                        resets += 1
                        low_accept_streak = 0
                        endpoint_correction = 0.
                        accepted, rolled = 0, 0
                        reset, status = True, "reset_low_accept"
                except (RuntimeError, np.linalg.LinAlgError):
                    filter_state = initialize_forward_window(measured, args.controls,
                                                             args.behind_spans, noise)
                    resets += 1
                    endpoint_correction = 0.
                    low_accept_streak = 0
                    reset, status = True, "reset"
        elif filter_state is not None:
            low_accept_streak = 0
            translation, yaw = planar_increment(poses[frame - 1], poses[frame])
            dt = max(0., (int(timestamps[frame]) - int(timestamps[frame - 1])) / 1e9)
            filter_state.predict(translation, yaw, dt=dt)
        filtered = None
        if filter_state is not None:
            if not args.no_path_measurement:
                path_update = filter_state.update_path_position(
                    lateral_sigma=args.path_lateral_sigma_m)
                path_accepted = path_update["accepted"]
                path_residual = path_update["normal_residual"]
            endpoint_correction += enforce_forward_reach(
                filter_state, args.max_forward_m + .01)
            filtered = filter_state.evaluate(np.linspace(*filter_state.domain, 500))
        if filtered_history is not None:
            filtered_history.append(filtered)
            control_history.append(filter_state.cp.copy() if filter_state is not None else None)
        forward_reach = float(-filtered[-1, 1]) if filtered is not None else np.nan
        det_error, filt_error, gt_count = score_frame(anchor, normal, detected, filtered)
        detector_errors.extend(det_error.tolist())
        filter_errors.extend(filt_error.tolist())
        matched = len(det_error)
        row = dict(frame=frame, timestamp_ns=int(timestamps[frame]), status=status,
                   measurement_points=len(measured), accepted=accepted,
                   path_accepted=path_accepted, path_normal_residual_m=path_residual,
                   rolled=rolled, reset=int(reset),
                   forward_reach_m=forward_reach,
                   endpoint_correction_m=endpoint_correction,
                   gt_anchors=gt_count, common_anchors=matched,
                   coverage=matched / gt_count if gt_count else np.nan,
                   detector_rmse_m=float(np.sqrt(np.mean(det_error**2))) if matched else np.nan,
                   filter_rmse_m=float(np.sqrt(np.mean(filt_error**2))) if matched else np.nan,
                   detector_mae_m=float(np.mean(np.abs(det_error))) if matched else np.nan,
                   filter_mae_m=float(np.mean(np.abs(filt_error))) if matched else np.nan)
        rows.append(row)
        last = dict(name=path.stem, frame=frame, gt=truth, detected=detected,
                    filtered=filtered,
                    controls=filter_state.cp.copy() if filter_state is not None else None,
                    max_forward_m=args.max_forward_m)
    def aggregate(values: list[float]) -> dict:
        value = np.asarray(values)
        return dict(rmse_m=float(np.sqrt(np.mean(value**2))) if len(value) else None,
                    mae_m=float(np.mean(np.abs(value))) if len(value) else None,
                    p95_abs_m=float(np.percentile(np.abs(value), 95)) if len(value) else None)

    update_rows = [row for row in rows if row["status"] in ("updated", "no_update")]
    update_measurements = sum(row["measurement_points"] for row in update_rows)
    reaches = np.asarray([row["forward_reach_m"] for row in rows])
    reaches = reaches[np.isfinite(reaches)]
    summary = dict(scene=path.stem, frames=frame_count, reset_count=resets,
                   min_forward_reach_m=float(np.min(reaches)) if len(reaches) else None,
                   median_forward_reach_m=float(np.median(reaches)) if len(reaches) else None,
                   max_endpoint_correction_m=max((row["endpoint_correction_m"] for row in rows), default=0.),
                   endpoint_constrained_frames=sum(row["endpoint_correction_m"] > 0 for row in rows),
                   updated_frames=sum(row["status"] == "updated" for row in rows),
                   path_accepted_frames=sum(row["path_accepted"] for row in rows),
                   accepted_measurements=sum(row["accepted"] for row in update_rows),
                   update_measurements=update_measurements,
                   accepted_fraction=(sum(row["accepted"] for row in update_rows) /
                                      update_measurements if update_measurements else None),
                   total_rolls=sum(abs(row["rolled"]) for row in rows),
                   matched_anchors=len(filter_errors),
                   gt_anchors=int(sum(row["gt_anchors"] for row in rows)),
                   coverage=(len(filter_errors) / sum(row["gt_anchors"] for row in rows)
                             if any(row["gt_anchors"] for row in rows) else None),
                   detector=aggregate(detector_errors), filtered=aggregate(filter_errors),
                   settings=dict(max_forward_m=args.max_forward_m,
                                 measurement_spacing_m=args.measurement_spacing_m,
                                 metric_spacing_m=args.metric_spacing_m,
                                 controls=args.controls, behind_spans=args.behind_spans,
                                 max_distance_m=args.max_distance_m, gate=args.gate,
                                 robust_threshold_m=args.robust_threshold_m,
                                 reinit_accept_fraction=args.reinit_accept_fraction,
                                 reinit_patience=args.reinit_patience,
                                 path_lateral_sigma_m=(None if args.no_path_measurement else
                                                       args.path_lateral_sigma_m)),
                   note="Detector and filter errors use the same GT anchors in sensor XY; "
                        "poses are treated as exact planar odometry for prediction. "
                        + ("Each new sensor position is a soft curve-normal observation. "
                           if not args.no_path_measurement else "")
                        + "The far endpoint is constrained to at least the requested forward range.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir / path.stem
    with prefix.with_suffix(".csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    prefix.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_scene(prefix.with_suffix(".png"), rows, last)
    if filtered_history is not None:
        show_interactive(path, arrays, rows, filtered_history,
                         args.max_forward_m, control_history, args.frame)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", nargs="?", default="eval_new_data",
                        help="Bag stem, NPZ path, or 'all'")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--noise-json", type=Path, default=Path("detector_noise.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("dataset_demo_output_"))
    parser.add_argument("--max-forward-m", type=float, default=150.)
    parser.add_argument("--measurement-spacing-m", type=float, default=2.)
    parser.add_argument("--metric-spacing-m", type=float, default=2.)
    parser.add_argument("--controls", type=int, default=24)
    parser.add_argument("--behind-spans", type=int, default=3)
    parser.add_argument("--anchor-fraction", type=float, default=.05)
    parser.add_argument("--max-distance-m", type=float, default=20.)
    parser.add_argument("--gate", type=float, default=15.)
    parser.add_argument("--robust-threshold-m", type=float, default=.2,
                        help="Add variance to detector residuals above this size; 0 disables")
    parser.add_argument("--reinit-accept-fraction", type=float, default=.8,
                        help="Reinitialize after repeated frames below this association fraction; 0 disables")
    parser.add_argument("--reinit-patience", type=int, default=3)
    parser.add_argument("--path-lateral-sigma-m", type=float, default=.05,
                        help="Allowed lateral sensor-to-curve offset (standard deviation)")
    parser.add_argument("--no-path-measurement", action="store_true",
                        help="Disable the soft sensor-position observation")
    parser.add_argument("--max-frames", type=int, default=0, help="Limit frames for a quick run")
    parser.add_argument("--interactive", action="store_true",
                        help="Open a frame browser after processing one scene")
    parser.add_argument("--frame", type=int, default=0, help="Initial frame in interactive mode")
    args = parser.parse_args()
    if (args.max_forward_m <= 0 or args.measurement_spacing_m <= 0 or
            args.metric_spacing_m <= 0 or args.max_distance_m <= 0 or args.gate <= 0 or
            not np.isfinite(args.robust_threshold_m) or args.robust_threshold_m < 0 or
            not 0 <= args.reinit_accept_fraction <= 1 or args.reinit_patience < 1 or
            not np.isfinite(args.path_lateral_sigma_m) or args.path_lateral_sigma_m <= 0 or
            args.controls < args.behind_spans + 6 or args.behind_spans < 1 or
            not 0 < args.anchor_fraction < 1 or args.max_frames < 0):
        parser.error("Invalid range, spacing, control, gate, or frame option")
    if args.scene == "all":
        paths = sorted(args.dataset_dir.glob("*.npz"))
    else:
        direct = Path(args.scene)
        paths = [direct if direct.suffix == ".npz" or len(direct.parts) > 1
                 else args.dataset_dir / f"{args.scene}.npz"]
    if not paths or any(not path.is_file() for path in paths):
        parser.error("No matching dataset NPZ files found")
    if args.interactive and len(paths) != 1:
        parser.error("--interactive requires one scene; choose a bag stem or NPZ path")
    if args.interactive and (args.frame < 0 or (args.max_frames and args.frame >= args.max_frames)):
        parser.error("--frame must be within the processed frame range")
    import matplotlib
    if not args.interactive:
        matplotlib.use("Agg", force=True)
    elif matplotlib.get_backend().lower() in {"agg", "pdf", "svg", "ps", "pgf", "cairo", "template"}:
        parser.error("--interactive needs a graphical Matplotlib backend; set MPLBACKEND=MacOSX or QtAgg")
    noise = Noise(**json.loads(args.noise_json.read_text())["noise"])
    summaries = []
    for path in paths:
        result = run_scene(path, noise, args)
        summaries.append(result)
        baseline, filtered = result["detector"]["rmse_m"], result["filtered"]["rmse_m"]
        coverage = result["coverage"]
        metric_text = (f"detector RMSE={baseline:.3f} m, filtered RMSE={filtered:.3f} m"
                       if baseline is not None and filtered is not None else "no common scored anchors")
        coverage_text = f"{coverage:.1%}" if coverage is not None else "n/a"
        print(f"{path.stem}: frames={result['frames']}, coverage={coverage_text}, "
              f"{metric_text}, "
              f"resets={result['reset_count']}")
    if len(summaries) > 1:
        matched = sum(item["matched_anchors"] for item in summaries)
        gt_count = sum(item["gt_anchors"] for item in summaries)
        def pooled(kind: str, metric: str) -> float | None:
            if not matched:
                return None
            values = [(item["matched_anchors"], item[kind][metric]) for item in summaries
                      if item["matched_anchors"] and item[kind][metric] is not None]
            if metric == "rmse_m":
                return float(np.sqrt(sum(n*value**2 for n, value in values) / matched))
            return float(sum(n*value for n, value in values) / matched)
        overall = dict(scenes=len(summaries), frames=sum(item["frames"] for item in summaries),
                       matched_anchors=matched, gt_anchors=gt_count,
                       coverage=matched / gt_count if gt_count else None,
                       reset_count=sum(item["reset_count"] for item in summaries),
                       endpoint_constrained_frames=sum(item["endpoint_constrained_frames"] for item in summaries),
                       updated_frames=sum(item["updated_frames"] for item in summaries),
                       path_accepted_frames=sum(item["path_accepted_frames"] for item in summaries),
                       accepted_measurements=sum(item["accepted_measurements"] for item in summaries),
                       detector=dict(rmse_m=pooled("detector", "rmse_m"),
                                     mae_m=pooled("detector", "mae_m")),
                       filtered=dict(rmse_m=pooled("filtered", "rmse_m"),
                                     mae_m=pooled("filtered", "mae_m")))
        (args.output_dir / "summary.json").write_text(
            json.dumps(dict(overall=overall, scenes=summaries), indent=2) + "\n")
        print(f"Overall: coverage={overall['coverage']:.1%}, "
              f"detector RMSE={overall['detector']['rmse_m']:.3f} m, "
              f"filtered RMSE={overall['filtered']['rmse_m']:.3f} m")
    print(f"Wrote metrics and plots to {args.output_dir}")


if __name__ == "__main__":
    main()
