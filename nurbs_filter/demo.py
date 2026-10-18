"""Sensor follows the true curve; detections cover only 10 metres ahead.

python demo.py                  # writes demo.png, prints error statistics
python demo.py --animate        # also opens a matplotlib animation
"""
import argparse
import numpy as np
from scipy.interpolate import CubicSpline
if __package__:
    from .rolling_nurbs import RollingNURBSKF, Noise, rotation, sample_curve
    from .forward_filter import initialize_forward_window
else:
    from rolling_nurbs import RollingNURBSKF, Noise, rotation, sample_curve
    from forward_filter import initialize_forward_window


def world_curve(x):
    x = np.asarray(x)
    return np.column_stack([x, 2*np.sin(x/17)+.4*np.sin(x/5)])


def slope(x):
    return 2/17*np.cos(x/17)+.4/5*np.cos(x/5)


def run(steps=160, seed=7, detection_range=10.):
    if steps < 1 or detection_range <= 0:
        raise ValueError('Need positive steps and detection range')
    rng = np.random.default_rng(seed)
    noise = Noise(white=.035, frame_translation=.12, smooth_shape=.07,
                  process_per_sqrt_second=.002, birth=.25)
    frames, errors = [], []
    filt = None
    prev_pose = None
    # Persistent arc-length lookup for both the sensor and visible true curve.
    xx = np.linspace(0., .45*(steps-1)+detection_range+2., 20000)
    speed = np.sqrt(1+slope(xx)**2)
    arc = np.r_[0., np.cumsum(.5*(speed[:-1]+speed[1:])*np.diff(xx))]
    x_at_s = lambda s: np.interp(s, arc, xx)
    for frame in range(steps):
        sensor_s = .45*frame
        x = float(x_at_s(sensor_s))
        origin = world_curve([x])[0]
        yaw = np.arctan(slope(x))
        rot = rotation(yaw)
        # Detector geometry changes every frame; its parameterization also changes.
        warp = .65+1.1*(.5+.5*np.sin(frame*.17))
        bias = rng.normal(0, noise.frame_translation, 2)
        anchors = np.linspace(0., detection_range, 12)
        dist = anchors[:, None]-anchors[None, :]
        gp_cov = noise.smooth_shape**2*np.exp(-.5*(dist/4.)**2)
        gp = np.linalg.cholesky(gp_cov+1e-10*np.eye(12)) @ rng.normal(size=(12, 2))
        field = CubicSpline(anchors, gp, axis=0)

        def detector(t):
            ahead = detection_range*np.asarray(t)**warp
            curve_x = x_at_s(sensor_s+ahead)
            return (world_curve(curve_x)-origin) @ rot+bias+field(ahead)

        measured = sample_curve(detector, count=55)
        if filt is None:
            filt = initialize_forward_window(measured, n=18, noise=noise)
            info = {'accepted': len(measured)}
        else:
            old_origin, old_yaw = prev_pose
            delta_t = rotation(old_yaw).T @ (origin-old_origin)
            filt.predict(delta_t, yaw-old_yaw, dt=.1,
                         odometry_cov=np.zeros((3, 3)))
            filt.recenter(reference=(0., 0.), anchor_fraction=4/15)
            info = filt.update_points(measured)
        prev_pose = (origin.copy(), yaw)
        u = np.linspace(*filt.domain, 160)
        estimated = filt.evaluate(u)
        estimated_world = estimated @ rot.T+origin
        measured_world = measured @ rot.T+origin
        # Approximate signed perpendicular distance to the known graph.
        def err(points):
            # Only score the visible forward region (trim noisy endpoints).
            trim = .02*detection_range
            mask = ((points[:, 0] > x_at_s(sensor_s+trim)) &
                    (points[:, 0] < x_at_s(sensor_s+detection_range-trim)))
            q = points[mask]
            return (q[:, 1]-world_curve(q[:, 0])[:, 1])/np.sqrt(1+slope(q[:, 0])**2)
        errors.append([np.sqrt(np.mean(err(measured_world)**2)),
                       np.sqrt(np.mean(err(estimated_world)**2))])
        frames.append(dict(origin=origin, rot=rot, measured=measured.copy(),
                           estimated=estimated.copy(), cp=filt.cp.copy(),
                           world=estimated_world, accepted=info['accepted'],
                           first_id=int(filt.control_ids[0]),
                           detection_range=detection_range, sensor_s=sensor_s,
                           visible_truth=world_curve(x_at_s(sensor_s+np.linspace(0, detection_range, 150)))))
    return filt, frames, np.asarray(errors)


def plot_result(frames, errors, output):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    f = frames[-1]
    range_m = f['detection_range']
    x = np.linspace(f['origin'][0]-.45*range_m, f['origin'][0]+1.2*range_m, 500)
    true = world_curve(x)
    measured_world = f['measured'] @ f['rot'].T+f['origin']
    ax[0, 0].plot(*true.T, 'k--', label='Static truth')
    ax[0, 0].plot(*measured_world.T, color='#db8643', label='Detection')
    ax[0, 0].plot(*f['world'].T, color='#087e8b', label='Filtered')
    ax[0, 0].plot(*f['origin'], 'k^', label='Sensor')
    ax[0, 0].set_title('Final estimate transformed to world frame')
    true_local = (true-f['origin']) @ f['rot']
    ax[0, 1].plot(*true_local.T, 'k--', label='Truth')
    ax[0, 1].plot(*f['measured'].T, color='#db8643', label='Detection')
    ax[0, 1].plot(*f['estimated'].T, color='#087e8b', label='Filtered')
    ax[0, 1].plot(*f['cp'].T, 'o:', ms=3, alpha=.6, label='Active control points')
    ax[0, 1].plot(0, 0, 'k^', label='Sensor')
    ax[0, 1].set_title('Same estimate stored in current moving frame')
    for a in ax[0]:
        a.legend(fontsize=8)
        a.set_aspect('equal', adjustable='box')
        a.set_xlabel('x [m]')
        a.set_ylabel('y [m]')
    # Explicit bounds: equal metre scales, no aspect-driven limit expansion.
    ax[0, 0].set_xlim(f['origin'][0]-.45*range_m, f['origin'][0]+1.2*range_m)
    ax[0, 0].set_ylim(f['origin'][1]-.3*range_m, f['origin'][1]+.3*range_m)
    ax[0, 1].set_xlim(-.45*range_m, 1.2*range_m)
    ax[0, 1].set_ylim(-.3*range_m, .3*range_m)
    ax[1, 0].plot(errors[:, 0], color='#db8643', label='Detection')
    ax[1, 0].plot(errors[:, 1], color='#087e8b', label='Filtered')
    ax[1, 0].set(xlabel='Frame', ylabel='RMS normal error [m]', title='Error on common visible region')
    ax[1, 0].legend()
    ids = np.array([f['first_id'] for f in frames])
    ax[1, 1].step(np.arange(len(frames)), ids, where='post', label='First active control ID')
    ax[1, 1].step(np.arange(len(frames)), ids+len(f['cp'])-1,
                  where='post', label='Last active control ID')
    ax[1, 1].set(xlabel='Frame', ylabel='Persistent control-point ID', title='Fixed-size window rolls along the curve')
    ax[1, 1].legend()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def animate(frames):
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True,
                             gridspec_kw={'height_ratios': [1, 2]})
    range_m = frames[0]['detection_range']
    global_x = (-.45*range_m, frames[-1]['origin'][0]+1.2*range_m)
    full_truth = world_curve(np.linspace(*global_x, 1500))
    global_y = (full_truth[:, 1].min()-.2*range_m,
                full_truth[:, 1].max()+.2*range_m)
    def draw(k):
        f = frames[k]
        x = np.linspace(f['origin'][0]-.45*range_m, f['origin'][0]+1.2*range_m, 300)
        truth = world_curve(x)
        for a in axes:
            a.clear()
        axes[0].plot(*full_truth.T, 'k--', label='Static truth / sensor trajectory')
        axes[0].plot(*(f['measured'] @ f['rot'].T+f['origin']).T,
                     color='#db8643', label='Detection')
        axes[0].plot(*f['world'].T, color='#087e8b', label='Filtered')
        axes[0].plot(*f['origin'], 'k^', label='Sensor')
        axes[0].set_title(f'World frame; frame {k}; sensor distance {f["sensor_s"]:.1f} m')
        axes[1].plot(*((truth-f['origin']) @ f['rot']).T, 'k--')
        axes[1].plot(*f['measured'].T, color='#db8643')
        axes[1].plot(*f['estimated'].T, color='#087e8b')
        axes[1].plot(*f['cp'].T, 'o:', ms=3)
        axes[1].plot(0, 0, 'k^')
        axes[1].set_title(f'Moving frame; detection: 0 to {range_m:g} m ahead; first control ID {f["first_id"]}')
        for a in axes:
            a.set_aspect('equal', adjustable='box')
            a.set_xlabel('x [m]')
            a.set_ylabel('y [m]')
        # Both sets of limits remain identical for every animation frame.
        axes[0].set_xlim(*global_x)
        axes[0].set_ylim(*global_y)
        axes[1].set_xlim(-.45*range_m, 1.2*range_m)
        axes[1].set_ylim(-.3*range_m, .3*range_m)
        axes[0].legend(loc='upper left', fontsize=8)
    animation = FuncAnimation(fig, draw, frames=len(frames), interval=80, repeat=True)
    plt.show()
    return animation


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=160)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--output', default='demo.png')
    parser.add_argument('--animate', action='store_true')
    parser.add_argument('--range', type=float, default=10., dest='detection_range',
                        help='Forward detection distance along the true curve [m]')
    args = parser.parse_args()
    filt, frames, errors = run(args.steps, args.seed, args.detection_range)
    warm = min(20, len(errors)//4)
    rms = np.sqrt(np.mean(errors[warm:]**2, axis=0))
    print(f'After warmup: detector RMS={rms[0]:.4f} m, filtered RMS={rms[1]:.4f} m')
    print(f'Active controls={filt.n}, rolls={filt.total_rolls}, final accepted={frames[-1]["accepted"]}')
    plot_result(frames, errors, args.output)
    if args.animate:
        animate(frames)
