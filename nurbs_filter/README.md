# Moving-frame NURBS filter with a rolling control-point window

This implementation estimates one static, simple planar curve while the sensor
moves. All control points and their covariance are stored in the CURRENT sensor
frame. The state contains a fixed number of control points, not a history of
measurement frames. There is no world-frame map and no batch sliding-window
pose optimization.

Files: `rolling_nurbs.py` is the reusable filter; `demo.py` is a simulation and
optional animation; `test_filter.py` checks mathematical invariants.

## 1. State and NURBS model

For degree p, n active control points, and positive fixed weights w_i,

    x = [P_0.x, P_0.y, ..., P_(n-1).x, P_(n-1).y]^T,
    C_k(u) = sum_i r_i(u) P_(i,k),
    r_i(u) = N_(i,p)(u) w_i / sum_j N_(j,p)(u) w_j.

The weights are not estimated. Default weights equal one, the B-spline special
case of NURBS; arbitrary positive unequal fixed weights are supported by the
constructor. With fixed weights and associations, evaluation is linear in x:

    C_k(u) = (r(u) tensor I_2) x_k.

Use a persistent uniform knot grid, with active window index b:

    t_j = (b+j) h,          j = 0,...,n+p,
    valid domain = [(b+p)h, (b+n)h].

These knots are NOT endpoint-clamped. The domain length is (n-p)h, not nh.
The basis and base-interval convention follow SciPy BSpline:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.html

u is a persistent curve coordinate, not the current detector's normalized
parameter. Initialization uses approximate arc length to define its gauge;
subsequent updates never re-normalize the state or move its knots.

## 2. Static-world prediction in the moving frame

Define an odometry increment by

    q_old = R_delta q_new + t_delta.

Thus t_delta is the NEW frame's origin in OLD-frame coordinates; R_delta
maps vectors from the new frame into the old one. Then a world-static control
point has prediction

    P_i^- = A (P_i^+ - t_delta),     A = R_delta^T,
    F = I_n tensor A,
    Sigma^- = F Sigma^+ F^T + G U G^T + Q.

For additive odometry errors [delta_t.x, delta_t.y, delta_theta], with
translation errors expressed in the old frame,

    J = [[0,-1],[1,0]],
    G_i = [-A, -J P_i^-].

Stack the G_i vertically. The SAME pose error affects all points; G U G^T
therefore has off-diagonal control-point blocks. Replacing this by independent
per-control-point noise loses essential correlations.

For world poses (R_old,t_old), (R_new,t_new), where a world point satisfies
q_world = R_k q_k + t_k, compute

    R_delta = R_old^T R_new,
    t_delta = R_old^T (t_new-t_old).

Because the rational basis sums to one, with exact odometry and Q=0,

    R_new C_new^-(u)+t_new = R_old C_old^+(u)+t_old.

This is the exact static-world prediction identity. Measurement corrections
still change the estimated curve; a noisy estimate cannot be both updated and
literally identical in the world at every frame. Noisy/biasing odometry also
prevents guaranteed absolute world stationarity without additional constraints.

## 3. Rolling control-point state

For a forward roll, change b to b+1 and form

    P'_0,...,P'_(n-2) = P_1,...,P_(n-1),
    P'_(n-1) = 2 P_(n-1)-P_(n-2)+eta,
    eta ~ Normal(0, sigma_birth^2 I_2).

Continue the last fixed weight at the new point. Let S be the n-by-n linear
map implementing these assignments, T=S tensor I_2, and E insert the new
two-dimensional acceleration noise at the final control point. Then

    x' = T x,
    Sigma' = T Sigma T^T + E (sigma_birth^2 I_2) E^T.

This retains the full covariance among surviving controls and the correlations
of the extrapolated new point with the existing points. Discarding a Gaussian
state component means retaining its covariance marginal, not conditioning on
its previous mean or applying a covariance Schur complement.

Old valid interval: [(b+p)h,(b+n)h].
New valid interval: [(b+p+1)h,(b+n+1)h].

On the common interval, all active basis functions, control points, and weights
are unchanged. Both curve geometry and curve covariance are preserved exactly.
Only the newly exposed span depends on the extrapolated control point. There
are p boundary-support controls beyond what an endpoint-interpolating mental
model would suggest; this is why simply shifting a clamped spline is incorrect.

Backward rolling uses the analogous leading-point extrapolation. Discarded
points are not stored; revisiting an old region creates a new prior rather than
recovering an old map.

`recenter()` projects a moving-frame reference (usually the sensor origin)
onto the predicted curve. It rolls until this parameter is approximately at
`anchor_fraction` of the window, leaving the rest of the window ahead. For fast
motion, an externally tracked anchor parameter or reinitialization is needed.

## 4. Measurements with unrelated parameterizations

A detection supplies a callable D_k(v) in the CURRENT frame. Sample it densely,
resample approximately uniformly in arc length, and obtain q_j. This sampling
does not establish parameter correspondences.

Associate geometrically against the prediction:

    u_j = argmin_(u in valid domain) ||q_j-C^-(u)||^2.

The implementation uses a dense coarse search followed by local bounded
minimization. For each association, define

    tau_j = dC^-/du(u_j) / ||dC^-/du(u_j)||,
    n_j = J tau_j,
    e_j = n_j^T (q_j-C^-(u_j)),
    H_j = n_j^T (r(u_j) tensor I_2).

Freeze u_j and n_j for ONE update. This is a point-to-normal EKF correction:
at an interior closest point, the parameter-shift term vanishes to first order
because n_j^T dC/du=0. It avoids interpreting tangential resampling changes as
normal curve motion. Tangential control-point modes are weakly/unobservable
from unlabelled curve geometry and remain governed by the prior/gauge.

Reject degenerate tangents, associations at window endpoints, distances beyond
the spatial gate, and standardized residual outliers. The geometry must be
close enough for association to be valid. Self intersections, nearby parallel
branches, and very large pose jumps need an explicit branch/order model.

## 5. Detector covariance and update

Samples of one detector curve are correlated. Defaults model

    q_j = C_true(u_j)+b+g(s_j)+epsilon_j,
    b ~ Normal(0, sigma_translation^2 I_2),
    Cov[g(s_i),g(s_j)] = sigma_shape^2
        exp(-(s_i-s_j)^2/(2 ell^2)) I_2,
    Cov[epsilon_i,epsilon_j] = delta_ij sigma_white^2 I_2.

After normal projection,

    R_ij = delta_ij sigma_white^2
       + (n_i^T n_j) [sigma_translation^2
          + sigma_shape^2 exp(-(s_i-s_j)^2/(2 ell^2))].

Thus more samples do not erase a shared detector offset. These are tunable
assumptions, not a fitted/calibrated detector model. The code also accepts the
full Cartesian sample covariance V through `covariance_xy`; it uses R=L V L^T,
where L places n_j^T at sample j. V must correspond to the resampled points in
their interleaved [x0,y0,x1,y1,...] order and yield a positive definite retained R.

After gating, perform

    S_z = H Sigma^- H^T + R,
    K = Sigma^- H^T S_z^-1,
    x^+ = x^- + K e,
    Sigma^+ = (I-KH) Sigma^- (I-KH)^T + K R K^T.

The code solves a Cholesky system rather than explicitly inverting S_z.
Its Gaussian covariance is conditional on the selected associations and gates;
it does not include branch-association uncertainty. Do not repeat the same
frame as independent evidence in a sequence of ordinary KF updates.

## 6. Usage

Install dependencies and run:

```sh
python -m pip install -r requirements.txt
python -m unittest -v test_filter.py
python demo.py
python demo.py --animate
```

The animation requires a graphical Matplotlib backend. The default simulation
also works headlessly and saves demo.png. Core usage:

```python
import numpy as np
from rolling_nurbs import RollingNURBSKF, Noise, sample_curve

# detector0/detector_k are vectorized callables accepting parameter arrays
# and returning (m,2) coordinates in their respective sensor frames.
first = sample_curve(detector0, parameter_range=(0, 1), count=60)
kf = RollingNURBSKF.from_points(first, n=18, noise=Noise())

# For every SUBSEQUENT frame:
kf.predict(translation=delta_t_old_frame, yaw=delta_yaw, dt=dt,
           odometry_cov=increment_covariance_3x3)
kf.recenter(reference=(0, 0), anchor_fraction=.25)
statistics = kf.update_curve(detector_k, parameter_range=(0, 1), count=60)
u = np.linspace(*kf.domain, 200)
filtered_curve_in_current_frame = kf.evaluate(u)
```

Initialization is an approximate least-squares fit with a weak curvature penalty
and a configurable, heuristic initial covariance, not a calibrated Bayesian
posterior. Supply good initial control points/covariance to the constructor if
available. Do not assimilate the initialization frame a second time.

Increase n or reduce physical knot spacing for tighter bends; increase birth
uncertainty if forward curvature extrapolation is unreliable. Set process
noise to zero for an exactly static curve with a trusted shape model, or retain
a small value to allow model mismatch. The fixed weights and knot grid are
intentional: estimating weights/knots needs a nonlinear state and extra gauge
constraints.

The odometry-noise model assumes independent increment errors uncorrelated
with the old curve state. Differences of uncertain absolute poses generally do
NOT satisfy this assumption. Persistent odometry bias, pose-state correlation,
or repeated revisits require a joint curve/pose filter or factor-graph model.
Likewise, the default detector model assumes independent errors across frames;
temporally correlated detector bias needs an augmented bias state.

## 7. Validation

Six tests verify exact static-world prediction, covariance rotation, the
odometry Jacobian by finite differences, common pose-noise correlations, exact
geometry/covariance preservation when rolling in either direction with unequal
weights, invariance to detector reparameterization, and an update that reduces
normal error while preserving positive semidefinite covariance (some checks
share a test).

Seed 7, 160 frames: 18 active controls, 32 rolls; after warmup the synthetic
detector RMS normal error is 0.1427 m and filter error is 0.0305 m. Error is
evaluated against a known graph using an approximate perpendicular distance
over the common visible region. This is an illustrative exact-odometry run,
not a general performance guarantee or uncertainty calibration experiment.

## 8. Calibrating detector noise from exported centerlines

The `centerline_dataset_all_generated` directory contains detected and ground
truth centerlines in the same sensor frame. Fit the four detector fields of
`Noise` and save a JSON report with:

```sh
python calibrate_noise.py centerline_dataset_all_generated --output detector_noise.json
```

The script samples detector XY every 0.5 m from 0.5 to 150 m, projects each
sample onto the interior of the frame's ground truth polyline, rejects normal
errors larger than 2 m, and fits the covariance of the remaining signed normal
errors. It removes a mean error at each sampled distance before fitting. The
output includes match counts, mean bias, and centered RMS error. Sampling range,
spacing, endpoint margin, and outlier cutoff are command-line options.

Use the fitted detector covariance in the filter like this:

```python
import json
from rolling_nurbs import Noise

with open("detector_noise.json") as source:
    noise = Noise(**json.load(source)["noise"])
```

This calibrates detector noise only. The process and birth fields retain their
defaults; they cannot be identified from same-frame detector versus label
errors. The GT centerline comes from a merged rail map and odometry, so label
and association errors contribute to these estimates. The fit applies to
the selected distance range and inlier cutoff.
For the full 150 m range, the fitted values are larger than a near-field fit
because distant centerline errors are larger. The single `Noise` model is an
average over all matched distances, rather than a range-dependent covariance.

## 9. Run the filter on exported scenes

Use the calibrated detector covariance and saved poses to run the filter on
one scene or the whole exported directory:

```sh
python demo_dataset.py eval_new_data
python demo_dataset.py eval_new_data --interactive
python demo_dataset.py all --output-dir dataset_demo_output
```

Each scene writes a JSON summary, a per-frame CSV, and a PNG showing the last
frame and the error time series. The default directory is `dataset_demo_output_`;
the example above selects `dataset_demo_output`. The `all` run also writes
`summary.json` with pooled metrics across scenes. The filter uses detected centerlines and relative
sensor poses. Each new sensor position is also treated as a soft observation
that the curve passes nearby. Pose increments are treated as exact planar odometry;
the dataset does not provide increment covariance. If the sensor leaves the
active curve window, the demo reinitializes from that frame's detection and
records a reset.

`--interactive` opens a window for one scene after processing it. Drag the
frame slider, click the error chart to jump to a frame, use Prev/Next or the
left/right arrow keys to step, Shift+arrows to move ten frames, and Play/Pause
or Space to animate. The Near/Full button zooms in on the control points near
the sensor. `--frame 100` selects the initial frame. This mode needs a
graphical Matplotlib backend; batch mode works headlessly.

The scene plot also shows the 24 active NURBS control points. Measurements are
sampled at fixed sensor-forward distances through 150 m, and the rolling anchor
is placed near the sensor to leave most of the window ahead. The demo applies
a tapered longitudinal constraint to the trailing control points so the
**filtered NURBS itself** reaches at least 150 m forward in sensor -Y. The same
linear projection is applied to the state covariance. `forward_reach_m` and
`endpoint_correction_m` in the CSV show the achieved range and the size of each
constraint; the JSON summarizes them. Change the range with `--max-forward-m`.

The path observation projects the current sensor origin onto the estimated
curve and updates only its lateral (curve-normal) offset. It uses a 0.05 m
standard deviation by default, so the curve is encouraged to pass near the
sensor without being forced through it. One new position is assimilated per
frame; older positions are already carried by the rolling state. The update
is gated at 2 m to avoid snapping to a distant branch. `path_accepted` and
`path_normal_residual_m` appear in the frame CSV. Set
`--path-lateral-sigma-m` to change its strength or `--no-path-measurement`
to compare against detector-only filtering.

Detector residuals above 0.2 m receive additional measurement variance, which
reduces the influence of isolated large errors. If fewer than 80% of sampled
detector points associate for three consecutive frames, the demo considers
reinitialization. It does so only when the current detection agrees with
recent detections after transforming them into the current sensor frame; this
avoids resetting onto a sudden detector glitch. `--robust-threshold-m`,
`--reinit-accept-fraction`, and `--reinit-patience` tune these rules. Set the
first two options to zero to disable them.

For scoring, ground-truth points are sampled every 2 m from the sensor to the
selected forward limit. The detector and filtered curves are each projected
onto the *same* ground-truth anchors. Reported RMSE, MAE, and 95th-percentile
absolute error are signed lateral errors in sensor XY on anchors where both
curves have an interior projection. Coverage and reset count accompany the
errors so missing predictions are visible. Results are conditional on the
exported merged-rail labels and saved odometry, not an absolute map-accuracy
test. Use `--max-frames` for a short smoke run; `--help` lists range, sampling,
filter, and gating options.
