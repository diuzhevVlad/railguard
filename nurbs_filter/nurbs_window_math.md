# NURBS curve estimation in a moving frame with a rolling control-point window

Complete mathematical specification of the accompanying planar Python filter.
The state is a fixed-size spatial window of curve control points, not a time
window of poses. Measurements are curves in the current sensor frame; odometry
transports the estimate between frames. The physical scene curve is static.

## 1. Notation and assumptions

| Symbol | Meaning |
|---|---|
| \(W\), \(B_k\) | Stationary world frame and sensor frame at time \(k\) |
| \(R_k,t_k\) | Sensor-to-world rotation and translation |
| \(P_{i,k}\in\mathbb R^2\) | Active control point \(i\) in \(B_k\) |
| \(n,p,h\) | Number of controls, spline degree, persistent knot spacing |
| \(b_k\in\mathbb Z\) | First active global control/knot index |
| \(x_k\in\mathbb R^{2n}\) | Interleaved control-point state |
| \(\Sigma_k\in\mathbb R^{2n\times2n}\) | Full state covariance |
| \(u\), \(v\) | Persistent state parameter and arbitrary detector parameter |
| \(C_k(u)\), \(D_k(v)\) | Estimated curve and detector curve in \(B_k\) |
| Superscripts \(-,+\) | Predicted and corrected quantities |
| \(\otimes\) | Kronecker product |

The implemented assumptions are:

1. One locally simple, nonbranching planar curve, with sufficiently close
   predictions for local geometric association.
2. Fixed positive rational weights and a persistent, uniform, non-clamped
   knot grid. A uniform rational spline is a valid special case of NURBS.
3. Approximately Gaussian errors after local linearization.
4. Odometry increments independent of the previous curve-state error, and
   detector errors independent across frames. Within-frame detector samples
   are correlated.
5. No archive of discarded controls and no jointly estimated world pose.

Define

\[
R(\theta)=
\begin{bmatrix}\cos\theta&-\sin\theta\\
\sin\theta&\cos\theta\end{bmatrix},
\qquad
J=\begin{bmatrix}0&-1\\1&0\end{bmatrix}.
\]

Then \(R'(\theta)=R(\theta)J=JR(\theta)\).

## 2. NURBS representation

The local knot vector for window index \(b\) is

\[
t_j=(b+j)h,\qquad j=0,\ldots,n+p.
\]

The valid base interval is

\[
\mathcal U_b=[t_p,t_n]=[(b+p)h,(b+n)h],
\qquad |\mathcal U_b|=(n-p)h.
\]

There are \(n+p+1\) knots. Knots are not repeated at the interval endpoints.
The first and last controls therefore do not generally lie on the curve.

For completeness, the B-spline basis is defined by

\[
N_{i,0}(u)=
\begin{cases}
1,&t_i\leq u<t_{i+1},\\
0,&\text{otherwise},
\end{cases}
\]

\[
N_{i,d}(u)=
\frac{u-t_i}{t_{i+d}-t_i}N_{i,d-1}(u)
+
\frac{t_{i+d+1}-u}{t_{i+d+1}-t_{i+1}}N_{i+1,d-1}(u).
\]

A term with zero denominator is defined as zero. Endpoint values on the
base interval use the continuous one-sided extension.

For fixed weights \(w_i>0\), define

\[
a_i(u)=w_iN_{i,p}(u),\qquad
\omega(u)=\sum_{i=0}^{n-1}a_i(u),\qquad
r_i(u)=\frac{a_i(u)}{\omega(u)}.
\]

On the valid interval,

\[
\sum_i r_i(u)=1,\qquad
C_k(u)=\sum_{i=0}^{n-1}r_i(u)P_{i,k}.
\]

Unit weights give a polynomial B-spline. Unequal fixed weights retain rational
geometry while preserving linearity with respect to the control coordinates.

Stack the state and define its evaluation matrix:

\[
x_k=
\begin{bmatrix}P_{0,k}\\\vdots\\P_{n-1,k}\end{bmatrix},
\qquad
B_b(u)=
\begin{bmatrix}r_0(u)I_2&\cdots&r_{n-1}(u)I_2\end{bmatrix}.
\]

Thus

\[
C_k(u)=B_b(u)x_k.
\]

The curve covariance and cross-covariance at fixed parameters are

\[
\operatorname{Cov}[C_k(u),C_k(v)]
=B_b(u)\Sigma_kB_b(v)^T.
\]

In particular, \(\Sigma_C(u)=B_b(u)\Sigma_kB_b(u)^T\).
This is conditional on the parameter values and the selected representation.

## 3. Curve derivatives

The B-spline derivative is

\[
N'_{i,p}(u)=
\frac{p}{t_{i+p}-t_i}N_{i,p-1}(u)
-
\frac{p}{t_{i+p+1}-t_{i+1}}N_{i+1,p-1}(u).
\]

Differentiate again for \(N''_{i,p}\) on each smooth span. Then

\[
r_i'=\frac{a_i'\omega-a_i\omega'}{\omega^2},
\]

\[
r_i''=
\frac{a_i''}{\omega}
-\frac{2a_i'\omega'}{\omega^2}
-\frac{a_i\omega''}{\omega^2}
+\frac{2a_i(\omega')^2}{\omega^3}.
\]

Consequently,

\[
C_k'(u)=\sum_i r_i'(u)P_{i,k},\qquad
C_k''(u)=\sum_i r_i''(u)P_{i,k}.
\]

At a nondegenerate point,

\[
\tau_k(u)=\frac{C_k'(u)}{\|C_k'(u)\|},
\qquad n_k(u)=J\tau_k(u).
\]

For reference, the signed planar curvature is

\[
\kappa(u)=
\frac{C_x'(u)C_y''(u)-C_y'(u)C_x''(u)}
{\|C'(u)\|^3}.
\]

The filter uses the first derivative; the second derivative helps explain
the local behavior of geometric association.

## 4. Moving-frame kinematics

The absolute sensor pose convention is

\[
q^W=R_kq^{B_k}+t_k.
\]

An exactly static world control point satisfies

\[
P_i^{B_k}=R_k^T(P_i^W-t_k).
\]

Define the odometry increment from the previous sensor frame to the current
sensor frame by

\[
R_\Delta=R_{k-1}^TR_k,\qquad
t_\Delta=R_{k-1}^T(t_k-t_{k-1}).
\]

Equivalently,

\[
q^{B_{k-1}}=R_\Delta q^{B_k}+t_\Delta.
\]

The vector \(t_\Delta\) is the new sensor origin expressed in the old frame.
It is not a world-frame translation and not the translation of the inverse
transform.

Writing \(A=R_\Delta^T\), static-world prediction gives

\[
P_{i,k}^-=A(P_{i,k-1}^+-t_\Delta).
\]

In stacked form,

\[
x_k^-=Fx_{k-1}^++c,\qquad
F=I_n\otimes A,\qquad
c=-\mathbf1_n\otimes(At_\Delta).
\]

This is an affine state transition for known odometry. There is no control-point
velocity state: the apparent motion is entirely due to sensor motion.

### Proof of world-frame stationarity

Since the rational basis sums to one,

\[
\begin{aligned}
C_k^-(u)
&=\sum_i r_i(u)A(P_{i,k-1}^+-t_\Delta)\\
&=A\bigl(C_{k-1}^+(u)-t_\Delta\bigr).
\end{aligned}
\]

Using \(R_kA=R_{k-1}\) and
\(t_k-R_{k-1}t_\Delta=t_{k-1}\),

\[
\boxed{
R_kC_k^-(u)+t_k
=R_{k-1}C_{k-1}^+(u)+t_{k-1}.
}
\]

This identity is exact before correction, for exact odometry and unchanged
physical curve points. After rolling it holds on the common valid domain.
Measurement corrections intentionally revise the world-frame estimate.

## 5. Odometry uncertainty and prediction covariance

Represent odometry uncertainty in the supplied increment coordinates:

\[
\delta\xi=
\begin{bmatrix}\delta t_x&\delta t_y&\delta\theta\end{bmatrix}^T,
\qquad
\operatorname{Cov}(\delta\xi)=U_k.
\]

Translations are additive in old-frame coordinates. Define the uncertain
transition as

\[
f_i(P_i,\delta\xi)
=R(\theta_\Delta+\delta\theta)^T
\bigl(P_i-t_\Delta-\delta t\bigr).
\]

At the nominal increment,

\[
\frac{\partial f_i}{\partial P_i}=A,\qquad
\frac{\partial f_i}{\partial\delta t}=-A,\qquad
\frac{\partial f_i}{\partial\delta\theta}=-JP_{i,k}^-.
\]

Thus

\[
G_i=
\begin{bmatrix}
-A&-JP_{i,k}^-
\end{bmatrix}\in\mathbb R^{2\times3},
\qquad
G=\begin{bmatrix}G_0\\\vdots\\G_{n-1}\end{bmatrix}.
\]

Explicitly, if \(P_{i,k}^-=(x_i,y_i)^T\), the last column of \(G_i\)
is \((y_i,-x_i)^T\).

For independent previous-state and increment errors,

\[
\boxed{
\Sigma_k^-=F\Sigma_{k-1}^+F^T+GU_kG^T+Q_k.
}
\]

The implementation uses

\[
Q_k=\sigma_{\mathrm{process}}^2\Delta t\,I_{2n},
\]

where \(\sigma_{\mathrm{process}}\) has units of length per square root of time.
For a strictly static and exactly represented curve, \(Q_k=0\) is appropriate.
A small nonzero value permits representation mismatch.

One odometry error acts on every control. Therefore,

\[
\operatorname{Cov}_{\mathrm{odom}}(P_i,P_j)=G_iU_kG_j^T,
\]

including \(i\ne j\). Independent noise per control would omit these correlations.
For pure translation uncertainty, every pair receives the same contribution
\(A\,U_{tt}\,A^T\).

If \(C_{x\xi}=\operatorname{Cov}(\delta x_{k-1},\delta\xi)\ne0\), the correct
first-order expression additionally contains

\[
FC_{x\xi}G^T+GC_{x\xi}^TF^T.
\]

Those terms are not tracked in this implementation. Differences of correlated
absolute pose estimates generally cannot be treated as independent increments.
Also, an SE(2) left/right perturbation covariance must be converted to this
additive coordinate convention before use.

## 6. Forward rolling transition

Every active control has persistent global identity \(b+i\).
A forward roll replaces \(b\) with \(b+1\), keeps \(n\) controls, and defines

\[
P_i'=P_{i+1},\quad i=0,\ldots,n-2,
\]

\[
P_{n-1}'=2P_{n-1}-P_{n-2}+\eta,\qquad
\eta\sim\mathcal N(0,V_{\mathrm{birth}}).
\]

The default is \(V_{\mathrm{birth}}=\sigma_{\mathrm{birth}}^2I_2\).
This is a prior on the newly introduced second control-point difference,
not a claim that the physical curve has zero curvature.

Let \(S_+\in\mathbb R^{n\times n}\) have entries

\[
(S_+)_{i,i+1}=1\quad(i<n-1),\qquad
(S_+)_{n-1,n-2}=-1,\qquad
(S_+)_{n-1,n-1}=2,
\]

with all other entries zero. Put

\[
T_+=S_+\otimes I_2,\qquad
E_+=e_{n-1}\otimes I_2.
\]

Then the Gaussian rolling transition is

\[
\boxed{
x'=T_+x,\qquad
\Sigma'=T_+\Sigma T_+^T+E_+V_{\mathrm{birth}}E_+^T.
}
\]

Weights are shifted with their persistent controls and the new boundary weight
is continued:

\[
(w_0',\ldots,w_{n-1}')
=(w_1,\ldots,w_{n-1},w_{n-1}).
\]

The covariance of the newborn control is

\[
\begin{aligned}
\Sigma_{\mathrm{new,new}}
={}&4\Sigma_{n-1,n-1}+\Sigma_{n-2,n-2}\\
&-2\Sigma_{n-1,n-2}-2\Sigma_{n-2,n-1}
+V_{\mathrm{birth}},
\end{aligned}
\]

and its cross-covariance with a retained old control \(P_j\) is

\[
\operatorname{Cov}(P_j,P_{\mathrm{new}})
=2\Sigma_{j,n-1}-\Sigma_{j,n-2}.
\]

The new control must not be initialized independently of its parents.

## 7. Backward rolling and repeated shifts

A backward roll sets \(b'=b-1\) and

\[
P_0'=2P_0-P_1+\eta,\qquad
P_i'=P_{i-1}\quad(i=1,\ldots,n-1).
\]

The scalar map has

\[
(S_-)_{0,0}=2,\quad(S_-)_{0,1}=-1,\quad
(S_-)_{i,i-1}=1\quad(i\ge1).
\]

Use \(T_-=S_-\otimes I_2\), \(E_-=e_0\otimes I_2\), and

\[
\Sigma'=T_-\Sigma T_-^T+E_-V_{\mathrm{birth}}E_-^T.
\]

Weights become \((w_0,w_0,w_1,\ldots,w_{n-2})\).

For \(m\) repeated forward rolls, with the same \(T=T_+\) and independent
birth noise \(Q_b=E_+V_{\mathrm{birth}}E_+^T\),

\[
x^{(m)}=T^mx,\qquad
\Sigma^{(m)}
=T^m\Sigma(T^m)^T+\sum_{j=0}^{m-1}T^jQ_b(T^j)^T.
\]

The code applies single rolls sequentially, including each covariance update.
Backward motion does not resurrect old discarded controls: it extrapolates
new controls because no persistent map is stored.

### Marginalization, not conditioning

If \(M\) selects the retained controls, their marginal is

\[
x_{\mathrm{ret}}=Mx,\qquad
\Sigma_{\mathrm{ret}}=M\Sigma M^T.
\]

A covariance Schur complement would describe conditioning on discarded values.
Discarding them supplies no such observation, so that operation is inappropriate.

## 8. Why rolling preserves the existing curve

For a forward roll, the old and new domains are

\[
\mathcal U_b=[(b+p)h,(b+n)h],\qquad
\mathcal U_{b+1}=[(b+p+1)h,(b+n+1)h].
\]

Their common interval is

\[
\mathcal O=[(b+p+1)h,(b+n)h].
\]

On this interval the dropped basis has zero contribution. The newborn basis
starts at \((b+n)h\), so it also has zero contribution in the interior of
\(\mathcal O\). Every contributing retained basis has exactly the same
persistent knot support, weight and control identity as before.

Consequently, with \(B_{\mathrm{old}},B_{\mathrm{new}}\) denoting evaluation
matrices for the two windows,

\[
B_{\mathrm{new}}(u)T_+=B_{\mathrm{old}}(u),\qquad
B_{\mathrm{new}}(u)E_+=0,\quad u\in\mathcal O.
\]

Therefore

\[
C_{\mathrm{new}}(u)=C_{\mathrm{old}}(u)
\]

and, for \(u,v\in\mathcal O\),

\[
\begin{aligned}
B_{\mathrm{new}}(u)\Sigma'B_{\mathrm{new}}(v)^T
=B_{\mathrm{old}}(u)\Sigma B_{\mathrm{old}}(v)^T.
\end{aligned}
\]

These are exact representation identities, not linearization approximations.
They include endpoint limits for the implemented degrees \(p\ge1\).
Changing to a newly clamped knot vector after every shift would invalidate
the argument by changing the boundary basis functions.

## 9. Choosing when to roll

After odometry prediction, project a reference point \(q_{\mathrm{ref}}\) in
the current sensor frame onto the predicted curve:

\[
u_{\mathrm{ref}}
=\arg\min_{u\in[a,c]}
\|q_{\mathrm{ref}}-C_k^-(u)\|^2,\qquad
[a,c]=\mathcal U_b.
\]

Usually \(q_{\mathrm{ref}}=(0,0)^T\). For anchor fraction \(0<\alpha<1\),

\[
u_{\mathrm{target}}=a+\alpha(c-a),
\qquad
m=\operatorname{round}
\left(\frac{u_{\mathrm{ref}}-u_{\mathrm{target}}}{h}\right).
\]

Apply \(m\) forward rolls if \(m>0\), or \(|m|\) backward rolls if \(m<0\).
This positions the reference approximately \(\alpha\) of the way into the
window while retaining fixed control density and parameter identities.

The algorithm assumes the old window still contains a useful reference
projection. Endpoint projection or an excessively large shift triggers
reinitialization rather than an ungrounded geometric association.
The current default maximum absolute shift is \(n-p-1\).

## 10. Arbitrary measurement representation and sampling

The detector supplies \(D_k:[v_{\min},v_{\max}]\to\mathbb R^2\) in \(B_k\).
It need only be sampleable; it need not share the state representation.

For dense parameters \(v_\ell\), evaluate \(d_\ell=D_k(v_\ell)\) and approximate
detector arc length by

\[
s_0=0,\qquad
s_\ell=\sum_{r=1}^{\ell}\|d_r-d_{r-1}\|.
\]

After removing repeated arc-length samples, interpolate coordinates at

\[
\bar s_j=\frac{j}{M-1}s_{\mathrm{last}},
\qquad j=0,\ldots,M-1,
\]

to obtain the measurement points \(q_j\).

Arc-length resampling reduces sensitivity to detector parameterization and
prevents arbitrary oversampling of slow-parameter regions. It does not make
the samples independent, nor does it establish correspondence with state \(u\).
In particular, equating \(v\) or the current normalized detector arc fraction
to state \(u\) is invalid when the visible segment moves.

## 11. Geometric association

For every measurement point, compute

\[
u_j=\arg\min_{u\in\mathcal U_b}
\frac12\|q_j-C_k^-(u)\|^2.
\]

The implementation first searches a dense curve grid, then minimizes within
a neighboring parameter bracket. This is local association, not a guarantee
of the globally correct branch.

At an interior stationary point,

\[
(C_k^-(u_j)-q_j)^TC_k^{-\prime}(u_j)=0.
\]

The second derivative of the objective is

\[
\|C_k^{-\prime}(u_j)\|^2
+(C_k^-(u_j)-q_j)^TC_k^{-\prime\prime}(u_j).
\]

A positive value supports a locally isolated minimum. Near degenerate tangents,
high curvature offsets, intersections or nearby branches, association may be
unstable or ambiguous. The supplied implementation rejects invalid tangents
and endpoint matches, but does not estimate branch uncertainty.

For an explicitly tracked ordered branch, one could instead impose monotonic
parameter correspondences. That is an extension, not an implemented guarantee.

## 12. Normal residuals and their first-order derivation

At the association, define

\[
\tau_j=\frac{C_k^{-\prime}(u_j)}
{\|C_k^{-\prime}(u_j)\|},
\qquad n_j=J\tau_j,
\]

\[
e_j=n_j^T(q_j-C_k^-(u_j)).
\]

Freeze \(u_j,n_j\) for one update. The scalar measurement model is

\[
z_j=n_j^Tq_j,\qquad
h_j(x)=n_j^TB_b(u_j)x,
\]

with Jacobian

\[
\boxed{H_j=n_j^TB_b(u_j).}
\]

To see why discarding tangential residuals is useful, perturb the curve by
\(\delta x\) and allow its closest-point parameter to change by \(\delta u_j\).
To first order,

\[
\delta C=B_b(u_j)\delta x+C_k^{-\prime}(u_j)\delta u_j.
\]

Since \(n_j^TC_k^{-\prime}(u_j)=0\), the parameter-shift term disappears from
the projected residual. At an exact interior closest point, \(q_j-C(u_j)\)
is normal and \(\delta n_j\) is perpendicular to \(n_j\), so
\(\delta n_j^T(q_j-C(u_j))=0\) as well. Thus

\[
\delta e_j\approx-H_j\delta x.
\]

The positive Jacobian \(H_j\) belongs to the predicted measurement \(h_j\);
the residual \(z_j-h_j(x)\) has derivative \(-H_j\).
This explains the sign in the Kalman correction \(x^+=x^-+Ke\).

Only geometry is observed, not labels along the curve. Tangential modes can
be weakly observed or unobservable; for a straight curve, normal-only
measurements do not observe control-point coordinates along its tangent.
The prior and persistent parameter grid provide the chosen representation.

## 13. Correlated measurement covariance

A smooth detected curve is one structured observation. Model its samples by

\[
q_j=C_{\mathrm{true}}(\bar u_j)+\beta+g(s_j)+\epsilon_j,
\]

where

\[
\beta\sim\mathcal N(0,\sigma_{\mathrm{translation}}^2I_2),
\]

\[
\operatorname{Cov}(g(s_i),g(s_j))
=\sigma_{\mathrm{shape}}^2
\exp\!\left[-\frac{(s_i-s_j)^2}{2\ell^2}\right]I_2,
\]

\[
\operatorname{Cov}(\epsilon_i,\epsilon_j)
=\delta_{ij}\sigma_{\mathrm{white}}^2I_2.
\]

The three components are mutually independent. Arc coordinates \(s_j\) here
are cumulative distances on this frame's sampled detection.

The full Cartesian sample covariance has blocks

\[
V_{ij}=
\left[
\sigma_{\mathrm{translation}}^2+
\sigma_{\mathrm{shape}}^2
\exp\!\left(-\frac{(s_i-s_j)^2}{2\ell^2}\right)
\right]I_2
+\delta_{ij}\sigma_{\mathrm{white}}^2I_2.
\]

Let \(L\in\mathbb R^{M\times2M}\) place \(n_j^T\) at the columns for sample \(j\).
Normal projection gives

\[
\boxed{R=LVL^T,}
\]

or elementwise,

\[
\boxed{
R_{ij}
=\delta_{ij}\sigma_{\mathrm{white}}^2
+(n_i^Tn_j)
\left[
\sigma_{\mathrm{translation}}^2+
\sigma_{\mathrm{shape}}^2
\exp\!\left(-\frac{(s_i-s_j)^2}{2\ell^2}\right)
\right].
}
\]

Different normal directions can produce negative off-diagonal entries.
Nevertheless \(R\) is positive semidefinite because it is a covariance
projection; positive white noise makes it positive definite.

If the detector has known fixed-basis coefficient covariance
\(\Sigma_{\mathrm{det}}\), and sampled Cartesian coordinates obey
\(q=A_{\mathrm{det}}c_{\mathrm{det}}\), use

\[
V=A_{\mathrm{det}}\Sigma_{\mathrm{det}}A_{\mathrm{det}}^T
+V_{\mathrm{extra}}.
\]

For a nonlinear detector representation, replace \(A_{\mathrm{det}}\) with its
sample-coordinate Jacobian. The code accepts this full \(V\) directly, in
interleaved sample coordinates. It must correspond to the actual sampled points.

### Why sample count must not erase a common bias

For a scalar offset \(y\), consider

\[
q_j=y+\beta+\epsilon_j,\quad
\operatorname{Var}(\beta)=\sigma_b^2,\quad
\operatorname{Var}(\epsilon_j)=\sigma_\epsilon^2.
\]

The sample mean has variance

\[
\operatorname{Var}(\bar q)=\sigma_b^2+\frac{\sigma_\epsilon^2}{M}.
\]

As \(M\to\infty\), the uncertainty approaches \(\sigma_b^2\), not zero.
Treating all samples as independent would incorrectly remove this floor.

## 14. Gating

Before correction, define each marginal innovation variance

\[
\nu_j=H_j\Sigma_k^-H_j^T+R_{jj}.
\]

The implementation retains a sample only if

\[
a+\varepsilon_u<u_j<c-\varepsilon_u,\qquad
\|C_k^{-\prime}(u_j)\|>\varepsilon_\tau,
\]

\[
\|q_j-C_k^-(u_j)\|<d_{\max},\qquad
e_j^2<\gamma^2\nu_j.
\]

Defaults are \(\varepsilon_u=10^{-3}h\),
\(\varepsilon_\tau=10^{-8}\), \(d_{\max}=0.9\) length units and \(\gamma=5\).
The earlier reference check for rolling uses a separate \(10^{-4}h\) margin.

After selecting indices \(\mathcal I\), retain

\[
e\leftarrow e_{\mathcal I},\qquad
H\leftarrow H_{\mathcal I,:},\qquad
R\leftarrow R_{\mathcal I,\mathcal I}.
\]

The surviving off-diagonal covariances must remain. No accepted samples means
no correction; prediction and any completed rolling are retained.
These are marginal gates, not a joint multivariate hypothesis test.
The reported covariance does not account for selection bias from gating.

## 15. Kalman correction

Stack accepted residuals and measurement rows. The local measurement equation is

\[
e\approx H(x_{\mathrm{true}}-x_k^-)+\epsilon_n,
\qquad \epsilon_n\sim\mathcal N(0,R).
\]

Then

\[
S_z=H\Sigma_k^-H^T+R,
\]

\[
K=\Sigma_k^-H^TS_z^{-1},
\]

\[
\boxed{x_k^+=x_k^-+Ke.}
\]

Use the Joseph covariance form:

\[
\boxed{
\Sigma_k^+
=(I-KH)\Sigma_k^-(I-KH)^T+KRK^T.
}
\]

Numerically, solve the Cholesky system

\[
S_zY=H\Sigma_k^-,\qquad K=Y^T,
\]

instead of explicitly forming an inverse. Symmetrize the covariance after
floating-point updates:

\[
\Sigma\leftarrow\tfrac12(\Sigma+\Sigma^T).
\]

Conditioned on frozen correspondences, normals, weights, knots and \(R\), this
is an ordinary linear Gaussian correction. The whole estimator is an
approximate geometric EKF because association and projection depend on the
predicted state and measurements. It is not an exact posterior over all
possible correspondences.

Do not apply the same frame repeatedly as independent measurements. A proper
iterated EKF would retain the original prior and relinearize a single-frame
likelihood; that is not the implemented algorithm.

## 16. Initialization from one curve

For ordered samples \(q_j\), use cumulative detector arc length as initial
state parameters \(u_j=s_j\). With \(L=s_{\mathrm{last}}\), choose

\[
b=-p,\qquad h=\frac{L}{n-p},
\]

so that the valid initial interval is \([0,L]\).

Let \(\mathcal B_{ji}=r_i(u_j)\), let \(Y\in\mathbb R^{M\times2}\) contain the
samples, and let \(D_2\) be the second-difference matrix:

\[
(D_2P)_i=P_i-2P_{i+1}+P_{i+2}.
\]

Initialization solves

\[
\widehat P=
\arg\min_P
\|\mathcal BP-Y\|_F^2+\lambda^2\|D_2P\|_F^2,
\qquad \lambda=0.02.
\]

Equivalently, solve the augmented least-squares system

\[
\begin{bmatrix}\mathcal B\\\lambda D_2\end{bmatrix}P
\approx
\begin{bmatrix}Y\\0\end{bmatrix}.
\]

The current initializer uses unit weights and the heuristic covariance

\[
\Sigma_0=0.3^2I_{2n}.
\]

This is not a calibrated posterior from the least-squares objective.
If a measurement covariance and a Gaussian curvature prior are specified,
a corresponding information matrix could instead be constructed from their
weighted normal equations. The supplied constructor permits an externally
chosen initial covariance.

Initialization already consumes the first detection. Do not update with
that same frame again as independent evidence.

## 17. Forward-only initialization used by the revised demo

The demo observes only the next 10 m, but the rolling state also retains
some controls behind the sensor. It first fits \(n_0=14\) controls to that
forward observation, then prepends \(m=4\) uncertain extrapolated controls,
giving \(n=18\).

Number the prepended rows by \(r=0,\ldots,m-1\), from outermost to innermost,
and set \(d=m-r\). Their means are

\[
P_{\mathrm{pre},r}=(1+d)P_0-dP_1.
\]

Let \(A_{\mathrm{init}}\) map old controls to the enlarged state: its leading
row \(r\) contains coefficients \((1+d,-d)\) on \(P_0,P_1\), and its remaining
rows form an identity matrix.

Independent backward-birth innovations \(\eta_1,\ldots,\eta_m\) accumulate as

\[
P_{\mathrm{pre},r}
=(1+d)P_0-dP_1+
\sum_{j=1}^{d}(d-j+1)\eta_j.
\]

Define \(B_{\mathrm{init},rj}=\max(d-j,0)\) for zero-based \(j=0,\ldots,m-1\),
and zero rows for the retained original controls. Then

\[
T_{\mathrm{init}}=A_{\mathrm{init}}\otimes I_2,\qquad
E_{\mathrm{init}}=B_{\mathrm{init}}\otimes I_2,
\]

\[
\Sigma_{\mathrm{init}}
=T_{\mathrm{init}}\Sigma_0T_{\mathrm{init}}^T
+E_{\mathrm{init}}
(I_m\otimes V_{\mathrm{birth}})
E_{\mathrm{init}}^T.
\]

Set \(b_{\mathrm{new}}=b_{\mathrm{old}}-4\).
The original measured interval is preserved exactly; only the region behind
it is extrapolated. This initialization uses no hidden ground-truth points.

For a measured length near 10 m,

\[
h\approx\frac{10}{14-3}=\frac{10}{11}\ \mathrm m
\]

in the initial parameter gauge. The full 18-control valid interval has length
\(15h\approx13.64\), with about \(4h\approx3.64\) behind and \(11h\approx10\)
ahead initially. The demo uses anchor fraction \(\alpha=4/15\).
Physical lengths later need not equal parameter lengths exactly.

## 18. Revised demo trajectory and detection support

The stationary reference curve is the graph

\[
c(x)=
\begin{bmatrix}
x\\ f(x)
\end{bmatrix},
\qquad
f(x)=2\sin(x/17)+0.4\sin(x/5).
\]

Its slope is

\[
f'(x)=\frac2{17}\cos(x/17)+\frac{0.4}{5}\cos(x/5).
\]

Define true arc length from zero:

\[
S(x)=\int_0^x\sqrt{1+f'(z)^2}\,dz,
\qquad x(s)=S^{-1}(s),
\]

where \(S^{-1}\) is the inverse of the arc-length map.
Numerically this is a dense trapezoidal-integral lookup followed by
one-dimensional interpolation.

At frame \(k\),

\[
s_k=0.45k\ \mathrm m,\qquad
t_k=c(x(s_k)),\qquad
\theta_k=\arctan f'(x(s_k)).
\]

Thus the sensor is on the true curve and points along its tangent.
The time step is \(0.1\) s, corresponding to approximately \(4.5\) m/s.
The odometry increments are computed exactly from these poses.

For detector parameter \(v\in[0,1]\), define

\[
\rho_k=0.65+1.1\left[\frac12+\frac12\sin(0.17k)\right],
\qquad a(v)=L_{\mathrm{det}}v^{\rho_k},
\qquad L_{\mathrm{det}}=10\ \mathrm m.
\]

The noisy detector curve in sensor coordinates is

\[
D_k(v)=
R_k^T\left[c(x(s_k+a(v)))-t_k\right]
+\beta_k+g_k(a(v)).
\]

Therefore its true support is exactly the forward arc-length interval
\([s_k,s_k+10]\). It contains no true observations behind the sensor.
Noise can move detected coordinates slightly beyond those geometric endpoints.
The changing exponent alters detector parameterization without altering true
support.

The simulated smooth field is obtained by sampling the squared-exponential
Gaussian covariance at 12 forward-distance anchors and interpolating them
with a cubic spline. Each frame draws a new field and translation offset.
The detector covariance used by the filter is a tunable approximation to that
simulation; its nonzero white component is an additional conservative term,
not an independently simulated white-noise component.

## 19. Error metric used in the demo

For a world-frame estimated point \(q=(q_x,q_y)\), use the local graph-based
approximation to signed perpendicular error

\[
\epsilon_\perp(q)\approx
\frac{q_y-f(q_x)}{\sqrt{1+f'(q_x)^2}}.
\]

This is exact to first order near the graph, not the exact global closest-point
distance for finite offsets.

Only points in the forward visible region are scored. The implementation
trims 2% of detection length at both ends, using world \(x\) bounds corresponding
to arc distances \(s_k+0.02L_{\mathrm{det}}\) and
\(s_k+0.98L_{\mathrm{det}}\). It computes

\[
E_k=\sqrt{\frac1{M_k}\sum_{j=1}^{M_k}\epsilon_\perp(q_j)^2}.
\]

After a warmup of \(w=\min(20,\lfloor N/4\rfloor)\) frames, the reported
aggregate is

\[
E_{\mathrm{aggregate}}
=\sqrt{\frac1{N-w}\sum_{k=w}^{N-1}E_k^2}.
\]

Detector and estimated curves are sampled separately; this is an illustrative
regional RMS comparison, not a paired-point estimator benchmark.

For seed 7 and 160 frames, the revised demo reports detector RMS 0.1438 m,
filtered RMS 0.0449 m, 18 active controls and 44 rolls.

## 20. Full processing sequence

For each frame after initialization:

1. Convert odometry to the declared old-to-new coordinate convention.
2. Predict all controls with the inverse relative transform.
3. Propagate the full covariance, including common odometry uncertainty.
4. Associate the sensor reference to the predicted curve and roll the window.
5. Sample the current detector curve approximately uniformly in arc length.
6. Associate each measured point to the predicted curve in the current frame.
7. Compute tangents, normals, residuals and the correlated sample covariance.
8. Gate points and retain the matching rows and covariance submatrix.
9. Apply one Kalman correction and the Joseph covariance update.
10. Return \(C_k^+(u)=B_b(u)x_k^+\) on the current valid parameter interval.

Only step 1 needs a pose convention; the state and measurement processing
remain entirely in the current moving frame. World poses are needed only
for visualization or external evaluation.

## 21. Mathematical limits and interpretation

**Static-world prediction versus a changing estimate.** Exact odometry makes
the prediction stationary in world coordinates. Corrections change the
estimated geometry, newborn spans introduce priors, and uncertain odometry
can introduce drift. An estimate cannot be both revised by noisy evidence
and literally identical in world coordinates at every time.

**Unobservable motion directions.** One unlabelled straight curve provides
normal constraints but little or no information about displacement along
its tangent. More samples of that same geometry do not resolve the ambiguity.
Odometry and the state prior carry those directions.

**Association uncertainty.** The Gaussian covariance is conditional on the
chosen nearest points and gates. It excludes uncertainty about branch choice
and the nonlinear sampling/association procedure.

**Temporal detector bias.** The supplied covariance correlates samples within
a frame. A slowly varying bias across frames needs an augmented bias state
with its own dynamics; otherwise successive corrections can be overconfident.

**Odometry bias and revisits.** Persistent pose bias, state/pose correlations
and map reuse call for a joint pose/curve estimator, factor graph, or archived
map. Adding independent control noise cannot recover those missing relations.

**Fixed weights and knots.** Estimating weights or knot positions would make
the state nonlinear and add representation ambiguities. Multiplying every
weight by the same positive number leaves a rational curve unchanged, so
weight estimation also requires a normalization constraint.

**Parameter units.** Initial \(u\) approximates arc length. It is subsequently
a persistent coordinate, not a guaranteed physical metre scale, and is never
reset to the current detector's \([0,1]\) domain.

## 22. Correspondence with the Python implementation

| Mathematical operation | Implementation |
|---|---|
| Rational basis and first derivative | RollingNURBSKF.basis |
| Curve evaluation | RollingNURBSKF.evaluate |
| Moving-frame prediction and covariance | RollingNURBSKF.predict |
| One Gaussian control-window shift | RollingNURBSKF.roll |
| Reference-based roll scheduling | RollingNURBSKF.recenter |
| Nearest-point geometric association | RollingNURBSKF.project |
| Normal correction and covariance | RollingNURBSKF.update_points |
| Arbitrary callable curve input | RollingNURBSKF.update_curve |
| Approximate arc-length resampling | sample_curve |
| Regularized first-frame initialization | RollingNURBSKF.from_points |
| Forward-only demo initialization | initialize_forward_window in demo.py |

The numerical checks cover exact world-frame prediction, covariance rotation,
the odometry Jacobian by finite differences, common odometry-error correlation,
rolling geometry and covariance with unequal rational weights, detector
reparameterization, and a covariance-preserving error-reducing correction.
The revised demo additionally checks sensor position, tangent heading,
10 m forward support, finite errors and fixed animation limits.

Basis and valid-interval implementation reference:
[SciPy BSpline documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.BSpline.html).
The moving-frame, rolling-state and measurement derivations above specify
the accompanying implementation and its stated approximation assumptions.
