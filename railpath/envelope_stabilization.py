"""Bounded, single-scan alignment of an envelope to observed rail pairs.

No odometry, GT map, frame history, or obstacle labels are used. Unsupported
scans keep the model prediction. The train's width and height are unchanged.
"""
from dataclasses import replace
import numpy as np
from scipy.ndimage import gaussian_filter1d


def _segmented_anchors(center, rails):
    rails=np.unique(rails,axis=0)
    depth=-rails[:,1]
    cx=np.interp(depth,-center[:,1],center[:,0])
    cz=np.interp(depth,-center[:,1],center[:,2])
    good=(depth>=2)&(depth<=80)&(abs(rails[:,0]-cx)<1.3)&(abs(rails[:,2]-cz)<.5)
    rails,depth,cx=rails[good],depth[good],cx[good]
    bins=np.floor(depth/2).astype(np.int32)
    anchors=[]
    for b in np.unique(bins):
        sides=[]
        for sign in (-1,1):
            p=rails[(bins==b)&((rails[:,0]-cx)*sign>.35)]
            if len(p)<3:break
            sides.append(np.array([np.median(p[:,0]),np.median(p[:,1]),np.quantile(p[:,2],.75)]))
        if len(sides)!=2:continue
        left,right=sides
        if 1.2<right[0]-left[0]<1.8 and abs(left[2]-right[2])<=.12:
            anchors.append((left+right)/2)
    return np.asarray(anchors).reshape(-1,3)


def _surface_anchors(xyz, center):
    # Restrict before interpolation: most returns are outside these rail strips.
    d=-xyz[:,1]
    cd=-center[:,1]
    support=center[(cd>=2)&(cd<=70)]
    endpoints=np.array([[np.interp(t,cd,center[:,0]),-t,np.interp(t,cd,center[:,2])]
                        for t in (2.,70.)])
    support=np.concatenate((support,endpoints))
    lo=support.min(axis=0);hi=support.max(axis=0)
    # This broad box contains every possible point in the exact strips below.
    # It avoids two interpolations for walls/ceiling without downsampling or
    # changing accepted support (padding is wider than the exact strip).
    broad=((d>=2)&(d<=70)&(xyz[:,0]>=lo[0]-1.02)&(xyz[:,0]<=hi[0]+1.02)
           &(xyz[:,2]>=lo[2]-.26)&(xyz[:,2]<=hi[2]+.56))
    # Finite lower and upper bounds also reject NaN/Inf in every coordinate.
    xyz=xyz[broad]
    d=-xyz[:,1]
    cx=np.interp(d,-center[:,1],center[:,0]);cz=np.interp(d,-center[:,1],center[:,2])
    x=xyz[:,0]-cx;h=xyz[:,2]-cz
    p=np.unique(xyz[(abs(abs(x)-.76)<.25)&(h>-.25)&(h<.55)],axis=0)
    d=-p[:,1];cx=np.interp(d,-center[:,1],center[:,0]);cz=np.interp(d,-center[:,1],center[:,2])
    x=p[:,0]-cx;h=p[:,2]-cz;bins=np.floor(d/2).astype(np.int32);anchors=[]
    for b in np.unique(bins):
        sides=[]
        for sign in (-1,1):
            ids=np.flatnonzero((bins==b)&(x*sign>0))
            if len(ids)<10 or np.ptp(d[ids])<.6:break
            top=np.quantile(h[ids],.9)
            ids=ids[abs(h[ids]-top)<.035]
            if len(ids)<3:break
            sides.append(np.array([np.median(p[ids,0]),np.median(d[ids]),top]))
        if len(sides)!=2:continue
        left,right=sides
        if not 1.3<right[0]-left[0]<1.75 or abs(left[2]-right[2])>.10:continue
        depth=(left[1]+right[1])/2
        anchors.append([(left[0]+right[0])/2,-depth,
                        np.interp(depth,-center[:,1],center[:,2])+(left[2]+right[2])/2])
    anchors=np.asarray(anchors).reshape(-1,3)
    if not len(anchors):return anchors
    # A short box/person must not be mistaken for a new track level. Require
    # a continuous run of four paired bins spanning at least five metres.
    chains=np.split(anchors,np.flatnonzero(np.diff(-anchors[:,1])>3.)+1)
    anchors=max(chains,key=len)
    if len(anchors)<4 or np.ptp(anchors[:,1])<5:return np.empty((0,3))
    return anchors


def _robust_line(depth, residual):
    x=np.asarray(depth)/50.
    weights=np.ones(len(x))
    for _ in range(5):
        coef=np.polynomial.polynomial.polyfit(x,residual,1,w=np.sqrt(weights))
        error=residual-np.polynomial.polynomial.polyval(x,coef)
        scale=max(.015,1.4826*np.median(abs(error-np.median(error))))
        weights=np.minimum(1.,2.5*scale/np.maximum(abs(error),1e-8))
    return np.polynomial.polynomial.polyval(x,coef)


def stabilize_rail_prediction(xyz, prediction, lateral_fade_m=10.):
    """Return a new prediction, preserving input arrays and physical dimensions.

    Segmented rail pairs correct lateral bias by at most 0.5 m. The correction
    fades back to the raw model beyond the last observed pair. None retains
    the legacy constant offset for controlled comparisons. A separate
    high-support geometric check corrects a clearly too-low track level by
    at most 0.4 m, only near its observed support. Small height discrepancies
    are deliberately left alone to preserve small-object sensitivity.
    """
    if lateral_fade_m is not None and (not np.isfinite(lateral_fade_m) or lateral_fade_m<0):
        raise ValueError('Lateral fade must be finite and nonnegative, or None')
    original=np.asarray(prediction.center)
    rails=prediction.rail_points
    if (len(original)<2 or rails is None or not np.isfinite(original).all()
            or np.any(np.diff(-original[:,1])<=0)):
        return prediction
    # A prepared prediction can be reused for display and classification.
    if prediction.details and prediction.details.get('envelope_stabilization',{}).get('version')==2:
        return prediction
    center=original.copy();depth=-center[:,1]
    rails=np.asarray(rails)
    rails=rails[np.isfinite(rails).all(axis=1)]
    anchors=_segmented_anchors(center,rails)
    lateral=False
    if len(anchors)>=5 and np.ptp(anchors[:,1])>=10:
        ad=-anchors[:,1]
        residual=anchors[:,0]-np.interp(ad,depth,center[:,0])
        residual=gaussian_filter1d(residual,1)
        center[:,0]+=np.interp(depth,ad,np.clip(residual,-.5,.5))
        lateral=True
    surface=_surface_anchors(np.asarray(xyz),center)
    height=False
    if len(surface):
        sd=-surface[:,1]
        residual=surface[:,2]-np.interp(sd,depth,center[:,2])
        if np.median(residual)>=.12:
            fitted=_robust_line(sd,residual)
            residual=np.clip(residual,fitted-.08,fitted+.08)
            weight=np.minimum(np.clip((depth-sd[0]+5)/5,0,1),np.clip((sd[-1]+5-depth)/5,0,1))
            center[:,2]+=weight*np.interp(depth,sd,np.clip(gaussian_filter1d(residual,.8),-.4,.4))
            height=True
    if lateral and lateral_fade_m is not None:
        support_end=float(-anchors[-1,1])
        weight=((depth<=support_end).astype(float) if lateral_fade_m==0 else
                np.clip((support_end+lateral_fade_m-depth)/lateral_fade_m,0,1))
        center[:,0]=original[:,0]+weight*(center[:,0]-original[:,0])
    details=dict(prediction.details or {})
    details['envelope_stabilization']=dict(version=2,lateral_fade_m=lateral_fade_m,paired_rail_bins=len(anchors),
        surface_rail_bins=len(surface),lateral_applied=lateral,height_applied=height,
        max_lateral_change_m=float(np.max(abs(center[:,0]-original[:,0]))),
        max_height_change_m=float(np.max(abs(center[:,2]-original[:,2]))))
    lengths=np.r_[0.,np.cumsum(np.linalg.norm(np.diff(center,axis=0),axis=1))]+prediction.distance_along_m[0]
    continuation=prediction.continuation
    if continuation is not None and len(continuation):
        continuation=np.asarray(continuation).copy()
        for axis in (0,2):
            continuation[:,axis]+=np.interp(-continuation[:,1],depth,center[:,axis]-original[:,axis])
    return replace(prediction,center=center,distance_along_m=lengths.astype(np.float32),
                   continuation=continuation,details=details)
