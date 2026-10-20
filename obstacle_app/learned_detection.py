"""Lazy single-scan RailGuard inference; independent of the map/odometry cache."""
from dataclasses import dataclass, replace
from pathlib import Path
import os
import sys
import threading
import time
import uuid

import numpy as np

from .cloud import xyz_view, FIELD_FORMATS
from .detection import RailPrediction
from .scan_detection import ScanDetector
from .world_map import source_fingerprint
from railpath.tracked_detection import CANDIDATE_CONFIG, detect_candidates
from .tracked_bag_detection import TrackedBagDetector
from .model_selection import model_range_for_topic, model_files

ROOT = Path(__file__).resolve().parent.parent
DETECTION_FORWARD_LIMIT_M = 150.0


def scan_input(msg):
    """Keep XYZ and ring aligned, including organized rows and padded records."""
    view = xyz_view(msg)
    xyz = np.column_stack([view[k].reshape(-1) for k in ('x', 'y', 'z')])
    valid = np.isfinite(xyz).all(1) & (np.einsum('ij,ij->i', xyz, xyz) > .25)
    ring = None
    field = next((f for f in msg.fields if f.name == 'ring'), None)
    if field is not None:
        if field.count != 1 or field.datatype not in FIELD_FORMATS:
            raise ValueError('Unsupported LiDAR ring field')
        dtype = np.dtype(('>' if msg.is_bigendian else '<') + FIELD_FORMATS[field.datatype])
        if field.offset < 0 or field.offset + dtype.itemsize > msg.point_step:
            raise ValueError('Invalid LiDAR ring field offset')
        raw = np.ndarray((msg.height, msg.width), dtype=dtype, buffer=msg.data,
                         offset=field.offset, strides=(msg.row_step, msg.point_step)).reshape(-1)
        if not np.isfinite(raw[valid]).all():
            raise ValueError('Nonfinite LiDAR ring IDs')
        ring = raw[valid].astype(np.int64)
        if np.any((ring < 0) | (ring >= 128)):
            raise ValueError('RailGuard expects Pandar128 ring IDs in 0..127')
    return xyz[valid].astype(np.float32, copy=False), ring


def adapt_prediction(out, elapsed_ms=0., device='unknown'):
    """Use the full finite predicted curve; retain the vote horizon for display."""
    curve = np.asarray(out['centerline'], dtype=np.float32)
    if curve.ndim != 2 or curve.shape[1] != 3:
        raise ValueError('RailGuard returned an invalid centerline')
    horizon = float(out.get('horizon', 0))
    if not np.isfinite(horizon):
        horizon = 0.
    sigma = out.get('sigma')
    sigma = np.asarray(sigma, dtype=float) if sigma is not None else None
    finite = np.isfinite(curve).all(1)
    if sigma is not None:
        if sigma.shape != (len(curve),):
            raise ValueError('RailGuard uncertainty shape does not match its curve')
        finite &= np.isfinite(sigma) & (sigma >= 0)
    # Never connect across invalid sections or classify extrapolated rows as observed.
    invalid = np.flatnonzero(~finite)
    end = int(invalid[0]) if len(invalid) else len(curve)
    usable = curve[:end]
    distance = -usable[:, 1]
    supported_count = int(np.count_nonzero((distance > 0) & (distance <= horizon)))
    path = usable
    continuation = usable[max(0,supported_count-1):] if len(usable)>supported_count else np.empty((0,3),np.float32)
    status = 'ready' if len(path) >= 2 else 'unavailable'
    center, lengths = np.empty((0,3),np.float32), np.empty(0,np.float32)
    if status == 'ready':
        lengths = np.r_[0., np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
        unique = np.r_[True, np.diff(lengths)>1e-6]
        path, lengths = path[unique], lengths[unique]
        if len(lengths) >= 2:
            # Dense samples avoid holes in the envelope's nearest-center query.
            sample = np.r_[np.arange(0., lengths[-1], .25), lengths[-1]]
            center = np.column_stack([np.interp(sample,lengths,path[:,a]) for a in range(3)]).astype(np.float32)
            first_step = np.linalg.norm(path[1]-path[0])
            first_depth_step = float(-path[1,1]+path[0,1])
            origin_offset = max(0., float(-path[0,1]))*first_step/max(first_depth_step,1e-6)
            lengths = (sample+origin_offset).astype(np.float32)
        else:
            status='unavailable';center=np.empty((0,3),np.float32);lengths=np.empty(0,np.float32)
    sigma_values = [float(v) if np.isfinite(v) else None for v in sigma[:end]] if sigma is not None else None
    checked_sigma = sigma[:end][distance <= DETECTION_FORWARD_LIMIT_M] if sigma is not None else None
    details = dict(source='railguard_model_refiner', status=status, horizon_m=horizon, model_range_m=len(curve),
                   checked_forward_m=min(DETECTION_FORWARD_LIMIT_M, float(-center[-1,1])) if len(center) else 0.,
                   lookahead_limit_m=DETECTION_FORWARD_LIMIT_M, inference_ms=round(elapsed_ms,2), device=device,
                   sigma_m=sigma_values,
                   sigma_max_checked_m=float(np.max(checked_sigma)) if checked_sigma is not None and len(checked_sigma) else None)
    rails = np.asarray(out.get('rails', np.empty((0, 3))), dtype=np.float32)
    if rails.ndim != 2 or rails.shape[1] != 3:
        raise ValueError('RailGuard returned invalid segmented rail points')
    rails = rails[np.isfinite(rails).all(1)]
    return RailPrediction(center, lengths, continuation, details, rails, DETECTION_FORWARD_LIMIT_M)


class RailGuardRuntime:
    """One shared GPU model, serialized access; no frame history or odometry."""
    def __init__(self, weights=None, device=None, refiner=None):
        self.weights=Path(weights or os.environ.get('OBSTACLE_WEIGHTS_DIR',ROOT/'400m_weights'))
        self.refiner=Path(refiner) if refiner is not None else next(
            (self.weights/name for name in ('refine','refine_200m','refine_400m')
             if (self.weights/name).is_dir()), self.weights/'refine')
        self.device=device or os.environ.get('OBSTACLE_RAIL_DEVICE',os.environ.get('RAILGUARD_DEVICE','auto'))
        self.lock=threading.Lock()
        self.model=None
        self.range_m=None

    def load(self):
        with self.lock:
            if self.model is not None:
                return
            for file in [self.weights/'path/best.pt',self.refiner/'last.pt',self.refiner/'calib.json']:
                if not file.is_file():
                    raise ValueError(f'Missing RailGuard file: {file}')
            import torch
            # Establish the decoder/refiner geometry before the neural modules
            # import ROWS. Do not resize/interpolate learned checkpoint tensors.
            refiner_state=torch.load(self.refiner/'last.pt',map_location='cpu',weights_only=True)
            rows=int(refiner_state['kap_w'].shape[0])-1
            knots=refiner_state['knots'].numpy().astype(float)
            if (rows not in (200,400) or not np.array_equal(knots,np.arange(0,rows+1,25))
                    or tuple(refiner_state['kap_w'].shape)!=(rows+1,len(knots))
                    or tuple(refiner_state['tok_pe'].shape)!=(rows//4,256)
                    or tuple(refiner_state['q_embed'].shape)!=(rows//25-1,256)):
                raise ValueError('Unsupported RailGuard refiner checkpoint geometry')
            configured=os.environ.get('RAILPATH_ROWS')
            loaded=sys.modules.get('railpath.data')
            if ((configured is not None and int(configured)!=rows)
                    or (loaded is not None and loaded.ROWS!=rows)):
                raise ValueError(f'Checkpoint requires {rows} m. Restart the process with RAILPATH_ROWS={rows}.')
            os.environ['RAILPATH_ROWS']=str(rows)
            self.range_m=rows
            del refiner_state
            from railpath import RailPath
            from railpath.refiner import RefinerRunner
            if self.device=='auto':
                self.device='cuda' if torch.cuda.is_available() else 'cpu'
            if self.device=='cuda' and not torch.cuda.is_available():
                raise ValueError('CUDA unavailable. Enable Docker GPU access or set OBSTACLE_RAIL_DEVICE=cpu.')
            torch.set_num_threads(max(1,min(int(os.environ.get('RAILGUARD_CPU_THREADS','4')),os.cpu_count() or 1)))
            model=RailPath(self.weights/'path/best.pt',refiner=self.refiner,device=self.device)
            state=torch.load(self.weights/'path/best.pt',map_location='cpu',weights_only=True)
            missing=set(model.model.state_dict())-set(state)
            if missing:
                raise ValueError(f'RailGuard checkpoint is missing {len(missing)} model tensors')
            del state
            model.refiner=RefinerRunner(self.refiner,knots,self.weights/'path/best.pt',model.device)
            # Warm up both networks so the first displayed frame doesn't compile kernels.
            image=np.zeros((8,128,1000),np.float32)
            model.predict_image(image)
            model.refiner.run(image,np.full(rows,-1.1,np.float32),50.,
                dict(params=np.zeros(len(knots)+2,np.float32),evidence=np.zeros(len(knots),np.float32)))
            self.model=model

    def predict(self, msg):
        xyz,ring=scan_input(msg)
        return self.predict_arrays(xyz,ring)

    def predict_arrays(self, xyz, ring=None):
        self.load()
        started=time.perf_counter()
        if not len(xyz):
            return adapt_prediction(dict(centerline=np.full((self.range_m,3),np.nan),sigma=None,horizon=0),device=self.device)
        with self.lock:
            # Sparse-ring interpolation is scan-specific, including after seeks and bag switches.
            self.model.sparse=None
            out=self.model.predict(xyz,ring=ring)
        return adapt_prediction(out,(time.perf_counter()-started)*1000,self.device)


class LearnedRailProvider:
    def __init__(self,runtime):
        self.runtime=runtime

    def predict(self,timestamp_ns,point_cloud=None):
        if point_cloud is None:
            raise ValueError('Learned rail prediction requires the current scan')
        return self.runtime.predict(point_cloud)


@dataclass
class LearnedJob:
    id: str
    bag_id: str
    topic: str
    fingerprint: str
    train_width_m: float=2.1
    train_height_m: float=3.0
    max_forward_m: float=150.0
    state: str='queued'
    progress: float=0.
    message: str='Loading RailGuard model and refiner'
    device: str='pending'
    model_range_m: int | None=None
    initial_speed_m_s: float=0.
    model_id: str=''


class LearnedDetectionManager:
    def __init__(self, detector_factory=None, runtime_factory=None):
        # Import lazily to avoid a cycle when the spawned worker imports this module.
        from .model_worker import ProcessRailRuntime
        self.runtime_factory=runtime_factory or ProcessRailRuntime
        self.runtimes={}
        self.factory=detector_factory
        self.jobs={}
        self.detectors={}
        self.lock=threading.Lock()
        self.closed=False

    def runtime_for(self, range_m):
        with self.lock:
            if self.closed:
                raise RuntimeError("Detection manager has shut down")
            if range_m not in self.runtimes:
                weights,refiner=model_files(range_m)
                self.runtimes[range_m]=self.runtime_factory(weights,refiner,range_m)
            return self.runtimes[range_m]

    def close(self):
        with self.lock:
            self.closed=True
            runtimes=list(self.runtimes.values())
        for runtime in runtimes:
            runtime.close()

    def start(self,bag_id,bag,topic,train_width_m=2.1,train_height_m=3.0,max_forward_m=150.,initial_speed_m_s=0.):
        fingerprint=source_fingerprint(bag)
        with self.lock:
            for job in self.jobs.values():
                if (job.bag_id,job.topic,job.fingerprint,job.train_width_m,job.train_height_m,job.max_forward_m,job.initial_speed_m_s)==(bag_id,topic,fingerprint,train_width_m,train_height_m,max_forward_m,initial_speed_m_s) and job.state!='failed':
                    return job
            if len(self.jobs)>=16:
                expired=next((key for key,value in self.jobs.items() if value.state in ('done','failed')),None)
                if expired:
                    self.jobs.pop(expired);self.detectors.pop(expired,None)
            job=LearnedJob(uuid.uuid4().hex,bag_id,topic,fingerprint,train_width_m,train_height_m,max_forward_m)
            self.jobs[job.id]=job
            job.initial_speed_m_s=initial_speed_m_s
            job.model_range_m=model_range_for_topic(topic)
            job.model_id=f'{job.model_range_m}m_weights'
        threading.Thread(target=self._load,args=(job,),daemon=True).start()
        return job

    def _load(self,job):
        try:
            job.state='running';job.progress=.1
            if self.factory is None:
                runtime=self.runtime_for(job.model_range_m)
                job.message=f'Loading {job.model_id} and refiner'
                if job.max_forward_m>job.model_range_m:
                    raise ValueError(f'Detection distance exceeds the selected {job.model_range_m} m model')
                runtime.load()
                detector=ScanDetector(LearnedRailProvider(runtime),
                                      filter_config=CANDIDATE_CONFIG, candidate_detector=detect_candidates)
                job.device=runtime.device
                if runtime.range_m != job.model_range_m:
                    raise ValueError('Loaded model does not match the selected LiDAR')
            else:
                detector=self.factory(job.bag_id,job.topic);job.device='test'
            detector.filter_config=replace(detector.filter_config,train_width_m=job.train_width_m,train_height_m=job.train_height_m)
            detector.max_forward_m=job.max_forward_m
            with self.lock:
                self.detectors[job.id]=detector
                model_label=f' {job.model_range_m} m' if job.model_range_m else ''
                job.state='done';job.progress=1.;job.message=f'RailGuard{model_label} + refiner ready ({job.device})'
        except Exception as exc:
            job.state='failed';job.message=str(exc)

    def get(self,job_id):
        with self.lock:
            return self.jobs.get(job_id)

    def detector(self,job):
        """Open independent tracking for one playback connection or frame preview."""
        with self.lock:
            detector = self.detectors[job.id]
        # Each websocket owns its stream. A standalone frame request is a fresh
        # one-scan preview, never a request to replay hidden historical scans.
        if self.factory is not None:
            return detector
        return TrackedBagDetector(detector, job.initial_speed_m_s)


def prediction_payload(prediction):
    return dict(rail_center=prediction.center.tolist(),
                rail_points=prediction.rail_points.tolist() if prediction.rail_points is not None else [],
                rail_continuation=prediction.continuation.tolist() if prediction.continuation is not None else [],
                rail_prediction=prediction.details)
