"""One spawned inference worker per model range, isolating railpath's ROWS."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
import threading

from .learned_detection import RailGuardRuntime, scan_input


_runtime = None


def _initialize(weights, refiner, device, range_m):
    global _runtime
    os.environ["RAILPATH_ROWS"] = str(range_m)
    _runtime = RailGuardRuntime(weights, device, refiner=refiner)


def _load():
    _runtime.load()
    return _runtime.device, _runtime.range_m


def _predict(xyz, ring):
    return _runtime.predict_arrays(xyz, ring)


class ProcessRailRuntime:
    def __init__(self, weights, refiner, range_m, device=None):
        self.weights, self.refiner = weights, refiner
        self.range_m = range_m
        self.device = device or os.environ.get("OBSTACLE_RAIL_DEVICE", "auto")
        self.pool = None
        self.lock = threading.Lock()
        self.closed = False

    def load(self):
        with self.lock:
            if self.closed:
                raise RuntimeError("Model worker has shut down")
            if self.pool is not None:
                return
            pool = ProcessPoolExecutor(
                max_workers=1, mp_context=multiprocessing.get_context("spawn"),
                initializer=_initialize,
                initargs=(self.weights, self.refiner, self.device, self.range_m))
            try:
                device, actual_range = pool.submit(_load).result()
                if actual_range != self.range_m:
                    raise ValueError(f"Expected {self.range_m} m weights, got {actual_range} m")
            except Exception:
                pool.shutdown(wait=True, cancel_futures=True)
                raise
            self.device, self.pool = device, pool

    def predict(self, msg):
        self.load()
        xyz, ring = scan_input(msg)
        with self.lock:
            if self.closed:
                raise RuntimeError("Model worker has shut down")
            # Only arrays cross the process boundary, not ROS-specific classes.
            return self.pool.submit(_predict, xyz, ring).result()

    def close(self):
        with self.lock:
            self.closed = True
            if self.pool is not None:
                self.pool.shutdown(wait=True, cancel_futures=True)
                self.pool = None
