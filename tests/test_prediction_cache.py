import tempfile
import unittest
from pathlib import Path

import numpy as np

from obstacle_app.prediction_cache import (load_prediction_cache,
                                           save_prediction_cache)
from railpath.obstacle_filter import RailPrediction


class PredictionCacheTests(unittest.TestCase):
    def test_round_trip_and_source_invalidation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bag = root / "bag"
            bag.mkdir()
            (bag / "metadata.yaml").write_text("test")
            (bag / "bag.db3").write_bytes(b"source")
            weights = root / "weights"
            for relative in ("path/best.pt", "refine/last.pt", "refine/calib.json"):
                path = weights / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(relative.encode())
            predictions = [
                (10, RailPrediction(
                    np.array([[0, -1, 0], [0, -2, 0]], np.float32),
                    np.array([1, 2], np.float32),
                    rail_points=np.array([[-.7, -1, 0], [.7, -1, 0]], np.float32),
                    max_forward_m=150.0)),
                (20, RailPrediction(
                    np.array([[1, -3, 0]], np.float32),
                    np.array([3], np.float32), rail_points=None,
                    max_forward_m=None)),
            ]
            cache = root / "cache"
            saved = save_prediction_cache(cache, bag, "/lidar", weights, predictions)
            self.assertTrue(saved.is_file())
            loaded = load_prediction_cache(cache, bag, "/lidar", weights)
            self.assertEqual(set(loaded), {10, 20})
            np.testing.assert_array_equal(loaded[10].center, predictions[0][1].center)
            np.testing.assert_array_equal(
                loaded[10].rail_points, predictions[0][1].rail_points)
            self.assertEqual(loaded[10].max_forward_m, 150.0)
            self.assertIsNone(loaded[20].max_forward_m)
            self.assertEqual(loaded[20].rail_points.shape, (0, 3))

            (bag / "metadata.yaml").write_text("changed")
            self.assertIsNone(load_prediction_cache(cache, bag, "/lidar", weights))


if __name__ == "__main__":
    unittest.main()
