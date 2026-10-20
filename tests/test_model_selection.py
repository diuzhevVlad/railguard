from obstacle_app.model_selection import model_range_for_topic, model_files
from obstacle_app.learned_detection import LearnedDetectionManager, LearnedJob, RailGuardRuntime


def test_topic_identity_not_observed_distance():
    for topic in ("/iv_points", "iv_points"):
        assert model_range_for_topic(topic) == 400
    for topic in ("/lidar_points", "/sensing/lidar/hesai128/pointcloud"):
        assert model_range_for_topic(topic) == 200


def test_checkpoint_layouts_and_overrides(tmp_path, monkeypatch):
    for rows in (200, 400):
        root = tmp_path/f"{rows}m_weights"
        (root/f"refine_{rows}m").mkdir(parents=True)
        monkeypatch.setenv(f"OBSTACLE_WEIGHTS_{rows}_DIR", str(root))
        assert model_files(rows) == (root, root/f"refine_{rows}m")
        assert RailGuardRuntime(root).refiner == root/f"refine_{rows}m"
        (root/"refine").mkdir()
        assert model_files(rows) == (root, root/"refine")


def test_jobs_keep_their_model_when_switching_bags(tmp_path, monkeypatch):
    from obstacle_app import learned_detection
    created = []
    class Runtime:
        def __init__(self, weights, refiner, rows):
            self.range_m, self.device = rows, "test"
            self.closed = False
            created.append(self)
        def load(self):
            pass
        def close(self):
            self.closed = True
    manager = LearnedDetectionManager(runtime_factory=Runtime)
    from types import SimpleNamespace
    monkeypatch.setattr(learned_detection, "BagOdometry", lambda bag,topic,*args: SimpleNamespace(bag=bag,topic=topic,close=lambda:None))
    try:
        for key, rows in (("normal",200), ("falcon",400), ("normal_again",200)):
            job = LearnedJob(key, key, "/iv_points" if rows == 400 else "/lidar_points",
                             key, model_range_m=rows, model_id=f"{rows}m_weights")
            manager.bags[key] = tmp_path/key
            manager._load(job)
            assert job.state == "done", job.message
            provider = manager.detector(job).detector.predictions
            assert provider.runtime.range_m == rows
        assert [r.range_m for r in created] == [200,400]
        assert manager.detectors["normal"].detector.predictions.runtime is created[0]
        assert manager.detectors["falcon"].detector.predictions.runtime is created[1]
        too_far = LearnedJob("far", "source/normal", "/lidar_points", "far",
                             max_forward_m=400, model_range_m=200, model_id="200m_weights")
        manager._load(too_far)
        assert too_far.state == "failed" and "200 m" in too_far.message
    finally:
        manager.close()
    assert all(r.closed for r in created)
