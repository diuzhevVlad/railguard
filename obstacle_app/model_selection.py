"""Model profiles for the LiDAR topics supported by the bag application."""
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def model_range_for_topic(topic):
    # This is sensor identity, not the farthest observed return: a Falcon scan
    # facing a nearby wall still requires its own model and axis convention.
    return 400 if topic in ("/iv_points", "iv_points") else 200


def model_files(range_m):
    weights = Path(os.environ.get(f"OBSTACLE_WEIGHTS_{range_m}_DIR",
                                  ROOT / f"{range_m}m_weights"))
    refiner = weights / "refine"
    if not refiner.is_dir():
        refiner = weights / f"refine_{range_m}m"
    return weights, refiner
