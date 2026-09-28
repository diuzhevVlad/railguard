"""RailPath inference and obstacle filtering.

Load neural dependencies only when inference is requested, so importing
railpath.obstacle_filter needs only NumPy and SciPy.
"""
__all__ = ["RailPath", "make_image"]


def __getattr__(name):
    if name in __all__:
        from .inference import RailPath, make_image
        globals().update(RailPath=RailPath, make_image=make_image)
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
