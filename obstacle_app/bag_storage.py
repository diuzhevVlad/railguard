"""Conservative preflight estimates for app-compatible ROS 2 bag exports."""
from pathlib import Path
import numpy as np


def estimate_export_bytes(reader, source_bag: Path, start_ns, stop_ns, compression=None):
    if compression not in (None, 'zstd'):
        raise ValueError('Unsupported bag compression; choose None or zstd')
    size = sum(p.stat().st_size for p in source_bag.iterdir() if p.suffix in ('.db3', '.mcap'))
    end = min(stop_ns, reader.end_time)
    fraction = max(1, end-start_ns) / max(1, reader.duration)
    ratio = 1.
    if compression == 'zstd':
        import zstandard
        compressor = zstandard.ZstdCompressor(level=3)
        ratios = []
        for stamp in np.linspace(start_ns, max(start_ns,end-1), 12).astype(np.int64):
            message = next(reader.messages(start=int(stamp), stop=end), None)
            if message is not None and len(message[2]):
                raw = message[2]
                ratios.append(len(compressor.compress(raw))/len(raw))
        if ratios:
            # Largest observed ratio, 50% margin, and per-bag fixed overhead.
            ratio = max(ratios)
    return int(size*fraction*ratio*1.5) + 128*1024**2
