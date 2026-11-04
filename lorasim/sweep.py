"""Accuracy as a function of one parameter, for all algorithms at once."""
from dataclasses import replace

import numpy as np

from .config import ALGORITHMS, MAX_GRID_POINTS
from .simulation import grid_size, simulate

PARAMETERS = ("Grid spacing", "Zone size", "Scan frames", "Localization frames", "Scan positions",
              "Station correlation", "Station height", "k")

LOG_AXIS = {"Grid spacing", "Zone size", "Scan frames", "Localization frames", "Scan positions"}
UNITS = {"Grid spacing": "m", "Zone size": "m (longer side)", "Scan frames": "frames per station",
         "Localization frames": "frames per station", "Scan positions": "positions per grid node",
         "Station correlation": "ρ", "Station height": "m above ground", "k": "neighbours"}


def variants(cfg, parameter):
    """(x value, config) pairs for the chosen parameter, starting from the current settings."""
    out = []
    if parameter == "Grid spacing":
        limit = min(cfg.width, cfg.height) / 2
        for f in (0.25, 0.5, 1, 2, 4, 8):
            s = float(np.clip(cfg.grid_spacing * f, 1.0, 1000.0))
            if s <= limit:
                out.append((s, cfg.with_(grid_spacing=s)))
    elif parameter == "Zone size":
        side = max(cfg.width, cfg.height)
        for target in (100, 200, 500, 1000, 2000, 5000, 10000):
            f = target / side
            stations = tuple((x * f, y * f) for x, y in cfg.manual_stations)
            out.append((target, cfg.with_(width=cfg.width * f, height=cfg.height * f,
                                          grid_spacing=cfg.grid_spacing * f, manual_stations=stations)))
    elif parameter == "Scan frames":
        out = [(n, cfg.with_(scan_frames=n)) for n in (10, 20, 50, 100, 200, 500)]
    elif parameter == "Localization frames":
        out = [(n, cfg.with_(locate_frames=n)) for n in (1, 2, 5, 10, 20, 50, 100, 200)]
    elif parameter == "Scan positions":
        out = [(m, cfg.with_(scan_positions=m)) for m in (1, 2, 4, 8, 16) if m <= cfg.scan_frames]
    elif parameter == "Station correlation":
        out = [(rho, cfg.with_(env=replace(cfg.env, station_correlation=rho)))
               for rho in (0.0, 0.25, 0.5, 0.75, 0.9)]
    elif parameter == "Station height":
        out = [(h, cfg.with_(station_height=h)) for h in (1.0, 1.5, 2.0, 3.0, 5.0, 8.0)]
    elif parameter == "k":
        out = [(k, cfg.with_(k=k)) for k in range(1, 11)]
    else:
        raise ValueError(parameter)
    unique = {}
    for x, c in out:
        if grid_size(c) <= MAX_GRID_POINTS:
            unique.setdefault(x, c)
    return sorted(unique.items())


def run_sweep(cfg, parameter, progress=None):
    items = variants(cfg, parameter)
    xs = np.array([x for x, _ in items], dtype=float)
    mean = {a: np.empty(len(items)) for a in ALGORITHMS}
    p90 = {a: np.empty(len(items)) for a in ALGORITHMS}
    for i, (_, c) in enumerate(items):
        res = simulate(c, algorithms=ALGORITHMS)
        for a in ALGORITHMS:
            mean[a][i] = res.errors[a].mean()
            p90[a][i] = np.percentile(res.errors[a], 90)
        if progress:
            progress(i + 1, len(items))
    return xs, mean, p90
