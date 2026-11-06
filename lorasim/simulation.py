"""Scan, localization and accuracy evaluation."""
from dataclasses import dataclass

import numpy as np

from . import propagation, radio
from .config import K_NONE, K_RAYLEIGH, MAX_GRID_POINTS, NO_SIGNAL_RSSI, Config

MAX_SAMPLES_PER_CHUNK = 2_000_000


class GridTooLarge(ValueError):
    def __init__(self, points):
        super().__init__(f"The grid has {points:,} points; the limit is {MAX_GRID_POINTS:,}. "
                         "Increase the grid spacing or reduce the area.")
        self.points = points


def grid_axis(length, spacing):
    """Nodes from edge to edge; the actual spacing is the largest one not above the requested one."""
    count = int(np.ceil(length / spacing - 1e-9)) + 1
    return np.linspace(0.0, length, count)


def actual_spacing(cfg):
    gx, gy = grid_axis(cfg.width, cfg.grid_spacing), grid_axis(cfg.height, cfg.grid_spacing)
    return float(gx[1] - gx[0]), float(gy[1] - gy[0])


def grid_size(cfg):
    return len(grid_axis(cfg.width, cfg.grid_spacing)) * len(grid_axis(cfg.height, cfg.grid_spacing))


def station_positions(cfg, rng):
    w, h = cfg.width, cfg.height
    if cfg.station_layout == "Corners":
        pos = [(0, 0), (w, 0), (w, h), (0, h)]
    elif cfg.station_layout == "Side midpoints":
        pos = [(w / 2, 0), (w, h / 2), (w / 2, h), (0, h / 2)]
    elif cfg.station_layout == "Random":
        pos = rng.uniform((0, 0), (w, h), size=(4, 2))
    else:
        pos = cfg.manual_stations
    return np.asarray(pos, dtype=float)


@dataclass
class Scenario:
    cfg: Config
    stations: np.ndarray        # (S, 2)
    grid_x: np.ndarray
    grid_y: np.ndarray
    grid: np.ndarray            # (P, 2), x changes fastest
    shadow: list                # one ShadowField per station
    noise_floor: float
    snr_limit: float
    airtime_ms: float           # one frame on air
    slot_ms: float              # one frame plus the guard gap
    rngs: dict


def build_scenario(cfg):
    points = grid_size(cfg)
    if points > MAX_GRID_POINTS:
        raise GridTooLarge(points)
    seq = np.random.SeedSequence(cfg.seed)
    names = ("stations", "shadow", "common_shadow", "tests", "scan", "locate")   # order fixed: same seed, same terrain
    rngs = {n: np.random.default_rng(s) for n, s in zip(names, seq.spawn(len(names)))}

    env, r = cfg.env, cfg.radio
    stations = station_positions(cfg, rngs["stations"])
    sigma = env.shadowing_sigma
    shadow = [propagation.ShadowField(cfg.width, cfg.height, sigma, env.correlation_distance, rngs["shadow"])
              for _ in stations]
    if env.station_correlation > 0 and sigma > 0:
        common = propagation.ShadowField(cfg.width, cfg.height, sigma, env.correlation_distance,
                                         rngs["common_shadow"])
        shadow = [propagation.CorrelatedShadow(common, own, env.station_correlation) for own in shadow]
    gx, gy = grid_axis(cfg.width, cfg.grid_spacing), grid_axis(cfg.height, cfg.grid_spacing)
    xx, yy = np.meshgrid(gx, gy)
    airtime = radio.time_on_air_ms(r.spreading_factor, r.bandwidth_khz, r.payload_bytes)
    return Scenario(cfg, stations, gx, gy, np.column_stack([xx.ravel(), yy.ravel()]),
                    shadow, radio.noise_floor_dbm(r.bandwidth_khz, r.noise_figure_db),
                    radio.SNR_LIMIT_DB[r.spreading_factor], airtime, airtime + r.guard_ms, rngs)


def mean_power(sc, points):
    """Mean received power (dBm) at each point from each station, without small-scale effects."""
    cfg, r = sc.cfg, sc.cfg.radio
    out = np.empty((len(points), len(sc.stations)))
    for s, st in enumerate(sc.stations):
        d = np.hypot(points[:, 0] - st[0], points[:, 1] - st[1])
        p = r.tx_power_dbm + 2 * r.antenna_gain_dbi - propagation.link_loss_db(
            d, cfg.station_height, cfg.receiver_height, cfg.env, r.frequency_mhz)
        p -= sc.shadow[s](points[:, 0], points[:, 1])
        out[:, s] = p
    return out


@dataclass
class Measurement:
    rssi: np.ndarray        # (P, S) mean reported RSSI, NO_SIGNAL_RSSI if nothing arrived
    received: np.ndarray    # (P, S) number of frames received
    frames: int


def measure(sc, points, frames, rng, positions=1):
    """The receiver stands at each point and averages `frames` frames from every station.

    Multipath: every receiver position has its own fixed part (Rice with the K-factor); a share
    env.dynamic_share of the scattered power is new in every frame, the line of sight stays. With
    several positions the receiver is moved around the point (positions more than half a wavelength
    apart, close compared with the shadowing correlation distance) and the frames are spread over
    them in turn, so averaging also reduces the fixed part.
    """
    env = sc.cfg.env
    base = mean_power(sc, points)
    n_pos = max(1, min(int(positions), frames))
    share = env.dynamic_share
    shape = base.shape + (n_pos,)
    if share > 0:
        los = propagation.los_amplitude(env.k_factor_db, K_RAYLEIGH, K_NONE)
        fixed = propagation.rice_amplitude(env.k_factor_db, shape, rng, K_RAYLEIGH, K_NONE)
        offset = propagation.fading_stats_db(env.k_factor_db, share, K_RAYLEIGH, K_NONE)[0]
    else:
        fixed = propagation.multipath_db(env.k_factor_db, shape, rng, K_RAYLEIGH, K_NONE)
    position_of_frame = np.arange(frames) % n_pos
    n_st = base.shape[1]
    rssi = np.empty_like(base)
    received = np.empty(base.shape, dtype=int)
    chunk = max(1, MAX_SAMPLES_PER_CHUNK // (n_st * frames))
    for a in range(0, len(points), chunk):
        b = slice(a, a + chunk)
        if share > 0:
            multipath = propagation.dynamic_gain_db(fixed[b][:, :, position_of_frame], share, rng, los) - offset
        else:
            multipath = fixed[b][:, :, position_of_frame]
        power = (base[b, :, None] + multipath
                 + rng.normal(0.0, env.frame_noise_sigma, (len(base[b]), n_st, frames)))
        ok = power - sc.noise_floor >= sc.snr_limit       # frame decoded when its SNR is above the limit
        count = ok.sum(axis=2)
        with np.errstate(invalid="ignore", divide="ignore"):
            rssi[b] = np.where(ok, radio.reported_rssi(power, sc.noise_floor), 0).sum(axis=2) / count
        received[b] = count
    rssi[received == 0] = NO_SIGNAL_RSSI
    return Measurement(rssi, received, frames)


def nearest_neighbours(map_features, map_points, query, k):
    """Weighted k nearest neighbours in feature space (k=1 gives plain NN)."""
    k = min(k, len(map_points))
    g2 = (map_features ** 2).sum(axis=1)
    out = np.empty((len(query), 2))
    for a in range(0, len(query), 100):
        q = query[a:a + 100]
        d2 = (q ** 2).sum(axis=1)[:, None] + g2[None, :] - 2 * q @ map_features.T
        idx = np.argpartition(d2, k - 1, axis=1)[:, :k]
        dist = np.sqrt(np.maximum(np.take_along_axis(d2, idx, axis=1), 0.0))
        w = 1.0 / (dist + 1e-6)
        out[a:a + 100] = (w[:, :, None] * map_points[idx]).sum(axis=1) / w.sum(axis=1, keepdims=True)
    return out


@dataclass
class Result:
    scenario: Scenario
    scan: Measurement
    test_points: np.ndarray
    located: Measurement
    estimates: dict             # algorithm -> (T, 2)
    errors: dict                # algorithm -> (T,)
    scan_time_s: float          # including the duty-cycle limit
    scan_radio_time_s: float    # frames on air only
    locate_time_s: float
    locates_per_hour: float     # allowed by the duty cycle of one station
    frame_loss: float

    @property
    def cfg(self):
        return self.scenario.cfg

    def stats(self, algorithm=None):
        e = self.errors[algorithm or self.cfg.algorithm]
        return {"mean": float(e.mean()), "median": float(np.median(e)),
                "p90": float(np.percentile(e, 90)), "max": float(e.max())}


def simulate(cfg, algorithms=None, test_points=None):
    algorithms = algorithms or (cfg.algorithm,)
    sc = build_scenario(cfg)
    scan = measure(sc, sc.grid, cfg.scan_frames, sc.rngs["scan"], cfg.scan_positions)
    if test_points is None:
        test_points = sc.rngs["tests"].uniform((0, 0), (cfg.width, cfg.height), size=(cfg.test_points, 2))
    test_points = np.asarray(test_points, dtype=float)
    located = measure(sc, test_points, cfg.locate_frames, sc.rngs["locate"], cfg.locate_positions)

    estimates = {}
    for alg in algorithms:
        k = 1 if alg == "NN" else cfg.k
        estimates[alg] = nearest_neighbours(scan.rssi, sc.grid, located.rssi, k)
    errors = {a: np.linalg.norm(e - test_points, axis=1) for a, e in estimates.items()}

    times = timing(sc, len(sc.grid))
    loss = 1 - scan.received.sum() / (len(sc.grid) * len(sc.stations) * cfg.scan_frames)
    return Result(sc, scan, test_points, located, estimates, errors, *times, float(loss))


def timing(sc, points):
    """Scan and localization time when the stations take turns and each obeys its duty cycle.

    A station may transmit at most duty_cycle of the time on average (ERC/REC 70-03), so at one
    grid node it needs at least frames x airtime / duty_cycle; the other stations transmit in
    its pauses. One localization is a short burst that fits into the hourly budget.
    """
    cfg = sc.cfg
    n_st = len(sc.stations)
    dc = cfg.radio.duty_cycle
    sequential = n_st * cfg.scan_frames * sc.slot_ms / 1000
    per_station = cfg.scan_frames * sc.airtime_ms / 1000 / dc
    scan_time = points * max(sequential, per_station)
    scan_radio = points * sequential
    burst = cfg.locate_frames * sc.airtime_ms / 1000            # one station's airtime per localization
    locate_time = max(n_st * cfg.locate_frames * sc.slot_ms / 1000, burst / dc if burst > 3600 * dc else 0)
    per_hour = 3600 * dc / burst
    return scan_time, scan_radio, locate_time, per_hour
