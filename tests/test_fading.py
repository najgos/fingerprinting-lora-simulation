"""Multipath split into a fixed and a per-frame part, spatial averaging, total sigma, station correlation."""
import math

import numpy as np
import pytest

from lorasim import ENVIRONMENTS, Config, Environment
from lorasim.config import K_NONE, K_RAYLEIGH
from lorasim.propagation import CorrelatedShadow, ShadowField, link_loss_db, reference_loss_db
from lorasim.simulation import build_scenario, mean_power, measure, shadowing_sigma_used

RAYLEIGH_DB = 10 / math.log(10) * math.pi / math.sqrt(6)      # 5.57 dB
FS = reference_loss_db(868.0)


def residual(env, frames, positions, seed=0, width=100.0):
    """Measured mean RSSI minus the mean power (path loss and shadowing) at every grid node."""
    sc = build_scenario(Config(env=env, width=width, height=width))
    m = measure(sc, sc.grid, frames, np.random.default_rng(seed), positions)
    return m.rssi - mean_power(sc, sc.grid)


def multipath_env(share):
    return Environment(1.0, FS, 2.0, 0.0, 13.0, K_RAYLEIGH, 0.0, dynamic_share=share)


@pytest.mark.parametrize("positions", [1, 4, 16])
def test_spatial_averaging_reduces_fixed_multipath(positions):
    r = residual(multipath_env(0.0), 16, positions)
    assert r.std() == pytest.approx(RAYLEIGH_DB / math.sqrt(positions), rel=0.1)
    assert abs(r.mean()) < 0.3


def test_frames_reduce_only_the_part_that_changes():
    static_1, static_50 = residual(multipath_env(0.0), 1, 1).std(), residual(multipath_env(0.0), 50, 1).std()
    assert static_50 == pytest.approx(static_1, rel=0.1)          # fixed multipath: frames do not help
    assert residual(multipath_env(1.0), 50, 1).std() < 1.0       # new fading every frame: averages out
    half = residual(multipath_env(0.5), 50, 1).std()
    assert 1.0 < half < static_1


def test_dynamic_multipath_has_zero_mean_and_rayleigh_spread_per_frame():
    r = residual(multipath_env(0.5), 1, 1)
    assert abs(r.mean()) < 0.3
    assert r.std() == pytest.approx(RAYLEIGH_DB, rel=0.1)


def test_total_sigma_leaves_the_rest_for_shadowing():
    bor = ENVIRONMENTS["Built-up, indoor nodes (Bor 2016)"].env
    assert shadowing_sigma_used(bor) == (pytest.approx(math.sqrt(3.57 ** 2 - 2.0 ** 2)), True)
    bonn = ENVIRONMENTS["Urban, rooftop gateways (Rademacher 2021)"].env
    sigma, ok = shadowing_sigma_used(bonn)
    assert ok and sigma == pytest.approx(math.sqrt(9.9 ** 2 - RAYLEIGH_DB ** 2 - 2.0 ** 2), rel=0.01)
    rayleigh_in_small_sigma = Environment(40.0, 127.41, 2.08, 3.57, 13.0, K_RAYLEIGH, 2.0,
                                          sigma_includes_fading=True)
    assert shadowing_sigma_used(rayleigh_in_small_sigma) == (0.0, False)
    shadow_only = Environment(40.0, 127.41, 2.08, 3.57, 13.0, K_RAYLEIGH, 2.0)
    assert shadowing_sigma_used(shadow_only) == (3.57, True)


def test_single_frames_spread_with_the_total_sigma():
    env = Environment(1.0, FS, 2.0, 8.0, 13.0, K_RAYLEIGH, 2.0, sigma_includes_fading=True)
    cfg = Config(env=env, width=300, height=300)
    sc = build_scenario(cfg)
    m = measure(sc, sc.grid, 1, np.random.default_rng(1))
    r = cfg.radio
    d = np.hypot(sc.grid[:, None, 0] - sc.stations[None, :, 0], sc.grid[:, None, 1] - sc.stations[None, :, 1])
    curve = r.tx_power_dbm + 2 * r.antenna_gain_dbi - link_loss_db(
        d, cfg.station_height, cfg.receiver_height, env, r.frequency_mhz)
    assert (m.rssi - curve).std() == pytest.approx(8.0, rel=0.1)


def test_station_correlation_keeps_sigma_and_correlates_stations():
    env = Environment(40.0, 127.41, 2.08, 4.0, 40.0, K_NONE, 0.0, station_correlation=0.5)
    cfg = Config(env=env, width=2000, height=2000, grid_spacing=100)
    sc = build_scenario(cfg)
    rng = np.random.default_rng(0)
    x, y = rng.uniform(200, 1800, 20000), rng.uniform(200, 1800, 20000)
    s = np.array([f(x, y) for f in sc.shadow])
    assert np.all(np.abs(s.std(axis=1) - 4.0) < 0.6)
    c = np.corrcoef(s)[np.triu_indices(4, 1)]
    assert np.all(np.abs(c - 0.5) < 0.15)
    assert abs(c.mean() - 0.5) < 0.08


def test_zero_station_correlation_keeps_the_terrain():
    base = Environment(40.0, 127.41, 2.08, 4.0, 40.0, K_NONE, 0.0)
    corr = Environment(40.0, 127.41, 2.08, 4.0, 40.0, K_NONE, 0.0, station_correlation=0.5)
    a = build_scenario(Config(env=base, seed=3))
    b = build_scenario(Config(env=corr, seed=3))
    assert all(isinstance(f, ShadowField) for f in a.shadow)
    assert all(isinstance(f, CorrelatedShadow) for f in b.shadow)
    x, y = np.array([10.0, 50.0, 90.0]), np.array([20.0, 60.0, 80.0])
    for fa, fb in zip(a.shadow, b.shadow):
        assert np.array_equal(fa(x, y), fb.own(x, y))
