import numpy as np
import pytest

from lorasim import ENVIRONMENTS, Config, Environment, GridTooLarge, simulate
from lorasim.config import ALGORITHMS, K_NONE, NO_SIGNAL_RSSI, SUB_BANDS, Radio
from lorasim.sweep import PARAMETERS, run_sweep, variants

FREE_SPACE = ENVIRONMENTS["Free space (ITU-R P.525)"].env
BONN = ENVIRONMENTS["Urban, rooftop gateways (Rademacher 2021)"].env
NOISE_FREE = Environment(40.0, 127.41, 2.08, 0.0, 13.0, K_NONE, 0.0)
BAND_1, BAND_10 = SUB_BANDS


def test_rssi_decreases_with_distance_from_station():
    res = simulate(Config(env=FREE_SPACE, station_layout="Corners"))
    grid, rssi = res.scenario.grid, res.scan.rssi[:, 0]          # station 0 sits at (0, 0)
    order = np.argsort(np.hypot(grid[:, 0], grid[:, 1]))
    smoothed = np.convolve(rssi[order], np.ones(15) / 15, mode="valid")
    assert np.all(np.diff(smoothed) <= 0.05)


@pytest.mark.parametrize("algorithm", ["NN", "WkNN"])
def test_noise_free_fingerprint_hits_grid_node(algorithm):
    cfg = Config(env=NOISE_FREE, algorithm=algorithm)
    nodes = np.array([[0, 0], [25, 40], [55, 95], [100, 100]], dtype=float)
    res = simulate(cfg, test_points=nodes)
    assert res.errors[algorithm].max() < 1e-3


def test_same_seed_gives_same_result_and_new_seed_changes_it():
    a = simulate(Config(seed=7))
    b = simulate(Config(seed=7))
    c = simulate(Config(seed=8))
    assert np.array_equal(a.estimates["WkNN"], b.estimates["WkNN"])
    assert not np.array_equal(a.estimates["WkNN"], c.estimates["WkNN"])


@pytest.mark.parametrize("length, spacing", [(100, 5), (100, 15), (100, 30), (250, 7), (100, 1000)])
def test_grid_covers_the_whole_area(length, spacing):
    from lorasim.simulation import grid_axis
    axis = grid_axis(length, spacing)
    assert axis[0] == 0 and axis[-1] == pytest.approx(length)
    assert np.all(np.diff(axis) <= spacing + 1e-9)
    assert np.diff(axis).max() > spacing * 0.5 or len(axis) == 2


def test_grid_limit():
    with pytest.raises(GridTooLarge):
        simulate(Config(width=10_000, height=10_000, grid_spacing=10))


def test_out_of_range_station_is_marked_as_no_signal():
    cfg = Config(width=10_000, height=10_000, grid_spacing=500, env=BONN)
    res = simulate(cfg)
    lost = res.scan.received == 0
    assert lost.any()
    assert np.all(res.scan.rssi[lost] == NO_SIGNAL_RSSI)
    assert res.frame_loss > 0


def test_scan_time_obeys_duty_cycle():
    for band, dc in ((BAND_1, 0.01), (BAND_10, 0.10)):
        cfg = Config(scan_frames=100, locate_frames=10, env=FREE_SPACE, radio=Radio(guard_ms=0, sub_band=band))
        res = simulate(cfg)
        airtime = res.scenario.airtime_ms / 1000
        assert res.scan_radio_time_s == pytest.approx(441 * 4 * 100 * airtime)
        # each station may be on air only dc of the time, which dominates 4 stations taking turns
        assert res.scan_time_s == pytest.approx(441 * 100 * airtime / dc)
        assert res.locate_time_s == pytest.approx(4 * 10 * airtime)
        assert res.locates_per_hour == pytest.approx(3600 * dc / (10 * airtime))


def test_all_sweeps_run():
    cfg = Config(test_points=50)
    for p in PARAMETERS:
        assert len(variants(cfg, p)) >= 3
    xs, mean, p90 = run_sweep(cfg, "k")
    assert len(xs) == 10 and set(mean) == set(ALGORITHMS)
    assert np.all(p90["WkNN"] >= mean["WkNN"] * 0.5)
