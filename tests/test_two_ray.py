"""Two-ray (ground reflection) model, ground types and antenna heights."""
import math

import numpy as np
import pytest

from lorasim import ENVIRONMENTS, Config, simulate
from lorasim.config import Environment, K_NONE
from lorasim.propagation import (free_space_loss_db, ground_permittivity, ground_reflection, link_loss_db,
                                 path_loss_db, two_ray_loss_db)

OPEN = ENVIRONMENTS["Open area, line of sight (two-ray)"].env
WAVELENGTH = 299_792_458.0 / 868e6


def test_ground_from_itu_r_p2040_table_3():
    eps = ground_permittivity("Medium dry ground", 868.0)
    assert eps.real == pytest.approx(15 * 0.868 ** -0.1)
    sigma = -eps.imag * 0.868 / 17.98                      # back to S/m
    assert sigma == pytest.approx(0.035 * 0.868 ** 1.63)
    assert ground_permittivity("Wet ground", 868.0).real > ground_permittivity("Very dry ground", 868.0).real


def test_reflection_tends_to_minus_one_at_grazing_incidence():
    eps = ground_permittivity("Medium dry ground", 868.0)
    assert abs(ground_reflection(np.array(1e-4), eps) + 1) < 0.01
    angles = np.sin(np.radians(np.linspace(0.01, 90, 500)))
    assert np.all(np.abs(ground_reflection(angles, eps)) <= 1.0)


def test_far_from_the_breakpoint_loss_grows_40_db_per_decade():
    ht, hr = 2.0, 1.2
    d = np.array([3000.0, 30000.0])
    pl = two_ray_loss_db(d, ht, hr, 868.0, "Medium dry ground")
    assert pl[1] - pl[0] == pytest.approx(40.0, abs=1.0)
    plane_earth = 40 * np.log10(d) - 20 * np.log10(ht * hr)
    assert np.all(np.abs(pl - plane_earth) < 1.5)


def test_before_the_breakpoint_loss_stays_near_free_space():
    ht, hr = 2.0, 1.2
    breakpoint = 4 * ht * hr / WAVELENGTH
    d = np.linspace(5.0, breakpoint / 2, 200)
    pl = two_ray_loss_db(d, ht, hr, 868.0, "Medium dry ground")
    fspl = free_space_loss_db(np.hypot(d, ht - hr), 868.0)
    assert np.all(pl >= fspl - 20 * math.log10(2) - 1e-9)   # two waves add at most 6 dB
    assert np.all(pl <= fspl + 10)


def test_log_distance_uses_3d_distance_and_equals_2d_for_equal_heights():
    env = Environment(40.0, 127.41, 2.08, 0.0, 13.0, K_NONE, 0.0)
    d = np.array([0.0, 3.0, 50.0])
    assert np.array_equal(link_loss_db(d, 1.5, 1.5, env, 868.0), path_loss_db(d, env, 868.0))
    assert link_loss_db(np.array(3.0), 7.0, 1.0, env, 868.0) == pytest.approx(
        path_loss_db(np.hypot(3.0, 6.0), env, 868.0))


def test_receiver_under_the_station_is_finite():
    assert np.isfinite(two_ray_loss_db(np.array([0.0]), 2.0, 1.2, 868.0, "Medium dry ground")).all()


def test_open_area_preset_reaches_every_node_and_rssi_falls_with_distance():
    noiseless = Environment(**{**OPEN.__dict__, "shadowing_sigma": 0.0, "k_factor_db": K_NONE,
                               "frame_noise_sigma": 0.0})
    res = simulate(Config(env=noiseless))
    assert res.frame_loss == 0
    grid, rssi = res.scenario.grid, res.scan.rssi[:, 0]          # station 0 at (0, 0)
    far = np.hypot(grid[:, 0], grid[:, 1]) > 10                   # interference ripples only near the station
    order = np.argsort(np.hypot(grid[far, 0], grid[far, 1]))
    assert np.all(np.diff(np.convolve(rssi[far][order], np.ones(9) / 9, mode="valid")) <= 0.05)


def test_default_is_the_open_area_preset():
    assert Config().env == OPEN
    res = simulate(Config(test_points=100))
    assert res.errors["WkNN"].mean() < 30
