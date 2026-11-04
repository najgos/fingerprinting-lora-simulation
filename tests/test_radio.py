import math

import numpy as np
import pytest

from lorasim import propagation, radio
from lorasim.config import ENVIRONMENTS, K_NONE, K_RAYLEIGH, Radio

BOR = ENVIRONMENTS["Built-up, indoor nodes (Bor 2016)"].env
BONN = ENVIRONMENTS["Urban, rooftop gateways (Rademacher 2021)"].env


@pytest.mark.parametrize("sf, expected_ms", [(7, 41.216), (9, 144.384), (12, 991.232)])
def test_time_on_air_matches_semtech_formula(sf, expected_ms):
    # 125 kHz, 10-byte payload, CR 4/5, 8-symbol preamble, explicit header, CRC on
    assert radio.time_on_air_ms(sf, 125, 10) == pytest.approx(expected_ms, abs=1e-3)


def test_sensitivity_follows_spreading_factor():
    assert radio.sensitivity_dbm(7, 125, 6) == pytest.approx(-124.5, abs=0.1)
    assert radio.sensitivity_dbm(12, 125, 6) == pytest.approx(-137.0, abs=0.1)


def test_reported_values_are_quantized_and_snr_saturates():
    assert radio.reported_rssi(np.array(-80.3), -117.0) == -80.5
    assert radio.reported_snr(np.array(30.0)) == radio.SNR_REPORT_MAX_DB
    assert radio.reported_snr(np.array(-3.1)) == -3.0


def test_measured_models_reproduce_published_values():
    assert propagation.path_loss_db(40.0, BOR, 868) == pytest.approx(127.41)
    assert propagation.path_loss_db(400.0, BOR, 868) == pytest.approx(127.41 + 20.8)
    assert propagation.path_loss_db(1000.0, BONN, 868) == pytest.approx(132.41)
    assert propagation.path_loss_db(10000.0, BONN, 868) == pytest.approx(132.41 + 15.8)


def test_path_loss_never_below_free_space():
    d = np.array([1, 10, 100, 1000], dtype=float)
    fspl = propagation.free_space_loss_db(d, 868)
    assert fspl[0] == pytest.approx(31.2, abs=0.05)
    assert np.all(propagation.path_loss_db(d, BONN, 868) >= fspl - 1e-9)


def test_shadowing_has_requested_sigma_and_exponential_correlation():
    rng = np.random.default_rng(0)
    field = propagation.ShadowField(2000, 2000, sigma=4.0, correlation_distance=40.0, rng=rng)
    x = rng.uniform(200, 1800, 20000)
    y = rng.uniform(200, 1800, 20000)
    a = field(x, y)
    assert a.std() == pytest.approx(4.0, rel=0.1)
    assert np.corrcoef(a, field(x + 40.0, y))[0, 1] == pytest.approx(math.exp(-1), abs=0.08)
    # exponential, not Gaussian: at half the distance exp(-0.5) = 0.61 (a Gaussian would give 0.78)
    assert np.corrcoef(a, field(x + 20.0, y))[0, 1] == pytest.approx(math.exp(-0.5), abs=0.08)
    assert abs(np.corrcoef(a, field(x + 400.0, y))[0, 1]) < 0.1


def test_multipath_spread_matches_rice_theory():
    rng = np.random.default_rng(1)
    rayleigh = propagation.multipath_db(K_RAYLEIGH, 200_000, rng, K_RAYLEIGH, K_NONE)
    # standard deviation of 10 log10 of an exponential variable: 10/ln10 * pi/sqrt(6)
    assert rayleigh.std() == pytest.approx(10 / math.log(10) * math.pi / math.sqrt(6), abs=0.05)
    assert abs(rayleigh.mean()) < 0.05
    strong_los = propagation.multipath_db(20.0, 200_000, rng, K_RAYLEIGH, K_NONE)
    assert strong_los.std() < 0.8
    assert np.all(propagation.multipath_db(K_NONE, 10, rng, K_RAYLEIGH, K_NONE) == 0)


def test_erp_of_project_antenna_is_within_the_limit():
    r = Radio()
    assert r.erp_dbm == pytest.approx(13.0 + 3.0 - 2.15)
    assert r.erp_dbm <= r.erp_limit_dbm
    assert Radio(tx_power_dbm=14.0).erp_dbm > r.erp_limit_dbm
