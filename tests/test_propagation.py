"""Path loss models and shadowing."""
import math
from dataclasses import replace

import numpy as np
import pytest

from lorasim import ENVIRONMENTS, propagation

GROUND = "Moist soil (30 % water)"
H_STATION, H_RECEIVER = 2.0, 1.2        # m, the defaults in Config


def test_free_space_loss_at_1_km_and_868_mhz():
    # Textbook form: 32.45 + 20*log10(d in km) + 20*log10(f in MHz) = 32.45 + 0 + 58.77 = 91.2 dB
    assert propagation.free_space_loss_db(1000, 868) == pytest.approx(91.2, abs=0.05)


def test_free_space_loss_grows_6_db_when_distance_doubles():
    near = propagation.free_space_loss_db(100, 868)
    far = propagation.free_space_loss_db(200, 868)
    assert far - near == pytest.approx(20 * math.log10(2))     # 6.02 dB


def test_log_distance_loss_never_drops_below_free_space():
    # An exponent below 2 would give less loss than free space; the model must clamp it.
    env = replace(ENVIRONMENTS["Free space (ITU-R P.525)"].env, path_loss_exponent=1.5)
    distances = np.array([1.0, 10.0, 100.0, 1000.0])
    loss = propagation.path_loss_db(distances, env, 868)
    assert np.all(loss >= propagation.free_space_loss_db(distances, 868))


def test_two_ray_loss_grows_40_db_per_decade_far_beyond_the_breakpoint():
    # Breakpoint 4 * h_station * h_receiver / wavelength = 28 m; 1 km and 10 km are far beyond it
    near, far = propagation.two_ray_loss_db(np.array([1000.0, 10_000.0]), H_STATION, H_RECEIVER, 868, GROUND)
    assert far - near == pytest.approx(40, abs=0.5)


@pytest.mark.parametrize("distance", [5.0, 10.0, 20.0])
def test_two_ray_loss_stays_near_free_space_before_the_breakpoint(distance):
    two_ray = propagation.two_ray_loss_db(distance, H_STATION, H_RECEIVER, 868, GROUND)
    # Free space on the 3-D distance between the antennas, which are at different heights
    free_space = propagation.free_space_loss_db(math.hypot(distance, H_STATION - H_RECEIVER), 868)
    # The reflected wave only adds interference ripples of a few dB
    assert abs(two_ray - free_space) < 4


def test_shadow_field_has_the_requested_sigma():
    rng = np.random.default_rng(0)          # fixed seed: the test gives the same result every run
    field = propagation.ShadowField(1000, 1000, sigma=4.0, correlation_distance=10.0, rng=rng)
    x, y = rng.uniform(0, 1000, size=(2, 20_000))
    assert field(x, y).std() == pytest.approx(4.0, rel=0.1)


def test_shadow_field_with_zero_sigma_is_flat():
    field = propagation.ShadowField(100, 100, sigma=0.0, correlation_distance=10.0, rng=np.random.default_rng(0))
    points = np.array([1.0, 50.0, 99.0])
    np.testing.assert_array_equal(field(points, points), [0.0, 0.0, 0.0])
