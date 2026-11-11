"""Grid, localization, reproducibility and timing."""
import numpy as np
import pytest

from lorasim import Config, GridTooLarge, simulate
from lorasim.simulation import build_scenario, grid_axis, nearest_neighbours

# A tiny hand-made radio map: 3 grid nodes, fingerprints from 2 stations (dBm)
MAP_POINTS = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
MAP_RSSI = np.array([[-50.0, -60.0], [-55.0, -65.0], [-70.0, -40.0]])


@pytest.fixture(scope="module")
def default_result():
    """One run with the default settings, shared by the tests in this file (it takes a moment)."""
    return simulate(Config())


def test_too_dense_grid_is_rejected():
    # 10 km x 10 km with 5 m spacing gives 2001 x 2001 = about 4 million nodes; the limit is 40 000
    cfg = Config(width=10_000, height=10_000, grid_spacing=5)
    with pytest.raises(GridTooLarge):
        build_scenario(cfg)


@pytest.mark.parametrize(("length", "spacing"), [(100, 5), (100, 15), (250, 7), (100, 1000)])
def test_grid_reaches_both_edges_without_exceeding_the_spacing(length, spacing):
    nodes = grid_axis(length, spacing)
    assert nodes[0] == 0.0
    assert nodes[-1] == pytest.approx(length)
    assert np.diff(nodes).max() <= spacing + 1e-9       # 1e-9: margin for floating-point rounding


def test_nn_returns_the_node_with_an_identical_fingerprint():
    query = MAP_RSSI[[1]]                               # exactly the fingerprint of node (10, 0)
    estimate = nearest_neighbours(MAP_RSSI, MAP_POINTS, query, k=1)
    np.testing.assert_allclose(estimate, [[10.0, 0.0]])


def test_wknn_lands_halfway_between_two_equally_similar_nodes():
    query = (MAP_RSSI[[0]] + MAP_RSSI[[1]]) / 2         # as similar to node (0, 0) as to node (10, 0)
    estimate = nearest_neighbours(MAP_RSSI, MAP_POINTS, query, k=2)
    np.testing.assert_allclose(estimate, [[5.0, 0.0]])


def test_same_seed_gives_same_result_and_new_seed_changes_it(default_result):
    again = simulate(Config(seed=1))                    # Config() uses seed 1 as well
    other = simulate(Config(seed=2))
    np.testing.assert_array_equal(default_result.errors["WkNN"], again.errors["WkNN"])
    assert not np.array_equal(default_result.errors["WkNN"], other.errors["WkNN"])


def test_scan_time_is_limited_by_the_duty_cycle(default_result):
    # Per node one station needs 20 frames x 41.216 ms / 1 % duty cycle = 82.4 s, more than the
    # 4 stations x 20 frames x 51.216 ms = 4.1 s of taking turns. 441 nodes x 82.4 s = 36 352.5 s.
    assert len(default_result.scenario.grid) == 441
    assert default_result.scan_time_s == pytest.approx(36_352.5, abs=0.1)
