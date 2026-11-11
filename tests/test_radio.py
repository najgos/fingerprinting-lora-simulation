"""Radio link budget: noise floor and receiver sensitivity."""
import pytest

from lorasim import radio


def test_noise_floor_at_125_khz():
    # -174 dBm/Hz + 10*log10(125 000 Hz) + noise figure 6 dB = -117.03 dBm
    assert radio.noise_floor_dbm(125, 6) == pytest.approx(-117.03, abs=0.01)


@pytest.mark.parametrize(("sf", "expected_dbm"), [(7, -124.5), (12, -137.0)])
def test_sensitivity_matches_the_values_in_config(sf, expected_dbm):
    # config.Radio.noise_figure_db states -124.5 dBm at SF7 and -137 dBm at SF12 (125 kHz, NF 6 dB)
    assert radio.sensitivity_dbm(sf, 125, 6) == pytest.approx(expected_dbm, abs=0.05)


@pytest.mark.parametrize(
    ("sf", "expected_ms"),
    [
        (7, 41.216),     # short symbols, no low data rate optimisation
        (9, 144.384),
        (12, 991.232),   # symbol 32.8 ms > 16 ms: low data rate optimisation on
    ],
)
def test_time_on_air_matches_semtech_calculator(sf, expected_ms):
    # 10-byte payload, 125 kHz, coding rate 4/5, 8-symbol preamble, explicit header, CRC on
    assert radio.time_on_air_ms(sf, 125, 10) == pytest.approx(expected_ms, abs=0.001)


@pytest.mark.parametrize(("signal_dbm", "expected_dbm"), [(-80.2, -80.0), (-80.3, -80.5)])
def test_reported_rssi_is_rounded_to_half_db(signal_dbm, expected_dbm):
    # The signal is 37 dB above the noise, so the noise adds almost nothing: only the rounding shows.
    # Multiples of 0.5 are exact in binary floating point, so == is safe here.
    assert radio.reported_rssi(signal_dbm, -117.0) == expected_dbm


def test_reported_rssi_measures_signal_plus_noise():
    # Equal signal and noise power: the total power is doubled, 10*log10(2) = 3 dB more
    assert radio.reported_rssi(-117.0, -117.0) == -114.0
