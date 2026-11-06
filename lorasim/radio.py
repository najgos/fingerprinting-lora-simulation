"""LoRa link parameters: time on air, noise floor, sensitivity and SX1262 RSSI reporting."""
import math

import numpy as np

# Demodulation SNR limit per spreading factor (Semtech AN1200.13, SX127x; not confirmed for SX1262)
SNR_LIMIT_DB = {7: -7.5, 8: -10.0, 9: -12.5, 10: -15.0, 11: -17.5, 12: -20.0}
CODING_RATE = 1            # fixed: 4/5
PREAMBLE_SYMBOLS = 8       # fixed


def time_on_air_ms(sf, bw_khz, payload_bytes, coding_rate=CODING_RATE, preamble=PREAMBLE_SYMBOLS,
                   explicit_header=True, crc=True):
    """Frame duration from Semtech AN1200.13."""
    t_sym = (2 ** sf) / (bw_khz * 1e3) * 1e3            # ms
    low_dr_opt = 1 if t_sym > 16.0 else 0
    ih = 0 if explicit_header else 1
    num = 8 * payload_bytes - 4 * sf + 28 + 16 * int(crc) - 20 * ih
    den = 4 * (sf - 2 * low_dr_opt)
    payload_symbols = 8 + max(math.ceil(num / den) * (coding_rate + 4), 0)
    return (preamble + 4.25) * t_sym + payload_symbols * t_sym


def noise_floor_dbm(bw_khz, noise_figure_db):
    return -174.0 + 10 * math.log10(bw_khz * 1e3) + noise_figure_db


def sensitivity_dbm(sf, bw_khz, noise_figure_db):
    return noise_floor_dbm(bw_khz, noise_figure_db) + SNR_LIMIT_DB[sf]


def reported_rssi(signal_dbm, noise_dbm):
    """SX1262 packet RSSI measures signal plus noise, 0.5 dB steps."""
    total = 10 * np.log10(10 ** (signal_dbm / 10) + 10 ** (noise_dbm / 10))
    return np.round(total * 2) / 2
