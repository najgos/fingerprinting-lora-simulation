"""Simulation settings, propagation parameter sets and regulatory sub-bands.

The comments and the Preset.source texts say where each number comes from or mark it as an assumption.
"""
from dataclasses import dataclass, field, replace

from .propagation import reference_loss_db

MAX_GRID_POINTS = 40_000
NO_SIGNAL_RSSI = -140.0   # dBm written into a fingerprint when no frame from a station arrived
NO_SIGNAL_SNR = -25.0     # dB, same idea for SNR

K_RAYLEIGH = -31.0        # K-factor at or below this value: no line of sight (Rayleigh fading)
K_NONE = 40.0             # K-factor at or above this value: no multipath fading


@dataclass(frozen=True)
class Environment:
    reference_distance: float     # m, d0 of the log-distance model
    reference_loss: float         # dB, mean path loss at d0
    path_loss_exponent: float     # n
    shadowing_sigma: float        # dB, log-normal shadowing
    correlation_distance: float   # m, shadowing correlation falls to 1/e (exponential model)
    k_factor_db: float            # Ricean K of the multipath at a fixed position
    frame_noise_sigma: float      # dB, frame-to-frame variation at one position (assumption)


@dataclass(frozen=True)
class Preset:
    env: Environment
    source: str                   # where the numbers come from
    valid_range_m: tuple          # link distances the source covers (min, max), None if unrestricted


ENVIRONMENTS = {
    "Free space (ITU-R P.525)": Preset(
        Environment(1.0, reference_loss_db(868.0), 2.0, 0.0, 13.0, K_NONE, 0.0),
        "Free-space loss, ITU-R P.525. No shadowing, no multipath, no frame noise.",
        None),
    "Built-up, indoor nodes (Bor 2016)": Preset(
        Environment(40.0, 127.41, 2.08, 3.57, 13.0, K_RAYLEIGH, 2.0),
        "Path loss and shadowing: Bor et al., MSWiM 2016 (Lancaster, built-up, indoor nodes, "
        "range about 100 m). Correlation distance: 3GPP TR 38.901 UMi NLOS. Rayleigh fading: "
        "3GPP NLOS. Frame noise: assumption.",
        (40.0, 100.0)),
    "Urban, rooftop gateways (Rademacher 2021)": Preset(
        Environment(1000.0, 132.41, 1.58, 9.9, 50.0, K_RAYLEIGH, 2.0),
        "Path loss and shadowing: Rademacher et al., VTC2021-Fall (Bonn, gateways on roofs and "
        "poles, links up to 13 km, mean 3 km). Correlation distance: 3GPP TR 38.901 UMa NLOS. "
        "Rayleigh fading: 3GPP NLOS. Frame noise: assumption.",
        (1000.0, 13000.0)),
}
DEFAULT_ENVIRONMENT = "Built-up, indoor nodes (Bor 2016)"

# ERC/REC 70-03 Annex 1: duty cycle and maximum e.r.p.
SUB_BANDS = {
    "868.0-868.6 MHz (1 %, 25 mW e.r.p.)": (0.01, 14.0),
    "869.4-869.65 MHz (10 %, 500 mW e.r.p.)": (0.10, 27.0),
}
DIPOLE_GAIN_DBI = 2.15    # e.r.p. = e.i.r.p. - 2.15 dB (3GPP TS 38.124, 8.2.3)

STATION_LAYOUTS = ("Corners", "Side midpoints", "Random", "Manual")
ALGORITHMS = ("WkNN", "NN")
FEATURES = ("RSSI", "RSSI + SNR")


@dataclass(frozen=True)
class Radio:
    frequency_mhz: float = 868.0
    sub_band: str = "868.0-868.6 MHz (1 %, 25 mW e.r.p.)"
    bandwidth_khz: float = 125.0
    spreading_factor: int = 7
    tx_power_dbm: float = 13.0           # 13 dBm + 3 dBi - 2.15 dB = 13.85 dBm e.r.p. (limit 14 dBm)
    antenna_gain_dbi: float = 3.0        # project antenna: 868 MHz, 3 dBi (seller's specification)
    noise_figure_db: float = 6.0         # assumption
    payload_bytes: int = 10              # coding rate 4/5 and an 8-symbol preamble are fixed (radio.py)
    guard_ms: float = 10.0               # gap between consecutive frames (assumption)
    snr_report_max_db: float = 12.0      # saturation of the reported packet SNR (assumption)

    @property
    def duty_cycle(self):
        return SUB_BANDS[self.sub_band][0]

    @property
    def erp_limit_dbm(self):
        return SUB_BANDS[self.sub_band][1]

    @property
    def erp_dbm(self):
        return self.tx_power_dbm + self.antenna_gain_dbi - DIPOLE_GAIN_DBI


@dataclass(frozen=True)
class Config:
    width: float = 100.0                 # m
    height: float = 100.0                # m
    grid_spacing: float = 5.0            # m
    station_layout: str = "Corners"
    manual_stations: tuple = ((0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0))
    radio: Radio = field(default_factory=Radio)
    env: Environment = ENVIRONMENTS[DEFAULT_ENVIRONMENT].env
    scan_frames: int = 200               # frames per station at every grid point
    locate_frames: int = 20              # frames per station for one localization
    test_points: int = 500
    algorithm: str = "WkNN"
    k: int = 4
    features: str = "RSSI"
    seed: int = 1

    def with_(self, **changes):
        return replace(self, **changes)
