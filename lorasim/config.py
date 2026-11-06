"""Simulation settings, propagation parameter sets and regulatory sub-bands.

The comments and the Preset.source texts say where each number comes from or mark it as an assumption.
"""
from dataclasses import dataclass, field, replace

from .propagation import reference_loss_db

MAX_GRID_POINTS = 40_000
NO_SIGNAL_RSSI = -140.0   # dBm written into a fingerprint when no frame from a station arrived

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
    dynamic_share: float = 0.0    # share of the multipath power that changes from frame to frame (0..1)
    station_correlation: float = 0.0     # correlation of the shadowing between any two stations (0..1)
    path_loss_model: str = "Log-distance"   # or "Two-ray" (d0, PL(d0) and n are then not used)
    ground: str = "Moist soil (30 % water)"  # ground type for the two-ray model (ITU-R P.527)


@dataclass(frozen=True)
class Preset:
    env: Environment
    source: str                   # where the numbers come from


ENVIRONMENTS = {
    "Open area, line of sight (two-ray)": Preset(
        Environment(1.0, reference_loss_db(868.0), 2.0, 4.0, 10.0, 9.0, 2.0,
                    station_correlation=0.5, path_loss_model="Two-ray", ground="Moist soil (30 % water)"),
        "Open ground without buildings, small obstacles only (benches). Mean loss: two-ray model, "
        "direct wave plus the wave reflected by flat ground; ground: ITU-R P.527-6 soil model "
        "(silty loam, 30 % water: assumption between its dry and wet examples). Shadowing 4 dB, "
        "correlation distance 10 m and Ricean K 9 dB: 3GPP TR 38.901 UMi LOS (made for base stations "
        "at 10 m, so a starting point to check with measurements). Station correlation 0.5: "
        "3GPP TR 36.942. Frame noise: assumption."),
    "Free space (ITU-R P.525)": Preset(
        Environment(1.0, reference_loss_db(868.0), 2.0, 0.0, 10.0, K_NONE, 0.0),
        "Free-space loss, ITU-R P.525. No shadowing, no multipath, no frame noise."),
}
DEFAULT_ENVIRONMENT = "Open area, line of sight (two-ray)"

@dataclass(frozen=True)
class SubBand:
    duty_cycle: float             # ERC/REC 70-03 Annex 1
    erp_limit_dbm: float          # ERC/REC 70-03 Annex 1
    low_mhz: float                # band edges
    high_mhz: float
    default_channel_mhz: float    # LoRaWAN EU868 channel (LoRaWAN Regional Parameters RP002)

    def fits(self, bandwidth_khz):
        """True if a channel of this bandwidth fits inside the sub-band."""
        return bandwidth_khz / 1000 <= self.high_mhz - self.low_mhz + 1e-9

    def channel_mhz(self, bandwidth_khz):
        """Centre frequency: the default channel, or the band centre if the bandwidth does not fit there."""
        if not self.fits(bandwidth_khz):
            raise ValueError(f"{bandwidth_khz:g} kHz does not fit into {self.low_mhz:g}-{self.high_mhz:g} MHz")
        half = bandwidth_khz / 2000
        f = self.default_channel_mhz
        if self.low_mhz - 1e-9 <= f - half and f + half <= self.high_mhz + 1e-9:
            return f
        return (self.low_mhz + self.high_mhz) / 2


SUB_BANDS = {
    "868.0-868.6 MHz (1 %, 25 mW e.r.p.)": SubBand(0.01, 14.0, 868.0, 868.6, 868.1),
    "869.4-869.65 MHz (10 %, 500 mW e.r.p.)": SubBand(0.10, 27.0, 869.4, 869.65, 869.525),
}
# e.r.p. is referred to a half-wave dipole, e.i.r.p. to an isotropic antenna (ITU Radio Regulations,
# Article 1); the dipole has 2.15 dBi, so e.r.p. = e.i.r.p. - 2.15 dB.
DIPOLE_GAIN_DBI = 2.15

STATION_LAYOUTS = ("Corners", "Side midpoints", "Random", "Manual")
ALGORITHMS = ("WkNN", "NN")


@dataclass(frozen=True)
class Radio:
    sub_band: str = "868.0-868.6 MHz (1 %, 25 mW e.r.p.)"
    bandwidth_khz: float = 125.0
    spreading_factor: int = 7
    tx_power_dbm: float = 13.0           # 13 dBm + 3 dBi - 2.15 dB = 13.85 dBm e.r.p. (limit 14 dBm)
    antenna_gain_dbi: float = 3.0        # project antenna: 868 MHz, 3 dBi (seller's specification)
    noise_figure_db: float = 6.0         # with the SNR limits: -124.5 dBm at SF7 and -137 dBm at SF12
                                         # (125 kHz), in line with the SX1262 datasheet sensitivity
    payload_bytes: int = 10              # coding rate 4/5 and an 8-symbol preamble are fixed (radio.py)
    guard_ms: float = 10.0               # gap between consecutive frames (assumption)

    @property
    def band(self):
        return SUB_BANDS[self.sub_band]

    @property
    def frequency_mhz(self):
        return self.band.channel_mhz(self.bandwidth_khz)

    @property
    def duty_cycle(self):
        return self.band.duty_cycle

    @property
    def erp_limit_dbm(self):
        return self.band.erp_limit_dbm

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
    station_height: float = 2.0          # m above ground, station antennas (assumption)
    receiver_height: float = 1.2         # m above ground, hand-held receiver (assumption)
    radio: Radio = field(default_factory=Radio)
    env: Environment = ENVIRONMENTS[DEFAULT_ENVIRONMENT].env
    scan_frames: int = 20                # frames per station at every grid point
    locate_frames: int = 20              # frames per station for one localization
    scan_positions: int = 1              # receiver positions around a grid node (more than half a
    locate_positions: int = 1            # wavelength apart); frames are spread over them in turn
    test_points: int = 500
    algorithm: str = "WkNN"
    k: int = 4
    seed: int = 1

    def with_(self, **changes):
        return replace(self, **changes)
