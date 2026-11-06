"""Log-distance path loss, exponentially correlated shadowing (shared between stations in part)
and Ricean multipath with a fixed and a per-frame part."""
import math

import numpy as np

C = 299_792_458.0
MAX_RASTER = 2048
CELLS_PER_CORRELATION_DISTANCE = 10    # coarser rasters overstate short-range correlation


def free_space_loss_db(distance_m, frequency_mhz):
    """ITU-R P.525: free-space basic transmission loss."""
    wavelength = C / (frequency_mhz * 1e6)
    return 20 * np.log10(4 * math.pi * np.maximum(distance_m, 1e-3) / wavelength)


def reference_loss_db(frequency_mhz):
    """Free-space loss at 1 m."""
    return float(free_space_loss_db(1.0, frequency_mhz))


def path_loss_db(distance_m, env, frequency_mhz):
    """Log-distance model PL(d0) + 10 n log10(d / d0), never below free-space loss."""
    d = np.maximum(distance_m, 1.0)
    pl = env.reference_loss + 10 * env.path_loss_exponent * np.log10(d / env.reference_distance)
    return np.maximum(pl, free_space_loss_db(d, frequency_mhz))


# Ground: soil model of ITU-R P.527-6 § 5.2 (valid at all frequencies up to 1000 GHz).
# Soil texture: silty loam from Table 2; temperature 23 °C as in Figs 14-16 of the Recommendation.
SOIL = {"sand": 30.63, "clay": 13.48, "rho_s": 2.59, "rho_b": 1.5750, "temperature_c": 23.0}
# Volumetric water content (m3/m3). 0.07 and 0.5 are the dry and wet soil examples of P.527-6;
# 0.3 is an assumption between them.
GROUND_TYPES = {
    "Dry soil (7 % water)": 0.07,
    "Moist soil (30 % water)": 0.30,
    "Wet soil (50 % water)": 0.50,
}


def pure_water_debye(f_ghz, temperature_c):
    """ITU-R P.527-6 eqs (6)-(13): real and imaginary part of the permittivity of pure water."""
    theta = 300 / (temperature_c + 273.15) - 1
    eps_s = 77.66 + 103.3 * theta
    eps_1 = 0.0671 * eps_s
    eps_inf = 3.52 - 7.52 * theta
    f1 = 20.20 - 146.4 * theta + 316 * theta ** 2
    f2 = 39.8 * f1
    real = (eps_s - eps_1) / (1 + (f_ghz / f1) ** 2) + (eps_1 - eps_inf) / (1 + (f_ghz / f2) ** 2) + eps_inf
    imag = ((f_ghz / f1) * (eps_s - eps_1) / (1 + (f_ghz / f1) ** 2)
            + (f_ghz / f2) * (eps_1 - eps_inf) / (1 + (f_ghz / f2) ** 2))
    return real, imag


def soil_permittivity(f_ghz, water_content, sand, clay, rho_s, rho_b, temperature_c):
    """ITU-R P.527-6 eqs (58)-(70): complex relative permittivity eps' - j eps'' of soil.

    Mixture of soil particles (sand, clay, silt), air, bound water and free water; `water_content`
    is the volumetric water content mv, sand and clay are percentages by volume.
    """
    alpha = 0.65
    beta_re = 1.2748 - 0.00519 * sand - 0.00152 * clay
    beta_im = 1.33797 - 0.00603 * sand - 0.00166 * clay
    eps_sm = (1.01 + 0.44 * rho_s) ** 2 - 0.062
    sigma_1 = 0.0467 + 0.2204 * rho_b - 0.004111 * sand - 0.006614 * clay
    sigma_2 = -1.645 + 1.939 * rho_b - 0.0225622 * sand + 0.01594 * clay
    x = f_ghz / 1.35
    sigma_re = x * (sigma_1 - sigma_2) / (1 + x ** 2)
    sigma_im = sigma_2 + (sigma_1 - sigma_2) / (1 + x ** 2)
    water_re, water_im = pure_water_debye(f_ghz, temperature_c)
    conduction = 18 / f_ghz * (rho_s - rho_b) / (rho_s * water_content)
    free_re = water_re + sigma_re * conduction
    free_im = water_im + sigma_im * conduction
    real = (1 + rho_b / rho_s * (eps_sm ** alpha - 1)
            + water_content ** beta_re * free_re ** alpha - water_content) ** (1 / alpha)
    imag = (water_content ** beta_im * free_im ** alpha) ** (1 / alpha)
    return complex(real, -imag)


def ground_permittivity(ground, frequency_mhz):
    """Complex relative permittivity of the ground type at the given frequency."""
    return soil_permittivity(frequency_mhz / 1000, GROUND_TYPES[ground], **SOIL)


def ground_reflection(sin_grazing, eps_c):
    """Fresnel reflection coefficient of the ground for vertical polarization (vertical antennas)."""
    root = np.sqrt(eps_c - (1 - sin_grazing ** 2))
    return (eps_c * sin_grazing - root) / (eps_c * sin_grazing + root)


def two_ray_loss_db(distance_m, h_tx, h_rx, frequency_mhz, ground):
    """Direct wave plus the wave reflected by flat ground (both antennas isotropic in elevation).

    Close to the station the loss follows free space with interference ripples; beyond the
    breakpoint 4 h_tx h_rx / wavelength it grows by about 40 dB per decade.
    """
    wavelength = C / (frequency_mhz * 1e6)
    d = np.asarray(distance_m, dtype=float)
    d1 = np.maximum(np.hypot(d, h_tx - h_rx), 1.0)
    d2 = np.maximum(np.hypot(d, h_tx + h_rx), d1)
    gamma = ground_reflection((h_tx + h_rx) / d2, ground_permittivity(ground, frequency_mhz))
    k = 2 * math.pi / wavelength
    field = np.exp(-1j * k * d1) / d1 + gamma * np.exp(-1j * k * d2) / d2
    return -20 * np.log10(wavelength / (4 * math.pi) * np.abs(field))


PATH_LOSS_MODELS = ("Log-distance", "Two-ray")


def link_loss_db(distance_2d, h_tx, h_rx, env, frequency_mhz):
    """Mean path loss of a link: log-distance on the 3-D distance, or the two-ray model."""
    if env.path_loss_model == "Two-ray":
        return two_ray_loss_db(distance_2d, h_tx, h_rx, frequency_mhz, env.ground)
    return path_loss_db(np.hypot(distance_2d, h_tx - h_rx), env, frequency_mhz)


class ShadowField:
    """Gaussian random field with standard deviation sigma and exponential correlation.

    The correlation between two positions is exp(-distance / correlation_distance), as in
    Gudmundson (1991) and 3GPP TR 38.901. White noise on a raster is filtered with the square
    root of the matching 2-D spectrum; values between raster cells are interpolated bilinearly.
    """

    def __init__(self, width, height, sigma, correlation_distance, rng):
        self.sigma = sigma
        if sigma <= 0:
            return
        margin = 4 * correlation_distance            # avoids wrap-around of the FFT filter
        span_x, span_y = width + 2 * margin, height + 2 * margin
        cell = max(correlation_distance / CELLS_PER_CORRELATION_DISTANCE, max(span_x, span_y) / MAX_RASTER)
        nx, ny = int(math.ceil(span_x / cell)) + 1, int(math.ceil(span_y / cell)) + 1
        white = rng.standard_normal((ny, nx))
        fx = np.fft.fftfreq(nx, d=cell)
        fy = np.fft.fftfreq(ny, d=cell)
        f2 = fx[None, :] ** 2 + fy[:, None] ** 2
        # 2-D spectrum of exp(-r/d) is proportional to (1 + (2 pi d f)^2)^(-3/2)
        amplitude = (1 + (2 * math.pi * correlation_distance) ** 2 * f2) ** -0.75
        smooth = np.real(np.fft.ifft2(np.fft.fft2(white) * amplitude))
        smooth /= smooth.std()
        self.values = smooth * sigma
        self.origin = -margin
        self.cell = cell
        self.coarse = cell > correlation_distance / CELLS_PER_CORRELATION_DISTANCE * 1.01

    def __call__(self, x, y):
        x = np.asarray(x, dtype=float)
        if self.sigma <= 0:
            return np.zeros_like(x)
        fx = (x - self.origin) / self.cell
        fy = (np.asarray(y, dtype=float) - self.origin) / self.cell
        ny, nx = self.values.shape
        x0 = np.clip(np.floor(fx).astype(int), 0, nx - 2)
        y0 = np.clip(np.floor(fy).astype(int), 0, ny - 2)
        tx, ty = fx - x0, fy - y0
        v = self.values
        top = v[y0, x0] * (1 - tx) + v[y0, x0 + 1] * tx
        bottom = v[y0 + 1, x0] * (1 - tx) + v[y0 + 1, x0 + 1] * tx
        return top * (1 - ty) + bottom * ty


class CorrelatedShadow:
    """Shadowing of one station: a part shared by all stations plus the station's own part.

    S_s = sqrt(rho) * common + sqrt(1 - rho) * own_s. Both fields have the same sigma and spatial
    correlation, so every station keeps sigma and any two stations are correlated with rho
    (3GPP TR 36.942 uses 0.5 between sites).
    """

    def __init__(self, common, own, rho):
        self.common, self.own, self.rho = common, own, rho
        self.sigma = own.sigma
        self.coarse = getattr(own, "coarse", False)

    def __call__(self, x, y):
        return math.sqrt(self.rho) * self.common(x, y) + math.sqrt(1 - self.rho) * self.own(x, y)


_RICE_OFFSET = {}
_FADING_STATS = {}


def _k_linear(k_factor_db, k_rayleigh):
    return 0.0 if k_factor_db <= k_rayleigh else 10 ** (k_factor_db / 10)


def los_amplitude(k_factor_db, k_rayleigh, k_none):
    """Amplitude of the line-of-sight part when the total mean power is 1 (scattered power 1 - los^2)."""
    if k_factor_db >= k_none:
        return 1.0
    k_lin = _k_linear(k_factor_db, k_rayleigh)
    return math.sqrt(k_lin / (k_lin + 1))


def rice_amplitude(k_factor_db, size, rng, k_rayleigh, k_none):
    """Complex amplitude with mean power 1: line-of-sight part plus Gaussian scattered part."""
    los = los_amplitude(k_factor_db, k_rayleigh, k_none)
    s = math.sqrt((1 - los ** 2) / 2)
    return los + s * (rng.standard_normal(size) + 1j * rng.standard_normal(size))


def _rice_power_db(k_lin, size, rng):
    los = math.sqrt(k_lin / (k_lin + 1))
    s = math.sqrt(1 / (2 * (k_lin + 1)))
    h = los + s * (rng.standard_normal(size) + 1j * rng.standard_normal(size))
    return 10 * np.log10(np.abs(h) ** 2)


def multipath_db(k_factor_db, size, rng, k_rayleigh, k_none):
    """Ricean (or Rayleigh) power gain in dB at fixed positions, zero mean in dB.

    The mean in dB is removed because path-loss models are fitted to measured dB values.
    """
    if k_factor_db >= k_none:
        return np.zeros(size)
    k_lin = _k_linear(k_factor_db, k_rayleigh)
    if k_lin not in _RICE_OFFSET:
        _RICE_OFFSET[k_lin] = float(_rice_power_db(k_lin, 400_000, np.random.default_rng(12345)).mean())
    return _rice_power_db(k_lin, size, rng) - _RICE_OFFSET[k_lin]


def dynamic_gain_db(fixed, dynamic_share, rng, los):
    """Power gain in dB of single frames: the line of sight stays, the scattered part changes in part.

    `fixed` holds the complex amplitudes (from rice_amplitude, mean power 1) at the position of each
    frame and `los` their line-of-sight part. A share `dynamic_share` of the scattered (multipath)
    power is new in every frame (moving people, vehicles, leaves); frames are independent. The line
    of sight is not affected, so the K-factor and the mean power stay the same. The mean in dB is
    not removed here (see fading_stats_db).
    """
    scattered_power = 1 - los ** 2
    new = math.sqrt(dynamic_share * scattered_power / 2) * (
        rng.standard_normal(fixed.shape) + 1j * rng.standard_normal(fixed.shape))
    h = los + math.sqrt(1 - dynamic_share) * (fixed - los) + new
    return 10 * np.log10(np.abs(h) ** 2)


def fading_stats_db(k_factor_db, dynamic_share, k_rayleigh, k_none):
    """Mean and standard deviation in dB of the single-frame multipath gain (fixed and dynamic part)."""
    key = (float(k_factor_db), float(dynamic_share))
    if key not in _FADING_STATS:
        if k_factor_db >= k_none:
            _FADING_STATS[key] = (0.0, 0.0)            # no multipath: nothing to change either
        else:
            rng = np.random.default_rng(12345)
            los = los_amplitude(k_factor_db, k_rayleigh, k_none)
            fixed = rice_amplitude(k_factor_db, 400_000, rng, k_rayleigh, k_none)
            g = dynamic_gain_db(fixed, dynamic_share, rng, los)
            _FADING_STATS[key] = (float(g.mean()), float(g.std()))
    return _FADING_STATS[key]
