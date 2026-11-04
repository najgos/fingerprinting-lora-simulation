"""Log-distance path loss, exponentially correlated shadowing and Ricean multipath."""
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


_RICE_OFFSET = {}


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
    k_lin = 0.0 if k_factor_db <= k_rayleigh else 10 ** (k_factor_db / 10)
    if k_lin not in _RICE_OFFSET:
        _RICE_OFFSET[k_lin] = float(_rice_power_db(k_lin, 400_000, np.random.default_rng(12345)).mean())
    return _rice_power_db(k_lin, size, rng) - _RICE_OFFSET[k_lin]
