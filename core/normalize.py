"""Invertible flux normalizations for mapping raw FITS values onto a training scale.

Each normalization is a monotonic map f: raw flux -> normalized value with an EXACT
inverse, so a network prediction in normalized space can be carried back to physical
flux. Stats (min/max, background, noise) are fit on TRAIN pixels only, then the same
map is applied to every patch.

Nothing is clipped: forward/inverse are exact inverses across the whole domain, so a
value outside the fitted range maps just outside [0, 1] rather than saturating. Clipping
to [0, 1] would send every over-range value to the same output and destroy invertibility,
which is the one property this module guarantees.

To add a normalization: subclass Normalization, implement fit/forward/inverse/to_dict,
and register the class. The round-trip test (inverse(forward(x)) == x) is the contract —
see test/test_normalize.py.
"""

import numpy as np
from astropy.stats import sigma_clipped_stats


class Normalization:
    method = None

    def forward(self, x):
        raise NotImplementedError

    def inverse(self, y):
        raise NotImplementedError

    def to_dict(self):
        raise NotImplementedError


class LinearNorm(Normalization):
    method = "linear"

    def __init__(self, lo, hi):
        self.lo = lo
        self.hi = hi

    @classmethod
    def fit(cls, pixels, **_):
        return cls(lo=float(pixels.min()), hi=float(pixels.max()))

    # f(x) = (x - lo) / (hi - lo)
    def forward(self, x):
        return (x - self.lo) / (self.hi - self.lo)

    def inverse(self, y):
        return y * (self.hi - self.lo) + self.lo

    def to_dict(self):
        return {"method": self.method, "lo": self.lo, "hi": self.hi}


class LogNorm(Normalization):
    """log10 stretch. Invertible on its natural domain x > shift - 1; values below that
    (darker than the fitted minimum) are out of domain and yield NaN rather than clip."""

    method = "log"

    def __init__(self, shift, hi_log):
        self.shift = shift
        self.hi_log = hi_log

    @classmethod
    def fit(cls, pixels, **_):
        lo = float(pixels.min())
        hi = float(pixels.max())
        return cls(shift=lo, hi_log=float(np.log10(hi - lo + 1.0)))

    # f(x) = log10(x - shift + 1) / hi_log ; the +1 puts the background floor at 0 and avoids log(0)
    def forward(self, x):
        return np.log10(x - self.shift + 1.0) / self.hi_log

    def inverse(self, y):
        return 10.0 ** (y * self.hi_log) + self.shift - 1.0

    def to_dict(self):
        return {"method": self.method, "shift": self.shift, "hi_log": self.hi_log}


class AsinhNorm(Normalization):
    """asinh stretch: ~linear within +-beta of the background, logarithmic above it, so
    faint structure stays near-linear while bright sources compress."""

    method = "asinh"

    def __init__(self, median, beta, lo_s, hi_s):
        self.median = median
        self.beta = beta
        self.lo_s = lo_s
        self.hi_s = hi_s

    @classmethod
    def fit(cls, pixels, asinh_softening=3.0, **_):
        # beta is a few times the sigma-clipped noise std; clipping ignores the sources so
        # beta tracks the background noise. On a near-noiseless field std -> 0, hence the
        # `or 1.0` floor (a sparse ideal field is better served by a linear normalization).
        _, median, std = sigma_clipped_stats(pixels, sigma=3.0)
        median = float(median)
        beta = asinh_softening * float(std) or 1.0
        lo = float(pixels.min())
        hi = float(pixels.max())
        return cls(
            median=median,
            beta=beta,
            lo_s=float(np.arcsinh((lo - median) / beta)),
            hi_s=float(np.arcsinh((hi - median) / beta)),
        )

    # s(x) = arcsinh((x - median) / beta) ; f(x) = (s(x) - lo_s) / (hi_s - lo_s)
    def forward(self, x):
        stretched = np.arcsinh((x - self.median) / self.beta)
        return (stretched - self.lo_s) / (self.hi_s - self.lo_s)

    def inverse(self, y):
        stretched = y * (self.hi_s - self.lo_s) + self.lo_s
        return self.median + self.beta * np.sinh(stretched)

    def to_dict(self):
        return {"method": self.method, "median": self.median, "beta": self.beta,
                "lo_s": self.lo_s, "hi_s": self.hi_s}


_REGISTRY = {cls.method: cls for cls in (LinearNorm, LogNorm, AsinhNorm)}


def fit_normalization(pixels, method, **kwargs):
    if method not in _REGISTRY:
        raise ValueError(f"unknown normalize method: {method}")
    return _REGISTRY[method].fit(np.asarray(pixels).reshape(-1), **kwargs)


def normalization_from_dict(params):
    method = params["method"]
    if method not in _REGISTRY:
        raise ValueError(f"unknown normalize method: {method}")
    fields = {key: value for key, value in params.items() if key != "method"}
    return _REGISTRY[method](**fields)
