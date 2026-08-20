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

import warnings

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


# A field is SPARSE -- a catalogue rendering rather than an image -- when this fraction of
# its pixels sit at exactly the sigma-clipped median. Such a field has a constant floor,
# not a background distribution, so there is no noise std for the knee to be a multiple of.
# The separation is not marginal, so the threshold does not need to be tuned:
#   m3201newK.fits           91.07%   (catalogue rendering, sparse)
#   m31bK50.fits              0.00%   (field with an unresolved-source floor)
#   Klong.fits                0.00%   (observed frame)
#   m3201newK_floored.fits    0.00%   (same catalogue, floor added)
_SPARSE_AT_MEDIAN = 0.20


def fit_asinh_beta(pixels, median, std, asinh_softening=3.0, asinh_beta=None):
    """Choose the asinh knee, or fail loudly rather than invent one.

    Normally beta = asinh_softening * sigma-clipped std, which puts the knee a few noise
    widths above the background. That fails on a SPARSE field -- point sources sitting on
    an exact constant floor -- because sigma clipping discards the only pixels carrying
    signal, leaving a "background" that is one repeated value.

    This used to read `beta = asinh_softening * float(std) or 1.0`. That fallback is not a
    scale at all -- it is 1.0 in whatever units the FITS happens to carry, so it is wrong
    by the image's own normalization. On m3201newK it landed ~57x too large, which
    flattens asinh into a linear map (asinh(x/beta) ~= x/beta for x << beta) and crushes
    every source toward zero. Measured on those patches: the median source lands at
    z = 0.0028 under the 1.0 fallback vs 0.098 under a fitted beta, against 0.052 on
    m31bK50, the pair that trains correctly. Worse, it was SILENT: the dataset generated
    and trained without complaint and only surfaced much later as a reconstruction whose
    invented background rivalled its faint sources.

    Testing `std == 0` is NOT enough to catch this, which is the subtler half of the bug.
    Let a handful of very faint sources survive the clip and the std comes back at, say,
    5e-07 -- nonzero, so the old expression sailed through, but 5e-07 is not a background
    width either. It is however many faint pixels happened to escape clipping, and the
    beta it implies is off by orders of magnitude in the other direction. So the test here
    is on the SHAPE of the clipped population: if most pixels sit at exactly the median,
    there is a constant floor and no distribution to measure, whatever the std says.

    When that happens, take the scale from the SOURCES instead -- the median of the pixels
    above the median -- and warn. On m3201newK that gives 0.0175 against m31bK50's fitted
    0.0105: the same order of magnitude, which is the whole point. Switching such a field
    to `linear` is NOT the alternative and is far worse: min-max against the brightest
    pixel puts the median source at z = 9.6e-05.

    Raises if no scale can be recovered at all, rather than guessing again.
    """
    if asinh_beta is not None:
        beta = float(asinh_beta)
        if not np.isfinite(beta) or beta <= 0:
            raise ValueError(f"asinh_beta must be finite and > 0, got {asinh_beta!r}")
        return beta

    pixels = np.asarray(pixels)
    at_median = float(np.mean(pixels == median))
    beta = float(asinh_softening) * float(std)
    if np.isfinite(beta) and beta > 0 and at_median < _SPARSE_AT_MEDIAN:
        return beta

    above = pixels[pixels > median]
    fallback = float(np.median(above)) if above.size else float("nan")
    if not np.isfinite(fallback) or fallback <= 0:
        raise ValueError(
            f"cannot fit an asinh beta: the sigma-clipped std is {float(std)!r}, so "
            f"asinh_softening * std is {beta!r}, and the {above.size} pixels above the "
            f"median give {fallback!r}. Pass an explicit asinh_beta.")
    warnings.warn(
        f"asinh: beta cannot be fit from the background -- this field is sparse "
        f"({at_median:.2%} of pixels sit at exactly the clipped median {median:.6g}, and "
        f"asinh_softening * std = {beta:.6g}). Falling back to the median of the "
        f"{above.size} pixels above it: beta = {fallback:.6g}. NOTE asinh_softening has "
        f"no effect on a field like this. Pin the knee with asinh_beta to choose it "
        f"yourself.",
        RuntimeWarning, stacklevel=3)
    return fallback


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
    def fit(cls, pixels, asinh_softening=3.0, asinh_beta=None, **_):
        # beta is a few times the sigma-clipped noise std; clipping ignores the sources so
        # beta tracks the background noise. See fit_asinh_beta for what happens when there
        # is no background to track -- that path used to invent beta = 1.0 in silence.
        _, median, std = sigma_clipped_stats(pixels, sigma=3.0)
        median = float(median)
        lo = float(pixels.min())
        hi = float(pixels.max())
        if not hi > lo:
            raise ValueError(f"asinh fit needs a non-degenerate range, got min == max == {lo!r}")
        beta = fit_asinh_beta(pixels, median, std, asinh_softening, asinh_beta)
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
