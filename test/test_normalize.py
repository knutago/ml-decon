"""Round-trip test: inverse(forward(x)) == x is the invertibility contract every
normalization must hold. Run as `uv run python -m test.test_normalize`.

Also guards the asinh knee on a SPARSE field -- point sources on exact zero, like
m3201newK.fits -- where sigma clipping discards every source and the clipped std comes
back as exactly 0. That case silently produced two unusable datasets under the old
`beta = asinh_softening * float(std) or 1.0`.
"""

import warnings

import numpy as np

from core.normalize import fit_normalization, normalization_from_dict

SEED = 0
METHODS = ["linear", "log", "asinh"]


def sample_pixels(rng):
    """A background floor plus a sparse bright tail, like an observed astronomical field."""
    background = rng.normal(100.0, 5.0, size=10000)
    sources = rng.uniform(200.0, 5.0e4, size=200)
    return np.concatenate([background, sources]).astype(np.float32)


def sparse_pixels(rng):
    """A CATALOGUE RENDERING: 96% exact zeros, the rest a lognormal source population.
    Scaled like m3201newK -- median source ~0.017, brightest ~300."""
    pixels = np.zeros(100000, dtype=np.float32)
    n = 4000
    pixels[:n] = np.exp(rng.normal(np.log(0.017), 1.2, size=n)).astype(np.float32)
    pixels[0] = 300.0
    rng.shuffle(pixels)
    return pixels


def check_roundtrip(method, pixels):
    norm = fit_normalization(pixels, method)
    recovered = norm.inverse(norm.forward(pixels))
    assert np.allclose(recovered, pixels, rtol=1e-4, atol=1e-2), \
        f"{method}: forward/inverse not identity (max err {np.abs(recovered - pixels).max():.3g})"

    # params survive a serialization round-trip
    rebuilt = normalization_from_dict(norm.to_dict())
    assert np.allclose(rebuilt.forward(pixels), norm.forward(pixels)), \
        f"{method}: normalization_from_dict changed the mapping"


def check_sparse_asinh(pixels):
    """On a sparse field the fit must warn and derive a beta from the sources, never
    invent a unit-free constant, and never leave the sources crushed against z = 0."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        norm = fit_normalization(pixels, "asinh", asinh_softening=3.0)
    assert any(issubclass(w.category, RuntimeWarning) for w in caught), \
        "sparse asinh fit must warn that beta could not come from the background"

    sources = pixels[pixels > 0]
    assert np.isclose(norm.beta, np.median(sources), rtol=1e-5), \
        f"beta should fall back to the median source flux, got {norm.beta:.6g}"
    assert norm.beta != 1.0, "beta = 1.0 is the old unit-free fallback, not a scale"

    # the real failure was not a crash but a squashed z domain: under beta = 1.0 the
    # median source landed at z = 0.0028 on m3201newK, against 0.052 on m31bK50.
    median_z = float(np.median(norm.forward(sources)))
    assert median_z > 0.02, \
        f"median source sits at z = {median_z:.5f}; the asinh has collapsed to linear"

    # and the map is still exactly invertible
    assert np.allclose(norm.inverse(norm.forward(pixels)), pixels, rtol=1e-4, atol=1e-6), \
        "sparse asinh: forward/inverse not identity"
    return median_z


def check_explicit_beta(pixels):
    norm = fit_normalization(pixels, "asinh", asinh_beta=0.014)
    assert norm.beta == 0.014, f"asinh_beta override ignored, got {norm.beta}"
    for bad in (0.0, -1.0, float("nan")):
        try:
            fit_normalization(pixels, "asinh", asinh_beta=bad)
        except ValueError:
            continue
        raise AssertionError(f"asinh_beta={bad} should have raised")


def check_degenerate():
    """No background AND no sources: there is no scale to recover, so raise."""
    for pixels in (np.zeros(1000, dtype=np.float32), np.full(1000, 7.0, dtype=np.float32)):
        try:
            fit_normalization(pixels, "asinh")
        except ValueError:
            continue
        raise AssertionError("a constant field should have raised, not returned a norm")


def main():
    rng = np.random.default_rng(SEED)
    pixels = sample_pixels(rng)
    for method in METHODS:
        check_roundtrip(method, pixels)
        print(f"{method}: round-trip OK")
    print("all normalizations invertible")

    sparse = sparse_pixels(rng)
    check_roundtrip("asinh", sparse)
    median_z = check_sparse_asinh(sparse)
    print(f"sparse asinh: warned, beta from sources, median source z = {median_z:.4f} OK")
    check_explicit_beta(sparse)
    print("asinh_beta override: honoured, and rejects non-positive values OK")
    check_degenerate()
    print("degenerate constant field: raises OK")


if __name__ == "__main__":
    main()
