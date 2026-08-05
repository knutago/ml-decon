"""Round-trip test: inverse(forward(x)) == x is the invertibility contract every
normalization must hold. Run as `uv run python -m test.test_normalize`.
"""

import numpy as np

from core.normalize import fit_normalization, normalization_from_dict

SEED = 0
METHODS = ["linear", "log", "asinh"]


def sample_pixels(rng):
    """A background floor plus a sparse bright tail, like an observed astronomical field."""
    background = rng.normal(100.0, 5.0, size=10000)
    sources = rng.uniform(200.0, 5.0e4, size=200)
    return np.concatenate([background, sources]).astype(np.float32)


def check_roundtrip(method, pixels):
    norm = fit_normalization(pixels, method)
    recovered = norm.inverse(norm.forward(pixels))
    assert np.allclose(recovered, pixels, rtol=1e-4, atol=1e-2), \
        f"{method}: forward/inverse not identity (max err {np.abs(recovered - pixels).max():.3g})"

    # params survive a serialization round-trip
    rebuilt = normalization_from_dict(norm.to_dict())
    assert np.allclose(rebuilt.forward(pixels), norm.forward(pixels)), \
        f"{method}: normalization_from_dict changed the mapping"


def main():
    rng = np.random.default_rng(SEED)
    pixels = sample_pixels(rng)
    for method in METHODS:
        check_roundtrip(method, pixels)
        print(f"{method}: round-trip OK")
    print("all normalizations invertible")


if __name__ == "__main__":
    main()
