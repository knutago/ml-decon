"""Contract tests for idl_repro, the Python port of the IDL rescale/addnoise/dotrim chain.

    uv run python -m test.test_idl_repro

Runs on synthetic arrays only -- no FITS, so it is runnable without the (out-of-repo) data.
The data-dependent check that the port actually reproduces the delivered *_avg/_scl/_obs2/
_trim files lives in verify_pairs.py next to those files; this asserts the maths each
idl_repro function promises:

  * rescale is an exact scalar multiply, and its factor is recoverable from total ratios;
  * addnoise's noise-free backbone is (scale*scl + sky)/nexp, clipped at 1e5, with effective
    constants matching addnoise_params, and its noise is zero-mean with variance
    signal/nexp^2 + rn^2/nexp^3 (so fit_addnoise/fit_noise_model recover the inputs);
  * dotrim's center crop matches IDL's dsz/2 integer-offset indexing, even and odd.
"""

import numpy as np

from dataset import idl_repro

SEED = 0
BAND, CONFIG = "K", "20"  # nexp=55: small enough that noise is easy to measure precisely


def test_rescale_exact_and_recoverable():
    """apply_rescale is an exact multiply; rescale_factor recovers it from total ratios."""
    rng = np.random.default_rng(SEED)
    avg = rng.uniform(1.0, 100.0, size=(200, 200))
    k = 1.234e7
    scl = idl_repro.apply_rescale(avg, k)
    assert np.array_equal(scl, avg * k), "apply_rescale is not an exact scalar multiply"
    # scl = avg*k, so total(scl)/total(avg) == k to floating precision
    k_hat = idl_repro.rescale_factor(avg, scl)
    assert abs(k_hat - k) / k < 1e-12, f"rescale_factor off: {k_hat} vs {k}"
    print(f"rescale: exact multiply, factor recovered to {abs(k_hat - k) / k:.1e}")


def test_addnoise_backbone_and_clip():
    """Noise-free apply_addnoise == (scale*scl + sky)/nexp, matches addnoise_params, clips."""
    p = idl_repro.addnoise_params(BAND, CONFIG)
    scl = np.linspace(0.0, 4.0, 500 * 500).reshape(500, 500)
    clean = idl_repro.apply_addnoise(scl, BAND, CONFIG)  # rng=None -> backbone
    expected = (scl * p["raw_scale"] + p["raw_sky"]) / p["nexp"]
    assert np.allclose(clean, expected, rtol=0, atol=0), "backbone != (scale*scl+sky)/nexp"
    # effective constants are raw/nexp
    assert abs(p["scale"] - p["raw_scale"] / p["nexp"]) < 1e-9
    assert abs(p["sky"] - p["raw_sky"] / p["nexp"]) < 1e-9
    # saturation: a huge input clips at SAT_LEVEL
    hot = idl_repro.apply_addnoise(np.array([[1e9]]), BAND, CONFIG)
    assert hot[0, 0] == idl_repro.SAT_LEVEL, "1e5 saturation clip not applied"
    print(f"addnoise backbone: exact, scale={p['scale']:.5g} sky={p['sky']:.5g}, clip OK")


def test_addnoise_noise_model_recovers_inputs():
    """Seeded noisy obs2: fit_addnoise recovers scale/sky, fit_noise_model recovers nexp,
    and the noise is zero-mean."""
    p = idl_repro.addnoise_params(BAND, CONFIG)
    scl = np.linspace(0.0, 4.0, 500 * 500).reshape(500, 500)
    rng = np.random.default_rng(SEED)
    obs2 = idl_repro.apply_addnoise(scl, BAND, CONFIG, rng=rng)

    scale_hat, sky_hat = idl_repro.fit_addnoise(scl, obs2)
    assert abs(scale_hat - p["scale"]) / p["scale"] < 1e-3, "scale not recovered"
    assert abs(sky_hat - p["sky"]) / p["sky"] < 1e-3, "sky not recovered"

    clean = idl_repro.apply_addnoise(scl, BAND, CONFIG)
    resid = obs2 - clean
    assert abs(resid.mean()) < 1e-3 * resid.std(), "noise is not zero-mean"

    # fit_noise_model fits var = signal/D + rn^2/D and returns D; the true slope is
    # 1/nexp^2, so D == nexp^2 and sqrt(D) recovers nexp.
    nm = idl_repro.fit_noise_model(clean, resid)
    nexp_hat = nm["nexp"] ** 0.5
    assert abs(nexp_hat - p["nexp"]) / p["nexp"] < 0.05, \
        f"nexp not recovered: {nexp_hat:.1f} vs {p['nexp']}"
    print(f"addnoise noise: scale/sky recovered, mean~0, nexp {nexp_hat:.1f} vs {p['nexp']}")


def test_center_crop_matches_idl_indexing():
    """center_crop keeps img[dsz//2 : dsz//2+out] on each axis, for even and odd dsz."""
    for dy, dx in [(8, 8), (7, 5)]:  # even, then odd differences
        full = np.arange((20 + dy) * (20 + dx)).reshape(20 + dy, 20 + dx)
        out_shape = (20, 20)
        got = idl_repro.center_crop(full, out_shape)
        y0, x0 = dy // 2, dx // 2
        expected = full[y0:y0 + 20, x0:x0 + 20]
        assert np.array_equal(got, expected), f"center_crop wrong for dsz=({dy},{dx})"
    print("dotrim: center crop matches IDL dsz//2 indexing (even and odd)")


def test_scaling_recipe_affine_and_inverse():
    """scaling_recipe's net affine is (k*scale, sky), and its inverse round-trips avg."""
    k = 2.5e7
    r = idl_repro.scaling_recipe(BAND, CONFIG, k)
    p = idl_repro.addnoise_params(BAND, CONFIG)
    assert r["a"] == k * p["scale"], "net slope a != k*scale"
    assert r["b"] == p["sky"], "net offset b != sky"
    # forward trim = a*avg + b, then the reported inverse recovers avg exactly
    avg = np.array([1e-6, 2e-6, 5e-6])
    trim = r["a"] * avg + r["b"]
    recovered = (trim - r["b"]) / r["a"]
    assert np.allclose(recovered, avg, rtol=1e-12, atol=0), "inverse does not round-trip avg"
    # the report renders without error and names the inverse
    assert "avg = (trim -" in idl_repro.format_scaling_report(r)
    print(f"scaling recipe: net affine a={r['a']:.4g} b={r['b']:.4g}, inverse round-trips")


def main():
    test_rescale_exact_and_recoverable()
    test_addnoise_backbone_and_clip()
    test_addnoise_noise_model_recovers_inputs()
    test_center_crop_matches_idl_indexing()
    test_scaling_recipe_affine_and_inverse()
    print("all idl_repro contracts hold")


if __name__ == "__main__":
    main()
