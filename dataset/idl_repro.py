"""Python reproduction of the IDL image-simulation chain (rescale -> addnoise -> dotrim).

Library only: importing this runs nothing. Use script/verify_pairs.py to test it against
the IDL-produced FITS files and build the scale/sky table, and test/test_idl_repro.py for
the data-free contract tests.

The original IDL turned an ideal image ``*_avg.fits`` into a mock observation
``*_trim.fits`` in three steps, one .pro file each:

    rescale.pro   avg  -> scl   scl = avg * total(ideal)/total(avg)          (pure rescale)
    addnoiseNN.pro scl -> obs2  obs2 = (scale*scl + sky + noise)/nexp        (photometry + noise)
    dotrim.pro    obs2 -> trim  trim = center-crop of obs2 to the ideal size (geometry)

Two facts drive the design:

  * The noise in addnoise came from ``randomn(s,...)`` with an undefined seed, so it is
    NOT byte-reproducible. Its form is, though. So we reproduce the deterministic backbone
    and test the noise statistically.
  * The scale/sky constants come from the addnoiseNN.pro routines (see ADDNOISE_PARAMS via
    addnoise_params), and are independently RECOVERED from the files as a cross-check
    (they agree to 0.00%).

The key subtlety of the real routines: each writes ``(img1/nexp) < 1e5`` -- it divides the
whole frame by nexp and clips at 1e5 ADU. So the constants seen in the written obs2 are
``scale/nexp`` and ``sky/nexp``, and the per-pixel noise variance is
``signal/nexp^2 + rn^2/nexp^3`` (shot + read, carried through the /sqrt(nexp) then the
/nexp). gain is 1, so its division is a no-op. Units are raw FITS ADU throughout.
"""

import numpy as np

# --- Constants from the addnoiseNN.pro routines and doaddnoise.pro -----------------
# gain=1 makes the gain division a no-op, and obs2 never reached the 1e5 clip (max ~9.99e4).
EXPTIME = 3600.0        # seconds; doaddnoise.pro passes 3600 to every call
READ_NOISE = 15.0       # rn, e- (== ADU since gain=1)
GAIN = 1.0
SAT_LEVEL = 1e5         # ADU
ATTENUATION = 10 ** (-0.4 * (25.0 - 15.0))  # zeropoint term 10^(-0.4*(25-15)) = 1e-4

# Per (config, band): scale coefficient and sky rate (ADU/s), transcribed from addnoiseNN.pro.
# raw scale = coef * ATTENUATION * EXPTIME; raw sky = sky_rate * EXPTIME.
_SCALE_COEF = {
    "20": {"J": 301572.0, "H": 358902.0, "K": 191237.0},
    "30": {"J": 678536.0, "H": 807529.0, "K": 430284.0},
    "50": {"J": 1884823.0, "H": 2243135.0, "K": 1195233.0},
    "100": {"J": 7539294.0, "H": 8972542.0, "K": 4780934.0},
}
_SKY_RATE = {
    "20": {"J": 3.8, "H": 31.6, "K": 24.3 + 23.7},
    "30": {"J": 3.6, "H": 29.6, "K": 22.8 + 22.2},
    "50": {"J": 4.2, "H": 34.7, "K": 26.8 + 26.0},
    "100": {"J": 4.2, "H": 34.7, "K": 26.8 + 26.0},
}
# nexp per tag, from the doaddnoise.pro *_obs2 calls.
_NEXP = {
    "J20": 86, "H20": 157, "K20": 55, "J30": 183, "H30": 347, "K30": 148,
    "J50": 444, "H50": 850, "K50": 367, "J100": 609, "H100": 1783, "K100": 971,
}


def addnoise_params(band, config):
    """Effective addnoise constants for one (band, config), derived from source.

    Returns dict with the raw (pre-/nexp) scale and sky the IDL formed, the nexp divisor,
    and the effective scale/sky that appear in the written obs2 (raw/nexp) -- the values
    fit_addnoise recovers from the files.
    """
    raw_scale = _SCALE_COEF[config][band] * ATTENUATION * EXPTIME
    raw_sky = _SKY_RATE[config][band] * EXPTIME
    nexp = _NEXP[f"{band}{config}"]
    return {
        "raw_scale": raw_scale, "raw_sky": raw_sky, "nexp": nexp,
        "scale": raw_scale / nexp, "sky": raw_sky / nexp,
    }


def apply_addnoise(scl, band, config, rng=None):
    """Reproduce addnoiseNN.pro: scl -> obs2, faithfully including /nexp and the 1e5 clip.

    With rng=None the shot+read noise term is omitted, giving the deterministic backbone
    (scl*scale + sky)/nexp clipped at SAT_LEVEL. Pass a numpy Generator to add noise from
    the same shot+read model the IDL used; it matches statistically, not bit-for-bit.
    """
    p = addnoise_params(band, config)
    img1 = scl * p["raw_scale"] + p["raw_sky"]  # pre-noise, pre-/nexp signal
    if rng is not None:
        noise = rng.standard_normal(img1.shape) * np.sqrt(img1) \
            + rng.standard_normal(img1.shape) * READ_NOISE
        img1 = img1 + noise / np.sqrt(p["nexp"])
    img1 = img1 / GAIN
    return np.minimum(img1 / p["nexp"], SAT_LEVEL)


def rescale_factor(avg, scl):
    """Recover the rescale.pro scalar: scl = avg * k, k = total(scl)/total(avg).

    Returns k. rescale.pro multiplied the ideal image by total(ideal)/total(avg); since
    scl = avg * k that ratio equals total(scl)/total(avg), which we can form without the
    (now-missing) ideal image.
    """
    return float(scl.sum() / avg.sum())


def apply_rescale(avg, k):
    """Reproduce rescale.pro given the recovered factor k."""
    return avg * k


def fit_addnoise(scl, obs2):
    """Recover the addnoise photometry constants: obs2 ~ scale*scl + sky + noise.

    Least-squares slope/intercept over all pixels. The noise is zero-mean so it does not
    bias the fit. Returns (scale, sky).
    """
    a = np.vstack([scl.ravel(), np.ones(scl.size)]).T
    (scale, sky), *_ = np.linalg.lstsq(a, obs2.ravel(), rcond=None)
    return float(scale), float(sky)


def fit_noise_model(clean, resid, num_bins=40, sample=2_000_000, seed=0):
    """Recover the noise model from residuals: var(resid) = clean/nexp + rn^2/nexp.

    addnoise added randomn*sqrt(signal) (shot) + randomn*rn (read), scaled by 1/sqrt(nexp),
    so residual variance is linear in the signal level. Binning residual variance against
    the clean signal and fitting a line gives slope 1/nexp and intercept rn^2/nexp.

    Returns dict with effective nexp, read noise rn, and the raw line fit. These are
    effective values: the real nexp is entangled with unrecoverable per-routine factors,
    but the model *form* is what the test checks.
    """
    x = clean.ravel()
    r = resid.ravel()
    if x.size > sample:
        rng = np.random.default_rng(seed)
        pick = rng.choice(x.size, sample, replace=False)
        x = x[pick]
        r = r[pick]
    order = np.argsort(x)
    x = x[order]
    r = r[order]
    edges = np.linspace(x.min(), np.percentile(x, 99.5), num_bins + 1)
    idx = np.digitize(x, edges)
    bin_signal = []
    bin_var = []
    for i in range(1, num_bins + 1):
        m = idx == i
        if m.sum() > 50:
            bin_signal.append(x[m].mean())
            bin_var.append(r[m].var())
    bin_signal = np.array(bin_signal)
    bin_var = np.array(bin_var)
    design = np.vstack([bin_signal, np.ones_like(bin_signal)]).T
    (slope, intercept), *_ = np.linalg.lstsq(design, bin_var, rcond=None)
    nexp = 1.0 / slope if slope > 0 else np.inf
    rn = float(np.sqrt(max(intercept, 0.0) * nexp))
    return {"nexp": float(nexp), "rn": rn, "var_slope": float(slope),
            "var_intercept": float(intercept)}


def center_crop(img, out_shape):
    """Reproduce dotrim.pro: keep the centered out_shape sub-image.

    IDL: imgo = img[dsz/2 : dsz/2+sz0-1] with dsz = size(obs2) - size(ideal), integer
    division. Numpy floor-division // matches IDL integer division for the non-negative
    offsets here.
    """
    dy = img.shape[0] - out_shape[0]
    dx = img.shape[1] - out_shape[1]
    y0 = dy // 2
    x0 = dx // 2
    return img[y0:y0 + out_shape[0], x0:x0 + out_shape[1]]


def scaling_recipe(band, config, k):
    """Arithmetic relating the original sharp image (avg) to the trimmed observation.

    Report only: returns the constants and the ordered per-pixel operations; it touches no
    pixels. k is rescale.pro's factor total(ideal)/total(avg) for the specific image -- read
    it from the rescale_k column of repro_table.csv for the tag in question.

    The noise-free scaling collapses to the affine  trim = a*avg + b  with a = k*scale and
    b = sky, where scale/sky are the effective (per-pixel obs2) constants raw/nexp. The
    random addnoise term and the dotrim crop are listed for provenance but carry no scaling.
    """
    p = addnoise_params(band, config)
    a = k * p["scale"]
    b = p["sky"]
    forward = [
        ("rescale.pro", f"x * {k:.6g}", "k = total(ideal)/total(avg)"),
        ("addnoiseNN.pro", f"x * {p['raw_scale']:.6g}",
         f"raw_scale = {_SCALE_COEF[config][band]:.6g} * {ATTENUATION:g} * {EXPTIME:.0f}"),
        ("addnoiseNN.pro", f"+ {p['raw_sky']:.6g}",
         f"raw_sky = {_SKY_RATE[config][band]:g} * {EXPTIME:.0f}"),
        ("addnoiseNN.pro", "+ noise",
         f"shot+read / sqrt(nexp={p['nexp']}); random, NOT reproducible"),
        ("addnoiseNN.pro", f"/ {GAIN:.0f}", "gain (no-op)"),
        ("addnoiseNN.pro", f"/ {p['nexp']}", f"nexp divide, then clip at {SAT_LEVEL:g}"),
        ("dotrim.pro", "center crop", "geometry only, no scaling"),
    ]
    return {
        "band": band, "config": config, "k": k, "nexp": p["nexp"],
        "raw_scale": p["raw_scale"], "raw_sky": p["raw_sky"],
        "scale": p["scale"], "sky": p["sky"], "a": a, "b": b,
        "forward": forward,
    }


def format_scaling_report(recipe, tag=None):
    """Render scaling_recipe() as a human-readable forward+inverse report (ADU)."""
    r = recipe
    name = tag or f"m31b{r['band']}{r['config']}"
    lines = [
        f"{name}: scaling between original sharp image (avg) and trimmed observation",
        "",
        "  Forward   sharp 'avg'  ->  trim   (per pixel, in order):",
    ]
    for stage, op, note in r["forward"]:
        lines.append(f"    {op:<20s} # {stage}: {note}")
    lines += [
        "",
        f"  Net noise-free scaling:   trim = {r['a']:.6g} * avg + {r['b']:.6g}",
        f"    a = k * scale = {r['k']:.6g} * {r['scale']:.6g}",
        f"    b = sky       = {r['b']:.6g}   (= raw_sky / nexp)",
        "",
        "  Inverse   trim  ->  original sharp-image scale:",
        f"    avg = (trim - {r['b']:.6g}) / {r['a']:.6g}",
    ]
    return "\n".join(lines)
