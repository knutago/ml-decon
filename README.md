# ml-decon

Machine-learning deconvolution of astronomical images: recover the underlying
intensity field (`ideal`) from a PSF-blurred, noisy observation (`observed`).

The current focus is the **evaluation workflow** — a model-agnostic scorer that takes an
`(observed, predicted, ideal)` triple and returns metrics, so classical baselines
(Richardson–Lucy, Wiener) and every member's network are scored through one path.
Read [`CLAUDE.md`](CLAUDE.md) first: it's the working agreement (layout, naming, data
rules) binding on both humans and coding agents.

## Layout

| Package | Purpose |
|---|---|
| `core/` | config schema, seeding/device, normalization — no ML, no result I/O |
| `dataset/` | FITS → patch arrays, torch Dataset |
| `evaluation/` | metrics and reporting — the shared, model-agnostic contract |
| `script/` | thin CLI entry points; orchestration only |
| `test/` | runnable assert scripts (not pytest) |

`model/`, `config/` don't exist yet — created when needed. `core` depends on nothing
internal; `dataset`/`evaluation` depend on `core`; `script` depends on all. Never import
`script` from a package.

## Setup

Python 3.12 via [`uv`](https://docs.astral.sh/uv/):

```
uv sync
```

Data lives **outside** the repo — no FITS, `.npy`, or checkpoints are ever committed. Each
member's config YAML holds per-user absolute data paths.

## Running

```
uv run python -m dataset.gen_data config/<experiment>.yaml   # build a patch dataset
uv run python -m test.<name>                                 # run a contract test
```

See `CLAUDE.md` for the full set of intended entry points (`script.train`, `script.eval`).

## IDL simulation port

`dataset/idl_repro.py` reproduces the legacy IDL chain that built the mock observations
(`*_avg.fits` → `*_trim.fits`) via rescale → addnoise → dotrim, with scale/sky constants
from the `addnoiseNN.pro` routines. The noise-free stages are reproduced exactly (dotrim
is byte-identical); the addnoise noise is checked statistically since the IDL seed is lost.

```
uv run python -m script.verify_pairs   <data_dir>   # verify port vs IDL FITS; write repro_table.csv
uv run python -m script.report_scaling <trim_fits>  # arithmetic mapping trim <-> sharp-image scale
uv run python -m test.test_idl_repro                # data-free contract tests
```
