# ml-decon — working agreement

Machine-learning deconvolution of astronomical images. Public, shared repo: several
people work here, most of them driving coding agents. Everything below is binding on
agents *and* on humans. If a rule here conflicts with a personal `~/.claude/CLAUDE.md`,
this file wins for work inside this repository.

## Current phase

**Phase 1 — build the evaluation workflow. Phase 2 — test models against it.**

We are in phase 1. The point is a scorer that is *model-agnostic*: it takes an
`(observed, predicted, ideal)` triple and returns metrics, with no knowledge of what
produced `predicted`. Classical baselines (Richardson–Lucy, Wiener) and every member's
network must be scorable through the exact same path. Anything that couples the metrics
to one architecture is a defect, not a shortcut.

Do not build model architectures ahead of the evaluation contract unless asked.

## Repository state

`core/`, `dataset/`, `evaluation/`, and `test/` exist.
`model/`, `script/`, and `config/` do not exist yet. `core/config.py` describes
**only** `seed` + `data` — there is no model or training config, deliberately, until the
evaluation contract is settled. Do not assume a module exists because an import
references it — check.

Target layout (create directories as they are actually needed, not upfront):

```
core/         config schema, seeding/device, normalization — no ML, no I/O of results
dataset/      FITS -> patch arrays, torch Dataset
evaluation/   metrics and reporting; the shared contract. Model-agnostic.
model/        architectures, one file per architecture
script/       thin CLI entry points (train.py, eval.py) — orchestration only, no algorithms
config/       one YAML per experiment
test/         runnable assert scripts
```

Directory names are **singular** (`script`, `test`, `model`). Keep it consistent.

Layering rule: `core` depends on nothing internal; `dataset`, `evaluation`, `model`
depend on `core`; `script` depends on all of them. Never import `script` from a package.
If logic is worth testing, it does not live in `script/`.

## Vocabulary — use these words, in code and in prose

| Term | Meaning |
|---|---|
| `observed` | the input: the real, PSF-blurred, noisy image |
| `ideal` | the target: the underlying intensity field we are trying to recover |
| `predicted` | model or baseline output, in the same space as `ideal` |

Never `lr`/`hr`, `input`/`output`, `gt`, `x`/`y`, `clean`/`dirty`. This is a
deconvolution problem, not super-resolution, and the naming should not imply otherwise.
If a change makes a name stale, rename it in the same change — a variable still called
`observed` after it starts holding predictions is a bug report waiting to happen.

## Code style

Python 3.12, run via `uv`. No enforced linter or formatter — the conventions below are
maintained by review, so follow them precisely rather than approximately.

- **snake_case** for modules, functions, variables; **PascalCase** for classes;
  **UPPER_SNAKE** for module-level constants.
- **Descriptive names, no abbreviations.** `index_patch`, not `i`. `num_val_blocks`,
  not `nv`. Tolerated: `idx`, `num`, `cfg` when the rest of the name carries the meaning.
- **Math is the exception**: single letters (`x`, `sigma`, `A`) are fine *only* when a
  comment or docstring right there gives the formulation and the source (paper, equation
  number) and the units.
- 4-space indent, ~100 column soft limit, standard-library / third-party / local import
  groups separated by blank lines.
- Prefer plain functions and `pathlib.Path`. No classes for things that are functions,
  no config objects for things that are arguments.

### Docstrings and comments

Every module gets a docstring with, in this order: one line on what it does, the exact
command to run it if it is runnable, and the *why* behind any non-obvious design choice.
This is the single most useful convention we have — an agent reading one file should
learn how the piece fits the pipeline. See `dataset/gen_data.py` for the shape.

Comments carry only what the code cannot:

- the expected keys/shape of a loosely-typed argument;
- a non-obvious *why* — a workaround, a hidden constraint, a subtle invariant;
- the derivation or source of a formula.

Never restate the code, never narrate the task (`# added for the eval PR`), never leave a
bare `TODO`. If a block needs a `# do the thing` header, extract a function named
`do_the_thing()` instead.

## Configuration is the experiment

A run is fully specified by one YAML file plus a top-level `seed`. Defaults live in the
dataclasses in `core/config.py`, never in the YAML, and unknown keys are a hard error.

**Every script that writes outputs must dump the resolved config next to them** as
`resolved_config.yaml`. Reproducing a run means reproducing those parameters, not bytes.
An output directory without a resolved config is not a result — it is an anecdote.

Seed Python, NumPy, and Torch from `config.seed`. We do **not** enable
`torch.use_deterministic_algorithms` or cudnn-deterministic: byte-identity costs real
speed and we only need parameter-level reproducibility. Say so if you change that.

## Data

**Data paths are per-user absolute paths in each member's config YAML.** Data never
enters the repo — no FITS, no `.npy`, no checkpoints, no PNGs, not even small ones. Git
history is forever and these files are large.

Consequences to work with, not around:

- A config from another member will not resolve on your machine. Never edit someone
  else's paths in a shared file, and never hardcode a path in code — read it from config.
- Never invent or guess a data path. If a run needs data you cannot see, stop and ask.
- Keep personal experiment configs out of shared ones. If you add a config for your own
  machine, say so in the PR so nobody mistakes it for a group baseline.

Domain gotchas that have already bitten this codebase:

- FITS is bottom-up: always `imshow(..., origin="lower")`.
- macOS writes AppleDouble sidecars (`._*.fits`) onto non-HFS drives; filter them when
  globbing a data directory.
- Normalizations must be invertible, and `inverse(forward(x)) == x` is a tested contract.
  Fit normalization on **train pixels only** — fitting on all pixels leaks the val set.
- Train/val split is by **spatial block**, not by patch, so train and val pixels never
  overlap. Any change to splitting must preserve that.
- Metrics like PSNR assume data in `[0, 1]`. Do not score raw FITS values with them, and
  state which space a metric operates in when you add one.

## Evaluation-specific rules

- Adding, removing, or changing a metric changes every number the group has reported.
  Propose it before implementing, and say what it does to existing results.
- Metrics take arrays, not models. If a metric function needs a `torch.nn.Module`, the
  abstraction is wrong.
- Report the metric's space (normalized vs. physical) and its assumptions alongside the
  value. A bare PSNR with no stated peak is not a result anyone can compare against.

## Working with agents

- **Talk before doing.** Do not implement anything that has not been discussed, and do
  not implement it in a way that has not been discussed. If the agreed approach turns out
  not to work, stop and raise it rather than quietly changing course.
- **Minimal diffs.** Touch only what the task requires. No drive-by refactors, no
  reformatting lines you did not have to edit, no import reordering. In a repo where
  several people run agents concurrently, a noisy diff is a merge conflict.
- **No silent fixes.** Spotting an unrelated bug is welcome; fixing it in the same change
  is not. Surface it and let the author decide.
- **No speculative abstraction.** Three similar lines beat a premature base class. Do not
  add config knobs, helpers, or plugin points for needs we have not hit.
- **Report honestly.** If tests fail, show the output. If a step was skipped, say which.
  Never describe an unrun script as working.

## Git

- Work on a branch; changes reach `main` through a pull request.
- **Agents never push and never merge.** Agents may commit locally and create branches;
  every `git push`, PR merge, tag, and anything else that mutates the remote is a human
  action, requested explicitly each time. Approval once does not carry to the next push.
- Commit messages: imperative mood, one line on *what* and *why*, no task narration.
- Never commit data, checkpoints, figures, `.venv`, or notebook outputs.

## Running things

```
uv run python -m dataset.gen_data config/<experiment>.yaml
uv run python -m script.train    config/<experiment>.yaml
uv run python -m script.eval     <run_dir>
uv run python -m test.<name>
```

Tests are runnable assert scripts under `test/`, not a pytest suite. Each asserts one
contract and prints what it checked.
