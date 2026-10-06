# AGENTS.md

This file provides guidance to AI coding agents when working with code in this repository.

## What this is

MNE-NIRS extends MNE-Python's fNIRS support: GLM analysis (`mne_nirs.statistics`, built on
nilearn and statsmodels), signal enhancement, quality metrics, SNIRF writing, fOLD and
other I/O helpers, simulation, fNIRS-specific datasets, and visualization. MNE-Python owns
the data containers (`Raw`, `Epochs`, `Info`) and the fNIRS readers; this package builds on
them.

## Follow MNE-Python's conventions

This package is a subsidiary of MNE-Python. Unless something below says otherwise, follow
[MNE-Python's AGENTS.md](https://github.com/mne-tools/mne-python/blob/main/AGENTS.md)
(read it rather than guessing): keep changes small, naming, numpydoc style, imports,
deprecation policy, compact tests, license rules for adapted code, and in particular its
[policy on AI assistance](https://github.com/mne-tools/mne-python/blob/main/CONTRIBUTING.md#policy-on-ai-assistance-in-contributions):

- Work test-first: write (or extend) a test that fails for the right reason, then make it
  pass. Promote anything a throwaway script caught into a real test.
- Do not open pull requests, push, or commit unless explicitly asked; the human submitting
  the change must review, understand, and disclose AI use in the PR description.
- Keep changes minimal and scoped to the request; mention, don't silently fix, unrelated
  problems you notice.

What does *not* carry over from MNE-Python:

- The changelog is the GitHub releases page, so there are no towncrier fragments to add.
- There are no lazy `__init__.pyi` stubs: each subpackage's `__init__.py` imports its
  public names directly, and new public API must also be added to `doc/api.rst`.
- There is no `docdict` here. Use MNE-Python's `@verbose` / `@fill_doc` from `mne.utils`
  (the legacy decorators that MNE-Python keeps for downstream packages), not its `_static`
  variants or hook.
- The default branch is `main`.

## Things to know

- Minimum versions of MNE-Python and other dependencies are in `pyproject.toml`; code that
  depends on a newer MNE-Python gates on `mne.utils.check_version` (grep for existing
  examples). Private MNE-Python helpers (`_validate_type`, `_check_option`, ...) are used
  freely, so they can break when MNE-Python's `main` changes.
- `visualisation` is spelled the British way (`mne_nirs.viz` is an alias).
- Tests need data: the MNE testing dataset plus the datasets fetched in
  `tools/github_actions_download.sh` (the fOLD tests instead use small bundled files in
  `mne_nirs/io/fold/tests/data` and skip without `xlrd`).
- `mne_nirs/conftest.py` turns warnings into errors; add narrowly scoped `ignore` lines
  there (with a comment naming the source) for third-party warnings.

## Tests and docs

`pytest -n auto mne_nirs/` runs the unit tests (CI uses pytest-xdist). Tests write to a
per-process fake home (the `protect_config` fixture), so resolve test data paths at
module level, e.g. `data_path(download=False)`, rather than inside test functions. The
gallery examples in `examples/` are not unit tests: they run when the docs (Sphinx +
sphinx-gallery) are built on CircleCI with `make -C doc html`, which also reports
coverage. Run `pre-commit run --all-files` (ruff and ruff-format) before
handing work back.
