# ms3 dependency audit

**Scope:** `install_requires` in `setup.cfg` (ms3 2.6.5, 2026-10-13).
**Question:** which runtime dependencies are strictly necessary, which can become
extras, which can be removed.

## Summary

| Dependency | Transitive pkgs pulled | Import sites | Verdict |
|---|---|---|---|
| `frictionless[pandas,visidata]` | 28 | `utils/frictionless_helpers.py` only | **Make optional**; drop both extras |
| `pytablewriter==1.0.0` | 9 | 2 call sites | **Remove** |
| `pathos` | 5 | 3 `Pool.starmap` calls | **Remove** (stdlib `multiprocessing`) |
| `GitPython` | 3 | `corpus.py`, ~8 helpers in `utils/functions.py` | **Make optional** |
| `typing_extensions` | 0 (already pulled by bs4) | `Self` x2 | Remove via conditional import |
| `webcolors` | 0 | ~10 calls | Keep (or inline later, low value) |
| `tqdm` | 0 | 1 site; no caller passes `progress=True` | Lazy import, optional |
| `beautifulsoup4`, `lxml`, `pandas`, `numpy` | - | everywhere | **Core**, keep |

Transitive counts come from `pip install --dry-run --report` in a fresh
Python 3.12 environment. A `0` means pure-python with no further requirements.

Expected result: core install drops from roughly 55 packages to roughly 10.

## Findings per dependency

### frictionless

- `visidata` extra: zero references in `src/`. Pulls `visidata` and `windows-curses`.
- `pandas` extra: no `to_pandas` / `from_pandas` anywhere. Only pulls `pyarrow`.
- Core library: used solely in `utils/frictionless_helpers.py` through
  `fl.Schema` (URL probe and `to_yaml`), `fl.validate`, `fl.Report`,
  `fl.FrictionlessException`. Descriptor dicts are built by hand.
- Caveat: `frictionless=True` is the default on `store_*` paths in `corpus.py`,
  `piece.py` and `operations.py`, so `ms3 extract` and `ms3 transform` hit it.
- Plan: lazy import inside the helpers. When the library is missing, still write
  TSV and descriptor JSON, skip validation, warn once. The schema YAML write
  needs a small fallback (JSON, or `pyyaml` if present).

### pytablewriter

- Only consumer is `dataframe2markdown`, called from `write_markdown`
  (README metadata table) and `concatenated_metadata2markdown`.
- Pinned to `==1.0.0`, which also constrains the resolver for downstream users.
- Plan: replace with a 15-line hand-rolled markdown table writer.

### pathos

- Three `Pool(cpu_count()).starmap(...)` sites: `corpus.py` (score parsing and
  TSV parsing) and `utils/functions.py` (MuseScore conversion).
- `LoggedClass` already defines `__getstate__` / `__setstate__` that strip the
  logger, so stdlib pickle should cope.
- Risk: returned `Score` objects carry a bs4 soup. Must run `parallel=True` to
  confirm stdlib pickling works. If it does not, keep as `ms3[parallel]` extra.

### GitPython

- `Corpus.__init__` always calls `get_git_repo`. Other uses: version info for
  metadata, `get_facet_at_git_revision`, comparison against a git revision.
- Three test files import `git` directly.
- Plan: lazy import. `get_git_repo` returns `None` when the library is missing;
  revision-based features raise with an install hint. Add `GitPython` to the
  `testing` extra.

### typing_extensions

- Only `Self`, two sites. `typing.Self` exists from Python 3.11;
  `python_requires` is `>=3.10`.
- bs4 >= 4.13 already requires `typing-extensions`, so it costs nothing today.
- Plan: `try: from typing import Self except ImportError: from typing_extensions import Self`,
  then drop from `install_requires`.

### webcolors

- Pure python, no dependencies. `CSS_COLORS` list already copied into
  `utils/constants.py`. Inlining would be a 148-entry dict plus ~30 lines.
- Verdict: keep. Only remove for purism.

### tqdm

- Single site in `scan_directory`, gated on `progress=`. No caller sets it.
- Plan: lazy import with no-op fallback.

### Core (keep)

- `beautifulsoup4` + `lxml`: the `"xml"` parser feature *is* lxml-xml. No
  alternative. `bs4_parser.py:4321` also uses the `"lxml"` HTML parser.
- `pandas`, `numpy`: used throughout.

## Related, out of scope

- `numpy<2.0.0` pin (commit `b75e6710`, no rationale recorded). Current pandas
  releases support numpy 2. Worth testing the lift separately.

## Proposed `setup.cfg`

```ini
install_requires =
    beautifulsoup4
    lxml
    pandas>=1.5.3,<3
    numpy<2.0.0
    webcolors
    tqdm

[options.extras_require]
git = GitPython
frictionless = frictionless
all = ms3[git,frictionless]
testing =
    setuptools
    pytest
    pytest-cov
    GitPython
```

## Implementation order

1. Drop `[pandas,visidata]` extras and `pytablewriter`. Safe, no behavior change.
2. Replace `pathos` with stdlib `multiprocessing`. Run tests with `parallel=True`.
3. Lazy-import `GitPython` behind `ms3[git]`.
4. Lazy-import `frictionless` behind `ms3[frictionless]`, with graceful fallback.
5. Conditional `Self` import; drop `typing_extensions`.

---

Prepared by Claude Fable 5.1 (Claude Code), 2026-10-13. Findings based on static
grep of `src/ms3` and pip dry-run resolution; no code changed.
