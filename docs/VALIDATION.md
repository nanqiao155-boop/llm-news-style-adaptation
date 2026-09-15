# Local validation — Round 1

All final commands ran with the staging folder as working directory and
`PYTHONDONTWRITEBYTECODE=1`. The validation environment is outside the release folder.

| Command / check | Result |
|---|---|
| `python -m pytest -q -p no:cacheprovider` | 19 passed |
| `python scripts/validate_release.py` | PASS; source compile/import, JSON/JSONL/CSV/YAML, Markdown links, file policy, secrets and paths |
| `python scripts/plot_public_results.py` | Six PNG figures generated from aggregate-only files |
| Gradio `build_app()` and callback tests | PASS; correct five-output generator; PASS and FAIL branches |
| Loopback HTTP launch, request and shutdown | HTTP 200, neutral title, no brand asset reference; server closed |
| Source SHA-256 comparison | 85 selected input file hashes unchanged |
| Visual inspection | All six figures readable, zero-based bar axes, no logo or private annotations |

Tests cover frozen Release-Adjusted caps/penalties, identical PASS output with skipped
Reviser, one FAIL revision, downstream stop after Reviewer error, invalid input,
schema retry bound, split/Test guard, repetition threshold and Gradio callbacks.
Structural validation compiles all Python in memory without creating bytecode.
The public scan contains no matches requiring review and no files above 50 MiB.
See `validation_report.json`, `secret_scan_report.csv` and the root large-file report.

## Corrections during preparation

The first pytest invocation used the source working directory and could not import
the newly created public demo module; it was corrected to staging. Staging bytecode
caches were removed. Copied CSV newline artifacts were normalized; no numeric result
was changed. The default address in historical provider code now uses a reserved
placeholder. A numeric RAM-size literal was explicitly identified as a phone-pattern
false positive and formatted with Python digit separators without changing its value.
Strength-sweep Repeat>=3 counts and severe Repeat>=5 counts are recorded separately.

## Not executed / limitations

No training, model inference, live Judge/Reviewer API request, model download,
quantization or full private-data pipeline was run. Those require excluded artifacts,
GPU/model runtime or configured services. Historical lightweight reference runners
were syntax-checked only; they intentionally retain dependencies on excluded
workspace artifacts and are labelled non-portable. Remote link availability was not
verified; local Markdown targets were checked. No Git operation was performed.

Packages were installed into a separate validation venv with access to existing
system packages. pip reported unrelated inherited TensorFlow/Keras dependency
conflicts; this is not a clean-environment training compatibility certificate.
The exercised public application and all selected tests passed. Direct tested package
versions are recorded in `validated_environment.json`; requirements use bounded ranges.

## Round 2 final gate

The user confirmed public distribution rights and froze the README technical story.
Only stale release-approval wording and the local-only review document were removed.
No license was created. The repository scanner now skips Git metadata so it can
continue to validate public content after initialization.

- `python -m pytest -q -p no:cacheprovider`: 19 passed.
- `python scripts/validate_release.py`: PASS; 52 Python compile checks, 39 imports,
  28 JSON, one synthetic JSONL, nine CSV, nine YAML and 24 local Markdown targets.
- Secret scan: NO_CREDIBLE_SECRET; no large files.
- Data directory contains only README and the explicitly synthetic sample.
- GitHub CLI was absent from PATH and the standard installation locations checked.
- Existing Git user.name and user.email were both unset; no identity was invented.
- Git initialization, commit, repository creation, push, git diff/status and remote
  verification were not performed because publication tooling is unavailable.
- Full model/private-data experiments remain unexecuted for the reasons listed above.

Current release status: GITHUB_CLI_REQUIRED.

## Round 2 continuation

Official GitHub CLI installation and authentication are complete. The user approved
GitHub username/noreply identity, configured only in this repository. Final pre-commit
checks: 19 tests passed; structural validation PASS; NO_CREDIBLE_SECRET.

## Publication completed

Public repository created and main pushed normally, without force. Remote commit and
124-file tree matched the verified local repository; README, six neutral figures and
aggregate results are present. Data contains only its README and synthetic fixture.
No weights, private env, brand logo, archive, defense files or internal review document
were published. No LICENSE was added. Topics were set. Working-tree verification is
performed after the final documentation sync. Earlier blockers above are historical.
