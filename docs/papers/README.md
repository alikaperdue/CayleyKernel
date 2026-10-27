# Paper verification and saved evidence

Paper reviews document finite checks, conventions and limitations alongside
committed tests. They are not proofs of unrestricted mathematical claims.

## Running current tests

From the repository root, with the package and test dependencies available:

```bash
python3 -m pytest -v tests/papers/
```

For the 16-paper collection or an individual companion, use the commands in
[the collection index](desktop-review/README.md) or the relevant paper README.
NumPy is needed for numerical companions. GPU methods require CuPy, a built
kernel and a usable CUDA device; missing prerequisites cause explicit skips.
Several suites perform expensive exhaustive or million-component experiments.
A test failure after device prerequisites succeed is not an expected skip.

The ordinary test commands do not need the original PDFs and do not rewrite
saved evidence. Retained paper-specific `run` modules can optionally write
new reports; use a separate output directory if preserving a historical run.
Any `/path/to/papers/` argument is a placeholder for your own downloaded PDF,
used only by retained runners to hash it for provenance.

## Historical snapshots

Committed JSON, CSV, SVG and already tracked text logs are reference evidence
from particular runs. Their environment, test outcomes, timestamps (where
recorded), source hashes and PDF hashes apply to those runs, not automatically
to the current source. Later additions may change coverage and test counts.

Source-hash entries may name retired report-generation helpers. Those entries
remain intact as historical provenance; the helper files are not required to
execute the current tests. Relative hash keys in older reports can be relative
to the corresponding test package rather than the repository root. No hash
has been recomputed or result relabeled as a new run during this documentation
cleanup.

For the desktop collection, per-paper snapshots supersede the counts in the
older `desktop-review/summary.json`. That aggregate remains historical, and
the collection README explicitly distinguishes the two runs. Consult the
linked per-paper JSON for the outcomes summarized in its table.

## What belongs in Git

Keep explanations, findings, compact reference JSON/CSV/SVG evidence, and
required test fixtures. Downloaded PDFs, OCR text, rendered pages and new raw
console logs remain local. Ignoring a path does not remove an already tracked
file; existing text logs are preserved. Save new console logs outside the
repository or in an ignored location instead of overwriting historical ones.
