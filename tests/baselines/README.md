# GPU references

`rtx-5070-ti-sm120.json` preserves the measured local
`benchmark-results/gpu-local-new.json` report from October 31, 2026.
Its CUDA and wrapper hashes match the source when this reference was added.
The report records a dirty working tree; its source/binary hashes and original
provenance are retained verbatim. These are measurements, not target timings.

The CUDA workflow requires a self-hosted Linux runner labelled `rtx-5070-ti`
with this GPU, CUDA toolkit (including nvcc supporting sm_120), driver/runtime,
PCI placement, OS and Python environment. Validation deliberately rejects
environment differences, even on another GPU with the same architecture.
Run manually through workflow_dispatch; set repository variable
`CAYLEY_GPU_CI=true` to also run on pushes to main after provisioning the runner.
CPU/Rust CI runs automatically on Linux, macOS and Windows.

Comparison allows 20% median slowdown and never rewrites this report.
For a different runner, record a new complete report using
`--run-benchmarks --benchmark-record tests/baselines/<device>-<environment>.json`,
review its samples and provenance, and update the workflow's path, runner label
and Python/CuPy versions. Never replace a failing reference automatically.
