# 600-matrix GPU survey and retries

The original survey completed 600 cases. The retry selected exactly its 65
unresolved cases, retained the same matrix coefficients, and completed all 65.
The archived original results match the initial results byte-for-byte. Retry
vector hashes were checked. No GPU rerun was needed for this review.

| Level | Dimension | Initial verified | Recovered by retry | Final verified | Unresolved |
| --- | --- | --- | --- | --- | --- |
| 3 | 8 | 100 | 0 | 100 | 0 |
| 4 | 16 | 97 | 3 | 100 | 0 |
| 5 | 32 | 88 | 11 | 99 | 1 |
| 6 | 64 | 82 | 18 | 100 | 0 |
| 7 | 128 | 83 | 16 | 99 | 1 |
| 8 | 256 | 85 | 15 | 100 | 0 |
| Total | | 535 | 63 | 598 | 2 |

The retry made 169 starting-point attempts. Forty-three of the 63 recovered
cases succeeded on its first attempt, which negates the saved lower-coordinate
candidate x (not the entire eigenvector). This indicates that starting-point
choice explains many initial failures. The other 20 recoveries needed more
attempts. Both remaining cases exhausted all 12 starts.

## Independent audit

The original 600 candidates were checked by independent recursive NumPy
multiplication in [survey-600-audit.json](results/survey-600-audit.json).
The 65 retry candidates received the same treatment, checking both rows of
`A*(1,x)=(1,x)*lambda`, in
[survey-600-retry-audit.json](results/survey-600-retry-audit.json).
All status classifications agreed at absolute Euclidean tolerance 1e-8. The
largest independently computed residual among the 63 accepted retry vectors
was 6.14e-10 (rounded upward).

## Cases still unresolved

| Level | Dimension | Matrix seed | Original residual | Best retry residual |
| --- | --- | --- | --- | --- |
| 5 | 32 | 20266086 | 0.27705536 | 0.23148321 |
| 7 | 128 | 20268071 | 0.70859384 | 0.25374934 |

These residuals are far above tolerance, not borderline roundoff. They mean
the solver did not find an eigenpair, not that the matrix has none. Searches
were limited to vectors `(1,x)` and a particular Newton/GMRES strategy.
The next informative work is targeted investigation of these two matrices:
an alternative solver, conditioning diagnostics, and a broader eigenvector
parameterization. That follow-up has not been performed here.

## Interpretation and scope

The sampled family now has verified eigenpairs for 598/600 matrices (99.67%).
This is a finite empirical result, not a population success probability or a
proof of existence for every matrix. The entries have sparse, normalized
random coefficients; arbitrary dense-entry matrices were not sampled.
Neither the remaining failure pattern nor the counts show monotonic worsening
with algebra level. All 100 level-8 cases were eventually resolved while one
case remains unresolved at each of levels 5 and 7.

Original and retry vectors and histories remain in
`scripts/results/eigen-survey-600/` and
`scripts/results/eigen-survey-600-retry/`. These are ignored local run outputs;
the audit JSON files are compact paper evidence.
