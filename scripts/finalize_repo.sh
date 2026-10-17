#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Alika Perdue
#
# ============================================================================
# scripts/finalize_repo.sh -- CayleyKernel
#
# One-shot finalisation of the repository layout and its first git commit.
# Review this whole file before running it.  It performs four stages:
#
#   Stage 1  Housekeeping:
#            - delete leftover build/cache/egg-info directories
#            - remove the local scratch directory .trash/
#            - chmod +x on shell scripts
#            - (skip with KEEP_TRASH=1 to keep .trash/)
#
#   Stage 2  Fresh git history:
#            - remove any existing .git/ (there are no commits to preserve)
#            - git init -b main, add everything, create the first commit
#
#   Stage 3  GitHub remote (SSH):
#            - add remote origin git@github.com:alikaperdue/CayleyKernel.git
#            - push (only when PUSH=1)
#
#   Stage 4  Verify the commit contents.
#
# Usage:
#   bash scripts/finalize_repo.sh                # everything except the push
#   PUSH=1 bash scripts/finalize_repo.sh         # also push to GitHub
#   KEEP_TRASH=1 bash scripts/finalize_repo.sh   # keep .trash/ on disk
# ============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GIT_USER_NAME="${GIT_USER_NAME:-Alika Perdue}"
GIT_USER_EMAIL="${GIT_USER_EMAIL:-alikaperdue@gmail.com}"
REMOTE_URL="git@github.com:alikaperdue/CayleyKernel.git"

echo "==> CayleyKernel repository finalisation"
echo "    working directory: $ROOT"

# ----------------------------------------------------------------------------
# Stage 1: housekeeping
# ----------------------------------------------------------------------------
echo ""
echo "==> [1/4] Housekeeping"

rm -rf .pytest_cache
rm -rf cuda_kernel.egg-info CayleyKernel.egg-info
find . -path ./.trash -prune -o -type d -name __pycache__ -print -exec rm -rf {} + 2>/dev/null || true

if [[ "${KEEP_TRASH:-0}" == "1" ]]; then
    echo "    KEEP_TRASH=1 -> leaving .trash/ on disk (still git-ignored)"
else
    rm -rf .trash
    echo "    removed .trash/ (old notebooks, .bak files, scratch scripts)"
fi

chmod +x scripts/*.sh || true
echo "    housekeeping done"

# ----------------------------------------------------------------------------
# Stage 2: fresh git history
# ----------------------------------------------------------------------------
echo ""
echo "==> [2/4] Fresh git history"

if [[ -d .git ]]; then
    if git rev-parse HEAD >/dev/null 2>&1; then
        echo "ERROR: .git/ already contains commits." >&2
        echo "       This script expects an empty repository (git init only)." >&2
        echo "       Aborting so nothing is lost." >&2
        exit 1
    fi
    rm -rf .git
    echo "    removed empty .git/"
fi

git init -b main
git config user.name  "$GIT_USER_NAME"
git config user.email "$GIT_USER_EMAIL"

# Safety check: nothing staged should be a binary or ignored artifact.
git add -A
if git ls-files | grep -E '\.(so|whl)$|__pycache__|\.bak$' ; then
    echo "ERROR: unexpected binary/cache files were staged (see above)." >&2
    exit 1
fi

git commit -m "CayleyKernel: first release of O(1)-order GPU Cayley-Dickson doubling products

Initial public commit: branch-free closed-form basis sign kernel for
arbitrary 2^n-dimensional Cayley-Dickson algebras, batched in a single
launch via grid-stride loop.

  * cuda/cuda_kernel.cu     NVIDIA CUDA kernel (O(1) per output component)
  * cuda/amd_kernel.cu      AMD HIP/ROCm port
  * CayleyKernel            Python wrapper (CuPy/ctypes), batch + broadcast
  * tests/                  recursive reference, algebraic identities,
                            zero divisors, slow & benchmark suites
  * Makefile, scripts/      kernel build, wheel build, repo tooling
  * Apache-2.0"

echo "    first commit created"

# ----------------------------------------------------------------------------
# Stage 3: GitHub remote (SSH)
# ----------------------------------------------------------------------------
echo ""
echo "==> [3/4] GitHub remote"

git remote remove origin 2>/dev/null || true
git remote add origin "$REMOTE_URL"
echo "    remote origin -> $REMOTE_URL"

# Quick SSH reachability probe (does not modify anything).
if ssh -T -o BatchMode=yes -o ConnectTimeout=5 git@github.com 2>&1 | grep -q "successfully authenticated"; then
    echo "    SSH auth to GitHub: OK"
else
    echo "    NOTE: SSH auth to GitHub did not succeed yet."
    echo "          Check your key with:  ssh -T git@github.com"
    echo "          (see README / chat notes for key setup)."
fi

if [[ "${PUSH:-0}" == "1" ]]; then
    echo "    pushing to origin main ..."
    git push -u origin main
else
    echo "    PUSH not set; skipping push.  Run:  PUSH=1 bash scripts/finalize_repo.sh"
fi

# ----------------------------------------------------------------------------
# Stage 4: verify
# ----------------------------------------------------------------------------
echo ""
echo "==> [4/4] Verification"
echo ""
echo "--- git status ---"
git status --short --branch
echo ""
echo "--- tracked files ---"
git ls-files
echo ""
echo "--- commit ---"
git log --stat -1
echo ""
echo "==> Done.  Next: create the empty repo on GitHub (alikaperdue/CayleyKernel,"
echo "    public), then:  PUSH=1 bash scripts/finalize_repo.sh"
