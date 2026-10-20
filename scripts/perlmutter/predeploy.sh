#!/bin/bash
# Run the LoRA band gap pre-deploy suite (T0 + T1 in tests/predeploy) and, if it passes on a
# clean checkout, stamp the commit in `.predeploy_ok`. `finetune_ddp_tft_lora_bandgap.sh` and
# `smoke_lora_bandgap.sh` refuse to start unless that stamp matches the commit they run.
# See docs/plans/2026-10-19-lora-bandgap-predeploy-tests.md.
#
# Usage (repo root; Perlmutter login node or any machine with the `zatom` env):
#   bash scripts/perlmutter/predeploy.sh                      # T0 + T1, ~10 min
#   PRETRAINED_CKPT_PATH=/path/to/base.ckpt bash scripts/perlmutter/predeploy.sh
#                                                             # also T0-7 / T1-1 on the real checkpoint
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

ZIMG=registry.nersc.gov/dasrepo/acmwhb/zatom:0.0.1
if command -v shifter >/dev/null 2>&1; then
    # Same image as the jobs, so library versions match exactly
    PYTHON=(shifter --image="$ZIMG" --env=PROJECT_ROOT="$PWD" python)
else
    PYTHON=(python)
fi

"${PYTHON[@]}" -m pytest tests/predeploy -x -q -p no:cacheprovider -o addopts="" "$@"

if ! git diff --quiet HEAD; then
    echo "Tests passed, but the working tree has uncommitted changes; not stamping."
    echo "Commit, then rerun so the stamp names the code that was actually tested."
    exit 1
fi
git rev-parse HEAD > .predeploy_ok
echo "Pre-deploy suite passed on $(cat .predeploy_ok); launchers may now be submitted."
