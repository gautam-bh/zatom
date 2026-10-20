#!/bin/bash
# T2-3: resubmitting a run resumes it (preemption path) instead of restarting from scratch.
#
# Run AFTER a passing `smoke_lora_bandgap.sh` job (T2-2). This resubmits the smoke test with
# the same run id and date, so it writes to the same run dir and checkpoint dir, then checks
# the second job's log shows Lightning restoring a step checkpoint and training past it.
#
# Usage (repo root, Perlmutter login node):
#   bash tests/predeploy/gpu/t2_3_resume_smoke.sh <pretrained_ckpt> <first_smoke_job_id>
set -euo pipefail

PRETRAINED_CKPT_PATH=${1:?Usage: $0 <pretrained_ckpt> <first_smoke_job_id>}
FIRST_JOB=${2:?Usage: $0 <pretrained_ckpt> <first_smoke_job_id>}
LOG_DIR=scripts/perlmutter/regular/logs
FIRST_OUT="$LOG_DIR/lora_bandgap_smoke_${FIRST_JOB}.out"

RUN_ID=$(grep -m1 '^Run ID: ' "$FIRST_OUT" | cut -d' ' -f3)
RUN_DATE=$(grep -m1 '^Run date: ' "$FIRST_OUT" | cut -d' ' -f3)
[ -n "$RUN_ID" ] && [ -n "$RUN_DATE" ] || { echo "Could not read run id/date from $FIRST_OUT"; exit 1; }
echo "Resubmitting smoke run $RUN_ID ($RUN_DATE)"

SECOND_JOB=$(sbatch --parsable --wait scripts/perlmutter/regular/smoke_lora_bandgap.sh \
    "$PRETRAINED_CKPT_PATH" "$RUN_ID" "$RUN_DATE") || true
SECOND_ERR="$LOG_DIR/lora_bandgap_smoke_${SECOND_JOB}.err"
SECOND_OUT="$LOG_DIR/lora_bandgap_smoke_${SECOND_JOB}.out"

# Lightning prints this when `trainer.fit(ckpt_path=...)` restores a checkpoint
if ! grep -qa "Restored all states from the checkpoint at .*[0-9]-[0-9]*\.ckpt" "$SECOND_ERR" "$SECOND_OUT"; then
    echo "T2-3 FAIL: job $SECOND_JOB did not restore a step checkpoint (it trained from scratch?)"
    exit 1
fi
if grep -qa "No checkpoint files found in the given directory" "$SECOND_ERR" "$SECOND_OUT"; then
    echo "T2-3 FAIL: job $SECOND_JOB found no checkpoints in the run's directory"
    exit 1
fi
echo "T2-3 PASS: job $SECOND_JOB resumed $RUN_ID from its last step checkpoint"
