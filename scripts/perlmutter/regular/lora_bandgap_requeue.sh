#!/bin/bash
# Helpers that let `finetune_ddp_tft_lora_bandgap.sh` run under a preemptible QOS (overrun).
# Sourced by the launcher, not run. Tested by tests/predeploy/test_t0_preempt.py.
#
# Under overrun a job can be preempted after 2 hours, and with `--time-min` SLURM may grant
# less wall time than `--time`. Either way the run must continue in a later allocation
# from its last checkpoint, not restart from the pretrained weights:
#   - preemption: `#SBATCH --requeue` puts the same job id back in the queue;
#   - time limit: SLURM never requeues a timed-out job, so `run_with_timeout_requeue`
#     requeues it on the `--signal=B:USR1@<secs>` warning.
# A requeued job reruns the whole batch script, so `load_or_save_run_identity` restores the
# first start's run id and date; otherwise `RUN_DATE=$(date ...)` would point the rerun at a
# new, empty checkpoint dir.

# load_or_save_run_identity STATE_FILE
#   First start of a job: write RUN_ID, RUN_DATE and ARCHITECTURE to STATE_FILE.
#   Requeued start (same SLURM_JOB_ID, hence same STATE_FILE): read them back.
load_or_save_run_identity() {
    local state_file=$1
    if [ -f "$state_file" ]; then
        # shellcheck disable=SC1090
        source "$state_file"
        echo "Requeued start (restart count ${SLURM_RESTART_COUNT:-unknown}): reusing run identity from $state_file"
    else
        mkdir -p "$(dirname "$state_file")"
        printf 'RUN_ID=%q\nRUN_DATE=%q\nARCHITECTURE=%q\n' "$RUN_ID" "$RUN_DATE" "$ARCHITECTURE" >"$state_file"
    fi
}

# run_with_timeout_requeue CMD...
#   Run CMD in the background and wait for it, requeueing the job on SIGUSR1. Bash defers
#   traps while a foreground command runs, so CMD must not run in the foreground. Sets
#   TRAIN_STATUS to CMD's exit status.
run_with_timeout_requeue() {
    trap 'echo "Time limit near: requeueing SLURM job $SLURM_JOB_ID"; scontrol requeue "$SLURM_JOB_ID"' USR1
    "$@" &
    local pid=$!
    while true; do
        wait "$pid"
        TRAIN_STATUS=$?
        # `wait` also returns when the trap fires; keep waiting while CMD is still running
        kill -0 "$pid" 2>/dev/null || break
    done
    trap - USR1
}
