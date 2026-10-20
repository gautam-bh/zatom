#!/bin/bash -l

######################### Batch Headers #########################
# Short end-to-end smoke test of `finetune_ddp_tft_lora_bandgap.sh`. Identical container,
# DDP topology, config and code path as the real run -- only the number of batches is cut
# down. Purpose: prove the things that can only fail at runtime (W&B online auth inside
# the training loop, `wandb.Table` serialization, the `model_checkpoint` monitor metric
# actually existing, checkpoints landing in the scratch dir with resume-compatible names,
# and `trainer.test()` finding `best_model_path`) before committing to an 18-hour job.
#
# NOTE: `limit_train_batches=64` with `accumulate_grad_batches=16` gives 4 optimizer steps,
# so `last_model_checkpoint.every_n_train_steps=2` (global steps, not batches) actually fires.
# Both numbers moved together when the batch size dropped 32 -> 8; leaving
# `limit_train_batches` at 16 would have yielded a single optimizer step and silently
# stopped exercising the periodic-checkpoint path this test is here to cover.
#
# HISTORY -- three runs of the previous version of this script (56547853, 56548543,
# 56750303) all died with `torch.OutOfMemoryError` in `training_step` before reaching a
# single one of the checks above, and reported only "FAIL -- training command exited 1".
# Two changes came out of that:
#   1. The cause is fixed at the source: attention sees `batch_size x multiplicity` graphs,
#      which was 32 x 8 = 256 per A100. `batch_size` drops to 8 here and in
#      `finetune_ddp_tft_lora_bandgap.sh`, holding `multiplicity` at 8 so the run keeps all
#      8 rotations per structure. See `configs/experiment/lora_bandgap.yaml` for the
#      arithmetic and for why batch size, not multiplicity, is the right lever.
#   2. This script no longer reports memory failures as generic failures. It samples GPU
#      memory throughout, classifies the exit, and prints peak usage even on a PASS --
#      a pass with no headroom is not a pass worth promoting to an 18-hour job.
#SBATCH -C gpu&hbm80g
#SBATCH --qos=debug
#SBATCH --image=registry.nersc.gov/dasrepo/acmwhb/zatom:0.0.1
#SBATCH --module=gpu,nccl-plugin
#SBATCH --account=m5008
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=00:30:00
#SBATCH --job-name=lora-bandgap-smoke
#SBATCH --output=scripts/perlmutter/regular/logs/lora_bandgap_smoke_%j.out
#SBATCH --error=scripts/perlmutter/regular/logs/lora_bandgap_smoke_%j.err

PROJECT_DIR="/global/u1/g/gautama/projects/zatom"
cd "$PROJECT_DIR" || exit

# Pre-deploy gate: refuse to start unless `scripts/perlmutter/predeploy.sh` passed on this
# exact commit. Every failed job so far (see docs/plans/2026-10-19-lora-bandgap-predeploy-tests.md)
# would have been caught by that suite in minutes instead of after a queue wait.
# Set SKIP_PREDEPLOY=1 to override.
if [ "${SKIP_PREDEPLOY:-0}" != "1" ] && [ "$(cat .predeploy_ok 2>/dev/null)" != "$(git rev-parse HEAD)" ]; then
    echo "ERROR: the pre-deploy suite has not passed on commit $(git rev-parse --short HEAD)."
    echo "       Run 'bash scripts/perlmutter/predeploy.sh' on a login node, then resubmit."
    exit 1
fi

export TORCH_HOME="/pscratch/sd/${USER:0:1}/$USER/torch_cache"
export HF_HOME="/pscratch/sd/${USER:0:1}/$USER/hf_cache"
export WANDB_CACHE_DIR="/pscratch/sd/${USER:0:1}/$USER/wandb_cache"
export WANDB_ARTIFACT_DIR="/pscratch/sd/${USER:0:1}/$USER/wandb_artifacts"

mkdir -p "$TORCH_HOME" "$HF_HOME" "$WANDB_CACHE_DIR" "$WANDB_ARTIFACT_DIR"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export WANDB_MODE=${WANDB_MODE:-online}
if [ -z "${WANDB_API_KEY:-}" ] && ! grep -qs "api.wandb.ai" "$HOME/.netrc"; then
    echo "ERROR: No W&B credentials found (WANDB_API_KEY unset and no api.wandb.ai entry in ~/.netrc)."
    exit 1
fi

if [ "$WANDB_MODE" != "offline" ]; then
    echo "Pre-flight: verifying W&B authentication from inside the container..."
    if ! timeout 180 shifter python -c "import wandb; print('W&B auth OK, user =', wandb.Api().viewer.username)"; then
        echo "ERROR: W&B pre-flight authentication check failed. Aborting."
        exit 1
    fi
fi

DEFAULT_RUN_ID="smoke$(tr -dc 'a-z0-9' </dev/urandom | head -c6)"
DEFAULT_RUN_DATE="$(date +%Y-%m-%d_%H-%M-%S)"
DEFAULT_ARCHITECTURE="tft_160M"

PRETRAINED_CKPT_PATH=${1:?Usage: sbatch $0 <pretrained_ckpt_path> [run_id] [run_date] [architecture]}
RUN_ID=${2:-$DEFAULT_RUN_ID}
RUN_DATE=${3:-$DEFAULT_RUN_DATE}
ARCHITECTURE=${4:-$DEFAULT_ARCHITECTURE}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-8}
ACCUM_GRAD_BATCHES=${ACCUM_GRAD_BATCHES:-16}
# Keep 4 optimizer steps regardless of how the two above are tuned, so the
# `every_n_train_steps=2` checkpoint path stays covered.
LIMIT_TRAIN_BATCHES=$((ACCUM_GRAD_BATCHES * 4))

TASK_NAME="finetune_fm"
RUN_NAME="lora_bandgap_smoke_arch-${ARCHITECTURE}"
RUN_DIR_NAME="${RUN_NAME}_${RUN_DATE}"

CKPT_PATH="/pscratch/sd/${USER:0:1}/$USER/zatom_checkpoints/$TASK_NAME/${RUN_DIR_NAME}/checkpoints"
mkdir -p "$CKPT_PATH"

# `model_checkpoint.monitor` in `configs/callbacks/lora_bandgap.yaml`. If the val loop never
# emits this key, the real run's best-checkpoint selection silently never fires and
# `trainer.test()` has no `best_model_path` to load -- 18 hours for nothing. That is the
# single most important thing this smoke test exists to prove, so assert on it by name.
MONITOR_METRIC="val_matbench/loss"

OUT_LOG="scripts/perlmutter/regular/logs/lora_bandgap_smoke_${SLURM_JOB_ID}.out"
ERR_LOG="scripts/perlmutter/regular/logs/lora_bandgap_smoke_${SLURM_JOB_ID}.err"

echo -e "Smoke test details:\n========================================================================\n"
echo "Run name: $RUN_NAME"
echo "Run ID: $RUN_ID"
echo "Run date: $RUN_DATE"
echo "Run directory: logs/$TASK_NAME/runs/$RUN_DIR_NAME"
echo "Checkpoint directory: $CKPT_PATH"
echo "Pretrained checkpoint: $PRETRAINED_CKPT_PATH"
echo "W&B mode: $WANDB_MODE"
echo "Train batch size: $TRAIN_BATCH_SIZE per GPU x multiplicity 8 = $((TRAIN_BATCH_SIZE * 8)) graphs through attention"
echo "Accumulation: $ACCUM_GRAD_BATCHES batches -> effective batch $((TRAIN_BATCH_SIZE * 8 * 4 * ACCUM_GRAD_BATCHES)); $LIMIT_TRAIN_BATCHES train batches = 4 optimizer steps"
echo "Monitored metric: $MONITOR_METRIC"
echo "SLURM job ID: $SLURM_JOB_ID  node: $SLURMD_NODENAME"
echo "Current time: $(date)"

# Sample GPU memory every 5 s so a PASS still reports its peak. Three prior runs OOM'd and
# left no record of how close to the ceiling anything was.
MEMLOG="/tmp/smoke_${SLURM_JOB_ID}_gpumem.csv"
( while true; do
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | tr '\n' ' '
    echo ""
    sleep 5
  done ) > "$MEMLOG" 2>/dev/null &
SAMPLER_PID=$!
trap 'kill $SAMPLER_PID 2>/dev/null' EXIT

echo -e "\nExecuting script train_fm.py:\n========================================================================\n"

bash -c "
    unset NCCL_CROSS_NIC \
    && HYDRA_FULL_ERROR=1 WANDB_RESUME=allow WANDB_RUN_ID=$RUN_ID WANDB_MODE=$WANDB_MODE PYTHONUNBUFFERED=1 TORCH_HOME=$TORCH_HOME HF_HOME=$HF_HOME WANDB_CACHE_DIR=$WANDB_CACHE_DIR WANDB_ARTIFACT_DIR=$WANDB_ARTIFACT_DIR \
    srun --kill-on-bad-exit=1 shifter python zatom/train_fm.py \
    experiment=lora_bandgap \
    data.datamodule.batch_size.train=$TRAIN_BATCH_SIZE \
    data.datamodule.batch_size.val=$TRAIN_BATCH_SIZE \
    data.datamodule.batch_size.test=$TRAIN_BATCH_SIZE \
    trainer.accumulate_grad_batches=$ACCUM_GRAD_BATCHES \
    pretrained_ckpt_path=$PRETRAINED_CKPT_PATH \
    ckpt_path=$CKPT_PATH \
    callbacks.model_checkpoint.dirpath=$CKPT_PATH \
    callbacks.last_model_checkpoint.dirpath=$CKPT_PATH \
    callbacks.last_model_checkpoint.every_n_train_steps=2 \
    model/architecture=$ARCHITECTURE \
    date=$RUN_DATE \
    name=$RUN_NAME \
    task_name=$TASK_NAME \
    logger.wandb.mode=$WANDB_MODE \
    trainer.num_nodes=$SLURM_JOB_NUM_NODES \
    trainer.devices=$SLURM_NTASKS_PER_NODE \
    trainer.max_epochs=1 \
    trainer.max_time='00:00:18:00' \
    ++trainer.limit_train_batches=$LIMIT_TRAIN_BATCHES \
    ++trainer.limit_val_batches=2 \
    ++trainer.limit_test_batches=2
"
TRAIN_STATUS=$?

kill $SAMPLER_PID 2>/dev/null

echo -e "\n========================================================================"
echo "Smoke test finished for SLURM job $SLURM_JOB_ID at $(date)"

# Peak memory is reported on every outcome, not just failures: a pass with no headroom is
# not a pass worth promoting to an 18-hour job. Every batch is padded to the dataset's
# largest structure (444 atoms), so this peak is the run's steady-state peak.
PEAK_LINE="$(sort -t' ' -k2 -rn "$MEMLOG" 2>/dev/null | head -1)"
echo "Peak GPU memory observed (index used_MiB pairs, sampled every 5 s): ${PEAK_LINE:-<no samples>}"
cp "$MEMLOG" "$PROJECT_DIR/scripts/perlmutter/regular/logs/smoke_${SLURM_JOB_ID}_gpumem.csv" 2>/dev/null

echo "Checkpoints written:"
ls -la "$CKPT_PATH"

# The training command's exit status must be the script's exit status. Without this the
# trailing `ls` above becomes the batch script's result, srun failures are masked, and
# `sacct` reports COMPLETED for a run that OOM'd on the first batch (see job 56548543).
if [ "$TRAIN_STATUS" -ne 0 ]; then
    # Classify the failure. Three consecutive runs failed for one reason (OOM) while
    # reporting the same opaque message as any other failure would, which is why the cause
    # took three debug-queue jobs and a separate diagnostic script to pin down.
    if grep -qa "torch.OutOfMemoryError\|CUDA out of memory" "$ERR_LOG" "$OUT_LOG" 2>/dev/null; then
        echo "RESULT: FAIL (memory) -- training OOM'd; the plumbing this test exists to check was never reached."
        echo "  Requested block at failure:"
        grep -ao "Tried to allocate [0-9.]* [GM]iB" "$ERR_LOG" 2>/dev/null | sort -u | sed 's/^/    /'
        echo "  Attention is [B_eff, heads, L, L] fp32 and B_eff = batch_size x augmentations.multiplicity,"
        echo "  so the two factors are interchangeable for memory. Preferred lever: halve TRAIN_BATCH_SIZE"
        echo "  (currently $TRAIN_BATCH_SIZE) and double ACCUM_GRAD_BATCHES (currently $ACCUM_GRAD_BATCHES) to hold the"
        echo "  effective batch fixed while keeping all 8 rotations per structure. L is fixed at the"
        echo "  dataset's largest structure (every batch is padded to it), so B_eff is the only lever."
        echo "  Only drop model.augmentations.multiplicity (currently 8) if batch size is already at 1;"
        echo "  it trades away SO(3) gradient-variance reduction, which is the reason the setting exists."
        exit "$TRAIN_STATUS"
    fi
    echo "RESULT: FAIL -- training command exited $TRAIN_STATUS (not a memory failure; check $ERR_LOG)"
    exit "$TRAIN_STATUS"
fi

# A smoke test that produced no checkpoints has not proven what it exists to prove:
# checkpoints landing in the scratch dir with resume-compatible names.
if [ -z "$(ls -A "$CKPT_PATH" 2>/dev/null)" ]; then
    echo "RESULT: FAIL -- training exited 0 but no checkpoints were written to $CKPT_PATH"
    exit 1
fi

# Exit 0 plus a checkpoint on disk is still not proof that `model_checkpoint` selected it on
# the monitored metric -- `last_model_checkpoint` writes on a step count and would satisfy
# the check above on its own. Assert the monitor key was actually emitted by the val loop.
if ! grep -qa "$MONITOR_METRIC" "$OUT_LOG" "$ERR_LOG" 2>/dev/null; then
    echo "RESULT: FAIL -- '$MONITOR_METRIC' never appeared in the logs, so"
    echo "  callbacks.model_checkpoint.monitor will never fire during the real run and"
    echo "  trainer.test() will have no best_model_path to load."
    exit 1
fi

# The monitor-metric checkpoint is the one the real run resumes and tests from, so require
# a file matching its filename pattern, not merely any checkpoint.
if ! ls "$CKPT_PATH"/lora-bandgap-epoch@*.ckpt >/dev/null 2>&1; then
    echo "RESULT: FAIL -- no 'lora-bandgap-epoch@*.ckpt' in $CKPT_PATH; model_checkpoint did not write."
    echo "  Contents were:"
    ls -A "$CKPT_PATH" | sed 's/^/    /'
    exit 1
fi

echo "RESULT: PASS"
echo "  - training completed $LIMIT_TRAIN_BATCHES batches / 4 optimizer steps without OOM"
echo "  - '$MONITOR_METRIC' was emitted by the validation loop"
echo "  - model_checkpoint wrote lora-bandgap-epoch@*.ckpt to scratch"
echo "  - trainer.test() ran to completion"
