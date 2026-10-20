#!/bin/bash -l

######################### Batch Headers #########################
# Memory-isolation diagnostic for the lora_bandgap OOM (jobs 56547853, 56548543).
#
# HISTORICAL: this diagnosis is concluded (B_eff = batch_size x multiplicity; the fix is
# batch_size=8 with accumulate_grad_batches=16, see configs/experiment/lora_bandgap.yaml).
# The analysis below is for the batch-32 runs. For a deterministic peak-memory measurement
# at the current settings use tests/predeploy/gpu/t2_1_peak_memory.py (T2-1) instead.
#
# Both prior smoke tests OOM'd on the FIRST training step with ~74-77 GiB of live tensors,
# and cutting the batch size 128 -> 32 barely moved that number. That insensitivity is the
# whole puzzle: if the training batch dominated memory, a 4x cut would have shown up.
#
# Two phases pin it down:
#   Phase 1 - training only (no sanity check, no validation), batch 32.
#   Phase 2 - training WITH validation, batch 8.
#
# Reading the result:
#   P1 pass, P2 fail -> validation leaves memory behind; the training batch is not the cause.
#   P1 pass, P2 pass -> batch size is the effective lever; pick the largest that fits.
#   P1 fail          -> the training step alone needs >79 GiB at batch 32 (pad length 444).
#SBATCH -C gpu&hbm80g
#SBATCH --qos=debug
#SBATCH --image=registry.nersc.gov/dasrepo/acmwhb/zatom:0.0.1
#SBATCH --module=gpu,nccl-plugin
#SBATCH --account=m5008
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4
#SBATCH --time=00:29:00
#SBATCH --job-name=lora-bandgap-memdiag
#SBATCH --output=/global/u1/g/gautama/projects/zatom/scripts/perlmutter/regular/logs/lora_bandgap_memdiag_%j.out
#SBATCH --error=/global/u1/g/gautama/projects/zatom/scripts/perlmutter/regular/logs/lora_bandgap_memdiag_%j.err

PROJECT_DIR="/global/u1/g/gautama/projects/zatom"
cd "$PROJECT_DIR" || exit

export TORCH_HOME="/pscratch/sd/${USER:0:1}/$USER/torch_cache"
export HF_HOME="/pscratch/sd/${USER:0:1}/$USER/hf_cache"
export WANDB_CACHE_DIR="/pscratch/sd/${USER:0:1}/$USER/wandb_cache"
export WANDB_ARTIFACT_DIR="/pscratch/sd/${USER:0:1}/$USER/wandb_artifacts"
mkdir -p "$TORCH_HOME" "$HF_HOME" "$WANDB_CACHE_DIR" "$WANDB_ARTIFACT_DIR"

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Diagnostic only -- keep W&B out of the picture entirely so a logging failure cannot be
# confused with a memory failure. Nothing here is a run worth recording.
export WANDB_MODE=disabled

PRETRAINED_CKPT_PATH=${1:?Usage: sbatch $0 <pretrained_ckpt_path>}
ARCHITECTURE="tft_160M"
TASK_NAME="finetune_fm"

# Sample GPU memory every 5 s so a PASSING phase still reports its peak -- a pass with no
# headroom is not a pass worth shipping to an 18-hour job.
MEMLOG="/tmp/memdiag_${SLURM_JOB_ID}.csv"
( while true; do
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | tr '\n' ' '
    echo ""
    sleep 5
  done ) > "$MEMLOG" 2>/dev/null &
SAMPLER_PID=$!
trap 'kill $SAMPLER_PID 2>/dev/null' EXIT

run_phase () {
    local phase_name="$1"; shift
    local run_date="$(date +%Y-%m-%d_%H-%M-%S)"
    local run_name="memdiag_${phase_name}_arch-${ARCHITECTURE}"
    local ckpt_dir="/pscratch/sd/${USER:0:1}/$USER/zatom_checkpoints/memdiag/${run_name}_${run_date}/checkpoints"
    mkdir -p "$ckpt_dir"

    echo ""
    echo "########################################################################"
    echo "### PHASE: $phase_name   started $(date)"
    echo "### overrides: $*"
    echo "########################################################################"

    bash -c "
        unset NCCL_CROSS_NIC \
        && HYDRA_FULL_ERROR=1 WANDB_MODE=disabled PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
           TORCH_HOME=$TORCH_HOME HF_HOME=$HF_HOME WANDB_CACHE_DIR=$WANDB_CACHE_DIR WANDB_ARTIFACT_DIR=$WANDB_ARTIFACT_DIR \
        srun --kill-on-bad-exit=1 shifter python zatom/train_fm.py \
        experiment=lora_bandgap \
        pretrained_ckpt_path=$PRETRAINED_CKPT_PATH \
        ckpt_path=$ckpt_dir \
        callbacks.model_checkpoint.dirpath=$ckpt_dir \
        callbacks.last_model_checkpoint.dirpath=$ckpt_dir \
        model/architecture=$ARCHITECTURE \
        date=$run_date \
        name=$run_name \
        task_name=$TASK_NAME \
        logger.wandb.mode=disabled \
        test=False \
        trainer.num_nodes=$SLURM_JOB_NUM_NODES \
        trainer.devices=$SLURM_NTASKS_PER_NODE \
        trainer.max_epochs=1 \
        $*
    "
    local rc=$?
    echo "### PHASE $phase_name EXIT CODE: $rc   finished $(date)"
    return $rc
}

# Phase 1: no validation at all. Isolates the training step's own memory.
run_phase "p1_trainonly_bs32" \
    "data.datamodule.batch_size.train=32 data.datamodule.batch_size.val=32 data.datamodule.batch_size.test=32 \
     trainer.accumulate_grad_batches=4 \
     ++trainer.num_sanity_val_steps=0 ++trainer.limit_val_batches=0 ++trainer.limit_train_batches=4 \
     ++trainer.check_val_every_n_epoch=1000"
P1_RC=$?

# Phase 2: validation enabled, much smaller batch. Isolates whether batch size is the lever.
run_phase "p2_withval_bs8" \
    "data.datamodule.batch_size.train=8 data.datamodule.batch_size.val=8 data.datamodule.batch_size.test=8 \
     trainer.accumulate_grad_batches=16 \
     ++trainer.num_sanity_val_steps=2 ++trainer.limit_val_batches=2 ++trainer.limit_train_batches=4"
P2_RC=$?

kill $SAMPLER_PID 2>/dev/null

echo ""
echo "########################################################################"
echo "### DIAGNOSTIC SUMMARY"
echo "###   Phase 1 (train only,  bs=32): exit $P1_RC"
echo "###   Phase 2 (with val,    bs=8 ): exit $P2_RC"
echo "### Peak GPU memory observed (MiB, sampled every 5 s):"
sort -t' ' -k2 -rn "$MEMLOG" 2>/dev/null | head -3
echo "### Full memory trace: $MEMLOG (node-local; copied below)"
echo "########################################################################"
cp "$MEMLOG" "$PROJECT_DIR/scripts/perlmutter/regular/logs/memdiag_${SLURM_JOB_ID}_gpumem.csv" 2>/dev/null
