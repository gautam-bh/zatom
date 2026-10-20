#!/bin/bash -l

######################### Batch Headers #########################
#SBATCH -C gpu&hbm80g                                          # request higher-memory (80G) GPU nodes
#SBATCH --qos=regular                                          # standard (non-shared) full-node batch queue
#SBATCH --image=registry.nersc.gov/dasrepo/acmwhb/zatom:0.0.1  # use specified container image
#SBATCH --module=gpu,nccl-plugin                               # load GPU and optimized NCCL plugin modules
#SBATCH --account=m5008                                        # use specified account for billing (e.g., `m5008_g` for AI4Science proposal, `dasrepo` for all else)
#SBATCH --nodes=1                                              # NOTE: this needs to match Lightning's `Trainer(num_nodes=...)`
#SBATCH --gpus-per-node=4                                      # use all 4 A100 GPUs on the node
#SBATCH --ntasks-per-node=4                                    # NOTE: this needs to match Lightning's `Trainer(devices=...)`
#SBATCH --time=18:00:00                                        # wall-clock ceiling with slack; actual usage is capped at ~11 node-hours via `trainer.max_time` below
#SBATCH --job-name=lora-bandgap-tft160M
#SBATCH --output=scripts/perlmutter/regular/logs/lora_bandgap_tft160M_%j.out
#SBATCH --error=scripts/perlmutter/regular/logs/lora_bandgap_tft160M_%j.err

# Determine location of the project's directory
# PROJECT_ID="dasrepo"
# PROJECT_DIR="/global/cfs/cdirs/$PROJECT_ID/$USER/Repositories/zatom"            # long term storage community drive
# PROJECT_DIR="/pscratch/sd/${USER:0:1}/$USER/Repositories/zatom"                 # high-performance storage scratch drive with an 8-week purge policy
PROJECT_DIR="/global/u1/g/gautama/projects/zatom"                                 # existing repo checkout (no pscratch clone set up)
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

# Establish environment variables
export TORCH_HOME="/pscratch/sd/${USER:0:1}/$USER/torch_cache"                    # high-performance storage scratch drive with an 8-week purge policy
export HF_HOME="/pscratch/sd/${USER:0:1}/$USER/hf_cache"                          # high-performance storage scratch drive with an 8-week purge policy
export WANDB_CACHE_DIR="/pscratch/sd/${USER:0:1}/$USER/wandb_cache"               # high-performance storage scratch drive with an 8-week purge policy
export WANDB_ARTIFACT_DIR="/pscratch/sd/${USER:0:1}/$USER/wandb_artifacts"        # high-performance storage scratch drive with an 8-week purge policy

mkdir -p "$TORCH_HOME"
mkdir -p "$HF_HOME"
mkdir -p "$WANDB_CACHE_DIR"
mkdir -p "$WANDB_ARTIFACT_DIR"

export PYTHONUNBUFFERED=1  # flush stdout promptly so the progress bar appears in the SLURM .out log

# Reduce allocator fragmentation. The training step allocates in bursts right after the
# generation-heavy validation loop, which leaves the cache fragmented; without this the
# run can OOM on a 1-3 GiB allocation while several GiB sit reserved-but-unallocated.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# W&B auth: batch jobs have no TTY for `wandb login`, so a key must come from the
# environment or ~/.netrc.
# NOTE: configs/logger/wandb.yaml hardcodes `mode: "online"`, which overrides the
# WANDB_MODE env var — so the mode is also passed as a Hydra override below.
#
# Offline mode is NOT a safe fallback and must not be used as one: the validation loop
# logs a `wandb.Table`, and serializing it calls `_get_artifact_entry_ref_url()` ->
# `max_cli_version()`, which queries api.wandb.ai/graphql *regardless of WANDB_MODE*.
# Job 56395971 was already offline and still died on a 401 two minutes into training.
# So missing credentials are a hard failure here, not a downgrade.
export WANDB_MODE=${WANDB_MODE:-online}
if [ -z "${WANDB_API_KEY:-}" ] && ! grep -qs "api.wandb.ai" "$HOME/.netrc"; then
    echo "ERROR: No W&B credentials found (WANDB_API_KEY unset and no api.wandb.ai entry in ~/.netrc)."
    echo "       Run 'wandb login' once on a login node, then resubmit."
    echo "       (Offline mode is not a workaround — see the note above this check.)"
    exit 1
fi

# Pre-flight auth check: one authenticated round trip before the 11-hour run starts, so
# an expired/revoked key fails in ~15 s instead of after startup has already burned time.
if [ "$WANDB_MODE" != "offline" ]; then
    echo "Pre-flight: verifying W&B authentication from inside the container..."
    if ! timeout 180 shifter python -c "import wandb; print('W&B auth OK, user =', wandb.Api().viewer.username)"; then
        echo "ERROR: W&B pre-flight authentication check failed. Aborting before training starts."
        echo "       Re-run 'wandb login' on a login node and resubmit."
        exit 1
    fi
fi

# Define run details
DEFAULT_RUN_ID="$(tr -dc 'a-z0-9' </dev/urandom | head -c8)"  # NOTE: generate once and hardcode below if you need a stable ID across resubmissions
DEFAULT_RUN_DATE="$(date +%Y-%m-%d_%H-%M-%S)"      # NOTE: hardcode this if you need to resume a specific prior run
DEFAULT_ARCHITECTURE="tft_160M"                    # NOTE: must match the checkpoint's architecture; one of (`tft_80M`, `tft_160M`, `tft_300M`)

PRETRAINED_CKPT_PATH=${1:?Usage: sbatch $0 <pretrained_ckpt_path> [run_id] [run_date] [architecture]}  # path to the pretrained (non-LoRA) base checkpoint to fine-tune from
RUN_ID=${2:-$DEFAULT_RUN_ID}
RUN_DATE=${3:-$DEFAULT_RUN_DATE}
ARCHITECTURE=${4:-$DEFAULT_ARCHITECTURE}

TASK_NAME="finetune_fm"
# NOTE: do NOT append $RUN_DATE here. Hydra's run dir is `${name}_${date}` (see
# configs/hydra/default.yaml), so a dated `name` produced `..._<date>_<date>` run dirs
# while CKPT_PATH pointed at the undated `..._<date>` path — a directory Hydra never
# wrote to. Every resubmission therefore found it empty, logged "No checkpoint files
# found ... Using provided pretrained checkpoint path", and silently restarted training
# from scratch. That is what created the paired stray dirs under logs/finetune_fm/runs/.
RUN_NAME="lora_bandgap_arch-${ARCHITECTURE}"
RUN_DIR_NAME="${RUN_NAME}_${RUN_DATE}"  # must match Hydra's `${name}_${date}` run dir

# Checkpoints go to scratch, not $HOME. Each checkpoint is ~645 MB and
# `last_model_checkpoint` keeps every one it writes (save_top_k=-1, every 5000 steps),
# on top of `model_checkpoint`'s top-3 + last.ckpt. $HOME is at 24.7/40 GiB, so an
# 11-hour run would blow the quota mid-training; pscratch is at 0.3% of 20 TiB.
CKPT_PATH="/pscratch/sd/${USER:0:1}/$USER/zatom_checkpoints/$TASK_NAME/${RUN_DIR_NAME}/checkpoints"  # path at which to find/save checkpoints for this run (also used to resume after preemption)
mkdir -p "$CKPT_PATH"

# Inform user of job details
echo -e "Job details:\n========================================================================\n"

echo "Run name: $RUN_NAME"
echo "Run ID: $RUN_ID"
echo "Run start time: $RUN_DATE"
echo "Run directory: logs/$TASK_NAME/runs/$RUN_DIR_NAME"
echo "Checkpoint directory: $CKPT_PATH"
echo "Pretrained checkpoint: $PRETRAINED_CKPT_PATH"
echo "W&B mode: $WANDB_MODE"
# Code is bind-run from $PROJECT_DIR against a fixed image tag, so record both
echo "Code commit: $(git rev-parse HEAD)$(git diff --quiet HEAD || echo ' (with uncommitted changes)')"
echo "Container image: $(grep -m1 '^#SBATCH --image=' "$0" | cut -d= -f2 | awk '{print $1}')"

echo -e "\nSLURM job name: $SLURM_JOB_NAME"
echo "SLURM job ID: $SLURM_JOB_ID"
echo "SLURM master node: $SLURMD_NODENAME"
echo "SLURM all nodes: $SLURM_NODELIST"
echo "SLURM node count: $SLURM_JOB_NUM_NODES"

echo -e "\nCUDA visible devices: $CUDA_VISIBLE_DEVICES"

echo -e "\nCurrent time: $(date)"
echo "Current directory: $(pwd)"
echo "Current node: $(hostname)"

echo -e "\nExecuting script train_fm.py:\n========================================================================\n"

# Run script
# NOTE: `trainer.max_time` is set to ~11 hours (well under the `#SBATCH --time` ceiling above)
# so Lightning stops gracefully (saves a final checkpoint) instead of being hard-killed by SLURM.
#
# NOTE: the batch size / grad-accumulation overrides below are required, not cosmetic.
# `configs/experiment/lora_bandgap.yaml` overrides `/data: joint` but never overrides the
# batch size, so it inherited the joint *pretraining* default of 128 per rank. Job 56547853
# reached the first training step and immediately OOM'd (79 GiB A100, 73.8 GiB live). LoRA
# does not reduce activation memory -- gradients still flow through every frozen layer.
# 32 + accumulate_grad_batches=4 is the repo's own proven matbench recipe (see
# scripts/perlmutter/interactive/finetune_ddp_tft_matbench.sh), and the LR resolver already
# rescales by sqrt(world_size / base_world_size) for the 4-GPU topology used here.
#
# 32 was still not enough: `model.augmentations.multiplicity=8` replicates the batch inside
# `training_step`, so attention saw 32 x 8 = 256 graphs per A100 and OOM'd (job 56750303).
# 8 x 16 keeps the same effective batch of 4096 and the same 8 rotations per structure,
# while cutting what any single forward pass has to hold to 64 graphs. Verified by
# smoke_lora_bandgap.sh before this script is worth 18 hours.
bash -c "
    unset NCCL_CROSS_NIC \
    && HYDRA_FULL_ERROR=1 WANDB_RESUME=allow WANDB_RUN_ID=$RUN_ID WANDB_MODE=$WANDB_MODE PYTHONUNBUFFERED=1 TORCH_HOME=$TORCH_HOME HF_HOME=$HF_HOME WANDB_CACHE_DIR=$WANDB_CACHE_DIR WANDB_ARTIFACT_DIR=$WANDB_ARTIFACT_DIR \
    srun --kill-on-bad-exit=1 shifter python zatom/train_fm.py \
    experiment=lora_bandgap \
    data.datamodule.batch_size.train=8 \
    data.datamodule.batch_size.val=8 \
    data.datamodule.batch_size.test=8 \
    trainer.accumulate_grad_batches=16 \
    pretrained_ckpt_path=$PRETRAINED_CKPT_PATH \
    ckpt_path=$CKPT_PATH \
    callbacks.model_checkpoint.dirpath=$CKPT_PATH \
    callbacks.last_model_checkpoint.dirpath=$CKPT_PATH \
    model/architecture=$ARCHITECTURE \
    date=$RUN_DATE \
    name=$RUN_NAME \
    task_name=$TASK_NAME \
    logger.wandb.mode=$WANDB_MODE \
    trainer.num_nodes=$SLURM_JOB_NUM_NODES \
    trainer.devices=$SLURM_NTASKS_PER_NODE \
    trainer.max_time='00:11:00:00'
"
TRAIN_STATUS=$?

# pscratch has an 8-week purge policy. Copy the checkpoints worth keeping (model_checkpoint's
# top-k + last.ckpt, not every periodic step checkpoint) to CFS. Non-fatal: the run's
# result stands even if the copy fails, but say so loudly.
CFS_CKPT_DIR=${CFS_CKPT_DIR:-/global/cfs/cdirs/m5008/$USER/zatom_checkpoints/$TASK_NAME/$RUN_DIR_NAME}
if ls "$CKPT_PATH"/lora-bandgap-epoch@*.ckpt "$CKPT_PATH"/last.ckpt >/dev/null 2>&1; then
    if mkdir -p "$CFS_CKPT_DIR" && cp -p "$CKPT_PATH"/lora-bandgap-epoch@*.ckpt "$CKPT_PATH"/last.ckpt "$CFS_CKPT_DIR"/ 2>/dev/null; then
        echo "Copied best/last checkpoints to $CFS_CKPT_DIR"
    else
        echo "WARNING: could not copy checkpoints to $CFS_CKPT_DIR; they exist only on pscratch ($CKPT_PATH), which is purged after 8 weeks."
    fi
fi

# Inform user of run completion
echo "Run completed for SLURM job $SLURM_JOB_NAME with ID $SLURM_JOB_ID (training exit status $TRAIN_STATUS)"
# Without this the script's status is the last `echo`, so `sacct` reports COMPLETED for a
# run that crashed (the same masking smoke_lora_bandgap.sh already guards against).
exit "$TRAIN_STATUS"
