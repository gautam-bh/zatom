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

# W&B auth: batch jobs have no TTY for `wandb login`, so a key must come from the
# environment or ~/.netrc; otherwise fall back to offline logging so wandb.init()
# doesn't kill the job (upload later with `wandb sync <run_dir>`).
# NOTE: configs/logger/wandb.yaml hardcodes `mode: "online"`, which overrides the
# WANDB_MODE env var — so the mode is also passed as a Hydra override below.
export WANDB_MODE=${WANDB_MODE:-online}
if [ -z "${WANDB_API_KEY:-}" ] && ! grep -qs "api.wandb.ai" "$HOME/.netrc"; then
    echo "WARNING: No W&B credentials found (WANDB_API_KEY unset and no api.wandb.ai entry in ~/.netrc)."
    echo "         Falling back to WANDB_MODE=offline. Run 'wandb login' once on a login node to fix."
    export WANDB_MODE=offline
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
RUN_NAME="lora_bandgap_arch-${ARCHITECTURE}_${RUN_DATE}"

CKPT_PATH="logs/$TASK_NAME/runs/${RUN_NAME}/checkpoints/"  # path at which to find/save checkpoints for this run (also used to resume after preemption)
mkdir -p "$CKPT_PATH"

# Inform user of job details
echo -e "Job details:\n========================================================================\n"

echo "Run name: $RUN_NAME"
echo "Run ID: $RUN_ID"
echo "Run start time: $RUN_DATE"
echo "Pretrained checkpoint: $PRETRAINED_CKPT_PATH"

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
bash -c "
    unset NCCL_CROSS_NIC \
    && HYDRA_FULL_ERROR=1 WANDB_RESUME=allow WANDB_RUN_ID=$RUN_ID WANDB_MODE=$WANDB_MODE PYTHONUNBUFFERED=1 TORCH_HOME=$TORCH_HOME HF_HOME=$HF_HOME WANDB_CACHE_DIR=$WANDB_CACHE_DIR WANDB_ARTIFACT_DIR=$WANDB_ARTIFACT_DIR \
    srun --kill-on-bad-exit=1 shifter python zatom/train_fm.py \
    experiment=lora_bandgap \
    pretrained_ckpt_path=$PRETRAINED_CKPT_PATH \
    ckpt_path=$CKPT_PATH \
    model/architecture=$ARCHITECTURE \
    date=$RUN_DATE \
    name=$RUN_NAME \
    task_name=$TASK_NAME \
    logger.wandb.mode=$WANDB_MODE \
    trainer.num_nodes=$SLURM_JOB_NUM_NODES \
    trainer.devices=$SLURM_NTASKS_PER_NODE \
    trainer.max_time='00:11:00:00'
"

# Inform user of run completion
echo "Run completed for SLURM job $SLURM_JOB_NAME with ID $SLURM_JOB_ID"
