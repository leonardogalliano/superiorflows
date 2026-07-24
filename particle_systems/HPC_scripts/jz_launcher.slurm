#!/usr/bin/env bash
#SBATCH --job-name=jz_job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=log_output/%x_%j.out
#SBATCH --error=log_error/%x_%j.err

if [ -z "$1" ]; then
    echo "Usage: sbatch [sbatch options] jz_launcher.slurm <job_type|script_path> [args...]"
    echo "Job types: train, sample, likelihood, patches"
    exit 1
fi

JOB_TYPE="$1"
shift

case "$JOB_TYPE" in
    train)
        SCRIPT_NAME="training_particles.py"
        ;;
    sample)
        SCRIPT_NAME="sampling_particles.py"
        ;;
    likelihood|eval|logprob)
        SCRIPT_NAME="evaluate_log_prob.py"
        ;;
    patches)
        SCRIPT_NAME="training_patches.py"
        ;;
    *)
        SCRIPT_NAME="$JOB_TYPE"
        ;;
esac

# Support both package-relative and absolute/current-directory script paths to ensure flexibility across different working directories.
if [[ "$SCRIPT_NAME" != /* ]]; then
    if [ -f "particle_systems/$SCRIPT_NAME" ]; then
        EXPERIMENT_SCRIPT="particle_systems/$SCRIPT_NAME"
    elif [ -f "$SCRIPT_NAME" ]; then
        EXPERIMENT_SCRIPT="$SCRIPT_NAME"
    else
        echo "Error: Script $SCRIPT_NAME not found in particle_systems/ or current directory."
        exit 1
    fi
else
    EXPERIMENT_SCRIPT="$SCRIPT_NAME"
fi

# Clean and dynamically load modules based on the exported GPU_TYPE
module purge
if [ "$GPU_TYPE" = "a100" ]; then
    module load arch/a100
elif [ "$GPU_TYPE" = "h100" ]; then
    module load arch/h100
fi
# V100 is the default architecture on Jean Zay and does not require an arch/ module
module load cuda/12.2.0

# Fix dynamic linking for JAX XLA
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
VENV_NVIDIA_LIBS=$(find $WORK/superiorflows/.venv/lib/python3.12/site-packages/nvidia -type d -name "lib" | tr '\n' ':')
export LD_LIBRARY_PATH="${VENV_NVIDIA_LIBS}${LD_LIBRARY_PATH}"

# Enforce cache routing as a safety net for the computation nodes
export UV_CACHE_DIR=$WORK/.uv_cache

# Setup workspace
cd $WORK/superiorflows || exit 1
mkdir -p log_output log_error

echo
echo "================================================="
echo "Starting job on $HOSTNAME"
echo "Allocated GPU Architecture: ${GPU_TYPE:-v100}"
echo "Executing: $EXPERIMENT_SCRIPT"
echo "Passed arguments: $@"
echo "Started at: $(date)"
echo "================================================="

/usr/bin/time -v srun uv run python "$EXPERIMENT_SCRIPT" "$@"

echo "================================================="
echo "Finished at: $(date)"
echo "================================================="
echo
