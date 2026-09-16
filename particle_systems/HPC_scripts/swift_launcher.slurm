#!/usr/bin/env bash
#SBATCH --job-name=swift_job
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=20:00:00
#SBATCH --qos=normal
#SBATCH --output=log_output/%x_%j.out
#SBATCH --error=log_error/%x_%j.err

mkdir -p log_output log_error

if [ -z "$1" ]; then
    echo "Usage: sbatch swift_launcher.slurm <job_type|script_path> [args...]"
    echo "Job types: train, sample, likelihood, patches, sample_patches"
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
    sample_patches|sampling_patches|sample-patches)
        SCRIPT_NAME="sampling_patches.py"
        ;;
    *)
        SCRIPT_NAME="$JOB_TYPE"
        ;;
esac

# Support both package-relative and current-directory paths to ensure compatibility with different invocation styles.
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

export PYTHONPATH="$PWD:$PYTHONPATH"

echo
echo "Starting job on $HOSTNAME"
echo "Executing: $EXPERIMENT_SCRIPT"
echo "Passed arguments: $@"
echo "Started at: $(date)"
echo

srun uv run python "$EXPERIMENT_SCRIPT" "$@"
    
echo
echo "Finished at: $(date)"
echo
