#!/usr/bin/env bash
#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=20:00:00
#SBATCH --qos=normal
#SBATCH --output=log_output/training_%j.out
#SBATCH --error=log_error/training_%j.err

mkdir -p log_output log_error

EXPERIMENT_SCRIPT="/home/lgallian/superiorflows/particle_systems/training_particles.py"
export PYTHONPATH="$PWD:$PYTHONPATH"

echo
echo "Starting training job on $HOSTNAME"
echo "Executing: $EXPERIMENT_SCRIPT"
echo "Passed arguments: $@"

echo "Training started at: $(date)"

srun uv run python $EXPERIMENT_SCRIPT "$@"
    
echo "Training finished at: $(date)"
echo