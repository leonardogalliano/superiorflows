#!/usr/bin/env bash
#SBATCH --job-name=sample
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=20:00:00
#SBATCH --qos=normal
#SBATCH --output=log_output/sampling_%j.out
#SBATCH --error=log_error/sampling_%j.err

mkdir -p log_output log_error

EXPERIMENT_SCRIPT="/home/lgallian/superiorflows/particle_systems/sampling_particles.py"
export PYTHONPATH="$PWD:$PYTHONPATH"

echo
echo "\nStarting sampling job on $HOSTNAME"
echo "Executing: $EXPERIMENT_SCRIPT"
echo "Passed arguments: $@"

echo "Sampling started at: $(date)"

srun uv run python $EXPERIMENT_SCRIPT "$@"
    
echo "Sampling finished at: $(date)\n"
echo