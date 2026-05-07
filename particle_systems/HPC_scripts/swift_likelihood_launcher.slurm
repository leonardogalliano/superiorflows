#!/usr/bin/env bash
#SBATCH --job-name=logprob
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=20:00:00
#SBATCH --qos=normal
#SBATCH --output=log_output/likelihood_%j.out
#SBATCH --error=log_error/likelihood_%j.err

mkdir -p log_output log_error

EXPERIMENT_SCRIPT="/home/lgallian/superiorflows/particle_systems/evaluate_log_prob.py"
export PYTHONPATH="$PWD:$PYTHONPATH"

echo
echo "Starting likelihood evaluation job on $HOSTNAME"
echo "Executing: $EXPERIMENT_SCRIPT"
echo "Passed arguments: $@"

echo "Likelihood evaluation started at: $(date)"

srun uv run python $EXPERIMENT_SCRIPT "$@"

echo "Likelihood evaluation finished at: $(date)"
echo
