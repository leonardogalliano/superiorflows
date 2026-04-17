#!/usr/bin/env bash
#SBATCH --job-name=sample
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --time=20:00:00
#SBATCH --output=log_output/sampling_%j.out
#SBATCH --error=log_error/sampling_%j.err

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
echo "Starting sampling job on $HOSTNAME"
echo "Allocated GPU Architecture: $GPU_TYPE"
echo "Passed arguments: $@"
echo "Sampling started at: $(date)"
echo "================================================="

# Dispatch the job via the CLI entry point defined in pyproject.toml
/usr/bin/time -v srun uv run /lustre/fswork/projects/rech/czd/ulu16wv/superiorflows/particle_systems/sampling_particles.py "$@"

echo "================================================="
echo "Sampling finished at: $(date)"
echo "================================================="
echo