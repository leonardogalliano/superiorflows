#!/usr/bin/env bash

# Require at least two arguments: the SLURM script path and the GPU type
if [ "$#" -lt 2 ]; then
    echo "Usage: ./submit.sh <path_to_slurm_script> <v100|a100|h100> [additional arguments for the script]"
    exit 1
fi

SCRIPT_PATH="$1"
GPU_TYPE=$(echo "$2" | tr '[:upper:]' '[:lower:]')
shift 2 # Shift twice so "$@" retains only the subsequent script arguments

# Fail fast if the provided script path is invalid
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: The target script '$SCRIPT_PATH' does not exist or is not a regular file."
    exit 1
fi

# Route the SLURM parameters based on the requested hardware
if [ "$GPU_TYPE" = "v100" ]; then
    ACCOUNT="czd@v100"
    CONSTRAINT="v100-32g"
    QOS="qos_gpu-t3"
elif [ "$GPU_TYPE" = "a100" ]; then
    ACCOUNT="czd@a100"
    CONSTRAINT="a100"
    QOS="qos_gpu_a100-t3"
elif [ "$GPU_TYPE" = "h100" ]; then
    ACCOUNT="czd@h100"
    CONSTRAINT="h100"
    QOS="qos_gpu_h100-t3"
else
    echo "Error: Unknown GPU architecture '$GPU_TYPE'. Please use v100, a100, or h100."
    exit 1
fi

echo "Submitting $SCRIPT_PATH to the $GPU_TYPE partition..."

# Submit the specific payload script, dynamically overriding the SBATCH parameters
sbatch \
    --account=$ACCOUNT \
    --constraint=$CONSTRAINT \
    --qos=$QOS \
    --export=ALL,GPU_TYPE=$GPU_TYPE \
    "$SCRIPT_PATH" "$@"