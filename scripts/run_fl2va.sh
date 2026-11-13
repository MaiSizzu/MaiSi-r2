#!/bin/bash
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=dcu:4
#SBATCH --cpus-per-task=32
#SBATCH -t 08:00:00
#SBATCH -o run_fl2va.log
#
# FL2VA end-to-end inference: first frame + text prompt -> video.
# The official baseline uses 4 GPUs. A single DCU cannot hold the transformer
# (cudaMalloc returns Code 2), and a 1-GPU quota makes SLURM reject the job
# (request 4 exceeds group max tres limit 1).
# Output spec: 768 x 928, 124 frames.

set -o pipefail

PKG=${PKG:-$PWD}
SS=${SS:-$PKG}

source "$PKG/env/dcu_env.sh"
source "$PKG/env/paths.env"
source "${H3_VENV}/bin/activate"

export LD_LIBRARY_PATH=${INFINI_ROOT:-$HOME/.infini}/lib:$DTK_ROOT/cuda/cuda-12/lib64:$DTK_ROOT/lib:/opt/hyhal/lib:/opt/hyhal/lib64:$LD_LIBRARY_PATH
export LD_PRELOAD=$DTK_ROOT/cuda/cuda-12/targets/x86_64-linux/lib/libcuda.so.1
export PYTHONPATH=$SS/InfiniCore/python:$SS/InfiniLM/python

if [ ! -d "$MODEL_ROOT" ]; then echo "ERROR: MODEL_ROOT not found: $MODEL_ROOT"; exit 2; fi
if [ ! -f "$FIRST_FRAME" ]; then echo "ERROR: first frame not found: $FIRST_FRAME"; exit 3; fi
mkdir -p "$OUTPUT_DIR"

cd "$SS/InfiniLM" || exit 1
python examples/minimax_h3.py fl2va --model-path="$MODEL_ROOT" --text="$PROMPT" --first-frame="$FIRST_FRAME" --gpus="$GPUS" --num-frames="$NUM_FRAMES" --steps="$STEPS" --output="$OUTPUT_DIR/fl2va_${NUM_FRAMES}f_${STEPS}s.mp4"
echo "fl2va_exit=$?"
