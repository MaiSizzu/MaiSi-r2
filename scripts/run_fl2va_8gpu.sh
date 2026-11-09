#!/bin/bash
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=dcu:8
#SBATCH --cpus-per-task=64
#SBATCH -t 04:00:00
#SBATCH -o run_fl2va_8gpu.%j.log
#
# FL2VA 端到端推理（8 卡版）。相对 run_fl2va.sh 的改动只有三处：
#   1. --gres=dcu:4 -> dcu:8
#   2. GPUS 默认 0..7
#   3. 增加 SMOKE=1 冒烟模式（小分辨率 + 少步数），用于先验证链路再烧卡时
#
# 冒烟：SMOKE=1 sbatch --export=ALL,PKG=$PWD,SS=$PWD/src scripts/run_fl2va_8gpu.sh
# 正式：sbatch --export=ALL,PKG=$PWD,SS=$PWD/src scripts/run_fl2va_8gpu.sh
# 回退到 4 卡：GPUS=0,1,2,3 且把 --gres 改成 dcu:4

set -o pipefail

PKG=${PKG:-$PWD}
SS=${SS:-$PKG}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}

source "$PKG/env/dcu_env.sh"
source "$PKG/env/paths.env"
source "${H3_VENV}/bin/activate"

export LD_LIBRARY_PATH=${INFINI_ROOT:-/work2/infini}/lib:$DTK_ROOT/cuda/cuda-12/lib64:$DTK_ROOT/lib:/opt/hyhal/lib:/opt/hyhal/lib64:$LD_LIBRARY_PATH
export LD_PRELOAD=$DTK_ROOT/cuda/cuda-12/targets/x86_64-linux/lib/libcuda.so.1
export PYTHONPATH=$SS/InfiniCore/python:$SS/InfiniLM/python

echo "[env] visible DCUs:"
python - <<'PY'
import torch
n = torch.cuda.device_count()
print("  device_count =", n)
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    print(f"  [{i}] {p.name}  {p.total_memory/1024**3:.1f} GiB")
PY

if [ ! -d "$MODEL_ROOT" ]; then echo "ERROR: MODEL_ROOT not found: $MODEL_ROOT"; exit 2; fi
if [ ! -f "$FIRST_FRAME" ]; then echo "ERROR: first frame not found: $FIRST_FRAME"; exit 3; fi
if [ ! -f "${INFINI_ROOT:-/work2/infini}/lib/libinfiniop.so" ]; then
  echo "ERROR: libinfiniop.so missing under ${INFINI_ROOT:-/work2/infini}/lib"; exit 4
fi
mkdir -p "$OUTPUT_DIR"

if [ "${SMOKE:-0}" = "1" ]; then
  NUM_FRAMES=8
  STEPS=2
  HW_ARGS="--height 256 --width 256"
  TAG=smoke
else
  HW_ARGS=""
  TAG="${NUM_FRAMES}f_${STEPS}s"
fi

cd "$SS/InfiniLM" || exit 1
python examples/minimax_h3.py fl2va \
  --model-path="$MODEL_ROOT" \
  --text="$PROMPT" \
  --first-frame="$FIRST_FRAME" \
  --gpus="$GPUS" \
  --num-frames="$NUM_FRAMES" \
  --steps="$STEPS" \
  $HW_ARGS \
  --output="$OUTPUT_DIR/fl2va_${TAG}.mp4"
echo "fl2va_exit=$?"
