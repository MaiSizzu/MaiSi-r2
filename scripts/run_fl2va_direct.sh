#!/bin/bash
# run_fl2va_direct.sh — 不依赖 Slurm 的 FL2VA 启动器（容器 / 平台终端 / 交互式节点通用）
#
# 冒烟（先跑这个，几十秒，用来确认链路）：
#   SMOKE=1 GPUS=0,1,2,3,4,5,6,7 bash scripts/run_fl2va_direct.sh
# 正式（8 卡，124 帧 50 步）：
#   GPUS=0,1,2,3,4,5,6,7 bash scripts/run_fl2va_direct.sh
# 回退（8 卡报错时，4 卡 TP 更稳）：
#   GPUS=0,1,2,3 bash scripts/run_fl2va_direct.sh
#
# 全部路径可用环境变量覆盖：PKG SS MODEL_ROOT FIRST_FRAME OUTPUT_DIR H3_VENV INFINI_ROOT PY

set -o pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PKG=${PKG:-$(cd "$HERE/.." && pwd)}
SS=${SS:-$PKG/src}
GPUS=${GPUS:-auto}     # auto = 用当前可见的全部 DCU（批作业分配几张就用几张）

# ---- DTK ----
if [ -z "$DTK_ROOT" ] || [ ! -d "$DTK_ROOT" ]; then
  for c in /public/software/compiler/dtk-* /opt/dtk* /opt/rocm*; do
    [ -d "$c" ] && DTK_ROOT=$c && break
  done
fi
if [ -n "$DTK_ROOT" ] && [ -f "$DTK_ROOT/env.sh" ]; then source "$DTK_ROOT/env.sh"; fi
export HIP_PATH=${HIP_PATH:-$DTK_ROOT}
export ROCM_PATH=${ROCM_PATH:-$DTK_ROOT}
export CUDA_HOME=${CUDA_HOME:-$DTK_ROOT/cuda/cuda-12}
export CUDA_PATH=$CUDA_HOME
export PATH=$CUDA_HOME/bin:$PATH

# ---- 包内路径配置（MODEL_ROOT / FIRST_FRAME / OUTPUT_DIR / PROMPT ...）----
[ -f "$PKG/env/paths.env" ] && source "$PKG/env/paths.env"

MID=${INFINI_ROOT:-/work2/infini}
export LD_LIBRARY_PATH=$MID/lib:$DTK_ROOT/cuda/cuda-12/lib64:$DTK_ROOT/lib:/opt/hyhal/lib:/opt/hyhal/lib64:$LD_LIBRARY_PATH
PRELOAD=$DTK_ROOT/cuda/cuda-12/targets/x86_64-linux/lib/libcuda.so.1
[ -f "$PRELOAD" ] && export LD_PRELOAD=$PRELOAD
export PYTHONPATH=$SS/InfiniCore/python:$SS/InfiniLM/python:$PYTHONPATH

# ---- Python ----
if [ -z "$PY" ]; then
  if [ -n "$H3_VENV" ] && [ -x "$H3_VENV/bin/python" ]; then PY=$H3_VENV/bin/python
  elif [ -x "$PKG/../venv/h3/bin/python" ]; then PY=$PKG/../venv/h3/bin/python
  elif command -v python3 >/dev/null 2>&1; then PY=python3
  else PY=python; fi
fi

# ---- 参数 ----
if [ "${SMOKE:-0}" = "1" ]; then
  FRAMES=8; STEPS=2; HW_ARGS="--height 256 --width 256"; TAG=smoke
else
  FRAMES=${NUM_FRAMES:-124}; STEPS=${STEPS:-50}; HW_ARGS=""; TAG="${FRAMES}f_${STEPS}s"
fi
# ---- 权重路径兜底：平台换了挂载点也能自动找到 ----
if [ ! -f "$MODEL_ROOT/model_index.json" ]; then
  for c in /work2/ai_data/models/MiniMax/MiniMax-H3/FL2VA \
           "$HOME/models/MiniMax-H3/FL2VA" "$PKG/../models/FL2VA" \
           "$PKG/models/FL2VA"; do
    if [ -f "$c/model_index.json" ]; then MODEL_ROOT=$c; break; fi
  done
fi
if [ ! -f "$FIRST_FRAME" ]; then
  for c in "$PKG/inputs/first_frame.jpg" "$PKG/inputs/first_frame.png" \
           "$HOME/first_frame.jpg" "$HOME/first_frame.png"; do
    if [ -f "$c" ]; then FIRST_FRAME=$c; break; fi
  done
fi

# ---- 输出目录兜底 ----
if ! mkdir -p "$OUTPUT_DIR" 2>/dev/null; then
  echo "WARN: 无法创建 $OUTPUT_DIR，改用 $PKG/outputs"
  OUTPUT_DIR=$PKG/outputs
  mkdir -p "$OUTPUT_DIR" || { echo "ERROR: 无法创建输出目录"; exit 5; }
fi
LOG=$OUTPUT_DIR/fl2va_${TAG}_$(date +%m%d_%H%M%S).log

echo "=============================================="
echo " PKG        = $PKG"
echo " MODEL_ROOT = $MODEL_ROOT"
echo " FIRST_FRAME= $FIRST_FRAME"
echo " OUTPUT_DIR = $OUTPUT_DIR"
echo " GPUS       = $GPUS"
echo " frames     = $FRAMES   steps = $STEPS   $HW_ARGS"
echo " PY         = $PY"
echo " LOG        = $LOG"
echo "=============================================="

# ---- 前置断言（不满足就直接退，不烧卡时）----
[ -d "$MODEL_ROOT" ] || { echo "ERROR: MODEL_ROOT 不存在: $MODEL_ROOT"; exit 2; }
[ -f "$FIRST_FRAME" ] || { echo "ERROR: 首帧不存在: $FIRST_FRAME"; exit 3; }
[ -f "$MID/lib/libinfiniop.so" ] || echo "WARN: $MID/lib/libinfiniop.so 不存在，若 import infinicore 失败请先构建"
[ -f "$SS/InfiniLM/examples/minimax_h3.py" ] || { echo "ERROR: 缺 InfiniLM 入口"; exit 4; }

echo "[env] 可见 DCU："
TMPN=$OUTPUT_DIR/.dcu_count.$$
"$PY" - "$TMPN" <<'PY'
import sys
try:
    import torch
except Exception as e:
    print(f"  [FAIL] import torch 失败: {type(e).__name__}: {e}")
    sys.exit(1)
n = torch.cuda.device_count()
print("  device_count =", n)
for i in range(n):
    p = torch.cuda.get_device_properties(i)
    print(f"  [{i}] {p.name}  {p.total_memory/1024**3:.1f} GiB")
open(sys.argv[1], "w").write(str(n))
PY
VIS=$(cat "$TMPN" 2>/dev/null); rm -f "$TMPN"; VIS=${VIS:-0}

# ---- GPUS=auto：跟随批作业实际分配的卡数（平台常常重编号，别写死 0-7）----
if [ "$GPUS" = "auto" ]; then
  if [ "$VIS" -gt 0 ]; then
    GPUS=$(seq -s, 0 $((VIS - 1)))
  else
    GPUS=0
  fi
  echo "[env] GPUS=auto → 使用可见的全部 $VIS 张卡：$GPUS"
fi
NGPU=$(echo "$GPUS" | awk -F, '{print NF}')
echo "[env] 张量并行度 = $NGPU"

# ---- 卡数硬校验：在登录节点跑会长跑 CPU，白烧时间 ----
if [ "$VIS" -lt "$NGPU" ]; then
  echo "ERROR: 只看到 $VIS 张 DCU，但 --gpus 要 $NGPU 张。"
  echo "       你现在多半在登录节点/没申请到卡。用批作业申请 8 卡，或 GPUS=0,1,2,3 降到 4 卡。"
  exit 6
fi

cd "$SS/InfiniLM" || exit 1
START=$(date +%s)
"$PY" examples/minimax_h3.py fl2va \
  --model-path="$MODEL_ROOT" \
  --text="$PROMPT" \
  --first-frame="$FIRST_FRAME" \
  --gpus="$GPUS" \
  --num-frames="$FRAMES" \
  --steps="$STEPS" \
  $HW_ARGS \
  --output="$OUTPUT_DIR/fl2va_${TAG}.mp4" 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
END=$(date +%s)

echo "----------------------------------------------"
echo "fl2va_exit=$RC   耗时 $((END-START)) 秒（约 $(( (END-START)*NGPU )) 卡秒）"
[ -f "$OUTPUT_DIR/fl2va_${TAG}.mp4" ] && ls -lh "$OUTPUT_DIR/fl2va_${TAG}.mp4"
echo "关键耗时（若日志里有 timings.print_report）："
grep -A3 "timings.print_report\|print_report" "$LOG" | tail -20
exit "$RC"
