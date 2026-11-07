#!/bin/bash
# h3_8card.sh — 8 卡 FL2VA 一键编排（自检 / 冒烟 / 正式 / 性能落表）
#
# 用法（在包根目录 /work2/songshan/MaiSi-r2 下执行）：
#   bash scripts/h3_8card.sh check      # 只读自检，几秒，不烧卡
#   bash scripts/h3_8card.sh smoke      # 8 卡冒烟：256x256 / 8 帧 / 2 步
#   bash scripts/h3_8card.sh full       # 8 卡正式：124 帧 / 50 步
#   bash scripts/h3_8card.sh perf       # 正式跑 + 把耗时写进 perf_8card.csv
#   bash scripts/h3_8card.sh fallback   # 8 卡报错时改 4 卡冒烟（官方基线）
#
# 可选覆盖：GPUS=0,1,2,3 MODEL_ROOT=... FIRST_FRAME=... OUTPUT_DIR=...
# 说明：不自带任何硬编码路径，全部沿用 env/paths.env 与自动探测。

set -o pipefail

MODE=${1:-check}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PKG=${PKG:-$(cd "$HERE/.." && pwd)}
SS=${SS:-$PKG/src}
GPUS=${GPUS:-auto}     # auto = 跟随本次批作业实际分配的卡数
LOGDIR=${LOGDIR:-$PKG/logs}
mkdir -p "$LOGDIR" 2>/dev/null || LOGDIR=/tmp
TS=$(date +%m%d_%H%M%S)

echo "########## h3_8card.sh  mode=$MODE  $(date '+%F %T') ##########"
echo "PKG=$PKG  SS=$SS  GPUS=$GPUS"

# ---------- 公共：DCU / 工具链环境 ----------
if [ -z "$DTK_ROOT" ] || [ ! -d "$DTK_ROOT" ]; then
  for c in /public/software/compiler/dtk-* /opt/dtk* /opt/rocm*; do
    [ -d "$c" ] && DTK_ROOT=$c && break
  done
  export DTK_ROOT
fi
[ -n "$DTK_ROOT" ] && [ -f "$DTK_ROOT/env.sh" ] && source "$DTK_ROOT/env.sh"

[ -f "$PKG/env/paths.env" ] && source "$PKG/env/paths.env"
MID=${INFINI_ROOT:-/work2/infini}

PY=${PY:-}
if [ -z "$PY" ]; then
  if [ -n "$H3_VENV" ] && [ -x "$H3_VENV/bin/python" ]; then PY=$H3_VENV/bin/python
  elif [ -x "$PKG/../venv/h3/bin/python" ]; then PY=$PKG/../venv/h3/bin/python
  elif command -v python3 >/dev/null 2>&1; then PY=python3
  else PY=python; fi
fi

# ---------- GPUS=auto：跟随本次作业实际分到的卡（平台常把卡重编号）----------
if [ "$GPUS" = "auto" ]; then
  GPUS=$("$PY" -c "import torch;print(','.join(str(i) for i in range(torch.cuda.device_count())))" 2>/dev/null)
  [ -z "$GPUS" ] && GPUS=auto
fi
if [ "$GPUS" = "auto" ]; then NGPU=0; else NGPU=$(echo "$GPUS" | awk -F, '{print NF}'); fi

# 让 check 模式里的 import infinicore 也能用（与 run_fl2va_direct.sh 保持一致）export PYTHONPATH=$SS/InfiniCore/python:$SS/InfiniLM/python:$PYTHONPATH
export LD_LIBRARY_PATH=$MID/lib:${DTK_ROOT:-}/cuda/cuda-12/lib64:${DTK_ROOT:-}/lib:/opt/hyhal/lib:/opt/hyhal/lib64:$LD_LIBRARY_PATH
[ -f "${DTK_ROOT:-}/cuda/cuda-12/targets/x86_64-linux/lib/libcuda.so.1" ] \
  && export LD_PRELOAD=${DTK_ROOT:-}/cuda/cuda-12/targets/x86_64-linux/lib/libcuda.so.1

# =====================================================================
# check
# =====================================================================
if [ "$MODE" = "check" ]; then
  echo "=========== 1. 身份与调度 ==========="
  echo "host=$(hostname)  user=$(whoami)  pwd=$(pwd)"
  echo "作业ID: JOBID=${SLURM_JOB_ID:-none}  PARTITION=${SLURM_JOB_PARTITION:-none}"
  command -v whichpartition >/dev/null 2>&1 && { echo -n "whichpartition: "; whichpartition 2>/dev/null; }

  echo "=========== 2. 8 张 DCU 是否都可见 ==========="
  if command -v hy-smi >/dev/null 2>&1; then
    echo "-- hy-smi --showproductname --"
    hy-smi --showproductname 2>/dev/null | head -20
    echo "-- 卡数统计 --"
    echo "hy-smi 报告卡数: $(hy-smi -L 2>/dev/null | grep -c . || echo '?')"
  else
    echo "[WARN] 没有 hy-smi，只能用 torch 看"
  fi

  echo "=========== 3. torch / 驱动栈 ==========="
  "$PY" - <<'PY'
import os, sys
print("python", sys.version.split()[0], sys.executable)
try:
    import torch
    print("torch", torch.__version__, "hip", getattr(torch.version, "hip", None))
    n = torch.cuda.device_count()
    print("torch.cuda.device_count() =", n)
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        print(f"  [{i}] {p.name}  {p.total_memory/1024**3:.1f} GiB")
    # 每卡快速分配测试（1 GiB），确认不是只可见不可用
    bad = []
    for i in range(n):
        try:
            with torch.cuda.device(i):
                x = torch.empty(1024 * 1024 * 256, dtype=torch.float16, device=f"cuda:{i}")
                x.zero_()
                del x
        except Exception as e:
            bad.append((i, type(e).__name__, str(e)[:80]))
    print("分配自检:", "全部通过" if not bad else f"失败卡 {bad}")
except Exception as e:
    print("[FAIL] import torch:", type(e).__name__, e)
PY

  echo "=========== 4. 自编译产物 ==========="
  for f in "$MID/lib/libinfiniop.so" "$MID/lib/libinfinicore.so"; do
    [ -f "$f" ] && echo "  [OK] $f  $(ls -lh "$f" | awk '{print $5}')" || echo "  [WARN] 缺 $f"
  done
  ls "$MID/lib" 2>/dev/null | grep -Ei "ccl|hccl|rccl" | sed 's/^/  CCL库: /' || echo "  [WARN] 未见 CCL 库（多卡 all-reduce 依赖它）"

  echo "=========== 5. 权重 / 首帧 / 磁盘 ==========="
  if [ -f "$MODEL_ROOT/model_index.json" ]; then
    echo "  [OK] MODEL_ROOT=$MODEL_ROOT"
    du -sh "$MODEL_ROOT" 2>/dev/null | sed 's/^/       /'
  else
    echo "  [FAIL] MODEL_ROOT 无 model_index.json: $MODEL_ROOT"
    echo "         搜索备选："
    find /work2 /public/ai_data /public/home/"$(whoami)" -maxdepth 5 -name model_index.json 2>/dev/null | head -5 | sed 's/^/         /'
  fi
  [ -f "$FIRST_FRAME" ] && echo "  [OK] FIRST_FRAME=$FIRST_FRAME" || echo "  [FAIL] 缺首帧 $FIRST_FRAME"
  echo "  输出目录: $OUTPUT_DIR"
  df -h "$OUTPUT_DIR" 2>/dev/null | tail -1 | sed 's/^/  /'
  echo "  内存: $(free -g 2>/dev/null | awk '/Mem:/{print $2\" GB 总 / \"$7\" GB 可用\"}')"
  echo "  CPU : $(nproc) 核"

  echo "=========== 6. 源码树 ==========="
  [ -f "$SS/InfiniLM/examples/minimax_h3.py" ] && echo "  [OK] InfiniLM 入口" || echo "  [FAIL] 缺 InfiniLM"
  [ -d "$SS/InfiniCore/python/infinicore" ] && echo "  [OK] InfiniCore python" || echo "  [FAIL] 缺 InfiniCore python"
  ND=$(find "$SS/InfiniCore/build/ninetoothed" -name '*.cpp' 2>/dev/null | wc -l)
  echo "  九齿产物 .cpp 数: ${ND:-0}（应为 908）"
  "$PY" -c "import infinicore, infinicore.sync; print('  [OK] import infinicore', getattr(infinicore,'__version__','?'))" 2>&1 | tail -2

  echo "=========== 结论 ==========="
  if [ -f "$MODEL_ROOT/model_index.json" ] && [ -f "$FIRST_FRAME" ]; then
    echo "可以跑。下一步："
    echo "  bash scripts/h3_8card.sh smoke     # 8 卡冒烟（1-2 分钟）"
  else
    echo "先解决上面 [FAIL] 项，再跑 smoke。"
  fi
  exit 0
fi

# =====================================================================
# smoke / full / perf / fallback
# =====================================================================
RUNNER=$PKG/scripts/run_fl2va_direct.sh
[ -f "$RUNNER" ] || { echo "ERROR: 缺 $RUNNER"; exit 1; }

export GPUS
case "$MODE" in
  smoke)     export SMOKE=1 ;;
  full|perf) export SMOKE=0 ;;
  fallback)  export SMOKE=1 GPUS=0,1,2,3; NGPU=4 ;;
  *) echo "未知模式: $MODE（check|smoke|full|perf|fallback）"; exit 2 ;;
esac

echo ">>> 启动：GPUS=$GPUS TP=$NGPU SMOKE=${SMOKE:-0}"
START=$(date +%s)
bash "$RUNNER" 2>&1 | tee "$LOGDIR/${MODE}_${TS}.log"
RC=${PIPESTATUS[0]}
END=$(date +%s)
DUR=$((END-START))

echo "=============================================="
echo "mode=$MODE  exit=$RC  墙钟耗时=${DUR}s  (${NGPU}卡 × ${DUR}s = $((DUR*NGPU)) 卡秒)"
echo "完整日志: $LOGDIR/${MODE}_${TS}.log"
ls -lh "$OUTPUT_DIR"/fl2va_*.mp4 2>/dev/null | tail -3

if [ "$MODE" = "perf" ] && [ "$RC" -eq 0 ]; then
  CSV=$OUTPUT_DIR/perf_8card.csv
  [ -f "$CSV" ] || echo "time,gpus,frames,steps,wall_s,card_seconds,video" > "$CSV"
  VIDEO=$(ls -t "$OUTPUT_DIR"/fl2va_*.mp4 2>/dev/null | head -1)
  echo "$(date '+%F %T'),$NGPU,${NUM_FRAMES:-124},${STEPS:-50},$DUR,$((DUR*NGPU)),$(basename "$VIDEO" 2>/dev/null)" >> "$CSV"
  echo "性能已落表: $CSV"
  grep -Ei "timing|elapsed|耗时|generate|vae|dit" "$LOGDIR/${MODE}_${TS}.log" | tail -20
fi

if [ "$RC" -ne 0 ]; then
  echo "---- 常见原因速查 ----"
  echo "1) head 数不能被 $NGPU 整除 / TP 初始化失败  ->  bash scripts/h3_8card.sh fallback   (4 卡)"
  echo "2) 显存不足（VAE 解码或 DiT 激活）          ->  降分辨率或帧数，先用 smoke 验证"
  echo "3) 缺 CCL 库 / all-reduce 挂住              ->  检查 \$INFINI_ROOT/lib 下 ccl 库，或先 4 卡跑通"
  echo "4) import infinicore 失败                  ->  回 check 模式看第 4 项"
fi
exit "$RC"
