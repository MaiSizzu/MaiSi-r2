#!/bin/bash
# preflight.sh — 新平台上跑 FL2VA 之前的一次性自检（不烧卡时，几秒钟）
# 用法： bash scripts/preflight.sh
# 只读检查，不启动推理。

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PKG=${PKG:-$(cd "$HERE/.." && pwd)}
SS=${SS:-$PKG/src}
FAIL=0
WARN=0

ok()   { echo "  [OK]   $*"; }
bad()  { echo "  [FAIL] $*"; FAIL=$((FAIL+1)); }
warn() { echo "  [WARN] $*"; WARN=$((WARN+1)); }

echo "================ 1. 机器 ================"
echo "  host   : $(hostname)"
echo "  user   : $(whoami)"
echo "  pwd    : $(pwd)"
echo "  包路径 : $PKG"

echo "================ 2. DCU 驱动 ================"
if command -v hy-smi >/dev/null 2>&1; then
  ok "hy-smi 存在"
  hy-smi --showproductname 2>/dev/null | head -12 | sed 's/^/    /'
  hy-smi 2>/dev/null | grep -Ei "DCU|MiB|%" | head -12 | sed 's/^/    /'
elif command -v rocm-smi >/dev/null 2>&1; then
  ok "rocm-smi 存在"
else
  warn "没有 hy-smi / rocm-smi，无法从驱动层确认卡数（下面用 torch 看）"
fi

echo "================ 3. 工具链 ================"
# DTK 自动探测
if [ -z "$DTK_ROOT" ] || [ ! -d "$DTK_ROOT" ]; then
  for c in /public/software/compiler/dtk-* /opt/dtk* /opt/rocm*; do
    [ -d "$c" ] && DTK_ROOT=$c && break
  done
fi
if [ -n "$DTK_ROOT" ] && [ -d "$DTK_ROOT" ]; then
  ok "DTK_ROOT=$DTK_ROOT"
  if [ -f "$DTK_ROOT/env.sh" ]; then ok "DTK env.sh 存在"; else warn "DTK_ROOT 下没有 env.sh（新版布局可能不需要）"; fi
else
  bad "找不到 DTK（DTK_ROOT 未设置且 /public/software/compiler/dtk-* 不存在）"
fi
command -v hipcc >/dev/null 2>&1 && ok "hipcc: $(command -v hipcc)" || warn "hipcc 不在 PATH（只跑推理不需要）"

echo "================ 4. Python / torch ================"
PY=${PY:-}
if [ -z "$PY" ]; then
  if [ -n "$H3_VENV" ] && [ -x "$H3_VENV/bin/python" ]; then PY=$H3_VENV/bin/python
  elif [ -x "$PKG/../venv/h3/bin/python" ]; then PY=$PKG/../venv/h3/bin/python
  elif command -v python3 >/dev/null 2>&1; then PY=python3
  else PY=python; fi
fi
echo "  解释器 : $PY"
"$PY" - <<'PY' || FAIL=$((FAIL+1))
import sys
print(f"  [OK]   python {sys.version.split()[0]}  ({sys.executable})")
try:
    import torch
    print(f"  [OK]   torch {torch.__version__}  hip={getattr(torch.version,'hip',None)}")
    n = torch.cuda.device_count()
    print(f"  [{'OK' if n>0 else 'FAIL'}]   torch.cuda.device_count() = {n}")
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        print(f"         [{i}] {p.name}  {p.total_memory/1024**3:.1f} GiB")
    sys.exit(0 if n > 0 else 1)
except Exception as e:
    print(f"  [FAIL] import torch 失败: {type(e).__name__}: {e}")
    sys.exit(1)
PY

echo "================ 5. 模型权重 ================"
CAND="$MODEL_ROOT
/work2/ai_data/models/MiniMax/MiniMax-H3/FL2VA
$HOME/models/MiniMax-H3/FL2VA
$PKG/../models/FL2VA"
FOUND=""
for d in $CAND; do
  [ -n "$d" ] && [ -f "$d/model_index.json" ] && FOUND=$d && break
done
if [ -z "$FOUND" ]; then
  # 兜底搜索（限深度，避免扫全盘）
  FOUND=$(find /work2 /public /home /mnt /data -maxdepth 5 -name model_index.json 2>/dev/null \
          | grep -i "minimax\|h3\|fl2va" | head -1 | xargs -r dirname)
fi
if [ -n "$FOUND" ]; then
  ok "MODEL_ROOT=$FOUND"
  du -sh "$FOUND" 2>/dev/null | sed 's/^/    /'
  [ -f "$FOUND/transformer/diffusion_pytorch_model.safetensors.index.json" ] \
    && ok "transformer 权重索引存在" || warn "未见 transformer 权重索引，确认目录结构"
else
  bad "找不到 FL2VA 权重（需 model_index.json）。用 MODEL_ROOT=/实际/路径 覆盖"
fi

echo "================ 6. 输入首帧 ================"
FIRST=${FIRST_FRAME:-$PKG/inputs/first_frame.jpg}
if [ ! -f "$FIRST" ]; then
  FIRST=$(find "$PKG" -maxdepth 3 \( -name "*.jpg" -o -name "*.png" \) 2>/dev/null | head -1)
fi
[ -n "$FIRST" ] && [ -f "$FIRST" ] && ok "FIRST_FRAME=$FIRST" \
  || bad "缺少首帧图片。放一张到 $PKG/inputs/first_frame.jpg，或 export FIRST_FRAME=/路径"

echo "================ 7. 自编译算子库 ================"
IR=${INFINI_ROOT:-/work2/infini}
if [ -f "$IR/lib/libinfiniop.so" ]; then
  ok "libinfiniop.so: $IR/lib/libinfiniop.so"
  ls -lh "$IR/lib/libinfiniop.so" | sed 's/^/    /'
else
  warn "没有 $IR/lib/libinfiniop.so"
  warn "若平台上没有现成安装，需要先按 README 构建 InfiniCore（约 20-40 分钟）"
fi

echo "================ 8. 源码树 ================"
[ -f "$SS/InfiniLM/examples/minimax_h3.py" ] && ok "InfiniLM 入口存在" || bad "缺 $SS/InfiniLM/examples/minimax_h3.py"
[ -d "$SS/InfiniCore/python/infinicore" ] && ok "InfiniCore python 包存在" || bad "缺 InfiniCore python 包"
[ -d "$SS/InfiniCore/build/ninetoothed" ] && ok "九齿生成产物存在（$(find "$SS/InfiniCore/build/ninetoothed" -name '*.cpp' | wc -l) 个 .cpp）" || warn "无 build/ninetoothed，重建时需先跑 build_ntops.py"

echo "================ 汇总 ================"
echo "  FAIL=$FAIL  WARN=$WARN"
if [ "$FAIL" -eq 0 ]; then
  echo "  结论：可以跑。先冒烟："
  echo "    SMOKE=1 GPUS=0,1,2,3,4,5,6,7 bash scripts/run_fl2va_direct.sh"
else
  echo "  结论：有阻塞项，先解决上面的 [FAIL]"
fi
exit "$FAIL"
