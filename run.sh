#!/bin/bash
# =============================================================================
# run.sh — InfiniFabric 上跑 MiniMax-H3 FL2VA 的唯一入口
#
# 项目里只放这一个文件，程序参数填下面之一：
#   check     只读自检（2 分钟，不写盘、不编译），第一次必须先跑这个
#   setup     解包 + 装 diffusers + 生成首帧（5 分钟）
#   build     编译 InfiniCore 并安装到 AFS（ccache 热了以后 3~5 分钟）
#   smoke     冒烟 256x256 / 8 帧 / 2 步（3 分钟）
#   full      正式 124 帧 / 50 步（约 60 分钟）
#   all       setup + build + smoke，编不完就停在编译，下轮接着来
#   localall  同 all，但不复用 AFS 上的编译树（AFS 编译太慢时用这个）
#             （编译日志实时打到控制台，被 50 分钟上限杀也能看到死在哪一步）
#   retry4    smoke 失败时改用 2 卡重试
#
# 编译树怎么来（都为了绕开 AFS 上 8668 个小文件的慢速遍历，cp -a 一次要 31 分钟）：
#   · $AFS/ic_work 已存在 → 直接复用，能接着编上次没编完的目标
#   · 不存在 → 把 24 MB 的 MaiSi-r2-full.tar.gz 在本机内存盘展开（约 15 秒）
# 编好的阶段会打戳（outputs/.stamp_*），下一轮直接跳过，不再重编。
# 想强制全量重编：删掉 $AFS/ic_work 和 $AFS/outputs/.stamp_* 再跑。
#
# 第二个参数是卡数，默认跟随实际可见卡数（auto）。
#   full 8      → 8 卡张量并行
#   full 2      → 2 卡（TP=8 报 heads 不整除时用）
#
# 所有大文件都落在个人 AFS（$IFAB_AFS_HOME），不占项目空间。
# =============================================================================

MODE=${1:-check}
WANT_GPU=${2:-auto}

# ---------- 定位 AFS ----------
AFS="${IFAB_AFS_HOME:-}"
if [ -z "$AFS" ]; then
  for c in "$HOME/afs" "$HOME"; do
    if [ -d "$c" ] && [ -w "$c" ]; then AFS="$c"; break; fi
  done
fi
AFS=${AFS:-$HOME}

# ---------- 容器里的工具约束 ----------
# 1) xmake 以 root 运行会自锁，必须显式放行
export XMAKE_ROOT=y
# 2) xmake 的全局目录也放到 AFS：编译树跨作业复用时，全局缓存若每轮换位置，
#    xmake 会认为环境变了而重跑 configure（白扔 3~4 分钟）
if mkdir -p "$AFS/xmake-global" 2>/dev/null && [ -w "$AFS/xmake-global" ]; then
  export XMAKE_GLOBALDIR=${XMAKE_GLOBALDIR:-$AFS/xmake-global}
else
  export XMAKE_GLOBALDIR=${XMAKE_GLOBALDIR:-${TMPDIR:-/tmp}/xmake-global}
fi
# 3) 容器读不到 /etc/gitconfig（Permission denied 会让 git clone 直接 fatal），全部绕开
export GIT_CONFIG_NOSYSTEM=1
export GIT_CONFIG_GLOBAL=${GIT_CONFIG_GLOBAL:-$AFS/.gitconfig}
export GIT_TERMINAL_PROMPT=0

PKG=$AFS/MaiSi-r2
INFINI=$AFS/infini
OUT=$AFS/outputs
PYLIBS=$AFS/pylibs
# infinicore 的 Python 包目录是上个作业 uid 留下的只读树，写不进去，所以在可写盘上拼一个
# overlay（包文件 + 编好的 .so），PYTHONPATH 前置它；/dev/shm 优先，装不下再退 AFS
OVL=${OVL:-/dev/shm/ic_py}
FIRST_FRAME=$AFS/inputs/first_frame.png
TS=$(date +%m%d_%H%M%S)
mkdir -p "$OUT" 2>/dev/null
LOG=$OUT/run_${MODE}_${TS}.log
: > "$LOG" 2>/dev/null || LOG=/dev/null
# 注意 1：不能用 `exec > >(tee ...)`，部分容器没有 /dev/fd（probe4 已踩过）
# 注意 2：tail 必须在 exec 重定向之前启动，否则它会把自己的输出也写回日志（死循环）
TAILPID=""
if [ "$LOG" != /dev/null ]; then
  tail -n +1 -f "$LOG" 2>/dev/null & TAILPID=$!
fi
exec >>"$LOG" 2>&1
_cleanup_log() { [ -n "$TAILPID" ] && { sleep 1; kill "$TAILPID" 2>/dev/null; }; }
trap _cleanup_log EXIT

hr() { echo "----------------------------------------------------------------"; }
say() { echo "[$(date +%H:%M:%S) +$((SECONDS))s] $*"; }
die() { echo; echo "!!! $*"; echo "!!! 日志: $LOG"; exit 1; }

echo "############ run.sh mode=$MODE gpu=$WANT_GPU  $(date '+%F %T') ############"
# 版本自报：不用登录终端，日志第一屏就能确认跑的是哪个文件（md5 可与我给的核对）
_self=${BASH_SOURCE[0]:-$0}
if [ -f "$_self" ]; then
  _sz=$(wc -c < "$_self" 2>/dev/null | tr -d ' ')
  echo "SELF = $(md5sum "$_self" 2>/dev/null | awk '{print $1}')  ${_sz:-?} 字节  $_self"
else
  echo "SELF = (拿不到脚本自身路径：$_self)"
fi
echo "AFS  = $AFS"
echo "PKG  = $PKG"
echo "LOG  = $LOG"
hr

# =============================================================================
# 1. 环境探测
# =============================================================================
say "探测 DTK / 编译器"
DTK_ROOT=${DTK_ROOT:-}
if [ -z "$DTK_ROOT" ] || [ ! -d "$DTK_ROOT" ]; then
  for c in /opt/dtk /opt/dtk-* /public/software/compiler/dtk-* /opt/rocm*; do
    [ -d "$c" ] && DTK_ROOT=$c && break
  done
fi
echo "  DTK_ROOT = ${DTK_ROOT:-<未找到>}"
if [ -n "$DTK_ROOT" ]; then
  echo "  内容:"; ls "$DTK_ROOT" 2>/dev/null | head -20 | sed 's/^/    /'
  [ -f "$DTK_ROOT/env.sh" ] && { source "$DTK_ROOT/env.sh"; echo "  已 source env.sh"; }
fi

export HIP_PATH=${HIP_PATH:-$DTK_ROOT}
export ROCM_PATH=${ROCM_PATH:-$DTK_ROOT}
# hipcc 在 $DTK_ROOT/hip/bin，hy-smi 在 /opt/hyhal/bin，但默认都不在 PATH 里
export PATH=$DTK_ROOT/hip/bin:$DTK_ROOT/bin:$DTK_ROOT/llvm/bin:/opt/hyhal/bin:$PATH

# ---------- CUDA 兼容层（xmake 和运行库都要它）----------
CUDA_COMPAT=""
for c in "$DTK_ROOT/cuda/cuda-12" "$DTK_ROOT/cuda" $(ls -d "$DTK_ROOT"/cuda/cuda-* 2>/dev/null); do
  # 必须真的有 nvcc（xmake 要靠它回版本号），只有目录不算
  if [ -x "$c/bin/nvcc" ]; then CUDA_COMPAT=$c; break; fi
done
if [ -z "$CUDA_COMPAT" ]; then
  say "DTK 下没有可用的 cuda 兼容目录（缺 bin/nvcc），造一个 shim 给 xmake 用"
  CUDA_COMPAT=$AFS/cudashim
  if [ -d "$DTK_ROOT/cuda/cuda-12" ] && [ -z "$(ls -A "$AFS/cudashim" 2>/dev/null)" ]; then
    # 有真实兼容层目录但缺 nvcc：把它的内容软链过来，再补一个假 nvcc
    mkdir -p "$CUDA_COMPAT"
    for d in "$DTK_ROOT/cuda/cuda-12"/*; do ln -sfn "$d" "$CUDA_COMPAT/$(basename "$d")" 2>/dev/null; done
  fi
  mkdir -p "$CUDA_COMPAT/bin" "$CUDA_COMPAT/lib64"
  if [ ! -x "$CUDA_COMPAT/bin/nvcc" ]; then
    cat > "$CUDA_COMPAT/bin/nvcc" <<'NVCC'
#!/bin/sh
case "$*" in
  *--version*|*-V*) echo "nvcc: NVIDIA (R) Cuda compiler driver"; echo "Cuda compilation tools, release 12.0, V12.0.140";;
esac
exit 0
NVCC
    chmod +x "$CUDA_COMPAT/bin/nvcc"
  fi
  if [ ! -e "$CUDA_COMPAT/lib64/libamdhip64.so" ]; then
    ln -sfn "$DTK_ROOT/lib/"*.so* "$CUDA_COMPAT/lib64/" 2>/dev/null
  fi
fi
export CUDA_HOME=$CUDA_COMPAT
export CUDA_PATH=$CUDA_COMPAT
export PATH=$CUDA_COMPAT/bin:$PATH
echo "  CUDA_HOME = $CUDA_COMPAT"

# ---------- 可见卡数 ----------
VIS=$(python3 - <<'PY' 2>/dev/null || echo 0
import torch
print(torch.cuda.device_count())
PY
)
VIS=${VIS:-0}
echo "  可见 DCU = $VIS"
if [ "$WANT_GPU" = "auto" ]; then
  if [ "$VIS" -gt 0 ]; then WANT_GPU=$VIS; else WANT_GPU=1; fi
fi
echo "  本次使用 = $WANT_GPU 卡"

hr

# =============================================================================
# 2. 权重 / 首帧 / 包体 定位
# =============================================================================
SHARED=${SHARED:-/public/home/aclsylqidf/shared}

find_model() {
  # 实测：MiniMax-H3 根目录没有 model_index.json，
  # FL2VA/ 和 Ref2VA/ 和 diffusers/ 各有一个；fl2va 任务要用 FL2VA 那个入口
  local cands=(
    "$SHARED/MiniMax-H3/FL2VA"
    "$SHARED/MiniMax-H3"
    "$SHARED/MiniMax-H3/diffusers"
    "$AFS/models/MiniMax-H3/FL2VA"
    "/work2/ai_data/models/MiniMax/MiniMax-H3/FL2VA"
  )
  for c in "${cands[@]}"; do
    if [ -f "$c/model_index.json" ]; then echo "$c"; return; fi
    if [ -f "$c/diffusers/model_index.json" ]; then echo "$c"; return; fi
  done
  # 兜底搜索（限深度，60 秒）
  local f
  f=$(timeout 60 find "$SHARED" -maxdepth 4 -name model_index.json 2>/dev/null | head -1)
  if [ -n "$f" ]; then dirname "$f"; fi
}

say "定位模型权重"
MODEL_ROOT=$(find_model)
if [ -n "$MODEL_ROOT" ]; then
  echo "  MODEL_ROOT = $MODEL_ROOT"
  ls -la "$MODEL_ROOT" 2>/dev/null | head -12 | sed 's/^/    /'
  for sub in diffusers FL2VA text_encoder transformer; do
    [ -d "$MODEL_ROOT/$sub" ] && echo "    有 $sub/"
  done
else
  echo "  MODEL_ROOT = <未找到>"
fi

say "定位代码包"
TARBALL=""
for c in "$AFS/MaiSi-r2-full.tar.gz" "$AFS/pkg/MaiSi-r2-full.tar.gz" "$AFS/MaiSi-r2.tar.gz" \
         "./MaiSi-r2-full.tar.gz" "$AFS/MaiSi-r2.tgz"; do
  [ -f "$c" ] && TARBALL=$c && break
done
echo "  TARBALL = ${TARBALL:-<未找到>}"
echo "  已解包  = $( [ -d "$PKG/src/InfiniLM" ] && echo yes || echo no )"

say "定位首帧"
if [ ! -f "$FIRST_FRAME" ]; then
  for c in "$AFS/inputs/first_frame.jpg" "$AFS/inputs/first_frame.png"; do
    [ -f "$c" ] && FIRST_FRAME=$c && break
  done
fi
echo "  FIRST_FRAME = $FIRST_FRAME $( [ -f "$FIRST_FRAME" ] && echo '(存在)' || echo '(待生成)' )"

say "AFS 余量"
df -h "$AFS" 2>/dev/null | tail -1 | sed 's/^/  /'

hr

# =============================================================================
# 3. 各模式
# =============================================================================
ensure_pkg() {
  if [ -d "$PKG/src/InfiniLM" ] && [ -d "$PKG/src/InfiniCore" ]; then
    say "代码包已解包，跳过"
    return 0
  fi
  [ -n "$TARBALL" ] || die "AFS 里找不到 MaiSi-r2-full.tar.gz，请先上传到「个人存储」"
  say "解包 $TARBALL → $AFS （约 150 MB / 8668 个文件，需要 1-3 分钟）"
  tar xzf "$TARBALL" -C "$AFS" || die "解包失败"
  [ -d "$PKG/src/InfiniLM" ] || die "解包后仍找不到 $PKG/src/InfiniLM"
  say "解包完成：$(find "$PKG" -type f | wc -l) 个文件"
}

ensure_deps() {
  # all = setup + build，会连续调两次；第一次过了就直接跳过，省一分钟
  [ "${DEPS_OK:-0}" = "1" ] && { say "依赖已校验过，跳过"; return 0; }
  # 平台不允许同名覆盖，所以新版用 -v3 后缀；优先命中 v3
  local ZIP=""
  ZIP=$(ls -t "$AFS"/pylibs*-v3.zip "$AFS"/dl/pylibs*-v3.zip 2>/dev/null | head -1)
  [ -n "$ZIP" ] || ZIP=$(ls -t "$AFS"/pylibs*-v2.zip "$AFS"/dl/pylibs*-v2.zip 2>/dev/null | head -1)
  [ -n "$ZIP" ] || ZIP=$(ls -t "$AFS"/pylibs*.zip "$AFS"/dl/pylibs*.zip 2>/dev/null | head -1)
  if [ -z "$ZIP" ]; then
    for c in "$AFS/pylibs-py311.zip" "$AFS/pylibs.zip" "$AFS/dl/pylibs-py311.zip"; do
      [ -f "$c" ] && ZIP=$c && break
    done
  fi

  if [ -z "$ZIP" ]; then
    say "!! 个人存储里没有 pylibs-py311.zip"
    say "!! 说明：本容器的 pip 会因为读 /etc/debian_version 无权限而直接崩，pip 路线不可用"
    die "缺离线依赖包 pylibs-py311.zip（要传到「个人存储」，不是项目）"
  fi

  local ZSIZE=0
  ZSIZE=$(stat -c%s "$ZIP" 2>/dev/null || echo 0)
  say "  依赖包 = $(basename "$ZIP")  $((ZSIZE/1048576)) MB"
  if [ "$ZSIZE" -gt 0 ] && [ "$ZSIZE" -lt 12000000 ]; then
    say "  !! 警告：这个包只有 $((ZSIZE/1048576)) MB，是缺 httpx2/httpcore2 的旧版"
    say "  !! 需要 16 MB 的 pylibs-py311-v2.zip，否则依赖校验必失败"
  fi

  # 指纹：换了新包必须重新解包（旧的 .ready 是空文件，自然对不上）
  local STAMP=""
  STAMP=$(sha256sum "$ZIP" 2>/dev/null | cut -c1-16)
  [ -n "$STAMP" ] || STAMP=$(python3 -c "import os,sys;print(os.path.getsize(sys.argv[1]))" "$ZIP" 2>/dev/null)

  if [ -f "$PYLIBS/.ready" ] && [ -d "$PYLIBS/diffusers" ] \
     && [ "$(cat "$PYLIBS/.ready" 2>/dev/null)" = "$STAMP" ]; then
    say "pylibs 已就绪（指纹 $STAMP），跳过解包"
  else
    say "解包离线依赖 $ZIP → $PYLIBS（全程不调用 pip，指纹 $STAMP）"
    mkdir -p "$PYLIBS"
    python3 - "$ZIP" "$PYLIBS" <<'PY' || die "解包 pylibs 失败"
import sys, zipfile
src, out = sys.argv[1], sys.argv[2]
with zipfile.ZipFile(src) as z:
    names = z.namelist()
    z.extractall(out)
print("  解包条目 =", len(names))
PY
    [ -d "$PYLIBS/diffusers" ] || die "解包后没有 diffusers 目录，检查 $ZIP 是否完整"
    echo "$STAMP" > "$PYLIBS/.ready"
  fi

  # transformers>=5 把 Qwen2TokenizerFast 并进了基类，H3 的 encoders.py 顶层直接 import 会炸 → 打兼容补丁（幂等）
  local ENC=$PYLIBS/diffusers/modular_pipelines/minimax_h3/encoders.py
  if [ -f "$ENC" ]; then
    grep -q "Qwen2TokenizerFastFallback" "$ENC" || python3 - "$ENC" <<'PY'
import sys
p = sys.argv[1]
s = open(p, encoding="utf-8").read()
old = "from transformers import Qwen2TokenizerFast, Qwen3VLForConditionalGeneration, Qwen3VLProcessor"
new = """try:
    from transformers import Qwen2TokenizerFast
except ImportError:  # Qwen2TokenizerFastFallback: transformers>=5 已把 *TokenizerFast 并进基类
    from transformers import Qwen2Tokenizer as Qwen2TokenizerFast
from transformers import Qwen3VLForConditionalGeneration, Qwen3VLProcessor"""
if old in s:
    open(p, "w", encoding="utf-8").write(s.replace(old, new, 1))
    print("  已打补丁：H3 encoders.py 兼容 Qwen2TokenizerFast")
else:
    print("  ! 未找到预期 import 行，H3 encoders 补丁跳过（若已打过则正常）")
PY
  fi

  if [ ! -d "$PYLIBS/transformers" ]; then
    say "  !! 当前包里没有 transformers，只能用系统自带的旧版（要求 hub<1.0，和 diffusers 0.42 死冲突）"
    say "  !! 现在用的是 $(basename "$ZIP")，请把 pylibs-py311-v3.zip 传到个人存储根目录再跑"
  fi

  say "校验依赖（diffusers 0.42 + transformers 5.x + hub 2.x 是配套的一套）"
  PYTHONPATH="$PYLIBS:$PYTHONPATH" python3 - <<'PY'
import sys

def show(tag, fn):
    try:
        print("  %-12s = %s" % (tag, fn())); return True
    except Exception as e:
        print("  %-12s = FAIL %s" % (tag, str(e).split("\n")[0][:110]))
        import traceback
        for line in traceback.format_exc().strip().split("\n")[-6:]:
            print("      " + line.strip()[:170])
        return False

show("requests", lambda: __import__("requests").__version__)
show("httpx2", lambda: __import__("httpx2").__version__)
show("hub", lambda: __import__("huggingface_hub").__version__)
ok_d = show("diffusers", lambda: __import__("diffusers").__version__)
show("accelerate", lambda: __import__("accelerate").__version__)
show("safetensors", lambda: __import__("safetensors").__version__)

def h3():
    from diffusers import ModularPipeline  # noqa: F401
    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ImageReference  # noqa: F401
    import diffusers.modular_pipelines.minimax_h3.encoders as enc  # 真正会 import Qwen2TokenizerFast 的模块
    return "OK (%s)" % enc.Qwen2TokenizerFast.__name__
ok_h = show("H3 管线", h3)

def tr():
    import transformers
    v = transformers.__version__
    try:
        major = int(v.split(".")[0])
    except Exception:
        major = 0
    if major < 5:
        raise RuntimeError("%s 太旧：transformers 4.x 要求 hub<1.0，与 diffusers 0.42 冲突，需要 5.x" % v)
    from transformers import Qwen3VLForConditionalGeneration  # noqa: F401
    from transformers import AutoProcessor, AutoTokenizer, AutoConfig, DynamicCache  # noqa: F401
    return v
ok_t = show("transformers", tr)

if ok_d and ok_h and ok_t:
    sys.exit(0)
sys.exit(3)
PY
  local rc=$?
  if [ $rc -eq 0 ]; then say "依赖全部就绪"; DEPS_OK=1; return 0; fi
  die "依赖校验失败（见上）"
}

ensure_first_frame() {
  [ -f "$FIRST_FRAME" ] && { say "首帧已存在"; return 0; }
  say "生成首帧 512x512"
  mkdir -p "$(dirname "$FIRST_FRAME")"
  python3 - "$FIRST_FRAME" <<'PY'
import sys, zlib, struct
out = sys.argv[1]
W = H = 512
try:
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (W, H))
    d = ImageDraw.Draw(im)
    for y in range(H):
        t = y / (H - 1)
        d.line([(0, y), (W, y)], fill=(int(120 + 100 * t), int(150 + 60 * t), int(200 - 60 * t)))
    d.ellipse([180, 300, 340, 400], fill=(90, 70, 60))
    d.ellipse([230, 250, 290, 320], fill=(110, 85, 70))
    im.save(out)
    print("  PIL 生成:", out, im.size)
except Exception as e:
    print("  PIL 不可用(%s)，改用内置 PNG 编码" % e)
    rows = b""
    for y in range(H):
        rows += b"\x00" + bytes([80, 120 + y * 80 // H, 200 - y * 120 // H]) * W
    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff)
    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0))
    png += chunk(b"IDAT", zlib.compress(rows, 6))
    png += chunk(b"IEND", b"")
    open(out, "wb").write(png)
    print("  内置编码生成:", out)
PY
  [ -f "$FIRST_FRAME" ] || die "首帧生成失败"
}

ensure_xmake() {
  command -v xmake >/dev/null 2>&1 && { say "xmake 已存在: $(command -v xmake)"; return 0; }
  mkdir -p "$AFS/bin" "$AFS/dl"

  # 路径 1（首选）：上传的官方 xmake 二进制（不依赖容器网络 / git）
  # 官方 bundle 的链接方式变过：v3.0.8 起是动态链接 glibc 2.38（容器 glibc 旧就报
  # "GLIBC_2.38 not found"），只有 v3.0.4~v3.0.7 是 musl 静态单文件、任何 glibc 都能跑。
  # → 从高版本往下逐个试跑，能起来的那份才用。
  local CANDS="" sz
  CANDS=$( { ls -1 "$AFS"/xmake-bundle-* "$AFS"/dl/xmake-bundle-* \
                 "$AFS"/xmake*linux*x86_64 "$AFS"/dl/xmake*linux*x86_64 2>/dev/null; } \
           | while read -r f; do case "$f" in *.tar.gz|*.tgz|*.zip) continue ;; esac; [ -f "$f" ] && echo "$f"; done | sort -u -V -r )
  if [ -z "$CANDS" ]; then
    # 兜底：任何以 xmake 开头、大于 1 MB 的非压缩文件都当成二进制试
    for f in "$AFS"/xmake* "$AFS"/dl/xmake*; do
      [ -f "$f" ] || continue
      case "$f" in *.tar.gz|*.tgz|*.zip) continue ;; esac
      sz=$(stat -c%s "$f" 2>/dev/null || echo 0)
      [ "$sz" -gt 1000000 ] && CANDS="$CANDS $f"
    done
  fi

  local ok=0 BIN="" d e m
  if [ -n "$CANDS" ]; then
    say "候选 xmake 二进制（高版本优先，逐个试跑）："
    say "  架构=$(uname -m)  nproc=$(nproc 2>/dev/null || echo '?')  TMPDIR=${TMPDIR:-<空>}"
    for BIN in $CANDS; do
      say "  $(basename "$BIN")  $(stat -c%s "$BIN" 2>/dev/null) 字节  md5=$(md5sum "$BIN" 2>/dev/null | cut -d' ' -f1)"
      # AFS 优先（能跨作业复用；实测这个 NFS 允许 exec，否则上一轮不会报 GLIBC 而是 Permission denied）
      for d in "$AFS/xbin" "${TMPDIR:-/tmp}/xbin" "$HOME/.xbin"; do
        [ -n "$d" ] || continue
        mkdir -p "$d" 2>/dev/null || continue
        rm -f "$d/xmake" 2>/dev/null
        cp -f "$BIN" "$d/xmake" 2>/dev/null || continue
        chmod +x "$d/xmake" 2>/dev/null
        if [ -x "$d/xmake" ] && "$d/xmake" --version >/dev/null 2>&1; then
          export PATH="$d:$PATH"
          export XMAKE_GLOBALDIR="$d/.global"
          mkdir -p "$XMAKE_GLOBALDIR" 2>/dev/null
          say "  √ 可用：$(basename "$BIN") → $d"
          ok=1
          break
        fi
        m=$(ls -l "$d/xmake" 2>/dev/null | awk '{print $1}')
        e=$("$d/xmake" --version 2>&1 | head -2 | tr '\n' ' ')
        say "    $d 跑不起来：权限=$m 错误=${e:-<无输出>}"
      done
      [ "$ok" = "1" ] && break
    done
  fi
  [ "$ok" = "1" ] || say "  !! 上传的二进制都起不来（多半是 glibc 太旧），继续试离线 tarball / 源码编译"

  # 路径 2：AFS 里已有的离线 tarball（tar 会带权限位，不受 chmod 被拒影响）
  if ! command -v xmake >/dev/null 2>&1; then
    for f in "$AFS"/xmake*.tar.gz "$AFS"/dl/xmake*.tar.gz; do
      [ -f "$f" ] || continue
      say "试离线包 $f"
      mkdir -p "$AFS/xbin"
      tar xzf "$f" -C "$AFS/xbin" --strip-components=1 2>/dev/null || tar xzf "$f" -C "$AFS/xbin"
      [ -x "$AFS/xbin/xmake" ] && { export PATH="$AFS/xbin:$PATH"; say "  xmake 解到 $AFS/xbin"; }
      break
    done
  fi

  # 路径 3（兜底）：gitee 源码编译。已实测容器 git 读配置失败，仅在 1/2 都缺时尝试
  if ! command -v xmake >/dev/null 2>&1; then
    command -v git >/dev/null 2>&1 && command -v make >/dev/null 2>&1 || {
      say "没有 git/make，源码编译这条路也走不了"
    }
    say "尝试 gitee 源码编译 xmake（4-8 分钟）"
    rm -rf "$AFS/dl/xmake-src"
    if git -c http.sslVerify=false clone --depth 1 https://gitee.com/tboox/xmake.git "$AFS/dl/xmake-src" 2>&1 | tail -3; then
      ( cd "$AFS/dl/xmake-src" && ./configure >/dev/null 2>&1 && make -j4 >/dev/null 2>&1 )
      if [ -x "$AFS/dl/xmake-src/build/xmake" ]; then
        mkdir -p "$AFS/xbin"
        cp -f "$AFS/dl/xmake-src/build/xmake" "$AFS/xbin/xmake" 2>/dev/null
        chmod +x "$AFS/xbin/xmake" 2>/dev/null
        [ -x "$AFS/xbin/xmake" ] && export PATH="$AFS/xbin:$PATH"
      fi
    fi
  fi

  command -v xmake >/dev/null 2>&1 || die "xmake 装不上。把 xmake-bundle-v3.0.7.linux.x86_64 传到「个人存储」根目录再跑，脚本会自动 chmod 并使用它（3.0.8 起是动态链接，容器 glibc 太旧跑不起来）"
  local XV
  XV=$(xmake --version 2>/dev/null | head -1)
  say "xmake = $XV"
  # 项目 hygon.lua:72 用了 add_orders，这是 xmake 3.x 才有的接口；2.x 会报 attempt to call a nil value
  case "$XV" in
    *v3.*|*v4.*|*v5.*|*v6.*) : ;;
    *) die "当前是 xmake 2.x，本项目需要 3.x（add_orders 不存在）。请把 xmake-bundle-v3.0.7.linux.x86_64 传到「个人存储」根目录" ;;
  esac
}

do_check() {
  hr; say "自检"
  echo "  python3   = $(command -v python3)  $(python3 -V 2>&1)"
  echo "  hipcc     = $(command -v hipcc || echo '<无>')"
  echo "  clang     = $([ -x "$DTK_ROOT/llvm/bin/clang" ] && "$DTK_ROOT/llvm/bin/clang" --version | head -1)"
  echo "  cmake     = $(command -v cmake || echo '<无>')"
  echo "  ninja     = $(command -v ninja || echo '<无>')"
  echo "  git       = $(command -v git || echo '<无>')"
  echo "  xmake     = $(command -v xmake || echo '<无>')"
  local XB=""
  XB=$( { ls -1 "$AFS"/xmake-bundle-* "$AFS"/dl/xmake-bundle-* "$AFS"/xmake*linux*x86_64 2>/dev/null; } \
        | while read -r f; do case "$f" in *.tar.gz|*.tgz|*.zip) continue ;; esac; [ -f "$f" ] && echo "$f"; done \
        | sort -u -V -r | head -1 )
  if [ -n "$XB" ]; then
    echo "  离线 xmake = $(basename "$XB")  $(du -h "$XB" 2>/dev/null | cut -f1)  （build 时自动启用）"
  else
    echo "  离线 xmake = <无>  建议上传 xmake-bundle-v3.0.7.linux.x86_64 到个人存储根目录（必须是 3.x，且 3.0.7 是最后一个静态版）"
  fi
  local NT=""
  for f in "$PKG/src/InfiniCore/build/ninetoothed/ninetoothed.h" "$PKG/patch_inc/ninetoothed.h"; do
    [ -s "$f" ] && { NT=$f; break; }
  done
  echo "  九齿头文件 = ${NT:-<无，build 时自动写入 build/ninetoothed/>}"
  PYTHONPATH="$PYLIBS:$PYTHONPATH" python3 - <<'PY'
import sys
print("  python    =", sys.version.split()[0])
try:
    import torch; print("  torch     =", torch.__version__, "hip", torch.version.hip, "dev", torch.cuda.device_count())
except Exception as e:
    print("  torch     = FAIL", e)
try:
    import PIL; print("  Pillow    =", PIL.__version__)
except Exception as e:
    print("  Pillow    = FAIL(首帧生成会退回内置编码)")
try:
    import diffusers; print("  diffusers  =", diffusers.__version__)
except Exception:
    print("  diffusers  = 未安装（setup 时装）")
try:
    from diffusers import ModularPipeline
    from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3ImageReference
    print("  H3 管线   = OK（ModularPipeline + minimax_h3 都在）")
except Exception as e:
    print("  H3 管线   = 缺：%s" % str(e).split("\n")[0][:90])
PY
  echo "  --- 运行库 ---"
  ldconfig -p 2>/dev/null | grep -E 'libamdhip64|libhipblas|librocblas|librccl' | awk '{print "   ", $1}' | sort -u | head -8
  echo "  --- 产物 ---"
  echo "    libinfiniop.so : $(ls -lh "$INFINI/lib/libinfiniop.so" 2>/dev/null || echo '<未编译>')"
  echo "    _infinicore*.so: $(find "$PKG/src/InfiniCore/python" -name '_infinicore*.so' 2>/dev/null | head -1 || echo '<未编译>')"
  hr
  say "自检结束。若 MODEL_ROOT 有值、可见卡数 ≥ 需要的卡数，就可以跑 setup。"
}

do_setup() {
  ensure_pkg
  ensure_deps
  ensure_first_frame
  say "setup 完成"
}

patch_build_paths() {
  [ -n "$IC" ] || IC=$PKG/src/InfiniCore
  local PB=$PYLIBS/pybind11/include
  [ -d "$PB" ] || PB=$PYLIBS/pybind11/pybind11/include
  local PYINC
  PYINC=$(python3 -c "import sysconfig;print(sysconfig.get_paths()['include'])" 2>/dev/null)
  say "重写 xmake 里的 scnet 绝对路径"
  echo "  pybind11 include = $PB"
  echo "  python   include = ${PYINC:-<未取到>}"
  [ -d "$PKG/patch_inc" ] || die "缺 $PKG/patch_inc（CUB 兼容头，编译必需）"
  [ -d "$PB" ] || die "缺 pybind11 头文件，先跑 setup"
  local f
  for f in "$IC/xmake.lua" "$IC"/xmake/*.lua; do
    [ -f "$f" ] || continue
    # 先 grep 再 sed：已经改过的文件不要重写，否则 mtime 每轮都变，
    # 增量编译树会被迫重跑 configure、白扔 3~4 分钟
    grep -q "/work2/songshan/patch_inc" "$f" 2>/dev/null &&
      sed -i "s#/work2/songshan/patch_inc#$PKG/patch_inc#g" "$f"
    grep -q "/work2/songshan/xmake-pkgs/pybind11" "$f" 2>/dev/null &&
      sed -i "s#/work2/songshan/xmake-pkgs/pybind11[^\"]*#$PB#g" "$f"
    if [ -n "$PYINC" ] && grep -q "/public/software/apps/anaconda3" "$f" 2>/dev/null; then
      sed -i "s#/public/software/apps/anaconda3[^\"]*include/python3\.[0-9]*#$PYINC#g" "$f"
    fi
  done
  local left
  left=$(grep -rn "/work2/songshan\|anaconda3" "$IC/xmake.lua" "$IC"/xmake/*.lua 2>/dev/null | wc -l)
  echo "  残留 scnet 路径 = $left 处"
  [ "$left" -eq 0 ] || grep -rn "/work2/songshan\|anaconda3" "$IC/xmake.lua" "$IC"/xmake/*.lua 2>/dev/null | head -3 | sed 's/^/    /'
  # clang-15 → clang-17，新版本会多出新警告；-Werror 会让整轮编译白跑，先放宽
  if [ -f "$IC/xmake/hygon.lua" ] && grep -q 'set_warnings("all", "error")' "$IC/xmake/hygon.lua" 2>/dev/null; then
    sed -i 's#set_warnings("all", "error")#set_warnings("all")#' "$IC/xmake/hygon.lua"
    echo "  已放宽 -Werror（clang-17 新警告不再中断编译）"
  fi
}

# 选编译目录：优先原地编（AFS），不可写就把 126 MB 源码树复制到本机硬盘再编
# （作业之间可能换 uid，上一个作业建的目录当前 uid 写不进去 → xmake 建目录直接 Permission denied）
pick_ic() {
  SRC_IC=$PKG/src/InfiniCore
  IC=$SRC_IC
  local bad=0
  if ! touch "$SRC_IC/.wtest" 2>/dev/null; then bad=1; fi
  rm -f "$SRC_IC/.wtest" 2>/dev/null
  if [ -d "$SRC_IC/build" ] && ! touch "$SRC_IC/build/.wtest" 2>/dev/null; then bad=1; fi
  rm -f "$SRC_IC/build/.wtest" 2>/dev/null
  [ "$bad" = "0" ] && return 0

  say "  !! $SRC_IC 不可写（上个作业 uid 遗留），改用可写的编译树"
  # 两条路，都只为避免「遍历 AFS 上的 8668 个小文件」：
  #   路径 1：AFS 上已有建好的 $AFS/ic_work → 直接复用（能接着编上次没编完的对象）
  #          判据是「有 xmake.lua 且能建文件」；不认 .clone_ok，因为上一轮的作业
  #          （跑 cp -a 那版）没写这个标记，白扔掉已经编好 infiniop 的树太亏
  #   路径 2：没有树 → 用 24 MB 的 tar 包在本机内存盘展开（约 20 秒）
  # 上一轮用 cp -a 从 AFS 复制 445 MB，光复制就烧了 1917 秒，必须彻底避开。
  local W=$AFS/ic_work reuse=0
  if [ "${FORCE_LOCAL:-0}" = "1" ]; then
    say "  FORCE_LOCAL=1：跳过 AFS 上的 ic_work，强制在本机展开"
  elif [ -f "$W/xmake.lua" ] && touch "$W/.wtest" 2>/dev/null; then
    rm -f "$W/.wtest" 2>/dev/null
    IC=$W
    reuse=1
    say "  复用 AFS 编译树 = $IC"
    say "  （上次编好的对象还在，本轮只补没编完的目标）"
  fi
  if [ "$reuse" = "0" ]; then
    local LT
    if [ -d /dev/shm ] && [ -w /dev/shm ]; then LT=/dev/shm/songshan_pkg; else LT=${TMPDIR:-/tmp}/songshan_pkg; fi
    [ -n "$TARBALL" ] && [ -f "$TARBALL" ] || die "AFS 上没有 ic_work 树，也找不到 tar 包，无法建编译树"
    rm -rf "$LT" 2>/dev/null
    mkdir -p "$LT" || die "建不了 $LT"
    say "  在本机内存盘展开代码包 → $LT（tar 包 24 MB，约 20 秒）"
    tar xzf "$TARBALL" -C "$LT" || die "解包失败（$TARBALL）"
    [ -d "$LT/MaiSi-r2/src/InfiniCore" ] || die "解包后缺 MaiSi-r2/src/InfiniCore"
    IC=$LT/MaiSi-r2/src/InfiniCore
  fi
  say "  编译目录 = $IC"
  return 0
}

# ---------------------------------------------------------------------------
# 九齿运行时头文件：NineToothedTensor / NineToothedStream / NineToothedResult
# 以及 ninetoothed::AutoTuningCache、ninetoothed::ThreadSafeUnorderedMap。
# build/ninetoothed/*.h 里是 #include "ninetoothed.h"，但代码包里没有这个头
# （原环境是从 pip 包 ninetoothed 的缓存目录取到的）。这里补一份：
# 写到引用者同目录 build/ninetoothed/ 下，quoted include 直接命中，不依赖 -I；
# 再往 patch_inc、include 各写一份兜底。内容与 ninetoothed 0.26 的 aot 模板一致。
# ---------------------------------------------------------------------------
write_nt_header() {
  cat > "$1" <<'NINETOOTHED_H_EOF'
#ifndef NINETOOTHED_H
#define NINETOOTHED_H

#include <stdint.h>

#define NINETOOTHED_TRUE 1

#define NINETOOTHED_FALSE 0

#define NINETOOTHED_NONE 0

enum NineToothedDataType {
    NINETOOTHED_INT8,
    NINETOOTHED_INT16,
    NINETOOTHED_INT32,
    NINETOOTHED_INT64,
    NINETOOTHED_UINT8,
    NINETOOTHED_UINT16,
    NINETOOTHED_UINT32,
    NINETOOTHED_UINT64,
    NINETOOTHED_FLOAT16,
    NINETOOTHED_BFLOAT16,
    NINETOOTHED_FLOAT32,
    NINETOOTHED_FLOAT64
};

typedef struct {
    void *data;
    uint64_t *shape;
    int64_t *strides;
} NineToothedTensor;

typedef void *NineToothedStream;

typedef int NineToothedResult;

#ifdef __cplusplus
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace ninetoothed {

class AutoTuningCache {
public:
    AutoTuningCache(const std::string& path) : path{path} {}

    std::vector<int> lookup(const std::vector<int>& config) {
        load();

        auto iter = cache.find(config);

        if (iter != cache.end()) {
            return iter->second;
        }

        return default_meta;
    }

private:
    void load() {
        auto mtime = std::filesystem::last_write_time(path);

        if (mtime == last_mtime) {
            return;
        }

        last_mtime = mtime;

        cache.clear();

        std::ifstream file{path};
        std::string config_line;
        std::string meta_line;

        if (std::getline(file, config_line) && std::getline(file, meta_line)) {
            default_meta = parse(meta_line);
        }

        while (std::getline(file, config_line) && std::getline(file, meta_line)) {
            cache[parse(config_line)] = parse(meta_line);
        }
    }

    static std::vector<int> parse(const std::string& line) {
        std::vector<int> values;
        std::stringstream ss{line};
        std::string token;

        while (std::getline(ss, token, ',')) {
            auto trimmed = trim(token);

            if (!trimmed.empty()) {
                values.push_back(std::stoi(trimmed));
            }
        }

        return values;
    }

    static std::string trim(const std::string& s) {
        auto start = s.find_first_not_of(" \t\r\n");

        if (start == std::string::npos) {
            return "";
        }

        auto end = s.find_last_not_of(" \t\r\n");

        return s.substr(start, end - start + 1);
    }

    std::string path;

    std::map<std::vector<int>, std::vector<int>> cache;

    std::vector<int> default_meta;

    std::filesystem::file_time_type last_mtime;
};

}

#include <mutex>
#include <shared_mutex>
#include <unordered_map>

namespace ninetoothed {

template<typename Key, typename Value>
class ThreadSafeUnorderedMap {
public:
    Value& operator[](const Key& key) {
        {
            std::shared_lock lock{mutex};

            auto iter = map.find(key);

            if (iter != map.end()) {
                return iter->second;
            }
        }

        std::unique_lock lock{mutex};

        auto iter = map.find(key);

        if (iter == map.end()) {
            iter = map.emplace(key, Value{}).first;
        }

        return iter->second;
    }

private:
    std::unordered_map<Key, Value> map;

    mutable std::shared_mutex mutex;
};

}
#endif

#endif // NINETOOTHED_H
NINETOOTHED_H_EOF
}

ensure_nt_header() {
  local d=$IC/build/ninetoothed ok=0 t base ref
  [ -d "$d" ] || die "缺 $d（九齿预生成算子源码），代码包不完整"
  # 关键：内容相同就不要重写。这个头被 953 个生成的 .h 引用，
  # 每轮无脑覆盖会让 mtime 变新，xmake 判定全部过期 → 整个 infiniop 重编一遍。
  ref=$d/.nt_ref.$$.h   # 放在编译树里：一定可写，不依赖 TMPDIR 是否可用
  write_nt_header "$ref" 2>/dev/null && [ -s "$ref" ] && base=$(md5sum "$ref" 2>/dev/null | cut -d' ' -f1)
  rm -f "$ref" 2>/dev/null
  for t in "$d/ninetoothed.h" "$PKG/patch_inc/ninetoothed.h" "$IC/include/ninetoothed.h"; do
    if [ -s "$t" ]; then
      if [ -n "$base" ] && [ "$(md5sum "$t" 2>/dev/null | cut -d' ' -f1)" = "$base" ]; then
        ok=1; continue
      fi
    fi
    mkdir -p "$(dirname "$t")" 2>/dev/null
    if write_nt_header "$t.tmp" 2>/dev/null && mv -f "$t.tmp" "$t" 2>/dev/null && [ -s "$t" ]; then
      say "  ninetoothed.h -> $t"
      ok=1
    else
      rm -f "$t.tmp" 2>/dev/null
    fi
  done
  [ "$ok" = "1" ] || die "ninetoothed.h 写不进去（build/ninetoothed、patch_inc、include 都失败）"
  [ -s "$d/ninetoothed.h" ] || say "  !! 主位置没写成，靠 patch_inc / include 兜底（若报找不到 ninetoothed.h 就贴日志）"
}

# _infinicore 的安装位置和 Python 的 import 位置对不上：
#   xmake install -> python/infinicore/_infinicore*.so
#   代码里却是   from infinicore.lib import _infinicore  -> 要 python/infinicore/lib/
# 而 $SRC_IC 是上个作业 uid 的只读树，cp 回去会静默失败（2>/dev/null 把报错吃掉）。
# 于是把「完整包 + lib/ 下的 .so」拼到可写目录，PYTHONPATH 前置，绕开写权限。
ensure_infinicore_overlay() {
  # 单独跑 smoke / full 时没走 pick_ic，这里补一下
  [ -n "$SRC_IC" ] || SRC_IC=$PKG/src/InfiniCore
  if [ -z "$IC" ]; then
    # 有 .so 的树才是真编译树；只认 xmake.lua 会在标记文件丢失时选错
    if ls "$AFS/ic_work"/python/infinicore/_infinicore*.so >/dev/null 2>&1; then
      IC=$AFS/ic_work
    elif [ -f "$AFS/ic_work/xmake.lua" ]; then
      IC=$AFS/ic_work
    else
      IC=$SRC_IC
    fi
  fi
  local so
  so=$(ls "$IC"/python/infinicore/_infinicore*.so 2>/dev/null | head -1)
  if [ -z "$so" ]; then
    so=$(ls "$IC"/build/linux/x86_64/release/_infinicore*.so 2>/dev/null | head -1)
  fi
  if [ -z "$so" ]; then
    # 最后兜底：xmake 的 install 前缀因版本而异，全树找一遍（很慢，只在上面都落空时跑）
    so=$(find "$IC" -maxdepth 7 -name '_infinicore*.so' 2>/dev/null | head -1)
  fi
  if [ -z "$so" ]; then
    say "  !! 编译树里找不到 _infinicore*.so（$IC）"
    return 1
  fi

  # overlay 放哪：优先 /dev/shm（读写快），空间不够就退回 AFS
  local so_kb avail_kb
  so_kb=$(( $(stat -c %s "$so" 2>/dev/null || echo 0) / 1024 ))
  avail_kb=$(df -k /dev/shm 2>/dev/null | awk 'NR==2{print $4}')
  if [ -n "$avail_kb" ] && [ "$avail_kb" -lt $((so_kb * 4 + 65536)) ]; then
    say "  /dev/shm 余量 $((avail_kb/1024))MB 放不下 overlay（.so 约 $((so_kb/1024))MB），改放 AFS"
    OVL=$AFS/ic_py
  fi
  if ! mkdir -p "$OVL" 2>/dev/null || ! touch "$OVL/.wtest" 2>/dev/null; then
    OVL=$AFS/ic_py
    mkdir -p "$OVL" 2>/dev/null || { say "  !! overlay 目录不可写：$OVL"; return 1; }
  fi
  rm -f "$OVL/.wtest" 2>/dev/null

  local fp; fp=$(stat -c '%s-%Y' "$so" 2>/dev/null || echo unknown)
  if [ "$(cat "$OVL/.fp" 2>/dev/null)" != "$fp" ] || [ ! -f "$OVL/infinicore/__init__.py" ]; then
    say "  组装 infinicore overlay（包目录只读，在可写副本上补 .so）"
    rm -rf "$OVL/infinicore.tmp" 2>/dev/null
    if cp -a "$SRC_IC/python/infinicore" "$OVL/infinicore.tmp" 2>/dev/null; then
      rm -rf "$OVL/infinicore" 2>/dev/null
      mv -f "$OVL/infinicore.tmp" "$OVL/infinicore" 2>/dev/null || { rm -rf "$OVL/infinicore.tmp"; return 1; }
      echo "$fp" > "$OVL/.fp"
    else
      rm -rf "$OVL/infinicore.tmp" 2>/dev/null
      mkdir -p "$OVL/infinicore/lib"
      say "  !! 包目录复制失败，只补 lib/"
    fi
  fi
  mkdir -p "$OVL/infinicore/lib"
  cp -f "$so" "$OVL/infinicore/lib/" || return 1
  export INFINICORE_OVERLAY=$OVL
  say "  _infinicore.so 就位：$OVL/infinicore/lib/$(basename "$so")  $(ls -lh "$so" 2>/dev/null | awk '{print $5}')"

  # 能写回包目录就写（老路径的保底），写不进去不算错
  if mkdir -p "$SRC_IC/python/infinicore/lib" 2>/dev/null; then
    cp -f "$so" "$SRC_IC/python/infinicore/lib/" 2>/dev/null \
      && cp -f "$so" "$SRC_IC/python/infinicore/" 2>/dev/null \
      && say "  同时写回包目录（双保险）"
  fi

  # 缺库要立刻看见，别等 Python 报 import 失败
  if command -v ldd >/dev/null 2>&1; then
    local miss
    miss=$(ldd "$so" 2>/dev/null | grep 'not found' | head -5)
    [ -n "$miss" ] && { echo "  !! .so 缺动态库（LD_LIBRARY_PATH 要补）："; echo "$miss" | sed 's/^/    /'; }
  fi
  return 0
}

has_dynsym() {   # $1=so 文件  $2=符号名
  if command -v nm >/dev/null 2>&1; then
    nm -D --defined-only "$1" 2>/dev/null | awk '{print $NF}' | grep -qx "$2"
  elif command -v readelf >/dev/null 2>&1; then
    readelf -Ws "$1" 2>/dev/null | awk '{print $8}' | grep -qx "$2"
  elif command -v strings >/dev/null 2>&1; then
    strings -a "$1" 2>/dev/null | grep -qx "$2"
  else
    # 没有 binutils 时，把 ELF 的符号串表按 \0 拆行再比对
    tr '\0' '\n' < "$1" 2>/dev/null | grep -qx "$2"
  fi
}

# 九齿生成的算子源码直接调 CUDA driver API（cuModuleLoadData / cuModuleGetFunction
# / cuDeviceGetAttribute / cuLaunchKernel / cuCtxGetId ...），但它内嵌的 kernel 其实是
# ELF(AMDGPU) code object —— 这套代码本来就是给 HIP 用的，只是符号名写成了 cu*。
# 于是 libinfiniop.so 留下 9 个未定义符号；其中 cuCtxGetId 连标准 CUDA 都没有，
# DTK 的 libcuda 更不会提供（11-07 08:41 那单：8 个候选一个都不给）。
# 重编 infiniop 要 38 分钟，所以运行时补一层薄壳，把这 9 个符号全部转发到 HIP。

build_cuda_shim() {
  local dir=$AFS/cudashim
  local src=$dir/h3_cuda_shim.c
  local so=$dir/libh3cuda.so
  mkdir -p "$dir" 2>/dev/null || { say "  !! 建不了目录 $dir"; return 1; }

  cat > "$src" <<'CEOF'
/* h3_cuda_shim.c —— 把九齿生成算子用到的 CUDA driver API 转发到 HIP。
 * 不包含任何头文件：不依赖 cuda.h 也不依赖 hip 头，任何 cc 都能编。 */
typedef int CUresult;
typedef void *CUmodule;
typedef void *CUfunction;
typedef void *CUcontext;
typedef void *CUstream;

extern int hipModuleLoadData(void **module, const void *image);
extern int hipModuleGetFunction(void **func, void *module, const char *name);
extern int hipModuleUnload(void *module);
extern int hipModuleLaunchKernel(void *f,
                                 unsigned int gx, unsigned int gy, unsigned int gz,
                                 unsigned int bx, unsigned int by, unsigned int bz,
                                 unsigned int shared, void *stream,
                                 void **params, void **extra);
extern int hipDeviceGetAttribute(int *pi, int attr, int dev);
extern int hipGetDevice(int *dev);
extern int hipFuncSetCacheConfig(void *func, int config);
extern int hipFuncSetAttribute(void *func, int attr, int value);
extern const char *hipGetErrorString(int err);

CUresult cuInit(unsigned int flags) { (void)flags; return 0; }

CUresult cuModuleLoadData(CUmodule *mod, const void *image) {
  return hipModuleLoadData((void **)mod, image);
}

CUresult cuModuleGetFunction(CUfunction *fn, CUmodule mod, const char *name) {
  return hipModuleGetFunction((void **)fn, mod, name);
}

CUresult cuModuleUnload(CUmodule mod) { return hipModuleUnload(mod); }

CUresult cuLaunchKernel(CUfunction f,
                        unsigned int gx, unsigned int gy, unsigned int gz,
                        unsigned int bx, unsigned int by, unsigned int bz,
                        unsigned int shared, CUstream stream,
                        void **params, void **extra) {
  return hipModuleLaunchKernel(f, gx, gy, gz, bx, by, bz, shared, stream, params, extra);
}

CUresult cuDeviceGetAttribute(int *pi, int attr, int dev) {
  int v = -1;
  if (!pi) return 1;
  if (hipDeviceGetAttribute(&v, attr, dev) == 0 && v > 0) { *pi = v; return 0; }
  /* CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN：DCU 上按 64 KB 兜底 */
  *pi = (attr == 97) ? 65536 : 0;
  return 0;
}

CUresult cuGetErrorString(int code, const char **str) {
  const char *s = hipGetErrorString(code);
  if (str) *str = s ? s : "unknown";
  return 0;
}

/* 缓存 / 动态共享内存偏好只是 hint，失败也不该让主流程 exit */
CUresult cuFuncSetCacheConfig(CUfunction f, int config) {
  (void)hipFuncSetCacheConfig(f, config);
  return 0;
}

CUresult cuFuncSetAttribute(CUfunction f, int attr, int value) {
  (void)hipFuncSetAttribute(f, attr, value);
  return 0;
}

/* 非标准 API（Triton 拿它当 kernel cache 的 key）。按当前设备给一个稳定 id。 */
CUresult cuCtxGetId(CUcontext ctx, unsigned long long *id) {
  int dev = 0;
  (void)ctx;
  if (hipGetDevice(&dev) != 0 || dev < 0) dev = 0;
  if (id) *id = (unsigned long long)dev + 1ULL;
  return 0;
}
CEOF

  local cc=""
  local c
  for c in gcc cc clang "$DTK_ROOT/bin/clang" "$DTK_ROOT/bin/clang++"; do
    [ -n "$c" ] || continue
    if command -v "$c" >/dev/null 2>&1; then cc=$c; break; fi
  done
  if [ -z "$cc" ]; then
    say "  !! 找不到 C 编译器（gcc / cc / clang 都没有），shim 编不出来"
    return 1
  fi

  # 找得到 libamdhip64 就显式链上（记 DT_NEEDED + rpath）；
  # 找不到就先留未定义符号，运行时由进程里已加载的 HIP 解析
  local hipd="" hiplib="" f
  for f in "$DTK_ROOT"/lib/libamdhip64.so "$DTK_ROOT"/lib64/libamdhip64.so \
           "$DTK_ROOT"/hip/lib/libamdhip64.so "$DTK_ROOT"/hip/lib64/libamdhip64.so \
           /opt/hyhal/lib/libamdhip64.so /opt/hyhal/lib64/libamdhip64.so; do
    if [ -e "$f" ]; then hiplib=$f; hipd=$(dirname "$f"); break; fi
  done
  if [ -z "$hiplib" ]; then
    f=$(ls "$DTK_ROOT"/lib/libamdhip64.so.* 2>/dev/null | head -1)
    [ -n "$f" ] && { hiplib=$f; hipd=$(dirname "$f"); }
  fi
  if [ -z "$hiplib" ] && command -v ldconfig >/dev/null 2>&1; then
    f=$(ldconfig -p 2>/dev/null | awk '/libamdhip64\.so/{print $NF; exit}')
    [ -n "$f" ] && [ -e "$f" ] && { hiplib=$f; hipd=$(dirname "$f"); }
  fi

  local -a larg=()
  [ -n "$hiplib" ] && larg=("$hiplib" "-Wl,-rpath,$hipd")

  # 薄壳依赖的 9 个 HIP 入口，先确认这份运行时里真的都有
  if [ -n "$hiplib" ] && command -v nm >/dev/null 2>&1; then
    local m="" hs
    for hs in hipModuleLoadData hipModuleGetFunction hipModuleUnload hipModuleLaunchKernel \
              hipDeviceGetAttribute hipGetDevice hipFuncSetCacheConfig hipFuncSetAttribute \
              hipGetErrorString; do
      has_dynsym "$hiplib" "$hs" || m="$m $hs"
    done
    if [ -n "$m" ]; then
      say "  !! $(basename "$hiplib") 里没有这些 HIP 入口：$m"
    else
      say "  HIP 入口核对：9/9 都在 $(basename "$hiplib")"
    fi
  fi

  say "  编译 CUDA→HIP 薄壳（$(basename "$cc")）"
  if ! "$cc" -O2 -fPIC -shared -o "$so" "$src" ${larg[@]+"${larg[@]}"} 2>"$dir/cc.log"; then
    say "  !! shim 编译失败："
    tail -8 "$dir/cc.log" 2>/dev/null | sed 's/^/    /'
    return 1
  fi
  [ -s "$so" ] || { say "  !! shim 没产出（$so 为空）"; return 1; }
  H3_CUDA_SHIM=$so
  if [ -n "$hipd" ]; then
    say "  shim 就位：$so  $(ls -lh "$so" 2>/dev/null | awk '{print $5}')（链接 $hipd/libamdhip64）"
  else
    say "  shim 就位：$so  $(ls -lh "$so" 2>/dev/null | awk '{print $5}')（HIP 符号留待运行时解析）"
  fi

  # 两道便宜的自检：依赖库是否找得到、ld.so 能否真的把它加载起来
  if command -v ldd >/dev/null 2>&1; then
    local miss
    miss=$(ldd "$so" 2>/dev/null | grep 'not found' | head -4)
    if [ -n "$miss" ]; then
      say "  !! shim 缺依赖库："
      printf '%s\n' "$miss" | sed 's/^/    /'
    fi
  fi
  if [ -x /bin/true ]; then
    if LD_PRELOAD="$so" /bin/true 2>"$dir/preload.log"; then
      say "  预加载自检通过（ld.so 能加载，依赖齐）"
    else
      say "  !! 预加载自检失败："
      tail -4 "$dir/preload.log" 2>/dev/null | sed 's/^/    /'
    fi
  fi
  return 0
}

# 子脚本里那行 `[ -f "$PRELOAD" ] && export LD_PRELOAD=$PRELOAD` 会把我们的 shim
# 直接顶掉。包目录是上个作业 uid 的只读树（改不了），所以在可写目录生成一份打过
# 补丁的副本：shim 放最前、DTK libcuda 跟在后面；PYTHONPATH 改成追加，保证
# overlay（带 lib/_infinicore.so 的那份）优先于只读包目录。
patch_fl2va_launcher() {
  local src=$PKG/scripts/run_fl2va_direct.sh
  local dst=$AFS/ic_py/run_fl2va_patched.sh
  [ -f "$src" ] || return 1
  [ -n "${H3_CUDA_SHIM:-}" ] || return 1
  mkdir -p "$AFS/ic_py" 2>/dev/null || return 1

  local line out="" n=0
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in
      '[ -f "$PRELOAD" ] && export LD_PRELOAD=$PRELOAD')
        out+='H3_SHIM='"$H3_CUDA_SHIM"$'\n'
        out+='[ -f "$PRELOAD" ] || PRELOAD='$'\n'
        out+='export LD_PRELOAD="$H3_SHIM${PRELOAD:+:$PRELOAD}${LD_PRELOAD:+:$LD_PRELOAD}"'$'\n'
        n=$((n+1))
        ;;
      'export PYTHONPATH=$SS/InfiniCore/python:$SS/InfiniLM/python:$PYTHONPATH')
        out+='export PYTHONPATH="$PYTHONPATH:$SS/InfiniCore/python:$SS/InfiniLM/python"'$'\n'
        n=$((n+1))
        ;;
      *)
        out+=$line$'\n'
        ;;
    esac
  done < "$src"

  if [ $n -lt 2 ]; then
    say "  !! 启动器结构和预期不一致（只替换了 $n 处），不冒险，放弃打补丁"
    return 1
  fi
  printf '%s' "$out" > "$dst" 2>/dev/null || return 1
  chmod +x "$dst" 2>/dev/null
  if ! bash -n "$dst" 2>/dev/null; then
    say "  !! 补丁后的启动器语法不过，放弃"
    rm -f "$dst"
    return 1
  fi
  H3_LAUNCHER=$dst
  say "  启动器已打补丁：$dst（shim 前置 / PYTHONPATH 追加）"
  return 0
}

ensure_cuda_compat() {
  [ -n "${CUDA_COMPAT_OK:-}" ] && return 0

  # 1) 把 libinfiniop.so 里未定义的 cu* 符号列全（以实际符号表为准，不猜）
  local -a cuundef=()
  local s u
  if command -v nm >/dev/null 2>&1 && [ -f "$INFINI/lib/libinfiniop.so" ]; then
    u=$(nm -D --undefined-only "$INFINI/lib/libinfiniop.so" 2>/dev/null \
        | awk '{print $NF}' | grep -E '^cu[A-Z]' | sort -u)
    if [ -n "$u" ]; then
      while IFS= read -r s; do [ -n "$s" ] && cuundef+=("$s"); done <<< "$u"
    fi
  fi
  if [ ${#cuundef[@]} -gt 0 ]; then
    say "  libinfiniop.so 未定义的 CUDA 符号 ${#cuundef[@]} 个：$(printf '%s ' "${cuundef[@]}" | cut -c1-220)"
  fi

  # 2) 编薄壳
  if ! build_cuda_shim; then
    say "  !! 薄壳没建起来，import 仍会报 undefined symbol"
    return 1
  fi

  # 3) 逐个核对覆盖（缺哪个一眼看到，不用再猜）
  local miss=""
  for s in ${cuundef[@]+"${cuundef[@]}"}; do
    has_dynsym "$H3_CUDA_SHIM" "$s" || miss="$miss $s"
  done
  if [ -n "$miss" ]; then
    say "  !! 薄壳还缺这些符号（需要补实现）：$miss"
  elif [ ${#cuundef[@]} -gt 0 ]; then
    say "  符号核对：${#cuundef[@]}/${#cuundef[@]} 全覆盖"
  fi

  # 4) 预加载：只放我们的薄壳。DTK 的 libcuda.so.1 由子脚本自己按原样加上，
  #    顺序上正好排在薄壳后面（同名符号我们优先）
  H3_CUDA_PRELOAD="$H3_CUDA_SHIM"
  export H3_CUDA_PRELOAD

  # 5) Python 侧兜底：万一子脚本把 LD_PRELOAD 冲掉，解释器启动时也用
  #    RTLD_GLOBAL 把薄壳拉进全局符号表
  local ovl="${INFINICORE_OVERLAY:-$OVL}"
  if [ -d "$ovl" ]; then
    cat > "$ovl/sitecustomize.py" <<'PYEOF'
import ctypes, os
for _p in os.environ.get("H3_CUDA_PRELOAD", "").split(":"):
    if _p and os.path.exists(_p):
        try:
            ctypes.CDLL(_p, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass
PYEOF
    say "  Python 兜底：$ovl/sitecustomize.py"
  fi

  # 6) 打补丁启动器，保证 LD_PRELOAD 里薄壳始终在最前
  patch_fl2va_launcher || say "  警告：启动器没补丁成，只能靠 sitecustomize 兜底"

  export CUDA_COMPAT_OK=1
  return 0
}

do_build() {
  ensure_pkg
  ensure_deps
  ensure_xmake
  pick_ic
  cd "$IC" || die "进不去 $IC"

  # 上一个作业 32 并发编 infiniop 花了 2317 秒，而节点有 128 核；这里按内存一起放宽。
  # 每个 hipcc 模板实例化峰值约 3~4 GB，按 MemTotal/4 收口，避免 OOM 直接整轮白跑。
  local NC MEMG MJ
  NC=$(nproc 2>/dev/null || echo 8)
  MEMG=$(awk '/^MemTotal:/{printf "%d", $2/1048576}' /proc/meminfo 2>/dev/null)
  MJ=${MEMG:-0}
  local JOBS=${JOBS:-$NC}
  [ "$JOBS" -gt 48 ] 2>/dev/null && JOBS=48
  if [ "${MJ:-0}" -ge 16 ]; then
    MJ=$(( MJ / 4 ))
    [ "$JOBS" -gt "$MJ" ] 2>/dev/null && JOBS=$MJ
  fi
  [ "$JOBS" -lt 4 ] 2>/dev/null && JOBS=4
  say "编译并发 = $JOBS（nproc=$NC，内存 ${MEMG:-?} GB）"
  # ccache 诊断：上一轮 infiniop 冷编 2317 秒、这轮 202 秒，靠的就是它命中缓存。
  # 只读不改，免得把已经热起来的缓存目录换掉
  if command -v ccache >/dev/null 2>&1; then
    say "ccache = $(ccache --version 2>/dev/null | head -1)"
    echo "  ccache 目录 = $(ccache --get-config cache_dir 2>/dev/null || echo '<默认>')"
    ccache -s 2>/dev/null | grep -iE "cache size|hit rate|cache miss|hits" | head -4 | sed 's/^/    /'
  else
    echo "  ccache = 未安装（编译只能靠树里的对象复用）"
  fi

  export PYTHON=$(command -v python3)
  export PYTHONPATH="$IC/src:$IC/python:$PYLIBS"
  # xmake.lua 在 configure 阶段就用 INFINI_ROOT 定 set_installdir / add_linkdirs，
  # 不设会落到 $HOME/.infini，编 _infinicore 时链接不到 -linfiniop
  export INFINI_ROOT=$INFINI

  patch_build_paths
  ensure_nt_header

  say "1/4 九齿 blockDim 修正（幂等）"
  if [ -f "$PKG/patches/ninetoothed_blockdim_fix.py" ]; then
    python3 "$PKG/patches/ninetoothed_blockdim_fix.py" 2>&1 | tail -5
  fi

  say "2/4 xmake configure"
  CCS=""
  ldconfig -p 2>/dev/null | grep -q librccl && CCS="--ccl=y"
  [ -n "$CCS" ] && echo "  发现 RCCL，启用 $CCS" || echo "  未发现 RCCL，不启用 --ccl"
  # xmake 默认会去 GitHub 拉 xmake-repo（容器走代理被拦成 502）。
  # 触发点是 option("omp") 默认 true → cpu.lua 里 add_requires("openmp")；用 --omp=n 关掉，
  # 再把全局网络切 private 做双保险，整个 configure 就不碰网络。
  xmake g --network=private >/dev/null 2>&1 && echo "  xmake 全局网络已切 private（不联网拉包）"
  local CFG=$OUT/xmake_configure.log
  local XOPT="--hygon-dcu=y --ninetoothed=y --omp=n --cpu=y"
  xmake f -y $XOPT $CCS --cuda="$CUDA_COMPAT" -cv >"$CFG" 2>&1
  local crc=$?
  tail -25 "$CFG"
  # 有的 xmake 版本没有内建 --cuda 选项，认不出就退一步：项目自己的 hygon 分支会用 CUDA_HOME
  if [ $crc -ne 0 ] && grep -qiE "unknown option|invalid option|unrecognized|no such option" "$CFG"; then
    say "  --cuda 不被当前 xmake 接受，去掉它重试"
    xmake f -y $XOPT $CCS -cv >"$CFG" 2>&1
    crc=$?
    tail -25 "$CFG"
  fi
  if [ $crc -ne 0 ] && grep -qiE "xmake-repo|repository|502|proxy|clone" "$CFG"; then
    say "  !! 仍然在联网拉 xmake-repo：把 --omp=n 之外的 add_requires 也关掉，或本地预置 repository"
  fi
  [ $crc -eq 0 ] || die "xmake configure 失败（看上面的 Cuda SDK / 编译器报错）"

  # 阶段跳过：上一轮已经编好并装到 AFS 的部分不再重编（作业只有 50 分钟，
  # 重编一遍 infiniop 就要 38 分钟，等于永远走不到 _infinicore）
  local STAMP_IOP=$OUT/.stamp_infiniop STAMP_CORE=$OUT/.stamp_infinicore
  local need_iop=1 need_core=1
  if [ -f "$STAMP_IOP" ] && [ -f "$INFINI/lib/libinfiniop.so" ]; then need_iop=0; fi
  # 产物判据以编译树 $IC 为准（$SRC_IC 是上个作业 uid 的只读树，别拿它当判据）
  if ls "$IC"/python/infinicore/_infinicore*.so >/dev/null 2>&1; then
    # stamp 丢了但 .so 还在（11-07 那轮就是：编完但 stamp 没写），直接复用，别再花 280s configure
    if [ -f "$STAMP_CORE" ]; then
      need_core=0
    elif [ "${FORCE_BUILD:-0}" = "1" ]; then
      say "  FORCE_BUILD=1，忽略已有 _infinicore.so 重新编译"
    else
      say "4/4 发现已编译好的 _infinicore.so（无 stamp），直接复用；要强制重编请 FORCE_BUILD=1"
      touch "$STAMP_CORE" 2>/dev/null
      need_core=0
    fi
  fi

  if [ "$need_iop" = "0" ]; then
    say "3/4 跳过编译 infiniop（$INFINI/lib/libinfiniop.so 已装好，跨作业复用）"
  else
  say "3/4 编译 infiniop（-j$JOBS，首次约 40 分钟；增量只补没编完的）"
  local BLOG=$OUT/build_infiniop.log
  # 用 tee 实时打到日志：编译要跑几分钟，原来全写进文件，作业被 50 分钟上限杀时
  # 日志里只剩一行标题，看不出卡在哪
  xmake build -j"$JOBS" infiniop 2>&1 | tee "$BLOG"
  local brc=${PIPESTATUS[0]}
  if [ $brc -ne 0 ]; then
    echo "  --- 报错摘要（完整日志 $BLOG）---"
    grep -n -m8 -E "error:|fatal error:|undefined reference|cannot find -l" "$BLOG" | cut -c1-200 | sed 's/^/    /'
    if grep -q "undefined reference to .cu[A-Z]" "$BLOG"; then
      echo "    !! 九齿内核用了 CUDA driver API（cuModuleLoadData / cuLaunchKernel），DTK 兼容层没链上"
    fi
    die "infiniop 编译失败"
  fi
  # AFS 上旧文件被占时会报 "Not access because it is busy"（上一轮就栽在 addcmul.h 上），
  # 这类错重试一次基本就过；只要 libinfiniop.so 到位，头文件沿用上次装的也能编
  local ILOG=$OUT/install_infiniop.log ii=1 irc=1
  while [ $ii -le 3 ]; do
    xmake install -y -o "$INFINI" infiniop >"$ILOG" 2>&1
    irc=$?
    [ $irc -eq 0 ] && break
    if grep -qi "busy" "$ILOG"; then
      say "  安装报 EBUSY（AFS 旧文件被占），5 秒后重试 $ii/3"
      sleep 5
    else
      break
    fi
    ii=$((ii+1))
  done
  tail -6 "$ILOG"
  if [ $irc -ne 0 ]; then
    say "  !! infiniop 安装未完全成功（多为 include 复制被占），检查 libinfiniop.so 是否已就位"
  fi
  [ -f "$INFINI/lib/libinfiniop.so" ] && touch "$STAMP_IOP" 2>/dev/null
  fi   # end of need_iop

  if [ "$need_core" = "0" ]; then
    say "4/4 跳过编译 _infinicore（上一轮已生成并拷回包目录）"
  else
  say "4/4 编译 python 绑定 _infinicore"
  local PLOG=$OUT/build_infinicore.log
  xmake build -j"$JOBS" _infinicore 2>&1 | tee "$PLOG"
  local prc=${PIPESTATUS[0]}
  if [ $prc -ne 0 ]; then
    echo "  --- 报错摘要（完整日志 $PLOG）---"
    grep -n -m8 -E "error:|fatal error:|undefined reference|cannot find -l" "$PLOG" | cut -c1-200 | sed 's/^/    /'
    say "python 绑定编译失败（infiniop 已编好；增量树还在 $IC，重跑 build 会接着来）"
    return 1
  fi
  xmake install _infinicore 2>&1 | tail -10
  ls "$IC"/python/infinicore/_infinicore*.so >/dev/null 2>&1 && touch "$STAMP_CORE" 2>/dev/null
  fi   # end of need_core

  # 无论本轮编没编，都把 .so 摆到能被 import 的位置（跳过编译时也要建 overlay）
  ensure_infinicore_overlay || say "  !! overlay 组装失败，import infinicore 大概率还是失败"

  echo "  --- 产物核对 ---"
  ls -lh "$INFINI/lib/"*.so 2>/dev/null | head -5
  # 定点 ls，别 find 整棵只读树（在一次日志里 find+du 白花了 3 分钟）
  ls -lh "$IC"/python/infinicore/_infinicore*.so 2>/dev/null | head -3
  ls -lh "$SRC_IC"/python/infinicore/lib/_infinicore*.so 2>/dev/null | head -3
  ls -lh "$OVL"/infinicore/lib/_infinicore*.so 2>/dev/null | head -2
  [ -f "$INFINI/lib/libinfiniop.so" ] || die "没生成 libinfiniop.so"
  # 增量树留在 AFS 上，下轮直接接着编
  say "编译树 $IC（下轮增量复用）"
  say "build 完成"
}

# 只跑 import、不加载权重：这一关过了才值得花 50 分钟跑管线
# =============================================================================
# InfiniLM 的 python 扩展 _infinilm —— 九齿只编了 InfiniCore，这一层从来没编过
#   examples/minimax_h3.py → infinilm.cache → from infinilm.lib import _infinilm
#   xmake.lua 把 .so 装到 python/infinilm/，而 lib/__init__.py 只往
#   python/infinilm/lib/ 里找（和 _infinicore 同一个坑），所以两头都补。
# =============================================================================
ensure_infinilm() {
  [ -n "${INFINILM_OK:-}" ] && return 0
  local src=$PKG/src/InfiniLM
  local root=$AFS/il_root
  local ovl=${INFINICORE_OVERLAY:-$OVL}
  [ -d "$src/csrc" ] || { say "  !! 找不到 $src/csrc"; return 1; }
  [ -d "$ovl" ] || { say "  !! overlay 没建好，先跑 ensure_infinicore_overlay"; return 1; }

  # 1) 之前编过就直接摆到位（overlay 若在 /dev/shm，每轮要重新拷一次）
  local so
  so=$(ls "$root"/_infinilm*.so 2>/dev/null | head -1)
  if [ -n "$so" ] && [ -f "$OUT/.stamp_infinilm" ]; then
    cp -f "$so" "$ovl/" 2>/dev/null
    mkdir -p "$src/python/infinilm/lib" 2>/dev/null && cp -f "$so" "$src/python/infinilm/lib/" 2>/dev/null
    export INFINILM_OK=1 INFINILM_LIBDIR=$ovl
    say "  _infinilm.so 复用：$(basename "$so")  $(ls -lh "$so" 2>/dev/null | awk '{print $5}')"
    return 0
  fi

  # 2) 准备一个 InfiniLM 能直接用的 INFINI_ROOT：四个库 + 头文件
  #    （InfiniCore 的 install 只装了 infiniop/infinirt，cpp_api 和 infiniccl 留在 build 里）
  mkdir -p "$root/lib" "$root/include" 2>/dev/null || { say "  !! $root 不可写"; return 1; }
  local l f base item
  for l in libinfinicore_cpp_api.so libinfiniop.so libinfinirt.so libinfiniccl.so; do
    [ -f "$root/lib/$l" ] && continue
    if [ -f "$INFINI/lib/$l" ]; then cp -f "$INFINI/lib/$l" "$root/lib/" 2>/dev/null && continue; fi
    f=$(ls "$IC/build/linux/x86_64/release/$l" 2>/dev/null | head -1)
    if [ -z "$f" ]; then f=$(find "$IC/build" -maxdepth 4 -name "$l" 2>/dev/null | head -1); fi
    if [ -n "$f" ]; then cp -f "$f" "$root/lib/" 2>/dev/null; else say "  !! 缺 $l（链接 _infinilm 会失败）"; fi
  done
  for item in infinicore infiniop infinicore.h infinicore.hpp infiniop.h infinirt.h infiniccl.h; do
    [ -e "$root/include/$item" ] && continue
    for base in "$INFINI/include" "$IC/include"; do
      if [ -e "$base/$item" ]; then
        ln -sfn "$base/$item" "$root/include/$item" 2>/dev/null
        # 少数文件系统不让建软链，退化成拷贝（头文件很小）
        [ -e "$root/include/$item" ] || cp -a "$base/$item" "$root/include/" 2>/dev/null
        break
      fi
    done
    [ -e "$root/include/$item" ] || say "  !! 缺头文件 $item"
  done

  # 3) 编译目录：包目录可写就就地编（对象落在 AFS，下轮增量）；
  #    不可写才用软链树（源码只读、只让 build/ 落盘），避免再花 6 分钟拷 1752 个文件
  local work=$src
  if ! touch "$src/.wtest" 2>/dev/null; then
    work=$AFS/il_work
    mkdir -p "$work" 2>/dev/null
    for f in "$src"/*; do
      base=$(basename "$f")
      [ "$base" = build ] && continue
      ln -sfn "$f" "$work/$base" 2>/dev/null
    done
    # 软链建不起来（少数文件系统不支持）就老老实实拷一份源码树
    if [ ! -e "$work/xmake.lua" ]; then
      say "  软链不可用，改拷源码树（22MB / 1752 个文件，会慢几分钟）"
      cp -a "$src"/. "$work"/ 2>/dev/null
      rm -rf "$work/build" 2>/dev/null
    fi
    say "  包目录只读，改用软链编译树 $work"
  else
    rm -f "$src/.wtest" 2>/dev/null
  fi

  ensure_xmake
  local j=${IL_JOBS:-48}
  local pb=$PYLIBS/pybind11/include
  [ -d "$pb" ] || pb=$PYLIBS/pybind11/pybind11/include
  [ -d "$pb" ] || { say "  !! 缺 pybind11 头文件（$pb），先跑 setup"; return 1; }
  local pyinc
  pyinc=$(python3 -c "import sysconfig;print(sysconfig.get_paths()['include'])" 2>/dev/null)
  [ -n "$pyinc" ] && [ -d "$pyinc" ] || pyinc=/usr/local/include/python3.11

  local log=$OUT/build_infinilm_$(date +%H%M%S).log
  say "  编译 _infinilm（$work，$j 并发，日志 $(basename "$log")）"
  (
    cd "$work" || exit 1
    export INFINI_ROOT=$root
    export PYBIND11_INCLUDE=$pb
    export PYTHON_INCLUDE=$pyinc
    export XMAKE_GLOBALDIR=${XMAKE_GLOBALDIR:-$AFS/.xmake_global}
    mkdir -p "$XMAKE_GLOBALDIR" 2>/dev/null
    xmake f -y -p linux -a x86_64 2>&1
    exec xmake build -y -j"$j" _infinilm 2>&1
  ) > "$log" 2>&1 &
  local pid=$! p rc=0
  # 每 20 秒报一次进度：既不让日志一片空白，也不刷屏
  while kill -0 "$pid" 2>/dev/null; do
    sleep 20
    kill -0 "$pid" 2>/dev/null || break
    p=$(grep -oE '\[ *[0-9]+%\]' "$log" 2>/dev/null | tail -1)
    [ -n "$p" ] && say "    _infinilm 进度 $p"
  done
  wait "$pid"; rc=$?
  if [ $rc -ne 0 ]; then
    say "  !! _infinilm 编译失败 rc=$rc，最后 20 行："
    tail -20 "$log" 2>/dev/null | sed 's/^/    /'
    return 1
  fi

  so=$(ls "$work"/build/linux/x86_64/release/_infinilm*.so 2>/dev/null | head -1)
  [ -z "$so" ] && so=$(find "$work/build" -maxdepth 5 -name '_infinilm*.so' 2>/dev/null | head -1)
  if [ -z "$so" ]; then
    say "  !! 编完却找不到 _infinilm*.so，最后 20 行："
    tail -20 "$log" 2>/dev/null | sed 's/^/    /'
    return 1
  fi

  cp -f "$so" "$root/" 2>/dev/null
  cp -f "$so" "$ovl/" 2>/dev/null || say "  !! 拷进 overlay 失败：$ovl"
  mkdir -p "$src/python/infinilm/lib" 2>/dev/null && cp -f "$so" "$src/python/infinilm/lib/" 2>/dev/null
  touch "$OUT/.stamp_infinilm" 2>/dev/null
  export INFINILM_OK=1 INFINILM_LIBDIR=$ovl
  say "  _infinilm.so 就位：$ovl/$(basename "$so")  $(ls -lh "$so" 2>/dev/null | awk '{print $5}')"
  return 0
}

preflight_import() {
  local py=python3
  command -v python3 >/dev/null 2>&1 || py=python
  say "  预检：import infinicore + _infinicore + _infinilm（不加载权重）"
  local out
  out=$(LD_PRELOAD="${H3_CUDA_PRELOAD:-}${LD_PRELOAD:+:$LD_PRELOAD}" "$py" - 2>&1 <<'PYEOF'
import infinicore
print("infinicore OK:", infinicore.__file__)
from infinicore.lib import _infinicore
print("_infinicore OK:", _infinicore.__file__)
from infinilm.lib import _infinilm
print("_infinilm OK:", _infinilm.__file__)
PYEOF
)
  local rc=$?
  if [ $rc -eq 0 ]; then
    printf '%s\n' "$out" | tail -4 | sed 's/^/    /'
  else
    say "  !! 预检失败，根因就在下面几行："
    printf '%s\n' "$out" | tail -6 | sed 's/^/    /'
  fi
  return $rc
}

do_run() {
  local tag=$1
  [ -f "$PKG/scripts/run_fl2va_direct.sh" ] || die "缺 run_fl2va_direct.sh"
  [ -n "$MODEL_ROOT" ] || die "找不到权重 model_index.json，检查 $SHARED/MiniMax-H3"
  [ -f "$INFINI/lib/libinfiniop.so" ] || say "警告：libinfiniop.so 还没编译，import infinicore 大概率失败（先跑 build）"
  ensure_first_frame

  export PKG SS INFINI_ROOT=$INFINI
  export MODEL_ROOT FIRST_FRAME OUTPUT_DIR=$OUT
  export GPUS=$(seq -s, 0 $((WANT_GPU - 1)))
  # .so 必须在 infinicore/lib/ 下才能被 from infinicore.lib import _infinicore 找到；
  # 包目录只读，用 overlay 兜住（单独跑 smoke 没走 build 时也在这里补一次）
  ensure_infinicore_overlay || say "警告：infinicore overlay 没建起来，import 大概率失败"
  ensure_cuda_compat || say "警告：CUDA→HIP 薄壳没建好，import 仍会报 undefined symbol"
  ensure_infinilm || say "警告：_infinilm 没编出来，import infinilm.cache 会失败"
  local pyic="$PKG/src/InfiniCore/python"
  [ -n "$INFINICORE_OVERLAY" ] && pyic="$INFINICORE_OVERLAY:$pyic"
  # overlay 放最前：既保证用的是新编的 infinicore，也让它的 sitecustomize.py 先生效
  export PYTHONPATH="$pyic:$PYLIBS:$PKG/src/InfiniLM/python"
  export LD_LIBRARY_PATH="$INFINI/lib:$DTK_ROOT/lib:/opt/hyhal/lib:/opt/hyhal/lib64:$LD_LIBRARY_PATH"
  # DTK 的 CUDA 兼容层（libcudart / libcuda stub）在 cuda-12/lib64
  local cudalib
  for cudalib in "$DTK_ROOT"/cuda/*/lib64 "$DTK_ROOT"/cuda/*/targets/x86_64-linux/lib; do
    [ -d "$cudalib" ] && LD_LIBRARY_PATH="$cudalib:$LD_LIBRARY_PATH"
  done
  # libinfinicore_cpp_api.so / libinfiniccl.so 只编在 build 目录里，没 install 到 infini/lib
  [ -d "$IC/build/linux/x86_64/release" ] && LD_LIBRARY_PATH="$IC/build/linux/x86_64/release:$LD_LIBRARY_PATH"
  # _infinilm.so 链接的是 $AFS/il_root/lib 里那四个库
  [ -d "$AFS/il_root/lib" ] && LD_LIBRARY_PATH="$AFS/il_root/lib:$LD_LIBRARY_PATH"
  [ -n "$CUDA_COMPAT" ] && [ -d "$CUDA_COMPAT/lib64" ] && LD_LIBRARY_PATH="$CUDA_COMPAT/lib64:$LD_LIBRARY_PATH"
  export LD_LIBRARY_PATH

  if [ "$tag" = "smoke" ]; then export SMOKE=1; fi

  # 便宜的自检：只 import，不加载权重。挂了 200 秒内就看到根因，不用等整条管线
  preflight_import || say "  预检没过，仍继续启动完整管线（便于对比完整报错）"

  local launcher=$PKG/scripts/run_fl2va_direct.sh
  local patched=0
  if [ -n "${H3_LAUNCHER:-}" ] && [ -f "$H3_LAUNCHER" ]; then launcher=$H3_LAUNCHER; patched=1; fi
  say "启动 $tag：GPUS=$GPUS  MODEL_ROOT=$MODEL_ROOT"
  say "  启动器 = $launcher"
  if [ $patched -eq 1 ]; then
    # 补丁版自己会 export LD_PRELOAD（shim 最前），父进程不必再设，
    # 免得 launcher 里每条 ls/sed 都去加载一遍 HIP 运行时
    bash "$launcher"
  elif [ -n "${H3_CUDA_PRELOAD:-}" ]; then
    say "  只为这条命令预加载：$H3_CUDA_PRELOAD"
    LD_PRELOAD="$H3_CUDA_PRELOAD${LD_PRELOAD:+:$LD_PRELOAD}" bash "$launcher"
  else
    bash "$launcher"
  fi
  local rc=$?
  echo
  if [ $rc -eq 0 ]; then
    say "$tag 成功，产物："
    ls -lh "$OUT"/*.mp4 2>/dev/null | tail -3
  else
    say "$tag 失败 rc=$rc"
    echo "  常见原因与对策："
    echo "   · heads 不能被卡数整除 → 减少卡数：full 2"
    echo "   · import infinicore 失败 → 先跑 build"
    echo "   · 显存不足 → 降 num_frames"
  fi
  return $rc
}

# ---------- 分发 ----------
case "$MODE" in
  check)  do_check ;;
  setup)  do_setup ;;
  build)  do_build ;;
  smoke)  do_setup; do_run smoke ;;
  full)   ensure_deps; do_run full ;;
  retry2|retry4)
          ensure_deps
          WANT_GPU=2
          if [ "$MODE" = "retry4" ] && [ "$VIS" -ge 4 ]; then WANT_GPU=4; fi
          do_run smoke ;;
  all|localall)
          if [ "$MODE" = "localall" ]; then
            export FORCE_LOCAL=1
            say "强制本机展开编译树（不复用 AFS 上的 ic_work）"
          fi
          do_setup
          if do_build; then
            if [ "$SECONDS" -gt "${BUDGET:-2400}" ]; then
              say "已用 ${SECONDS}s，超过 ${BUDGET:-2400}s 预算，剩下的时间不够跑推理，本轮到此为止"
              say "下一轮再提交一次 all：已编好的阶段会跳过，编译从断点继续"
            else
              do_run smoke
            fi
          fi
          ;;
  *)      echo "用法: run.sh {check|setup|build|smoke|full|all|localall|retry2} [卡数]" ;;
esac

RC=$?
hr
say "结束 rc=$RC  日志 $LOG"
echo "视频在 $OUT ；报告在 $AFS"
exit $RC
