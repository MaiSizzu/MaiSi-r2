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
#
# 13:11 那单暴露的最后一个坑：Qwen3VLVideoProcessor 硬依赖 torchvision，缺了它
# 整个 Qwen3VLProcessor 都构造不出来 → components.processor = None → encoders.py
# 里 processor.image_processor(...) 直接崩。而 image processor 本身有纯 PIL 后端，
# keyframes 完全够用。解法见 §5.5（给 video_processor 挂空壳）与 §5.5 的预检。
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

# ---------- 临时目录 ----------
# 平台的 TMPDIR 指向 /dev/shm，而这台机器 /dev/shm 只有几十 MB。
# 48 路并行编译的临时文件会把它占满，之后 HIP 初始化就拿不到设备了
# （11-07 10:33 那单：+20s 时 torch 还能看到 8 卡，编完 21 分钟后变成 0 卡）。
# 所以编译/解包的临时文件一律挪到 AFS，/dev/shm 只留给运行时自己用。
if [ -z "${H3_TMP_FIXED:-}" ]; then
  ORIG_TMP=${TMPDIR:-/tmp}
  if mkdir -p "$AFS/tmp" 2>/dev/null && [ -w "$AFS/tmp" ]; then
    export TMPDIR=$AFS/tmp
  fi
  export H3_TMP_FIXED=1 ORIG_TMP
fi

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
# 纯 Python 小依赖补充（janus / xxhash），见 ensure_pydeps：离线 pylibs 里没有它们
PYDEPS=$AFS/pydeps
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

  # 这段校验要真 import torch+diffusers+transformers，在 AFS 上实测 6~7 分钟。
  # 骨架没变就没必要每单重付：指纹只管「包 + python 版本 + 这段脚本版本」，
  # 包换了（.ready 变）或我改了校验逻辑（DEPS_VER 变）都会自动失效。
  local DFP=""
  DFP=$(printf '%s|%s|%s' "$(cat "$PYLIBS/.ready" 2>/dev/null)" \
        "$(python3 -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null)" \
        "deps2" | md5sum | cut -c1-16)
  if [ -f "$PYLIBS/.deps_ok" ] && [ "$(cat "$PYLIBS/.deps_ok" 2>/dev/null)" = "$DFP" ] \
     && [ -d "$PYLIBS/diffusers" ] && [ -d "$PYLIBS/transformers" ]; then
    say "校验依赖：指纹 $DFP 与上一单一致（已通过），跳过导入（省约 6 分钟）"
    DEPS_OK=1
    return 0
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
  if [ $rc -eq 0 ]; then
    say "依赖全部就绪"
    echo "$DFP" > "$PYLIBS/.deps_ok" 2>/dev/null || true
    DEPS_OK=1
    return 0
  fi
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
  PYTHONPATH="$PYDEPS:$PYLIBS:$PYTHONPATH" python3 - <<'PY'
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
for _m in ("janus", "xxhash"):
    try:
        _mod = __import__(_m)
        print("  %-9s =" % _m, getattr(_mod, "__version__", "?"), (getattr(_mod, "__file__", "") or "")[-46:])
    except Exception:
        print("  %-9s = 缺（smoke 时由 ensure_pydeps 补到 $PYDEPS）" % _m)
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


# TP>1 必须有 allreduce，但集群上编 InfiniCore 没开 --ccl：
# infiniccl_cuda.h 走 INFINICCL_DEVICE_API_NOOP(cuda)，infinicclCommInitAll 固定返回
# 5 = INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED（11:45 那单的 Error Code 5 就是它）。
# 开 --ccl 要 RCCL + nccl.h + 重编十几分钟，节点上也没有 rccl。
# 换条路：InfiniLM 的 TP 是单进程 + 每卡一个 RankWorker 线程，八张卡在同一地址空间，
# 所以用「线程栅栏 + hipMemcpy 汇聚」自己实现 allreduce，再用 LD_PRELOAD 覆盖同名符号。
build_ccl_shim() {
  local dir=$AFS/cudashim
  local src=$dir/h3_ccl.c
  local so=$dir/libh3ccl.so
  mkdir -p "$dir" 2>/dev/null || { say "  !! 建不了目录 $dir"; return 1; }

  cat > "$src" <<'H3CCLEOF'
/* ====================================================================
 * h3_ccl.c —— 自实现 InfiniCCL（线程栅栏 + hipMemcpy 汇聚）
 *
 * 背景：集群上编 InfiniCore 没开 --ccl，于是 infinicclCommInitAll 是
 *       INFINICCL_DEVICE_API_NOOP 的空壳，固定返回 5
 *       (INFINI_STATUS_DEVICE_TYPE_NOT_SUPPORTED)，TP=8 根本起不来。
 *       开 --ccl 需要 RCCL + nccl.h + 重编十几分钟，节点上也没有 rccl。
 *
 * 思路：InfiniLM 的 TP 是「单进程 + 每张卡一个 RankWorker 线程」，
 *       八张卡在同一地址空间、同一进程里并发跑 forward。所以集合通信
 *       不需要网络，只要一条线程栅栏 + hipMemcpy 就够了：
 *         各 rank 把自己的数据 D2H 到 host staging → 一个线程归约求和
 *         → H2D 写回所有 rank 的 recvbuf。语义与 ncclAllReduce 一致：
 *         阻塞、返回时 recvbuf 已就绪。
 *
 * 用 LD_PRELOAD 覆盖 libinfiniccl.so 里的同名符号（PLT 解析，先加载者优先）。
 * ==================================================================== */
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef _WIN32
#include <windows.h>
typedef CRITICAL_SECTION h3_mx;
typedef CONDITION_VARIABLE h3_cv;
#define H3_MX_INIT(m) InitializeCriticalSection(m)
#define H3_MX_LOCK(m) EnterCriticalSection(m)
#define H3_MX_UNLOCK(m) LeaveCriticalSection(m)
#define H3_CV_INIT(c) InitializeConditionVariable(c)
#define H3_CV_WAIT(c, m) do { while (!SleepConditionVariableCS((c), (m), INFINITE)) {} } while (0)
#define H3_CV_BCAST(c) WakeAllConditionVariable(c)
static long h3_xchg(volatile long *p, long v) { return InterlockedExchange(p, v); }
static void h3_yield(void) { SwitchToThread(); }
#else
#include <pthread.h>
#include <sched.h>
typedef pthread_mutex_t h3_mx;
typedef pthread_cond_t h3_cv;
#define H3_MX_INIT(m) pthread_mutex_init((m), NULL)
#define H3_MX_LOCK(m) pthread_mutex_lock(m)
#define H3_MX_UNLOCK(m) pthread_mutex_unlock(m)
#define H3_CV_INIT(c) pthread_cond_init((c), NULL)
#define H3_CV_WAIT(c, m) pthread_cond_wait((c), (m))
#define H3_CV_BCAST(c) pthread_cond_broadcast(c)
static long h3_xchg(volatile long *p, long v) { return __atomic_exchange_n(p, v, __ATOMIC_ACQ_REL); }
static void h3_yield(void) { sched_yield(); }
#endif

/* 只声明用到的 HIP 入口，不依赖 hip 头文件（编译环境干净） */
extern int hipSetDevice(int dev);
extern int hipDeviceSynchronize(void);
extern int hipStreamSynchronize(void *stream);
extern int hipMemcpy(void *dst, const void *src, size_t n, int kind);
extern int hipGetDevice(int *dev);

#define H3_MAXR 16
#define H3_CHUNK 262144ULL /* 每次搬运的元素上限，控制 host 缓冲在几 MB */

#define H3_D2H 2
#define H3_H2D 1

/* 日志：直接写 stderr，不经 printf 缓冲 */
static void h3_log(const char *s) {
    fputs(s, stderr);
    fflush(stderr);
}

static char *h3_num(char *p, long v) {
    char t[24];
    int k = 0, i;
    unsigned long long u;
    if (v < 0) {
        *p++ = '-';
        u = (unsigned long long)(-v);
    } else {
        u = (unsigned long long)v;
    }
    do {
        t[k++] = (char)('0' + (int)(u % 10ULL));
        u /= 10ULL;
    } while (u && k < 22);
    for (i = k - 1; i >= 0; --i) *p++ = t[i];
    return p;
}

static void h3_say(const char *a, long v, const char *b) {
    char buf[256];
    char *p = buf;
    const char *q;
    for (q = a; q && *q && p < buf + 200; ++q) *p++ = *q;
    p = h3_num(p, v);
    for (q = b; q && *q && p < buf + 240; ++q) *p++ = *q;
    *p++ = '\n';
    *p = 0;
    h3_log(buf);
}

typedef struct {
    int n;                     /* rank 数 */
    int ids[H3_MAXR];          /* 各 rank 的 device id */
    h3_mx mu;
    h3_cv cv;
    unsigned long long gen;    /* 集合通信代次 */
    int arrive;                /* 本轮已到达的 rank 数 */
    size_t count[H3_MAXR];
    int dtype[H3_MAXR];
    void *sbuf[H3_MAXR];
    void *rbuf[H3_MAXR];
    int op[H3_MAXR];
    void *stage[H3_MAXR];      /* host 暂存，每 rank 一块 */
    size_t cap;                /* stage 容量（元素数） */
    size_t esz_cap;            /* 分配时的元素字节数 */
    unsigned long long seq;
    unsigned long long moved;  /* 累计搬运字节（用于报进度） */
} h3grp;

typedef struct {
    int device_type;
    int device_id;
    int rank;
    int world;
    h3grp *grp;
} h3comm;

static int h3_esz(int t) {
    switch (t) {
    case 1: case 2: case 3: case 7: return 1;              /* BYTE BOOL I8 U8 */
    case 4: case 8: case 12: case 19: return 2;           /* I16 U16 F16 BF16 */
    case 5: case 9: case 13: return 4;                    /* I32 U32 F32 */
    case 6: case 10: case 14: return 8;                   /* I64 U64 F64 */
    default: return 0;
    }
}

/* ---- 半精度/bf16 与 float 互换（无 F16C 依赖，纯整数位运算） ---- */
static float h3_h2f(unsigned short h) {
    unsigned int s = (h >> 15) & 1u, e = (h >> 10) & 0x1fu, m = h & 0x3ffu;
    union { unsigned int u; float f; } x;
    if (e == 0x1fu) {
        x.u = (s << 31) | 0x7f800000u | (m << 13);
        return x.f;
    }
    if (e == 0) {
        double d = (double)m / 16777216.0;
        return (float)(s ? -d : d);
    }
    x.u = ((e - 15u + 127u) << 23) | (m << 13);
    return (s ? -1.0f : 1.0f) * x.f;
}

static unsigned short h3_f2h(float f) {
    union { unsigned int u; float f; } x;
    unsigned int u, s, e, m, r, rem;
    int exp, shift;
    x.f = f;
    u = x.u;
    s = (u >> 31) & 1u;
    e = (u >> 23) & 0xffu;
    m = u & 0x7fffffu;
    if (e == 0xffu) {
        if (m == 0) return (unsigned short)((s << 15) | 0x7c00u);
        return (unsigned short)((s << 15) | 0x7e00u);
    }
    if (e == 0) return (unsigned short)(s << 15); /* 下溢到 0（够用） */
    exp = (int)e - 127 + 15;
    if (exp >= 0x1f) return (unsigned short)((s << 15) | 0x7c00u);
    if (exp <= 0) {
        if (exp < -10) return (unsigned short)(s << 15);
        m |= 0x800000u;
        shift = 14 - exp;
        r = m >> shift;
        rem = m & ((1u << shift) - 1u);
        if (rem > (1u << (shift - 1)) || (rem == (1u << (shift - 1)) && (r & 1u))) r += 1u;
        return (unsigned short)((s << 15) | r);
    }
    r = ((unsigned int)exp << 10) | (m >> 13);
    rem = m & 0x1fffu;
    if (rem > 0x1000u || (rem == 0x1000u && (r & 1u))) r += 1u;
    return (unsigned short)((s << 15) | r);
}

static float h3_bf2f(unsigned short v) {
    union { unsigned int u; float f; } x;
    x.u = ((unsigned int)v) << 16;
    return x.f;
}

static unsigned short h3_f2bf(float f) {
    union { unsigned int u; float f; } x;
    unsigned int u;
    x.f = f;
    u = x.u;
    if ((u & 0x7fffffffu) > 0x7f800000u) return (unsigned short)((u >> 16) | 0x0040u);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (unsigned short)(u >> 16);
}

/* 归约：把 W 份 stage 逐元素求和（AVG 再除以 W），结果写回 stage[0]。
 * 中间用 double 累加，避免 fp16 累加 8 次就掉精度；
 * 写回时按目标 dtype 打包（fp16/bf16 要转回位型，不能直接强转）。 */
static double h3_id_d(double a) { return a; }
static double h3_wr_f16(double a) { return (double)h3_f2h((float)a); }
static double h3_wr_bf16(double a) { return (double)h3_f2bf((float)a); }

static int h3_reduce(h3grp *g, size_t n, int dtype, int op) {
    int W = g->n, i;
    size_t j;
    const double sc = (op == 4) ? 1.0 / (double)(W > 0 ? W : 1) : 1.0;

#define H3_CASE(TY, RD, WR)                                          \
    do {                                                             \
        const TY *s_[H3_MAXR];                                       \
        TY *o_ = (TY *)g->stage[0];                                  \
        for (i = 0; i < W; ++i) s_[i] = (const TY *)g->stage[i];     \
        for (j = 0; j < n; ++j) {                                    \
            double a_ = 0.0;                                         \
            for (i = 0; i < W; ++i) a_ += RD((double)s_[i][j]);      \
            o_[j] = (TY)WR(a_ * sc);                                 \
        }                                                            \
    } while (0)

    switch (dtype) {
    case 12: H3_CASE(unsigned short, h3_h2f, h3_wr_f16); break;  /* F16 */
    case 19: H3_CASE(unsigned short, h3_bf2f, h3_wr_bf16); break; /* BF16 */
    case 13: H3_CASE(float, h3_id_d, h3_id_d); break;
    case 1: case 2: case 7: H3_CASE(unsigned char, h3_id_d, h3_id_d); break;
    case 3: H3_CASE(signed char, h3_id_d, h3_id_d); break;
    case 4: H3_CASE(short, h3_id_d, h3_id_d); break;
    case 8: H3_CASE(unsigned short, h3_id_d, h3_id_d); break;
    case 5: H3_CASE(int, h3_id_d, h3_id_d); break;
    case 9: H3_CASE(unsigned int, h3_id_d, h3_id_d); break;
    case 6: H3_CASE(long long, h3_id_d, h3_id_d); break;
    case 10: H3_CASE(unsigned long long, h3_id_d, h3_id_d); break;
    case 14: H3_CASE(double, h3_id_d, h3_id_d); break;
    default: return 10;
    }
    return 0;
}

#undef H3_CASE

/* stage 缓冲按「元素数×8 字节」分配：任何 dtype 都塞得下，且只增不减 */
static int h3_ensure(h3grp *g, size_t chunk) {
    int i;
    size_t bytes;
    if (g->cap >= chunk) return 0;
    bytes = chunk * 8u + 64u;
    for (i = 0; i < g->n; ++i) {
        if (g->stage[i]) free(g->stage[i]);
        g->stage[i] = malloc(bytes);
        if (!g->stage[i]) {
            g->cap = 0;
            h3_say("[h3ccl] !! 分配汇聚缓冲失败 chunk=", (long)chunk, "");
            return 1;
        }
    }
    /* 归约结果统一由 stage[0] 向各 rank 广播 */
    g->cap = chunk;
    return 0;
}

static int h3_collective(h3grp *g, int rank, size_t count, int dtype, int op,
                         void *sbuf, void *rbuf, void *stream, int kind) {
    size_t esz = (size_t)h3_esz(dtype);
    size_t chunk, off, n;
    int i;
    unsigned long long mygen;

    if (!esz) {
        h3_say("[h3ccl] !! 不支持的 dtype=", (long)dtype, "");
        return 10;
    }
    if (count == 0) return 0;

    /* 本卡算完，sendbuf 才是最终值 */
    if (stream) hipStreamSynchronize(stream);
    else hipDeviceSynchronize();

    chunk = count > (size_t)H3_CHUNK ? (size_t)H3_CHUNK : count;

    H3_MX_LOCK(&g->mu);

    mygen = g->gen;
    g->count[rank] = count;
    g->dtype[rank] = dtype;
    g->op[rank] = op;
    g->sbuf[rank] = sbuf;
    g->rbuf[rank] = rbuf;

    if (++g->arrive < g->n) {
        while (g->gen == mygen) H3_CV_WAIT(&g->cv, &g->mu);
        H3_MX_UNLOCK(&g->mu);
        return 0;
    }

    /* 最后一个到达的当 leader：所有 rank 的数据都已登记，独占做完归约 */
    g->arrive = 0;
    if (h3_ensure(g, chunk) != 0) {
        g->gen++;
        H3_CV_BCAST(&g->cv);
        H3_MX_UNLOCK(&g->mu);
        return 1;
    }

    for (off = 0; off < count; off += chunk) {
        int src = (kind == 0) ? 0 : (kind - 1); /* 归约结果是 stage[0]；广播只有 root 有效 */
        n = count - off;
        if (n > chunk) n = chunk;
        for (i = 0; i < g->n; ++i) {
            if (kind != 0 && i != src) continue;
            hipSetDevice(g->ids[i]);
            hipMemcpy(g->stage[i], (char *)g->sbuf[i] + off * esz, n * esz, H3_D2H);
        }
        if (kind == 0) {
            if (h3_reduce(g, n, dtype, op) != 0) break;
        }
        for (i = 0; i < g->n; ++i) {
            hipSetDevice(g->ids[i]);
            hipMemcpy((char *)g->rbuf[i] + off * esz, g->stage[src], n * esz, H3_H2D);
        }
        g->moved += (unsigned long long)(n * esz) * (unsigned long long)(g->n * 2);
    }
    if (g->n > 0) hipSetDevice(g->ids[rank]);

    g->seq++;
    if (g->seq <= 3 || (g->seq % 500ULL) == 0) {
        char buf[256];
        char *p = buf;
        const char *q;
        for (q = "[h3ccl] allreduce #"; *q; ++q) *p++ = *q;
        p = h3_num(p, (long)g->seq);
        for (q = "  count="; *q; ++q) *p++ = *q;
        p = h3_num(p, (long)count);
        for (q = "  dtype="; *q; ++q) *p++ = *q;
        p = h3_num(p, (long)dtype);
        for (q = "  ranks="; *q; ++q) *p++ = *q;
        p = h3_num(p, (long)g->n);
        *p++ = '\n';
        *p = 0;
        h3_log(buf);
    }

    g->gen++;
    H3_CV_BCAST(&g->cv);
    H3_MX_UNLOCK(&g->mu);
    return 0;
}

static h3grp *h3_new_group(int n, const int *ids) {
    h3grp *g = (h3grp *)calloc(1, sizeof(h3grp));
    int i;
    if (!g) return NULL;
    g->n = n;
    for (i = 0; i < n; ++i) g->ids[i] = ids ? ids[i] : i;
    H3_MX_INIT(&g->mu);
    H3_CV_INIT(&g->cv);
    return g;
}

/* ---------------- 对外 C API（覆盖 libinfiniccl 同名符号） ---------------- */

int infinicclCommInitAll(int device_type, void **comms, int ndevice, const int *device_ids) {
    h3grp *g;
    h3comm *c;
    char buf[256];
    char *p;
    const char *q;
    int i;

    if (!comms || ndevice <= 0 || ndevice > H3_MAXR) {
        h3_say("[h3ccl] !! commInitAll 参数不对 ndevice=", (long)ndevice, "");
        return 3;
    }
    g = h3_new_group(ndevice, device_ids);
    if (!g) return 1;
    for (i = 0; i < ndevice; ++i) {
        c = (h3comm *)calloc(1, sizeof(h3comm));
        if (!c) return 1;
        c->device_type = device_type;
        c->device_id = g->ids[i];
        c->rank = i;
        c->world = ndevice;
        c->grp = g;
        comms[i] = (void *)c;
    }
    p = buf;
    for (q = "[h3ccl] commInitAll 自实现 n="; *q; ++q) *p++ = *q;
    p = h3_num(p, (long)ndevice);
    for (q = "  device_type="; *q; ++q) *p++ = *q;
    p = h3_num(p, (long)device_type);
    for (q = "  ids="; *q; ++q) *p++ = *q;
    for (i = 0; i < ndevice && p < buf + 230; ++i) {
        p = h3_num(p, (long)g->ids[i]);
        *p++ = ' ';
    }
    *p++ = '\n';
    *p = 0;
    h3_log(buf);
    return 0;
}

int infinicclGetCommName(void *comm, char *comm_name, size_t comm_name_size) {
    h3comm *c = (h3comm *)comm;
    char tmp[48];
    char *p = tmp;
    const char *q;
    size_t n;
    if (!c || !comm_name) return 4;
    for (q = "infiniccl-h3-"; *q; ++q) *p++ = *q;
    p = h3_num(p, (long)c->device_id);
    *p = 0;
    n = (size_t)(p - tmp);
    if (n + 1 > comm_name_size) return 3;
    memcpy(comm_name, tmp, n + 1);
    return 0;
}

int infinicclCommDestroy(void *comm) {
    if (!comm) return 4;
    /* group 由各 comm 共享，进程退出时统一回收 */
    free(comm);
    return 0;
}

int infinicclGroupStart(void *comm) {
    (void)comm;
    return 0;
}

int infinicclGroupEnd(void *comm) {
    (void)comm;
    return 0;
}

int infinicclAllReduce(void *sendbuf, void *recvbuf, size_t count, int datatype,
                       int op, void *comm, void *stream) {
    h3comm *c = (h3comm *)comm;
    if (!c || !c->grp) return 4;
    if (op != 0 && op != 4) {
        h3_say("[h3ccl] !! 只支持 SUM/AVG，收到 op=", (long)op, "");
        return 2;
    }
    return h3_collective(c->grp, c->rank, count, datatype, op, sendbuf, recvbuf, stream, 0);
}

int infinicclBroadcast(const void *sendbuf, void *recvbuf, size_t count, int datatype,
                       int root, void *comm, void *stream) {
    h3comm *c = (h3comm *)comm;
    if (!c || !c->grp) return 4;
    if (root < 0 || root >= c->grp->n) return 3;
    return h3_collective(c->grp, c->rank, count, datatype, 0, recvbuf, recvbuf, stream, root + 1);
}

int infinicclSend(const void *sendbuf, size_t count, int datatype, int peer,
                  void *comm, void *stream) {
    (void)sendbuf; (void)count; (void)datatype; (void)peer; (void)comm; (void)stream;
    h3_log("[h3ccl] !! Send 未实现（只在 pp>1 用到）\n");
    return 2;
}

int infinicclRecv(void *recvbuf, size_t count, int datatype, int peer,
                  void *comm, void *stream) {
    (void)recvbuf; (void)count; (void)datatype; (void)peer; (void)comm; (void)stream;
    h3_log("[h3ccl] !! Recv 未实现（只在 pp>1 用到）\n");
    return 2;
}

int infinicclAllGather(void *sendbuf, void *recvbuf, size_t send_count, int datatype,
                       void *comm, void *stream) {
    (void)sendbuf; (void)recvbuf; (void)send_count; (void)datatype;
    (void)comm; (void)stream;
    h3_log("[h3ccl] !! AllGather 未实现（本路径未用到）\n");
    return 2;
}

int infinicclAllGatherV(void *sendbuf, void *recvbuf, const size_t *recv_counts, int nranks,
                        int datatype, void *comm, void *stream) {
    (void)sendbuf; (void)recvbuf; (void)recv_counts; (void)nranks;
    (void)datatype; (void)comm; (void)stream;
    return 2;
}

int infinicclReduceScatter(void *sendbuf, void *recvbuf, size_t recv_count, int datatype,
                           int op, void *comm, void *stream) {
    (void)sendbuf; (void)recvbuf; (void)recv_count; (void)datatype;
    (void)op; (void)comm; (void)stream;
    h3_log("[h3ccl] !! ReduceScatter 未实现（本路径未用到）\n");
    return 2;
}

int infinicclReduceScatterV(void *sendbuf, void *recvbuf, const size_t *send_counts, int nranks,
                            int datatype, int op, void *comm, void *stream) {
    (void)sendbuf; (void)recvbuf; (void)send_counts; (void)nranks;
    (void)datatype; (void)op; (void)comm; (void)stream;
    return 2;
}

int infinicclGetUniqueId(void *unique_id) {
    char *p = (char *)unique_id;
    if (!p) return 4;
    memset(p, 0, 128);
    memcpy(p, "h3ccl-id", 8);
    return 0;
}

/* pp>1 才用；单进程内多线程 InitRank：按 unique_id 归组，等齐 nranks 再返回 */
static h3grp *g_reg[4];
static int g_reg_n[4];
static int g_reg_ready[4];
static int g_reg_cnt;
static volatile long g_reg_lock;

int infinicclCommInitRank(void **comm, int nranks, void *unique_id, int rank) {
    h3grp *g = NULL;
    h3comm *c;
    int slot = -1, i, dev = 0;

    if (!comm || nranks <= 0 || nranks > H3_MAXR) return 3;
    while (h3_xchg(&g_reg_lock, 1) != 0) h3_yield();
    for (i = 0; i < g_reg_cnt; ++i) {
        if (g_reg_n[i] == nranks) {
            slot = i;
            break;
        }
    }
    if (slot < 0) {
        int ids[H3_MAXR];
        slot = g_reg_cnt++;
        for (i = 0; i < nranks; ++i) ids[i] = i;
        g_reg[slot] = h3_new_group(nranks, ids);
        g_reg_n[slot] = nranks;
        g_reg_ready[slot] = 0;
    }
    g = g_reg[slot];
    h3_xchg(&g_reg_lock, 0);

    hipGetDevice(&dev);
    c = (h3comm *)calloc(1, sizeof(h3comm));
    if (!c) return 1;
    c->device_type = 8;
    c->device_id = dev;
    c->rank = rank;
    c->world = nranks;
    c->grp = g;

    H3_MX_LOCK(&g->mu);
    if (g->ids[rank] < 0) g->ids[rank] = dev;
    g_reg_ready[slot]++;
    if (g_reg_ready[slot] >= nranks) {
        H3_CV_BCAST(&g->cv);
    } else {
        while (g_reg_ready[slot] < nranks) H3_CV_WAIT(&g->cv, &g->mu);
    }
    H3_MX_UNLOCK(&g->mu);

    *comm = (void *)c;
    h3_say("[h3ccl] CommInitRank 自实现 nranks=", (long)nranks, "");
    return 0;
}

#if defined(__GNUC__) && !defined(_WIN32)
__attribute__((constructor)) static void h3_ccl_boot(void) {
    h3_log("[h3ccl] 自实现集合通信已加载（LD_PRELOAD 生效）\n");
}
#endif
H3CCLEOF

  local cc=""
  local c
  for c in gcc cc clang "$DTK_ROOT/bin/clang"; do
    [ -n "$c" ] || continue
    if command -v "$c" >/dev/null 2>&1; then cc=$c; break; fi
  done
  if [ -z "$cc" ]; then
    say "  !! 找不到 C 编译器，集合通信补不了（TP>1 会继续报 Error Code 5）"
    return 1
  fi

  local hipd="" hiplib="" f
  for f in "$DTK_ROOT"/lib/libamdhip64.so "$DTK_ROOT"/lib64/libamdhip64.so \
           "$DTK_ROOT"/hip/lib/libamdhip64.so "$DTK_ROOT"/hip/lib64/libamdhip64.so \
           /opt/hyhal/lib/libamdhip64.so /opt/hyhal/lib64/libamdhip64.so; do
    [ -e "$f" ] && { hiplib=$f; hipd=$(dirname "$f"); break; }
  done
  if [ -z "$hiplib" ] && command -v ldconfig >/dev/null 2>&1; then
    f=$(ldconfig -p 2>/dev/null | awk '/libamdhip64\.so/{print $NF; exit}')
    [ -n "$f" ] && [ -e "$f" ] && { hiplib=$f; hipd=$(dirname "$f"); }
  fi

  say "  编译集合通信薄壳（$(basename "$cc")，自实现 allreduce 不依赖 RCCL）"
  local -a larg=(-pthread)
  [ -n "$hiplib" ] && larg+=("$hiplib" "-Wl,-rpath,$hipd")
  if ! "$cc" -O2 -fPIC -shared -o "$so" "$src" ${larg[@]+"${larg[@]}"} 2>"$dir/ccl_cc.log"; then
    say "  !! 集合通信薄壳编译失败："
    tail -12 "$dir/ccl_cc.log" 2>/dev/null | sed 's/^/    /'
    return 1
  fi
  [ -s "$so" ] || { say "  !! 集合通信薄壳没产出（$so 为空）"; return 1; }

  # 少一个符号就会在运行时炸，这里逐个核对
  local want="infinicclCommInitAll infinicclAllReduce infinicclCommDestroy infinicclGetCommName infinicclGroupStart infinicclGroupEnd"
  local miss="" s2
  for s2 in $want; do has_dynsym "$so" "$s2" || miss="$miss $s2"; done
  if [ -n "$miss" ]; then
    say "  !! 集合通信薄壳缺符号：$miss"
    return 1
  fi
  H3_CCL_SHIM=$so
  if [ -n "$hipd" ]; then
    say "  集合通信薄壳就位：$so  $(ls -lh "$so" 2>/dev/null | awk '{print $5}')（链 $hipd/libamdhip64）"
  else
    say "  集合通信薄壳就位：$so  $(ls -lh "$so" 2>/dev/null | awk '{print $5}')（HIP 符号留待运行时解析）"
  fi

  if LD_PRELOAD="$so" /bin/true 2>"$dir/ccl_preload.log"; then
    say "  预加载自检通过（ld.so 能加载）"
  else
    say "  !! 集合通信薄壳预加载失败："
    tail -4 "$dir/ccl_preload.log" 2>/dev/null | sed 's/^/    /'
    return 1
  fi
  # 原生实现在哪：libinfiniccl.so 里（动态，最好）还是被静态链进了别的 .so
  if [ -f "$INFINI/lib/libinfiniccl.so" ] && has_dynsym "$INFINI/lib/libinfiniccl.so" infinicclCommInitAll; then
    say "  原生 infinicclCommInitAll 在 libinfiniccl.so 里 → 预加载可以覆盖它"
  else
    say "  提示：没在 libinfiniccl.so 里找到原生符号，可能被静态链进 _infinilm.so（预加载同样能覆盖）"
  fi
  return 0
}

# =============================================================================
# 5.5 processor 兜底（torchvision 缺失 → video_processor 加载失败 → processor = None）
# =============================================================================
# 13:11 那单的经验：8 卡 TP 起来了、73G 权重也加载完了，最后死在这一行——
#     Qwen3VLVideoProcessor requires the Torchvision library but it was not found
#     AttributeError: 'NoneType' object has no attribute 'image_processor'
# 根因不在 image processor：transformers 给 qwen3_vl 配了两套后端，
#     {"torchvision": "Qwen2VLImageProcessor", "pil": "Qwen2VLImageProcessorPil"}
# 没有 torchvision 时自动走 PIL 那套，keyframes 完全够用。卡住的是同级的
# video_processor —— 它被 define_import_structure 按文件名标成依赖 torchvision，
# 一缺就成 Dummy，于是整个 Qwen3VLProcessor 构造不出来。MiniMax-H3 只用图片，
# 所以给它挂个空壳顶上（详见脚本内的注释）。
write_proc_fix() {
  local dyn=$PYDEPS/dyn entry=$AFS/ic_py/h3_entry.py
  mkdir -p "$dyn" "$AFS/ic_py" 2>/dev/null || return 1
  # 用内容比对而不是版本号：先前写死 "procfix1"，补丁内容改了但 AFS 上还是老文件，
  # 结果新 run.sh 跑起来用的还是旧补丁。改成先写临时文件再 cmp，内容一变就换。
  local tmpd=$dyn/.h3_proc_fix.new tmpe=$AFS/ic_py/.h3_entry.new changed=0

  cat > "$tmpd" <<'PYEOF'
"""让 MiniMax-H3 的 processor 在缺少 torchvision 的容器里也能加载。

transformers 给 Qwen3-VL 的 image processor 配了两套后端：

    ("qwen3_vl", {"torchvision": "Qwen2VLImageProcessor",
                  "pil":        "Qwen2VLImageProcessorPil"})

没有 torchvision 时 AutoImageProcessor 自动挑 pil 那套（Qwen2VLImageProcessorPil，
PilBackend 是纯 PIL + numpy，一行 tvF 都不碰），MiniMax-H3 的 keyframes 走的正是它。

真正卡死的是一整条和 video_processor 有关的链：

1. `transformers/__init__.py` 里

       try:
           if not is_torchvision_available():
               raise OptionalDependencyNotAvailable()
       except OptionalDependencyNotAvailable:
           from .utils import dummy_torchvision_objects
       else:
           _import_structure["video_processing_utils"] = [...]

   于是 `transformers.VideoProcessorMixin` 是个 Dummy，hasattr 直接 False。

2. `base_file_requirements` 又按文件名把 `video_processing_*.py` 整类标成依赖
   (vision, torch, torchvision)，所以 `transformers.VIDEO_PROCESSOR_MAPPING` 也是
   Dummy。

3. `ProcessorMixin.check_argument_for_proper_class` 拿字符串 "VideoProcessorMixin"
   去 `get_possibly_dynamic_module()`：第一步 hasattr 落空（第 1 条），就退到
   lookup_locations 里翻 `*_MAPPING._extra_content`，翻到 VIDEO_PROCESSOR_MAPPING
   时直接

       AttributeError: type object 'VIDEO_PROCESSOR_MAPPING' has no attribute '_extra_content'

   整个 Qwen3VLProcessor 就此构造失败，components.processor 变 None，
   encoders.py:282 的 processor.image_processor(...) 跟着炸。

MiniMax-H3 全程只喂 keyframes（图片），从不走视频分支，所以这里做三件事：
  a. 把真·VideoProcessorMixin 塞回 transformers 顶层，让 hasattr 能命中；
  b. AutoVideoProcessor 加载失败时返回一个继承真类的空壳；
  c. 类型校验遇到空壳直接用真类判断，不再走那条会崩的查询链。

本模块 import 即生效，幂等。
"""

from __future__ import annotations

import logging

_LOG = logging.getLogger("h3.procfix")

_APPLIED = False
_VPM = None  # transformers.video_processing_utils.VideoProcessorMixin 的真类


def _null_video_processor_class():
    """造一个只占位的 video_processor 类（继承 VideoProcessorMixin 才能过类型校验）。"""
    global _VPM
    from transformers.video_processing_utils import VideoProcessorMixin

    _VPM = VideoProcessorMixin

    class NullVideoProcessor(VideoProcessorMixin):
        # Qwen2-VL / Qwen3-VL 的 patch 参数；processor 侧只读 merge_size
        merge_size = 2
        patch_size = 16
        temporal_patch_size = 2
        model_input_names: list[str] = []

        def __init__(self, **kwargs):
            # 故意不调 super().__init__：那套参数校验对空壳没有意义
            pass

        def preprocess(self, videos=None, **kwargs):
            raise RuntimeError(
                "[h3] 当前环境没有 torchvision，video_processor 只是占位；"
                "MiniMax-H3 只用 keyframes（走 image_processor），不该走到这里"
            )

        def to_dict(self):
            return {"video_processor_type": "NullVideoProcessor"}

        def save_pretrained(self, *args, **kwargs):
            return None

    return NullVideoProcessor


def _promote_video_processor_names():
    """把被 torchvision 挡在门外的 video_processing 名字补回 transformers 顶层。

    `get_possibly_dynamic_module()` 的第一步就是 hasattr(transformers, name)。把真对象
    写进模块 __dict__ 之后，这一步直接命中，不会再退到会崩的 lookup_locations。

    每个名字单独取，缺一个不影响别的（真实环境里 BaseVideoProcessor 是有的，但这里
    不赌——少补一个也只是少一层保险，不该让整条兜底链跟着废掉）。
    """
    import transformers

    aliases = {}
    for name in ("VideoProcessorMixin", "BaseVideoProcessor"):
        try:
            module = __import__("transformers.video_processing_utils", fromlist=[name])
            aliases[name] = getattr(module, name)
        except Exception:  # noqa: BLE001
            continue
    try:
        from transformers.models.auto.video_processing_auto import VIDEO_PROCESSOR_MAPPING

        aliases["VIDEO_PROCESSOR_MAPPING"] = VIDEO_PROCESSOR_MAPPING
    except Exception:  # noqa: BLE001
        pass

    names = []
    for name, obj in aliases.items():
        if transformers.__dict__.get(name) is obj:
            continue
        try:
            setattr(transformers, name, obj)
            names.append(name)
        except Exception:  # noqa: BLE001
            pass
    if names:
        _LOG.info("[h3] 已补回 transformers 顶层：%s", ", ".join(names))


def _patch_auto_video_processor(null_cls):
    from transformers.models.auto.video_processing_auto import AutoVideoProcessor

    current = AutoVideoProcessor.from_pretrained
    if getattr(current, "_h3_patched", False):
        return
    original = getattr(current, "__func__", current)

    def from_pretrained(cls, *args, **kwargs):
        try:
            return original(cls, *args, **kwargs)
        except Exception as exc:  # noqa: BLE001 —— 这里就是要吞掉，否则整个 processor 都没了
            head = str(exc).strip().splitlines()[0][:160] if str(exc).strip() else ""
            _LOG.warning(
                "[h3] video_processor 加载失败，改挂空壳（不影响 keyframes）：%s: %s",
                type(exc).__name__,
                head,
            )
            return null_cls()

    from_pretrained._h3_patched = True
    AutoVideoProcessor.from_pretrained = classmethod(from_pretrained)


def _patch_argument_check():
    """类型校验遇到 video_processor 直接用真类判断，绕开会崩的查询链（见模块说明 3）。"""
    global _VPM
    from transformers.processing_utils import ProcessorMixin

    if _VPM is None:
        # 空壳类那步万一没成，这里自己再取一次，别让校验补丁跟着废
        try:
            from transformers.video_processing_utils import VideoProcessorMixin

            _VPM = VideoProcessorMixin
        except Exception:  # noqa: BLE001
            pass

    current = ProcessorMixin.check_argument_for_proper_class
    if getattr(current, "_h3_patched", False):
        return
    original = current

    def check_argument_for_proper_class(self, argument_name, argument):
        if argument is None:
            return None
        if _VPM is not None and isinstance(argument, _VPM):
            return _VPM
        return original(self, argument_name, argument)

    check_argument_for_proper_class._h3_patched = True
    ProcessorMixin.check_argument_for_proper_class = check_argument_for_proper_class


def apply() -> bool:
    """每一步独立成败：任何一步挂了，剩下的照装，绝不整体放弃。"""
    global _APPLIED
    if _APPLIED:
        return True

    try:
        import transformers  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        print(f"[h3] 没装上 processor 兜底：import transformers 失败 {type(exc).__name__}: {exc}", flush=True)
        return False

    def step(label, fn):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            print(f"[h3] {label}没装上：{type(exc).__name__}: {exc}", flush=True)
            return None

    null_cls = step("空壳类", _null_video_processor_class)
    step("顶层名字补全", _promote_video_processor_names)
    if null_cls is not None:
        step("AutoVideoProcessor 兜底", lambda: _patch_auto_video_processor(null_cls))
    step("类型校验兜底", _patch_argument_check)

    installed = null_cls is not None and getattr(_VPM, "__name__", "") == "VideoProcessorMixin"
    _APPLIED = installed
    if installed:
        print("[h3] processor 兜底已装：video_processor 挂空壳，image_processor 走 PIL 后端", flush=True)
    else:
        print("[h3] processor 兜底只装上一部分，processor 可能仍然起不来", flush=True)
    return installed


apply()
PYEOF

  cat > "$tmpe" <<'PYEOF'
"""examples/minimax_h3.py 的薄包装：先给 transformers 装好 processor 兜底，再原样跑入口。

FL2VA 的入口是 `python examples/minimax_h3.py fl2va --model-path=...`，而补丁必须赶在
AutoProcessor 加载之前生效。命令行里塞 `-c` 会打乱 argv[0] 和相对路径，所以用一个
runpy 包装：argv 原样传给入口脚本，`__name__ == "__main__"` 也保持成立。
"""

import os
import runpy
import sys

try:
    import h3_proc_fix  # noqa: F401  —— import 即生效
except Exception as exc:  # noqa: BLE001 —— 兜底装不上也要让主流程继续，别拦路
    print(f"[h3] processor 兜底没装上：{type(exc).__name__}: {exc}", flush=True)

_script = os.environ.get("H3_ENTRY_SCRIPT") or os.path.join(
    os.getcwd(), "examples", "minimax_h3.py"
)
if not os.path.isfile(_script):
    sys.exit(f"[h3] 找不到入口脚本：{_script}（cwd={os.getcwd()}）")

# 直接 `python examples/minimax_h3.py` 时 sys.path[0] 是 examples/，改走 runpy 后
# 会变成这个中转脚本所在的目录；补回来，行为跟原启动方式一致
_here = os.path.dirname(os.path.abspath(_script))
if _here not in sys.path:
    sys.path.insert(0, _here)

sys.argv[0] = _script
runpy.run_path(_script, run_name="__main__")
PYEOF

  # 内容一致就不动（省得每单都刷新 mtime）；不一致说明补丁升级了，换掉
  if cmp -s "$tmpd" "$dyn/h3_proc_fix.py" 2>/dev/null; then
    rm -f "$tmpd"
  else
    mv -f "$tmpd" "$dyn/h3_proc_fix.py" 2>/dev/null && changed=1
  fi
  if cmp -s "$tmpe" "$entry" 2>/dev/null; then
    rm -f "$tmpe"
  else
    mv -f "$tmpe" "$entry" 2>/dev/null && changed=1
  fi

  [ -s "$dyn/h3_proc_fix.py" ] && [ -s "$entry" ] || { say "  !! processor 兜底脚本没写出来"; return 1; }
  H3_ENTRY_PY=$entry
  export H3_ENTRY_PY
  if [ $changed -eq 1 ]; then
    say "  processor 兜底已更新：$dyn/h3_proc_fix.py + $entry"
  else
    say "  processor 兜底已是最新（内容未变）"
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

  # entry 中转脚本必须先就位，下面替换 python 调用行时才有东西可指
  write_proc_fix || say "  !! processor 兜底脚本没备好，这一单仍会在 processor 处失败"

  local line out="" n=0 pf=0
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in
      '[ -f "$PRELOAD" ] && export LD_PRELOAD=$PRELOAD')
        out+='H3_SHIM='"${H3_CUDA_PRELOAD:-$H3_CUDA_SHIM}"$'\n'
        if [ "${H3_DROP_LIBCUDA:-0}" = 1 ]; then
          # 实测：预加载 DTK 的 libcuda 会让 hip 看不到卡，这种情况只留薄壳
          out+='PRELOAD='$'\n'
        else
          out+='[ -f "$PRELOAD" ] || PRELOAD='$'\n'
        fi
        out+='export LD_PRELOAD="$H3_SHIM${PRELOAD:+:$PRELOAD}${LD_PRELOAD:+:$LD_PRELOAD}"'$'\n'
        n=$((n+1))
        ;;
      'export PYTHONPATH=$SS/InfiniCore/python:$SS/InfiniLM/python:$PYTHONPATH')
        out+='export PYTHONPATH="$PYTHONPATH:$SS/InfiniCore/python:$SS/InfiniLM/python"'$'\n'
        n=$((n+1))
        ;;
      '"$PY" examples/minimax_h3.py fl2va \')
        # 换成 h3_entry.py 中转，好在 AutoProcessor 加载之前装上 processor 兜底
        out+='H3_ENTRY_SCRIPT="$PWD/examples/minimax_h3.py" "$PY" '"${H3_ENTRY_PY:-$AFS/ic_py/h3_entry.py}"$' fl2va \\'$'\n'
        n=$((n+1))
        pf=1
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
  if [ $pf -eq 0 ]; then
    say "  !! 启动器里没匹配到 python 调用行，processor 兜底挂不上（会退回 processor=None 的老问题）"
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

  # 4) 集合通信薄壳：TP>1 的 allreduce 靠它（InfiniCore 没开 --ccl，原生是空壳）
  local cclok=0
  if build_ccl_shim; then
    cclok=1
  else
    say "  !! 集合通信薄壳没建好：TP>1 会继续报 Error Code 5，只能退 TP=1（显存多半不够）"
  fi

  # 5) 预加载：只放我们的薄壳。DTK 的 libcuda.so.1 由子脚本自己按原样加上，
  #    顺序上正好排在薄壳后面（同名符号我们优先）
  if [ $cclok -eq 1 ]; then
    H3_CUDA_PRELOAD="$H3_CUDA_SHIM:$H3_CCL_SHIM"
  else
    H3_CUDA_PRELOAD="$H3_CUDA_SHIM"
  fi
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
# InfiniCore 的 install 只装了 infiniop/infinirt，cpp_api 和 infiniccl 留在 build 目录，
# 头文件也没装全。这里把它们补进 $INFINI，让编译和运行共用同一棵 INFINI_ROOT。
ensure_infini_root() {
  local iroot=$INFINI l f base item
  mkdir -p "$iroot/lib" "$iroot/include" 2>/dev/null || { say "  !! $iroot 不可写"; return 1; }
  for l in libinfinicore_cpp_api.so libinfiniop.so libinfinirt.so libinfiniccl.so; do
    [ -f "$iroot/lib/$l" ] && continue
    f=$(ls "$IC/build/linux/x86_64/release/$l" 2>/dev/null | head -1)
    [ -n "$f" ] || f=$(find "$IC/build" -maxdepth 4 -name "$l" 2>/dev/null | head -1)
    if [ -n "$f" ]; then cp -f "$f" "$iroot/lib/" 2>/dev/null; else say "  !! 缺 $l（链接 _infinilm 会失败）"; fi
  done
  for item in infinicore infiniop infinicore.h infinicore.hpp infiniop.h infinirt.h infiniccl.h; do
    [ -e "$iroot/include/$item" ] && continue
    for base in "$IC/include" "$INFINI/include"; do
      [ -e "$base/$item" ] || continue
      ln -sfn "$base/$item" "$iroot/include/$item" 2>/dev/null
      # 少数文件系统不让建软链，退化成拷贝（头文件很小）
      [ -e "$iroot/include/$item" ] || cp -a "$base/$item" "$iroot/include/" 2>/dev/null
      break
    done
    [ -e "$iroot/include/$item" ] || say "  !! 缺头文件 $item"
  done
  return 0
}

ensure_infinilm() {
  [ -n "${INFINILM_OK:-}" ] && return 0
  local src=$PKG/src/InfiniLM
  local root=$AFS/il_root     # 只用来存 _infinilm.so 成品
  local iroot=$INFINI         # 编译和运行共用同一个 INFINI_ROOT，不再另起一棵树
  local ovl=${INFINICORE_OVERLAY:-$OVL}
  [ -d "$src/csrc" ] || { say "  !! 找不到 $src/csrc"; return 1; }
  [ -d "$ovl" ] || { say "  !! overlay 没建好，先跑 ensure_infinicore_overlay"; return 1; }

  # 1) 库和头文件先备齐（复用旧 .so 时同样要做，所以放在复用判断之前）
  ensure_infini_root || say "  !! INFINI_ROOT 没备齐，链接/加载可能失败"

  # 2) 之前编过就直接摆到位（overlay 若在 /dev/shm，每轮要重新拷一次）
  local so
  so=$(ls "$root"/_infinilm*.so 2>/dev/null | head -1)
  if [ -n "$so" ] && [ -f "$OUT/.stamp_infinilm" ]; then
    cp -f "$so" "$ovl/" 2>/dev/null
    mkdir -p "$src/python/infinilm/lib" 2>/dev/null && cp -f "$so" "$src/python/infinilm/lib/" 2>/dev/null
    export INFINILM_OK=1 INFINILM_LIBDIR=$ovl
    say "  _infinilm.so 复用：$(basename "$so")  $(ls -lh "$so" 2>/dev/null | awk '{print $5}')"
    return 0
  fi

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
    export INFINI_ROOT=$iroot
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

# =============================================================================
# 卡在不在，决定这一单还值不值得跑
#   10:33 那单的教训：+20s 时 torch 还能看到 8 卡，编了 21 分钟之后 HIP 一张都看不到，
#   等于后面 40 分钟全白烧。所以先花 10 秒探一次，并把 /dev/shm 余量打出来
#   ——平台的 TMPDIR 就在 /dev/shm 上，并行编译最容易把它写满。
# =============================================================================
hip_count() {   # $1=要预加载的库（可空）；输出 数字 / errN / na
  local py=python3
  command -v python3 >/dev/null 2>&1 || py=python
  LD_PRELOAD=${1:-} "$py" - "$DTK_ROOT/lib/libamdhip64.so" "$DTK_ROOT/lib64/libamdhip64.so" \
      "$DTK_ROOT/hip/lib/libamdhip64.so" "$DTK_ROOT/hip/lib64/libamdhip64.so" 2>/dev/null <<'PYEOF'
import ctypes, sys
names = ["libamdhip64.so", "libamdhip64.so.6", "libamdhip64.so.5", "libamdhip64.so.4"]
names += [p for p in sys.argv[1:] if p]
lib = None
for nm in names:
    try:
        lib = ctypes.CDLL(nm)
        break
    except OSError:
        pass
if lib is None:
    print("na")
else:
    n = ctypes.c_int(-1)
    rc = lib.hipGetDeviceCount(ctypes.byref(n))
    print(n.value if rc == 0 else "err%d" % rc)
PYEOF
}

shm_free_mb() { df -k /dev/shm 2>/dev/null | awk 'NR==2{print int($4/1024)}'; }

# /dev/shm 在这台机器上很小（早期实测不足 200MB），HIP 初始化和 IPC 都要用它。
# 编译临时文件已经挪到 AFS，这里只清我们自己早几轮留下的残留，绝不碰在用的 overlay。
free_shm() {
  local d b
  for d in /dev/shm/songshan_pkg /dev/shm/songshan_tar; do
    [ -d "$d" ] || continue
    case "$PKG" in "$d"*) say "  $d 正在用，不清理"; continue ;; esac
    case "${OVL:-}${INFINICORE_OVERLAY:-}" in *"$d"*) say "  $d 正在用，不清理"; continue ;; esac
    b=$(du -sm "$d" 2>/dev/null | awk '{print $1}')
    rm -rf "$d" 2>/dev/null && say "  腾出 /dev/shm：清掉 $d（${b:-?}MB）"
  done
  say "  /dev/shm 余量 $(shm_free_mb)MB"
}

probe_dcu() {   # $1=plain|preload → 0 有卡 / 2 没卡 / 3 判断不了
  local mode=${1:-plain} pre="" n shm info
  [ "$mode" = preload ] && pre=${H3_CUDA_PRELOAD:-}
  shm=$(shm_free_mb)
  info="kfd=$([ -e /dev/kfd ] && echo y || echo n) dri=$([ -d /dev/dri ] && echo y || echo n)"
  n=$(hip_count "$pre")
  case "$n" in
    na)
      say "    HIP 探测($mode)：加载不到 libamdhip64，判断不了  /dev/shm ${shm:-?}MB  $info"
      return 3
      ;;
    err*)
      say "    HIP 探测($mode)：hipGetDeviceCount 返回 $n  /dev/shm ${shm:-?}MB  $info"
      return 2
      ;;
  esac
  if ! [ "$n" -ge 1 ] 2>/dev/null; then
    say "    HIP 探测($mode)：0 张卡  /dev/shm ${shm:-?}MB  $info"
    return 2
  fi
  say "    HIP 探测($mode)：$n 张卡  /dev/shm ${shm:-?}MB  $info"
  DCU_COUNT=$n
  return 0
}

# =============================================================================
# 5.5 纯 Python 小依赖：janus / xxhash
# =============================================================================
# 11:29 那一单断在 `import janus`（infinilm/llm/request.py 在第 13 行就导入了它）。
# 离线 pylibs 里没有 janus / xxhash，但对我们这条路径，它俩只是"import 得过去就行"：
#   · janus   —— 只有 LLM / AsyncLLMEngine 的异步队列会真用；FL2VA 走 multimodal runner
#   · xxhash  —— 只用来算前缀缓存的分块键（16 字节摘要），换 blake2b-128 语义等价
# 所以在 AFS 上放一份极小的等价实现，PYTHONPATH 前置；缺别的包时再加一轮联网补齐。
ensure_pydeps() {
  local pd=$PYDEPS
  mkdir -p "$pd" 2>/dev/null || { say "  !! $pd 不可写"; return 1; }

  local stamp="janus2-xxh1"
  if [ "$(cat "$pd/.stamp" 2>/dev/null)" != "$stamp" ] || [ ! -f "$pd/janus/__init__.py" ]; then
    say "  补纯 Python 小依赖 → $pd（janus 等价实现 + xxhash 等价实现）"
    mkdir -p "$pd/janus"
    cat > "$pd/janus/__init__.py" <<'PYEOF'
"""janus 的最小等价实现：一个 queue.Queue 同时暴露 sync_q / async_q。

InfiniLM 只用到 Queue() + sync_q.{put,get,put_nowait,get_nowait,qsize,task_done,empty}
+ async_q.{put,get,put_nowait,get_nowait,task_done,empty}，语义与 janus 1.x 一致：
两侧共用同一个底层队列，task_done / qsize 自然对齐。
"""

import asyncio
import queue as _queue

__version__ = "1.0.0+h3shim"


class _AsyncSide:
    """asyncio 一侧：阻塞操作丢到线程池，绝不卡事件循环。"""

    __slots__ = ("_q",)

    def __init__(self, q):
        self._q = q

    async def put(self, item):
        self._q.put(item)

    def put_nowait(self, item):
        return self._q.put_nowait(item)

    async def get(self):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._q.get)

    def get_nowait(self):
        try:
            return self._q.get_nowait()
        except _queue.Empty:
            raise asyncio.QueueEmpty from None

    def task_done(self):
        return self._q.task_done()

    async def join(self):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._q.join)

    def qsize(self):
        return self._q.qsize()

    def empty(self):
        return self._q.empty()

    def full(self):
        return self._q.full()


class Queue:
    def __init__(self, maxsize=0):
        self._q = _queue.Queue(maxsize)
        self._async_q = _AsyncSide(self._q)
        self._closed = False

    @property
    def sync_q(self):
        return self._q

    @property
    def async_q(self):
        return self._async_q

    @property
    def closed(self):
        return self._closed

    def close(self):
        self._closed = True

    def shutdown(self, immediate=False):
        self._closed = True

    async def aclose(self):
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
PYEOF

    cat > "$pd/xxhash.py" <<'PYEOF'
"""xxhash 的最小等价实现：只覆盖 InfiniLM 用到的 xxh3_128()。

前缀缓存的分块键只要求"稳定 + 128 位"，算法本身可替换，这里用 blake2b-128。
对外接口（update / digest / hexdigest / intdigest / copy / reset）与 xxhash 一致。
"""

import hashlib

__version__ = "0.0.0+h3shim"

_ALGO = "blake2b"


class _Hasher:
    def __init__(self, digest_size=8, seed=0, data=b""):
        self._size = digest_size
        self._seed = seed
        self._h = hashlib.blake2b(digest_size=digest_size)
        if data:
            self.update(data)

    def update(self, data):
        if isinstance(data, str):
            data = data.encode()
        elif isinstance(data, memoryview):
            data = data.tobytes()
        self._h.update(bytes(data))
        return self

    def digest(self):
        return self._h.digest()

    def hexdigest(self):
        return self._h.hexdigest()

    def intdigest(self):
        return int.from_bytes(self._h.digest(), "big")

    def copy(self):
        other = _Hasher(self._size, self._seed)
        other._h = self._h.copy()
        return other

    def reset(self):
        self._h = hashlib.blake2b(digest_size=self._size)


def _make(size, data=b"", *args, **kwargs):
    return _Hasher(size, data=data)


def xxh3_128(data=b"", *args, **kwargs):
    return _Hasher(16, data=data)


def xxh3_64(data=b"", *args, **kwargs):
    return _Hasher(8, data=data)


def xxh128(data=b"", *args, **kwargs):
    return _Hasher(16, data=data)


def xxh64(data=b"", *args, **kwargs):
    return _Hasher(8, data=data)


def xxh32(data=b"", *args, **kwargs):
    return _Hasher(4, data=data)


def xxh3_128_digest(data=b"", *args, **kwargs):
    return hashlib.blake2b(bytes(data), digest_size=16).digest()


def xxh3_64_digest(data=b"", *args, **kwargs):
    return hashlib.blake2b(bytes(data), digest_size=8).digest()


def xxh128_digest(data=b"", *args, **kwargs):
    return xxh3_128_digest(data)


def xxh64_digest(data=b"", *args, **kwargs):
    return xxh3_64_digest(data)


def xxh32_digest(data=b"", *args, **kwargs):
    return hashlib.blake2b(bytes(data), digest_size=4).digest()


def digest(data, digest_size=8, *args, **kwargs):
    return hashlib.blake2b(bytes(data), digest_size=digest_size).digest()


def h3_128(data, *args, **kwargs):
    return xxh3_128_digest(data)


class VERSION:
    VERSION = "0.0.0+h3shim"
PYEOF
    echo "$stamp" > "$pd/.stamp" 2>/dev/null
  fi

  # 自检：真把两个包 import 起来跑一遍最小动作，别只看文件在不在
  local py=python3
  command -v python3 >/dev/null 2>&1 || py=python
  local out
  out=$(PYTHONPATH="$pd${PYTHONPATH:+:$PYTHONPATH}" "$py" - 2>&1 <<'PYEOF'
import janus, xxhash
q = janus.Queue()
q.sync_q.put({"a": 1})
assert q.async_q.get_nowait() == {"a": 1}, "janus 两侧没共用同一个队列"
assert q.sync_q.qsize() == 0
h = xxhash.xxh3_128()
h.update(b"InfiniLM-prefix-cache-v1\0")
h.update(b"\0")
assert len(h.digest()) == 16, "xxhash 摘要不是 16 字节"
assert xxhash.xxh3_128(b"x").hexdigest() == xxhash.xxh3_128(b"x").hexdigest()
print("janus %s / xxhash %s" % (getattr(janus, "__version__", "?"), getattr(xxhash, "__version__", "?")))
PYEOF
)
  if [ $? -eq 0 ]; then
    say "  janus + xxhash 可用：$out"
    return 0
  fi
  say "  !! janus/xxhash 自检失败："
  printf '%s\n' "$out" | tail -8 | sed 's/^/    /'
  return 1
}

# =============================================================================
# 5.6 出片编码兜底
# =============================================================================
# 入口脚本最后一步是 diffusers 的 encode_video()，它内部 `import av`（PyAV）——
# 离线 pylibs 里没有 av，真跑起来会"生成完 40 分钟、最后一步写 mp4 才炸"。
# 所以先写好一份同签名的兜底实现（ffmpeg 命令行 → cv2 → PNG 序列），
# 再把 diffusers 里 `av = _import_av()` 改成 try/except 转过来。有 av 时一行不改。
write_media_fallback() {
  local pd=$PYDEPS
  mkdir -p "$pd" 2>/dev/null || return 1
  local stamp="media1"
  if [ "$(cat "$pd/.media_stamp" 2>/dev/null)" = "$stamp" ] && [ -s "$pd/media_fallback.py" ]; then
    return 0
  fi
  cat > "$pd/media_fallback.py" <<'PYEOF'
"""encode_video 的兜底实现（PyAV 不在时顶上）。

签名、入参格式与 diffusers.utils.export_utils.encode_video 完全一致：
  video: list[PIL.Image] | np.ndarray | torch.Tensor([T,H,W,C])
  audio: torch.Tensor / np.ndarray
出片优先级：ffmpeg 命令行（带音轨） → OpenCV mp4v（无音轨） → PNG 序列（最坏情况留证据）。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import wave

import numpy as np

__version__ = "h3fallback-1"


def _to_uint8_frames(video):
    """统一成 [T,H,W,3] uint8。"""
    if hasattr(video, "detach") or isinstance(video, np.ndarray):
        arr = video
        if hasattr(arr, "detach"):
            arr = arr.detach().to("cpu").float().numpy()
        arr = np.asarray(arr)
    else:
        frames = []
        for f in video:
            if hasattr(f, "convert"):
                frames.append(np.asarray(f.convert("RGB")))
            else:
                arr_f = f
                if hasattr(arr_f, "detach"):
                    arr_f = arr_f.detach().to("cpu").float().numpy()
                frames.append(np.asarray(arr_f))
        arr = np.stack(frames, 0)

    if arr.ndim != 4:
        raise ValueError("video 形状应为 [T,H,W,C]，实际 %r" % (arr.shape,))
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        lo = float(arr.min()) if arr.size else 0.0
        hi = float(arr.max()) if arr.size else 0.0
        if 0.0 <= lo and hi <= 1.001:
            arr = arr * 255.0
        arr = np.clip(np.rint(arr.astype(np.float32)), 0, 255).astype(np.uint8)
    return arr


def _write_wav(path, audio, sample_rate):
    a = audio
    if hasattr(a, "detach"):
        a = a.detach().to("cpu").float().numpy()
    a = np.asarray(a, dtype=np.float32)
    if a.ndim == 2 and a.shape[0] <= 8 and a.shape[0] < a.shape[1]:
        a = a.T                      # [ch, n] → [n, ch]
    if a.ndim == 2 and a.shape[1] == 1:
        a = a[:, 0]
    if a.size and float(np.max(np.abs(a))) > 1.5:
        a = a / 32768.0              # 已经是整型量纲
    a = np.clip(a, -1.0, 1.0)
    a = np.nan_to_num(a)
    with wave.open(path, "wb") as w:
        w.setnchannels(1 if a.ndim == 1 else int(a.shape[1]))
        w.setsampwidth(2)
        w.setframerate(int(sample_rate))
        w.writeframes((a * 32767.0).astype("<i2").tobytes())


def _ffmpeg_bin():
    ff = shutil.which("ffmpeg")
    if ff:
        return ff
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _encode_ffmpeg(frames, fps, out_path, wav):
    ff = _ffmpeg_bin()
    if not ff:
        return False
    t, h, w, _ = frames.shape
    if h % 2:
        frames, h = frames[:, : h - 1], h - 1
    if w % 2:
        frames, w = frames[:, :, : w - 1], w - 1
    cmd = [
        ff, "-y", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", "%dx%d" % (w, h), "-r", str(int(round(fps))), "-i", "-",
    ]
    if wav:
        cmd += ["-i", wav]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p"]
    if wav:
        cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest"]
    cmd += ["-movflags", "+faststart", out_path]

    print("[media] ffmpeg 出片：%d 帧 %dx%d @%sfps%s" % (t, w, h, fps, " + 音轨" if wav else ""))
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        for frame in frames:
            proc.stdin.write(frame.tobytes())
        proc.stdin.close()
    except (BrokenPipeError, OSError):
        pass
    err = proc.stderr.read().decode("utf-8", "ignore")
    rc = proc.wait()
    if rc != 0:
        print("[media] !! ffmpeg 失败 rc=%d: %s" % (rc, err.strip()[-400:]))
        return False
    return os.path.isfile(out_path) and os.path.getsize(out_path) > 0


def _encode_cv2(frames, fps, out_path):
    try:
        import cv2
    except Exception:
        return False
    h, w = frames.shape[1], frames.shape[2]
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    if not writer.isOpened():
        return False
    for frame in frames:
        writer.write(frame[:, :, ::-1])          # RGB → BGR
    writer.release()
    print("[media] cv2 出片（mp4v，无音轨）：%s" % out_path)
    return os.path.isfile(out_path) and os.path.getsize(out_path) > 0


def encode_video(video, fps, output_path, audio=None, audio_sample_rate=None, video_chunks_number=1):
    frames = _to_uint8_frames(video)
    print("[media] PyAV 不可用 → 走兜底出片：%d 帧 %dx%d" % (frames.shape[0], frames.shape[2], frames.shape[1]))
    wav = None
    if audio is not None:
        if not audio_sample_rate:
            print("[media] !! 给了音频但没有采样率，这一版不带音轨")
        else:
            try:
                tmpdir = tempfile.mkdtemp(prefix="h3aud_")
                wav = os.path.join(tmpdir, "audio.wav")
                _write_wav(wav, audio, audio_sample_rate)
            except Exception as exc:      # 音频坏了也别把画面一起赔掉
                print("[media] !! 写 wav 失败（%s），改为无声出片" % exc)
                wav = None
    if _encode_ffmpeg(frames, fps, output_path, wav):
        print("[media] 出片完成：%s" % output_path)
        return
    if _encode_cv2(frames, fps, output_path):
        print("[media] 出片完成（无音轨）：%s" % output_path)
        return
    png_dir = os.path.splitext(output_path)[0] + "_frames"
    os.makedirs(png_dir, exist_ok=True)
    from PIL import Image

    for i, frame in enumerate(frames):
        Image.fromarray(frame).save(os.path.join(png_dir, "f%05d.png" % i))
    print("[media] !! 既没有 ffmpeg 也没有 cv2，PNG 序列已存：%s（%d 帧）" % (png_dir, frames.shape[0]))
PYEOF
  echo "$stamp" > "$pd/.media_stamp" 2>/dev/null
  return 0
}

# 把 diffusers 里 `av = _import_av()` 换成 try/except：没有 av 时转兜底实现
patch_encode_video() {
  local f=$PYLIBS/diffusers/utils/export_utils.py
  [ -f "$f" ] || { say "  !! 找不到 $f，出片兜底没法挂钩"; return 1; }
  grep -q "media_fallback" "$f" && return 0
  local py=python3
  command -v python3 >/dev/null 2>&1 || py=python
  "$py" - "$f" <<'PYEOF'
import io
import sys

path = sys.argv[1]
src = io.open(path, encoding="utf-8").read()
old = "    av = _import_av()\n"
new = (
    "    try:\n"
    "        av = _import_av()\n"
    "    except ImportError:\n"
    "        from media_fallback import encode_video as _h3_fallback_encode_video\n"
    "        return _h3_fallback_encode_video(\n"
    "            video, fps, output_path, audio, audio_sample_rate, video_chunks_number\n"
    "        )\n"
)
if old not in src:
    print("  !! export_utils.py 里没找到 'av = _import_av()'，跳过")
    sys.exit(1)
io.open(path, "w", encoding="utf-8").write(src.replace(old, new, 1))
print("  diffusers.encode_video 已挂上兜底：%s" % path)
PYEOF
}

ensure_media_encoder() {
  write_media_fallback || { say "  !! 兜底出片模块没写出来"; return 1; }
  local py=python3
  command -v python3 >/dev/null 2>&1 || py=python

  local avout
  avout=$(PYTHONPATH="$PYDEPS/dyn:$PYDEPS" "$py" -c "import av; print(av.__version__)" 2>&1)
  if [ $? -eq 0 ]; then
    say "  出片编码器 = PyAV $avout（diffusers 原生路径，不需要兜底）"
    return 0
  fi

  local ff cv
  ff=$(command -v ffmpeg 2>/dev/null || true)
  cv=$(PYTHONPATH="$PYDEPS/dyn:$PYDEPS" "$py" -c "import cv2; print(cv2.__version__)" 2>/dev/null | tail -1)
  say "  出片编码器 = PyAV 缺失，走兜底：ffmpeg=${ff:-<无>}  cv2=${cv:-<无>}"

  # 有网就顺手装上：先 av（原生路径，带音轨），不行退 imageio-ffmpeg（自带 ffmpeg 可执行文件）
  if [ -z "$ff" ] && [ -z "$cv" ]; then
    if net_ok; then
      if pyfetch_install av; then
        avout=$(PYTHONPATH="$PYDEPS/dyn:$PYDEPS" "$py" -c "import av; print(av.__version__)" 2>/dev/null)
        if [ -n "$avout" ]; then
          say "  PyAV $avout 可用（wheel 解压装上的），走原生出片"
          return 0
        fi
      fi
      say "  PyAV 没成，退一步装 imageio-ffmpeg（自带 ffmpeg 二进制）"
      if pyfetch_install imageio-ffmpeg; then
        ff=$(PYTHONPATH="$PYDEPS/dyn:$PYDEPS" "$py" -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())" 2>/dev/null | tail -1)
        [ -n "$ff" ] && say "  兜底 ffmpeg = $ff" && { patch_encode_video || true; return 0; }
      fi
    else
      say "  !! 这台机器连不上 PyPI（离线），只能靠自带实现"
    fi
    say "  !! 没有 av / ffmpeg / cv2 —— 最后只能存 PNG 序列，报告里要说明"
  fi
  patch_encode_video || say "  !! 兜底没挂上，出片那一步仍会报 PyAV 缺失"
  return 0
}

# 依赖表：把这条链真正会 import 的顶层包一次列全，别一轮一轮试错
dep_table() {
  local py=python3
  command -v python3 >/dev/null 2>&1 || py=python
  PYTHONPATH="$PYDEPS/dyn:$PYDEPS${PYTHONPATH:+:$PYTHONPATH}" "$py" - 2>&1 <<'PYEOF'
import importlib.util

# (分组, 模块名, 是否必需, pip 包名)
NEED = [
    ("核心", "janus", True, "janus"),
    ("核心", "xxhash", True, "xxhash"),
    ("基础", "numpy", True, "numpy"),
    ("基础", "torch", True, "torch"),
    ("基础", "PIL", True, "pillow"),
    ("模型", "safetensors", True, "safetensors"),
    ("模型", "transformers", True, "transformers"),
    ("模型", "tokenizers", True, "tokenizers"),
    ("模型", "diffusers", True, "diffusers"),
    ("模型", "accelerate", True, "accelerate"),
    ("模型", "huggingface_hub", True, "huggingface-hub"),
    ("工具", "tqdm", True, "tqdm"),
    ("工具", "yaml", True, "pyyaml"),
    ("工具", "regex", True, "regex"),
    ("工具", "psutil", False, "psutil"),
    ("出片", "av", False, "av"),
    ("出片", "cv2", False, "opencv-python-headless"),
    ("出片", "imageio_ffmpeg", False, "imageio-ffmpeg"),
    ("工具", "imageio", False, "imageio"),
]

missing = []
for grp, name, must, dist in NEED:
    try:
        spec = importlib.util.find_spec(name)
    except Exception:
        spec = None
    if spec is None:
        print("    %-4s %-18s %s" % (grp, name, "缺" if must else "缺（可选）"))
        if must:
            missing.append(dist)
    else:
        loc = spec.origin
        if not loc or loc == "namespace":
            locs = list(spec.submodule_search_locations or [])
            loc = locs[0] if locs else "?"
        print("    %-4s %-18s OK   %s" % (grp, name, str(loc)[:60]))
print("MISSING=" + " ".join(missing))
PYEOF
}

net_ok() {
  command -v curl >/dev/null 2>&1 || return 1
  curl -sS -m 8 -o /dev/null https://pypi.tuna.tsinghua.edu.cn/simple/ 2>/dev/null
}

# 直接从镜像下 wheel 解压。pip 在容器里会崩在 /etc/debian_version（PermissionError，
# 11:55 那单实测 rc=2），而 wheel 本身就是个 zip，解压到 PYTHONPATH 目录即可用。
pyfetch_install() {   # $@ = PyPI 包名
  local pd=$PYDEPS/dyn py=python3
  command -v python3 >/dev/null 2>&1 || py=python
  mkdir -p "$pd" 2>/dev/null || return 1
  say "  直接下 wheel 解压（绕开 pip）：$*"
  "$py" - "$pd" "$@" <<'PYEOF'
import os
import re
import subprocess
import sys
import zipfile

pd = sys.argv[1]
pkgs = sys.argv[2:]
BASE = "https://pypi.tuna.tsinghua.edu.cn"
CP = "cp%d%d" % sys.version_info[:2]


def fetch(url, out):
    """先 urllib，失败退 curl —— 容器里证书/代理经常只给 curl 配好了。"""
    try:
        import urllib.request

        with urllib.request.urlopen(url, timeout=120) as resp:
            data = resp.read()
        with open(out, "wb") as fh:
            fh.write(data)
        return os.path.getsize(out) > 0
    except Exception:
        pass
    try:
        rc = subprocess.call(["curl", "-sS", "-L", "-m", "300", "-o", out, url])
        return rc == 0 and os.path.getsize(out) > 0
    except Exception:
        return False


def pick(pkg):
    idx = os.path.join(pd, ".idx_%s.html" % pkg.replace("-", "_"))
    if not fetch("%s/simple/%s/" % (BASE, pkg), idx):
        print("  !! %s：索引页拿不到" % pkg)
        return None
    text = open(idx, encoding="utf-8", errors="ignore").read()
    from urllib.parse import urljoin

    best = None
    for href in re.findall(r'href="([^"#]+\.whl)', text):
        name = href.rsplit("/", 1)[-1]
        if "none-any" in name:
            score = 1
        elif CP in name and "x86_64" in name and ("manylinux" in name or "linux_x86_64" in name):
            score = 3
        else:
            continue
        url = urljoin("%s/simple/%s/" % (BASE, pkg), href)
        if best is None or score > best[0]:
            best = (score, name, url)
    if best is None:
        print("  !! %s：镜像里没有适配 %s 的 wheel" % (pkg, CP))
    return best


ok = 0
for pkg in pkgs:
    got = pick(pkg)
    if not got:
        continue
    _, name, url = got
    print("     %s" % name)
    whl = os.path.join(pd, ".dl.whl")
    if not fetch(url, whl):
        print("  !! %s 下载失败" % name)
        continue
    try:
        with zipfile.ZipFile(whl) as zf:
            zf.extractall(pd)
        ok += 1
        print("     解开 → %s" % pd)
    except Exception as exc:
        print("  !! %s 解压失败: %s" % (name, exc))
    finally:
        try:
            os.remove(whl)
        except OSError:
            pass
sys.exit(0 if ok else 1)
PYEOF
}

# 依赖表还有缺的才走这条路：先 8 秒探通网络，再下 wheel（pip 只作最后兜底）
pip_heal() {
  local pd=$PYDEPS/dyn py=python3
  command -v python3 >/dev/null 2>&1 || py=python
  mkdir -p "$pd" 2>/dev/null || return 1
  net_ok || { say "  !! 这台机器连不上 PyPI（离线），只能靠自带实现"; return 1; }

  # 先走 wheel 解压：不依赖 pip 的发行版探测，容器里更可靠
  if pyfetch_install "$@"; then
    say "  补齐成功（wheel 解压）"
    return 0
  fi

  "$py" -m pip --version >/dev/null 2>&1 || { say "  !! wheel 没成，也没有 pip 兜底"; return 1; }
  say "  wheel 没成，退 pip：$* → $pd"
  local tmo="" log rc
  command -v timeout >/dev/null 2>&1 && tmo="timeout 240"
  log=$($tmo "$py" -m pip install -q --no-input --no-cache-dir --disable-pip-version-check \
        --target "$pd" -i https://pypi.tuna.tsinghua.edu.cn/simple \
        --timeout 10 --retries 1 "$@" 2>&1)
  rc=$?
  if [ $rc -eq 0 ]; then
    say "  联网补齐成功"
    return 0
  fi
  say "  !! 联网补齐失败 rc=$rc"
  printf '%s\n' "$log" | tail -6 | sed 's/^/    /'
  return 1
}

preflight_import() {
  local py=python3
  command -v python3 >/dev/null 2>&1 || py=python
  export PYTHONPATH="$PYDEPS/dyn:$PYDEPS${PYTHONPATH:+:$PYTHONPATH}"

  # 第一步：依赖表。缺什么一次列全，避免"补一个又冒一个"
  say "  预检 1/3：第三方依赖体检（一次列全，不逐个试错）"
  local tbl miss
  tbl=$(dep_table)
  printf '%s\n' "$tbl" | sed '/^MISSING=/d' | sed 's/^/  /'
  miss=$(printf '%s\n' "$tbl" | sed -n 's/^MISSING=//p')
  if [ -n "$miss" ]; then
    say "  !! 还缺：$miss"
    if pip_heal $miss; then
      tbl=$(dep_table)
      printf '%s\n' "$tbl" | sed '/^MISSING=/d' | sed 's/^/  /'
      miss=$(printf '%s\n' "$tbl" | sed -n 's/^MISSING=//p')
    fi
    if [ -n "$miss" ]; then
      say "  !! 依赖补不齐（$miss）—— 先解决这几个包，别往下跑"
      return 3
    fi
  fi

  # 第二步：真 import。只 import 不加载权重，$tag 越大越接近真实路径
  say "  预检 2/3：import infinicore → _infinicore → _infinilm → infinilm → multimodal"
  local pl
  # 注意：不能用 `while read` 处理——最后一个条目没有换行符时 read 返回非 0，
  # 循环体不执行，列表会静默少一项（13:22 那单就把 libh3ccl.so 漏掉了，看着像没预加载）
  pl=$(printf '%s' "${H3_CUDA_PRELOAD:-}" | tr ':' '\n' | sed 's#.*/##' | tr '\n' ' ')
  say "    环境 INFINI_ROOT=${INFINI_ROOT:-?}  LD_PRELOAD=${pl:-<空>}  /dev/shm 余量 $(shm_free_mb)MB"
  local out
  out=$(LD_PRELOAD="${H3_CUDA_PRELOAD:-}${LD_PRELOAD:+:$LD_PRELOAD}" "$py" - 2>&1 <<'PYEOF'
import infinicore
print("infinicore OK:", infinicore.__file__)
from infinicore.lib import _infinicore
print("_infinicore OK:", _infinicore.__file__)
from infinilm.lib import _infinilm
print("_infinilm OK:", _infinilm.__file__)
import infinilm
print("infinilm OK（janus / xxhash 这一步通了）")
from infinilm.multimodal.minimax_h3 import MiniMaxH3TransformerRunner
print("multimodal.minimax_h3 OK（FL2VA 入口类可导入）")
PYEOF
)
  local rc=$?
  if [ $rc -ne 0 ]; then
    say "  !! 预检失败，根因就在下面几行（末尾 25 行）："
    printf '%s\n' "$out" | tail -25 | sed 's/^/    /'
    if printf '%s\n' "$out" | grep -q 'ModuleNotFoundError'; then
      say "     ↳ 是缺包：把这一行贴我，或确认 $PYDEPS/dyn 里有没有它"
    fi
    if printf '%s\n' "$out" | grep -q 'undefined symbol'; then
      say "     ↳ 是 cu*→HIP 薄壳没覆盖到，把符号名贴我"
    fi
    return $rc
  fi
  printf '%s\n' "$out" | tail -5 | sed 's/^/    /'
  return 0
}

# processor 自检单独一个函数：它只要几十秒，不该被「骨架指纹一致就跳过」省掉。
# 13:11 那单是加载完 73G 权重、一路跑进 encoders.py 才炸在 processor 上的
# （Qwen3VLVideoProcessor requires the Torchvision library），代价 40 分钟。
preflight_processor() {
  local py=python3
  command -v python3 >/dev/null 2>&1 || py=python
  # ensure_cuda_compat 有 CUDA_COMPAT_OK 快速返回，正常路径下 patch_fl2va_launcher
  # 已经把它写好了；这里再兜一次，免得某个分支漏写导致预检自己先崩
  [ -s "$PYDEPS/dyn/h3_proc_fix.py" ] || write_proc_fix >/dev/null 2>&1 || true
  [ -s "$PYDEPS/dyn/h3_proc_fix.py" ] || {
    say "  !! processor 兜底脚本没写出来（$PYDEPS/dyn/h3_proc_fix.py）"
    return 4
  }
  export PYTHONPATH="$PYDEPS/dyn:$PYDEPS${PYTHONPATH:+:$PYTHONPATH}"

  say "  预检 3/3：processor 加载（torchvision 缺失时 image_processor 应走 PIL 后端）"
  local gout grc
  gout=$(MODEL_ROOT="$MODEL_ROOT" LD_PRELOAD="${H3_CUDA_PRELOAD:-}${LD_PRELOAD:+:$LD_PRELOAD}" \
    "$py" - 2>&1 <<'PYEOF'
import glob
import os
import sys

import h3_proc_fix  # noqa: F401  —— 装 processor 兜底（幂等）

root = os.environ.get("MODEL_ROOT", "")
if not root or not os.path.isdir(root):
    print("!! MODEL_ROOT 不可用：%r" % root)
    sys.exit(1)

# processor 可能在根目录、也可能在 processor/ 或 diffusers/processor/；按配置文件找，不猜
found = None
level1 = [root] + sorted(p for p in glob.glob(os.path.join(root, "*")) if os.path.isdir(p))
for base in level1:
    for c in [base] + sorted(p for p in glob.glob(os.path.join(base, "*")) if os.path.isdir(p)):
        if os.path.isfile(os.path.join(c, "processor_config.json")) or os.path.isfile(
            os.path.join(c, "preprocessor_config.json")
        ):
            found = c
            break
    if found:
        break
if found is None:
    print("!! 没在 %s 下两层里找到 processor 配置，目录结构：" % root)
    for n in sorted(os.listdir(root))[:20]:
        print("      ", n)
    sys.exit(1)
print("processor 目录 = %s" % found)

from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

proc = Qwen3VLProcessor.from_pretrained(found)
ip = getattr(proc, "image_processor", None)
if ip is None:
    print("!! processor 起来了但 image_processor 是 None —— 兜底没生效")
    sys.exit(1)
print("processor OK: %s" % type(proc).__name__)
print(
    "  image_processor = %s  merge_size = %s  backend = %s"
    % (type(ip).__name__, getattr(ip, "merge_size", "?"), getattr(ip, "backend", "?"))
)
print("  video_processor = %s" % type(getattr(proc, "video_processor", None)).__name__)
print("  create_mm_token_type_ids = %s"
      % ("有" if callable(getattr(proc, "create_mm_token_type_ids", None)) else "缺"))

# encoders.py 就调这一句，keyframes 必须能过
import numpy as np
from PIL import Image

arr = (np.arange(64 * 64 * 3, dtype=np.uint8) % 251).reshape(64, 64, 3)
res = ip(images=[Image.fromarray(arr)], return_tensors="pt")
for k in ("pixel_values", "image_grid_thw"):
    if k not in res:
        print("!! image_processor 输出缺 %s，实际有：%s" % (k, list(res.keys())))
        sys.exit(1)
print("  图片预处理 OK: pixel_values=%s image_grid_thw=%s"
      % (tuple(res["pixel_values"].shape), res["image_grid_thw"].tolist()))
PYEOF
)
  grc=$?
  if [ $grc -eq 0 ]; then
    printf '%s\n' "$gout" | sed 's/^/    /'
    return 0
  fi
  say "  !! processor 自检失败（这次只花几十秒，不是 40 分钟），根因在下面："
  printf '%s\n' "$gout" | tail -20 | sed 's/^/    /'
  return $grc
}


# 骨架指纹：包 + python 版本 + 所有运行时 .so 的 size:mtime。
# 任何一处重编/换包都会变，变了就老老实实走完整预检。
skel_fp() {
  local s="v1|$(cat "$PYLIBS/.ready" 2>/dev/null)|$(python3 -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null)"
  local f
  for f in "$INFINI/lib/libinfiniop.so" "$INFINI/lib/libinfiniccl.so" \
           "$AFS/cudashim/libh3cuda.so" "$AFS/cudashim/libh3ccl.so" \
           "$PKG/src/InfiniLM/python/infinilm/lib/_infinilm.cpython-311-x86_64-linux-gnu.so"; do
    [ -f "$f" ] && s="$s|$(basename "$f"):$(stat -c '%s:%Y' "$f" 2>/dev/null)"
  done
  for f in "$AFS"/ic_py/infinicore/lib/_infinicore*.so; do
    [ -f "$f" ] && s="$s|ic:$(stat -c '%s:%Y' "$f" 2>/dev/null)"
  done
  printf '%s' "$s" | md5sum | cut -c1-16
}

do_run() {
  local tag=$1
  [ -f "$PKG/scripts/run_fl2va_direct.sh" ] || die "缺 run_fl2va_direct.sh"
  [ -n "$MODEL_ROOT" ] || die "找不到权重 model_index.json，检查 $SHARED/MiniMax-H3"
  [ -f "$INFINI/lib/libinfiniop.so" ] || say "警告：libinfiniop.so 还没编译，import infinicore 大概率失败（先跑 build）"
  ensure_first_frame

  # 库路径先备好：探卡要靠它找到 libamdhip64 及其依赖库
  export PKG SS INFINI_ROOT=$INFINI
  export MODEL_ROOT FIRST_FRAME OUTPUT_DIR=$OUT
  export LD_LIBRARY_PATH="$INFINI/lib:$DTK_ROOT/lib:/opt/hyhal/lib:/opt/hyhal/lib64:${LD_LIBRARY_PATH:-}"
  local cudalib
  for cudalib in "$DTK_ROOT"/cuda/*/lib64 "$DTK_ROOT"/cuda/*/targets/x86_64-linux/lib; do
    [ -d "$cudalib" ] && LD_LIBRARY_PATH="$cudalib:$LD_LIBRARY_PATH"
  done
  [ -d "$IC/build/linux/x86_64/release" ] && LD_LIBRARY_PATH="$IC/build/linux/x86_64/release:$LD_LIBRARY_PATH"
  [ -n "$CUDA_COMPAT" ] && [ -d "$CUDA_COMPAT/lib64" ] && LD_LIBRARY_PATH="$CUDA_COMPAT/lib64:$LD_LIBRARY_PATH"
  export LD_LIBRARY_PATH

  # 先确认这一单真有卡：没卡的话，后面编 20 分钟 + 加载 73G 全是白烧
  say "探测可见 DCU（有卡才往下跑）"
  local pc=0
  probe_dcu plain || pc=$?
  if [ $pc -eq 2 ]; then
    say "  !! 这一单 HIP 看不到任何 DCU —— 不是代码问题，别在这里耗卡时"
    say "     常见原因：作业没拿到卡 / 卡被别的作业占着 / /dev/shm 被写满（见上面余量）"
    say "     处理：直接重提一单（smoke / 8 卡）；反复出现就把这几行贴我"
    return 2
  fi
  if [ -n "${DCU_COUNT:-}" ] && [ "${WANT_GPU:-0}" -gt "$DCU_COUNT" ] 2>/dev/null; then
    say "  只探到 $DCU_COUNT 张卡（原计划 $WANT_GPU 张），按实际卡数跑"
    WANT_GPU=$DCU_COUNT
  fi

  export GPUS=$(seq -s, 0 $((WANT_GPU - 1)))
  # .so 必须在 infinicore/lib/ 下才能被 from infinicore.lib import _infinicore 找到；
  # 包目录只读，用 overlay 兜住（单独跑 smoke 没走 build 时也在这里补一次）
  ensure_infinicore_overlay || say "警告：infinicore overlay 没建起来，import 大概率失败"
  ensure_cuda_compat || say "警告：CUDA→HIP 薄壳没建好，import 仍会报 undefined symbol"
  # 预加载会不会把卡弄丢（10:33 那单的疑点）：探一次就能分辨，并能自动降级
  if [ -n "${H3_CUDA_PRELOAD:-}" ]; then
    local pp=0
    probe_dcu preload || pp=$?
    if [ $pp -eq 2 ] && [ $pc -eq 0 ]; then
      say "  !! plain 有卡但 preload 没卡 → 弃用 DTK libcuda，只留薄壳，重新打补丁"
      H3_CUDA_PRELOAD="$H3_CUDA_SHIM${H3_CCL_SHIM:+:$H3_CCL_SHIM}"
      H3_DROP_LIBCUDA=1
      export H3_CUDA_PRELOAD H3_DROP_LIBCUDA
      patch_fl2va_launcher >/dev/null 2>&1 || true
      probe_dcu preload || say "  !! 只留薄壳也看不到卡，问题不在预加载"
    fi
  fi
  ensure_infinilm || say "警告：_infinilm 没编出来，import infinilm.cache 会失败"
  # janus / xxhash 不在离线 pylibs 里，得先补上，否则 import infinilm 第 13 行就断
  ensure_pydeps || say "警告：janus/xxhash 没补上，import infinilm 会报 ModuleNotFoundError"
  # 出片那一步靠 PyAV，离线环境多半没有：先备好 ffmpeg/cv2 兜底，别等生成完才炸
  ensure_media_encoder || say "警告：出片兜底没备好，最后写 mp4 可能失败"
  # 干完重活再探一次：10:33 那单的卡就是在 21 分钟编译过程中丢的，这里能立刻看见
  local pd=0
  probe_dcu preload || pd=$?
  if [ $pd -eq 2 ]; then
    say "  !! 编译后卡不见了（开工时 ${DCU_COUNT:-?} 张）—— 这一单的卡在作业中途丢了"
    say "     不是代码问题：重提一单即可；每单都这样就是平台侧问题，把上面几行贴我"
    return 2
  fi
  local pyic="$PKG/src/InfiniCore/python"
  [ -n "${INFINICORE_OVERLAY:-}" ] && pyic="$INFINICORE_OVERLAY:$pyic"
  # overlay 放最前：既保证用的是新编的 infinicore，也让它的 sitecustomize.py 先生效
  export PYTHONPATH="$pyic:$PYLIBS:$PYDEPS/dyn:$PYDEPS:$PKG/src/InfiniLM/python"
  # 启动器把 python 的 stdout 接进 `| tee`，管道 → 块缓冲（8KB），print 全憋在内存里，
  # 日志看起来"半天不动"，其实在跑。11:45 那单停在「张量并行度 = 8」就是这个原因。
  export PYTHONUNBUFFERED=1

  if [ "$tag" = "smoke" ]; then export SMOKE=1; fi

  # 便宜的自检：只 import，不加载权重。挂了 5 分钟内就看到根因，不用等整条管线。
  # 但这段导入本身要 5 分钟，而启动器马上会把同一批模块再导一遍——骨架没变就跳过。
  free_shm
  local SKFP=""
  SKFP=$(skel_fp)
  if [ -f "$AFS/.skel_ok" ] && [ "$(cat "$AFS/.skel_ok" 2>/dev/null)" = "$SKFP" ]; then
    say "  预检：骨架指纹 $SKFP 与上一单一致（已导入通过），跳过（省约 5 分钟）"
  else
    preflight_import || {
      say "  !! 预检没过，直接中止：这条链已经断了，再加载 73G 权重也必然失败，不烧这 40 分钟"
      return 3
    }
    echo "$SKFP" > "$AFS/.skel_ok" 2>/dev/null || true
  fi

  # processor 检查只要几十秒，不跟着骨架指纹一起跳过
  preflight_processor || {
    say "  !! processor 起不来，中止：这一环断了，加载完 73G 权重照样死在 encoders.py"
    return 3
  }

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
    if ls "$OUT"/*.mp4 >/dev/null 2>&1; then
      say "$tag 成功，产物："
      ls -lh "$OUT"/*.mp4 2>/dev/null | tail -3
    else
      # 退出码 0 但没 mp4：说明出片那一步退化成 PNG 序列了，别当成成功
      say "$tag 退出码 0，但没有 mp4 —— 视频没编码出来"
      ls -d "$OUT"/*_frames 2>/dev/null | tail -2 | while read -r d; do
        echo "  PNG 序列：$d（$(ls "$d" 2>/dev/null | wc -l) 帧）"
      done
      echo "  原因：PyAV / ffmpeg / cv2 三样都缺。帧是真算出来了，只是没封装成视频"
    fi
  else
    say "$tag 失败 rc=$rc"
    case "$rc" in
      2)
        echo "  这一单 HIP 看不到 DCU（不是代码问题）：重提一单即可；反复出现把探测那几行贴我"
        ;;
      3)
        echo "  预检没过：报错就在上面「预检失败」那几行里，改完再提"
        ;;
      *)
        local lastlog
        lastlog=$(ls -t "$OUT"/fl2va_${tag}_*.log 2>/dev/null | head -1)
        if [ -n "$lastlog" ] && grep -q "Error Code 5" "$lastlog" 2>/dev/null; then
          echo "  集合通信没接管（Error Code 5）：往上找「集合通信薄壳」那行，看是不是没编出来/没预加载"
        elif [ -n "$lastlog" ] && grep -qi "out of memory\|OUT_OF_RESOURCES\|memory access fault" "$lastlog" 2>/dev/null; then
          echo "  显存/访存出错 → 降 num_frames，或换 retry2"
        elif [ -n "$lastlog" ] && grep -q "InfiniCore C API Error" "$lastlog" 2>/dev/null; then
          echo "  InfiniCore 内部报错 → 往上翻 20 行看 C++ 侧打印的文件:行号"
        fi
        echo "  常见原因与对策："
        echo "   · heads 不能被卡数整除 → 减少卡数：retry2"
        echo "   · 显存不足 → 降 num_frames"
        ;;
    esac
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
