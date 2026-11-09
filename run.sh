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
#   compare   官方 minimax_h3_compare.py 的 PyTorch↔InfiniLM 对比
#             （默认只跑 fl2va；PyTorch 基准片生成一次就存在 outputs/compare/，
#               之后靠 --reference-fl2va-video 复用，不会每次重跑）
#             H3_TASKS=fl2va,ref2va 两个任务都跑；H3_REUSE_OURS=1 本方片子也复用
#             H3_FORCE_REF=1 强制重跑基准（旧片留 .bak）
#   report    只算 SSIM/尺寸/帧数，不生成、不占卡（两条片子都在 compare 目录里时用）
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
#
# 14:30 那单：processor 通了、73G 权重也加载完（文本编码器 17.6s / diffusion 13.1s），
# 进入 [generate] 之后约十秒直接 Segmentation fault（rc=139），没有 Python traceback，
# 也没有任何 allreduce 日志。原生崩溃不写栈，所以这一版先装「现场保护」：
#   · faulthandler（SIGSEGV 时打印所有线程的 Python 栈）
#   · 心跳线程（每 45s 打一次阶段 + 全线程栈，信号处理器没跑起来也有最后现场）
#   · 块级进度 [blk] >>> / <<< 包住每个 pipeline block，最后一条就是出事那步
# 同时给两个薄壳加故障注入日志：CUDA 那条链以前有个静默崩溃路径（cuModuleGetFunction
# 拿到 NULL 直接被转发进 hipModuleLaunchKernel）；集合通信则在真搬运前打「进入」点，
# 好分辨是「还没走到集合通信」还是「走进了死在搬运里」。
# 万一 trace 只能定位到 block 而定位不到单次 kernel：H3_SERIAL=1 打开同步 launch
# （=2 连 HIP 运行时 memcpy 一起序列化），崩点会精确到那一条 dispatch，代价是变慢。
#
# 11-08 05:58 那单终于拿到现场了：faulthandler 的栈顶是
#   encoders.py:95 get_qwen3vl_prompt_embeds → minimax_h3.py:397 encode
#   → infiniLM/infinilm/infer_engine.py:548 forward_raw
# 也就是崩在**文本编码器的第一次 forward**，跟 diffusion、跟集合通信都无关
# （整份日志一条 [h3ccl] >> 进入 都没有，commInitAll 倒是成功的）。
# Python 栈到 super().forward(...) 就断，底下全是 C++，所以这一版补两样：
#   · libh3bt.so（build_bt_shim）：SIGSEGV 时先打 C 层栈（每帧给出「哪个 .so +
#     偏移」），再把信号交回 faulthandler 打 Python 栈，两份栈都留在日志里
#   · forward_raw 入参打点：进去那一刻手里拿的张量、形状、精度、在哪张卡。
#     三行看完就能分清是「喂错了东西」（好改）还是「算子/kernel 自己炸」（难改）
#
# 11-08 09:55 那单把上面两条都验了：入参全对（input_ids 1×76 int64 cuda:0、
# position_ids 3×76、四个 list(len=1)），三处钩子也确实挂上了，可还是 rc=139，
# 而且 [h3bt] 一行都没有、faulthandler 也没打 Python 栈 —— 两套都哑了，
# 说明它们装好之后被谁顶掉了，或者 v1 的处理器自己先崩了（它把 backtrace()
# 放在最前面，栈一坏就一个字都写不出来）。于是这一版：
#   · h3bt v2：先写常量串 → siginfo/RIP → dladdr → /proc/self/maps，最后才 unwind；
#     另给 h3bt_selftest()（fork 子进程故意崩一次，证明链是活的）和
#     h3bt_status()（真调用前比对处理器地址，被 HIP 顶掉就重装）
#   · [fr] 第一次真调用之前先跑两个便宜的自检：A 八张卡各跑一次 infinicore
#     微算子（matmul），B 同一台 engine 的纯文本 forward（4 token、无 pixel_values）。
#     A 挂 = 算子/设备层坏；A 过 B 挂 = 注意力/KV 主路坏；A、B 都过 = 视觉塔坏。
#     关掉：H3_FRPROBE=0（B 会往 KV cache 写 4 个槽，只是 smoke 期诊断用）
#
# 11-08 11:48 那单把上面这套跑出了结果：A[0]~A[7] 全部 OK（matmul 读数
# 32768.0 一位不差），B0 一开就崩 —— 算子层排除，视觉塔也排除，问题在
# 纯文本 forward 的公共路径上。但 v2 的 h3bt 把 /proc/self/maps 全量倒进了
# 日志（3000 多行），平台缓冲被冲爆，sig/RIP/出错模块和 C 栈全被挤掉 ——
# 用最没用的信息换掉了最有用的。所以 h3bt 升到 v3：
#   · 顺序按信息密度重排：常量串 → siginfo/RIP → dladdr(RIP) → unwind（每帧都
#     过 dladdr，格式统一是「模块 + 偏移」）→ 最后才是 maps
#   · maps 只挑 RIP/RSP/RBP 各自命中的那一行（三行），全量要 H3_BT_MAPS=1
#
# 同一单还派生出第三条线索：attention.cpp:26 是 num_attention_heads / tp_size
# 的整数除法，不能整除时静默取整，随后 view({bs,seq,heads,head_dim}) 的元素数
# 就和 q 对不上 → 越界 → SIGSEGV，且崩在 qkv_proj 之后、o_proj 的 allreduce
# 之前（这正好解释了「一条 [h3ccl] >> 进入 都没有」）。所以启动时会打一行
# heads / kv_heads / hidden，并检查能不能被卡数整除。
#   · H3_TEXT_TP / H3_DIT_TP：只把「文本编码器」/「DiT」单独降成 N 卡，主线
#     照样全卡。文本塔的活只是 76 个 token 的 prefill，TP=8 对它零收益，却要
#     多走一条 allreduce 和分片边界——而段错误恰好就落在这个 forward 里。降卡
#     等于整段绕开，不碰算子，是「先出一帧视频」的保底手段。
#     ★ 平台只能选 8 卡时，这是唯一能改分片数的杠杆（retry4/retry2 用不了）。
#     不设 = 保持现在的全卡行为（默认不动，免得给定位引入新变量）
#
# 注意组件 config 的实际位置：examples/minimax_h3.py 的 resolve_model_paths 里，
# fl2va 的 partition_dir 就是 diffusers/ 本身，所以 text_encoder 在
# <MODEL_ROOT>/diffusers/text_encoder；而 MODEL_ROOT 通常是 .../MiniMax-H3/FL2VA。
# 直接拼 $MODEL_ROOT/text_encoder 找不到 —— 11-08 那单就是这么丢掉 heads 那行的。
#
# 开关怎么传：平台提交只给位置参数（mode / gpu），环境变量带不进来。所以
# 在 AFS 根下建 h3.env（和 run.sh 同级那个 $AFS），每行一条 H3_XXX=值，
# 启动时会当环境变量 set 一遍，日志里会打印生效了几条。外部已经设过的不覆盖。
# 改开关不必重传 run.sh。
#
# compare 模式的取舍（官方脚本一次跑四件事：PyTorch fl2va / InfiniLM fl2va / 两边 ref2va，
# 每件 124 帧 50 步，一件都不便宜）：
#   · 只跑 fl2va，ref2va 要额外输入视频且时间翻倍，默认不跑
#   · PyTorch 基准片落在 outputs/compare/reference_fl2va.mp4，之后一律 provided，
#     官方脚本连子进程都不起（日志里是 using existing result），省掉最贵的那一趟
#   · 开工前先确认 skimage 在不在 —— 官方脚本是在解码对比那一刻才 import 它的，
#     不先查就会跑完几小时才在最后一步崩在 No module named 'skimage'
#
# smoke 的帧数不是 8（11-08 15:41 那单的结论）：
# 视频 VAE 只能编码 17n+5 帧，before_denoise.py:406 先把 num_frames 向上取到最近的
# 17n+5，再要求落在 [120,360]。8 帧取上去是 22，直接 ValueError —— 也就是说 8 这种
# 「小样」在这个模型里永远不可能合法，最小合法档就是 124（17*7+5）。
# 启动器里那行写死的是 FRAMES=8，包目录是别的 uid 的只读树改不了，所以
# patch_fl2va_launcher 生成补丁副本时会就地换成 ${H3_SMOKE_FRAMES:-124}。
# 想换档位：往 h3.env 写 H3_SMOKE_FRAMES=<17n+5 且 ≥120>，124 已经是最小合法值。
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
# 导入用的 pylibs 副本（/dev/shm）。留空表示还没建，建好后 setup_fast_pylibs 会填。
# 只是个「从哪 import」的开关：.ready / .deps_ok 这些戳和打过的补丁都还在 AFS 上。
PYLIBS_X=${PYLIBS_X:-}
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
# 开关文件：平台提交只给位置参数（mode / gpu），环境变量带不进来。所以留个
# 约定 —— AFS 根下放 h3.env，每行一条 H3_XXX=值，就当地 set 了一遍。
# 好处是改开关不用重传 200KB 的 run.sh，也不用赌平台有没有 env 入口。
# 外部已经设过的不覆盖：命令行/平台真能传时，以它为准。
load_flags_file() {
  local f=$1 n=0 k v line
  [ -f "$f" ] || { echo "  开关文件 $f：不存在"; return 0; }
  while IFS= read -r line || [ -n "$line" ]; do
    line=$(printf '%s' "$line" | tr -d '\r')
    case "$line" in ''|\#*) continue ;; esac
    # 先判有没有等号，再判变量名 —— 反过来的话，中文注释这类"没有等号"的行
    # 会先被变量名规则拦下，提示就成了看不懂，反而看不出真正原因
    case "$line" in *=*) : ;; *) echo "  !! 跳过没有等号的行：$line"; continue ;; esac
    k=${line%%=*}
    case "$k" in ''|*[!A-Za-z0-9_]*) echo "  !! 跳过变量名不合法的行：$line"; continue ;; esac
    case "$k" in [A-Za-z_]*) : ;; *) echo "  !! 跳过变量名不合法的行：$line"; continue ;; esac
    v=${line#*=}
    if [ -n "${!k:-}" ]; then
      echo "   · $k 已由外部设为 ${!k}，文件里这条不生效"
      continue
    fi
    export "$k=$v"
    n=$((n + 1))
  done < "$f"
  echo "  开关文件 $f：生效 $n 条"
}
load_flags_file "$AFS/h3.env"

# ---------- diffusers 的 attention 后端 ----------
# 视频/音频 VAE 的解码不走 InfiniLM，而是 diffusers 自己的 dispatch_attention_fn
# （autoencoder_kl_minimax_h3.py:331、autoencoder_kl_minimax_h3_audio.py:325，
#  两处都是 backend=self._attention_backend，类属性 None → 跟随全局）。
# 全局默认是 "native"，它只是「用 torch SDPA」的意思，具体挑哪个 kernel 由 torch 定：
# 本平台 torch 是 ROCm 版，flash_sdp 默认开着，而 flash_attn_2_cuda*.so 并不存在，
# 于是 scaled_dot_product_attention 在选后端时直接抛
#     RuntimeError: No matching libraries found for flash_attn_2_cuda*.so
# 11-09 02:13 那单就是这样倒在 VideoDecodeStep：DiT 2 步 25.4s 全跑完、allgather 也通了，
# 只剩最后一步出片被这个 kernel 选择挡死。
# "_native_math" 是 diffusers 现成的后端（attention_dispatch.py 的 _native_math_attention），
# 内部就是 with sdpa_kernel(SDPBackend.MATH)，纯 matmul+softmax，不问任何 flash 库。
# 精度上 MATH 反而更稳（softmax 内部升 f32），代价是显存换速度；VAE 解码是分块做的，
# 单块 S 有限，扛得住。想换回去/试别的：h3.env 里写 DIFFUSERS_ATTN_BACKEND=xxx 即可，
# 这里用的是 :- 默认值，外部设过就不覆盖。
export DIFFUSERS_ATTN_BACKEND="${DIFFUSERS_ATTN_BACKEND:-_native_math}"
say "diffusers attention backend = $DIFFUSERS_ATTN_BACKEND（VAE 解码走这条；native 会去要 flash 库）"

# ---------- 画布短边 ----------
# 模型设计画布是短边 768（resolve_canvas_size 的 canvas_short_edge，面积上限
# 768*1344），但 3000 秒装不下：DiT 的 attention 随 token 数平方涨，
# 256 短边实测 2 步 26.7s，按面积外推 768 就是每步上百秒、50 步上万秒。
# 往 256 压之后，124 帧 / 50 步 / seed 42 与模板一致，只有分辨率不同
# （模板没规定分辨率，但报告里要写清楚）。
# 为什么不用 --height/--width：minimax_h3_compare.py 拼子进程命令时是写死的，
# 不给这两个参数，命令行管不到它；而 compare 要过尺寸校验、要算 SSIM，
# 两边必须同尺寸。所以在画布解析这一层统一（h3_proc_fix 的 _patch_canvas_size）。
# 想换档：h3.env 里写 H3_CANVAS_SHORT=512（32 的倍数）。
export H3_CANVAS_SHORT="${H3_CANVAS_SHORT:-256}"
say "画布短边 = $H3_CANVAS_SHORT（默认 768；full 与 compare 都用这个，尺寸才对得上）"

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

# transformers / diffusers 的导入开销几乎全在「遍历包目录、对上万个 .py 逐个 stat」上，
# 而 $PYLIBS 在 AFS 上，一次 stat 就是一次网络往返 —— 05:58 那单的心跳栈看得清清楚楚：
#   transformers/utils/import_utils.py:3126 create_import_structure_from_path → os.path.isfile
# 光 import 就烧掉 600 秒，加上前面 8 分钟预检，走到 [generate] 要 19 分钟，
# 15 分钟的作业上限根本等不到崩溃现场。
# /dev/shm 每单有 450 GB，把 32 MB 的离线包直接解到本地盘只要几秒，之后导入是本地 IO。
# 只改「从哪 import」：.ready / .deps_ok 这些戳、以及上游补丁，仍然落在 AFS 上。
setup_fast_pylibs() {
  # 每个提前返回都留一句人话：09:55 那单这里一声不响就退出了，
  # 结果 preflight 和启动器全在 AFS 上导包（心跳栈里全是 AFS 的 transformers），
  # 白等 10 分钟还查不出为什么。
  if [ "${H3_FAST_PYLIBS:-1}" = "0" ]; then
    say "  本地导入副本：H3_FAST_PYLIBS=0，按开关关闭"
    return 0
  fi
  if [ -n "${PYLIBS_X:-}" ]; then
    # 已经设过就不要再往 PYTHONPATH 前面拼一遍（这函数会被调好几次）
    # 但要先确认真在：/dev/shm 每个作业都是新的，环境里若残留上一单的路径，
    # 静默复用等于把 PYTHONPATH 指到一个不存在的目录 —— 两条链又悄悄回 AFS 导包，
    # 而日志里一句提示都没有（11-09 03:18 那单就是这样：两遍 import 白等 800 秒）。
    if [ -d "$PYLIBS_X/transformers" ]; then
      say "  本地盘副本复用：$PYLIBS_X（不碰 AFS）"
      case ":${PYTHONPATH:-}:" in
        *":$PYLIBS_X:"*) : ;;
        *) export PYTHONPATH="$PYLIBS_X${PYTHONPATH:+:$PYTHONPATH}" ;;
      esac
      return 0
    fi
    say "  本地导入副本：环境里的 PYLIBS_X=$PYLIBS_X 已不存在（新容器的 /dev/shm 是空的），重建"
    PYLIBS_X=
  fi
  if [ -z "${PYLIBS:-}" ]; then
    say "  !! 本地导入副本：PYLIBS 还是空的（调用点太早），这一单仍从 AFS 导入"
    return 0
  fi
  local stamp=""
  stamp=$(cat "$PYLIBS/.ready" 2>/dev/null)
  if [ -z "$stamp" ]; then
    say "  !! 本地导入副本：$PYLIBS/.ready 读不到（空），这一单仍从 AFS 导入"
    return 0
  fi

  # 目录名带 v1：以后补丁逻辑变了就把 v1 提到 v2，保证不会复用旧的本地副本
  local shmb=${H3_SHM_DIR:-/dev/shm}
  local dst=$shmb/h3pylibs.${stamp}.v1
  if [ -f "$dst/.ok" ] && [ -d "$dst/transformers" ]; then
    export PYLIBS_X=$dst
    export PYTHONPATH="$dst${PYTHONPATH:+:$PYTHONPATH}"
    say "  导入路径 = $dst（本地盘副本，不碰 AFS）"
    return 0
  fi

  local freemb=""
  freemb=$(df -Pm "$shmb" 2>/dev/null | awk 'NR==2{print $4}')
  if [ -z "$freemb" ] || [ "$freemb" -lt 3000 ]; then
    say "  $shmb 余量 ${freemb:-?}MB 不够，pylibs 仍从 AFS 导入（会慢 5~10 分钟）"
    return 0
  fi

  local zip=${PYLIBS_ZIP:-}
  [ -n "$zip" ] || zip=$(ls -t "$AFS"/pylibs*-v3.zip "$AFS"/dl/pylibs*-v3.zip 2>/dev/null | head -1)
  [ -n "$zip" ] || zip=$(ls -t "$AFS"/pylibs*.zip "$AFS"/dl/pylibs*.zip 2>/dev/null | head -1)

  local t0=$SECONDS
  rm -rf "$dst" 2>/dev/null
  # mkdir 失败也要说话：静默 return 会让这一单又回 AFS 导包，日志里看不出原因
  mkdir -p "$dst" 2>/dev/null || { say "  !! 本地副本建不出来（$dst），这一单仍从 AFS 导入"; return 0; }
  say "  建本地导入副本 $dst（解包一次，之后每单导入省 5~10 分钟）"
  if [ -n "$zip" ] && [ -f "$zip" ]; then
    python3 - "$zip" "$dst" >/dev/null 2>&1 <<'PY'
import sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as z:
    z.extractall(sys.argv[2])
PY
  else
    cp -a "$PYLIBS/." "$dst/" 2>/dev/null
  fi
  [ -d "$dst/transformers" ] || {
    rm -rf "$dst" 2>/dev/null
    say "  !! 本地副本里没有 transformers，退回 AFS"
    return 0
  }
  # AFS 那份已经打过 Qwen2TokenizerFast 兼容补丁，直接把同一文件覆盖过来，免得上游两边不一致
  local enc=diffusers/modular_pipelines/minimax_h3/encoders.py
  [ -f "$PYLIBS/$enc" ] && cp -f "$PYLIBS/$enc" "$dst/$enc" 2>/dev/null
  touch "$dst/.ok" 2>/dev/null
  export PYLIBS_X=$dst
  export PYTHONPATH="$dst${PYTHONPATH:+:$PYTHONPATH}"
  say "  本地副本就绪（$((SECONDS - t0))s）：这两条链之后都从 /dev/shm 导入"
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
  # 这一步是真 import（torch + diffusers + transformers），在 AFS 上实测 6 分钟；
  # 换成本地盘副本后就只剩几秒，指纹失效时不至于又白烧 6 分钟
  setup_fast_pylibs
  PYTHONPATH="${PYLIBS_X:-$PYLIBS}:$PYTHONPATH" python3 - <<'PY'
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
  # 视觉塔和 DiT 都走 op::mha，而 Hygon 上它零注册。这一步只编一个 .cc（几秒），
  # 不碰 xmake；编不出来就明确告诉这一单会白跑，而不是等到 rip=0 才发现。
  if ensure_mha_fallback; then
    say "  视觉塔 op::mha 已有兜底实现（不会再 call 0）"
  else
    # 没有它 = 100% 在视觉塔第一步 call 0，跑下去只是白烧 9 分钟再崩。
    # 当场退出，把时间留给改代码。
    say "  !! mha 兜底没装上：这单必然在视觉塔 rip=0，直接停（往上翻 [mha] 那段）"
    die "mha 兜底缺失，跑下去必崩"
  fi
  # DiT 的 op::cast_ 在 Hygon 上只有 ATen 实现（没开 ENABLE_ATEN 就必抛）。
  # 文本塔不经过它，所以前面几单都是一路跑到 DiT 第一步才炸，白烧 20 分钟。
  if ensure_cast_fallback; then
    say "  DiT op::cast_ 已有覆盖（不会再 cast_ requires ATen）"
  else
    say "  !! cast_ 覆盖没装上：DiT 第一步必抛 cast_ requires ATen，直接停"
    say "     （H3_CAST=0 只用于确认诊断，不会让 DiT 跑通）"
    die "cast_ 覆盖缺失，跑下去必在 DiT 抛"
  fi
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
# 把刚编出来的 libinfinicore_cpp_api.so 刷到所有可能被加载的位置。
# 判据是内容（cmp -s），一致就不动，免得每轮白拷一份几十 MB 的库。
refresh_cpp_api() {
  local newapi="" f dst n=0
  if [ -n "$IC" ] && [ -f "$IC/build/linux/x86_64/release/libinfinicore_cpp_api.so" ]; then
    newapi=$IC/build/linux/x86_64/release/libinfinicore_cpp_api.so
  elif [ -n "$IC" ]; then
    newapi=$(find "$IC/build" -maxdepth 4 -name libinfinicore_cpp_api.so 2>/dev/null | head -1)
  fi
  if [ -z "$newapi" ] || [ ! -f "$newapi" ]; then
    say "  (编译树里没有 libinfinicore_cpp_api.so，跳过刷新：IC=${IC:-未定})"
    return 0
  fi
  for dst in "$INFINI/lib" "$OVL/infinicore/lib" "$OVL/lib" \
             "$SRC_IC/python/infinicore/lib" "$AFS/ic_py/infinicore/lib"; do
    [ -d "$dst" ] || continue
    if [ -f "$dst/libinfinicore_cpp_api.so" ] && cmp -s "$newapi" "$dst/libinfinicore_cpp_api.so"; then
      continue
    fi
    if cp -f "$newapi" "$dst/libinfinicore_cpp_api.so" 2>/dev/null; then
      n=$((n + 1))
      say "  libinfinicore_cpp_api.so 刷新 → $dst  $(ls -lh "$newapi" 2>/dev/null | awk '{print $5}')"
    else
      say "  !! cpp_api 刷新失败（只读或被占）：$dst"
    fi
  done
  [ "$n" = "0" ] && say "  libinfinicore_cpp_api.so 与编译树一致，不重复拷"
  return 0
}

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
  # 重编过 cpp_api 之后，$INFINI/lib 里的旧副本必须换掉。
  # _infinilm.so 通过 RPATH 去 $INFINI/lib 找 libinfinicore_cpp_api.so，
  # 而 ensure_infini_root 是「文件不存在才补」，永远刷不新旧文件 ——
  # 改了 C++ 算子（ops/multi_head_attention/mha_composed_fallback.cc）却还在跑旧实现，
  # 就会继续 rip=0。这里按内容比对，不一致就覆盖所有副本。
  refresh_cpp_api

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
 * 只包含 stdio.h（打印用），不依赖 cuda.h 也不依赖 hip 头，任何 cc 都能编。 */
#include <stdio.h>

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

/* 故障注入式日志：14:30 那单在 [generate] 之后十秒直接 SIGSEGV，没有 Python 栈，
 * 也没看到任何 allreduce 日志。CUDA driver 这条链有一个典型的静默崩溃路径：
 * cuModuleLoadData 失败（把 CUDA 的 PTX/cubin 喂给了 hipModuleLoadData）→
 * cuModuleGetFunction 拿到 NULL → 调用方不检查就把 NULL 传给 cuLaunchKernel →
 * hipModuleLaunchKernel 解引用空指针 → 段错误。所以这里三件事：
 *   1. 加载/取函数失败一律打日志，并把镜像头 4 字节抠出来（能一眼认出格式）
 *   2. cuLaunchKernel 遇到 NULL 函数句柄直接返回错误码，不往下转发
 *   3. 前几次成功 launch 打点，确认这条路径到底有没有被走到
 */
static long h3c_bad = 0;

CUresult cuInit(unsigned int flags) { (void)flags; return 0; }

CUresult cuModuleLoadData(CUmodule *mod, const void *image) {
  int rc = hipModuleLoadData((void **)mod, image);
  if (rc != 0) {
    unsigned int magic = image ? *(const unsigned int *)image : 0u;
    fprintf(stderr,
            "[h3cuda] !! cuModuleLoadData 失败 rc=%d magic=0x%08x"
            "（0x7f454c46=ELF/HSACO 正常；0x50ed55ba=PTX 不认；0xb1c0ffee=fatbin 不认）\n",
            rc, magic);
    fflush(stderr);
  }
  return rc;
}

CUresult cuModuleGetFunction(CUfunction *fn, CUmodule mod, const char *name) {
  int rc = hipModuleGetFunction((void **)fn, mod, name);
  if (rc != 0) {
    fprintf(stderr, "[h3cuda] !! cuModuleGetFunction 失败 rc=%d name=%s\n",
            rc, name ? name : "(null)");
    fflush(stderr);
  }
  return rc;
}

CUresult cuModuleUnload(CUmodule mod) { return hipModuleUnload(mod); }

CUresult cuLaunchKernel(CUfunction f,
                        unsigned int gx, unsigned int gy, unsigned int gz,
                        unsigned int bx, unsigned int by, unsigned int bz,
                        unsigned int shared, CUstream stream,
                        void **params, void **extra) {
  if (!f) {
    if (h3c_bad < 5) {
      /* 走到这里说明上游没查 cuModuleGetFunction 的返回值。以前这么传下去就是
       * hipModuleLaunchKernel 解引用空指针 —— 一崩连栈都没有。 */
      fprintf(stderr, "[h3cuda] !! cuLaunchKernel 收到 NULL 函数句柄，已拦下"
                      "（根因见上面的加载/取函数报错）\n");
      fflush(stderr);
      h3c_bad++;
    }
    return 1; /* CUDA_ERROR_INVALID_VALUE */
  }
  {
    static long n = 0;
    n++;
    if (n <= 3) {
      fprintf(stderr, "[h3cuda] launch #%ld grid=(%u,%u,%u) block=(%u,%u,%u) shared=%u\n",
              n, gx, gy, gz, bx, by, bz, shared);
      fflush(stderr);
    }
  }
  {
    int rc = hipModuleLaunchKernel(f, gx, gy, gz, bx, by, bz, shared, stream, params, extra);
    if (rc != 0 && h3c_bad < 5) {
      fprintf(stderr, "[h3cuda] !! hipModuleLaunchKernel 返回 %d（grid/block 见上一行）\n", rc);
      fflush(stderr);
      h3c_bad++;
    }
    return rc;
  }
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
    unsigned long long ag_seq; /* allgather 自己的计数（不能和 allreduce 共用，否则「前 3 次」会被对方吃掉） */
    unsigned long long rs_seq; /* reduceScatter 自己的计数 */
    unsigned long long enter;  /* 进入次数（用于前几次打点） */
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

    /* 进入点打点：11-07 14:30 那单崩在 [generate] 之后，一条 allreduce 日志都没有，
     * 分不清是「还没走到集合通信」还是「走进了但死在搬运里」。这里在真搬运之前打，
     * 前两次集合通信把 8 个 rank 的到达情况和 count/dtype 全列出来：
     *   - 只出现 7 条 → 有 rank 没到，问题在计算侧或线程侧
     *   - 8 条都出现但随后崩 → 死在 hipMemcpy/归约里
     *   - count 各不相同 → TP 切分不对，后面的越界读也是它引起的 */
    g->enter++;
    if (g->enter <= 16) {
        fprintf(stderr, "[h3ccl] >> 进入 #%llu rank=%d count=%ld dtype=%d op=%d\n",
                g->enter, rank, (long)count, dtype, op);
        fflush(stderr);
    }

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

    /* rank 间 count/dtype 不一致 = TP 切分不对。下面按 leader 的 count 搬运，
     * 比它长的 rank 会被多读一截（越界），比它短的读到脏数据。以前完全静默。 */
    {
        static long h3_mism = 0;
        int bad = 0, k;
        for (k = 0; k < g->n; ++k) {
            if (g->count[k] != count || g->dtype[k] != dtype) bad = 1;
        }
        if (bad && h3_mism < 5) {
            fprintf(stderr, "[h3ccl] !! rank 间 count/dtype 不一致：");
            for (k = 0; k < g->n; ++k) {
                fprintf(stderr, " r%d=%ld/%d", k, (long)g->count[k], g->dtype[k]);
            }
            fprintf(stderr, "\n");
            fflush(stderr);
            h3_mism++;
        }
    }

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
        /* 只打前 3 次是为了不把平台日志冲爆，但 11-08 12:03 那单因此被误读成
           「只跑了 3 层、forward 没走完」。第 3 次之后必须留一句说明。 */
        if (g->seq == 3)
            h3_log("[h3ccl] （前 3 次已报；之后每 500 次再报一行，静默不代表没走）\n");
    }

    g->gen++;
    H3_CV_BCAST(&g->cv);
    H3_MX_UNLOCK(&g->mu);
    return 0;
}

/* ---- allgather / allgatherV ----
 * 与 allreduce 共用同一套栅栏（mu + arrive + gen），但搬运模式不同，所以单独成函数，
 * 不去动已经跑通的 h3_collective：
 *   allreduce     每 rank 一份 → 归约 → 广播回每个 rank
 *   allgather     每 rank 一份 → 按 rank 顺序整块拼进每个 rank 的 output
 *   reduceScatter 每 rank 有 n 段 → 每段归约 → rank r 只留下第 r 段
 * 语义按 nccl 对应接口：阻塞返回，返回时 recvbuf 已就绪。
 *
 * 为什么要它：11-08 17:45 那单文本塔、cast_ 全过，进 DiT 第一个 block 就死在
 * allgather.cc:55 "Not Implemented" —— MiniMaxH3AdaLNProjection 是列并行，
 * 每层都要把 [1, out/tp] 的本地份拼回完整宽度（transformer.cpp:301）。 */
static int h3_gather(h3grp *g, int rank, size_t send_count, const size_t *recv_counts,
                     int dtype, void *sbuf, void *rbuf, void *stream) {
    size_t esz = (size_t)h3_esz(dtype);
    size_t chunk, off, n, maxc = 0, base = 0;
    int i, k;
    unsigned long long mygen;

    if (!esz) {
        h3_say("[h3ccl] !! allgather 不支持的 dtype=", (long)dtype, "");
        return 10;
    }
    if (send_count == 0) return 0;

    /* 各 rank 的 sendbuf 先算完再进来（和 allreduce 同样的理由） */
    if (stream) hipStreamSynchronize(stream);
    else hipDeviceSynchronize();

    H3_MX_LOCK(&g->mu);

    g->enter++;
    if (g->enter <= 16) {
        fprintf(stderr, "[h3ccl] >> ag 进入 #%llu rank=%d send=%ld dtype=%d\n",
                g->enter, rank, (long)send_count, dtype);
        fflush(stderr);
    }

    mygen = g->gen;
    g->count[rank] = send_count;
    g->dtype[rank] = dtype;
    g->sbuf[rank] = sbuf;
    g->rbuf[rank] = rbuf;

    if (++g->arrive < g->n) {
        while (g->gen == mygen) H3_CV_WAIT(&g->cv, &g->mu);
        H3_MX_UNLOCK(&g->mu);
        return 0;
    }
    g->arrive = 0;

    /* 每份长度：普通 allgather 全等 send_count，allgatherV 逐份不同。
     * 分块按最长那份，保证 stage 装得下任意一份的一块。 */
    for (k = 0; k < g->n; ++k) {
        size_t ck = recv_counts ? recv_counts[k] : send_count;
        if (ck > maxc) maxc = ck;
    }
    chunk = maxc > (size_t)H3_CHUNK ? (size_t)H3_CHUNK : maxc;
    if (chunk == 0) chunk = 1;
    if (h3_ensure(g, chunk) != 0) {
        g->gen++;
        H3_CV_BCAST(&g->cv);
        H3_MX_UNLOCK(&g->mu);
        return 1;
    }

    /* 拼接顺序 = rank 顺序：output 里第 k 块就是 rank k 的 sendbuf。
     * allgather.cc 就是按 shape[0] *= world_size 建的 output，所以必须整块顺序
     * 追加，不能按块交错（交错只能对上「每份 1 个元素」的特例）。 */
    for (k = 0; k < g->n; ++k) {
        size_t ck = recv_counts ? recv_counts[k] : send_count;
        for (off = 0; off < ck; off += chunk) {
            n = ck - off;
            if (n > chunk) n = chunk;
            hipSetDevice(g->ids[k]);
            hipMemcpy(g->stage[k], (char *)g->sbuf[k] + off * esz, n * esz, H3_D2H);
            for (i = 0; i < g->n; ++i) {
                hipSetDevice(g->ids[i]);
                hipMemcpy((char *)g->rbuf[i] + (base + off) * esz, g->stage[k], n * esz, H3_H2D);
            }
            g->moved += (unsigned long long)(n * esz) * (unsigned long long)(g->n + 1);
        }
        base += ck;
    }
    if (g->n > 0) hipSetDevice(g->ids[rank]);

    g->ag_seq++;
    if (g->ag_seq <= 3 || (g->ag_seq % 500ULL) == 0) {
        char buf[256];
        char *p = buf;
        const char *q;
        for (q = "[h3ccl] allgather #"; *q; ++q) *p++ = *q;
        p = h3_num(p, (long)g->ag_seq);
        for (q = "  send="; *q; ++q) *p++ = *q;
        p = h3_num(p, (long)send_count);
        for (q = "  total="; *q; ++q) *p++ = *q;
        p = h3_num(p, (long)base);
        for (q = "  dtype="; *q; ++q) *p++ = *q;
        p = h3_num(p, (long)dtype);
        for (q = "  ranks="; *q; ++q) *p++ = *q;
        p = h3_num(p, (long)g->n);
        *p++ = '\n';
        *p = 0;
        h3_log(buf);
        if (g->ag_seq == 3)
            h3_log("[h3ccl] （allgather 前 3 次已报；之后每 500 次再报一行）\n");
    }

    g->gen++;
    H3_CV_BCAST(&g->cv);
    H3_MX_UNLOCK(&g->mu);
    return 0;
}

/* ---- reduceScatter / reduceScatterV ---- */
static int h3_scatter(h3grp *g, int rank, size_t recv_count, const size_t *send_counts,
                      int dtype, int op, void *sbuf, void *rbuf, void *stream) {
    size_t esz = (size_t)h3_esz(dtype);
    size_t chunk, off, n, maxc = 0, base = 0;
    int i, k;
    unsigned long long mygen;

    if (!esz) {
        h3_say("[h3ccl] !! reduceScatter 不支持的 dtype=", (long)dtype, "");
        return 10;
    }
    if (recv_count == 0) return 0;
    if (op != 0 && op != 4) {
        h3_say("[h3ccl] !! reduceScatter 只支持 SUM/AVG，收到 op=", (long)op, "");
        return 2;
    }

    if (stream) hipStreamSynchronize(stream);
    else hipDeviceSynchronize();

    H3_MX_LOCK(&g->mu);

    g->enter++;
    if (g->enter <= 16) {
        fprintf(stderr, "[h3ccl] >> rs 进入 #%llu rank=%d recv=%ld dtype=%d op=%d\n",
                g->enter, rank, (long)recv_count, dtype, op);
        fflush(stderr);
    }

    mygen = g->gen;
    g->count[rank] = recv_count;
    g->dtype[rank] = dtype;
    g->op[rank] = op;
    g->sbuf[rank] = sbuf;
    g->rbuf[rank] = rbuf;

    if (++g->arrive < g->n) {
        while (g->gen == mygen) H3_CV_WAIT(&g->cv, &g->mu);
        H3_MX_UNLOCK(&g->mu);
        return 0;
    }
    g->arrive = 0;

    for (k = 0; k < g->n; ++k) {
        size_t ck = send_counts ? send_counts[k] : recv_count;
        if (ck > maxc) maxc = ck;
    }
    chunk = maxc > (size_t)H3_CHUNK ? (size_t)H3_CHUNK : maxc;
    if (chunk == 0) chunk = 1;
    if (h3_ensure(g, chunk) != 0) {
        g->gen++;
        H3_CV_BCAST(&g->cv);
        H3_MX_UNLOCK(&g->mu);
        return 1;
    }

    /* 第 k 段：每个 rank 从自己 sendbuf 的同一段取，归约后只写进 rank k 的 recvbuf
     * （reduce_scatter 的定义：rank r 拿到的就是「所有 rank 的第 r 段之和」）。 */
    for (k = 0; k < g->n; ++k) {
        size_t ck = send_counts ? send_counts[k] : recv_count;
        for (off = 0; off < ck; off += chunk) {
            n = ck - off;
            if (n > chunk) n = chunk;
            for (i = 0; i < g->n; ++i) {
                hipSetDevice(g->ids[i]);
                hipMemcpy(g->stage[i], (char *)g->sbuf[i] + (base + off) * esz, n * esz, H3_D2H);
            }
            if (h3_reduce(g, n, dtype, op) != 0) break;
            hipSetDevice(g->ids[k]);
            hipMemcpy((char *)g->rbuf[k] + off * esz, g->stage[0], n * esz, H3_H2D);
            g->moved += (unsigned long long)(n * esz) * (unsigned long long)(g->n + 1);
        }
        base += ck;
    }
    if (g->n > 0) hipSetDevice(g->ids[rank]);

    g->rs_seq++;
    if (g->rs_seq <= 3 || (g->rs_seq % 500ULL) == 0) {
        char buf[256];
        char *p = buf;
        const char *q;
        for (q = "[h3ccl] reduceScatter #"; *q; ++q) *p++ = *q;
        p = h3_num(p, (long)g->rs_seq);
        for (q = "  recv="; *q; ++q) *p++ = *q;
        p = h3_num(p, (long)recv_count);
        for (q = "  total="; *q; ++q) *p++ = *q;
        p = h3_num(p, (long)base);
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
    h3comm *c = (h3comm *)comm;
    if (!c || !c->grp) return 4;
    return h3_gather(c->grp, c->rank, send_count, NULL, datatype, sendbuf, recvbuf, stream);
}

int infinicclAllGatherV(void *sendbuf, void *recvbuf, const size_t *recv_counts, int nranks,
                        int datatype, void *comm, void *stream) {
    h3comm *c = (h3comm *)comm;
    if (!c || !c->grp) return 4;
    if (!recv_counts) return 4;
    if (nranks != c->grp->n) {
        h3_say("[h3ccl] !! AllGatherV 的 nranks 与通信域不符：", (long)nranks, "");
        return 3;
    }
    /* send_count 只做「非零」校验用；真实长度逐份从 recv_counts 取 */
    return h3_gather(c->grp, c->rank, recv_counts[c->rank], recv_counts,
                     datatype, sendbuf, recvbuf, stream);
}

int infinicclReduceScatter(void *sendbuf, void *recvbuf, size_t recv_count, int datatype,
                           int op, void *comm, void *stream) {
    h3comm *c = (h3comm *)comm;
    if (!c || !c->grp) return 4;
    return h3_scatter(c->grp, c->rank, recv_count, NULL, datatype, op, sendbuf, recvbuf, stream);
}

int infinicclReduceScatterV(void *sendbuf, void *recvbuf, const size_t *send_counts, int nranks,
                            int datatype, int op, void *comm, void *stream) {
    h3comm *c = (h3comm *)comm;
    if (!c || !c->grp) return 4;
    if (!send_counts) return 4;
    if (nranks != c->grp->n) {
        h3_say("[h3ccl] !! ReduceScatterV 的 nranks 与通信域不符：", (long)nranks, "");
        return 3;
    }
    return h3_scatter(c->grp, c->rank, send_counts[c->rank], send_counts,
                      datatype, op, sendbuf, recvbuf, stream);
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

  say "  编译集合通信薄壳（$(basename "$cc")，自实现 allreduce/allgather/reduceScatter，不依赖 RCCL）"
  local -a larg=(-pthread)
  [ -n "$hiplib" ] && larg+=("$hiplib" "-Wl,-rpath,$hipd")
  if ! "$cc" -O2 -fPIC -shared -o "$so" "$src" ${larg[@]+"${larg[@]}"} 2>"$dir/ccl_cc.log"; then
    say "  !! 集合通信薄壳编译失败："
    tail -12 "$dir/ccl_cc.log" 2>/dev/null | sed 's/^/    /'
    return 1
  fi
  [ -s "$so" ] || { say "  !! 集合通信薄壳没产出（$so 为空）"; return 1; }

  # 少一个符号就会在运行时炸，这里逐个核对
  # allgather/reduceScatter 是 11-08 17:45 那单补的：DiT 的 adaln 列并行每层都要 allgather，
  # 原来返回 Not Implemented，白烧一单
  local want="infinicclCommInitAll infinicclAllReduce infinicclBroadcast infinicclAllGather infinicclAllGatherV infinicclReduceScatter infinicclReduceScatterV infinicclCommDestroy infinicclGetCommName infinicclGroupStart infinicclGroupEnd"
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
  local tmpd=$dyn/.h3_proc_fix.new tmpe=$AFS/ic_py/.h3_entry.new tmpt=$dyn/.h3_trace.new
  local changed=0

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
import os

_LOG = logging.getLogger("h3.procfix")

_APPLIED = False
_VPM = None  # transformers.video_processing_utils.VideoProcessorMixin 的真类
_CANVAS_PATCHED = False


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


def _patch_canvas_size():
    """把默认画布短边压到 H3_CANVAS_SHORT（run.sh 里默认 256）。

    平台的 3000 秒装不下设计值（短边 768）：DiT 的 attention 随 token 数平方涨，
    256 短边实测 2 步 26.7s；按面积放大 10.9 倍、S 放大 9 倍外推，50 步要上万秒。
    full 与 compare 还必须同尺寸 —— compare 要过尺寸校验、才谈得上 SSIM，
    而 minimax_h3_compare.py 拉起子进程时不给 --height/--width（命令是拼死的），
    所以只能从画布解析这一层统一，命令行参数反而管不到 compare。

    只改 short_edge，max_pixels 原样传回：256 短边算出来的面积远小于预算
    （768*1344），不会触发那条按面积缩的分支。
    """
    global _CANVAS_PATCHED
    short = os.environ.get("H3_CANVAS_SHORT", "").strip()
    if not short or _CANVAS_PATCHED:
        return False
    n = int(short)
    if n <= 0 or n % 32:
        raise ValueError(f"H3_CANVAS_SHORT 必须是 32 的正整数倍，拿到 {short!r}")

    import diffusers.modular_pipelines.minimax_h3.modular_pipeline as mp
    import diffusers.modular_pipelines.minimax_h3.before_denoise as bd
    import diffusers.modular_pipelines.minimax_h3.before_encoder as be

    orig = mp.resolve_canvas_size

    def patched(aspect_width, aspect_height, canvas_multiple, short_edge, max_pixels, **kw):
        return orig(aspect_width, aspect_height, canvas_multiple, n, max_pixels, **kw)

    # 三个模块各自 from ... import 绑定了这个名字，只改 modular_pipeline 不够
    for mod in (mp, bd, be):
        mod.resolve_canvas_size = patched
    _CANVAS_PATCHED = True
    print(
        f"[h3] 画布短边已压到 {n}（默认 768；full 与 compare 都用这个，尺寸才对得上）",
        flush=True,
    )
    return True


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
    # 画布尺寸和 processor 无关，但两条路都要它：full 走 h3_entry，compare 走
    # sitecustomize，都从这里过。只有设了 H3_CANVAS_SHORT 才动手。
    step("画布尺寸兜底", _patch_canvas_size)

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
"""examples/minimax_h3.py 的薄包装。

做两件事，都在入口脚本之前：

1. `h3_trace.install()` —— 段错误现场保护（faulthandler + 心跳 + 块级进度）。
   必须最先装：万一崩在下面 `import h3_proc_fix` 或 transformers 导入里，也想看到栈。
2. `import h3_proc_fix` —— 给 transformers 装 processor 兜底，必须赶在 AutoProcessor
   加载之前。

FL2VA 的入口是 `python examples/minimax_h3.py fl2va --model-path=...`。命令行里塞
`-c` 会打乱 argv[0] 和相对路径，所以用 runpy 包装：argv 原样传给入口脚本，
`__name__ == "__main__"` 也保持成立。
"""

import os
import runpy
import sys

try:
    import h3_trace

    h3_trace.install()
except Exception as exc:  # noqa: BLE001 —— 现场保护装不上不能拦路
    print(f"[h3] 段错误现场保护没装上：{type(exc).__name__}: {exc}", flush=True)

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

  # 段错误现场保护：faulthandler + 心跳 + 块级进度（h3_entry 会 import 它）
  cat > "$tmpt" <<'PYEOF'
"""段错误现场保护 + diffusers modular pipeline 块级进度。

14:30 那单：processor 通了、73G 权重也加载完了，`[generate]` 之后约十秒直接
Segmentation fault（rc=139）。原生崩溃不写 Python traceback，日志停在

    [generate] Running MiniMax-H3 FL2VA
      frames: 8  steps: 2

就没了 —— 完全看不出死在哪一步、死的时候 8 个 RankWorker 在干什么。

这里补三样东西，都不改仓库代码：

1. faulthandler：SIGSEGV / SIGABRT / SIGFPE / SIGBUS 时把**所有** Python 线程的
   栈打到 stderr。就算崩在 C 扩展里，也能看到是从哪一行 Python 调进去的。

2. 心跳线程：每 N 秒打一行当前阶段，并把所有线程的栈 dump 一次。万一信号处理器
   没跑起来（栈被踩坏、崩在非 Python 线程里），最后一份心跳就是现场快照。

3. 块级进度：给 ModularPipeline 和它下面每个 block 的 `__call__` 挂日志，进入/退出
   各一行并带耗时。段错误前最后一条 `[blk] >>> X` 就是出事那一步，前面所有
   `[blk] <<< X  1.2s` 则证明那几步是干净的。

块为什么要「延迟挂」：`minimax_h3.py` 里 `from diffusers import ModularPipeline` 在
模块导入时就发生，而具体 block 类（MiniMaxH3ResizeStep 之类）在那之后才定义。所以
只能先挂住 `ModularPipeline.__call__`，等第一次 `pipe(...)` 时再遍历
`pipe._blocks.sub_blocks` 整棵树，按类挂。

环境变量：
    H3_TRACE=0     关掉全部（默认开）
    H3_HB=45       心跳间隔秒数，0 表示只打印不 dump 栈
"""

import builtins
import faulthandler
import functools
import os
import sys
import threading
import time

_STATE = {
    "installed": False,
    "patched": False,
    "patched_fr": False,
    "patched_tt": False,
    "patched_ttp": False,
    "patched_dtp": False,
    "depth": 0,
    "stage": "启动中",
    "t0": time.perf_counter(),
}
_SAID = set()


def _say(msg):
    try:
        sys.stderr.write(msg + "\n")
        sys.stderr.flush()
    except Exception:  # noqa: BLE001 —— 日志绝不能反过来把主流程搞挂
        pass


def _once(key, msg):
    if key in _SAID:
        return
    _SAID.add(key)
    _say(msg)


def _now():
    return time.perf_counter() - _STATE["t0"]


# ---------------------------------------------------------------- faulthandler


def _enable_faulthandler():
    # faulthandler 要一个带真实 fd 的流；sys.stderr 被换成 StringIO 时会抛 fileno，
    # 所以退到 sys.__stderr__（原始 fd 2）。两个都不行就放弃，不能拦路。
    last = None
    for stream in (sys.stderr, sys.__stderr__):
        if stream is None:
            continue
        try:
            faulthandler.enable(file=stream, all_threads=True)
        except TypeError:  # 老解释器没有 all_threads
            try:
                faulthandler.enable(file=stream)
            except Exception as exc:  # noqa: BLE001
                last = exc
                continue
        except Exception as exc:  # noqa: BLE001
            last = exc
            continue
        _say("[trace] faulthandler 已开：SIGSEGV/SIGABRT/SIGFPE 会打印所有线程的 Python 栈")
        return True
    _say("[trace] faulthandler 打不开（无可用 fd）：%s: %s" % (type(last).__name__, last))
    return False


def _install_native_bt():
    """装 C 层栈兜底（libh3bt.so）。

    顺序有讲究：本函数必须在 faulthandler.enable() 之后调用。它是以「链」的方式
    工作的——自己上位，崩时先打 C 栈，再把信号交回被它顶掉的那个处理器，也就是
    faulthandler。装早了就被 faulthandler 顶掉，装晚了才轮到它打头阵。

    两个自检（09:55 那单的教训：处理器"装了"却一个字都没打出来，等于没装）：
      · h3bt_selftest()：fork 子进程故意崩一次，证明整条链是活的
      · h3bt_status()  ：和 h3bt_self() 比，证明现在挂的还是我（没被 HIP 顶掉）
    """
    lib = os.environ.get("H3_BT_LIB", "")
    if not lib or not os.path.isfile(lib):
        return False
    if os.environ.get("H3_BT", "1").strip().lower() in ("0", "false", "no"):
        return False
    try:
        import ctypes

        so = ctypes.CDLL(lib, use_errno=False)
        so.h3bt_install.restype = ctypes.c_int
        so.h3bt_status.restype = ctypes.c_ulong
        so.h3bt_self.restype = ctypes.c_ulong
        so.h3bt_selftest.restype = ctypes.c_int
        rc = so.h3bt_install()
        if rc != 0:
            _say("[trace] 原生栈兜底装不上：h3bt_install() 返回 %d" % rc)
            return False
        _STATE["bt_so"] = so
        mine = int(so.h3bt_self())
        _STATE["bt_self"] = mine
        _say("[trace] 原生栈兜底已装：%s（处理器 @0x%x，崩时先打 C 栈再交回 faulthandler）"
             % (lib, mine))
        if os.environ.get("H3_BT_SELFTEST", "1").strip().lower() not in ("0", "false", "no"):
            st = int(so.h3bt_selftest())
            if st in (11, 139):
                _say("[trace]   自检通过：子进程确实被 SIGSEGV 干掉（%d），处理器链是活的" % st)
            else:
                _say("[trace]   !! 自检异常：子进程 rc=%d（期望 139）"
                     "—— 这一单的 C 栈很可能还是打不出来" % st)
        return True
    except Exception as exc:  # noqa: BLE001
        _say("[trace] 原生栈兜底加载失败：%s: %s" % (type(exc).__name__, exc))
        return False


def _bt_check(tag):
    """崩之前确认 C 栈兜底还在位。

    HIP / DTK 初始化时会自己装 SIGSEGV 处理器；它装在我们后面就把我们顶掉了。
    09:55 那单 C 栈和 Python 栈一个都没打出来，最像这种接管。
    这里比对地址，发现不对就立刻重装（h3bt_install 里有"已经是我就不装"的判断）。
    """
    so = _STATE.get("bt_so")
    if so is None:
        return
    try:
        mine = int(_STATE.get("bt_self") or 0)
        cur = int(so.h3bt_status())
        if cur == mine:
            _say("[trace] %s：C 栈兜底仍在位（0x%x）" % (tag, cur))
            return
        _say("[trace] !! %s：SIGSEGV 处理器被换掉了（现在 0x%x，我 0x%x）→ 重装" % (tag, cur, mine))
        so.h3bt_install()
        cur = int(so.h3bt_status())
        _say("[trace]    重装后 0x%x（%s）" % (cur, "成功" if cur == mine else "还是被占着"))
    except Exception as exc:  # noqa: BLE001
        _say("[trace] 检查 C 栈兜底失败：%s: %s" % (type(exc).__name__, exc))


def _start_heartbeat(interval):
    if interval <= 0:
        return

    def run():
        last = time.perf_counter()
        while True:
            time.sleep(max(1.0, interval - (time.perf_counter() - last)))
            last = time.perf_counter()
            try:
                _maybe_patch()  # 万一两条 import 路径都漏了，心跳兜一次
            except Exception:  # noqa: BLE001
                pass
            _say(
                "[hb] t=+%.0fs stage=%s threads=%d"
                % (_now(), _STATE["stage"], threading.active_count())
            )
            try:
                faulthandler.dump_traceback(all_threads=True)
            except Exception:  # noqa: BLE001
                pass

    th = threading.Thread(target=run, name="h3-heartbeat", daemon=True)
    th.start()
    _say("[trace] 心跳线程已起：每 %ds 打一次阶段 + 全线程栈" % int(interval))


# ---------------------------------------------------------------- 块级进度


def _walk(block, depth=0):
    yield block, depth
    subs = getattr(block, "sub_blocks", None)
    if not subs:
        return
    try:
        children = list(subs.values())
    except Exception:  # noqa: BLE001
        return
    for sub in children:
        for item in _walk(sub, depth + 1):
            yield item


def _trace_class(cls):
    orig = cls.__dict__.get("__call__")
    if orig is None:
        # 类自己没定义 __call__，继承链上那个已经挂过了（或不需要挂）
        return False
    if getattr(orig, "_h3_traced", False):
        return False

    @functools.wraps(orig)
    def wrapper(self, *args, **kwargs):
        # 用 type(self) 而不是 patch 时那个类：很多 step 类不自己定义 __call__，
        # 继承的是 ConditionalPipelineBlocks/SequentialPipelineBlocks 的实现，
        # 只有按实例类型取名才看得出到底是哪一步。
        name = type(self).__name__
        depth = _STATE["depth"]
        _STATE["depth"] = depth + 1
        _STATE["stage"] = name
        pad = "  " * depth
        _say("[blk] %s>>> %s" % (pad, name))
        t0 = time.perf_counter()
        try:
            return orig(self, *args, **kwargs)
        finally:
            _STATE["depth"] = depth
            _say("[blk] %s<<< %s  %.1fs" % (pad, name, time.perf_counter() - t0))

    wrapper._h3_traced = True
    cls.__call__ = wrapper
    return True


def _patch_tree(pipe):
    if _STATE.get("tree_done"):
        return
    _STATE["tree_done"] = True
    blocks = getattr(pipe, "_blocks", None)
    if blocks is None:
        _say("[trace] pipe 上没有 _blocks，块级日志没挂上（不影响主流程）")
        return
    names = []
    seen = set()
    for block, _depth in _walk(blocks):
        cls = type(block)
        if cls in seen:
            continue
        seen.add(cls)
        try:
            if _trace_class(cls):
                names.append(cls.__name__)
        except Exception as exc:  # noqa: BLE001
            _say("[trace] 挂 %s 失败：%s: %s" % (cls.__name__, type(exc).__name__, exc))
    _say("[trace] 块级日志已挂 %d 个：%s" % (len(names), ", ".join(names)))


def _hook_pipeline():
    mod = sys.modules.get("diffusers.modular_pipelines.modular_pipeline")
    ModularPipeline = getattr(mod, "ModularPipeline", None) if mod is not None else None
    if ModularPipeline is None:
        from diffusers.modular_pipelines.modular_pipeline import ModularPipeline

    orig = ModularPipeline.__dict__.get("__call__")
    if orig is None or getattr(orig, "_h3_traced", False):
        return False

    @functools.wraps(orig)
    def wrapper(self, *args, **kwargs):
        _STATE["stage"] = "pipeline"
        _say("[blk] >>> pipeline.__call__")
        _patch_tree(self)
        t0 = time.perf_counter()
        try:
            return orig(self, *args, **kwargs)
        finally:
            _say("[blk] <<< pipeline.__call__  %.1fs" % (time.perf_counter() - t0))

    wrapper._h3_traced = True
    ModularPipeline.__call__ = wrapper
    return True


def _is_empty_shell(obj):
    """这个对象是不是「空壳 Tensor」—— impl_ 为 nullptr，碰任何成员都 SIGSEGV。

    只对 infinicore 的 Tensor 有意义；torch.Tensor / list / dict / None 一律 False
    （它们由调用方自己处理）。

    ⚠ 判空必须问到 `_underlying` 上，不能直接 `bool(obj)`：

      infinicore.Tensor 是 python 包装类（infinicore/tensor.py:25），它只重载了
      __getattr__ 把 shape/dtype/device 转发给 _underlying，**自己并没有 __bool__**；
      而特殊方法是按类型槽查的，压根不走 __getattr__ 转发。于是 `bool(包装对象)`
      落到 object 的默认实现 —— 永远 True。空壳照样往下走，下一行 `tuple(obj.shape)`
      就去解引用 nullptr（读 0x18 偏移）直接 SIGSEGV。

      11-08 12:03 那单按「先问 bool 再摸 shape」修过一次，15:08 这单又原地崩了、
      rip 偏移一模一样（_infinicore.so+0x10ff98、fault_addr=0x18），原因就是当年
      那一行判空其实从来没生效过。

      真正的 __bool__ 只绑在 C++ 层（pybind11/tensor.hpp:53 -> bool(tensor)
      -> impl_ != nullptr），所以先拿到 _underlying 再问它。
    """
    if type(obj).__name__ != "Tensor":
        return False
    # 包装类走 _underlying；万一是裸的 C++ Tensor（没有这个属性），就用它自己
    target = getattr(obj, "_underlying", obj)
    if target is None:
        return False
    try:
        return not bool(target)
    except Exception:  # noqa: BLE001
        return False


def _fact(obj):
    """把入参印成「形状 / 精度 / 落在哪」三件事，一行看完。

    段错误如果是「喂错了东西」引起的，这一行就能定案；如果三样都对，
    那就是算子或 kernel 层面的问题，再往下才需要 C 层栈。

    ⚠ 空 Tensor 必须先过 _is_empty_shell()，绝不能先摸 .shape。
    _infinicore.Tensor 的 shape 是 `tensor->shape()`——operator-> 拿到的是
    内部 impl 指针，空壳时它就是 nullptr，读成员等于读 0x18，直接 SIGSEGV
    （11-08 12:03 那单：si_code=1 fault_addr=0x18，栈上是
    _PyObject_GenericGetAttrWithDict → _infinicore.so）。而 SIGSEGV 不是
    Python 异常，外面套 try 完全拦不住——这里是唯一的关口。
    """
    # torch 先单独试，拿到就用；拿不到不能让后面的判断跟着 NameError
    # （踩过：`isinstance(obj, torch.Tensor)` 在 import 失败后引用未定义名，
    #  异常又被自己的 try 吞掉，于是空壳判断整段被跳过，照样摸到了 shape）
    torch = None
    try:
        import torch as _torch

        torch = _torch
    except Exception:  # noqa: BLE001
        pass

    if torch is not None:
        try:
            if isinstance(obj, torch.Tensor):
                return "torch.Tensor%s dtype=%s dev=%s" % (
                    tuple(obj.shape),
                    str(obj.dtype).replace("torch.", ""),
                    obj.device,
                )
        except Exception:  # noqa: BLE001
            pass

    # 只认 infinicore 的 Tensor（走到这里的一定不是 torch.Tensor，上面已 return），
    # 按类名认就够。判空交给 _is_empty_shell —— 它问到 _underlying 上，
    # 直接 bool(obj) 是无效的（包装类没有 __bool__，永远 True，见该函数注释）。
    if _is_empty_shell(obj):
        return "Tensor(空：C++ 侧没填这个字段)"
    shape = getattr(obj, "shape", None)
    if shape is not None:
        return "%s%s dtype=%s dev=%s" % (
            type(obj).__name__,
            tuple(shape),
            getattr(obj, "dtype", "?"),
            getattr(obj, "device", "?"),
        )
    if isinstance(obj, (list, tuple)):
        # 列表里装的东西才是关键：pixel_values / image_grid_thw / image_bound 全是
        # list(len=1)，光看个数等于没看。前 3 个元素各给一份形状/精度/设备。
        try:
            items = ["[%d] %s" % (i, _fact(v)) for i, v in enumerate(obj[:3])]
            more = "" if len(obj) <= 3 else " …(+%d)" % (len(obj) - 3)
            return "%s(len=%d) {%s%s}" % (type(obj).__name__, len(obj), ", ".join(items), more)
        except Exception:  # noqa: BLE001
            return "%s(len=%d)" % (type(obj).__name__, len(obj))
    if obj is None:
        return "None"
    return "%s(%r)" % (type(obj).__name__, obj if isinstance(obj, (int, float, str, bool)) else "…")


def _brief_out(out):
    """一行说清 forward_raw 返回了什么（哪些字段有值、哪些是空壳）。

    这一行是 11-08 12:03 那单的结论来源：condition_encoder_mode 下 C++ 只填了
    logits（内容其实是第 50 层 hidden state，encode() 正是靠它），hidden_states
    是空 Tensor —— 谁去摸它的 .shape 谁就 SIGSEGV。所以打印必须过 _fact，
    不能直接 tuple(t.shape)。
    """
    try:
        if isinstance(out, dict):
            return " ".join("%s=%s" % (k, _fact(v)) for k, v in sorted(out.items()))
    except Exception:  # noqa: BLE001
        pass
    return _fact(out)


def _probe_guard(fn, tag):
    """跑一个自检；它自己报错就记一行，绝不能把主流程带崩。"""
    try:
        return fn()
    except BaseException as exc:  # noqa: BLE001
        msg = (str(exc) or "").splitlines()
        _say("[probe] !! %s 自己报错（不是段错误）：%s: %s"
             % (tag, type(exc).__name__, (msg[0][:200] if msg else "")))
        return None


def _probe_ops():
    """A：infinicore 微算子，8 张卡逐个走一遍。

    挂了 = 算子分发层本身是坏的，跟 H3 的模型代码无关。
    为什么要逐卡：TP=8 是「一个进程里 8 张卡」，只要有一张卡的 device
    没被 infinicore 真正认下来（ctx / stream 没建），第一个 forward 就会
    在某一路 rank 上崩，而日志上看起来只是"进了 C++ 就没了"。
    """
    import torch

    import infinicore

    try:
        ndev = int(torch.cuda.device_count())
    except Exception:  # noqa: BLE001
        ndev = 1
    _say("[probe] A0 torch 看到 %d 张卡" % ndev)
    ndev = max(1, min(ndev, 8))
    has_mm = hasattr(infinicore, "matmul")
    exp = (32.0 * 32.0 * 32.0) if has_mm else (32.0 * 32.0 * 2.0)
    bad = []
    for d in range(ndev):
        dev = "cuda:%d" % d
        try:
            at = torch.ones((32, 32), dtype=torch.float32, device=dev)
            bt = torch.ones((32, 32), dtype=torch.float32, device=dev)
            _say("[probe] A[%d] from_torch" % d)
            a = infinicore.from_torch(at)
            b = infinicore.from_torch(bt)
            c = infinicore.matmul(a, b) if has_mm else infinicore.add(a, b)
            if hasattr(infinicore, "sync_device"):
                infinicore.sync_device()
            out = torch.empty((32, 32), dtype=torch.float32, device=dev)
            infinicore.from_torch(out).copy_(c)
            if hasattr(infinicore, "sync_device"):
                infinicore.sync_device()
            s = float(out.sum().item())
            ok = abs(s - exp) < 0.5
            _say("[probe] A[%d] sum=%.1f 期望=%.1f → %s"
                 % (d, s, exp, "OK" if ok else "数值不对"))
            if not ok:
                bad.append(d)
        except BaseException as exc:  # noqa: BLE001
            msg = (str(exc) or "").splitlines()
            _say("[probe] !! A[%d] 报错：%s: %s"
                 % (d, type(exc).__name__, (msg[0][:160] if msg else "")))
            bad.append(d)
    return "全过" if not bad else ("坏的卡 %s" % bad)


def _probe_text_only(orig, engine):
    """B：同一台 engine 的纯文本 forward（4 个 token、不带 pixel_values）。

    11-08 12:03 这单把它跑通了：50 层全跑完（allreduce 日志只打前 3 次，后面
    静默 —— 别被「只有 #1/#2/#3」骗成「只跑了 3 层」），返回
    {logits=有效, hidden_states=空, output_ids={0}}。当时崩在**打印返回值**：
    摸空 hidden_states 的 .shape → nullptr+0x18。所以现在这个自检的结论是
    「主路没问题」，它只剩回归验证的价值，默认不再跑（H3_FRPROBE=1 可开）。

    代价：往 KV cache 写了 4 个槽；之后真调用从 0 开始覆盖，smoke 可以接受。
    """
    import torch

    import infinicore

    n = 4
    dev = "cuda:0"
    ids = torch.tensor([[11, 22, 33, 44]], dtype=torch.int64, device=dev)
    pos = torch.arange(n, dtype=torch.int64, device=dev).unsqueeze(0).repeat(3, 1).contiguous()
    kw = {
        "position_ids": infinicore.from_torch(pos),
        "past_kv_lengths": infinicore.from_torch(torch.zeros(1, dtype=torch.int32, device=dev)),
        "total_kv_lengths": infinicore.from_torch(torch.tensor([n], dtype=torch.int32, device=dev)),
        "input_offsets": infinicore.from_torch(torch.tensor([0, n], dtype=torch.int32, device=dev)),
    }
    _say("[probe] B0 纯文本 forward 开始（%d token、无 pixel_values）" % n)
    out = orig(engine, infinicore.from_torch(ids), **kw)
    if isinstance(out, dict):
        _say("[probe] B1 返回：%s"
             % ", ".join("%s=%s" % (k, _fact(v)) for k, v in sorted(out.items())))
    else:
        _say("[probe] B1 返回：%s" % _fact(out))
    return True


def _hook_forward_raw():
    """给 InferEngine.forward_raw 挂日志。

    为什么盯这个函数：11-08 05:58 那单崩在这里（infer_engine.py:548），
    而 Python 栈到 `super().forward(...)` 就断了——底下全是 C++。
    所以至少要留下「进去的时候手里拿的是什么」：哪些张量、什么形状、
    什么精度、在哪张卡上。第一次崩之前能打出来的就这三行。
    """
    try:
        # 关键：不能在这时候 `from infinilm.infer_engine import InferEngine`。
        # sys.modules 里的条目在 exec 之前就建好了，模块正文还没跑完时 import
        # 拿到的是半成品，getattr 会扑空（ModularPipeline 那次就是这么栽的）。
        # 所以先看 sys.modules 里那个对象上有没有类，没有就等下一次机会。
        mod = sys.modules.get("infinilm.infer_engine")
        InferEngine = getattr(mod, "InferEngine", None) if mod is not None else None
        if InferEngine is None:
            return False
    except Exception:  # noqa: BLE001
        return False

    orig = getattr(InferEngine, "forward_raw", None)
    if orig is None or getattr(orig, "_h3_traced", False):
        return False

    seen = {"n": 0}

    def wrapper(self, input_ids, **kwargs):
        seen["n"] += 1
        n = seen["n"]
        if n <= 2:
            try:
                lines = ["[fr] >>> forward_raw #%d" % n, "      input_ids: %s" % _fact(input_ids)]
                for k in sorted(kwargs):
                    if kwargs[k] is not None:
                        lines.append("      %s: %s" % (k, _fact(kwargs[k])))
                _say("\n".join(lines))
            except Exception:  # noqa: BLE001
                pass
        else:
            _say("[fr] >>> forward_raw #%d" % n)
        if n == 1:
            # 真调用之前把「崩在哪儿」劈成两半，顺手确认 C 栈兜底没被 HIP 顶掉。
            # 11-08 12:03 那单已经用它出结论了（A 算子全过、B 跑到 forward 结束），
            # 所以默认关 —— probe 会往 KV cache 里写 4 个槽，也会多花一两分钟，
            # 而它想证明的两件事现在都成立了。要复跑：H3_FRPROBE=1
            if os.environ.get("H3_FRPROBE", "0").strip().lower() not in ("0", "false", "no"):
                _probe_guard(_probe_ops, "A 微算子")
                _probe_guard(lambda: _probe_text_only(orig, self), "B 纯文本 forward")
                _STATE["stage"] = "text encoder forward"
            _bt_check("[fr] 真调用前")
            _say("[fr] --- 自检结束，开始真调用（下面这行之后就是 C++）---")
        t0 = time.perf_counter()
        out = orig(self, input_ids, **kwargs)
        # 返回值摘要：11-08 12:03 那单就是死在「打印返回值」这一步 —— C++ 在
        # condition_encoder_mode 下只填了 {logits}（内容其实是第 50 层 hidden
        # state），hidden_states 留空；谁去摸空壳的 .shape 谁就 SIGSEGV。
        # 现在一律走 _fact（空壳先问 bool），哪几个字段有值一眼可见。
        _say("[fr] <<< forward_raw #%d  %.1fs  ← %s"
             % (n, time.perf_counter() - t0, _brief_out(out)))
        return out

    wrapper._h3_traced = True
    InferEngine.forward_raw = wrapper
    return True


def _hook_to_torch_guard():
    """给 MiniMaxH3TransformerRunner._to_torch 套一层「空壳先拦」。

    这个函数是 native infinicore.Tensor -> torch.Tensor 的唯一漏斗，第一行就是
    `tuple(tensor.shape)`。也就是说：只要上游任何一环塞进来一个空壳，它就会
    以 SIGSEGV（fault_addr=0x18）收场，日志里只剩一串没有名字的栈 —— 与 12:03
    / 15:08 两单的现场完全一样，很难第一眼看出是谁给的。

    包一层把它变成**可读的 Python 异常**：崩也要崩在有名字、有 message 的地方。
    代价只有一次 isinstance 级别的判断，且异常路径本来就不会被走到。
    """
    mod = sys.modules.get("infinilm.multimodal.minimax_h3")
    cls = getattr(mod, "MiniMaxH3TransformerRunner", None) if mod is not None else None
    if cls is None:
        return False
    orig = getattr(cls, "_to_torch", None)
    if orig is None or getattr(orig, "_h3_guarded", False):
        return False

    def guard(tensor, device):
        if _is_empty_shell(tensor):
            raise RuntimeError(
                "native 侧给过来的是空 Tensor（impl_ 为 nullptr），转 torch 必崩；"
                "condition_encoder_mode 下 C++ 把 hidden state 填进的是 logits 字段，"
                "hidden_states 留空 —— 检查调用方读的是哪个字段"
            )
        return orig(tensor, device)

    guard._h3_guarded = True
    cls._to_torch = staticmethod(guard)
    return True


def _hook_tp(module_name, class_name, env_name, key, who):
    """<env_name>=N：<who>只用 N 张卡（主线照样全卡）。

    为什么要这个开关：文本塔的活只是把 76 个 token 的 prompt 编成 embedding，
    一遍 prefill，TP=8 对它一分钱收益都没有；但 TP=8 要多走一条 allreduce 和
    分片边界，而 11-08 那两单的 SIGSEGV 恰好就在这个 forward 里 —— 崩的位置
    离 TP 的分片逻辑太近了。降卡等于把整段路径绕开，是「先出一帧视频」的保底
    手段，不用碰任何算子。DiT 那边同理：平台只给 8 卡时，这是唯一还能改分片数
    的杠杆（retry4 / retry2 那条路走不通）。

    副作用只有一个：权重从 N 份变成 1 份全量，首卡显存占用上升。真 OOM 会报
    allocation 失败，不是段错误，看日志就能分清。

    默认不开（不设 = 现在的全卡行为），免得给定位引入新变量。
    """
    want = os.environ.get(env_name, "").strip()
    if want in ("", "0"):
        return False
    try:
        n = int(want)
    except ValueError:
        # key 加 patched_ 前缀，和 _maybe_patch 里用同一个去重键：两处都判定
        # 「不是正整数」时只留一句，免得日志里同一件事说两遍
        _once("patched_" + key + "_bad",
              "[trace] %s=%r 不是整数，按不降处理" % (env_name, want))
        return False
    if n <= 0:
        return False

    # 和 forward_raw 一样：只认「对象上真的有这个类」，不认 sys.modules 里的名字。
    # 模块正文可能还在 exec，此时类还没定义，返回 False 等下一次。
    mod = sys.modules.get(module_name)
    cls = getattr(mod, class_name, None) if mod is not None else None
    if cls is None:
        return False

    orig = getattr(cls, "__init__", None)
    if orig is None or getattr(orig, "_h3_tp", False):
        return False

    def wrapper(self, *args, **kwargs):
        ids = kwargs.get("tp_device_ids")
        if ids is None:
            # 位置参数形态：tp_device_ids 是第 5 个（path, *, ... 全 keyword-only），
            # 实际调用方不会这么传，真出现了就原样放过去并留一句话。
            _once(key + "_noids",
                  "[trace] %s没拿到 tp_device_ids，%s 这次没生效" % (who, env_name))
            return orig(self, *args, **kwargs)
        ids = [int(x) for x in ids]
        keep = min(n, len(ids))
        if keep >= len(ids):
            _once(key + "_noop", "[trace] %s已是 %d 卡，%s=%d 不用降"
                  % (who, len(ids), env_name, n))
            return orig(self, *args, **kwargs)
        kwargs["tp_device_ids"] = ids[:keep]
        # tp_size 入口没显式传（默认 1），但万一哪天传了 8，这里必须一起改，
        # 否则构造函数里 tp_size 与 len(tp_device_ids) 对不上会直接 raise。
        if kwargs.get("tp_size") not in (None, 1):
            kwargs["tp_size"] = keep
        _say("[trace] %s降 TP：%d 卡 → %d 卡（%s）—— 只为绕开 TP 分片路径"
             % (who, len(ids), keep, ",".join(str(x) for x in kwargs["tp_device_ids"])))
        return orig(self, *args, **kwargs)

    wrapper._h3_tp = True
    cls.__init__ = wrapper
    return True


def _hook_text_tp():
    return _hook_tp("infinilm.multimodal.minimax_h3", "MiniMaxH3ConditionEncoder",
                    "H3_TEXT_TP", "ttp", "文本编码器")


def _hook_dit_tp():
    # DiT 也走 InferEngine + 同一份 attention.cpp，所以 heads 不整除时一样会崩。
    # 但它比文本塔大得多，降卡只到「不 OOM」为止，别一上来就填 1。
    return _hook_tp("infinilm.multimodal.minimax_h3", "MiniMaxH3TransformerRunner",
                    "H3_DIT_TP", "dtp", "DiT 侧")


def _maybe_patch():
    """目标模块一旦就绪就立刻挂上（幂等，可反复调用）。

    三条纪律：
    - 两件事互相独立：pipeline 挂了不该拖累 forward_raw，反之亦然
    - 只有真挂上才置位；没挂上就留着重试（模块可能还在 exec 里）
    - 判断「就绪」看对象上有没有那个类，不看 sys.modules 里有没有那个名字
    """
    if not _STATE["patched"]:
        mod = sys.modules.get("diffusers.modular_pipelines.modular_pipeline")
        if mod is not None and getattr(mod, "ModularPipeline", None) is not None:
            try:
                if _hook_pipeline():
                    _STATE["patched"] = True
                    _say("[trace] 已挂 ModularPipeline.__call__（等第一次 pipe(...) 再铺开）")
            except Exception as exc:  # noqa: BLE001
                _say("[trace] 挂 pipeline 失败：%s: %s" % (type(exc).__name__, exc))

    if not _STATE["patched_fr"]:
        try:
            if _hook_forward_raw():
                _STATE["patched_fr"] = True
                _say("[trace] 已挂 InferEngine.forward_raw（会打印入参形状/精度/设备）")
        except Exception as exc:  # noqa: BLE001
            _say("[trace] 挂 forward_raw 失败：%s: %s" % (type(exc).__name__, exc))

    if not _STATE["patched_tt"]:
        try:
            if _hook_to_torch_guard():
                _STATE["patched_tt"] = True
                _say("[trace] 已给 _to_torch 加空壳拦截（空 Tensor 转 torch 会抛异常而不是 SEGV）")
        except Exception as exc:  # noqa: BLE001
            _say("[trace] 挂 _to_torch 失败：%s: %s" % (type(exc).__name__, exc))

    # 两个降卡开关走同一套逻辑：没开/值错就置位（结论不会再变），命中才挂。
    # 分开置位是必要的：DiT 的类要等 minimax_h3 正文 exec 到 404 行之后才存在，
    # 不能因为文本塔挂上了就把 DiT 也当成处理完。
    for _env, _key, _hook, _label in (
        ("H3_TEXT_TP", "patched_ttp", _hook_text_tp, "MiniMaxH3ConditionEncoder"),
        ("H3_DIT_TP", "patched_dtp", _hook_dit_tp, "MiniMaxH3TransformerRunner"),
    ):
        if _STATE[_key]:
            continue
        _want = os.environ.get(_env, "").strip()
        if _want in ("", "0"):
            _STATE[_key] = True
            continue
        if not _want.isdigit():
            # 值给错了就等于不降，而且这个结论不会再变——置位，别每次 import 重解析
            _once(_key + "_bad", "[trace] %s=%r 不是正整数，按不降处理" % (_env, _want))
            _STATE[_key] = True
            continue
        try:
            if _hook():
                _STATE[_key] = True
                _say("[trace] 已挂 %s.__init__（%s 生效）" % (_label, _env))
        except Exception as exc:  # noqa: BLE001
            _say("[trace] 挂 %s 降 TP 失败：%s: %s" % (_label, type(exc).__name__, exc))


def _hook_import():
    """两条路径都要覆盖：

    - `import x` / `from x import y` 走 builtins.__import__
    - diffusers 顶层是 _LazyModule，`from diffusers import ModularPipeline` 只是
      __getattr__ 触发 importlib.import_module，不经过 __import__

    所以两个都包一层。反正 _maybe_patch 幂等，多调几次没成本。
    """
    orig_import = builtins.__import__
    if not getattr(orig_import, "_h3_hooked", False):

        def hook(name, globals=None, locals=None, fromlist=(), level=0):
            mod = orig_import(name, globals, locals, fromlist, level)
            _maybe_patch()
            return mod

        hook._h3_hooked = True
        builtins.__import__ = hook

    import importlib

    orig_im = importlib.import_module
    if not getattr(orig_im, "_h3_hooked", False):

        def im_hook(name, package=None):
            mod = orig_im(name, package)
            _maybe_patch()
            return mod

        im_hook._h3_hooked = True
        importlib.import_module = im_hook


# ---------------------------------------------------------------- 入口


def install():
    if _STATE["installed"]:
        return True
    _STATE["installed"] = True

    if os.environ.get("H3_TRACE", "1").strip() in ("0", "false", "no"):
        _say("[trace] H3_TRACE=0，全部关掉")
        return False

    _enable_faulthandler()
    _install_native_bt()

    try:
        interval = float(os.environ.get("H3_HB", "45"))
    except ValueError:
        interval = 45.0
    _start_heartbeat(interval)

    try:
        _hook_import()
    except Exception as exc:  # noqa: BLE001
        _say("[trace] 装 import 钩子失败：%s: %s" % (type(exc).__name__, exc))

    return True
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
  if cmp -s "$tmpt" "$dyn/h3_trace.py" 2>/dev/null; then
    rm -f "$tmpt"
  else
    mv -f "$tmpt" "$dyn/h3_trace.py" 2>/dev/null && changed=1
  fi

  [ -s "$dyn/h3_proc_fix.py" ] && [ -s "$entry" ] && [ -s "$dyn/h3_trace.py" ] \
    || { say "  !! processor/现场保护补丁没写出来"; return 1; }
  H3_ENTRY_PY=$entry
  export H3_ENTRY_PY
  if [ $changed -eq 1 ]; then
    say "  processor 兜底已更新：$dyn/h3_proc_fix.py + $dyn/h3_trace.py + $entry"
  else
    say "  processor 兜底已是最新（内容未变）"
  fi
  return 0
}
# ---- write_proc_fix 结束 ----（上面内嵌的 Python 里有独立成行的 `}`，测试脚本靠这行定界）

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
      *'FRAMES=8; STEPS=2;'*)
        # 15:41 那单的结论：smoke 的 frames=8 根本不被接受。
        # before_denoise.py:406 会先把它「向上取到最近的 17n+5」= 22，再校验必须落在
        # [120,360] —— 22 直接 ValueError。视频 VAE 只能编码 17n+5 帧，所以 8 帧这种
        # 小样永远不可能合法，最小合法值就是 124（17*7+5）。
        # 这里就地改掉，不依赖重打 tarball：包目录是别的 uid 的只读树，改了也不生效。
        out+='  FRAMES=${H3_SMOKE_FRAMES:-124}; STEPS=${H3_SMOKE_STEPS:-2}; HW_ARGS="--height 256 --width 256"; TAG=smoke'$'\n'
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
  say "  smoke 帧数 = ${H3_SMOKE_FRAMES:-124}（17n+5 且 ≥120；8 帧会被 before_denoise.py 直接拒）"
  return 0
}

# 原生栈兜底：Python 栈只能停在「调进 C++ 的那一行」（这次就是 infer_engine.py:548），
# 底下到底崩在哪个 .so、哪段偏移，faulthandler 一个字节都给不出来。gcc 就在手上
# （薄壳已经在用它编），所以自己编一个信号处理器：SIGSEGV 时先打 C 层栈，
# 再把控制权交回 faulthandler 打 Python 栈——两边都留，不互相覆盖。
# 装法见 h3_trace：必须等 faulthandler.enable() 之后再装，才排得上队。
build_bt_shim() {
  local dir=$AFS/cudashim
  local src=$dir/h3_bt.c
  local tmps=$dir/.h3_bt.c.new
  local so=$dir/libh3bt.so
  mkdir -p "$dir" 2>/dev/null || return 1

  # 先写临时文件再比对内容：直接覆盖 $src 会把 mtime 刷新成「刚刚」，
  # 后面「so 比 src 新就复用」永远不成立，每单都白编一次。
  cat > "$tmps" <<'H3BTEOF'
/* ====================================================================
 * h3_bt.c —— 原生（C/C++）栈兜底  v3
 *
 * v1 全哑：把 backtrace() 放最前，栈一坏它自己先崩，一个字都写不出来。
 * v2 能进了 —— 11-08 11:48 那单日志里确实出现了 maps 转储。但顺序还是错：
 * 它在 unwind 之前把 /proc/self/maps 全量倒出来（3000 多行），平台日志缓冲
 * 被冲爆，sig/RIP/出错模块 和 后面的 C 栈一起被挤掉 —— 等于用最没用的信息
 * 换掉了最有用的那几行。
 *
 * v3 按「信息密度」重排，越靠前越不可能失败也越关键：
 *   ① 一条常量串                —— 它出现就证明处理器真被调到了
 *   ② siginfo + RIP/RSP/RBP     —— 纯内存读
 *   ③ dladdr(RIP)               —— 出错指令在哪个 .so 的哪个偏移
 *   ④ unwind，每帧都过 dladdr    —— C 栈，格式统一是「模块 + 偏移」
 *   ⑤ maps 里只挑命中的那几行    —— RIP/RSP/RBP 各一行，不再全量
 *   ⑥ 把信号交回被顶掉的处理器（Python 的 faulthandler）
 * 完整 maps 只在 H3_BT_MAPS=1 时要，日常绝不开。
 *
 * 另外三个入口：
 *   h3bt_selftest()  fork 一个子进程故意 SIGSEGV，验证整条链是活的
 *   h3bt_status()    现在 SIGSEGV 上挂的是谁的地址
 *   h3bt_self()      我自己的地址（跟 status 比就知道有没有被顶掉）
 * ==================================================================== */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <execinfo.h>
#include <fcntl.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>
#include <ucontext.h>

#define H3BT_MAX 64
#define H3BT_NSIG 4
#define H3BT_ALTBUF (64 * 1024)

static const int h3_sigs[H3BT_NSIG] = {SIGSEGV, SIGABRT, SIGBUS, SIGILL};
static struct sigaction h3_prev[H3BT_NSIG];
static volatile sig_atomic_t h3_busy = 0;
static volatile sig_atomic_t h3_selftest = 0;
static char h3_altbuf[H3BT_ALTBUF];
/* install 时读一次：信号处理器里不碰 getenv */
static int h3_want_maps = 0;

static void h3w(const char *s) {
    ssize_t r = write(2, s, strlen(s));
    (void)r;
}

/* 固定宽度十六进制：不用 snprintf，信号处理器里少一个不确定因素 */
static void h3hex(unsigned long v) {
    char b[19];
    int i;
    b[0] = '0';
    b[1] = 'x';
    for (i = 0; i < 16; i++) {
        unsigned int d = (unsigned int)((v >> (60 - 4 * i)) & 0xfUL);
        b[2 + i] = (char)(d < 10 ? ('0' + d) : ('a' + (d - 10)));
    }
    b[18] = '\0';
    h3w(b);
}

static void h3dec(long v) {
    char b[24];
    int i = 23;
    unsigned long u;
    b[i] = '\0';
    i--;
    u = (v < 0) ? (unsigned long)(-(v + 1)) + 1UL : (unsigned long)v;
    if (u == 0UL) {
        b[i] = '0';
        i--;
    }
    while (u != 0UL && i >= 0) {
        b[i] = (char)('0' + (int)(u % 10UL));
        u /= 10UL;
        i--;
    }
    if (v < 0 && i >= 0) {
        b[i] = '-';
        i--;
    }
    h3w(&b[i + 1]);
}

/* maps 行开头的十六进制数。手写而不用 strtoul：后者不在 async-signal-safe 名单里 */
static unsigned long h3hexval(const char **pp, int *ok) {
    const char *p = *pp;
    unsigned long v = 0UL;
    int n = 0;
    *ok = 0;
    while (n < 16) {
        char c = *p;
        int d;
        if (c >= '0' && c <= '9') {
            d = c - '0';
        } else if (c >= 'a' && c <= 'f') {
            d = c - 'a' + 10;
        } else if (c >= 'A' && c <= 'F') {
            d = c - 'A' + 10;
        } else {
            break;
        }
        v = (v << 4) | (unsigned long)d;
        p++;
        n++;
    }
    if (n > 0) {
        *ok = 1;
        *pp = p;
    }
    return v;
}

static int h3_in(unsigned long a, unsigned long s, unsigned long e) {
    return (a != 0UL) && (s <= a) && (a < e);
}

/* 只打 RIP / RSP / RBP 各自所在的那一行。
   v2 在这里倒全量，把日志冲爆，反而丢了 sig/RIP —— 这次的教训。 */
static void h3maps_keys(unsigned long a1, unsigned long a2, unsigned long a3) {
    int fd;
    static char lbuf[1024];
    int llen = 0;
    int hit1 = 0, hit2 = 0, hit3 = 0;
    char buf[4096];
    ssize_t k;

    fd = open("/proc/self/maps", O_RDONLY);
    if (fd < 0) {
        h3w("[h3bt] （/proc/self/maps 打不开）\n");
        return;
    }
    h3w("[h3bt] --- maps 命中段（只列 RIP/RSP/RBP 所在的，不打全量）---\n");
    while ((k = read(fd, buf, sizeof buf)) > 0) {
        ssize_t i;
        for (i = 0; i < k; i++) {
            char c = buf[i];
            int eol = (c == '\n') || (llen >= (int)sizeof lbuf - 1);
            if (!eol) {
                lbuf[llen++] = c;
                continue;
            }
            lbuf[llen] = '\0';
            if (llen > 0) {
                const char *p = lbuf;
                int ok1 = 0, ok2 = 0;
                unsigned long s = h3hexval(&p, &ok1);
                if (ok1 && *p == '-') {
                    unsigned long e;
                    p++;
                    e = h3hexval(&p, &ok2);
                    if (ok2) {
                        const char *tag = NULL;
                        if (!hit1 && h3_in(a1, s, e)) {
                            tag = "        <<< RIP（出错指令）";
                            hit1 = 1;
                        } else if (!hit2 && h3_in(a2, s, e)) {
                            tag = "        <<< RSP（出事时的栈顶）";
                            hit2 = 1;
                        } else if (!hit3 && h3_in(a3, s, e)) {
                            tag = "        <<< RBP（栈帧基址）";
                            hit3 = 1;
                        }
                        if (tag != NULL) {
                            h3w("[h3bt] ");
                            h3w(lbuf);
                            h3w(tag);
                            h3w("\n");
                        }
                    }
                }
            }
            llen = 0;
        }
    }
    close(fd);
    if (!hit1) {
        h3w("[h3bt] （RIP 没落在任何已映射段里 —— 说明是跳飞了，不是踩数据）\n");
    }
}

/* 全量 maps：只在 H3_BT_MAPS=1 时调用。留给「事后要自己反查段」的场合。 */
static void h3maps_all(void) {
    int fd = open("/proc/self/maps", O_RDONLY);
    char buf[4096];
    ssize_t k;
    if (fd < 0) {
        h3w("[h3bt] （/proc/self/maps 打不开）\n");
        return;
    }
    h3w("[h3bt] --- /proc/self/maps 全量（H3_BT_MAPS=1）---\n");
    while ((k = read(fd, buf, sizeof buf)) > 0) {
        ssize_t off = 0;
        while (off < k) {
            ssize_t r = write(2, buf + off, (size_t)(k - off));
            if (r <= 0) {
                break;
            }
            off += r;
        }
    }
    close(fd);
    h3w("[h3bt] --- maps end ---\n");
}

static void h3_frame(void *pc, int i) {
    const char *mod = "?";
    unsigned long off = 0UL;
    Dl_info info;
    memset(&info, 0, sizeof info);
    if (dladdr(pc, &info) != 0 && info.dli_fname != NULL) {
        mod = info.dli_fname;
        if (info.dli_fbase != NULL) {
            off = (unsigned long)((char *)pc - (char *)info.dli_fbase);
        }
    }
    h3w("[h3bt] #");
    if (i < 10) {
        h3w(" ");
    }
    h3dec((long)i);
    h3w(" ");
    h3w(mod);
    h3w(" + ");
    h3hex(off);
    /* 有符号名就顺手带上，省一次手工 addr2line */
    if (info.dli_sname != NULL && info.dli_saddr != NULL) {
        h3w("   ");
        h3w(info.dli_sname);
        h3w(" + ");
        h3hex((unsigned long)((char *)pc - (char *)info.dli_saddr));
    }
    h3w("\n");
}

static void h3_handler(int sig, siginfo_t *si, void *uc_) {
    ucontext_t *uc = (ucontext_t *)uc_;
    void *frames[H3BT_MAX];
    unsigned long rip = 0UL, rsp = 0UL, rbp = 0UL;
    int n, i;

    if (h3_busy) {
        h3w("\n[h3bt] !! 重入：上一轮正在打栈时又崩了，直接退出\n");
        _exit(128 + sig);
    }
    h3_busy = 1;

    /* ① 先落地一条常量串 */
    h3w("\n[h3bt] >>> CAUGHT sig=");
    h3dec((long)sig);
    h3w(" (");
    if (sig == SIGSEGV) {
        h3w("SIGSEGV");
    } else if (sig == SIGABRT) {
        h3w("SIGABRT");
    } else if (sig == SIGBUS) {
        h3w("SIGBUS");
    } else if (sig == SIGILL) {
        h3w("SIGILL");
    } else {
        h3w("?");
    }
    h3w(")\n");

    /* ② siginfo + 出事那条指令的地址 */
    h3w("[h3bt] si_code=");
    h3dec(si ? (long)si->si_code : 0L);
    h3w("  fault_addr=");
    h3hex(si ? (unsigned long)(uintptr_t)si->si_addr : 0UL);
    h3w("\n");

#ifdef __x86_64__
    if (uc != NULL) {
        rip = (unsigned long)(uintptr_t)uc->uc_mcontext.gregs[REG_RIP];
        rsp = (unsigned long)(uintptr_t)uc->uc_mcontext.gregs[REG_RSP];
        rbp = (unsigned long)(uintptr_t)uc->uc_mcontext.gregs[REG_RBP];
        h3w("[h3bt] rip=");
        h3hex(rip);
        h3w(" rsp=");
        h3hex(rsp);
        h3w(" rbp=");
        h3hex(rbp);
        h3w("\n");
    }
#else
    (void)uc;
#endif

    if (h3_selftest) {
        /* 自检子进程：走到这里已经证明「处理器能进、能写」 */
        h3w("[h3bt] selftest：整条链是活的（子进程到此为止）\n");
        _exit(128 + sig);
    }

    /* ③ 出错指令落在哪个模块的哪个偏移 */
#ifdef __x86_64__
    if (uc != NULL && rip != 0UL) {
        Dl_info info;
        memset(&info, 0, sizeof info);
        if (dladdr((void *)(uintptr_t)rip, &info) != 0 && info.dli_fname != NULL) {
            h3w("[h3bt] 出错指令在 ");
            h3w(info.dli_fname);
            h3w(" + ");
            h3hex(info.dli_fbase ? (rip - (unsigned long)(uintptr_t)info.dli_fbase) : 0UL);
            if (info.dli_sname != NULL) {
                h3w("   最近符号 ");
                h3w(info.dli_sname);
                h3w(" + ");
                h3hex(rip - (unsigned long)(uintptr_t)info.dli_saddr);
            }
            h3w("\n");
        } else {
            h3w("[h3bt] 出错指令不在任何已加载模块里（dladdr 没查到）\n");
        }
    }
#endif

    /* ④ 现在才 unwind：每帧都过 dladdr，输出统一是「模块 + 偏移」 */
    n = backtrace(frames, H3BT_MAX);
    h3w("[h3bt] --- C 栈（每帧 = 模块 + 偏移）---\n");
    for (i = 0; i < n; i++) {
        h3_frame(frames[i], i);
    }
    if (n >= H3BT_MAX) {
        h3w("[h3bt] （栈更深，只留了前 ");
        h3dec((long)H3BT_MAX);
        h3w(" 层）\n");
    }
    h3w("===== [h3bt] END BACKTRACE =====\n");

    /* ⑤ maps 只挑命中行。全量要显式开 —— v2 就是在这把日志冲爆的 */
    h3maps_keys(rip, rsp, rbp);
    if (h3_want_maps) {
        h3maps_all();
    }

    /* ⑥ 交回原处理器：faulthandler 接着打 Python 栈，然后照常死掉 */
    for (i = 0; i < H3BT_NSIG; i++) {
        if (h3_sigs[i] == sig) {
            sigaction(sig, &h3_prev[i], NULL);
            break;
        }
    }
    raise(sig);
    _exit(128 + sig);
}

int h3bt_install(void) {
    struct sigaction sa;
    struct sigaction cur;
    stack_t ss;
    const char *env;
    int i;
    int rc = 0;

    /* install 不是信号处理器，可以安心读环境变量；handler 里只认这个静态位 */
    env = getenv("H3_BT_MAPS");
    h3_want_maps = (env != NULL && env[0] == '1') ? 1 : 0;

    /* 主线程开一块备用栈：万一是栈溢出引起的 SIGSEGV，
     * 没有 SA_ONSTACK 的话处理器根本跑不起来（直接二次失败）。 */
    ss.ss_sp = h3_altbuf;
    ss.ss_size = sizeof h3_altbuf;
    ss.ss_flags = 0;
    (void)sigaltstack(&ss, NULL);

    for (i = 0; i < H3BT_NSIG; i++) {
        /* 已经是我了就别再装：重复装会把 h3_prev 覆盖成自己，
         * 崩的时候"交回原处理器"变成无限递归。 */
        if (sigaction(h3_sigs[i], NULL, &cur) == 0 && cur.sa_sigaction == h3_handler) {
            continue;
        }
        memset(&sa, 0, sizeof sa);
        sa.sa_sigaction = h3_handler;
        sa.sa_flags = SA_SIGINFO | SA_ONSTACK;
        sigemptyset(&sa.sa_mask);
        if (sigaction(h3_sigs[i], &sa, &h3_prev[i]) != 0) {
            rc = -1;
        }
    }
    return rc;
}

/* 现在 SIGSEGV 上挂的是谁的地址 */
unsigned long h3bt_status(void) {
    struct sigaction cur;
    if (sigaction(SIGSEGV, NULL, &cur) != 0) {
        return 0UL;
    }
    return (unsigned long)(uintptr_t)cur.sa_sigaction;
}

unsigned long h3bt_self(void) { return (unsigned long)(uintptr_t)h3_handler; }

/* 在一个子进程里故意崩一次：不改主进程状态，又能证明整条链是活的。
 * 返回 139 表示"子进程确实被 SIGSEGV 干掉了"，也就是处理器链正常。 */
int h3bt_selftest(void) {
    pid_t pid;
    int st = 0;
    pid = fork();
    if (pid < 0) {
        return -1;
    }
    if (pid == 0) {
        h3_selftest = 1;
        h3_busy = 0;
        raise(SIGSEGV);
        _exit(71); /* 走到这儿说明信号被吞了 */
    }
    if (waitpid(pid, &st, 0) < 0) {
        return -2;
    }
    if (WIFSIGNALED(st)) {
        return 128 + WTERMSIG(st);
    }
    if (WIFEXITED(st)) {
        return WEXITSTATUS(st);
    }
    return -3;
}

const char *h3bt_version(void) { return "h3bt/3"; }
H3BTEOF

  # 内容没变就直接复用编好的 .so（每单都重编没必要，也省 gcc 存在与否的抖动）
  if [ -s "$so" ] && [ -s "$src" ] && cmp -s "$tmps" "$src"; then
    rm -f "$tmps" 2>/dev/null
    export H3_BT_LIB=$so
    return 0
  fi
  mv -f "$tmps" "$src" 2>/dev/null || cp -f "$tmps" "$src" 2>/dev/null
  rm -f "$tmps" 2>/dev/null

  local cc="" c
  for c in gcc cc clang "$DTK_ROOT/bin/clang"; do
    [ -n "$c" ] || continue
    if command -v "$c" >/dev/null 2>&1; then cc=$c; break; fi
  done
  if [ -z "$cc" ]; then
    say "  !! 找不到 C 编译器，原生栈兜底装不上（不影响推理，只是崩了看不到 C 栈）"
    return 1
  fi

  if ! "$cc" -O2 -fPIC -shared -o "$so" "$src" -ldl 2>"$dir/bt_cc.log"; then
    say "  !! 原生栈兜底编译失败（不影响推理，只是崩了看不到 C 栈）："
    tail -8 "$dir/bt_cc.log" 2>/dev/null | sed 's/^/    /'
    return 1
  fi
  [ -s "$so" ] || { say "  !! 原生栈兜底没产出"; return 1; }
  export H3_BT_LIB=$so
  say "  原生栈兜底就位：$so（SIGSEGV 先打 C 栈，再交回 faulthandler）"
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

  # 4.5) 原生栈兜底：不影响 LD_PRELOAD，纯粹为崩溃时能看见 C 层栈
  build_bt_shim || true

  # 5) 预加载：只放我们的薄壳。DTK 的 libcuda.so.1 由子脚本自己按原样加上，
  #    顺序上正好排在薄壳后面（同名符号我们优先）
  if [ $cclok -eq 1 ]; then
    H3_CUDA_PRELOAD="$H3_CUDA_SHIM:$H3_CCL_SHIM"
  else
    H3_CUDA_PRELOAD="$H3_CUDA_SHIM"
  fi
  # mha 兜底 .so：构造期把组合实现注册进 libinfinicore_cpp_api.so 的 dispatcher
  if [ -n "${H3_MHA_PRELOAD:-}" ]; then
    H3_CUDA_PRELOAD="$H3_CUDA_PRELOAD:$H3_MHA_PRELOAD"
  fi
  # cast_ 覆盖 .so：同名符号顶掉 cpp_api 里那份（Hygon 没开 ATen，原实现必抛）
  if [ -n "${H3_CAST_PRELOAD:-}" ]; then
    H3_CUDA_PRELOAD="$H3_CUDA_PRELOAD:$H3_CAST_PRELOAD"
  fi
  export H3_CUDA_PRELOAD

  # 5) Python 侧兜底：万一子脚本把 LD_PRELOAD 冲掉，解释器启动时也用
  #    RTLD_GLOBAL 把薄壳拉进全局符号表
  local ovl="${INFINICORE_OVERLAY:-$OVL}"
  if [ -d "$ovl" ]; then
    cat > "$ovl/sitecustomize.py" <<'PYEOF'
import ctypes, os

# VAE 解码走 diffusers 的 dispatch_attention_fn，默认 backend=native 会在
# ROCm 上挑到 flash kernel，而本平台没有 flash_attn_2_cuda*.so，SDPA 直接抛
# "No matching libraries found"。_native_math 走 sdpa_kernel(MATH)，不问 flash。
# run.sh 主体已经 export 过一次；这里再兜一层，管的是那些把环境洗掉、
# 或者绕过启动器直接 sys.executable 起的子进程（compare 就是这种）。
os.environ.setdefault("DIFFUSERS_ATTN_BACKEND", "_native_math")

for _p in os.environ.get("H3_CUDA_PRELOAD", "").split(":"):
    if _p and os.path.exists(_p):
        try:
            ctypes.CDLL(_p, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass

# 只在显式要求时补 processor 兜底 + 现场保护。
# 为什么需要这一手：minimax_h3_compare.py 是用 sys.executable 直接起
# minimax_h3.py 子进程的，不走我们的 h3_entry.py，补丁装不上就会在
# Qwen3VLVideoProcessor 那里再炸一次（缺 torchvision）。而 run_fl2va 那条路
# 已经由 h3_entry 装了，重复装也是幂等的。
# 默认不开：每次探测用的小 python 都去 import transformers 要多花 3~5 秒。
if os.environ.get("H3_SITE_PATCH") == "1":
    try:
        import h3_trace

        h3_trace.install()
    except Exception:
        pass
    try:
        import h3_proc_fix
    except Exception:
        pass
PYEOF
    say "  Python 兜底：$ovl/sitecustomize.py（H3_SITE_PATCH=1 时连带装 processor 补丁）"
  fi

  # 6) 打补丁启动器，保证 LD_PRELOAD 里薄壳始终在最前
  patch_fl2va_launcher || say "  警告：启动器没补丁成，只能靠 sitecustomize 兜底"

  export CUDA_COMPAT_OK=1
  return 0
}

# =============================================================================
# Hygon 上没有 flash-attn 扩展时，MultiheadAttention 在 dispatcher 里一个实现都没有：
#   xmake.lua 只有 --flash-attn 非空才保留 ops/*/hygon/*.cc，
#   而通用版 mha_flashattn.cc 只编 NVIDIA/METAX/QY，
#   graph.hpp 的 INFINICORE_GRAPH_OP_DISPATCH 又不判空 -> call 0 -> SIGSEGV(rip=0)。
# 结果：首帧图片一进视觉塔（qwen3_5_vision.cpp 的 op::mha）就整轮崩掉。
# 下面这份 .cc 用 matmul+softmax+matmul 补一份组合实现，数学路径和 InfiniLM 的
# static_attn 后端完全一致（那条路在文本塔上已经跑通过）。
#
# 为什么用 heredoc 而不是让仓库里的文件自己过去：
#   ensure_pkg 看到 $PKG/src/InfiniLM 就「已解包，跳过」，仓库里的改动传不到平台；
#   只有 run.sh 本身每单都是新的。所以和 h3_trace.py / cuda shim 一个套路。
# 仓库原件：src/InfiniCore/src/infinicore/ops/multi_head_attention/mha_composed_fallback.cc
# 两边必须逐字节一致，_fix/test_mha_fallback.sh 会比对。
# =============================================================================
pick_mha_dir() {
  # 兜底 .cc / .so 放哪儿。顺序都是「每单必然能写」的真实证据：
  #   $AFS/cudashim —— 两个 shim 每单都在这里重编；
  #   $OUT          —— 日志就写在这儿；
  #   /tmp          —— 本机盘，丢了也只是重编几秒。
  local d
  for d in "$AFS/cudashim" "$OUT" "/tmp/h3mha"; do
    [ -n "$d" ] || continue
    mkdir -p "$d" 2>/dev/null || continue
    if ( : > "$d/.h3wtest" ) 2>/dev/null; then
      rm -f "$d/.h3wtest" 2>/dev/null
      printf '%s\n' "$d"
      return 0
    fi
  done
  return 1
}

# 往 $1 写一份兜底 .cc。stdout 打「路径 变化标记」（same / new）。
# 目录必须**已经可写**：平台上的 $PKG 是上个作业用另一个 uid 解包的，
# mkdir -p 对已存在目录静默成功，只有真写一次才试得出权限。
write_mha_src() {
  local dir=$1
  local f=$dir/mha_composed_fallback.cc
  local tmp=$dir/.mha_fallback.new.cc

  # 首选：直接用包里那份真源码（和仓库里 commit 的是同一个文件）。
  # 这样 run.sh 里的 heredoc 只是备份，不需要和源码逐字同步（之前就漂过一次：
  # 仓库里改了日志文案，heredoc 还是旧版）。编出来的 .so 因此永远是"仓库那版"。
  local ref="" c
  for c in "${PKG:-}/src/InfiniCore/src/infinicore/ops/multi_head_attention/mha_composed_fallback.cc" \
           "${IC:-}/src/infinicore/ops/multi_head_attention/mha_composed_fallback.cc"; do
    # 只认「带注册回查」的那版：包若是旧 tarball，里面的 .cc 没有自检，
    # 用完会在预加载自检里被误判成「注册没落到 cpp_api」，白丢一单。
    [ -n "$c" ] && [ -s "$c" ] && [ -r "$c" ] || continue
    grep -q "注册回查" "$c" 2>/dev/null || continue
    ref=$c
    break
  done
  if [ -n "$ref" ]; then
    if [ -s "$f" ] && cmp -s "$ref" "$f"; then
      printf '%s same\n' "$f"
      return 0
    fi
    if cp -f "$ref" "$tmp" 2>/dev/null && [ -s "$tmp" ] \
       && mv -f "$tmp" "$f" 2>/dev/null && [ -s "$f" ]; then
      printf '%s new\n' "$f"
      return 0
    fi
    rm -f "$tmp" 2>/dev/null
    # 拷不出来才往下走 heredoc（包里的源码被删/被截断时的最后一道保险）
  fi

  cat > "$tmp" 2>/dev/null <<'MHA_FALLBACK_CC_EOF'
// Hygon(DCU) 上没有 flash-attn 扩展时的 MultiheadAttention 兜底实现。
//
// 背景：xmake.lua 只有在 --flash-attn 不为空时才保留 ops/*/hygon/*.cc，
// 而 hygon/mha_flashattn_hygon.cc 又被 `#if defined(ENABLE_HYGON_API) &&
// defined(ENABLE_FLASH_ATTN)` 包住；通用版 mha_flashattn.cc 只编 NVIDIA/METAX/QY。
// 于是 README 的标准构建（--hygon-dcu=y --ninetoothed=y --ccl=y，无 --flash-attn）
// 下 MultiheadAttention 的 dispatcher 里 **一个函数指针都没有**，
// graph.hpp 的 INFINICORE_GRAPH_OP_DISPATCH 又不判空，直接 call 0 -> SIGSEGV(rip=0)。
//
// 这里用已注册的基础算子把注意力算出来（和 InfiniLM 的 static_attn 后端同一条数学路径）：
//     q·k^T (matmul, alpha=scale) -> softmax/causal_softmax -> ·v (matmul)
// 输入输出布局沿用调用方约定：[B, S, H, D]（vision 塔与 DiT 都是这么传的）。
//
// 编译条件刻意写成「Hygon 且没有 ENABLE_FLASH_ATTN」：只有这一种配置是坏的。
// 一旦有人用 --flash-attn 编（走 hygon/mha_flashattn_hygon.cc 的真实现），
// 本文件整体不参与编译，绝不会顶掉快路径。

// 两种编译形态，代码完全相同：
//   · 编进 libinfinicore_cpp_api.so（宏 ENABLE_HYGON_API）—— 需要重编该库时才走这条；
//   · 独立编成 libh3mha.so，用 LD_PRELOAD 挂（宏 H3_MHA_STANDALONE）—— 只编一个 .cc，
//     靠构造期把实现注册进 cpp_api 里的 dispatcher。后者是当前实际使用的路径：
//     重编 cpp_api 会连带触发 infiniop 全量重编（约 38 分钟），单作业 50 分钟装不下。
#if (defined(ENABLE_HYGON_API) || defined(H3_MHA_STANDALONE)) && !defined(ENABLE_FLASH_ATTN)

#include <cstdio>

#include "infinicore/ops/mha.hpp"

#include "infinicore/ops/causal_softmax.hpp"
#include "infinicore/ops/matmul.hpp"
#include "infinicore/ops/softmax.hpp"

#include <cstdio>
#include <stdexcept>
#include <string>

namespace infinicore::op::mha_impl::composed {

struct PlannedMeta {
    graph::GraphTensor out, q, k, v;
    float scale;
    bool is_causal;
};

void *plan(Tensor out,
           const Tensor &q,
           const Tensor &k,
           const Tensor &v,
           std::optional<Tensor> alibi_slopes,
           float scale,
           bool is_causal) {
    // 调用点（qwen3_5_vision / minimax_h3_transformer）都是 4D + 无 alibi；
    // 真出现别的用法时宁可在 plan 阶段报错，也不要算出错误结果。
    if (q->shape().size() != 4) {
        throw std::runtime_error(
            "MultiheadAttention(composed fallback): expect 4D [B,S,H,D] tensors, got rank "
            + std::to_string(q->shape().size()));
    }
    if (alibi_slopes.has_value()) {
        throw std::runtime_error(
            "MultiheadAttention(composed fallback): alibi_slopes is not supported");
    }

    return new PlannedMeta{
        graph::GraphTensor(out),
        graph::GraphTensor(q),
        graph::GraphTensor(k),
        graph::GraphTensor(v),
        scale,
        is_causal};
}

namespace {

std::string shape_str(const Tensor &t) {
    std::string s = "[";
    for (size_t i = 0; i < t->shape().size(); ++i) {
        if (i) s += ",";
        s += std::to_string(t->shape()[i]);
    }
    return s + "]";
}

void note_once() {
    static bool done = false;
    if (!done) {
        done = true;
        std::fprintf(stderr,
                     "[mha] 使用 InfiniCore 组合实现（matmul+softmax+matmul），"
                     "本平台无 flash-attn 扩展\n");
    }
}

} // namespace

void run(void *planned_meta) {
    auto *p = reinterpret_cast<PlannedMeta *>(planned_meta);
    Tensor out(p->out);
    Tensor q(p->q);
    Tensor k(p->k);
    Tensor v(p->v);

    const auto &qs = q->shape();
    const size_t batch_size = qs[0];
    const size_t seq_len = qs[1];
    const size_t num_heads = qs[2];
    const size_t head_dim = qs[3];

    const size_t num_kv_heads = k->size(2);
    const size_t value_head_dim = v->size(3);

    if (num_heads % num_kv_heads != 0) {
        throw std::runtime_error(
            "MultiheadAttention(composed fallback): num_heads " + std::to_string(num_heads)
            + " is not a multiple of num_kv_heads " + std::to_string(num_kv_heads)
            + " q=" + shape_str(q) + " k=" + shape_str(k) + " v=" + shape_str(v));
    }

    note_once();

    const size_t ngroup = num_heads / num_kv_heads;

    // [B, S, H, D] -> [B, H, S, D] -> [B*H_kv, ngroup*S, D]
    auto Q = q->permute({0, 2, 1, 3})
                 ->contiguous()
                 ->view({batch_size * num_kv_heads, ngroup * seq_len, head_dim});
    auto K = k->permute({0, 2, 1, 3})
                 ->contiguous()
                 ->view({batch_size * num_kv_heads, seq_len, head_dim});
    auto V = v->permute({0, 2, 1, 3})
                 ->contiguous()
                 ->view({batch_size * num_kv_heads, seq_len, value_head_dim});

    auto K_transposed = K->permute({0, 2, 1}); // [B*H_kv, D, S]

    // alpha 直接折进 matmul，省一次全量缩放
    auto attn_weight = op::matmul(Q, K_transposed, p->scale); // [B*H_kv, ngroup*S, S]

    auto flat = attn_weight->view({batch_size * num_heads, seq_len, seq_len});
    // causal 用 causal_softmax_（static_attn 已验证过原地可用）；
    // 非 causal 走独立的 softmax，避免 out/in 同一块内存时还要赌实现里的原地语义。
    //
    // 之后必须 view 回 GQA 展开的形状再乘 V：probs 的 batch 维得是 B*H_kv，
    // 留成 [B*H_q, S, S] 去乘 [B*H_kv, S, D_v]，ngroup>1 时 batch 维直接对不上。
    Tensor probs = flat->view({batch_size * num_kv_heads, ngroup * seq_len, seq_len});
    if (p->is_causal) {
        op::causal_softmax_(flat, flat);
    } else {
        probs = op::softmax(flat, -1)
                    ->view({batch_size * num_kv_heads, ngroup * seq_len, seq_len});
    }

    auto attn_out = op::matmul(probs, V); // [B*H_kv, ngroup*S, D_v]

    auto merged = attn_out->view({batch_size, num_heads, seq_len, value_head_dim})
                      ->permute({0, 2, 1, 3})
                      ->contiguous(); // [B, S, H, D_v]：必须和 out 同形

    // copy_from 的实现是 `if (src->shape() != this->shape()) throw`（copy.cc:21），
    // 所以这里千万不要 view 成 [B,S,H*D] —— 元素数一样也不行，形状必须逐维相同。
    out->copy_from(merged);
}

void cleanup(void **planned_meta_ptr) {
    delete *reinterpret_cast<PlannedMeta **>(planned_meta_ptr);
    *planned_meta_ptr = nullptr;
}

namespace {

void register_all() {
    // override=false：只填没人注册的槽位。Hygon 上这个表本来就是全 nullptr，
    // 所以一定会填上；将来若真编了 --flash-attn 的真实现，也不会被我们顶掉。
    MultiheadAttention::plan_dispatcher().registerAll(&plan, false);
    MultiheadAttention::run_dispatcher().registerAll(&run, false);
    MultiheadAttention::cleanup_dispatcher().registerAll(&cleanup, false);
    // 这行是判断「兜底到底有没有装上」的唯一凭据，日志里搜 [mha] 即可。
    std::fprintf(stderr, "[mha] 兜底已注册到 dispatcher（Hygon，无 flash-attn）\n");
}

} // namespace

#if defined(H3_MHA_STANDALONE)
// 独立 .so 走 LD_PRELOAD：构造期注册到 libinfinicore_cpp_api.so 里的 dispatcher。
// 链接依赖保证 cpp_api 先加载，所以此处调用 plan_dispatcher() 是安全的。
// try/catch 兜住一切异常：这个构造函数跑在所有 python 进程里（包括预检），
// 绝不能因为注册失败把进程带崩。
// 注意：这里**不能**写 static —— C++ 标准禁止在 extern "C" 的链接规范里带存储类，
// gcc 报 `invalid use of 'static' in linkage specification`（14:49 那单就死在这一个词）。
// 用 visibility("hidden") 把它藏起来，效果和 static 一样，但不违反链接规范。
extern "C" __attribute__((constructor, visibility("hidden"))) void h3_mha_fallback_ctor() {
    try {
        register_all();
        // 注册完立刻回查：静态 dispatcher 是函数内 static，正常只会有一份；
        // 万一拿到的不是 cpp_api 那份（跨 DSO 各一份），注册就是白做。
        // 这一行是「兜底到底有没有真装上」的唯一凭据，日志里搜「注册回查」即可。
        auto pfn = MultiheadAttention::plan_dispatcher().lookup(Device::Type::HYGON);
        auto rfn = MultiheadAttention::run_dispatcher().lookup(Device::Type::HYGON);
        std::fprintf(stderr,
                     "[mha] 注册回查：plan=%s run=%s\n",
                     pfn ? "非空" : "空(注册没落到 cpp_api 那份 dispatcher)",
                     rfn ? "非空" : "空(注册没落到 cpp_api 那份 dispatcher)");
    } catch (const std::exception &e) {
        std::fprintf(stderr, "[mha] 注册失败：%s\n", e.what());
    } catch (...) {
        std::fprintf(stderr, "[mha] 注册失败（未知异常），保持不注册状态\n");
    }
}
#else
namespace {
const bool registered = (register_all(), true);
} // namespace
#endif

} // namespace infinicore::op::mha_impl::composed

#endif // (ENABLE_HYGON_API || H3_MHA_STANDALONE) && !ENABLE_FLASH_ATTN
MHA_FALLBACK_CC_EOF
  if [ ! -s "$tmp" ]; then
    rm -f "$tmp" 2>/dev/null
    return 1
  fi
  if [ -s "$f" ] && cmp -s "$tmp" "$f"; then
    rm -f "$tmp" 2>/dev/null
    printf '%s same\n' "$f"
    return 0
  fi
  if mv -f "$tmp" "$f" 2>/dev/null && [ -s "$f" ]; then
    printf '%s new\n' "$f"
    return 0
  fi
  rm -f "$tmp" 2>/dev/null
  return 1
}

# 把兜底 .cc 塞进 $PKG 的编译树 —— 只给 do_build 用（改 xmake 那条老路）。
# 平台上 $PKG 常是别的 uid 解包的，写不进去是常态；独立 .so 路径不依赖它。
inject_mha_fallback() {
  if [ -z "${IC:-}" ]; then
    say "  !! IC 还没定，注入不了 mha 兜底实现"
    return 1
  fi
  local dir=$IC/src/infinicore/ops/multi_head_attention
  local f=$dir/mha_composed_fallback.cc
  mkdir -p "$dir" 2>/dev/null
  if ! ( : > "$dir/.h3wtest" ) 2>/dev/null; then
    say "  !! 写不进 $dir（这棵树是上个作业/另一个 uid 解包的）"
    say "     do_build 走不通；独立 .so 路径不受影响，别在这儿 die"
    return 1
  fi
  rm -f "$dir/.h3wtest" 2>/dev/null
  local out changed
  out=$(write_mha_src "$dir") || { say "  !! 注入失败（写不进去）：$f"; return 1; }
  set -- $out
  f=$1
  changed=${2:-new}
  if [ "$changed" = "same" ]; then
    say "  mha 兜底实现已在编译树里且内容一致：$(basename "$f")"
  else
    say "  mha 兜底实现注入：$f"
    say "    $(wc -c < "$f" | tr -d ' ') 字节  md5 $(md5sum "$f" 2>/dev/null | awk '{print $1}')"
  fi
  return 0
}

# 这份 .cc 的 md5：smoke 每轮拿它和 $OUT/.stamp_mha_fallback 比，
# 不一致就编一次（平台只给 smoke 入口，没法手动 build）。
# 改了上面的 heredoc 就必须同步这一行，_fix/test_mha_fallback.sh 会核对。
MHA_FALLBACK_CC_MD5=1f52123ee8e9a0de0b3f979a56fe0784

# =============================================================================
# Hygon 上 MultiheadAttention 在 dispatcher 里是零注册，op::mha 的 lookup 返回
# nullptr，graph.hpp 的 INFINICORE_GRAPH_OP_DISPATCH 又不判空 -> call 0 (rip=0)。
# 视觉塔第一步就是 op::mha，所以 13:28 那单注入成功、却还是在同一处崩。
#
# 为什么不重编 libinfinicore_cpp_api.so：
#   xmake 里 _infinicore 依赖 infiniop，触发构建就是全量（infiniop 约 38 分钟），
#   单作业 3000 秒装不下；而且这棵树的 build/ 里连 cpp_api 的产物都没有。
# 所以：把兜底实现单独编成一个 .so，LD_PRELOAD 挂上去，靠构造期把它注册进
#   cpp_api 里的 dispatcher —— 和已经跑通的 libh3cuda.so / libh3ccl.so 同一条路。
# 代价只有一个 .cc（几秒），完全不碰 xmake 目标。
# =============================================================================
ensure_mha_fallback() {
  [ -n "${IC:-}" ] || IC=$PKG/src/InfiniCore
  local api=$INFINI/lib/libinfinicore_cpp_api.so

  # 正常就在 $INFINI/lib（_infinilm.so 的 RPATH 也指这儿）。万一不在，
  # 按 refresh_cpp_api 的查找顺序补一遍，别为这个白丢一单。
  if [ ! -f "$api" ]; then
    local c
    for c in "$OVL/infinicore/lib/libinfinicore_cpp_api.so" \
             "$SRC_IC/python/infinicore/lib/libinfinicore_cpp_api.so" \
             "$AFS/ic_py/infinicore/lib/libinfinicore_cpp_api.so"; do
      [ -n "$c" ] && [ -f "$c" ] && { api=$c; break; }
    done
  fi

  if [ ! -f "$api" ]; then
    say "  !! 没有 libinfinicore_cpp_api.so（dispatcher 的宿主），兜底挂不上"
    return 1
  fi
  local apilib
  apilib=$(dirname "$api")

  # cpp_api 若把注册表符号藏了（-fvisibility=hidden），编译一定 undefined reference、
  # 预加载也一定失败。先用 nm 静态确认，省得等编译报错还要猜方向。
  if command -v nm >/dev/null 2>&1; then
    local nsym
    nsym=$(nm -D --defined-only "$api" 2>/dev/null | grep -c "MultiheadAttention")
    if [ "${nsym:-0}" -gt 0 ]; then
      say "  cpp_api 导出 MultiheadAttention 符号 $nsym 个（可链接）"
    else
      say "  !! cpp_api 里没有 MultiheadAttention 符号（被隐藏或版本不符）"
      say "     独立 .so 这条路走不通，得改成在 InfiniLM 侧绕开 op::mha"
      return 1
    fi
  fi

  # 不再往 $PKG 里写：那棵树是别的 uid 解包的，本作业没写权限（14:04 那单就死在这）。
  local dir
  dir=$(pick_mha_dir) || { say "  !! 找不到可写目录放兜底 .cc/.so"; return 1; }
  local so=$dir/libh3mha.so
  local log=$dir/mha_build.log

  local out changed src
  out=$(write_mha_src "$dir") || {
    say "  !! 兜底源码写不进 $dir（连 /tmp 都写不进去？）"
    return 1
  }
  set -- $out
  src=$1
  changed=${2:-new}
  say "  兜底源码：$src（$(wc -c < "$src" | tr -d ' ') 字节，$([ "$changed" = same ] && echo 未变 || echo 新写)）"

  # 源码与宿主库都没动就复用（.cc 内容不变时 write_mha_src 不落盘，mtime 也不动）
  if [ -f "$so" ] && [ "$changed" = "same" ] && [ "$so" -nt "$src" ] && [ "$so" -nt "$api" ]; then
    H3_MHA_PRELOAD=$so
    export H3_MHA_PRELOAD
    say "  mha 兜底 .so 复用：$so  $(ls -lh "$so" 2>/dev/null | awk '{print $5}')"
    return 0
  fi

  local cc=g++
  command -v g++ >/dev/null 2>&1 || cc=clang++
  command -v "$cc" >/dev/null 2>&1 || { say "  !! 没有 g++/clang++，编不了兜底 .so"; return 1; }

  say "  编译 mha 兜底 .so（$cc -shared，只编这一个 .cc，不动 xmake）"
  mkdir -p "$dir" 2>/dev/null
  rm -f "$so.tmp" 2>/dev/null
  local inc="-I$INFINI/include"
  [ -d "$IC/include" ] && inc="-I$IC/include $inc"
  if ! "$cc" -O2 -std=c++17 -fPIC -shared \
        -DH3_MHA_STANDALONE \
        $inc \
        -o "$so.tmp" "$src" \
        -L"$apilib" -linfinicore_cpp_api \
        -Wl,-rpath,"$apilib" \
        2>"$log"; then
    say "  !! 兜底 .so 编译失败（这单还会 rip=0，先别跑）："
    grep -n -m8 -E "error:|fatal error:|undefined reference|cannot find" "$log" 2>/dev/null | cut -c1-200 | sed 's/^/    /'
    tail -4 "$log" 2>/dev/null | sed 's/^/    /'
    return 1
  fi
  [ -s "$so.tmp" ] || { say "  !! 兜底 .so 编出来是空的"; return 1; }
  mv -f "$so.tmp" "$so" 2>/dev/null || { cp -f "$so.tmp" "$so" 2>/dev/null; rm -f "$so.tmp" 2>/dev/null; }

  # 必须在独立进程里试加载：这个 .so 会被 LD_PRELOAD 进每一个 python 进程。
  # 万一 cpp_api 没导出 plan_dispatcher（被 -fvisibility=hidden 藏了），ld.so 会
  # 直接拒绝加载，那就不只是 mha 的问题了，整个 smoke 都起不来。宁可这里失败。
  if ! LD_LIBRARY_PATH="$apilib:$INFINI/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
       LD_PRELOAD="$so" /bin/true 2>"$dir/mha_preload.log"; then
    say "  !! 兜底 .so 预加载失败（cpp_api 可能没导出符号）："
    tail -6 "$dir/mha_preload.log" 2>/dev/null | sed 's/^/    /'
    say "     不挂它——挂了会连 python 都起不来"
    return 1
  fi
  say "  mha 兜底 .so 就位：$so  $(ls -lh "$so" 2>/dev/null | awk '{print $5}')（预加载自检通过）"

  # 能加载还不够：必须看到构造期回查打出「plan=非空」。
  # 空的话就是注册落到了另一份 dispatcher 实例（跨 DSO 静态变量各一份），
  # 这条路彻底走不通，得改成在 InfiniLM 侧绕开 op::mha —— 别再去跑整单。
  if grep -q "注册回查：plan=非空" "$dir/mha_preload.log" 2>/dev/null; then
    say "  兜底注册回查通过：dispatcher 里 plan/run 都非空"
  else
    say "  !! 兜底 .so 能加载，但注册回查没通过："
    grep -m3 -E "\[mha\]" "$dir/mha_preload.log" 2>/dev/null | sed 's/^/    /'
    say "     注册没落到 cpp_api 那份 dispatcher —— 独立 .so 这条路走不通"
    return 1
  fi
  H3_MHA_PRELOAD=$so
  export H3_MHA_PRELOAD
  return 0
}

# =============================================================================
# Hygon 上 op::cast_ 只有 ATen 实现：
#   src/infinicore/ops/cast/cast.cc
#     #if defined(ENABLE_ATEN)  -> to_aten_tensor(out).copy_(to_aten_tensor(in));
#     #else                     -> throw std::runtime_error("cast_ requires ATen");
# 本平台的构建没开 ATen，所以只要有人调它必抛。
#
# DiT 第一次真的命中：csrc/models/minimax_h3/minimax_h3_transformer.cpp:525
#     video_embeds = cast_to(linear_forward(proj_in_, video_input), dtype_)
# proj_in_ 的输入被就地升到 F32（522 行），输出于是也是 F32，这里必须转回 bf16。
# 文本塔不经过这条路径 —— 所以 15:41 那单一路上到 DiT 的 forward_raw #2 才炸，
# 而且 8 个 rank 报的都是同一句，上层只看到 "RankWorker stopped during run"。
#
# 覆盖方式还是「独立 .so + LD_PRELOAD 同名符号」（_infinilm.so 对 cast_ 的调用
# 是跨 DSO 的 PLT 调用，预加载的先解析到）。但**不写 kernel**：
# 真命中的只有每步 4 处中等张量（13M / 2.2M / 15.7M 元素 + 每个 block 的 temb），
# D2H->CPU->H2D 一次几十毫秒，相对 DiT 单步可忽略；
# 而 kernel 要 hipcc + --offload-arch=gfx936 + 设备端自检，任一环节不对就整单白跑。
# =============================================================================
write_cast_src() {
  local dir=$1
  local f=$dir/cast_hygon_shim.cc
  local tmp=$dir/.cast_shim.new.cc

  cat > "$tmp" 2>/dev/null <<'CAST_SHIM_CC_EOF'
// Hygon(DCU) 上 infinicore::op::cast_ 的唯一实现走 ATen：
//   src/infinicore/ops/cast/cast.cc:38-47
//     void run_cast(...) {
//   #if defined(ENABLE_ATEN)
//       ... adaptor::to_aten_tensor(out).copy_(adaptor::to_aten_tensor(in));
//   #else
//       throw std::runtime_error("cast_ requires ATen");
//   #endif
//     }
// 本平台的构建没开 ENABLE_ATEN，所以只要有人调 cast_ 就必抛。
//
// DiT 第一次真的用到它是在 minimax_h3_transformer.cpp:525-526：
//   video_input = cast_to(video_hidden_states, F32)   // 入参本来就是 F32，短路
//   video_embeds = cast_to(linear_forward(proj_in_, video_input), dtype_)
// 入参被显式升到 F32，proj_in_ 的输出于是也是 F32，这里必须转回 bf16 —— 抛。
// 表现就是 8 个 rank 同时打 "exception during forward: cast_ requires ATen"，
// 上一层只看到 "RankWorker stopped during run"，一点线索都没有。
//
// 覆盖方式：独立 .so + LD_PRELOAD 同名符号（和 op::mha 兜底、cuda/ccl 薄壳同一条路）。
// _infinilm.so 对 cast_ 的调用是跨 DSO 的 PLT 调用，先加载者优先，
// 预加载的这份就顶掉了 cpp_api 里的那份。
//
// 为什么用 D2H -> CPU 转换 -> H2D，而不是写个 HIP kernel：
//   · DiT 里真正命中的转换只有 4 处中等张量（每步）：
//       proj_in 输出 2432×5376、audio_proj_in 输出 414×5376、norm_out 输出 2922×5376
//     外加每个 block 里 temb 那点（2×hidden，可忽略）；
//     一次 15M 元素的往返约 120MB，几十毫秒，相对 DiT 单步可忽略。
//   · kernel 要 hipcc + --offload-arch=gfx936+ 设备端自检，任何一个不对就是整单白跑；
//     这条路径只用到已经跑通的 infinirt memcpy（libh3cuda 薄壳早就把它接到 hip 上了）。
//
// 编译条件刻意只留 H3_CAST_STANDALONE：这份文件即使被 xmake 的 add_files 扫进
// cpp_api 的编译树，也是一个空 TU，绝不会和 cast.cc 撞符号。
#if defined(H3_CAST_STANDALONE)

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "infinicore/context/context.hpp"
#include "infinicore/dtype.hpp"
#include "infinicore/ops/cast.hpp"
#include "infinicore/tensor.hpp"

namespace {

// bf16 只是 f32 的高 16 位，转换可以纯位运算做，不依赖任何半精度头文件。
// f32 -> bf16 用 round-to-nearest-even，和 ATen 的 BFloat16(float) 一致；
// 不做这一步（直接截断）会让 DiT 的数值和参考实现慢慢漂开。
inline float bf16_to_f32(uint16_t v) {
    const uint32_t u = static_cast<uint32_t>(v) << 16;
    float f = 0.0f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

inline uint16_t f32_to_bf16(float f) {
    uint32_t u = 0;
    std::memcpy(&u, &f, sizeof(u));
    const uint32_t exp = u & 0x7f800000u;
    if (exp == 0x7f800000u) { // inf / nan
        if ((u & 0x007fffffu) != 0) {
            return static_cast<uint16_t>((u >> 16) | 0x0040u); // 保留为 quiet NaN
        }
        return static_cast<uint16_t>(u >> 16);
    }
    u += 0x7fffu + ((u >> 16) & 1u);
    return static_cast<uint16_t>(u >> 16);
}

enum Kind { K_F32, K_BF16, K_OTHER };

Kind kind_of(const infinicore::DataType &d) {
    if (d == infinicore::DataType::F32) {
        return K_F32;
    }
    if (d == infinicore::DataType::BF16) {
        return K_BF16;
    }
    return K_OTHER;
}

// 只实现模型真会用到的那几种；其余明确报错，不要静默算出垃圾。
void convert_on_host(void *dst, const void *src, size_t n,
                     const infinicore::DataType &from,
                     const infinicore::DataType &to) {
    const Kind ks = kind_of(from);
    const Kind kd = kind_of(to);
    if (ks == K_F32 && kd == K_BF16) {
        const auto *s = static_cast<const float *>(src);
        auto *d = static_cast<uint16_t *>(dst);
        for (size_t i = 0; i < n; ++i) {
            d[i] = f32_to_bf16(s[i]);
        }
        return;
    }
    if (ks == K_BF16 && kd == K_F32) {
        const auto *s = static_cast<const uint16_t *>(src);
        auto *d = static_cast<float *>(dst);
        for (size_t i = 0; i < n; ++i) {
            d[i] = bf16_to_f32(s[i]);
        }
        return;
    }
    throw std::runtime_error("cast_ shim: 未实现的转换 " + infinicore::toString(from)
                             + " -> " + infinicore::toString(to)
                             + "（本 shim 只做 F32<->BF16）");
}

size_t g_calls = 0;

void log_call(size_t n, const infinicore::DataType &from,
              const infinicore::DataType &to, double mbytes) {
    const size_t k = ++g_calls;
    // 前几次全打（能看出调用顺序），之后每 500 次一行：
    // 日志里看到 [cast] 就证明符号确实被我们顶掉了；一条都没有 = 覆盖没生效。
    if (k <= 8 || k % 500 == 0) {
        std::fprintf(stderr, "[cast] #%zu %s->%s numel=%zu (%.1f MB)\n", k,
                     infinicore::toString(from).c_str(),
                     infinicore::toString(to).c_str(), n, mbytes);
    }
}

} // namespace

namespace infinicore::op {

// 与 include/infinicore/ops/cast.hpp 的声明逐字一致，靠同名符号顶掉 cpp_api 里那份。
void cast_(Tensor out, const Tensor &in) {
    if (!out || !in || out->shape() != in->shape()) {
        throw std::runtime_error("cast_ expects equal non-empty shapes");
    }
    if (out->device() != in->device()) {
        throw std::runtime_error("cast_: output 与 input 必须在同一设备上。out: "
                                 + out->info() + " in: " + in->info());
    }

    const auto &src_dtype = in->dtype();
    const auto &dst_dtype = out->dtype();

    // 转换要在调用方那个 rank 自己的设备上做（8 个 rank 是 8 个线程/进程，
    // hip 的 current device 是线程本地状态，必须自己设）。
    context::setDevice(out->device());

    // 非连续输入先整理成连续：后面的宿主转换按线性内存走。
    Tensor src = in;
    if (!src->is_contiguous()) {
        src = in->contiguous();
    }

    if (src_dtype == dst_dtype) {
        // 同类型就是纯拷贝，交给已有实现（它会处理 out 的 strides）。
        out->copy_from(src);
        return;
    }

    const size_t n = src->numel();
    if (n == 0) {
        return;
    }
    if (static_cast<size_t>(out->numel()) != n) {
        throw std::runtime_error("cast_: numel 不一致。out: " + out->info()
                                 + " in: " + in->info());
    }

    const size_t s_size = src->element_size();
    const size_t d_size = out->element_size();
    log_call(n, src_dtype, dst_dtype,
             static_cast<double>(n) * static_cast<double>(s_size + d_size) / (1024.0 * 1024.0));

    std::vector<uint8_t> host_in(n * s_size);
    std::vector<uint8_t> host_out(n * d_size);

    if (out->device().getType() == Device::Type::CPU) {
        convert_on_host(host_out.data(), src->data(), n, src_dtype, dst_dtype);
        std::memcpy(out->data(), host_out.data(), n * d_size);
        return;
    }

    // 关键：先把生产者的队列排空。infinirt 的 D2H 走的是 cudaMemcpy(同步)，
    // 它和 infinirt 自己那条 stream 之间没有顺序保证，不 sync 可能读到旧值。
    context::syncStream();
    context::memcpyD2H(host_in.data(), src->data(), n * s_size);
    convert_on_host(host_out.data(), host_in.data(), n, src_dtype, dst_dtype);

    // H2D 用阻塞版：host_out 出了这个函数就析构，异步拷贝不安全。
    if (out->is_contiguous()) {
        context::memcpyH2D(out->data(), host_out.data(), n * d_size, /*async=*/false);
    } else {
        Tensor tmp = Tensor::empty(out->shape(), dst_dtype, out->device());
        context::memcpyH2D(tmp->data(), host_out.data(), n * d_size, /*async=*/false);
        out->copy_from(tmp);
    }
}

} // namespace infinicore::op

namespace {
// 预加载自检用：构造期打一行，确认这份 .so 真的被 ld.so 挂上了。
// （符号有没有被顶掉，要看运行时的 [cast] 行。）
__attribute__((constructor, visibility("hidden"))) void h3_cast_shim_banner() {
    std::fprintf(stderr,
                 "[cast] cast_ 覆盖已就位（Hygon 无 ATen，F32<->BF16 走 D2H->CPU->H2D）\n");
}
} // namespace

#endif // H3_CAST_STANDALONE
CAST_SHIM_CC_EOF

  if [ ! -s "$tmp" ]; then
    rm -f "$tmp" 2>/dev/null
    return 1
  fi
  if [ -s "$f" ] && cmp -s "$tmp" "$f"; then
    rm -f "$tmp" 2>/dev/null
    printf '%s same\n' "$f"
    return 0
  fi
  if mv -f "$tmp" "$f" 2>/dev/null && [ -s "$f" ]; then
    printf '%s new\n' "$f"
    return 0
  fi
  rm -f "$tmp" 2>/dev/null
  return 1
}

# 这份 .cc 的 md5：只是给 _fix/test_cast_shim.sh 的锚点 ——
# 改了上面的 heredoc 就必须同步这一行，测试会核对「heredoc == 仓库原件 == 这个戳」。
# 运行时重编靠 mtime（下面 "$so" -nt "$src"），不靠这个常量。
CAST_SHIM_CC_MD5=6744fc8b3b2022912cfb3259a8c23f30

ensure_cast_fallback() {
  if [ "${H3_CAST:-1}" = 0 ]; then
    say "  H3_CAST=0：跳过 cast_ 覆盖（DiT 会在 cast_ requires ATen 处停）"
    return 0
  fi
  [ -n "${IC:-}" ] || IC=$PKG/src/InfiniCore
  local api=$INFINI/lib/libinfinicore_cpp_api.so
  if [ ! -f "$api" ]; then
    local c
    for c in "$OVL/infinicore/lib/libinfinicore_cpp_api.so" \
             "$SRC_IC/python/infinicore/lib/libinfinicore_cpp_api.so" \
             "$AFS/ic_py/infinicore/lib/libinfinicore_cpp_api.so"; do
      [ -n "$c" ] && [ -f "$c" ] && { api=$c; break; }
    done
  fi
  if [ ! -f "$api" ]; then
    say "  !! 没有 libinfinicore_cpp_api.so，cast_ 覆盖挂不上"
    return 1
  fi
  local apilib
  apilib=$(dirname "$api")

  # 宿主必须**导出** cast_，否则我们这份顶不掉任何东西（-fvisibility=hidden 时会这样）。
  # 静态确认一次，省得白跑 20 分钟再看到同一句 "cast_ requires ATen"。
  if command -v nm >/dev/null 2>&1; then
    local nsym
    nsym=$(nm -D --defined-only "$api" 2>/dev/null | grep -c "op5cast_")
    if [ "${nsym:-0}" -gt 0 ]; then
      say "  cpp_api 导出 cast_ 符号 $nsym 个（可覆盖）"
    else
      say "  !! cpp_api 没导出 cast_（被隐藏？）—— 覆盖不会生效"
      return 1
    fi
  fi

  local dir
  dir=$(pick_mha_dir) || { say "  !! 找不到可写目录放 cast_ shim"; return 1; }
  local so=$dir/libh3cast.so
  local log=$dir/cast_build.log

  local out changed src
  out=$(write_cast_src "$dir") || {
    say "  !! cast_ shim 源码写不进 $dir"
    return 1
  }
  set -- $out
  src=$1
  changed=${2:-new}
  say "  cast_ 源码：$src（$(wc -c < "$src" | tr -d ' ') 字节，$([ "$changed" = same ] && echo 未变 || echo 新写)）"

  if [ -f "$so" ] && [ "$changed" = "same" ] && [ "$so" -nt "$src" ] && [ "$so" -nt "$api" ]; then
    H3_CAST_PRELOAD=$so
    export H3_CAST_PRELOAD
    say "  cast_ shim .so 复用：$so  $(ls -lh "$so" 2>/dev/null | awk '{print $5}')"
    return 0
  fi

  local cc=g++
  command -v g++ >/dev/null 2>&1 || cc=clang++
  command -v "$cc" >/dev/null 2>&1 || { say "  !! 没有 g++/clang++，编不了 cast_ shim"; return 1; }

  say "  编译 cast_ shim .so（$cc -shared，只编这一个 .cc，不动 xmake）"
  local inc="-I$INFINI/include"
  [ -d "$IC/include" ] && inc="-I$IC/include $inc"
  rm -f "$so.tmp" 2>/dev/null
  if ! "$cc" -O2 -std=c++17 -fPIC -shared \
        -DH3_CAST_STANDALONE \
        $inc \
        -o "$so.tmp" "$src" \
        -L"$apilib" -linfinicore_cpp_api \
        -Wl,-rpath,"$apilib" \
        2>"$log"; then
    say "  !! cast_ shim 编译失败："
    grep -n -m8 -E "error:|fatal error:|undefined reference|cannot find" "$log" 2>/dev/null | cut -c1-200 | sed 's/^/    /'
    tail -4 "$log" 2>/dev/null | sed 's/^/    /'
    return 1
  fi
  [ -s "$so.tmp" ] || { say "  !! cast_ shim 编出来是空的"; return 1; }
  mv -f "$so.tmp" "$so" 2>/dev/null || { cp -f "$so.tmp" "$so" 2>/dev/null; rm -f "$so.tmp" 2>/dev/null; }

  # 这个 .so 会被 LD_PRELOAD 进每一个 python 进程，先单独试加载。
  if ! LD_LIBRARY_PATH="$apilib:$INFINI/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
       LD_PRELOAD="$so" /bin/true 2>"$dir/cast_preload.log"; then
    say "  !! cast_ shim 预加载失败："
    tail -6 "$dir/cast_preload.log" 2>/dev/null | sed 's/^/    /'
    say "     不挂它——挂了会连 python 都起不来"
    return 1
  fi
  if grep -q "cast_ 覆盖已就位" "$dir/cast_preload.log" 2>/dev/null; then
    say "  cast_ shim 就位：$so  $(ls -lh "$so" 2>/dev/null | awk '{print $5}')（构造期自检通过）"
  else
    say "  !! cast_ shim 能加载但构造期没打标语，符号多半没编进去："
    tail -4 "$dir/cast_preload.log" 2>/dev/null | sed 's/^/    /'
    return 1
  fi
  H3_CAST_PRELOAD=$so
  export H3_CAST_PRELOAD
  return 0
}

do_build() {
  ensure_pkg
  ensure_deps
  ensure_xmake
  pick_ic
  cd "$IC" || die "进不去 $IC"
  # 先补算子源码：xmake configure 会扫 add_files，晚于它写文件就要再 configure 一次
  # 只是给 xmake 那条老路补源码；写不进编译树也照跑 —— 独立 .so 已经兜住了
  inject_mha_fallback || say "  !! 源码没能写进编译树（uid 不同），跳过；独立 .so 路径不受影响"

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
    # FORCE_BUILD 必须优先于 stamp：stamp 只表示「编过一次」，不代表「编的是当前源码」。
    # 改了 C++ 算子后如果还按 stamp 跳过，就会一直跑旧的 libinfinicore_cpp_api.so。
    if [ "${FORCE_BUILD:-0}" = "1" ]; then
      say "  FORCE_BUILD=1，忽略已有 _infinicore.so 与 stamp，强制重编（增量只补改动的 .cc）"
    elif [ -f "$STAMP_CORE" ]; then
      need_core=0
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
    # av 能 import ≠ 能出片：examples/minimax_h3.py 走的是 diffusers 的 encode_video，
    # 里面是 add_stream("libx264") + aac。缺任一编码器时启动阶段不会报错，
    # 而是等 DiT 全部跑完、写 mp4 那一刻才抛 UnknownCodecError —— 前面几十分钟白烧。
    # 所以在这里花 30 秒验编码器，把最后一步的翻车风险提前暴露。
    # 注意：cv2 / imageio / imageio-ffmpeg 都不需要，encode_video 只碰 PyAV。
    local avmiss
    avmiss=$(PYTHONPATH="$PYDEPS/dyn:$PYDEPS" "$py" - <<'PYEOF' 2>&1
import sys
try:
    import av
except Exception as e:
    print("import:%s" % e)
    sys.exit(0)
try:
    have = set(av.codecs_available)
except Exception:
    try:
        from av.codec.codec import codecs_available as _c
        have = set(_c)
    except Exception:
        print("unknown")
        sys.exit(0)
print(",".join(c for c in ("libx264", "aac") if c not in have))
PYEOF
)
    if [ -n "$avmiss" ]; then
      say "  !! PyAV $avout 能 import，但缺编码器：$avmiss"
      say "     encode_video 要到写 mp4 那一步才会炸（生成全跑完才失败），这会儿就得处理"
      say "     先别跑：把这两行发我，我出个换编码器的补丁（改 patch 不改推理）"
      H3_AV_ENCODER_MISS=$avmiss
    else
      say "  出片编码器 = PyAV $avout（libx264 + aac 都在，diffusers 原生路径）"
    fi
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
    # 官方对比脚本算 SSIM 要用 skimage（离线包里没有，compare 前会现场补）
    ("对比", "skimage", False, "scikit-image"),
    ("对比", "scipy", False, "scipy"),
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
  # 这一步正常只花几十秒。14:30 那单若在这里无声挂着，后面 40 分钟的权重加载和
  # 一小时的生成全是白烧，所以套个超时：卡住就直接判死，把卡时留给重提一单。
  local tmo=""
  command -v timeout >/dev/null 2>&1 && tmo="timeout -k 5 ${H3_PTMO:-900}"
  [ -n "$tmo" ] && say "    上限 ${H3_PTMO:-900}s 强制结束（正常几十秒；卡住就是真卡住）"
  local gout grc
  gout=$(MODEL_ROOT="$MODEL_ROOT" LD_PRELOAD="${H3_CUDA_PRELOAD:-}${LD_PRELOAD:+:$LD_PRELOAD}" \
    $tmo "$py" - 2>&1 <<'PYEOF'
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
  if [ $grc -eq 124 ] || [ $grc -eq 137 ]; then
    say "  !! processor 自检超过 ${H3_PTMO:-900}s 仍没结束 → 判定卡死（不是慢）"
    printf '%s\n' "$gout" | tail -20 | sed 's/^/    /'
    say "     停在 from_pretrained 或某个 import 等锁上。这一单重提一次通常就过；"
    say "     连续两单都卡在同一处，把上面那几行贴我"
    return 5
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

# 文本编码器的 heads 能不能被卡数整除 —— 11-08 11:48 那单把范围逼到了这里。
#
# csrc/layers/attention/attention.cpp:26 是 `num_attention_heads_ = total / tp_size`，
# C++ 的整数除法。不能整除时它不报错，直接向下取整，接着
#   view({batch, seq, num_attention_heads_, head_dim_})
# 的元素数就和 q 的实际元素数对不上 —— 越界，SIGSEGV。
#
# 这个错法跟这一单的现场完全对得上：A 自检（裸 infinicore 算子）全过，一进
# forward 就崩，而且 [h3ccl] 一条 ">> 进入" 都没有 —— 因为 qkv_proj 是列并行，
# 它后面才轮到 o_proj 的 allreduce，崩在 view 就永远走不到集合通信。
# 所以先把这几个数字打到日志里，一单就能定案。
# 组件 config 到底在哪，不能靠拼：examples/minimax_h3.py 的 resolve_model_paths
# 里 fl2va 的 partition_dir 就是 diffusers/ 本身，所以 text_encoder 在
# <MODEL_ROOT>/diffusers/text_encoder，而 MODEL_ROOT 常是 .../MiniMax-H3/FL2VA。
# 直接拼 "$MODEL_ROOT/text_encoder/config.json" 会找不到 —— 11-08 那单就因此
# 白丢了一次 heads 诊断。这里按 resolve_model_paths 的分支顺序逐一探。
find_component_cfg() {
  local root=$1 comp=$2 par c
  par=$(dirname "$root")
  # diffusers 排前面：fl2va 的 partition_dir 就是 modular_dir（= .../MiniMax-H3/
  # diffusers），那才是真正被 load 的那份。FL2VA/ 下也有一份同名目录，但只是
  # 分区的另一份拷贝；11-08 12:03 打出来的 heads=64 来自 FL2VA 那份，代码实际
  # 加载的却是 diffusers 那份，两者一致只是运气。
  for c in "$par/diffusers/$comp/config.json" \
           "$par/diffusers/$comp/$comp/config.json" \
           "$root/diffusers/$comp/config.json" \
           "$root/diffusers/$comp/$comp/config.json" \
           "$root/$comp/config.json" \
           "$root/$comp/$comp/config.json" \
           "$par/$comp/config.json"; do
    [ -f "$c" ] && { printf '%s\n' "$c"; return 0; }
  done
  return 1
}

# 为什么单看这几个数就能定案：attention.cpp:26 是
#   num_attention_heads_ = total_num_heads / tp_size
# C++ 的整数除法。不能整除时它不报错，直接向下取整，接着
#   view({batch, seq, num_attention_heads_, head_dim_})
# 的元素数就和 q 的实际元素数对不上 —— 越界，SIGSEGV。
#
# 这个错法跟这一单的现场完全对得上：A 自检（裸 infinicore 算子）全过，一进
# forward 就崩，而且 [h3ccl] 一条 ">> 进入" 都没有 —— 因为 qkv_proj 是列并行，
# 它后面才轮到 o_proj 的 allreduce，崩在 view 就永远走不到集合通信。
#
# 平台只给 8 卡时，最后那两行给的 H3_TEXT_TP / H3_DIT_TP 是唯一出路：只降这
# 一个组件的分片数，主线不动。
report_heads() {
  local cfg=$1 gpus=$2 label=${3:-text encoder} knob=${4:-H3_TEXT_TP} afs=${5:-}
  if [ ! -f "$cfg" ]; then
    say "  $label：config 不在（$cfg），跳过"
    return 0
  fi
  python3 - "$cfg" "$gpus" "$label" "$knob" "$afs" <<'PYHEADS'
import json
import sys

cfg, gpus, label, knob, afs = sys.argv[1:6]
n = len([g for g in gpus.split(",") if g.strip()])
try:
    d = json.load(open(cfg))
except Exception as exc:  # noqa: BLE001
    print("  %s：config 读不了（%s），跳过" % (label, exc))
    sys.exit(0)
tc = d.get("text_config", d) or {}
h = tc.get("num_attention_heads")
kv = tc.get("num_key_value_heads")
hs = tc.get("hidden_size")
hd = tc.get("head_dim")
print("  %s: hidden=%s heads=%s kv_heads=%s head_dim=%s 卡数=%d"
      % (label, hs, h, kv, hd, n))
print("     config: %s" % cfg)
if not isinstance(h, int):
    print("  （config 里没有 num_attention_heads，整除性判断跳过）")
    sys.exit(0)
if h % n == 0:
    print("  heads %d / %d 卡 = %d，能整除" % (h, n, h // n))
    sys.exit(0)
print("  !! %s 的 heads %d 不能被 %d 卡整除，会被整数除法截成 %d" % (label, h, n, h // n))
print("     后果：q 的元素数 ≠ view({bs,seq,heads,head_dim}) 要的元素数")
print("           → 越界 → SIGSEGV，而且一条 allreduce 都走不到")
ok = [k for k in (1, 2, 3, 4, 6, 8) if k <= n and h % k == 0]
if ok:
    best = max(ok)
    print("     能整除的卡数：%s" % "、".join("%d 卡" % k for k in ok))
    env = (afs + "/h3.env") if afs else "$AFS/h3.env"
    print("     平台只给 %d 卡、换不了卡数时 —— 别减卡，把它单独降 TP：" % n)
    print("       在 %s 里加一行   %s=%d" % (env, knob, best))
    print("       （这组件的 attention 就用 %d 卡分片，%d/%d=%d 正好整除；主线仍 %d 卡）"
          % (best, h, best, h // best, n))
    if best != n:
        print("     平台允许换卡数时也可以整条链路减卡：retry4 / retry2")
PYHEADS
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
      H3_CUDA_PRELOAD="$H3_CUDA_SHIM${H3_CCL_SHIM:+:$H3_CCL_SHIM}${H3_MHA_PRELOAD:+:$H3_MHA_PRELOAD}${H3_CAST_PRELOAD:+:$H3_CAST_PRELOAD}"
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
  # 导入路径优先本地盘：AFS 上光 transformers 建 import 结构就要好几分钟，
  # 而这段路径 preflight 和启动器各走一遍 —— 两遍合起来就是十分钟的纯等待。
  setup_fast_pylibs
  # overlay 放最前：既保证用的是新编的 infinicore，也让它的 sitecustomize.py 先生效
  export PYTHONPATH="$pyic:${PYLIBS_X:-$PYLIBS}:$PYDEPS/dyn:$PYDEPS:$PKG/src/InfiniLM/python"
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

  if [ "${H3_ENV_ONLY:-0}" = "1" ]; then
    say "  环境已就绪（H3_ENV_ONLY=1：只准备环境，不启动推理）"
    return 0
  fi

  # 段错误定位加速器（默认关，只在 trace 还不够用时开）：
  # 异步 launch 崩了以后，Python 栈只能指出「在哪个 block」，指不出「哪一次 launch」。
  # 打开串行化，崩点就精确到那一条 dispatch，代价是明显变慢（smoke 也够用）。
  #   H3_SERIAL=1  torch 侧每次 launch 后同步（CUDA/HIP 两个变量名都设，ROCm 构建读哪个都认）
  #   H3_SERIAL=2  再把 HIP 运行时本身也序列化（kernel 和 memcpy 全同步）
  if [ -n "${H3_SERIAL:-}" ] && [ "${H3_SERIAL}" != "0" ]; then
    export CUDA_LAUNCH_BLOCKING=1 HIP_LAUNCH_BLOCKING=1
    if [ "${H3_SERIAL}" -ge 2 ] 2>/dev/null; then
      export AMD_SERIALIZE_KERNEL=3 AMD_SERIALIZE_COPY=3
    fi
    say "  串行化调试已开（H3_SERIAL=$H3_SERIAL）：崩点可精确到单次 launch，会明显变慢"
  fi

  local launcher=$PKG/scripts/run_fl2va_direct.sh
  local patched=0
  if [ -n "${H3_LAUNCHER:-}" ] && [ -f "$H3_LAUNCHER" ]; then launcher=$H3_LAUNCHER; patched=1; fi
  say "启动 $tag：GPUS=$GPUS  MODEL_ROOT=$MODEL_ROOT"
  say "  启动器 = $launcher"
  # config 的位置按 resolve_model_paths 探，不硬拼（见 find_component_cfg 注释）。
  # 两个组件都可能不整除，所以各打一行——DiT 那边崩起来一样是 SIGSEGV。
  local tcfg dcfg
  if tcfg=$(find_component_cfg "$MODEL_ROOT" text_encoder); then
    report_heads "$tcfg" "$GPUS" "text encoder" H3_TEXT_TP "$AFS"
  else
    say "  text encoder：config 没找到（$MODEL_ROOT 和 diffusers/ 下都探过）"
  fi
  if dcfg=$(find_component_cfg "$MODEL_ROOT" transformer); then
    report_heads "$dcfg" "$GPUS" "DiT" H3_DIT_TP "$AFS"
  else
    say "  DiT：config 没找到（$MODEL_ROOT 和 diffusers/ 下都探过）"
  fi
  # 降 TP 是保底手段，不是默认行为，所以每次都要在日志里留一行：跑完回头看
  # 才知道这一单到底是全卡还是降过。不设时必须说「不降」，否则
  # 空着的那行会让人以为是忘了传。
  if [ -n "${H3_TEXT_TP:-}" ] && [ "${H3_TEXT_TP}" != "0" ]; then
    say "  文本编码器降 TP = $H3_TEXT_TP 卡（DiT 不受影响）—— 绕开 TP 分片路径"
  else
    say "  文本编码器 TP = 与主线一致（未降；要降就往 $AFS/h3.env 写 H3_TEXT_TP=1）"
  fi
  if [ -n "${H3_DIT_TP:-}" ] && [ "${H3_DIT_TP}" != "0" ]; then
    say "  DiT 降 TP = $H3_DIT_TP 卡（文本塔不受影响）—— 绕开 TP 分片路径"
  else
    say "  DiT TP = 与主线一致（未降；要降就往 $AFS/h3.env 写 H3_DIT_TP=4）"
  fi
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
        # 日志名由启动器按 TAG 拼：smoke 是 fl2va_smoke_<MMDD_HHMM>.log，
        # full 是 fl2va_124f_50s_<MMDD_HHMM>.log —— 里面没有 "full"，按 tag 找会
        # 一律落空，于是最贵的那一单反而一条对策都不打。取最新的一份即可。
        lastlog=$(ls -t "$OUT"/fl2va_*.log 2>/dev/null | head -1)
        if [ -n "$lastlog" ] && grep -q "Error Code 5" "$lastlog" 2>/dev/null; then
          echo "  集合通信没接管（Error Code 5）：往上找「集合通信薄壳」那行，看是不是没编出来/没预加载"
        elif [ -n "$lastlog" ] && grep -qi "out of memory\|OUT_OF_RESOURCES\|memory access fault" "$lastlog" 2>/dev/null; then
          echo "  显存/访存出错 → 先降 num_frames；还不行就降 H3_DIT_TP / H3_TEXT_TP"
        elif [ -n "$lastlog" ] && grep -q "InfiniCore C API Error" "$lastlog" 2>/dev/null; then
          echo "  InfiniCore 内部报错 → 往上翻 20 行看 C++ 侧打印的文件:行号"
        fi
        echo "  常见原因与对策："
        echo "   · heads 不能被卡数整除：平台只给 8 卡、减不了卡 —— 往 $AFS/h3.env 里加一行"
        echo "     H3_TEXT_TP=<能整除的卡数>（DiT 不整除就改用 H3_DIT_TP），具体填几看上面 heads 那行"
        echo "   · 显存不足 → 降 num_frames"
        ;;
    esac
  fi
  return $rc
}

# ---------- 官方对比（minimax_h3_compare.py）----------
# 官方脚本一次跑四件事：PyTorch 基准 fl2va、InfiniLM fl2va、两边 ref2va。
# 每件都是 124 帧 / 50 步，一件都不便宜，所以这里按「任务能单独跑」来编排：
#   · 默认 --tasks=fl2va             —— ref2va 要额外的参考视频，时间翻倍，不默认跑
#   · PyTorch 基准跑一次就落在 $OUT/compare/reference_fl2va.mp4，
#     之后一律用 --reference-fl2va-video 指过去，官方脚本直接标 provided，
#     不会重新生成（H3_FORCE_REF=1 才强制重跑）
#   · 只想看我们自己的片子：H3_REUSE_OURS=1，连 InfiniLM 那侧也走 provided
# 每条片子的来路都会打一行，日志里能分清哪个是刚生成的、哪个是复用的。
CMP_DIR=$OUT/compare
CMP_PY=$PKG/src/InfiniLM/examples/minimax_h3_compare.py

# 官方对比脚本在**解码那一刻**才 import skimage，而集群离线：
# pip_heal 里 net_ok 一查就返回失败，等于 compare 这条路走到最后一步必崩。
# 所以自备一份纯 numpy 的 structural_similarity（_fix/ssim_numpy.py），
# 口径按 skimage 的默认（uniform 7x7 窗 + 样本协方差 + K1/K2），
# 和真 skimage 0.26 逐工况对拍过，差 < 1e-15。
# 放在 $PYDEPS/dyn 下、靠 PYTHONPATH 前置命中，不动 pylibs 里的任何东西。
write_ssim_shim() {
  local d=$PYDEPS/dyn
  mkdir -p "$d/skimage/metrics" 2>/dev/null || { say "  !! 建不了 $d/skimage/metrics"; return 1; }
  cat > "$d/skimage/__init__.py" <<'PYEOF'
"""离线环境的 skimage 占位：只提供 metrics.structural_similarity。"""

__all__ = ["metrics"]
PYEOF
  cat > "$d/skimage/metrics/__init__.py" <<'PYEOF'
from .ssim import structural_similarity

__all__ = ["structural_similarity"]
PYEOF
  cat > "$d/skimage/metrics/ssim.py" <<'PYEOF'
"""skimage.metrics.structural_similarity 的最小 numpy 替代。

只为本赛题的对比脚本服务：`minimax_h3_compare.py` 里唯一一处调用是

    structural_similarity(gray_a, gray_b, data_range=255)

两张都是 PIL `convert("L")` 出来的二维 uint8 灰度图（长边已缩到 ≤512）。
所以这里实现的是 skimage 的默认口径：uniform 7x7 窗、样本协方差（cov_norm）、
K1=0.01、K2=0.03。gaussian_weights / gradient 这些分支官方脚本不用，直接拒绝。

为什么要自己写：集群离线，`pip_heal` 的 net_ok 一查就返回失败，
而官方脚本是在**解码对比那一刻**才 import skimage —— 不先兜住就会跑完
124 帧 × 50 步（每侧几十分钟）再崩在 No module named 'skimage'。
"""

import numpy as np

_K1 = 0.01
_K2 = 0.03
_DEFAULT_WIN = 7


def _box_mean(x, size):
    """size x size 均值滤波；边界按 reflect（等价于 scipy.ndimage 的默认 mode）。

    scipy 的 'reflect' 是 (d c b a | a b c d)，正好等于 numpy 的 'symmetric'。
    用两次 cumsum 做可分离求和，避免引入 scipy。
    """
    pad = size // 2
    xp = np.pad(np.asarray(x, dtype=np.float64), pad, mode="symmetric")
    c = np.cumsum(np.cumsum(xp, axis=0), axis=1)
    c = np.pad(c, ((1, 0), (1, 0)), mode="constant")
    s = c[size:, size:] - c[:-size, size:] - c[size:, :-size] + c[:-size, :-size]
    return s / float(size * size)


def structural_similarity(
    im1,
    im2,
    *,
    win_size=None,
    data_range=None,
    gaussian_weights=False,
    channel_axis=None,
    gradient=False,
    full=False,
    **kwargs,
):
    # 必须先升 float64 再算 a*a / a*b：uint8 上 200*200 会溢出成 8 位，
    # 方差直接被压平，SSIM 永远贴着 1（第一版就栽在这，随机图能算出 0.98）。
    a = np.asarray(im1, dtype=np.float64)
    b = np.asarray(im2, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("Input images must have the same dimensions.")
    if a.ndim != 2:
        if channel_axis is not None and a.ndim == 3:
            # 官方脚本走的是灰度，这条只是别让别的调用方直接炸
            a = a.mean(axis=channel_axis)
            b = b.mean(axis=channel_axis)
        else:
            raise ValueError("shim 只支持二维灰度图")
    if gaussian_weights:
        raise NotImplementedError("shim 只实现了 uniform 窗（官方脚本不用 gaussian）")
    if gradient or kwargs:
        raise NotImplementedError("shim 不支持 gradient / 额外参数")

    if data_range is None:
        dr = float(max(a.max(), b.max()) - min(a.min(), b.min())) or 1.0
    else:
        dr = float(data_range)

    size = _DEFAULT_WIN if win_size is None else int(win_size)
    if size % 2 == 0:
        raise ValueError("win_size must be odd")
    shortest = min(a.shape)
    if shortest < size:
        # 窗口比图还大：退到最大的合法奇数窗（skimage 会直接报错，但对比脚本
        # 用的是 256 级以上的帧，走不到这里；退一步比整轮崩掉划算）
        size = shortest if shortest % 2 == 1 else shortest - 1
        if size < 3:
            m = 1.0 if np.array_equal(a, b) else 0.0
            return (m, np.full(a.shape, m)) if full else m

    ux = _box_mean(a, size)
    uy = _box_mean(b, size)
    uxx = _box_mean(a * a, size)
    uyy = _box_mean(b * b, size)
    uxy = _box_mean(a * b, size)

    npix = float(size * size)
    cov_norm = npix / (npix - 1.0)  # 样本协方差
    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)

    c1 = (_K1 * dr) ** 2
    c2 = (_K2 * dr) ** 2
    s = ((2 * ux * uy + c1) * (2 * vxy + c2)) / (
        (ux * ux + uy * uy + c1) * (vx + vy + c2)
    )

    pad = (size - 1) // 2
    if pad:
        s = s[pad : s.shape[0] - pad, pad : s.shape[1] - pad]
    m = float(s.mean())
    return (m, s) if full else m
PYEOF
  [ -f "$d/skimage/metrics/ssim.py" ] || { say "  !! SSIM 兜底没写出来"; return 1; }
  say "  SSIM 兜底已写入 $d/skimage（纯 numpy，不需要 skimage/scipy）"
  return 0
}

# 官方脚本用 skimage 算 SSIM，离线 pylibs 里没有这个包。
# 它是在**解码对比**那一刻才 import 的，也就是说：不先查，就会跑完 124 帧 × 50 步
# （每侧几十分钟）才在最后一步崩在 No module named 'skimage' 上。
# 所以这里在开工前就探一次，缺了就现场补 wheel（不依赖 pip，走清华镜像）。
ensure_compare_deps() {
  local py=python3
  command -v python3 >/dev/null 2>&1 || py=python
  local probe='from skimage.metrics import structural_similarity as f; print("skimage OK", f.__module__)'
  local out
  out=$(PYTHONPATH="$PYDEPS/dyn:$PYDEPS${PYTHONPATH:+:$PYTHONPATH}" "$py" -c "$probe" 2>&1)
  if [ $? -eq 0 ]; then
    say "  对比依赖就位：$out"
    return 0
  fi
  say "  对比要 skimage（SSIM），离线包里没有 → 用自带实现（纯 numpy）"
  printf '%s\n' "$out" | tail -3 | sed 's/^/    /'
  # 默认不走 pip：集群离线，pip_heal 的 net_ok 一查就 false；就算有网，
  # scikit-image + scipy 解压也要几十秒，而自带实现已经和 skimage 的默认口径
  # 逐工况对拍到 1e-15。真要试联网装包：H3_SKIMAGE_PIP=1。
  if [ "${H3_SKIMAGE_PIP:-0}" = "1" ]; then
    # scikit-image 的硬依赖：scipy / networkx / lazy_loader / imageio / tifffile
    # （pyfetch_install 只解压指定包，不会替我们解析依赖，得自己列全）
    pip_heal scikit-image scipy networkx lazy-loader imageio tifffile || true
    out=$(PYTHONPATH="$PYDEPS/dyn:$PYDEPS${PYTHONPATH:+:$PYTHONPATH}" "$py" -c "$probe" 2>&1)
    if [ $? -eq 0 ]; then
      say "  对比依赖补齐（联网）：$out"
      return 0
    fi
  fi
  write_ssim_shim || return 3
  out=$(PYTHONPATH="$PYDEPS/dyn:$PYDEPS${PYTHONPATH:+:$PYTHONPATH}" "$py" -c "$probe" 2>&1)
  if [ $? -eq 0 ]; then
    say "  对比依赖就位（自带 SSIM）：$out"
    return 0
  fi
  printf '%s\n' "$out" | tail -8 | sed 's/^/    /'
  return 3
}

# 按目录里已有的片源组装参数（结果放 CMP_ARGS 全局数组）
cmp_build_args() {
  CMP_ARGS=(
    --model-path="$MODEL_ROOT"
    --output-dir="$CMP_DIR"
    --tasks="${H3_TASKS:-fl2va}"
  )
  [ -f "$FIRST_FRAME" ] && CMP_ARGS+=(--first-frame="$FIRST_FRAME")

  local ref=$CMP_DIR/reference_fl2va.mp4 ours=$CMP_DIR/infinilm_fl2va.mp4
  if [ "${H3_FORCE_REF:-0}" = "1" ]; then
    say "  PyTorch 基准：H3_FORCE_REF=1，强制重新生成（旧片会被覆盖）"
    [ -f "$ref" ] && mv -f "$ref" "$ref.bak" && say "    旧片已留作 reference_fl2va.mp4.bak"
  elif [ -s "$ref" ]; then
    CMP_ARGS+=(--reference-fl2va-video="$ref")
    say "  PyTorch 基准：复用 $ref（$(stat -c '%s' "$ref" 2>/dev/null) 字节，不重新生成）"
  else
    say "  PyTorch 基准：目录里没有现成的，这一单生成一次，之后就一直复用"
  fi

  if [ "${H3_REUSE_OURS:-0}" = "1" ]; then
    local fullout=$OUT/fl2va_124f_50s.mp4
    if [ -s "$ours" ]; then
      CMP_ARGS+=(--infinilm-fl2va-video="$ours")
      say "  InfiniLM 结果：H3_REUSE_OURS=1，复用 $ours（不重新生成）"
    elif [ -s "$fullout" ]; then
      # full 那条路的参数和官方对比完全对齐：prompt「A dog falls asleep」、
      # 124 帧、50 步、seed 42（minimax_h3.py 的默认值），所以这份片子就是
      # 官方脚本要生成的同一份，直接拿来比，省掉一整趟 124 帧生成。
      CMP_ARGS+=(--infinilm-fl2va-video="$fullout")
      say "  InfiniLM 结果：H3_REUSE_OURS=1，复用 full 的产出 $fullout"
      say "    （前提：prompt/帧数/步数/seed 都没改过；改过就别用这个开关）"
    else
      say "  InfiniLM 结果：H3_REUSE_OURS=1 但两条候选都没有，照常生成"
    fi
  fi

  # ref2va 两条片子也支持复用，避免重复跑
  local t
  for t in reference infinilm; do
    local f=$CMP_DIR/${t}_ref2va.mp4
    if [ -s "$f" ] && [ "${H3_FORCE_REF2VA:-0}" != "1" ]; then
      CMP_ARGS+=(--${t}-ref2va-video="$f")
      say "  ${t} ref2va：复用 $f"
    fi
  done
}

# 预加载链要和启动器完全一致：我们的薄壳在最前，DTK 的 libcuda 跟在后面。
# 少这一段，compare 的子进程会自己 import infinicore，符号解析就跟 smoke 走的路不一样了。
h3_preload_chain() {
  local pre="${H3_CUDA_PRELOAD:-}" f
  if [ "${H3_DROP_LIBCUDA:-0}" != "1" ]; then
    for f in "$DTK_ROOT"/cuda/*/targets/x86_64-linux/lib/libcuda.so.1 \
             "$DTK_ROOT"/cuda/*/lib64/libcuda.so.1; do
      [ -f "$f" ] && pre="${pre:+$pre:}$f" && break
    done
  fi
  printf '%s' "$pre"
}

do_compare() {
  [ -f "$CMP_PY" ] || die "缺官方对比脚本：$CMP_PY（代码包没解全？）"
  mkdir -p "$CMP_DIR" 2>/dev/null || die "建不了 $CMP_DIR"

  # 卡、CUDA→HIP 薄壳、集合通信、PYTHONPATH、三道预检，全走 smoke 那套；
  # H3_ENV_ONLY=1 让它准备完就停，不启动 run_fl2va_direct。
  H3_ENV_ONLY=1 do_run compare
  local rc=$?
  [ $rc -eq 0 ] || return $rc

  # compare 的子进程是 sys.executable 直接起的，不经过 h3_entry，靠 sitecustomize
  # 这道闸门把 processor 兜底 + 现场保护带进去。
  export H3_SITE_PATCH=1
  export H3_TRACE="${H3_TRACE:-1}"
  export H3_HB="${H3_HB:-300}"   # 124 帧跑很久，心跳 5 分钟一次，别把日志刷爆

  local py=python3
  command -v python3 >/dev/null 2>&1 || py=python

  # 先确认 SSIM 依赖在，缺了现在补；这东西不在的话，几小时生成完才崩在最后一步
  ensure_compare_deps || return $?

  local -a args
  cmp_build_args
  args=("${CMP_ARGS[@]}")
  args+=(--infinilm-gpus="$GPUS"
         --legacy-gpus="${LEGACY_GPUS:-0,1,2,3}"
         --legacy-memory-fraction="${LEGACY_MEM_FRAC:-0.4}")

  say "官方对比（每侧 124 帧 / 50 步）"
  say "  脚本 = $CMP_PY"
  say "  参数 = ${args[*]}"
  local clog=$OUT/compare_${TS}.log clrc=0
  local pre
  pre=$(h3_preload_chain)
  say "  预加载链 = ${pre:-<空>}"
  LD_PRELOAD="$pre${LD_PRELOAD:+:$LD_PRELOAD}" \
    "$py" "$CMP_PY" "${args[@]}" 2>&1 | tee "$clog"
  clrc=${PIPESTATUS[0]}

  echo
  say "对比结束 rc=$clrc  日志 $clog"
  local f
  for f in reference_fl2va infinilm_fl2va reference_ref2va infinilm_ref2va; do
    [ -s "$CMP_DIR/$f.mp4" ] && say "  $f.mp4  $(stat -c '%s' "$CMP_DIR/$f.mp4" 2>/dev/null) 字节  $(stat -c '%y' "$CMP_DIR/$f.mp4" 2>/dev/null | cut -d. -f1)"
  done
  say "  SSIM/尺寸/帧数见上面 [result] 段；报告填这一段就行"
  # 只跑出基准、我们这侧挂了也不算白跑：基准已落盘，下一单直接复用
  return $clrc
}

# 纯出报告：两条片子都现成，只算 SSIM / 时长，不碰卡
do_report() {
  [ -f "$CMP_PY" ] || die "缺官方对比脚本：$CMP_PY"
  local ref=$CMP_DIR/reference_fl2va.mp4 ours=$CMP_DIR/infinilm_fl2va.mp4
  [ -s "$ref" ] || die "缺基准片 $ref（先跑 compare，或把官方给的参考片放这里）"
  [ -s "$ours" ] || die "缺我们的片 $ours（先跑 full，或 H3_REUSE_OURS 指过去）"
  [ -n "$MODEL_ROOT" ] && [ -f "$MODEL_ROOT/model_index.json" ] \
    || die "找不到权重目录（--model-path 是必填参数，虽然 report 不会真去加载）"

  local py=python3
  command -v python3 >/dev/null 2>&1 || py=python
  ensure_compare_deps || return $?
  say "只出报告（不生成、不占卡）"
  say "  基准 = $ref  $(stat -c '%s' "$ref") 字节"
  say "  本方 = $ours  $(stat -c '%s' "$ours") 字节"
  PYTHONPATH="$PYLIBS:$PYDEPS/dyn:$PYDEPS${PYTHONPATH:+:$PYTHONPATH}" \
    "$py" "$CMP_PY" \
      --model-path="$MODEL_ROOT" \
      --output-dir="$CMP_DIR" \
      --tasks="${H3_TASKS:-fl2va}" \
      --reference-fl2va-video="$ref" \
      --infinilm-fl2va-video="$ours" 2>&1 | tee "$OUT/report_${TS}.log"
  local rc=${PIPESTATUS[0]}
  say "报告结束 rc=$rc  日志 $OUT/report_${TS}.log"
  return $rc
}

# ---------- 分发 ----------
case "$MODE" in
  check)  do_check ;;
  setup)  do_setup ;;
  build)  do_build ;;
  smoke)
          # 平台固定只提交 smoke，而 smoke 原本不编译。算子兜底现已改成
          # 「独立 .so + LD_PRELOAD」（见 ensure_mha_fallback，setup 阶段就编好），
          # 所以这里只在显式要求时才走 xmake —— 光 configure 一次就要 100 秒。
          NEED_BUILD=0
          if [ "${H3_BUILD:-0}" = "1" ] || [ "${FORCE_BUILD:-0}" = "1" ]; then
            NEED_BUILD=1
            say "H3_BUILD=${H3_BUILD:-0} FORCE_BUILD=${FORCE_BUILD:-0}，强制重编"
          fi
          if [ "$NEED_BUILD" = "1" ]; then
            _cwd0=$(pwd)
            do_build || die "编译失败，本轮不跑推理"
            cd "$_cwd0" 2>/dev/null || cd "$AFS" || cd /
            echo "$MHA_FALLBACK_CC_MD5" > "$OUT/.stamp_mha_fallback" 2>/dev/null \
              && say "已记下算子源码戳：$MHA_FALLBACK_CC_MD5"
          fi
          do_setup; do_run smoke ;;
  full)
          # full 必须和 smoke 走同一套准备，不能只 ensure_deps：
          # ensure_mha_fallback / ensure_cast_fallback 只在 do_setup 里。前者缺了
          # 视觉塔第一步就 call 0，后者缺了 DiT 第一步抛「cast_ requires ATen」——
          # 两种都要先烧掉二十分钟导入才暴露，是这里最贵的两个坑。
          # 这两步都有 md5 戳，编过的直接跳过，正常路径只花几秒。
          #
          # H3_ENV_ONLY=1 是「只把环境备好、不跑推理」的开关（调环境时用）。
          # 它留在 AFS/h3.env 里，full 会 rc=0 却一张片子都没有 ——「成功但没产物」
          # 是最难查的一种失败，所以在分发这一层直接拦住，别让它占掉一单 8 卡。
          if [ "${H3_ENV_ONLY:-0}" = "1" ]; then
            say "  !! h3.env 里 H3_ENV_ONLY=1：那是只备环境、不跑推理的开关"
            say "     full 这么提会 rc=0 却没有任何 mp4。删掉那一行再提交。"
            exit 4
          fi
          say "正式片：${NUM_FRAMES:-124} 帧 / ${STEPS:-50} 步 / 画布短边 $H3_CANVAS_SHORT"
          say "  产物 = $OUT/fl2va_${NUM_FRAMES:-124}f_${STEPS:-50}s.mp4"
          do_setup
          do_run full
          ;;
  compare) do_setup; do_compare ;;
  report) do_report ;;
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
  *)      echo "用法: run.sh {check|setup|build|smoke|full|compare|report|all|localall|retry2} [卡数]"
          echo "  compare 跑官方 minimax_h3_compare.py：默认只跑 fl2va，PyTorch 基准跑一次就复用"
          echo "    H3_TASKS=fl2va,ref2va  两个任务都跑    H3_REUSE_OURS=1  复用已有的本方片子"
          echo "    H3_FORCE_REF=1  强制重跑基准          LEGACY_GPUS=0,1,2,3  基准用的卡"
          echo "  report  只算 SSIM / 尺寸 / 帧数，不生成、不占卡（两条片子都得先在 compare 目录里）"
          echo "  段错误还在就开 H3_SERIAL=1（同步 launch，崩点精确到单次 dispatch；=2 更彻底但更慢）" ;;
esac

RC=$?
hr
say "结束 rc=$RC  日志 $LOG"
echo "视频在 $OUT ；报告在 $AFS"
exit $RC
