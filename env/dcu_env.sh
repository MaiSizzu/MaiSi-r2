#!/usr/bin/env bash
# 海光 DCU 环境（可移植版）
#
# 本脚本不写死任何集群路径：DTK 根目录、安装前缀、xmake 全部由环境变量或自动探测决定，
# 因此在一台新机器上 clone 后 source 本文件即可，不需要改内容。
#
# 用法：
#   source env/dcu_env.sh            # 或由 scripts/*.sh 内部 source
# 可覆盖的变量：
#   DTK_ROOT      DTK 安装根（缺省自动探测 /opt/dtk*、/public/software/compiler/dtk-*、$HOME/dtk*）
#   INFINI_ROOT   安装前缀（缺省 $HOME/.infini）
#   XMAKE_BIN     xmake 目录或可执行文件（缺省用 PATH 里的，其次用仓库 vendor/ 下的 3.0.7 静态版）
#   PYTHON_INCLUDE / PYBIND11_INCLUDE / H3_PATCH_INC  编译期头文件位置（缺省由 xmake.lua 自理）

# ---------- 1. DTK ----------
if [ -z "${DTK_ROOT:-}" ] || [ ! -d "${DTK_ROOT:-}" ]; then
  for _c in ${H3_DTK_CANDIDATES:-} /opt/dtk /opt/dtk-* /opt/rocm* \
            /public/software/compiler/dtk-* "$HOME"/dtk* "$HOME"/dtk-*; do
    [ -n "$_c" ] && [ -f "$_c/env.sh" ] && DTK_ROOT="$_c" && break
  done
  unset _c
fi
if [ -n "${DTK_ROOT:-}" ] && [ -f "$DTK_ROOT/env.sh" ]; then
  . "$DTK_ROOT/env.sh"
else
  echo "[env] 未找到 DTK：请 export DTK_ROOT=<dtk 安装根>（该目录下应有 env.sh）" >&2
fi
export DTK_ROOT

# ---------- 2. HIP / ROCm 相关 ----------
if [ -n "${DTK_ROOT:-}" ] && [ -d "$DTK_ROOT" ]; then
  export HIP_PATH="${HIP_PATH:-$DTK_ROOT}"
  export ROCM_PATH="${ROCM_PATH:-$DTK_ROOT}"
  export HIP_CLANG_PATH="${HIP_CLANG_PATH:-$DTK_ROOT/llvm/bin}"
  # DTK 的 cuda 兼容层版本随发布变化，取实际存在的那个
  if [ -z "${CUDA_PATH:-}" ]; then
    for _d in "$DTK_ROOT"/cuda/cuda-* "$DTK_ROOT"/cuda; do
      [ -d "$_d" ] && CUDA_PATH="$_d" && break
    done
    unset _d
  fi
  export CUDA_PATH
  # clang 版本号（15/16/17…）随 DTK 变化，不再写死
  if [ -z "${HIP_CLANG_INCLUDE_PATH:-}" ]; then
    for _d in "$DTK_ROOT"/llvm/lib/clang/*/; do
      [ -d "$_d" ] || continue
      export HIP_CLANG_INCLUDE_PATH="${_d%/}/include"
      export HIP_CLANG_RT_LIB="${_d%/}/lib/linux"
      break
    done
    unset _d
  fi
fi

# ---------- 3. 安装前缀 ----------
# 原先默认 /work2/infini 只在本队集群存在；改为用户目录，避免把产物写进别人的路径
export INFINI_ROOT="${INFINI_ROOT:-$HOME/.infini}"

# ---------- 4. xmake ----------
# 4.1 用户显式指定：XMAKE_BIN 是目录则取其 bin/，是可执行文件则取其所在目录
if [ -n "${XMAKE_BIN:-}" ]; then
  if [ -d "$XMAKE_BIN" ]; then
    export PATH="$XMAKE_BIN/bin:$PATH"
  else
    export PATH="$(dirname "$XMAKE_BIN"):$PATH"
  fi
fi
# 4.2 PATH 里没有 xmake 时，用仓库 vendor/ 下随包交付的 3.0.7 静态版
#     （3.0.8+ 起是动态链接，老容器 glibc 跑不起来，故固定 3.0.7）
if ! command -v xmake >/dev/null 2>&1; then
  _h3_repo="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." 2>/dev/null && pwd)"
  for _b in "$_h3_repo"/vendor/xmake-bundle-v3.0.7.* "$_h3_repo"/vendor/xmake-bundle-*; do
    [ -f "$_b" ] || continue
    _bin="${H3_ENV_BIN:-$HOME/.cache/h3-env/bin}"
    mkdir -p "$_bin" 2>/dev/null || break
    cp -f "$_b" "$_bin/xmake" 2>/dev/null || break
    chmod +x "$_bin/xmake" 2>/dev/null || break
    export PATH="$_bin:$PATH"
    echo "[env] xmake = 仓库内 $(basename "$_b")（已就绪）" >&2
    break
  done
  unset _b _bin _h3_repo
fi

# ---------- 5. 自检输出（一行，便于日志核对） ----------
if [ -n "${H3_ENV_VERBOSE:-}" ]; then
  echo "[env] DTK_ROOT=$DTK_ROOT" >&2
  echo "[env] INFINI_ROOT=$INFINI_ROOT" >&2
  echo "[env] xmake=$(command -v xmake || echo '<无>')" >&2
  echo "[env] python3=$(command -v python3 || echo '<无>')" >&2
fi
