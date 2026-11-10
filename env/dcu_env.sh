export DTK_ROOT=/public/software/compiler/dtk-25.04.1
export INFINI_ROOT=${INFINI_ROOT:-/work2/infini}
export PATH=${XMAKE_BIN:-/work2/xmake-local}/bin:$PATH
source $DTK_ROOT/env.sh
export HIP_PATH=$DTK_ROOT ROCM_PATH=$DTK_ROOT HIP_CLANG_PATH=$DTK_ROOT/llvm/bin CUDA_PATH=$DTK_ROOT/cuda/cuda-12
export HIP_CLANG_INCLUDE_PATH=$DTK_ROOT/llvm/lib/clang/15.0.0/include
export HIP_CLANG_RT_LIB=$DTK_ROOT/llvm/lib/clang/15.0.0/lib/linux
