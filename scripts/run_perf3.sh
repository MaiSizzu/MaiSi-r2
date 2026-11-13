#!/bin/bash
source ${PKG:-$PWD}/env/dcu_env.sh
if [ -n "${H3_VENV:-}" ] && [ -f "$H3_VENV/bin/activate" ]; then
  source "$H3_VENV/bin/activate"
else
  echo "H3_VENV 未设置或无效，沿用当前 PATH 里的 python3: $(command -v python3)"
fi
export LD_LIBRARY_PATH=${INFINI_ROOT:-$HOME/.infini}/lib:${CUDA_PATH:-$DTK_ROOT/cuda/cuda-12}/lib64:$DTK_ROOT/lib:/opt/hyhal/lib:/opt/hyhal/lib64:$LD_LIBRARY_PATH
export LD_PRELOAD=${CUDA_PATH:-$DTK_ROOT/cuda/cuda-12}/targets/x86_64-linux/lib/libcuda.so.1
export PYTHONPATH=${SS:-${PKG:-$PWD}}/InfiniCore/python:${SS:-${PKG:-$PWD}}/InfiniLM/python
cd ${SS:-${PKG:-$PWD}}/InfiniCore
L=${SS:-${PKG:-$PWD}}/swiglu_perf.log
python test/infiniop/swiglu.py --hygon --profile > $L 2>&1
echo "exit=$? log_lines=$(wc -l < $L) log=$L"
ls -l $L
python ${SS:-${PKG:-$PWD}}/scripts/parse_perf.py $L
