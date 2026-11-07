#!/bin/bash
source ${PKG:-/work2/songshan/MaiSi-r2}/env/dcu_env.sh
source ${H3_VENV:-/public/home/xdzs2026_b0408/venv/h3}/bin/activate
export LD_LIBRARY_PATH=${INFINI_ROOT:-/work2/infini}/lib:$DTK_ROOT/cuda/cuda-12/lib64:$DTK_ROOT/lib:/opt/hyhal/lib:/opt/hyhal/lib64:$LD_LIBRARY_PATH
export LD_PRELOAD=$DTK_ROOT/cuda/cuda-12/targets/x86_64-linux/lib/libcuda.so.1
export PYTHONPATH=${SS:-/work2/songshan}/InfiniCore/python:${SS:-/work2/songshan}/InfiniLM/python
cd ${SS:-/work2/songshan}/InfiniCore
L=${SS:-/work2/songshan}/swiglu_perf.log
python test/infiniop/swiglu.py --hygon --profile > $L 2>&1
echo "exit=$? log_lines=$(wc -l < $L) log=$L"
ls -l $L
python ${SS:-/work2/songshan}/scripts/parse_perf.py $L
