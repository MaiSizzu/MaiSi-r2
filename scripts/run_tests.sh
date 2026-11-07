#!/bin/bash
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=dcu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:30:00
#SBATCH -o /work2/songshan/run_tests.log
source ${PKG:-/work2/songshan/MaiSi-r2}/env/dcu_env.sh
source ${H3_VENV:-/public/home/xdzs2026_b0408/venv/h3}/bin/activate
export LD_LIBRARY_PATH=${INFINI_ROOT:-/work2/infini}/lib:$DTK_ROOT/cuda/cuda-12/lib64:$DTK_ROOT/lib:/opt/hyhal/lib:/opt/hyhal/lib64:$LD_LIBRARY_PATH
export LD_PRELOAD=$DTK_ROOT/cuda/cuda-12/targets/x86_64-linux/lib/libcuda.so.1
export PYTHONPATH=${SS:-/work2/songshan}/InfiniCore/python:${SS:-/work2/songshan}/InfiniLM/python
cd ${IC:-${SS:-/work2/songshan}/InfiniCore} || exit 1
echo "=== swiglu ==="
python test/infiniop/swiglu.py --hygon
echo "swiglu_exit=$?"
