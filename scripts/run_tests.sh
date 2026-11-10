#!/bin/bash
#SBATCH -p hx1hdnormal
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --gres=dcu:1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:30:00
#SBATCH -o run_tests.log
source ${PKG:-$PWD}/env/dcu_env.sh
if [ -n "${H3_VENV:-}" ] && [ -f "$H3_VENV/bin/activate" ]; then
  source "$H3_VENV/bin/activate"
else
  echo "H3_VENV 未设置或无效，沿用当前 PATH 里的 python3: $(command -v python3)"
fi
export LD_LIBRARY_PATH=${INFINI_ROOT:-$HOME/infini}/lib:$DTK_ROOT/cuda/cuda-12/lib64:$DTK_ROOT/lib:/opt/hyhal/lib:/opt/hyhal/lib64:$LD_LIBRARY_PATH
export LD_PRELOAD=$DTK_ROOT/cuda/cuda-12/targets/x86_64-linux/lib/libcuda.so.1
export PYTHONPATH=${SS:-${PKG:-$PWD}}/InfiniCore/python:${SS:-${PKG:-$PWD}}/InfiniLM/python
cd ${IC:-${SS:-${PKG:-$PWD}}/InfiniCore} || exit 1
echo "=== swiglu ==="
python test/infiniop/swiglu.py --hygon
echo "swiglu_exit=$?"
