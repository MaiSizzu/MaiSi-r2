#!/bin/bash
source ${PKG:-/work2/songshan/MaiSi-r2}/env/dcu_env.sh
cd ${PKG:-/work2/songshan/MaiSi-r2}/scripts
hipcc -O2 --offload-arch=gfx936 bench_swiglu.hip -o bench_swiglu || exit 1
./bench_swiglu swiglu_v2.cubin 2048 14336 256 200
./bench_swiglu swiglu_v2.cubin 16 5632 256 2000
./bench_swiglu swiglu_v2.cubin 16 5632 128 2000
