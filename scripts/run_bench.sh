#!/bin/bash
source ${PKG:-$PWD}/env/dcu_env.sh
cd ${PKG:-$PWD}/scripts
hipcc -O2 --offload-arch=gfx936 bench_swiglu.hip -o bench_swiglu || exit 1
./bench_swiglu swiglu_v2.cubin 2048 14336 256 200
./bench_swiglu swiglu_v2.cubin 16 5632 256 2000
./bench_swiglu swiglu_v2.cubin 16 5632 128 2000
