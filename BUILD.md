# 构建说明（海光 DCU / DTK）

本包在官方 README 的标准构建上加了两处适配，都以**独立 .so + LD_PRELOAD** 的方式接入，
不修改 InfiniCore 的 xmake.lua。这一页说明为什么这么接、以及怎么复现。

## 1. 工具链

| 项 | 值 |
| --- | --- |
| DTK | 25.04.1（`$DTK_ROOT=/public/software/compiler/dtk-25.04.1`） |
| 编译器 | hipcc（`$DTK_ROOT/hip/bin`），host 侧 g++ / clang++ |
| 构建系统 | xmake >= 3.1.2 |
| 九齿 | ninetoothed 0.26.0 |
| 设备 | 海光 DCU，`--offload-arch=gfx936`，波前宽度 64 |

环境变量见 `env/dcu_env.sh`（`source` 之后再进构建目录，顺序不可交换）。

## 2. 构建命令

```bash
source env/dcu_env.sh
cd src/InfiniCore

# 九齿产物与 blockDim 修正（幂等，期望 patched sites 867）
PYTHONPATH=src python scripts/build_ntops.py
python ../../patches/ninetoothed_blockdim_fix.py

# configure：注意这里没有 --flash-attn，也没有 --aten
xmake g --network=private            # 不联网拉 xmake-repo
CCS=""; ldconfig -p | grep -q librccl && CCS="--ccl=y"
xmake f -y --hygon-dcu=y --ninetoothed=y --omp=n --cpu=y $CCS \
         --cuda="$DTK_ROOT/cuda/cuda-12" -cv

xmake build infiniop && xmake install -y -o $INFINI_ROOT
```

`--omp=n` 不是性能选择：`option("omp")` 默认 true 会触发 `add_requires("openmp")`，
而容器里访问 xmake-repo 会被代理拦成 502，configure 直接失败。

### 与官方示例的三处差异

| 官方说明里的写法 | 本包 | 后果 |
| --- | --- | --- |
| `--aten=y` | 未开启 | `op::cast_` 只有 ATen 实现，未开启时它**直接抛异常**（见 §3.1） |
| `--flash-attn` | 未传入 | `ops/*/hygon/` 下的 mha 实现不参与编译，dispatcher 里一个函数指针都没有（见 §3.2） |
| `--ccl=y` | 运行时探测 `librccl`，有才开 | 平台没有 RCCL 时不启用，多卡走各自的设备内通信路径 |

## 3. 两个 shim 为什么存在

### 3.1 cast_（`ops/cast/cast_hygon_shim.cc`）

`src/infinicore/ops/cast/cast.cc` 的实现是二选一：

```cpp
#if defined(ENABLE_ATEN)
    to_aten_tensor(out).copy_(to_aten_tensor(in));
#else
    throw std::runtime_error("cast_ requires ATen");
#endif
```

本平台没开 ATen，所以这条调用必抛。DiT 第一次真正命中在
`csrc/models/minimax_h3/minimax_h3_transformer.cpp:525`：
`proj_in_` 的输入被就地升到 F32，输出也是 F32，必须转回 bf16。
8 个 rank 报的都是同一句 `cast_ requires ATen`，上层只看到 `RankWorker stopped during run`。

shim 用**同名符号覆盖**：`_infinilm.so` 对 `cast_` 的调用是跨 DSO 的 PLT 调用，
`LD_PRELOAD` 的先被解析到。实现走 **D2H → CPU 转换 → H2D**，不写 HIP kernel：
DiT 每步真正命中的转换只有 4 处中等张量（`proj_in` 输出 2432×5376、`audio_proj_in`
输出 414×5376、`norm_out` 输出 2922×5376，外加每个 block 的 temb），
一次 15M 元素往返约 120 MB、几十毫秒，相对 DiT 单步可忽略；
而 kernel 需要 hipcc + `--offload-arch=gfx936` + 设备端自检，任一环节不对就是整单白跑。

### 3.2 MultiheadAttention（`ops/multi_head_attention/mha_composed_fallback.cc`）

`xmake.lua` 只有在 `--flash-attn` 不为空时才保留 `ops/*/hygon/*.cc`，
而 `hygon/mha_flashattn_hygon.cc` 又被 `#if defined(ENABLE_HYGON_API) && defined(ENABLE_FLASH_ATTN)`
包住；通用版 `mha_flashattn.cc` 只编 NVIDIA / METAX / QY。
于是本构建下 `MultiheadAttention` 的 dispatcher 里没有任何实现，
`graph.hpp` 的 `INFINICORE_GRAPH_OP_DISPATCH` 又不判空，调用即 `SIGSEGV(rip=0)`。

shim 用 matmul + softmax + matmul 组合出前向，按 head 切分，不依赖 flash-attn 扩展。
**代价写在报告里：这是本包 28.0 s/步（PyTorch 3.171 s/步）的主要来源**，也是 768×928
下 softmax 需要 2×21 GB 常驻、直接 VMFault 的原因。

## 4. 为什么 shim 不写进 xmake.lua

四条原因：

1. **不污染上游构建**。两个 shim 是平台适配，不是 InfiniCore 的功能；写进 `xmake.lua`
   会让 `--flash-attn` 分支与官方文档的前提自相矛盾。两份源码的编译条件都是宏门控的：
   `cast_hygon_shim.cc` 只在 `H3_CAST_STANDALONE` 下展开，
   `mha_composed_fallback.cc` 只在 `(ENABLE_HYGON_API || H3_MHA_STANDALONE) && !ENABLE_FLASH_ATTN`
   下展开。**即使某天它们被 `add_files` 扫进编译树，也是两个空 TU，不会和上游实现撞符号。**
2. **符号覆盖需要独立 DSO**。要的是"同名符号 + PLT 先解析"，同一份 `_infinicore.so`
   里加文件做不到覆盖。
3. **构建时长**。mha 另一种形态是编进 `libinfinicore_cpp_api.so`（宏 `ENABLE_HYGON_API`），
   但重编 cpp_api 会连带触发 infiniop 全量重编，实测约 38 分钟；
   单作业最长 50 分钟，装不下"改一次编译一次"的迭代。独立 .so 只编一个 `.cc`，几十秒。
4. **注册回查**。shim 在构造函数里回查 dispatcher 的 `plan`/`run` 是否非空——跨 DSO
   静态变量各一份，注册可能落到另一份实例上。独立 .so 才能在加载自检阶段就判死，
   不必白跑一整单。

因此 `xmake.lua` 里**没有 `add_files` 引用这两个文件**，这是设计，不是漏接线。
`run.sh` 在运行时生成/复用 `.cc` 并编成 `.so`，`LD_PRELOAD` 挂进每个 python 进程。

## 5. 单独编译（不走 run.sh 时）

```bash
apilib=$INFINI_ROOT/lib
cc=g++; command -v g++ >/dev/null || cc=clang++

# mha 兜底
$cc -O2 -std=c++17 -fPIC -shared -DH3_MHA_STANDALONE \
    -I$INFINI_ROOT/include -I src/InfiniCore/include \
    -o mha_composed_fallback.so \
    src/InfiniCore/src/infinicore/ops/multi_head_attention/mha_composed_fallback.cc \
    -L$apilib -linfinicore_cpp_api -Wl,-rpath,$apilib

# cast_ 覆盖
$cc -O2 -std=c++17 -fPIC -shared -DH3_CAST_STANDALONE \
    -I$INFINI_ROOT/include -I src/InfiniCore/include \
    -o cast_hygon_shim.so \
    src/InfiniCore/src/infinicore/ops/cast/cast_hygon_shim.cc \
    -L$apilib -linfinicore_cpp_api -Wl,-rpath,$apilib
```

### 编译后必须做两步自检

```bash
# 1) 能否加载（cpp_api 若被 -fvisibility=hidden 藏了符号，ld.so 直接拒绝）
LD_LIBRARY_PATH=$apilib LD_PRELOAD=./mha_composed_fallback.so /bin/true

# 2) 注册是否落到 cpp_api 那份 dispatcher（期望打印「注册回查：plan=非空」）
```

第 2 步不过就是"能加载但没生效"，这条路走不通，别去跑整单。
`run.sh` 里这两步是硬门禁，任一失败即退出。

## 6. 复现顺序

```bash
export IFAB_AFS_HOME=<持久盘目录>
cp $IFAB_AFS_HOME/run.sh .          # 平台工作区每次重建，run.sh 会被还原成旧版
bash run.sh check                   # 只读自检：卡数/权重/首帧/库/九齿产物
bash run.sh compare                 # 出片 + 与参考片比对（Size/Length/SSIM）
bash run.sh full                    # 124 帧 / 50 步正式出片
```

**不要绕过 run.sh 直接调 `scripts/run_fl2va_direct.sh`**：绕过会回落
768×928 + FA2 的默认档，本构建没有 flash-attn，起不来。
