# 脉思团队 · 嵩山杯赛题一 初赛源码
队伍标识 xdzs2026_b0408 ｜ 2026-11-01 ｜ 配套报告随邮件另附

## 目录
src/InfiniCore（minimax-h3）、src/InfiniLM（minimax_h3）
patches/ 源码补丁 + 九齿 blockDim 修正脚本；env/ scripts/ evidence/

## 构建（顺序不可交换）
source env/dcu_env.sh
cd src/InfiniCore && git apply --check --reverse -p1 ../../patches/patches_infinicore_1031.patch && echo PATCH_OK
PYTHONPATH=src python scripts/build_ntops.py
python ../../patches/ninetoothed_blockdim_fix.py     # 期望 patched sites 867
xmake f -y --hygon-dcu=y --ninetoothed=y --ccl=y --cuda=$DTK_ROOT/cuda/cuda-12 -cv
xmake build infiniop && xmake install -y -o /work2/infini

## 验证
cd ../..
sbatch --export=ALL,PKG=$PWD,SS=$PWD/src,IC=$PWD/src/InfiniCore scripts/run_tests.sh    # 在包根目录执行，期望 81 条全过

## 合规
未改权重、评测程序与 test/；未量化、未减帧数；改动限于九齿编译器与海光后端适配。
九齿生成 908 个 .cpp 实现文件（另有 45 个 .h、4 个聚合派发 .c，目录合计 957 个文件），仅 swiglu 完成接入并 81/81 通过。


## 算子接入情况

四个算子均由九齿从高层张量语言生成，build/ninetoothed 下有 908 个 .cpp 实现文件（另有 45 个 .h、4 个聚合派发 .c，合计 957 个文件），
其中包含 divisibility 与 contiguity 布局特化变体。

| 算子 | 生成产物 | 已接入 InfiniCore | 自测 |
| --- | --- | --- | --- |
| flash_attention | 有 | 否，编译受 ENABLE_NVIDIA_API 门控 | 未测 |
| kv_caching | 有 | 否 | 未测 |
| relu | 有 | 否 | 未测 |
| swiglu | 有 | 是 | 81/81 通过 |

## 需如实披露的三处

1. flash_attention 的 scale 参数
   descriptor.h 中已有 scale 形参，与生成的 launch 签名不一致。该算子的编译受 ENABLE_NVIDIA_API
   宏门控，海光构建不参与编译，不影响海光产物。相关源码已还原为上游状态。

2. blockDim 定点替换
   生成产物中的启动参数由 N 乘 32 调整为 N 乘 64，共 867 处。依据为目标平台波前宽度 64，
   即 cubin 元数据 wavefront_size 为 0x40，以及最大工作组 256，即 max_flat_workgroup_size 为 0x100。
   替换由 patches/ninetoothed_blockdim_fix.py 完成，脚本幂等并带残留断言。
   未修改九齿 ssa.py 的回退路径，因为该处回退值为整数，不直接参与文本产物生成。

3. 编译期兼容层
   海光工具链缺少 CUB，构建时由 patch_inc 目录提供 CUB 到 rocprim 的头文件映射。
   该目录随包提供，需加入编译 include 路径。
   per_channel_quant_int8 中的 #if 1 已改为 #if defined(ENABLE_HYGON_API)，
   与 reduce.cuh、topksoftmax、lp_norm 的既有海光分支写法保持一致。

## 已知限制

- 本包不含模型权重。FL2VA 权重约 135 GB，需自行准备。
- flash_attention 在海光构建中不参与编译。
- 除 swiglu 外的三个算子仅生成产物，未完成接入与功能验证。
- per_channel_quant_int8、random_sample 等 NVIDIA 专用算子在海光上仅保证可编译。

## Dependencies

Python packages: see env/requirements.lock. Toolchain: DTK 25.04.1 (hipcc), xmake >= 3.1.2,
ninetoothed 0.26.0. CUB compatibility headers are provided in patch_inc/ and must be
added to the compiler include path.

## Model and data paths

All paths are configurable, no hardcoded absolute path exists in code or scripts:

    source env/paths.env      # MODEL_ROOT, FIRST_FRAME, OUTPUT_DIR, NUM_FRAMES, STEPS, GPUS

Weights are not included (~135 GB FL2VA checkpoint). MODEL_ROOT must point to a
diffusers-style directory containing model_index.json and the transformer weights.

## Run FL2VA end to end

    source env/dcu_env.sh && source env/paths.env
    sbatch --export=ALL,PKG=$PWD,SS=$PWD/src scripts/run_fl2va.sh

Requires 4 DCUs.

## 非 Slurm 平台（容器 / 网页终端 / InfiniFabric）

平台若不给 sbatch，用 `scripts/h3_8card.sh` 一键编排（路径全部自动探测，DTK / venv / OUTPUT_DIR 可用环境变量覆盖）：

    bash scripts/h3_8card.sh check    # 只读自检：卡数、权重、首帧、库、九齿产物
    bash scripts/h3_8card.sh smoke    # 冒烟：256x256 / 8 帧 / 2 步，约 1-2 分钟
    bash scripts/h3_8card.sh full     # 正式：124 帧 / 50 步
    bash scripts/h3_8card.sh perf     # 同 full，另把耗时落进 perf_8card.csv
    bash scripts/h3_8card.sh fallback # 8 卡失败时改 4 卡冒烟（官方基线）

底层是 `run_fl2va_direct.sh`（不依赖 Slurm，前台直接跑，日志 tee 到 OUTPUT_DIR）。拆开单跑也可以：

    bash scripts/preflight.sh                      # 只读自检
    SMOKE=1 GPUS=0,1,2,3,4,5,6,7 bash scripts/run_fl2va_direct.sh   # 冒烟
    GPUS=0,1,2,3,4,5,6,7 bash scripts/run_fl2va_direct.sh           # 正式

张量并行度等于 --gpus 的个数。8 卡报 head 数不整除时，回退 `GPUS=0,1,2,3` 即可（4 卡是官方基线）。

## Source patch verification

The source tree in src/ is already patched, so the patch is verified in reverse:
PATCH_OK means the tree matches patches_infinicore_1031.patch exactly. Applying it
forward reports "patch does not apply", which is expected.

## Run logs and evidence

evidence/run_tests.log                full log of the swiglu test run (81/81)
evidence/blockdim-before-after.diff   blockDim launch configuration before/after
