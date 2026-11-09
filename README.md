# 脉思团队 · 嵩山杯赛题一 初赛源码

队伍标识 xdzs2026_b0408 ｜ 2026-11-09 ｜ 配套报告随邮件另附

## 目录

    src/InfiniCore          minimax-h3
    src/InfiniLM            minimax_h3
    patches/                源码补丁 + 九齿 blockDim 修正脚本
    patch_inc/              海光工具链的 CUB 兼容头（需加入 include 路径）
    env/ scripts/ evidence/
    run.sh                  平台唯一入口（InfiniFabric / 无 Slurm）
    REPRODUCE.md            给评审的复现指南：前置条件、三条路径、失败对照
    BUILD.md                构建细节 + 两个平台适配 shim 的接线方式
    MANIFEST.txt            文件清单 8594 条（对应提交内容 8595 个文件，不含清单自身）

## 快速开始（InfiniFabric，无 sbatch）

> 评审先看 **REPRODUCE.md**：里面有前置条件清单、三种起跑方式，以及每条失败信息对应的处理。
> 仓库里是全部代码，但有**两样东西必须自备**：MiniMax-H3 权重（~135 GB）和离线依赖包
> `pylibs-py311-v3.zip`（~16 MB，容器里 pip 不可用）。清单见 REPRODUCE.md §0.1。

    export IFAB_AFS_HOME=<持久盘目录>          # 每单必设，外层没有 $AFS
    cp $IFAB_AFS_HOME/run.sh .                 # 平台会把工作区的 run.sh 还原成旧版
    bash run.sh check                          # 只读自检，2 分钟，第一次必须先跑
    bash run.sh compare                        # 出片 + 与 PyTorch 参考片比对
    bash run.sh full                           # 正式 124 帧 / 50 步

`run.sh` 子命令：`check` 只读自检 ｜ `setup` 解包+装依赖+生成首帧 ｜ `build` 编译
InfiniCore ｜ `smoke` 256×256/8 帧/2 步 ｜ `full` 124 帧/50 步 ｜ `compare` 官方
`minimax_h3_compare.py` 的 PyTorch↔InfiniLM 对比 ｜ `report` 只算 SSIM/尺寸/帧数，不占卡。

**不要绕过 run.sh 直接调 `scripts/run_fl2va_direct.sh`**：绕过会回落 768×928 + FA2
的默认档，本构建没有 `--flash-attn`，起不来。

## 构建（顺序不可交换）

    source env/dcu_env.sh
    cd src/InfiniCore && git apply --check --reverse -p1 ../../patches/patches_infinicore_1031.patch && echo PATCH_OK
    PYTHONPATH=src python scripts/build_ntops.py
    python ../../patches/ninetoothed_blockdim_fix.py     # 期望 patched sites 867
    xmake f -y --hygon-dcu=y --ninetoothed=y --omp=n --cpu=y --cuda=$DTK_ROOT/cuda/cuda-12 -cv
    xmake build infiniop && xmake install -y -o "$IFAB_AFS_HOME/infini"

与官方说明的三处差异（`--aten=y` 未开、`--flash-attn` 未传、`--ccl=y` 按
`librccl` 存在与否探测）、以及为此写的两个 shim，见 **BUILD.md**。
两个 shim 不在 `xmake.lua` 里，是刻意的：用独立 .so + `LD_PRELOAD` 同名符号覆盖，
宏门控保证它们即使被 `add_files` 扫进编译树也只是空 TU。

## 验证

    cd ../..
    sbatch --export=ALL,PKG=$PWD,SS=$PWD/src,IC=$PWD/src/InfiniCore scripts/run_tests.sh    # 包根目录执行，期望 81 条全过

## 算子接入情况

四个算子均由九齿从高层张量语言生成，`build/ninetoothed` 下有 908 个 `.cpp` 实现文件
（另有 45 个 `.h`、4 个聚合派发 `.c`，合计 957 个文件），其中包含 divisibility 与
contiguity 布局特化变体。

| 算子 | 生成产物 | 已接入 InfiniCore | 自测 |
| --- | --- | --- | --- |
| flash_attention | 有 | 否，编译受 ENABLE_NVIDIA_API 门控 | 未测 |
| kv_caching | 有 | 否 | 未测 |
| relu | 有 | 否 | 未测 |
| swiglu | 有 | 是 | 81/81 通过 |

端到端链路上实际生效的只有 swiglu。relu 缺一层包装（生成的 kernel 没有对应的
op 注册），kv_caching 的 paged KV 布局与 InfiniLM 的缓存结构对不上，
flash_attention 被 `--flash-attn` 门控挡在编译之外——这三条都**没有**在本包里
宣称已替换上游实现。

## 需如实披露的四点

1. flash_attention 的 scale 参数
   `descriptor.h` 中已有 scale 形参，与生成的 launch 签名不一致。该算子的编译受
   `ENABLE_NVIDIA_API` 宏门控，海光构建不参与编译，不影响海光产物。相关源码已还原为上游状态。

2. blockDim 定点替换
   生成产物中的启动参数由 N×32 调整为 N×64，共 867 处。依据为目标平台波前宽度 64
   （cubin 元数据 `wavefront_size = 0x40`）与最大工作组 256
   （`max_flat_workgroup_size = 0x100`）。替换由 `patches/ninetoothed_blockdim_fix.py`
   完成，脚本幂等并带残留断言。未修改九齿 `ssa.py` 的回退路径，因为该处回退值为整数，
   不直接参与文本产物生成。

3. 编译期兼容层
   海光工具链缺少 CUB，构建时由 `patch_inc/` 提供 CUB→rocprim 的头文件映射，
   该目录随包提供，需加入编译 include 路径。`per_channel_quant_int8` 中的 `#if 1`
   已改为 `#if defined(ENABLE_HYGON_API)`，与 `reduce.cuh`、`topksoftmax`、`lp_norm`
   的既有海光分支写法保持一致。

4. 两个运行时 shim
   `ops/cast/cast_hygon_shim.cc`（`cast_` 在本构建下只有 ATen 实现，未开 ATen 即抛异常）
   与 `ops/multi_head_attention/mha_composed_fallback.cc`（无 `--flash-attn` 时
   dispatcher 为空，调用即 `SIGSEGV(rip=0)`）。两者都是**平台适配**，不是九齿接入，
   不参与算子替换率的计算。详见 BUILD.md。

## 已知限制

- 本包不含模型权重（FL2VA 约 135 GB），需自行准备。
- **出片分辨率 256×320**。画布比例已按 768:928 固定（不再由首帧反推），但 768×928
  在本构建下不可用：mha 兜底是 matmul+softmax+matmul，768×928 时视频 token 26,448
  + 文本 708 = 27,570，单卡 7 head 下 softmax 前向需两份 `27,570²` bf16 中间量，
  共 21 GB 常驻，8 卡 full 在第 1 个去噪步即 `KERNEL VMFault`（rc=139）。
  该开销不随卡数下降（head 分了，`S²` 不分），需要 O(S) 分块累加的融合注意力才可行。
- **单步耗时 28.0 s/步**（8 卡，124 帧）。同机 PyTorch 参考 3.171 s/步，端到端 210 s。
  差距主要来自上面这条 mha 兜底路径与 `cast_` 的 D2H 往返。
- **compare 结果（124 帧）：Size PASS / Length PASS / Average SSIM 37.403%**。
  官方说明里的基准示例（768×928）为 93.584%，但那是 768×928 档、且参考片是
  PyTorch 在同档下生成的；本包跑的是 256×320 档，两者不是同一口径，不能直接相减。本包里 SSIM 偏低主要来自出片尺度与参考片的
  空间频率差异，模糊实验（对两侧都做 15 核高斯）把 SSIM 从 37.15% 抬到 72.79%，
  说明结构与时序是对上的，损失在细节。
- 出片存在彩色过饱和斑块。两侧都有，参考片在三个连续段位上更重
  （6.08% vs 3.90%、4.32% vs 3.58%、12.38% vs 4.55%），逐帧强度相关
  （r=0.437）但分布不一致，**未做进一步归因**，也没有把责任推给算子替换。
- flash_attention 在海光构建中不参与编译；除 swiglu 外的三个算子仅生成产物，
  未完成接入与功能验证。
- `per_channel_quant_int8`、`random_sample` 等 NVIDIA 专用算子在海光上仅保证可编译。
- CCL 部分没有做海光实现：`--ccl=y` 仅在 `librccl` 存在时启用，本平台未启用，
  多卡通信走各自的设备内路径。

## Dependencies

Python packages: 见 `env/requirements.lock`。Toolchain: DTK 25.10（hipcc，`run.sh`
自动探测 `$DTK_ROOT`；`env/dcu_env.sh` 里的 `dtk-25.04.1` 是早期路径）、
xmake >= 3.1.2、ninetoothed 0.26.0。CUB 兼容头在 `patch_inc/`，需加入编译 include 路径。

## Model and data paths

所有权重与数据路径都可以用环境变量覆盖。脚本里的默认值只是提交方环境的兜底，
换环境时请显式设置：

    export MODEL_ROOT=/path/to/MiniMax-H3/FL2VA   # 必设：权重不在包里
    source env/paths.env                          # FIRST_FRAME / OUTPUT_DIR / NUM_FRAMES / STEPS

`MODEL_ROOT` 优先于脚本内部的自动搜索（`$SHARED/MiniMax-H3/...`）；
不设的时候脚本会去找 `$SHARED/MiniMax-H3/FL2VA` 等候选，而 `SHARED` 的默认值
指向提交方所在组的共享目录，在别的环境里不存在。

Weights are not included (~135 GB FL2VA checkpoint). `MODEL_ROOT` must point to a
diffusers-style directory containing `model_index.json` and the transformer weights.

## Run FL2VA end to end

    source env/dcu_env.sh && source env/paths.env
    sbatch --export=ALL,PKG=$PWD,SS=$PWD/src scripts/run_fl2va.sh

Slurm 与 InfiniFabric 两条路径都支持；本包实测在 8 卡上跑通（`LEGACY_GPUS=0..7`），
4 卡是官方基线，TP=8 报 heads 不整除时回退 4 卡。

## 非 Slurm 平台（容器 / 网页终端 / InfiniFabric）

平台不给 sbatch 时用 `scripts/h3_8card.sh` 一键编排（路径自动探测，
DTK / venv / OUTPUT_DIR 可用环境变量覆盖）：

    bash scripts/h3_8card.sh check    # 只读自检：卡数、权重、首帧、库、九齿产物
    bash scripts/h3_8card.sh smoke    # 冒烟：256x256 / 8 帧 / 2 步，约 1-2 分钟
    bash scripts/h3_8card.sh full     # 正式：124 帧 / 50 步
    bash scripts/h3_8card.sh perf     # 同 full，另把耗时落进 perf_8card.csv
    bash scripts/h3_8card.sh fallback # 8 卡失败时改 4 卡冒烟（官方基线）

底层是 `run_fl2va_direct.sh`（不依赖 Slurm，前台直接跑，日志 tee 到 OUTPUT_DIR）：

    bash scripts/preflight.sh                      # 只读自检
    SMOKE=1 GPUS=0,1,2,3,4,5,6,7 bash scripts/run_fl2va_direct.sh   # 冒烟
    GPUS=0,1,2,3,4,5,6,7 bash scripts/run_fl2va_direct.sh           # 正式

张量并行度等于 `--gpus` 的个数。

## Source patch verification

`src/` 已是打好补丁的树，所以补丁是**反向**验证的：`PATCH_OK` 表示树与
`patches_infinicore_1031.patch` 完全一致。正向 apply 会报 "patch does not apply"，属预期。

## Run logs and evidence

    evidence/run_tests.log                swiglu 测试全量日志（81/81）
    evidence/blockdim-before-after.diff   blockDim 启动参数改前/改后
    evidence/mp4-probe.txt                出片实测元信息（ffmpeg 原始输出）
    BUILD.md                              构建、shim 接线与自检
    MANIFEST.txt                          文件清单（2026-11-09，8594 条）
