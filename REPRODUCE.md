# 复现指南（给评审）

一页说清：拿到这个仓库后，**怎么跑起来、每一步该看到什么、跑不动时是哪个前置条件没满足**。

## 0. 前置条件（缺一不可）

| 条件 | 要求 | 怎么确认 |
| --- | --- | --- |
| 加速卡 | 海光 DCU（gfx936），≥ 1 张；正式出片按报告口径为 8 张 | `hy-smi`，或 `python3 -c "import torch;print(torch.cuda.device_count())"` |
| 工具链 | DTK（含 hipcc、hyhal）；`run.sh` 自动探测 `$DTK_ROOT` | `ls /public/software/compiler/ \| grep dtk` |
| Python | 带 torch(ROCm) + diffusers + transformers 的 `python3` 在 PATH 里 | `python3 -c "import torch,diffusers,transformers"` |
| 权重 | MiniMax-H3 FL2VA，diffusers 风格目录（含 `model_index.json`），约 135 GB | 见 §2 |
| 磁盘 | 持久盘 ≥ 30 GB 可用（编译树 + 出片 + 日志） | `df -h` |

权重**不在仓库里**，这是官方模型，需自备。

## 1. 最短路径：clone 后直接跑

```bash
git clone https://github.com/MaiSizzu/MaiSi-r2.git
cd MaiSi-r2
export MODEL_ROOT=/path/to/MiniMax-H3/FL2VA     # 必须指向含 model_index.json 的那层
cp env/h3.env.example "${IFAB_AFS_HOME:-$HOME}/h3.env"   # 8 卡口径，见下
bash run.sh check                                # 只读自检，2 分钟，不写盘不编译
```

`run.sh` 会把自己所在目录当作代码包（日志里打 `PKG = ...（来源：run.sh 所在目录）`），
**不需要**额外准备什么 `MaiSi-r2-full.tar.gz`。

那个 `h3.env` 建议照抄：不复制也能跑，但会退回内置默认 `LEGACY_GPUS=0,1,2,3`
（4 卡、显存 0.4），出片与耗时都对不上报告里的 8 卡口径。文件放持久盘根
（`$AFS/h3.env`，`$AFS` 未设时就是 `$HOME/h3.env`），`run.sh` 每单开头会加载它。

`check` 通过的标准：日志里能看到「可见 DCU = N」「PKG = …（来源：…）」「MODEL_ROOT = …」，
并且没有 `!!` 开头的行。

然后按需要跑：

```bash
bash run.sh setup      # 解包依赖 + 生成首帧（约 5 分钟）
bash run.sh build      # 编译 InfiniCore 并安装到持久盘（ccache 热了 3~5 分钟，冷编约 40 分钟）
bash run.sh compare    # 出我方片 + PyTorch 基准片 + Size/Length/SSIM
bash run.sh full       # 正式出片 124 帧 / 50 步
```

## 2. 权重怎么指

`run.sh` 的解析顺序：

1. 环境变量 `MODEL_ROOT`（**优先**，指对了就跳过自动搜索）
2. `$SHARED/MiniMax-H3/FL2VA`、`$SHARED/MiniMax-H3`、`$AFS/models/...` 等候选
3. `SHARED` 默认值指向前提交方所在组的共享目录，**你们环境里不存在，所以请显式设 `MODEL_ROOT`**

`MODEL_ROOT` 要指向**含 `model_index.json` 的那一层**。MiniMax-H3 的目录结构是
`MiniMax-H3/FL2VA/diffusers/model_index.json`，此时 `MODEL_ROOT` 应设为
`MiniMax-H3/FL2VA`（不是 `MiniMax-H3`，也不是 `.../diffusers`）。

## 3. 平台提交路径（InfiniFabric）

如果按平台方式提交（工作区只有一个 `run.sh`）：

```
程序: run.sh    参数: compare
```

工作区里没有 `src/`，`run.sh` 会依次尝试：

1. `$AFS/MaiSi-r2`（持久盘上已解包的包）
2. 持久盘根目录的 `MaiSi-r2-full.tar.gz`（24 MB）
3. **从本仓库浅克隆**（`git clone --depth 1`，需要工作区能访问 GitHub）

三条都不成时，脚本会退出并打印三种解法，不会盲跑。若环境访问不了 GitHub，
把整包放到 `$AFS/MaiSi-r2` 或设 `H3_PKG=<包路径>` 即可。

## 4. 常见失败与定位

| 日志里看到 | 原因 | 处理 |
| --- | --- | --- |
| `AFS 里找不到 MaiSi-r2-full.tar.gz` | 旧版 run.sh（2,158,044 字节） | 用仓库里这版；或在包根目录直接跑 |
| `可见 DCU = 0` | 卡没分到 / `LD_LIBRARY_PATH` 没带 hyhal | 先 `hy-smi`，再 `source env/dcu_env.sh` |
| `cast_ requires ATen` | 两个运行时 shim 没挂上 | 用 `run.sh` 跑，别绕过它直接调官方脚本 |
| `flash_attn_2_cuda` 加载失败 | 回落到 768×928 + FA2 默认档 | 同上，必须走 `run.sh` |
| `KERNEL VMFault` + `blockSoftmax` | 768×928 下 softmax 需 2×10.64 GB | 已知限制，不是复现失败：用 256×320 档 |
| `Mismatch dtype ... rms_norm` | VAE 被降成 bf16 | `H3_VAE_FP32=1`（本包默认开） |

`run.sh` 的日志一律落在 `$AFS/outputs/run_<mode>_<时间>.log`，失败时最后一行会给路径。

## 5. 我们实际跑过什么、没跑过什么

如实说明，避免评审按不存在的结论验收：

| 项 | 状态 |
| --- | --- |
| 8 卡 DCU，256×320，124 帧 / 50 步 | **跑通**，出片 5.18 s / 24 fps |
| `compare`（PyTorch 基准 vs InfiniLM） | **跑通**，Size PASS / Length PASS / SSIM 37.403% |
| swiglu 算子自测 | **81/81 通过**（`evidence/run_tests.log`） |
| 768×928 正式出片 | **未跑通**：首个去噪步 VMFault，根因与计算吻合，不随卡数缓解 |
| VAE fp32 修复的量化效果 | **未取得**：探针崩在 DiT，从未进入 VAE 解码 |
| relu / kv_caching / flash_attention 接入 | **未完成**（仅生成九齿产物） |

## 6. 环境变量速查

| 变量 | 作用 | 默认 |
| --- | --- | --- |
| `MODEL_ROOT` | 权重根目录 | 自动搜索 |
| `H3_PKG` | 强制指定包路径 | `run.sh` 所在目录 / `$AFS/MaiSi-r2` |
| `IFAB_AFS_HOME` | 持久盘根（大文件都落这） | `$HOME/afs` → `$HOME` |
| `SHARED` | 权重共享目录 | 提交方组目录 |
| `H3_VAE_FP32` | 视频 VAE 走 fp32 | `1` |
| `STEPS` / `NUM_FRAMES` | 去噪步数 / 帧数 | 50 / 124 |
| `LEGACY_GPUS` | 卡号列表 | `0,1,2,3,4,5,6,7` |
| `H3_REPO_URL` / `H3_REPO_BRANCH` | 自动拉代码用的仓库 | 本仓库 / `main` |
