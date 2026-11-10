# vendor/ —— 离线依赖与编译工具

    pylibs-py311-v3.zip                    33 MB，5,378 个文件，解压后是一套可直接 import 的 site-packages
    xmake-bundle-v3.0.7.linux.x86_64      2.9 MB，musl 静态链接的 xmake 3.0.7，`build` 的第一关

## xmake 为什么也在这里

容器里没有 xmake，仓库里原本只有 `xmake.lua` 配置。`build` 走到 `ensure_xmake` 时若找不到
3.x 就直接 `die`，而这一步在评审的复现路径上出现得极早（`build` 开头），却长期没被写进
任何文档——等于「文档说四件就够，实际跑起来第五关就停」。

官方 bundle 只有 2.9 MB，塞进 `vendor/` 的成本比让评审去下载/编译低得多。选 **3.0.7**
是因为：3.0.8 起官方改成动态链接 glibc 2.38，容器里 glibc 更旧，会报
`GLIBC_2.38 not found`；3.0.4~3.0.7 是 musl 静态单文件，任何 glibc 都能跑。2.x 则缺
`add_orders`，本项目用了它，会直接报错。

`ensure_xmake` 的查找顺序：持久盘 `$IFAB_AFS_HOME/xmake-bundle-*` → `$AFS/dl/` →
**本目录（`vendor/`）** → 兜底扫描 `$AFS/xmake*`。找到多个时按版本号从高到低逐个试运行，
能起来的那份才用。想覆盖成别的版本，把文件放持久盘根目录即可，那里优先级更高。

## 依赖包为什么是这个形式

目标容器里 `pip` 用不了：读 `/etc/debian_version` 就抛异常（无权限），`pip install`
任何包都直接崩，也没有可用的内网源。但推理需要 diffusers 0.42 / transformers 5.x /
huggingface_hub 2.x 这一整套配套版本，系统自带的版本和它们死冲突。

所以做法是：先在能联网的机器上把这套包按「已安装的目录结构」收集齐，打成 zip；
`run.sh` 在 `ensure_deps` 里把它**解压**到 `$IFAB_AFS_HOME/pylibs`，再把这个目录放到
`PYTHONPATH` 最前面。不调用 pip、不联网、不写 site-packages。

## 怎么被用到

`find_pylibs_zip` 的查找顺序（见 `run.sh`）：

1. `$IFAB_AFS_HOME/pylibs*-v3.zip`（持久盘上有一份就用它，省得每次从仓库解压）
2. `$IFAB_AFS_HOME/dl/` 下同名文件
3. **`vendor/`（本目录，随包自带）**
4. `$AFS/vendor/`、run.sh 所在目录的 `vendor/`
5. 旧的 `-v2` / 无版本号包

显式指定：`PYLIBS_ZIP=/path/to/pylibs-py311-v3.zip bash run.sh setup`

解压后会校验指纹（zip 的 sha256 前 16 位，写进 `$AFS/pylibs/.ready`），
换包自动重解，同一个包不重复解。

## 版本与内容

| 包 | 关键内容 |
| --- | --- |
| diffusers | 0.42.1（含 `modular_pipelines/minimax_h3`） |
| transformers | 5.x |
| huggingface_hub | 2.x |
| 其他 | accelerate 1.15、safetensors、Pillow、numpy 等推理链所需 |

`transformers>=5` 把 `Qwen2TokenizerFast` 并进了基类，H3 的 `encoders.py` 顶层 import
会失败；`ensure_deps` 会在解压后对那一行打一个幂等补丁（`Qwen2TokenizerFastFallback`），
所以**不要**直接改 zip 里的文件——下次解压会被覆盖。

## 许可

这里的 `.whl`/`.py` 全部来自上游开源项目（Apache-2.0 / BSD / MIT 等），未做修改
（除上述 `encoders.py` 的运行时补丁，且补丁写在解压后的副本上，不在 zip 内）。
仅作为「离线镜像」随本提交分发，版权归各上游项目所有；需要时请以 PyPI 上的官方
发行版为准。

## 自己重建一份

在能联网、python 3.11 的环境里：

```bash
python -m pip download -d wheels -r env/requirements.lock   # 只下 wheel，不安装
python - <<'EOF'
import zipfile, os
with zipfile.ZipFile("pylibs-py311-v3.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk("site-packages"):
        for f in files:
            p = os.path.join(root, f)
            z.write(p, os.path.relpath(p, "site-packages"))
EOF
```

也可以直接复制一个能 `import torch, diffusers, transformers` 的 `site-packages` 目录
（不含 torch 的 `.so` 与 CUDA/ROCm 运行库——那些由平台镜像提供），按同样结构压平后打包。
`run.sh` 只按目录名取用，不校验来源。
