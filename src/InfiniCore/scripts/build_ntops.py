import concurrent.futures
import importlib
import pathlib
import sys

CURRENT_FILE_PATH = pathlib.Path(__file__)

SRC_DIR_PATH = CURRENT_FILE_PATH.parent.parent / "src"

# 与 infiniop.ninetoothed.build.BUILD_DIRECTORY_PATH 指向同一目录
# （<InfiniCore>/build/ninetoothed）。这里不 import 它，避免在没装
# ninetoothed 的机器上提前失败。
BUILD_DIR = CURRENT_FILE_PATH.parent.parent / "build" / "ninetoothed"


def _existing_products():
    if not BUILD_DIR.is_dir():
        return []
    return sorted(BUILD_DIR.glob("*.cpp")) + sorted(BUILD_DIR.glob("*.c"))


def _find_and_build_ops():
    ops_path = SRC_DIR_PATH / "infiniop" / "ops"

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []

        for op_dir in ops_path.iterdir():
            ninetoothed_path = op_dir / "ninetoothed"

            if not ninetoothed_path.is_dir():
                continue

            build_file = ninetoothed_path / "build.py"
            if not build_file.exists():
                continue

            futures.append(executor.submit(_build, ninetoothed_path))

        for future in concurrent.futures.as_completed(futures):
            future.result()


def _build(ninetoothed_path):
    module_path = ninetoothed_path / "build"
    relative_path = module_path.relative_to(SRC_DIR_PATH)
    import_name = ".".join(relative_path.parts)
    module = importlib.import_module(import_name)

    module.build()


if __name__ == "__main__":
    force = "--force" in sys.argv

    # 生成产物已随仓库交付：默认直接复用，不要求机器上装有 ninetoothed 这个
    # pip 包。只有修改过算子定义（src/infiniop/ops/*/ninetoothed/）时才需要
    # 用 --force 重新生成。
    existing = [] if force else _existing_products()
    if existing:
        print(
            "[build_ntops] 复用仓库内已有的 %d 个生成产物（%s），跳过重新生成。"
            % (len(existing), BUILD_DIR)
        )
        print(
            "[build_ntops] 仅在修改过算子定义后才需要重跑：\n"
            "[build_ntops]   pip install ninetoothed    # 首次需要\n"
            "[build_ntops]   python scripts/build_ntops.py --force"
        )
        raise SystemExit(0)

    try:
        import ninetoothed  # noqa: F401
    except ImportError:
        print(
            "[build_ntops] 未安装 ninetoothed，且仓库内没有可复用的生成产物。\n"
            "[build_ntops] 先执行：pip install ninetoothed\n"
            "[build_ntops] 若只是编译，请确认 build/ninetoothed/ 下有 *.cpp（随仓库提供）。",
            file=sys.stderr,
        )
        raise SystemExit(2)

    from infiniop.ninetoothed.build import BUILD_DIRECTORY_PATH

    BUILD_DIRECTORY_PATH.mkdir(parents=True, exist_ok=True)

    _find_and_build_ops()
