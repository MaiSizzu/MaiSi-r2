import glob
import os
import re
import sys
import tempfile

PATTERN = re.compile("([0-9]+) [*] 32")


def fix(directory):
    files = sorted(glob.glob(os.path.join(directory, "*.cpp")))
    if not files:
        sys.exit("no generated .cpp under " + directory)
    changed = 0
    for path in files:
        with open(path, "r", errors="surrogateescape") as fh:
            text = fh.read()
        hits = PATTERN.findall(text)
        if not hits:
            continue
        new_text = PATTERN.sub(lambda m: m.group(1) + " * 64", text)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
        with os.fdopen(fd, "w", errors="surrogateescape") as fh:
            fh.write(new_text)
        os.replace(tmp, path)
        changed += len(hits)
    residual = 0
    for path in files:
        with open(path, "r", errors="surrogateescape") as fh:
            residual += len(PATTERN.findall(fh.read()))
    print("patched sites:", changed, flush=True)
    print("residual sites:", residual, flush=True)
    if residual:
        sys.exit("ERROR: residual warp-size-32 launch found")
    if changed == 0:
        print("OK: already fixed, nothing to do", flush=True)


fix(sys.argv[1] if len(sys.argv) > 1 else "build/ninetoothed")
