#!/usr/bin/env python3
"""从 ScanNet test 划分中取 1/10 场景，在 test_sub10 目录下建立指向 .pth 的符号链接。"""
import glob
import os
import sys

_PROJECT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_DEFAULT_DATA = os.path.join(_PROJECT, "data", "scannet_processed")

DATA_ROOT = os.path.expanduser(
    sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_DATA
)
TEST_DIR = os.path.join(DATA_ROOT, "test")
OUT_DIR = os.path.join(DATA_ROOT, "test_sub10")


def main():
    files = sorted(glob.glob(os.path.join(TEST_DIR, "*.pth")))
    if not files:
        print(f"FAILED: no .pth in {TEST_DIR}", file=sys.stderr)
        sys.exit(1)
    n = max(1, len(files) // 10)
    sel = files[:n]
    os.makedirs(OUT_DIR, exist_ok=True)
    for f in sel:
        base = os.path.basename(f)
        dst = os.path.join(OUT_DIR, base)
        try:
            if os.path.lexists(dst):
                os.remove(dst)
        except OSError:
            pass
        os.symlink(os.path.abspath(f), dst)
    print(f"Linked {len(sel)} / {len(files)} test scenes -> {OUT_DIR}")
    for s in sel:
        print(" ", os.path.basename(s))


if __name__ == "__main__":
    main()
