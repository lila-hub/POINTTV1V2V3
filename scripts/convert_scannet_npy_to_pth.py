#!/usr/bin/env python3
"""将 scannet-compressed 解压后的 scene 子目录 (coord.npy 等) 转为 Pointcept 所需的 split/*.pth。"""
import os
import sys
import glob
import numpy as np
import torch

IGNORE = -1


def scene_dir_to_pth(scene_dir: str, out_pth: str) -> bool:
    coord_f = os.path.join(scene_dir, "coord.npy")
    if not os.path.isfile(coord_f):
        return False
    coord = np.load(coord_f)
    color = np.load(os.path.join(scene_dir, "color.npy")) if os.path.isfile(
        os.path.join(scene_dir, "color.npy")
    ) else np.zeros_like(coord)
    normal = np.load(os.path.join(scene_dir, "normal.npy")) if os.path.isfile(
        os.path.join(scene_dir, "normal.npy")
    ) else np.zeros_like(coord)
    n = coord.shape[0]
    scene_id = os.path.basename(scene_dir.rstrip(os.sep))

    sem = None
    for name in (
        "semantic_gt20.npy",
        "segment.npy",
        "semantic.npy",
        "label.npy",
    ):
        p = os.path.join(scene_dir, name)
        if os.path.isfile(p):
            sem = np.load(p).reshape(-1)
            break
    if sem is None:
        sem = np.ones(n, dtype=np.int64) * IGNORE

    inst = None
    for name in ("instance_gt.npy", "instance.npy"):
        p = os.path.join(scene_dir, name)
        if os.path.isfile(p):
            inst = np.load(p).reshape(-1)
            break
    if inst is None:
        inst = np.ones(n, dtype=np.int64) * IGNORE

    data = {
        "coord": coord.astype(np.float32),
        "color": color.astype(np.float32),
        "normal": normal.astype(np.float32),
        "scene_id": scene_id,
        "semantic_gt20": sem.astype(np.int64),
        "instance_gt": inst.astype(np.int64),
    }
    os.makedirs(os.path.dirname(out_pth), exist_ok=True)
    torch.save(data, out_pth)
    return True


def main():
    _proj = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    _default = os.path.join(_proj, "data", "scannet_processed")
    root = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else _default)
    for split in ("train", "val", "test"):
        sp = os.path.join(root, split)
        if not os.path.isdir(sp):
            continue
        subdirs = [d for d in glob.glob(os.path.join(sp, "*")) if os.path.isdir(d)]
        n_ok = 0
        for d in sorted(subdirs):
            name = os.path.basename(d)
            if name.endswith(".pth"):
                continue
            out = os.path.join(sp, f"{name}.pth")
            if os.path.isfile(out):
                continue
            if scene_dir_to_pth(d, out):
                n_ok += 1
        print(f"{split}: converted {n_ok} scenes (new .pth next to scene dirs)")


if __name__ == "__main__":
    main()
