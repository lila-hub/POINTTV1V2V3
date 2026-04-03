#!/usr/bin/env python3
"""读取子集推理结果，生成 10 张三维散点可视化与 Markdown 报告。"""
import glob
import hashlib
import os
import sys
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection

def _project_root():
    """本仓库根目录（pointv3/，脚本位于 scripts/ 下）。"""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


CLASS_NAMES = [
    "wall",
    "floor",
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "desk",
    "curtain",
    "refridgerator",
    "shower curtain",
    "toilet",
    "sink",
    "bathtub",
    "otherfurniture",
]


def _tab20_norm():
    """与 scatter(c=pred, cmap=tab20, vmin=0, vmax=19) 一致的颜色归一化。"""
    return mpl.colors.Normalize(vmin=0, vmax=19)


def _class_hex_colors():
    cmap = plt.cm.tab20
    norm = _tab20_norm()
    return [mpl.colors.to_hex(cmap(norm(i))) for i in range(20)]


def write_class_color_legend_png(vis_dir):
    """一张图：20 类 × tab20 颜色 + 名称（与预测可视化色带一致）。"""
    path = os.path.join(vis_dir, "class_color_legend_tab20.png")
    cmap = plt.cm.tab20
    norm = _tab20_norm()
    fig, ax = plt.subplots(figsize=(8, 10), dpi=120)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 20)
    ax.set_title("ScanNet20: class id -> color (tab20, matches 3D plots)", fontsize=12)
    for i in range(20):
        y = 19 - i
        color = cmap(norm(i))
        rect = mpatches.Rectangle((0.05, y + 0.15), 0.12, 0.7, facecolor=color, edgecolor="0.3", linewidth=0.5)
        ax.add_patch(rect)
        ax.text(0.22, y + 0.5, f"id={i:2d}  {CLASS_NAMES[i]}", va="center", fontsize=9, family="monospace")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)
    return path


# 3D 散点过多时既慢又糊成一片；下采样后仍具代表性
MAX_POINTS_3D = 120_000


def _subsample_indices(n: int, max_points: int, scene: str) -> np.ndarray:
    if n <= max_points:
        return np.arange(n, dtype=np.int64)
    seed = int.from_bytes(hashlib.md5(scene.encode("utf-8")).digest()[:4], "little")
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=max_points, replace=False)


def main():
    root = _project_root()
    data_sub = os.path.join(root, "data/scannet_processed/test_sub10")
    result_dir = os.path.join(root, "runs/ptv3_test_sub10/result")
    vis_dir = os.path.join(root, "runs/ptv3_test_sub10/vis")
    report_path = os.path.join(root, "runs/ptv3_test_sub10/SUB10_REPORT.md")
    os.makedirs(vis_dir, exist_ok=True)

    scenes = sorted(
        [
            os.path.basename(f)[:-4]
            for f in glob.glob(os.path.join(data_sub, "*.pth"))
        ]
    )
    if not scenes:
        print("No scenes in test_sub10", file=sys.stderr)
        sys.exit(1)

    fig_md = []
    for scene in scenes:
        pth = os.path.join(data_sub, f"{scene}.pth")
        pred_path = os.path.join(result_dir, f"{scene}_pred.npy")
        if not os.path.isfile(pred_path):
            print(f"skip {scene}: missing {pred_path}", file=sys.stderr)
            continue
        data = torch.load(pth, map_location="cpu")
        coord = np.asarray(data["coord"], dtype=np.float32)
        pred = np.load(pred_path)
        m = min(coord.shape[0], pred.shape[0])
        coord = coord[:m]
        pred = pred[:m].astype(np.int64)
        pred = np.clip(pred, 0, 19)

        idx = _subsample_indices(m, MAX_POINTS_3D, scene)
        c = coord[idx]
        p = pred[idx]

        cmap = plt.cm.tab20
        norm = _tab20_norm()
        xyz_min = coord.min(axis=0)
        xyz_max = coord.max(axis=0)
        xyz_rng = np.maximum(xyz_max - xyz_min, 1e-6)

        fig = plt.figure(figsize=(14, 6.2), dpi=120)
        # 两个视角：斜俯视 + 侧视，便于看出房间几何
        view_params = [
            (28, -62, "view A: oblique"),
            (12, 120, "view B: side"),
        ]
        axes3d = []
        sc_last = None
        for i, (elev, azim, subtitle) in enumerate(view_params):
            ax = fig.add_subplot(1, 2, i + 1, projection="3d")
            sc_last = ax.scatter(
                c[:, 0],
                c[:, 1],
                c[:, 2],
                c=p,
                cmap=cmap,
                norm=norm,
                s=0.35,
                linewidths=0,
                alpha=0.82,
                depthshade=True,
            )
            ax.view_init(elev=elev, azim=azim)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_title(subtitle, fontsize=10)
            ax.set_box_aspect(tuple(xyz_rng))
            axes3d.append(ax)
        fig.suptitle(
            f"{scene} - 3D pred (tab20), sampled up to {MAX_POINTS_3D} pts",
            fontsize=11,
        )
        plt.tight_layout()
        fig.subplots_adjust(top=0.86)
        cbar = fig.colorbar(
            sc_last,
            ax=axes3d,
            fraction=0.03,
            pad=0.06,
            ticks=range(0, 20),
        )
        cbar.set_label("class id (0–19)")
        out_png = os.path.join(vis_dir, f"{scene}_pred_3d.png")
        plt.savefig(out_png, bbox_inches="tight")
        plt.close(fig)
        rel = f"vis/{scene}_pred_3d.png"
        fig_md.append((scene, rel))
        print("wrote", out_png)

    hex_colors = _class_hex_colors()
    write_class_color_legend_png(vis_dir)

    log_path = os.path.join(os.path.dirname(vis_dir), "test_sub10.log")
    metrics_note = ""
    if os.path.isfile(log_path):
        with open(log_path, "r", encoding="utf-8", errors="ignore") as lf:
            for line in lf:
                if "Val result:" in line or "mIoU/mAcc/allAcc" in line:
                    metrics_note = line.strip()
                    break

    lines = [
        "# PTv3 ScanNet 测试子集（1/10）可视化报告",
        "",
        f"- 生成时间：{datetime.now().isoformat(timespec='seconds')}",
        f"- 子集目录：`{data_sub}`（test 集约 1/10 场景）",
        f"- 推理输出：`{result_dir}`",
        f"- 本报告：`{report_path}`",
        f"- 完整测试日志：`{log_path}`",
        "",
        "## 推理摘要",
        "",
        "| 项目 | 说明 |",
        "|------|------|",
        "| 场景数 | 10（在 `test` 共 100 个 `.pth` 中取约 1/10） |",
    ]
    if metrics_note:
        lines.extend(
            [
                f"| 日志汇总行 | `{metrics_note}` |",
            ]
        )
    lines.extend(
        [
            "| 指标解读 | ScanNet **官方 test** 无公开语义真值；转换数据里标签多为占位，**mIoU 等数值通常无意义**。下图为 **预测类别** 的 **三维散点**（双视角）。 |",
            "",
            "## 颜色与类别（matplotlib `tab20`）",
            "",
            "点颜色由 **预测类别 id（0–19）** 映射到 `tab20` 色带；**与 `make_sub10_vis_and_report.py` 中 `vmin=0, vmax=19` 的 `Normalize` 一致**。",
            "",
            "![类别颜色对照](vis/class_color_legend_tab20.png)",
            "",
            "| id | 类别 | 色值（hex，tab20） |",
            "|----|------|---------------------|",
        ]
    )
    for i, n in enumerate(CLASS_NAMES):
        lines.append(f"| {i} | {n} | `{hex_colors[i]}` |")
    lines.extend(
        [
            "",
            "## 可视化说明",
            "",
            "各场景为 **matplotlib 三维散点**：左/右为两种视角（方位角与仰角不同），颜色为 **预测语义类别 id**（0–19）。点云超过 "
            f"{MAX_POINTS_3D} 点时 **随机下采样**（每场景固定种子），以兼顾速度与可读性。",
            "",
        ]
    )
    for scene, rel in fig_md:
        lines.append(f"### {scene}")
        lines.append("")
        lines.append(f"![{scene}]({rel})")
        lines.append("")
    lines.extend(
        [
            "## 复现命令摘要",
            "",
            "```bash",
            "conda activate pointcept",
            "cd ~/pointv3 && python scripts/make_test_sub10_split.py",
            "cd ~/pointv3/Pointcept && export PYTHONPATH=./",
            "python tools/test.py --config-file ~/pointv3/models/ptv3/scannet-semseg-pt-v3m1-0-base/config.py \\",
            "  --num-gpus 1 --options save_path=~/pointv3/runs/ptv3_test_sub10 \\",
            "  weight=~/pointv3/models/ptv3/scannet-semseg-pt-v3m1-0-base/model/model_best.pth \\",
            "  data.test.split=test_sub10 num_worker=4",
            "python ~/pointv3/scripts/make_sub10_vis_and_report.py",
            "```",
            "",
        ]
    )
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("Report:", report_path)


if __name__ == "__main__":
    main()
