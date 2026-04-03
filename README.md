# PointTv1v2v3

本仓库在 **Pointcept** 上组织 **Point Transformer v1 / v2 / v3**（ScanNet 语义分割等）的实验脚本、配置与说明。代码主体来自上游 [Pointcept](https://github.com/Pointcept/Pointcept)，本仓库在根目录增加 **数据布局、权重路径、一键脚本与部分运行结果**。

**远程仓库**：<https://github.com/triplesf/PointTv1v2v3>

### 可视化示例（PTv3 · ScanNet test 子集）

以下为 **PTv3** 在 **test_sub10**（约 1/10 test 场景）上的预测 **三维散点** 图（由 `scripts/make_sub10_vis_and_report.py` 生成；每图含两种视角）。颜色表示 **预测语义类别 id**（`tab20` 色带），详细类别与色值见同目录下的报告与图例。

<p align="center">
  <b>scene0707_00</b> · <b>scene0711_00</b><br/>
  <img src="runs/ptv3_test_sub10/vis/scene0707_00_pred_3d.png" alt="scene0707_00 3D" width="48%" />
  <img src="runs/ptv3_test_sub10/vis/scene0711_00_pred_3d.png" alt="scene0711_00 3D" width="48%" />
</p>
<p align="center">
  <b>scene0714_00</b> · <b>scene0715_00</b><br/>
  <img src="runs/ptv3_test_sub10/vis/scene0714_00_pred_3d.png" alt="scene0714_00 3D" width="48%" />
  <img src="runs/ptv3_test_sub10/vis/scene0715_00_pred_3d.png" alt="scene0715_00 3D" width="48%" />
</p>

**类别—颜色对照**：[`class_color_legend_tab20.png`](runs/ptv3_test_sub10/vis/class_color_legend_tab20.png) · **报告**：[`SUB10_REPORT.md`](runs/ptv3_test_sub10/SUB10_REPORT.md)

---

## Pointcept 是什么

**Pointcept** 是开源的 **3D 点云深度学习框架**：统一数据管线、增强、`tools/train.py` / `tools/test.py` 入口，并在同一套代码里注册多种 **backbone** 与数据集（ScanNet、S3DIS 等）。本仓库中的 **`Pointcept/`** 目录即该框架的源码树；训练与测试均需进入该目录并设置 `export PYTHONPATH=./`。

---

## Point Transformer v1 / v2 / v3 在本仓库中的位置

三者均由 Pointcept **内置实现**，通过 **配置文件** 选择不同 `backbone.type`，并非三个互不相关的独立项目。

| 版本 | 实现代码（`Pointcept/pointcept/models/`） | ScanNet 语义分割配置示例（`Pointcept/configs/scannet/`） | 配置中的典型骨干名 |
|------|-------------------------------------------|--------------------------------------------------------|----------------------|
| **PT v1** | `point_transformer/`（如 `point_transformer_seg.py`） | `semseg-pt-v1-0-base.py` | `PointTransformer-Seg50` |
| **PT v2** | `point_transformer_v2/`（如 `point_transformer_v2m1_origin.py`、`point_transformer_v2m2_base.py`） | `semseg-pt-v2m2-0-base.py`、`semseg-pt-v2m1-0-origin.py` 等 | `PT-v2m1`、`PT-v2m2` 等 |
| **PT v3** | `point_transformer_v3/`（如 `point_transformer_v3m1_base.py`） | `semseg-pt-v3m1-0-base.py` | `PT-v3m1` |

**说明**：配置里另有 **`PPT-v1m1`**（Point Prompt Training 等，如 `semseg-ppt-v1m1-0-sc-st-spunet.py`），名字含 “v1”，但与上表中的 **Point Transformer v1（`semseg-pt-v1-*`）** 不是同一系列，选用时请看配置文件名与 `backbone.type`。

**运行方式（与 Pointcept 官方一致）**：在 `Pointcept/` 下执行，例如测试 PTv3：

```bash
cd Pointcept && export PYTHONPATH=./
python tools/test.py --config-file configs/scannet/semseg-pt-v3m1-0-base.py --num-gpus 1 --options save_path=... weight=... 
```

PTv1 / PTv2 仅需将 `--config-file` 换成上表中对应配置文件路径，并准备相应权重与数据。

---

## 目录说明（与当前仓库对齐）

| 路径 | 作用 |
|------|------|
| **`Pointcept/`** | Pointcept 框架源码：`configs/`（含 **PTv1 / PTv2 / PTv3** 等配置）、`tools/`（train/test）、`pointcept/models/`（含 `point_transformer*`、`point_transformer_v2`、`point_transformer_v3`）、`libs/`（pointops 等）。训练/测试时进入此目录并设置 `PYTHONPATH=./`。详见上文「Point Transformer v1 / v2 / v3 在本仓库中的位置」。 |
| **`scripts/`** | 本仓库维护的辅助脚本：子集划分、可视化报告、资产迁移、PTv2 全流程检测等（见下表）。 |
| **`data/`** | **本地 ScanNet 等预处理数据根目录**（如 `scannet_processed/`）。**默认不提交到 Git**（体积大）。首次可用 `relocate_assets.sh` 从 `~/datasets` 迁入。 |
| **`models/`** | **预训练权重**（如 Hugging Face 下载的 `ptv3/`）。**默认不提交**（单文件常超 GitHub 100MB 限制）。 |
| **`runs/`** | 推理与实验输出：各次 `save_path` 下的 `config.py`、`result/`（预测 `.npy` / submit）、`vis/`（可视化图）、`SUB10_REPORT.md` 等；**已纳入版本库的部分**以仓库内实际文件为准。 |
| **`logs/`** | 可选：其它后台测试等日志目录。 |
| **`LAYOUT.txt`** | 中文布局说明（与 `README` 互补）。 |
| **`relocate_assets.sh`** | 将 `~/datasets/scannet_processed`、`~/models/ptv3`、整个 `~/runs` 合并进本仓库对应目录，并链接 `Pointcept/data/scannet` → 本仓库 `data/scannet_processed`。 |
| **`POINTCEPT_GIT_NOTE.txt`** | 说明本仓库如何将原独立 `Pointcept/.git` 备份，便于与上游同步或恢复。 |
| **`import_check.txt`** | 简单环境/路径检查占位（若有）。 |

### `scripts/` 脚本一览

| 脚本 | 说明 |
|------|------|
| `run_sub10_pipeline.sh` | 1/10 ScanNet test 子集：划分 → `tools/test.py` → 可视化与 `SUB10_REPORT.md`。 |
| `make_test_sub10_split.py` | 从 `test/` 中选约 1/10 场景，生成 `test_sub10` 软链。 |
| `make_sub10_vis_and_report.py` | 读取子集预测与点云，生成多视角/3D 可视化 PNG 与报告。 |
| `sync_sub10_vis_from_home.sh` | 将 `~/runs/ptv3_test_sub10/vis` 同步到本仓库 `runs/`。 |
| `relocate_assets.sh`（根目录） | 资产迁入（见上）。 |
| `ptv2_pointcept_full_pipeline.sh` | PTv2：检查 Pointcept 配置、数据链、权重搜索、可选测试与 legacy 回退流程。 |
| `continue_ptv3_from_stage7.sh` / `complete_ptv3.sh` 等 | PTv3 下载与测试流水线（见脚本内注释）。 |
| `convert_scannet_npy_to_pth.py` | 数据格式转换辅助。 |
| `check_import_path.py` | 导入路径检查。 |

---

## 环境约定

- **Conda 环境名**：常用 `pointcept`（与 Pointcept 文档一致）。
- **数据**：`Pointcept/data/scannet` 应指向本仓库的 `data/scannet_processed`（`relocate_assets.sh` 会创建软链）。
- **权重**：测试时在配置或命令行中指定 `weight=` 指向本地 `models/ptv3/.../model_best.pth` 等。

---

## 克隆后快速准备

1. 克隆本仓库。  
2. 准备 **ScanNet 预处理数据** 与 **权重**（自行下载或从本机拷贝到 `data/`、`models/`）。  
3. 执行：  
   `bash relocate_assets.sh`（若数据仍在 `~/datasets`、`~/models`、`~/runs`）。  
4. 激活 conda，进入 `Pointcept/`，设置 `export PYTHONPATH=./`，再按 Pointcept 文档运行 `tools/train.py` / `tools/test.py`。

---

## Git 与 GitHub 说明

- **不跟踪**：`/data/`、`/models/`、`*.log`（见根目录 `.gitignore`），避免超大文件与密钥进入仓库。  
- **权重与全量数据**：请使用 [Hugging Face](https://huggingface.co/) 或网盘等单独分发。  
- 推送前请在本机执行：  
  `git push -u origin main`  
  （HTTPS 需 Personal Access Token；或改用 SSH。）

---

## 上游与引用

- Pointcept：<https://github.com/Pointcept/Pointcept>  
- 若需恢复本仓库内 Pointcept 独立 git 历史，见 **`POINTCEPT_GIT_NOTE.txt`**。

---

## 论文与相关链接

与本仓库中 **Point Transformer v1 / v2 / v3** 及框架直接相关的论文（与 [Pointcept README](https://github.com/Pointcept/Pointcept) 一致）：

| 工作 | 说明 | 链接 |
|------|------|------|
| **Point Transformer V3** | *Simpler, Faster, Stronger*（PTv3） | [arXiv:2312.10035](https://arxiv.org/abs/2312.10035) · [项目](https://github.com/Pointcept/PointTransformerV3) · [Bib](https://xywu.me/research/ptv3/bib.txt) |
| **Point Transformer V2** | *Grouped Vector Attention and Partition-based Pooling*（NeurIPS 2022） | [arXiv:2210.05666](https://arxiv.org/abs/2210.05666) · [Bib](https://xywu.me/research/ptv2/bib.txt) |
| **Point Transformer** | 原版 PT（ICCV 2021 Oral）（常称 PTv1） | [arXiv:2012.09164](https://arxiv.org/abs/2012.09164) · [Bib](https://hszhao.github.io/papers/iccv21_pointtransformer_bib.txt) |
| **Pointcept** | 点云感知代码库说明（引用仓库） | [GitHub](https://github.com/Pointcept/Pointcept) |

数据集与评测常用：**[ScanNet](http://www.scan-net.org/)**。
