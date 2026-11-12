# ID-V2V：身份保持式视频风格迁移

[![项目主页](https://img.shields.io/badge/Project-Page-yellow)](https://eyeline-labs.github.io/ID-V2V/) [![论文](https://img.shields.io/badge/Paper-arXiv-b31b1b?logo=arxiv&logoColor=red)](https://arxiv.org/abs/2607.22830)

<b>🎉 已被 SIGGRAPH Asia 2026 接收</b>

[Xu Yuancheng](https://yuancheng-xu.github.io/)<sup>1,2</sup>, [He Mingming](https://mingminghe.com/)<sup>2</sup>, [Pablo Salamanca](https://pablosalaman.ca/)<sup>1,2</sup>, [Ma Li](https://limacv.github.io/homepage/)<sup>2</sup>, [Yash Kant](https://yashkant.github.io/)<sup>1,2</sup>, [Emmett Steven](https://www.linkedin.com/in/emmettsteven)<sup>1</sup>, [Paul Debevec](https://www.pauldebevec.com/)<sup>1,2</sup>, [Ning Yu](https://ningyu1991.github.io/)<sup>1,2</sup> <br/>
<sup>1</sup>Netflix, <sup>2</sup>Eyeline Labs <br/>

# 👁️ 概览

![ID-V2V teaser](assets/teaser.jpg)

**ID-V2V** 是一项关于身份保持式视频风格迁移的探索性研究，仅供演示与参考。给定一段**源视频**和一帧**风格化关键帧**（可附带额外关键帧），它会生成一段新视频——在保持源视频中人物身份、表情、眼神、肢体动作等细节不变的前提下，让场景、光照和整体风格跟随关键帧变化。这使得"先拍摄、后风格化"的视觉创作流程成为可能。

<video src="https://github.com/user-attachments/assets/6783f6d0-8993-4143-9a4c-dde2a2cfdeee" controls muted width="720"></video>

本仓库包含：

- [环境搭建](#-环境搭建) — 安装依赖环境 + 下载模型权重
- [模型推理](#-模型推理) — 预处理 → 生成的完整流程，含输入/输出结构说明
- [详细使用场景](#-详细使用场景) — 风格迁移、不完美关键帧、长视频、多关键帧、重打光
- [两种模型变体](#-两种模型变体) — 默认 `idv2v` 与可选的法线-深度增强版
- [仓库目录结构](#-仓库目录结构)

社区已将 ID-V2V 移植到 [ComfyUI](https://github.com/Comfy-Org/ComfyUI/pull/15139) 和 [Fal API](https://fal.ai/models/fal-ai/id-v2v)。

# 🔧 环境搭建

统一使用 `uv` 管理环境，同时覆盖预处理与推理。

```bash
git clone https://github.com/Eyeline-Labs/ID-V2V && cd ID-V2V
uv sync && source .venv/bin/activate
```

Python 3.10, torch 2.6+cu118, transformers 5.9, xfuser, flash-attn 2.7.。官方测试环境为 8× A100-80GB。`ffmpeg` 首次运行时自动安装。

SAM3 权重托管在 Hugging Face 上，需先登录一次，然后将所有权重下载到 `checkpoints/` 目录：

```bash
hf auth login                          # 粘贴有 facebook/sam3 读取权限的 token
bash scripts/download_checkpoints.sh   # idv2v.pth + SAM3 + Wan2.1 的 T5 + VAE + tokenizer + CLIP (~96 GB)
```

下载内容包括 **`idv2v.pth`**（来自 [`Eyeline-Labs/ID-V2V`](https://huggingface.co/Eyeline-Labs/ID-V2V) → `checkpoints/idv2v.pth`），外加 SAM3 和 Wan2.1 的 T5 + VAE + tokenizer + CLIP——默认模型所需的全部内容。如果已有 checkpoint，可加 `--skip-idv2v` 跳过，并设置 `MODEL_CHECKPOINT` 指向本地路径。每个文件的来源见 [`checkpoints/README.md`](checkpoints/README.md)。

### ⚠️ RTX 5090 (Blackwell sm_120) 上的本地部署差异

README 描述的是 torch 2.6+cu118 + A100/sm_80 一把梭的官方路径。**Blackwell sm_120 GPU 不支持 torch 2.6**（最高只到 sm_90），torch 栈必须升级到 **2.7.0+cu128**。在 231 单卡 5090 上实测有三点需要特别注意：

| 项 | 官方 README | 231 实测 |
|---|---|---|
| torch + CUDA | 2.6+cu118 | **2.7.0+cu128**（sm_120 硬需求） |
| env 数量 | 1 个 `uv` venv 覆盖预处理+推理 | **2 个 conda env 分工**：`idv2v`（Python 3.10, 原生带 sam2, 负责预处理）+ `hyavatar`（Python 3.11, Blackwell 就绪, 负责推理）。强行合并需要在 hyavatar 装 sam2（慢编译 C extension）或重编 idv2v 的 flash-attn，性价比低 |
| SAM 权重 | SAM3（HF `facebook/sam3` 仓库） | 代码实际调用 **SAM2.1**（`sam2.1_hiera_base_plus.pt`，下载到 `checkpoints/VACE-Annotators/sam2/`）；README 写 SAM3 是泛称 |

此外，231 上 checkpoint 放在 NFS `/nfs-data/`（本地盘空间不够 73GB）、`STAGE_CHECKPOINT_TO_SHM=false`（126GB `/dev/shm` 放 73GB 会爆）、启动前 `swapoff -a` + `vm.max_map_count=16777216`。完整流程与踩坑见 [new_video_inference_guide.md](new_video_inference_guide.md) 和 [deploy_progress.md](deploy_progress.md)。

# 🎨 模型推理

ID-V2V 接收**一段源视频**、**一帧风格化首帧**（你想要的视觉风格）、可选的**额外关键帧**和**文本提示词**，生成一段风格跟随关键帧变化但身份与表演保持不变的视频。流程分两步：先**预处理**从源视频提取控制信号，再**生成**新视频。生成分辨率固定为 720p。

## 输入：样例目录

每次运行从一个独立的样例目录读取输入。完整示例（含多关键帧 + 长视频）如下：

```
my_sample/
├── source.mp4                 (必需)  源视频（任意长度）
├── stylized_first_frame.png   (必需)  风格化首帧——你想要的视觉效果 (0 号帧)
├── prompt.txt                 (必需)  文本提示词
└── keyframes/                 (可选)  任意帧位置的额外锚点
    ├── 40.png                         文件名 N = 在第 N 帧注入（从 0 起算，N ≥ 1）
    └── 80.png
```

`stylized_first_frame.png` 锁定第 0 帧；`keyframes/<N>.png` 锁定第 N 帧。这些风格化帧可以用任意图像编辑工具制作（如 NanoBanana）。若源视频超过 81 帧，模型会按 clip 分块依次生成。

## 预处理

默认 `idv2v` 模型依赖单一控制信号——**前景灰度图**（SAM3（代码实际使用 SAM2.1 `sam2.1_hiera_base_plus.pt`，见[环境搭建](#-环境搭建)的部署差异）分割出人物，背景置灰）：

```bash
# SAMPLE_DIR 是输入目录（这里用仓库自带的 restylization 示例；换成你自己的即可）
# SAM_PROMPT 是 SAM3（SAM2.1 实现）分割提示词（默认 "person"；如 "head", "dog"）：
SAMPLE_DIR=test_samples/restylization/two_sitting_woman SAM_PROMPT=person bash scripts/preprocess.sh
```

处理结果写入 `<SAMPLE_DIR>/preprocessing/orig_pixel.mp4`。唯一例外是重打光场景——不需要预处理，见[详细使用场景](#-详细使用场景)。

## 生成

在同一个 `SAMPLE_DIR` 上运行（使用"环境搭建"中下载的 `checkpoints/idv2v.pth`）：

```bash
SAMPLE_DIR=test_samples/restylization/two_sitting_woman bash scripts/infer.sh
```

- **多 GPU**：当 `GPU` 是逗号分隔且数量大于 1 时自动启用（如 `"0,1,2,3,4,5,6,7"`），通过 `torchrun` + USP 序列并行启动。单卡设置 `GPU="0"`（普通 `python` + CPU offload；速度较慢但一张卡能放下）。
- **关键帧**：从 `<SAMPLE_DIR>/keyframes/<N>.png` 自动发现。
- **`MAX_NUM_FRAMES`**：限制输出总长度。视频以重叠的 81 帧 clip 生成（`NUM_FRAMES_PER_CLIP=81`），更长的源视频只需生成更多 clip——没有独立的长视频模式。
- checkpoint 加载时使用 mmap，所有 GPU 共享同一份拷贝。`STAGE_CHECKPOINT_TO_SHM=true`（默认）会先把 `.pth` 拷到 `/dev/shm`（RAM），在慢速/网络存储上有大幅加速；但要求空闲 RAM ≥ checkpoint 大小——若 CPU OOM 请设为 `false`。
- **耗时参考**（单卡 RTX 5090，30 step）：81 帧单 clip ≈ 50 min；240 帧（3 clip）≈ 3.5 h。多 clip 时单 step 会从 ~98s 涨到 ~140s。
- **输入宽高比**：模型固定输出 1280×720，竖屏源视频（如 768×1344）会被 center-crop 掉上下部分，构图偏窄。输出侧对比视频已支持横竖屏混排（统一高度后横向拼接），不会因比例不同报错。

> 运行时会打印一条无害的 `diffsynth` 警告（`using --use_multi_control_vace but only 1 control conditions is provided`），属于预期行为，可安全忽略。

> 本机部署（231 服务器 + 5090）的完整流程、踩坑记录和参数说明见 [new_video_inference_guide.md](new_video_inference_guide.md) 与 [deploy_progress.md](deploy_progress.md)。

## 输出结构

```
my_sample/outputs/<run_name>/
│
│   ─── 始终保存 ───
├── run_config.json                     运行时解析后的配置（含 prompt）
├── first_frame.png                     实际使用的裁剪后风格化首帧
├── original_video.mp4                  源视频按输出尺寸重新编码，裁剪到输出长度
├── generated_video.mp4                 ← 最终拼接输出
├── flip_test.mp4                       交替播放生成帧和源帧的 A/B 对比
│
│   ─── 仅 SAVE_VERBOSE=true (默认) ───
├── condition_0.mp4                     喂给模型的控制信号
├── generated_clip0.mp4  generated_clip1.mp4  ...   每个 clip 的单独输出
└── generated_video_withOrig.mp4        并排对比（生成 | 源）可视化
```

`<run_name>` 默认格式为 `r{W}x{H}_f{F}_kf{N}_idv2v`（如 `r1280x720_f81_kf0_idv2v`；`F` = 每个 clip 的帧数 = `NUM_FRAMES_PER_CLIP`，非总长度，所以 240 帧多 clip 运行仍显示 `f81`；`N` = 关键帧数）。设置 `SAVE_VERBOSE=false` 可仅保留最小输出集。`flip_test.mp4` 是重点可视化：播放生成视频，在最多 5 个均匀分布的帧位置暂停，交替展示 **Generated**（绿色标签）↔ **Source**（红色标签），方便快速 A/B 对比。

# 🧩 详细使用场景

ID-V2V 由**关键帧**驱动——你提供风格化的帧。下方五个菜谱覆盖了常见用法。每个菜谱在 [`scripts/examples/`](scripts/examples) 有现成的运行脚本，在 [`test_samples/`](test_samples) 有对应的样例（**同时作为输入模板**）：打开样例文件夹查看输入布局，然后把 `SAMPLE_DIR` 指向你自己按相同布局组织的目录，运行对应脚本即可。每个样例还附带 `generated.mp4` 展示预期结果。请先完成[环境搭建](#-环境搭建)。

## 风格迁移 (Restylization)

**是什么**：改变人物 + 重新生成整个场景——背景、物体、整体风格。

**适用场景**：最通用的用法——在不重拍的情况下重塑一段表演周围的世界。你提供一帧改变了场景和背景但保持人物姿态与细微表情的风格化首帧，ID-V2V 将这个新外观延续到整段视频，同时严格保留源视频的身份、表情、眼神和口型。

**运行**：

```bash
# SAMPLE_DIR 指向输入目录（换成你自己的，按样例布局）：
SAMPLE_DIR=test_samples/restylization/two_sitting_woman bash scripts/examples/restylization.sh
```

默认以 "person" 作为 SAM3 prompt 进行预处理（前景灰度条件），然后生成视频。样例位于 `test_samples/restylization/`。

## 不完美关键帧 (Imperfect keyframe)

**是什么**：与风格迁移相同流程，但首帧**未能完美匹配**源视频。

**适用场景**（以及为什么重要）：NanoBanana 等图像编辑模型在风格化一帧时往往无法保持精确的姿态或表情对齐，导致风格化首帧与源视频第一帧**不匹配**。这不要紧——你不需要像素级精确的关键帧。第 0 帧跟随你提供的不完美关键帧，但从**第二帧开始** ID-V2V 会自动重新对齐到源视频的身份与表演，修正偏差。

**运行**——与风格迁移相同的命令和脚本，只有样例不同：

```bash
SAMPLE_DIR=test_samples/non_aligned_keyframe/woman_phone bash scripts/examples/restylization.sh
```

样例位于 `test_samples/non_aligned_keyframe/`。

## 长视频 (Longer video)

**是什么**：源视频超过单个 81 帧 clip 时的生成。ID-V2V **按 clip 依次生成**，每个新 clip 以**前一个 clip 的末尾**（重叠位置）作为条件，clip 之间无缝拼接、漂移受控。

**运行**：

```bash
SAMPLE_DIR=test_samples/longer_video/woman_dancing bash scripts/examples/longer_video.sh
```

示例默认 `MAX_NUM_FRAMES=240`（→ 3 个 clip）；按需设任意长度。预处理和生成流程与风格迁移相同。样例位于 `test_samples/longer_video/`。对于较长的源视频，建议增加关键帧数量以获得最佳效果。

## 多关键帧 (Multiple keyframes)

**是什么**：不仅锁定第一帧，还锁定更多位置的帧。

**适用场景**：有时候一帧不够——比如你想同时固定**首帧和末帧**（或中间帧），让视频精确落到你期望的位置。把编辑好的帧放到 `keyframes/<N>.png`，模型会锁定每一帧。关键帧可以是任意帧位置。

**运行**：

```bash
SAMPLE_DIR=test_samples/first_last_frame/two_women_spotlight bash scripts/examples/first_last_frame.sh
```

样例锁定两端——`stylized_first_frame.png`（首帧）+ `keyframes/80.png`（81 帧 clip 的末帧），`infer.sh` 会自动发现。样例位于 `test_samples/first_last_frame/`。

## 重打光 (Relighting)

**是什么**：**只改变光照**，保持完整场景——前景人物 + 背景——完全一致。

**适用场景**（以及与风格迁移的区别）：重打光是风格迁移的受限特例。风格迁移可以编辑背景、整体场景、大幅改变视频外观，重打光**只改变光照**。做法是用图像编辑模型重打光首帧（**保持一切包括背景不变**），然后直接喂给 ID-V2V——**不需要 SAM3 mask、不需要背景置灰、完全不需要预处理**。原始源视频本身就是控制信号，整个场景被完整保留，只有光照变化。

**运行**：

```bash
SAMPLE_DIR=test_samples/relighting/two_sitting_woman bash scripts/examples/relighting.sh
```

跳过预处理，直接使用 `scripts/infer_relighting.sh`（原始源视频作为条件）。样例位于 `test_samples/relighting/`。

# 🦿 两种模型变体

以上所有用法都使用**默认 `idv2v`** 模型（单一 per-frame VACE 控制信号驱动）。另有一个可选的增强变体，额外添加两个条件：

| 变体 | VACE 条件数 | 预处理脚本 | 推理脚本 | Checkpoint |
|---|---|---|---|---|
| **`idv2v`** — *默认，推荐* | **1**: 前景灰度像素（SAM3 分割的人物） | `scripts/preprocess.sh` | `scripts/infer.sh` | `idv2v.pth` |
| `idv2v_with_normal_depth` — *可选* | **3**: 前景灰度像素 **+ 表面法线 (DAViD) + 深度 (DepthAnything-V2)** | `scripts/idv2v_with_normal_depth/preprocess_with_depth.sh` | `scripts/idv2v_with_normal_depth/infer_with_depth.sh` | `idv2v_with_normal_depth.pth` |

默认 `idv2v` 保留（重打光时保留）SAM3 mask 内的内容（被分割的人物），自由生成帧的其余部分。`idv2v_with_normal_depth` 额外用源视频深度约束那些区域——当你希望深度也受控时使用，但前提是源视频的深度与你想生成的场景一致。下载 checkpoint 时加 `--with-depth` 即可获取该变体的权重 + DAViD + DepthAnything-V2：

```bash
bash scripts/download_checkpoints.sh --with-depth   # + idv2v_with_normal_depth.pth + DAViD + DepthV2 (~75 GB)
```

预处理和推理通过 `scripts/idv2v_with_normal_depth/preprocess_with_depth.sh` 和 `scripts/idv2v_with_normal_depth/infer_with_depth.sh` 执行；输出 `condition_0/1/2.mp4`（像素 + 法线 + 深度），使用 `..._idv2v_with_normal_depth` 后缀。**两个 checkpoint 结构相同但权重不兼容**——用错 `.pth` 不会报错，但输出质量会很差。checkpoint 与脚本必须匹配。详见 [`checkpoints/README.md`](checkpoints/README.md)。

# 📁 仓库目录结构

```
ID-V2V_public/
├── scripts/
│   ├── download_checkpoints.sh   下载公开权重（默认: SAM3 + Wan; --with-depth 额外加 DAViD + DepthV2）
│   ├── preprocess.sh             idv2v: SAM3 + origPixel（前景灰度条件）
│   ├── infer.sh                  idv2v: 单条件视频生成
│   ├── infer_relighting.sh       idv2v 重打光：原始源视频作为条件（无需预处理）
│   ├── examples/                 每个场景一个可直接运行的脚本
│   └── idv2v_with_normal_depth/  备选 idv2v_with_normal_depth 变体（次要）:
│       ├── preprocess_with_depth.sh   SAM3 + origPixel + DAViD + DepthV2
│       └── infer_with_depth.sh        三条件视频生成
├── src/idv2v/
│   ├── preprocess/               SAM3, origPixel, DAViD, DepthV2
│   └── inference/                pipeline.py（1–3 条件通用引擎） + flip_test.py
├── diffsynth_studio/             Wan 系列 pipeline
├── test_samples/                 按使用场景分组的可运行样例（source.mp4 + stylized_first_frame.png + prompt.txt + generated.mp4）
├── checkpoints/                  gitignore — 由 download_checkpoints.sh 填充
```

# 🙏 致谢

本项目基于 [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)（Wan 系列）、[SAM3](https://huggingface.co/facebook/sam3)（分割）、[DAViD](https://github.com/microsoft/DAViD)（表面法线，仅 `idv2v_with_normal_depth`）、[DepthAnything v2](https://github.com/DepthAnything/Depth-Anything-V2)（深度，仅 `idv2v_with_normal_depth`）和 [Stable-Video-Infinity](https://github.com/vita-epfl/Stable-Video-Infinity)（长视频生成）构建。
