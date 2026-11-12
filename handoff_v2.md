# ID-V2V 部署 Handoff（2026-11-12 · 第二轮）

> 上一个对话 50 轮到顶，本文件是下一个对话的起步点。读这一份 + deploy_progress.md 即可。

---

## 0. 一句话状态

**231 单机 ID-V2V 已跑通两个样例**（two_sitting_woman 49 帧 ≈55min、spirit_girl 240 帧 ≈3.5h），文档完整。**下一件事**：ComfyUI 方案 A 执行（qwen 容器改做 ID-V2V）。

---

## 1. 服务器

| 服务器 | IP | 用途 |
|---|---|---|
| **231（目标）** | `zenking@192.168.8.231` | RTX 5090 32GB Blackwell sm_120，**单卡**，inference + preprocess 全在这里 |
| 201 | `zenking@192.168.100.201` | 现在不通（ping 255），历史上用来 rsync 权重和 preprocess；当前已**不再参与** |

---

## 2. 231 双 env 分工（实测）

| | `idv2v` env | `hyavatar` env |
|---|---|---|
| Python | 3.10.12 | 3.11.15 |
| torch | 2.7.0+cu128 ✅ sm_120 | 2.7.0+cu128 ✅ sm_120 |
| 职责 | **preprocess**（sam2 + orig_pixel） | **inference**（DiT + VACE） |
| 为什么这样 | 原生有 sam2；但 flash-attn 编译旧，inference import diffsynth 会 `undefined symbol` | diffsynth Blackwell 就绪；**没 sam2（不装，preprocess 不走这里）** |
| 激活 | `conda activate idv2v` | `conda activate hyavatar` |
| 命令示例 | `python -m idv2v.preprocess.sam2_preprocess ...` | `bash scripts/infer.sh` |

---

## 3. 完整资产清单（≈ 89.3GB 实际被加载）

| # | 资产 | 大小 | 物理路径 |
|---|---|---|---|
| 1 | UMT5-XXL tokenizer | 21M | `/nfs-data/InfiniteTalk/weights/Wan2.1-I2V-14B-480P/google/umt5-xxl/` |
| 2 | T5 Encoder | 11G | 同上 `models_t5_umt5-xxl-enc-bf16.pth` |
| 3 | CLIP Encoder | 4.5G | 同上 `models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth` |
| 4 | CLIP tokenizer | 22M | 同上 `xlm-roberta-large/` |
| 5 | VAE | 485M | 同上 `Wan2.1_VAE.pth` |
| 6 | **ID-V2V DiT + VACE** | **73G** | `/nfs-data/aigc/ID-V2V/checkpoints/idv2v.pth`（1539 keys: DiT 1303 + VACE 236, `img_emb.proj.0.bias` **存在** → ComfyUI PR 15139 能自动识别） |
| 7 | SAM2.1 | 309M | `/data/aigc/ID-V2V/checkpoints/VACE-Annotators/sam2/sam2.1_hiera_base_plus.pt` |

**Symlink 结构**：
```
/data/aigc/ID-V2V/checkpoints/
├── idv2v.pth → /nfs-data/aigc/ID-V2V/checkpoints/idv2v.pth
├── VACE-Annotators/sam2/sam2.1_hiera_base_plus.pt        (本地)
└── wan/Wan-AI/
    ├── Wan2.1-I2V-14B-480P → /nfs-data/InfiniteTalk/weights/Wan2.1-I2V-14B-480P/
    ├── Wan2.1-I2V-14B-720P → 同上（同目录！）
    └── Wan2.1-T2V-14B      → 同上（同目录！）
```
**三个名字不同的 symlink 指向同一 NFS 目录** —— 让 pipeline.py 的 `model_id_with_origin_paths` 字符串匹配能找到文件；T2V-14B 和 I2V-14B-480P 的 tokenizer/T5/VAE/CLIP 完全一样，只有 DiT 不同，而 DiT 被 idv2v.pth 覆盖所以无所谓。

**不被加载但占空间**：base Wan DiT safetensors 7 分片 ≈61G（InfiniteTalk 带过来的，不动）。

**README / infer.sh 里提但不存在的**：`Wan2.1-VACE-14B/diffusion_pytorch_model*.safetensors`（VACE 权重打在 idv2v.pth 里，那个 glob 永远不展开）。

---

## 4. 已验证的代码改动

| 文件 | 改动 | 原因 |
|---|---|---|
| `src/idv2v/inference/pipeline.py` | 加 `import mmap as _mmap`；`stack_videos_left_right_and_save` 比例自适应 + 偶数对齐；`_load_finetuned_dit_vace` 加 `MAP_SHARED` | 对比视频崩溃 + mmap 大文件 OOM |
| `scripts/infer.sh` / `infer_relighting.sh` / `preprocess.sh` | `STAGE_CHECKPOINT_TO_SHM=true` → `${STAGE_CHECKPOINT_TO_SHM:-true}`；`conda activate idv2v` → `conda activate hyavatar` | 环境可覆盖；脚本默认值硬编码 |
| `scripts/download_checkpoints.sh` | `conda activate idv2v` → `conda activate hyavatar` | 同上 |
| `diffsynth_studio/diffsynth/models/downloader.py` | `from modelscope import snapshot_download` → try-except 跳过 | modelscope 依赖链无底洞，推理不走它 |

---

## 5. 文档索引

| 文件 | 用途 |
|---|---|
| [deploy_progress.md](file:///home/zj/aigc/ID-V2V/deploy_progress.md) | 部署报告，**最完整的事实源**。§九 env 分工、§十 资产清单 |
| [new_video_inference_guide.md](file:///home/zj/aigc/ID-V2V/new_video_inference_guide.md) | 新视频推理 5 步操作手册（用户跑新视频时看这个） |
| [README.zh.md](file:///home/zj/aigc/ID-V2V/README.zh.md) + [README.md](file:///home/zj/aigc/ID-V2V/README.md) | 各加了"⚠️ RTX 5090 部署差异"小节 + SAM3 注脚 |

---

## 6. ComfyUI 方案 A（明日主任务）

### 为什么选 A

| 选项 | 做法 | 风险 |
|---|---|---|
| A（推荐） | 停着的 qwen 容器改做 ID-V2V | **零风险**（h3 不动、qwen 本来就停着） |
| B | 直接升级 h3 ComfyUI 源码 | 有风险（CustomNodes 版本绑定，h3 在跑 H3 任务） |

### qwen 容器现状

- compose: `/data/aigc/comfyui-qwen-image21/compose.yaml`
- ComfyUI 源码路径: `/data/aigc/comfyui-qwen-image21/ComfyUI/` → **不是 git repo**（fatal: not a git repository），是从 tar 包或 docker cp 来的
- 镜像: `yanwk/comfyui-boot:cu130-slim-v2`（Python 3.13 + cu130，cu128 也 OK）
- 端口: 8190（现在 qwen 停着，端口空）
- compose 里: `--disable-all-custom-nodes --whitelist ComfyUI-TD-Qwen3TTS`、源码和 models 都 `:ro`、`shm_size: 2gb`、`PYTHONPATH` 绑 TTS 包
- **正在跑的 h3 容器**: 端口 8188，CustomNodes 全是 H3 相关（MiniMax-H3、krea2edit 等），**完全没有 PR 15139 的任何改动**

### 6 步执行清单

1. **重置 ComfyUI 源码**：`mv ComfyUI ComfyUI.qwen.bak && git clone https://github.com/Comfy-Org/ComfyUI.git ComfyUI`（主线已合入 PR 15139，天然带 `context_vace` 隔离、`vace_image_input=True` 自动检测、`ref_pad_image` 节点输入）
2. **改 compose.yaml 6 处**：容器名改 `comfyui-idv2v`、**去掉** `--disable-all-custom-nodes --whitelist` 那行、**去掉** `PYTHONPATH=/tts-packages:/runtime-packages:runtime-base/...`、`shm_size: 2gb` → `8gb`、所有 volume 的 `:ro` 去掉、保留端口 8190
3. **模型 symlink**（**零下载，全复用 231 NFS**）：
   ```
   models/unet/idv2v.pth → /nfs-data/aigc/ID-V2V/checkpoints/idv2v.pth
   models/text_encoders/models_t5_umt5-xxl-enc-bf16.pth → /nfs-data/InfiniteTalk/weights/Wan2.1-I2V-14B-480P/models_t5_umt5-xxl-enc-bf16.pth
   models/text_encoders/umt5-xxl/ → /nfs-data/InfiniteTalk/weights/Wan2.1-I2V-14B-480P/google/umt5-xxl/
   models/clip/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth → /nfs-data/InfiniteTalk/weights/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
   models/clip/xlm-roberta-large/ → /nfs-data/InfiniteTalk/weights/Wan2.1-I2V-14B-480P/xlm-roberta-large/
   models/vae/Wan2.1_VAE.pth → /nfs-data/InfiniteTalk/weights/Wan2.1-I2V-14B-480P/Wan2.1_VAE.pth
   ```
   **不需要**：base Wan DiT safetensors（PR 15139 让 ComfyUI 直接从 idv2v.pth 读 DiT + VACE）、独立 VACE-14B 目录、SAM2（继续用 idv2v env 跑 preprocess 抽帧喂 ComfyUI）
4. **启动 + 冒烟**：`docker compose up -d`、`docker compose logs -f` 看启动日志（必须出现 `vace_image_input=True`）、浏览器 `http://192.168.8.231:8190` 能看到 WanVideoModelSampling 节点有 `ref_pad_image` optional input
5. **写工作流 json**：`LoadImage (stylized_first_frame) + LoadImage (orig_pixel.mp4 第 0 帧) → WanVideoModelSampling (ref_pad_image=...) → CLIPTextEncode → KSampler → VAEDecode → SaveVideo`，存 `workflows/idv2v_restylization_api.json`
6. **短测对齐**：spirit_girl 现成产物（preprocess 已经跑完），先 steps=3 length=49（~10 min）冒烟 → 无漂移 → steps=30 length=49（≈55 min）跟 CLI two_sitting_woman 对齐

### PR 15139 关键改动（给 ComfyUI 识别 ID-V2V 必须的 3 件事）

| 改动 | 文件 | 作用 |
|---|---|---|
| VACE block 只看 text token | `comfy/ldm/wan/model.py` | `context_vace = context[:, context_img_len:]`，VACE blocks 不能对 CLIP image token 做 attention，否则输出漂移 |
| `img_emb.proj.0.bias` 检测自动启 I2V | `model_detection.py` + `supported_models.py` | `vace_image_input=True` → `image_to_video=True` |
| 节点加 `ref_pad_image` 输入 | `comfy_extras/nodes_wan.py` | 把前景灰度条件（orig_pixel.mp4）填进 DiT pad frames 位置 |

---

## 7. 明日其他待办

- [ ] **201 权重清理**（等 201 网络恢复）：删 `/data/aigc/ID-V2V/checkpoints/` 下重复的 73G + 77G，两边合计释放 ~150G
- [ ] **vm.max_map_count 持久化**：`echo "vm.max_map_count=16777216" | sudo tee -a /etc/sysctl.conf`（当前是手动 `sysctl -w`，重启会丢）
- [ ] **可选**：231 h3 容器要不要也升级到带 PR 15139（等 H3 任务空闲后再评估）

---

## 8. 关键命令速查

```bash
# SSH
ssh zenking@192.168.8.231

# preprocess（idv2v env）
source ~/miniconda3/etc/profile.d/conda.sh && conda activate idv2v
cd /data/aigc/ID-V2V
python -m idv2v.preprocess.sam2_preprocess --video_path S/source.mp4 --output_dir S/preprocessing --model_path checkpoints/VACE-Annotators/sam2/sam2.1_hiera_base_plus.pt --config_path configs/sam2.1/sam2.1_hiera_b+.yaml --min_area_ratio 0.005 --max_area_ratio 0.95 --top_n 5
python -m idv2v.preprocess.orig_pixel --video_path S/source.mp4 --mask_folder S/preprocessing --mask_image_file_name sam3Mask_id_all.png --result_save_path S/preprocessing/orig_pixel.mp4

# inference（hyavatar env）
conda activate hyavatar
pkill -9 -f inference/pipeline.py  # 启动前清残留
MAX_NUM_FRAMES=240 GPU=0 STAGE_CHECKPOINT_TO_SHM=false bash scripts/infer.sh

# 查看进度
tail -f /tmp/spirit_log.log | tr "\r" "\n"
tail -20 /tmp/spirit_log.log | tr "\r" "\n" | tail -18
grep -nE "Error|Traceback|ValueError" /tmp/spirit_log.log
nvidia-smi

# ComfyUI
docker compose -f /data/aigc/comfyui-qwen-image21/compose.yaml up -d
docker compose -f /data/aigc/comfyui-qwen-image21/compose.yaml logs -f
```

---

*创建于 2026-11-12 第二轮对话结束，为 50 轮上限后的 handoff*
