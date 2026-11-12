# ID-V2V 部署进展与事实核查

核查日期：2026-11-12（Asia/Shanghai）。本次只读检查了本地代码、231 / 201 的环境与文件、231 推理日志及视频元数据；没有重新跑推理、修改远端环境或清理权重。

## 1. 当前结论

- 当前主要验证机器为 `zenking@192.168.8.231:/data/aigc/ID-V2V`。这是单机实验部署，现有证据不足以称为“生产部署完成”。
- 231 的两个样例均有可由 ffprobe 读取的 **49 帧主视频**。`spirit_girl` 历史日志记录过 240 帧拼接，但该次随后在并排对比阶段报错；当前目录不能交付为 240 帧成果。
- 231 可以承担 SAM2 预处理与推理，但使用两个环境；预处理存在 overlay 写出异常，不能称为无异常的一键全流程。
- 201 当前可连接，权重已下载完成。它不必参与当前 231 推理流程，但并非“不通”或“已清理”。
- 本地与 231 部署脚本存在差异，直接同步本地脚本可能覆盖服务器修复。当前操作入口见 [handoff.md](handoff.md)。[new_video_inference_guide.md](new_video_inference_guide.md) 尚有旧版本号及完成状态描述，使用时以本次核查为准。

## 2. 231 硬件与环境快照

| 项目 | 本次实测 |
|---|---|
| 主机名 | `zenking` |
| GPU | 1 × NVIDIA GeForce RTX 5090，32607 MiB |
| NVIDIA 驱动 | 595.99.02 |
| 内存 / swap | 251 GiB / 0 B |
| 磁盘可用空间 | `/` 与 `/data` 同一文件系统，约 63 GiB；`/nfs-data` 约 1.7 TiB（动态值） |
| `vm.max_map_count` | 16777216（当前值，不代表程序必须达到此值） |
| 项目 `.venv` | 不存在；231 脚本会走 conda fallback |

| 包 / 能力 | `idv2v`（预处理） | `hyavatar`（推理） |
|---|---|---|
| Python | **3.10.12** | **3.11.6** |
| torch | 2.7.0+cu128 | 2.7.0+cu128 |
| torch CUDA runtime | 12.8 | 12.8 |
| sm_120 | 编译架构列表包含；CUDA 零张量验证成功 | 编译架构列表包含；CUDA 零张量验证成功 |
| transformers / tokenizers | 5.20.2 / 0.23.3 | 5.20.2 / 0.23.3 |
| diffusers | 0.41.0 | 0.41.0 |
| numpy / pandas | 2.2.6 / 2.3.3 | 2.2.6 / 2.2.3 |
| huggingface-hub | 1.33.0 | 1.33.0 |
| accelerate | 1.15.0 | **1.1.1** |
| safetensors | 0.8.0 | 0.8.0 |
| omegaconf | 2.3.1 | 2.4.0 |
| av | 17.1.0 | 18.1.0 |
| sentencepiece / ftfy | 0.2.3 / 6.3.1 | 0.2.3 / 6.3.1 |
| wcwidth | 0.8.2 | 0.9.4 |
| sam2 | 已安装 | 未安装 |
| diffsynth | 本次 import 失败：flash_attn_2_cuda undefined symbol | 既有推理日志与产物支持已运行 |

原文 `hyavatar Python 3.11.15`、`accelerate 1.15.0` 不符合当前环境。早期 `idv2v torch 2.6.0+cu118` 不能用于描述当前 231；201 的 idv2v 仍为 2.6.0+cu118。

`hyavatar` 的 site-packages 中存在指向 `/data/aigc/ID-V2V/src` 和 `/data/aigc/ID-V2V/diffsynth_studio` 的 `.pth` 文件。这只解决源码搜索路径，不等于完成了依赖安装与一致性校验。本地 `pyproject.toml` 要求 `==3.10.*`，并锁定 torch 2.6 与 cp310 flash-attn wheel；不是锁定“3.10.12”。不要在 231 上直接 `uv sync` 或按此配置重建 Blackwell 推理环境。

## 3. 样例：当前文件与历史日志分开记录

当前输出目录均为 `<样例>/outputs/r1280x720_f81_kf0_idv2v/`。`f81` 表示每 clip 的模型帧数，不表示最终视频帧数。

| 文件 / 配置 | `two_sitting_woman` | `spirit_girl` |
|---|---|---|
| `source.mp4` | **1920×1080，81 帧，25 fps，3.24s** | 768×1344，345 帧，24 fps，14.375s |
| `preprocessing/orig_pixel.mp4` | 1920×1080，81 帧，25 fps | 768×1344，345 帧，24 fps |
| `run_config.json` | max_num_frames=49，num_frames_per_clip=81，30 steps | max_num_frames=49，num_frames_per_clip=81，30 steps |
| `generated_video.mp4` | 1280×720，**49 帧，1.96s** | 1280×720，**49 帧，2.041667s** |
| `generated_clip0.mp4` | 1280×720，81 帧 | 1280×720，81 帧 |
| `generated_clip1/2.mp4` | 当前未见 | **当前未见** |
| `generated_video_withOrig.mp4` | 2560×720，49 帧 | 1690×720，49 帧 |
| `condition_0.mp4` / `original_video.mp4` | 均为 1280×720，49 帧 | 均为 1280×720，49 帧 |
| `flip_test.mp4` | 277 帧，11.08s | 265 帧，11.041667s |

`flip_test.mp4` 是可视化产物，时长不能当作生成主视频时长。这里验证了元数据可读性，未做逐帧完整解码或身份/画质验收。

### 3.1 two_sitting_woman 历史运行

231 `/tmp/idv2v_infer.log` 记录 30 steps 扩散耗时 **50分38秒**，模型生成 81 帧后裁成 49 帧，最终包含 `Inference done.` / `Done. Output:`。原报告约 55 分钟总耗时可作为历史估计，不能解释为模型只计算了 49 帧。

### 3.2 spirit_girl 历史 240 帧运行

231 `/tmp/spirit_log.log` 记录：

| Clip（0 基索引区间，右开） | 模型生成帧数 | 30 steps 耗时 |
|---|---|---|
| `[0, 81)` | 81 | 58分43秒 |
| `[80, 161)` | 81 | 1小时01分37秒 |
| `[159, 240)` | 81 | 1小时09分53秒 |

三个 clip 均生成 81 帧；相邻重叠 1 / 2 帧，拼接后 240 帧。原报告“81 + 81 + 78 帧”、前两个 clip 各约 50 分钟不准确。扩散累计 **3小时10分13秒**；约 3.5 小时总耗时仍只是历史估计。该日志 checkpoint mmap 调用耗时 33.1 秒，不代表全部权重已驻留内存。

日志随后因源视频和输出横竖比例不同，在 `stack_videos_left_right_and_save` 抛出 `ValueError`。因此它证明完成过 240 帧生成/拼接，不证明该次全流程无异常结束。当前 `spirit_girl` 的配置与主视频均为 49 帧，文件时间为 11月12日 10:45–11:45；输出目录内容已不同于历史 240 帧运行，240 帧文件的现存位置未确认。

## 4. 权重与加载链路

以下大小以 `stat -L` 实测字节数为准；GiB = 2^30 字节，避免混用 GB / GiB。

| 资产 | 字节数 | 约 GiB | 231 路径 |
|---|---:|---:|---|
| ID-V2V finetuned DiT + VACE | 77,776,433,523 | 72.43 | `/nfs-data/aigc/ID-V2V/checkpoints/idv2v.pth` |
| T5 encoder | 11,361,920,418 | 10.58 | 下述 Wan 目录 `models_t5_umt5-xxl-enc-bf16.pth` |
| CLIP encoder | 4,772,359,047 | 4.44 | 下述 Wan 目录 `models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth` |
| VAE | 507,609,880 | 0.47 | 下述 Wan 目录 `Wan2.1_VAE.pth` |
| SAM2.1 base-plus | 323,606,802 | 0.30 | `/data/aigc/ID-V2V/checkpoints/VACE-Annotators/sam2/sam2.1_hiera_base_plus.pt` |

以上五个权重合计约 **88.24 GiB**，另需 tokenizer 文件；不是完整环境或整个 NFS 目录的占用。本次确认路径及文件大小，没有重新计算大权重哈希。

```text
/data/aigc/ID-V2V/checkpoints/
├── idv2v.pth -> /nfs-data/aigc/ID-V2V/checkpoints/idv2v.pth
├── VACE-Annotators/sam2/sam2.1_hiera_base_plus.pt   # 本地文件
└── wan/Wan-AI/
    ├── Wan2.1-T2V-14B ---------┐
    ├── Wan2.1-I2V-14B-480P ---┼-> /nfs-data/InfiniteTalk/weights/Wan2.1-I2V-14B-480P
    └── Wan2.1-I2V-14B-720P ---┘
```

`pipeline.py` 的 finetuned 路径跳过 `diffusion_pytorch_model*` base DiT，另加载 T5、VAE、CLIP 与 tokenizer，再从 idv2v.pth 加载 DiT / VACE。因此这些别名对当前 finetuned 配置有运行证据；**不意味着 480P / 720P / T2V 的完整权重等价**。不能据此在去掉 finetuned checkpoint 后继续混用。

`VACE_GLOB` 只在无 finetuned checkpoint 的 fallback 路径、且 `pipe.vace is None` 时使用，不是所有模式“永远不读取”。InfiniteTalk 的 base DiT 分片虽不用于当前 ID-V2V 路径，仍可能被其他项目使用。

原文 ComfyUI 迁移表和 PR 15139 的结论未在本次验证。仅凭该原生 pipeline 跑通，不能证明 ComfyUI 可直接读取原始 idv2v.pth；节点支持、格式转换、检测逻辑和工作流仍需单独验收。

## 5. 代码状态与操作限制

| 文件 / 配置 | 本地 | 231 |
|---|---|---|
| `pipeline.py` | MAP_SHARED、libx264、横竖屏并排缩放/偶数尺寸 | 与本地 SHA-256 相同 |
| `sam2_preprocess.py` | SAM2 自动分割，仍有 overlay writer 问题 | 与本地 SHA-256 相同 |
| `scripts/infer.sh` | 强制 source `.venv`；SHM=true 硬编码 | `.venv` 不存在时激活 hyavatar；SHM 可用环境变量覆盖 |
| `scripts/preprocess.sh` | 强制 source `.venv`；SAM2 路径 | fallback 也激活 hyavatar，**而该环境没有 sam2** |
| `scripts/infer_relighting.sh` | 不可套用 infer.sh 已修复的结论 | SHM 仍硬编码 true |
| `scripts/download_checkpoints.sh` | 下载 SAM3 等资产，与当前 SAM2 路径不完全一致 | 有 hyavatar fallback；不是 SAM2 端到端安装器 |
| `downloader.py` | ModelScope import 有 ImportError fallback | 同类处理，文本格式不同 |

231 `infer.sh` 与本地均固定 `NUM_INFERENCE_STEPS=30`、`NUM_FRAMES_PER_CLIP=81`、`WIDTH=1280`、`HEIGHT=720`、`DEBUG=false`；不能假设这些变量支持环境覆盖。`MAX_NUM_FRAMES` 支持覆盖，默认 **240**，不是 81。固定 RUN_NAME 不包含 max_num_frames；同一样例反复运行会复用目录，覆盖产物或遗留旧文件。

### 已确认问题

- **SAM2 overlay 异常**：`/tmp/sam2_out.log` 记录 345 帧分割约 5分43秒、逐帧保存约 13分06秒，随后 `write_overlay_video` 报 `AttributeError: 'NoneType' object has no attribute 'codec_context'`。当前代码仍用 `iio.imopen(..., plugin="pyav")` 后直接 `write_frame`，没有显式初始化视频流。已有 orig_pixel.mp4 不能证明 `preprocess.sh` 一键成功；该脚本 `set -e` 会在 SAM2 返回非零时停止。
- **两个环境分工**：idv2v 可导入 sam2，但本次 `import diffsynth` 因旧 flash-attn 二进制符号错误失败；hyavatar 无 sam2。直接运行远端 preprocess.sh 会选错环境，应按交付单显式调用模块。
- **横竖屏对比**：当前 pipeline 已按统一高度和偶数宽度缩放。现存 1690×720 对比文件支持修复生效；历史日志仍保留旧代码报错。
- **画面裁切**：当前脚本输出 1280×720，中心裁剪会丢失竖屏上下内容。此处描述的是当前脚本设置，不能推断模型只支持横屏；保留全幅竖屏的处理方案尚未验收。

### 不能当成已证实根因的旧解释

- MAP_PRIVATE 本身不是立即复制整份权重；私有文件映射也可共享未修改的文件缓存页。改为 MAP_SHARED 的代码存在，但“73GB 必须约一千万 mmap 条目”“必须 max_map_count≥16777216”没有诊断证据。当前内核设置只是现场快照。
- mmap 加载返回耗时不等于完整模型加载耗时。此前 6.2 秒只适合作为某次历史日志值；本次 240 帧日志为 33.1 秒。
- 未在入口代码发现显式 `torch.compile`，不能把前几步较慢断言为 compile 已完成。显存采样也不能证明 DiT + VACE 全量常驻 GPU；单卡实际调用 `enable_vram_management(vram_buffer=10)`。
- 日志中的 `Killed` 不能单凭文字归因于 SSH 断连；没有内核 OOM / 信号来源证据。nohup 有助于脱离终端，但不保证免于 OOM 或会话清理。

不把 `swapoff`、全局清缓存、修改 sysctl、全局 pkill 或删除共享模型作为日常启动前置步骤。本次没有执行这些操作。

## 6. 201 历史部署的本次复核

- 201 当前主机名为 `a-X99`，2 × RTX 3090（每卡 24576 MiB），驱动 580.178.04。
- `/data/aigc/ID-V2V/.deploy_info.yml` 仍记录 `deployed_host: zj-jg`、commit `b9df1e501b2561c445923cbed13b5ff8770c0e7a`、部署时间 `2026-11-09T07:50:27Z`（北京时间 15:50:27）。这是历史部署元数据，不是当前 hostname。
- idv2v 环境为 Python **3.10.12**、torch **2.6.0+cu118**；`.venv/bin/python` 指向该 conda Python。
- `checkpoints/idv2v.pth` 已存在，大小 **77,776,433,523 字节**。旧“下载 2.3GB、等待完成”的 TODO 已过时。与 231 大小相同不等于已做哈希一致性校验。
- `environment.lock.txt` 确有 90 行，但包含大量其他包及 `file:///...` 构建路径，不能作为 ID-V2V 完整、可重建的已验证依赖锁。
- 没有本次在 201 重跑模型的证据；不再保留“24GB 必定可跑、20–40 分钟”等未经实测的承诺。
- 未执行原报告的 201 权重删除命令；共享模型、备份与其他使用方未核对前，不以“有 NFS 副本”为由删除。

## 7. 验收依据与剩余事项

本地代码证据：`scripts/infer.sh`、`scripts/preprocess.sh`、`scripts/infer_relighting.sh`、`scripts/download_checkpoints.sh`、`pyproject.toml`、`src/idv2v/inference/pipeline.py`、`src/idv2v/preprocess/sam2_preprocess.py`。远端证据：本节前述日志、各样例 run_config.json、ffprobe 元数据、stat/readlink、两环境 sys.version / 包元数据 / CUDA 零张量验证。

- [x] 确认当前环境、资产路径和两个 49 帧主视频元数据。
- [x] 区分历史 240 帧生成与当前 49 帧交付物。
- [ ] 修复 SAM2 overlay writer，重新验证预处理脚本正常退出。
- [ ] 统一本地/231 脚本的环境选择与参数覆盖行为，再做隔离目录的复现验证。
- [ ] 如需 240 帧交付，定位历史文件或重新运行并保存独立配置、日志与产物。
- [ ] 完成输出视频完整解码及身份、表情、口型、画质的人工验收。
- [ ] ComfyUI 迁移另立验证，不作为当前部署已完成能力。

本次仅修订本文件与 handoff.md；未修改代码、远端部署或其他指南。
