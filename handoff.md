# ID-V2V 当前交付与运行入口

核查日期：2026-11-12（Asia/Shanghai）。本文件替代 2026-11-09 的初始安装/下载交接说明；详细核查依据与历史记录见 [deploy_progress.md](deploy_progress.md)。

## 1. 交付状态

主要验证目录：`zenking@192.168.8.231:/data/aigc/ID-V2V`。231 是当前实验验证机器，不写为生产上线。

| 样例 | 当前主视频 | 当前配置 |
|---|---|---|
| `restylization/two_sitting_woman` | 1280×720，25 fps，49 帧，1.96s | max_num_frames=49，clip=81，30 steps |
| `restylization/spirit_girl` | 1280×720，24 fps，49 帧，2.041667s | max_num_frames=49，clip=81，30 steps |

两个主视频均位于 `<SAMPLE_DIR>/outputs/r1280x720_f81_kf0_idv2v/generated_video.mp4`，已核对 ffprobe 元数据，未完成画质人工验收。`spirit_girl` 曾在历史日志中生成/拼接 240 帧，但当次并排对比报错，当前目录已经是 49 帧配置与产物，不能按 240 帧交付。

## 2. 环境与资产

| 阶段 | 231 环境 | 已知限制 |
|---|---|---|
| SAM2 分割 / orig_pixel | `idv2v`：Python 3.10.12、torch 2.7.0+cu128 | sam2 已装；diffsynth 因 flash-attn ABI 符号错误无法导入；SAM2 overlay 写出会报错 |
| DiT / VACE 推理 | `hyavatar`：Python **3.11.6**、torch 2.7.0+cu128 | 无 sam2；已有推理运行证据 |

两个环境均实测能执行 CUDA 零张量，torch 架构列表含 sm_120。无需为了当前交接强行合并环境。

权重通过项目内 `checkpoints/` 访问：idv2v.pth 指向 `/nfs-data/aigc/ID-V2V/checkpoints/idv2v.pth`（72.43 GiB）；三个 Wan 路径别名指向 `/nfs-data/InfiniteTalk/weights/Wan2.1-I2V-14B-480P`；SAM2 base-plus 为项目内本地文件。finetuned 路径跳过 base DiT/VACE，仍需要 T5、VAE、CLIP 和 tokenizer。别名不适用于任意模型配置。

本地 pyproject 的 torch 2.6 / cu118 / cp310 flash-attn 组合不适用于重建当前 5090 推理环境。本文件不提供未经重建验证的全套安装命令。如需补装普通 PyPI 包，默认使用清华镜像：

```bash
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
```

## 3. 使用已有预处理结果运行推理（231 部署版本）

以下命令针对 **231 现存 scripts/infer.sh**：它支持 `STAGE_CHECKPOINT_TO_SHM=false`。本地同名脚本仍硬编码 true 且强制加载 `.venv`，不能直接套用；覆盖远端脚本前必须先处理差异。

```bash
ssh zenking@192.168.8.231
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hyavatar
cd /data/aigc/ID-V2V
python -c 'import sys, torch; print(sys.executable, torch.__version__)'

# 确认有没有正在运行的任务；若有，先确认归属，不要直接全局 pkill
pgrep -af '[i]nference/pipeline.py' || true

# 用新样例目录隔离本次产物，避免覆盖原有已验收输出
BASE=test_samples/restylization/two_sitting_woman
SAMPLE_DIR="${BASE}_verify_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$SAMPLE_DIR"
cp "$BASE/source.mp4" "$BASE/stylized_first_frame.png" "$BASE/prompt.txt" "$SAMPLE_DIR/"
# 复用相同 source.mp4 对应的预处理目录；该目录只供推理读取
ln -s "$(realpath "$BASE/preprocessing")" "$SAMPLE_DIR/preprocessing"
# 本例不复制可选 keyframes，现有 two_sitting_woman 配置也无 keyframe

LOG="/tmp/idv2v_$(basename "$SAMPLE_DIR").log"
SAMPLE_DIR="$SAMPLE_DIR" GPU=0 STAGE_CHECKPOINT_TO_SHM=false MAX_NUM_FRAMES=49 \
  nohup bash scripts/infer.sh > "$LOG" 2>&1 &
TASK_PID=$!
printf 'PID=%s\nLOG=%s\nSAMPLE_DIR=%s\n' "$TASK_PID" "$LOG" "$SAMPLE_DIR"
```

当前 infer.sh 固定 1280×720、每 clip 81 帧、30 steps，`NUM_INFERENCE_STEPS=2` 或 `DEBUG=true` 环境变量不会覆盖脚本赋值。`MAX_NUM_FRAMES=49` 仅限制最终总帧数，模型仍计算一个 81 帧 clip；不能据此预期减少到 49/81 的推理耗时。既有 two_sitting_woman 日志扩散阶段为 50分38秒。

要验证 spirit_girl，可将 BASE 改为其样例目录。要验证 240 帧，将 MAX_NUM_FRAMES 改为 240，并保留独立输出目录；历史三 clip 扩散累计约 3小时10分钟，整轮耗时受模型加载和资源竞争影响。

## 4. 新视频预处理（已知需人工确认的环节）

准备 `<SAMPLE_DIR>/source.mp4`、`stylized_first_frame.png`、`prompt.txt`。首帧应与源片内容和裁剪构图匹配。当前推理会将输入中心裁剪至 1280×720，竖屏可能丢失大部分上下画面。

231 的 preprocess.sh 在无 `.venv` 时也会激活 hyavatar，选错环境。因此目前显式使用 idv2v 环境调用模块：

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate idv2v
cd /data/aigc/ID-V2V
export CUDA_VISIBLE_DEVICES=0
S='test_samples/restylization/<新样例>'  # 替换为实际目录

python -m idv2v.preprocess.sam2_preprocess \
  --video_path "$S/source.mp4" \
  --output_dir "$S/preprocessing" \
  --model_path checkpoints/VACE-Annotators/sam2/sam2.1_hiera_base_plus.pt \
  --config_path configs/sam2.1/sam2.1_hiera_b+.yaml
```

当前该模块在分割、保存掩码后，可能在调试 overlay 视频写出时抛 `codec_context` 异常。**这不等于整步成功**；不要忽略任意错误后继续。只有确认报错局限于 overlay，且每一帧 `frame_N/sam3Mask_id_all.png` 都存在、尺寸正确并可读取，才可单独执行下步。若掩码缺失/不可读，先处理分割错误。本次未修复该代码。

```bash
# 在上述掩码检查通过后执行
python -m idv2v.preprocess.orig_pixel \
  --video_path "$S/source.mp4" \
  --mask_folder "$S/preprocessing" \
  --mask_image_file_name sam3Mask_id_all.png \
  --result_save_path "$S/preprocessing/orig_pixel.mp4"

# 确认 orig_pixel 与 source 帧数、fps、尺寸匹配，并查看实际前景质量
for f in "$S/source.mp4" "$S/preprocessing/orig_pixel.mp4"; do
  ffprobe -v error -select_streams v:0 \
    -show_entries stream=width,height,nb_frames,r_frame_rate,duration \
    -of json "$f"
done
```

SAM2 自动分割按面积和 top_n 筛选对象，不是 SAM3 的文本 person 分割；`SAM_PROMPT=person` 不控制这个 SAM2 模块。沿用 `sam3Mask_id_all.png` 文件名仅为兼容 orig_pixel 的读取约定。

## 5. 查看进度与验收

```bash
# LOG / SAMPLE_DIR 为启动时打印的值
tr '\r' '\n' < "$LOG" | tail -20
# 实时查看，Ctrl+C 退出查看
# tail -f "$LOG"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader

OUT="$SAMPLE_DIR/outputs/r1280x720_f81_kf0_idv2v"
cat "$OUT/run_config.json"
ffprobe -v error -select_streams v:0 \
  -show_entries stream=width,height,nb_frames,r_frame_rate,duration \
  -of json "$OUT/generated_video.mp4"
# 完整解码检查；随后还需人工看画质
ffmpeg -v error -i "$OUT/generated_video.mp4" -f null -
```

验收应同时检查进程退出状态、日志有无 traceback、配置与主视频是否一致、视频能否完整解码、预期对比产物是否存在。`nohup` 启动成功或出现 `INFER_DONE` 自定义标记均不足以证明推理成功；`flip_test` 时长也不能替代主视频帧数。

不需要默认清空系统 page cache、关闭 swap、修改 max_map_count 或杀掉全部 pipeline 进程。遇到 `Killed` 应先查信号/OOM 证据；遇到显存 OOM，应结合实际 VRAM/offload 诊断。减少扩散 steps 主要减少时间，不保证降低显存峰值。

## 6. 201 历史安装记录与后续事项

201 `zenking@192.168.100.201:/data/aigc/ID-V2V` 当前可连接，主机名 `a-X99`，2 × RTX 3090。历史 `.deploy_info.yml` 记录 2026-11-09 15:50:27（北京时间）、主机名 zj-jg、commit `b9df1e501b2561c445923cbed13b5ff8770c0e7a`。部署元数据不保证后续源码没有变化。

201 当前 idv2v 为 Python 3.10.12、torch 2.6.0+cu118，`.venv/bin/python` 指向该 conda 环境；idv2v.pth 已有 77,776,433,523 字节。原交接单的 Python 3.10.22、下载 2.3GB 的进度和未完成下载 TODO 已移除。201 的 90 行 environment.lock.txt 包含其他包和本地构建路径，不作为完整重建锁。没有本次 201 单卡推理验收，不能承诺 24GB 可跑或 20–40 分钟完成。

下一步：修复并验收 SAM2 overlay；对齐本地/231 脚本；如需 240 帧成果则定位历史文件或独立重跑。ComfyUI 迁移尚未验证。模型许可需按实际来源另查，原表未经核实的许可证结论不作为交付依据。未删除 201 或 InfiniteTalk 共享权重。

[new_video_inference_guide.md](new_video_inference_guide.md) 可作历史操作参考，其中 Python 版本、240 帧现存状态、预处理无异常等描述尚未同步；以本交付单与部署核查报告为准。
