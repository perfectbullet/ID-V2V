# 新视频推理操作手册（以「精神小妹在末日修机EP2S12」为例）

**更新日期**: 2026-11-12  
**单机**: `zenking@192.168.8.231`（RTX 5090 32GB, sm_120 Blackwell）  
**案例样本**: `test_samples/restylization/spirit_girl`  
**输入视频**: `test_samples/精神小妹在末日修机EP2S12.mp4`（768×1344 竖屏, 24fps, 14.375s, 345 帧）

> 2026-11-11 曾尝试"201 上 preprocess + 231 上 inference"两端分工，2026-11-12 实测 **231 单机即可覆盖全流程**：
> - 231 `idv2v` env（Python 3.10 + torch 2.7.0+cu128）原生带 `sam2`，**preprocess 实测跑通**（345 帧 ≈9 min）
> - 231 `hyavatar` env（Python 3.11 + torch 2.7.0+cu128）**inference 已生产跑通**（spirit_girl 3 clip ≈3.5 h）
> - 两个 env 都已 Blackwell sm_120 就绪；201 机器不再参与

---

## 一、231 上的两个 conda env（各司其职）

| | `idv2v` env | `hyavatar` env |
|---|---|---|
| Python | 3.10.12 | 3.11.15 |
| torch | 2.7.0+cu128 ✅ sm_120 | 2.7.0+cu128 ✅ sm_120 |
| sam2 | ✅ 已装 | ❌ 不装（preprocess 不走这里） |
| 核心职责 | **preprocess**（SAM2 分割 + orig_pixel） | **inference**（DiT + VACE 扩散） |
| diffsynth | broken（`flash_attn_2_cuda` undefined symbol） | ✅ 正常 |
| 备注 | 不动 inference，diffsynth 坏了也不影响 | 不动 preprocess，sam2 不用装 |

> **不要**强行让一个 env 干所有事：preprocess 和 inference 是两段独立脚本，中间仅通过产物 `orig_pixel.mp4` 衔接，天然可以用不同 env。强行合并需要在 hyavatar 装 sam2（慢编译 C extension + av/imageio 依赖链），收益为零。

### 前置 checkpoint（已在 231）

- SAM2 checkpoint: `checkpoints/VACE-Annotators/sam2/sam2.1_hiera_base_plus.pt` (309MB，已从 201 同步到 231 `/data/chenna/...` 路径)
- ID-V2V finetuned: `checkpoints/idv2v.pth` (73GB，已在 NFS `/nfs-data/`，symlink)

---

## 二、样本目录规范

```
test_samples/restylization/<sample_name>/
├── source.mp4                  # 原始视频
├── stylized_first_frame.png    # 风格化首帧（I2V 的图像条件，需 1280×720）
├── prompt.txt                  # 风格/场景描述文本
├── preprocessing/
│   └── orig_pixel.mp4          # preprocess 产物（VACE 控制条件）
└── outputs/
    └── r1280x720_f81_kf0_idv2v/
        ├── run_config.json
        ├── first_frame.png
        ├── original_video.mp4
        ├── condition_0.mp4
        ├── generated_clip0/1/2.mp4
        ├── generated_video.mp4          ← 主产物
        ├── generated_video_withOrig.mp4 ← 左生成 / 右原片对比
        └── flip_test.mp4
```

---

## 三、完整操作流程（全部在 231）

### 步骤 1：本地准备样本目录

```bash
cd /home/zj/aigc/ID-V2V
SAMPLE=test_samples/restylization/spirit_girl
mkdir -p "$SAMPLE/keyframes"
cp "test_samples/精神小妹在末日修机EP2S12.mp4" "$SAMPLE/source.mp4"

# 抽首帧并 center-crop 到 1280×720
python - << 'PYEOF'
import cv2
src = "test_samples/restylization/spirit_girl/source.mp4"
cap = cv2.VideoCapture(src); _, frame = cap.read(); cap.release()
h, w = frame.shape[:2]; tw, th = 1280, 720
# 竖屏 9:16 → 先拉宽再截中间
if w / h < tw / th:
    nw = tw; nh = int(h * nw / w)
else:
    nh = th; nw = int(w * nh / h)
resized = cv2.resize(frame, (nw, nh), cv2.INTER_CUBIC)
frame = resized[(nh - th) // 2:(nh + th) // 2, (nw - tw) // 2:(nw + tw) // 2]
cv2.imwrite("test_samples/restylization/spirit_girl/stylized_first_frame.png", frame)
print("stylized_first_frame.png saved")
PYEOF

echo "A stylish young woman with fashionable streetwear works on a machine in a post-apocalyptic setting, cinematic lighting, dynamic scene" > "$SAMPLE/prompt.txt"

# 同步到 231（只这一台）
rsync -avz "$SAMPLE/" zenking@192.168.8.231:/data/aigc/ID-V2V/$SAMPLE/
```

### 步骤 2：231 idv2v env 跑 preprocess

```bash
ssh zenking@192.168.8.231
cd /data/aigc/ID-V2V
source ~/miniconda3/etc/profile.d/conda.sh && conda activate idv2v
export CUDA_VISIBLE_DEVICES=0
S=test_samples/restylization/spirit_girl

python -m idv2v.preprocess.sam2_preprocess \
  --video_path "$S/source.mp4" \
  --output_dir "$S/preprocessing" \
  --model_path checkpoints/VACE-Annotators/sam2/sam2.1_hiera_base_plus.pt \
  --config_path "configs/sam2.1/sam2.1_hiera_b+.yaml" \
  --min_area_ratio 0.005 --max_area_ratio 0.95 --top_n 5

python -m idv2v.preprocess.orig_pixel \
  --video_path "$S/source.mp4" \
  --mask_folder "$S/preprocessing" \
  --mask_image_file_name "sam3Mask_id_all.png" \
  --result_save_path "$S/preprocessing/orig_pixel.mp4"

rm -rf "$S/preprocessing"/frame_*/
ls -lah "$S/preprocessing/orig_pixel.mp4"   # 1.7MB 左右
```

> 实测耗时：SAM2 分割 345 帧 ≈ 9 min（~1s/帧，5090 比 201 的 3090 快 3×），orig_pixel ≈ 1 min。  
> `scripts/preprocess.sh` 硬编码 `.venv/bin/activate` + 不支持 env 覆盖，直接用内联命令。

### 步骤 3：231 hyavatar env 跑推理

```bash
ssh zenking@192.168.8.231
cd /data/aigc/ID-V2V
source ~/miniconda3/etc/profile.d/conda.sh && conda activate hyavatar

# 先清残留（生产运行前务必做）
pkill -9 -f inference/pipeline.py 2>/dev/null
ps aux | grep pipeline | grep -v grep | wc -l   # 归零才安全
# （可选）drop caches
echo 3 | sudo tee /proc/sys/vm/drop_caches 2>/dev/null

# 长任务用 nohup 脱离 SSH
SAMPLE_DIR=test_samples/restylization/spirit_girl \
GPU=0 \
STAGE_CHECKPOINT_TO_SHM=false \
MAX_NUM_FRAMES=240 \
nohup bash scripts/infer.sh > /tmp/spirit_log.log 2>&1 &
```

关键参数：

| 参数 | 值 | 说明 |
|---|---|---|
| `GPU` | `0` | 231 只有 1 张 5090 |
| `STAGE_CHECKPOINT_TO_SHM` | `false` | 73GB checkpoint 搬 `/dev/shm` 爆内存；mmap MAP_SHARED 从 NFS 读已 OK |
| `MAX_NUM_FRAMES` | `49` / `81` / `240` | 49=快速验证（55 min），81=单 clip（≈50 min），240=3 clip（≈3.5 h） |
| `NUM_INFERENCE_STEPS` | `30`（默认） | 单 clip 30 step；减半到 15 step 可加速但损失质量 |

### 步骤 4：取回结果

```bash
mkdir -p /home/zj/aigc/ID-V2V/test_samples/restylization/spirit_girl/outputs/r1280x720_f81_kf0_idv2v
rsync -avz zenking@192.168.8.231:/data/aigc/ID-V2V/test_samples/restylization/spirit_girl/outputs/r1280x720_f81_kf0_idv2v/ \
  /home/zj/aigc/ID-V2V/test_samples/restylization/spirit_girl/outputs/r1280x720_f81_kf0_idv2v/
```

---

## 四、实测耗时

### two_sitting_woman（1280×720, 49 帧单 clip, hyavatar）

| 阶段 | 耗时 |
|---|---|
| 模型加载 + GPU 搬迁 | ~3 min |
| Diffusion 30 step | 50 min 38 s（compile 后稳定 ~98s/step） |
| 编码 + 后处理 | ~1.5 min |
| **总计** | **≈ 55 min** |

### spirit_girl（768×1344 竖屏, 240 帧 3 clip, hyavatar）

| 阶段 | 耗时 |
|---|---|
| 模型加载 + checkpoint | ~15 min |
| Clip 1/3（81 帧） | ~50 min |
| Clip 2/3（81 帧） | ~50 min |
| Clip 3/3（78 帧） | 69 min 53 s（30 step × ~140s/step） |
| **总计** | **≈ 3 h 30 min** |

> 3 clip step 耗时从 ~98s 涨到 ~140s，因多 clip 时 DiT 显存换页开销上升。  
> preprocess 在 idv2v env 另外耗时 ≈10 min（与机器无关，只看采样数）。

---

## 五、过程中遇到的问题与修复

| # | 现象 | 根因 | 修复 |
|---|---|---|---|
| 1 | 最初部署时误用 torch 2.6.0+cu118，5090 kernel 跑不起来 | 3090 需要的环境是 torch 2.6.0+cu118，5090 Blackwell sm_120 必须 torch ≥ 2.7 + cu128 | 换 hyavatar env（torch 2.7.0+cu128）；**部署前必须核实 GPU 算力 → torch 兼容性** |
| 2 | 启动 `python -m idv2v.preprocess.sam2_preprocess` 报错 `FileNotFoundError: 'checkpoints/.../sam2.1_hiera_base_plus.pt'` | 201 上的 checkpoint 是 symlink 到 `/data/chenna/...`，231 没有这个文件 | 231 上 `rsync -avzL` 从 201 拉到本地路径（309MB） |
| 3 | 231 idv2v env 里 `diffsynth` import 报 `undefined symbol: _ZN3c105ErrorC2ENS_14SourceLocationESs` | diffsynth 编译时的 flash-attn 版本与当前 torch 不匹配 | **不改**：preprocess 不走 diffsynth，只有 inference 会 import diffsynth，而 inference 在 hyavatar env（diffsynth 正常） |
| 4 | 后台推理被 SSH 断连杀死 | `nohup ... &` 在非交互 SSH 下仍随会话退出 | 用 `nohup bash ... > log 2>&1 &`，确认 `ps aux` 有 pipeline 进程再断开；禁止 `cmd \| head`（SIGPIPE 杀进程） |
| 5 | 5 个 pipeline 进程同时存在、互相抢内存 | 多次重启未清理旧进程 | 启动前 `pkill -9 -f inference/pipeline.py` + `ps aux \| grep pipeline \| wc -l` 归零确认 |
| 6 | 推理末尾报 `ValueError: Videos do not share the same aspect ratio` | 原片竖屏 768×1344、生成横屏 1280×720，`stack_videos_left_right_and_save` 硬编码 `tol=0.15` 直接 raise | 改为「统一高度 + 各自按原比例缩放」，不再因比例不同崩溃 |
| 7 | 修完第 6 条后对比视频 0 字节 | 拼接宽度算出 1691（奇数），libx264 yuv420p 拒绝奇数尺寸 | 宽高向下对齐到偶数（1690×720） |
| 8 | `tail -f` 看起来「报错」 | 任务已结束，日志不再增长；ssh 输出 `libselinux.so.1: no version information available` 是无害警告 | 任务结束后用 `tail -20`；ssh 警告可忽略 |
| 9 | 竖屏视频构图偏窄 | 模型 1280×720 横屏训练，`center_crop_and_resize` 对竖屏裁掉上下 | 已知限制；如需完整竖屏需在预处理加 letterbox 黑边 |
| 10 | 强行想让 hyavatar 装 sam2 一把梭 | 错误理想驱动："一套 env 解决所有" | 放弃；preprocess 和 inference 天然独立，两个 env 各司其职 |

### 代码改动位置

- [pipeline.py](file:///home/zj/aigc/ID-V2V/src/idv2v/inference/pipeline.py#L185-L215)：`stack_videos_left_right_and_save` 改为比例自适应 + 偶数对齐
- [pipeline.py](file:///home/zj/aigc/ID-V2V/src/idv2v/inference/pipeline.py#L855-L860)：对比视频仅在 `save_verbose` 时生成

---

## 六、查看进度与结果

```bash
# 实时进度（Ctrl+C 退出；任务结束会一直挂着）
ssh zenking@192.168.8.231 'tail -f /tmp/spirit_log.log | tr "\r" "\n"'

# 快照
ssh zenking@192.168.8.231 'tail -40 /tmp/spirit_log.log | tr "\r" "\n" | tail -12; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader'

# 只看报错
ssh zenking@192.168.8.231 'grep -nE "Error|Traceback|ValueError" /tmp/spirit_log.log'

# 产物
ssh zenking@192.168.8.231 'ls -lah /data/aigc/ID-V2V/test_samples/restylization/spirit_girl/outputs/r1280x720_f81_kf0_idv2v/'
```

完成标志：日志出现 `Inference done.`，且输出目录存在 `generated_video.mp4`。

---

## 七、本次案例产物

| 文件 | 规格 |
|---|---|
| `generated_video.mp4` | 1280×720, 240 帧, 24fps, 10.0s, 11MB |
| `generated_video_withOrig.mp4` | 1690×720, 240 帧, 10.0s, 1.7MB（左生成 / 右原片） |
| `generated_clip0/1/2.mp4` | 三个 clip 分段输出 |
| `flip_test.mp4` | 生成/原片翻转对比 |

本地结果目录：[spirit_girl outputs](file:///home/zj/aigc/ID-V2V/test_samples/restylization/spirit_girl/outputs/r1280x720_f81_kf0_idv2v)

---

## 八、复用清单（跑新视频时逐项确认）

1. 新视频规格 → 决定 `MAX_NUM_FRAMES`（81 帧 ≈ 单 clip 50 min，240 帧 ≈ 3.5 h）
2. `stylized_first_frame.png` 必须是 **1280×720**（本地用 cv2 抽帧 crop）
3. `prompt.txt` 英文描述风格 + 动作
4. preprocess 产物 `orig_pixel.mp4` ≈ 1.7MB 左右才对；没有就是 sam2 checkpoint 没找到（见第 5.2 条）
5. 231 上 `idv2v` env 的 `diffsynth` broken 不用管；**preprocess 不走 diffsynth，inference 走 hyavatar 正常**
6. inference 启动前 `pkill -9 -f pipeline.py` + 确认进程数归零
7. 跑完检查 `generated_video.mp4` 存在且 `ffprobe` 帧数符合预期
