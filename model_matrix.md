# 模型矩阵（三台服务器）

> 扫描时间：2026-11-10
> 扫描范围：`/data` 全盘 + `/nfs-data` (231)，`find -size +500M`
> SSH 用户：`zenking`（三台通）

| 简称 | IP | 主机名 | GPU | 主要存储 |
|---|---|---|---|---|
| **201** | `192.168.100.201` | a-X99 | 2× RTX 3090 24GB | `/dev/sda2` 1.9T（93% 用） |
| **230** | `192.168.100.230` | — | — | `/data`（含 `/data/192.168.8.231/` 子目录挂载 231 的盘） |
| **231** | `192.168.8.231` | zenking | — | `/nfs-data`（NFS 共享盘） |

---

## 🟡 ID-V2V 核心依赖

| 模型 | 大小 | 192.168.100.201 | 192.168.100.230 | 192.168.8.231 /nfs-data | 状态 |
|---|---|---|---|---|---|
| **idv2v.pth** (finetuned DiT+VACE) | ~72GB | 🔄 curl 下载中 | ❌ | ❌ | **唯一 hf-mirror 源，201 上 curl -C - 续传** |
| **Wan2.1 T5 UMT5-XXL** | 11GB | ✅ `/data/aigc/ID-V2V/checkpoints/wan/.../T2V-14B/models_t5_umt5-xxl-enc-bf16.pth` | ✅ `/data/192.168.8.231/zj/metahuman_work/InfiniteTalk/weights/Wan2.1-I2V-14B-480P/` | ✅ `/nfs-data/InfiniteTalk/weights/Wan2.1-I2V-14B-480P/` | ✅ 已从 230 拷贝到 201 |
| **Wan2.1 VAE** | 485MB | ✅ 同上路径 Wan2.1_VAE.pth | ✅ 同上 | ✅ 同上 | ✅ 已复用 |
| **Wan2.1 CLIP ViT-Huge-14** | 4.5GB | ✅ `.../I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth` | ✅ `/data/192.168.8.231/zj/.../Wan2.1-I2V-14B-480P/` | ✅ `/nfs-data/InfiniteTalk/weights/Wan2.1-I2V-14B-480P/` | ✅ 已复用 |
| **Wan2.1 tokenizer** (google/umt5-xxl) | ~28MB | ✅ `.../T2V-14B/google/umt5-xxl/` | ✅ | ✅ | ✅ 已复用 |
| ~~SAM3~~ (gated) | ~6.5GB | ❌ | ❌ | ❌ | **代码改走 SAM 2.1，跳过** |
| **SAM 2.1 large** | 858MB | ✅ symlink → `/data/chenna/BadgeGroove-OOD-Seg/sam2.1_hq_hiera_large.pt` | ✅ `/data/192.168.8.231/lbg/models/sam2.1_hiera_large.pt` | ❌ | ✅ 复用 BadgeGroove-OOD-Seg |
| **SAM 2.1 base+** | 309MB | ✅ symlink → `/data/chenna/BadgeGroove-OOD-Seg/sam2.1_hiera_base_plus.pt` | ❌ | ❌ | ✅ 复用 |
| **SAM 2.1 configs** | — | ✅ `checkpoints/VACE-Annotators/sam2/configs/sam2.1/` | ✅ 自带 | ❌ | ✅ 已复制 |
| **insightface buffalo_l** | ~300MB | ✅ symlink → `/data/aigc/insightface/server/.models/.bundles/buffalo_l-80ffe37d8a59/` | ❌ | ❌ | ✅ 复用 AvatarOps |

---

## 🎬 Wan2.1 全家桶（Text-to-Video / Image-to-Video）

| 模型 | 大小 | 192.168.100.201 | 192.168.100.230 | 192.168.8.231 /nfs-data | 备注 |
|---|---|---|---|---|---|
| Wan2.1-I2V-14B-480P (7×DiT shards) | 56GB | ❌ | ✅ `/data/192.168.8.231/zj/metahuman_work/InfiniteTalk/weights/Wan2.1-I2V-14B-480P/` | ✅ `/nfs-data/InfiniteTalk/weights/Wan2.1-I2V-14B-480P/` | ID-V2V **不需要**（有自己的 idv2v.pth） |
| Wan2.1-I2V-14B-720P fp8 e4m3fn | 16GB | ❌ | ✅ `/data/192.168.8.231/lbg/models/models/diffusion_models/` | ❌ | ComfyUI fp8 量化 |
| Wan2.1-I2V-14B-480P fp8 | 16GB | ❌ | ✅ 同上 | ❌ | ComfyUI fp8 |
| Wan2.2 I2V low-noise fp8 scaled | 14GB | ❌ | ✅ 同上 | ❌ | Wan 2.2 新版本 |
| Wan2.1 DiT Q4_K_S GGUF (480P) | 10GB | ❌ | ✅ `/data/192.168.8.231/lbg/models/models/diffusion_models/GGUF/` | ❌ | 量化 GGUF |
| Wan2.1 DiT Q4_K_S GGUF (720P) | 10GB | ❌ | ✅ 同上 | ❌ | 量化 GGUF |
| Wan2.1 InfiniTetalk-Single fp16 | 5.1GB | ❌ | ✅ `/data/192.168.8.231/lbg/models/models/diffusion_models/` | ❌ | InfiniteTalk 专用 |
| Wan2.1 FantasyPortrait fp16 | 2.0GB | ❌ | ✅ `/data/192.168.8.231/lbg/models/【Fantasy Portrait】.../models/diffusion_models/` | ❌ | Fantasy Portrait |
| Wan2.1 FusionX LoRA T2V | 303MB | ❌ | ✅ `/data/192.168.8.231/zj/.../Wan14BT2VFusioniX/FusionX_LoRa/` | ✅ | LoRA adapter |
| Wan2.1 FusionX LoRA I2V | 354MB | ❌ | ✅ 同上 | ✅ | LoRA adapter |
| Wan2.1 FusionX LoRA Phantom | 303MB | ❌ | ✅ 同上 | ✅ | LoRA adapter |
| Wan2.1 RealismBoost LoRA | 293MB | ❌ | ✅ `/data/192.168.8.231/zj/.../Wan14BT2VFusioniX/OtherLoRa's/` | ✅ | |
| Wan2.1 DetailEnhancer LoRA | 293MB | ❌ | ✅ 同上 | ✅ | |
| Wan2.1 lightx2v_I2V cfg_step_distill rank64 | 738MB | ❌ | ✅ `/data/192.168.8.231/lbg/models/models/loras/` | ❌ | LightX2V 采样蒸馏 |
| CLIP-ViT-H-fp16 (Wan通用) | 1.3GB | ❌ | ✅ `/data/192.168.8.231/lbg/models/models/clip_vision/` | ❌ | CLIP vision |
| Wan2.1 lightx2v LoRA rank32 (Comfy) | 303MB | ❌ | ❌ | ✅ `/nfs-data/InfiniteTalk/WanVideo_comfy/` | Comfy LoRA |

---

## 🧑‍🦱 数字人 / Avatar / Video Avatar

| 模型 | 大小 | 192.168.100.201 | 192.168.100.230 | 192.168.8.231 /nfs-data | 备注 |
|---|---|---|---|---|---|
| **HunyuanVideo-Avatar** T2V-720p | 30GB | ❌ | ❌ | ✅ `/nfs-data/avatarops/models/HunyuanVideo-Avatar/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt` | AvatarOps 主力 |
| HunyuanVideo-Avatar fp8 | 25GB | ❌ | ❌ | ✅ 同上 `mp_rank_00_model_states_fp8.pt` | 量化版本 |
| HunyuanVideo-Avatar LLaVA-LLaMA image (4 shards) | ~20GB | ❌ | ❌ | ✅ `.../ckpts/llava_llama_image/` | 视觉语言对齐 |
| HunyuanVideo-Avatar text_encoder_2 | 1.7GB | ❌ | ❌ | ✅ `.../ckpts/text_encoder_2/` | tf_model.h5 + pytorch_model.bin + safetensors |
| **LongCat-Video-Avatar-1.5** base_model (6 shards) | ~31GB | ❌ | ❌ | ✅ `/nfs-data/avatarops/models/LongCat-Video-Avatar-1.5/base_model/` | Meituan LongCat |
| LongCat-Video-Avatar text_encoder (5 shards) | ~24GB | ❌ | ❌ | ✅ `/nfs-data/avatarops/models/LongCat-Video/text_encoder/` | |
| LongCat-Video-Avatar-1.5 int8 base (4 shards) | ~15GB | ❌ | ❌ | ✅ `.../base_model_int8/quantized_model-*.safetensors` | 量化 |
| LongCat-Video-Avatar-1.5 DMD LoRA | 2.5GB | ❌ | ❌ | ✅ `.../lora/dmd_lora.safetensors` | |
| LongCat-Video-Avatar-1.5 whisper-large-v3 | ~11GB | ❌ | ❌ | ✅ `.../whisper-large-v3/` | ASR，含 fp32 双版本 |
| **InfiniteTalk** single fp16 | 9.3GB | ❌ | ✅ `/data/192.168.8.231/zj/metahuman_work/InfiniteTalk/weights/InfiniteTalk/single/` | ✅ `/nfs-data/InfiniteTalk/weights/InfiniteTalk/single/` | 单说话人 |
| InfiniteTalk multi fp16 | 9.3GB | ❌ | ✅ 同上 `multi/` | ✅ `/nfs-data/InfiniteTalk/weights/InfiniteTalk/multi/` | 多说话人 |
| InfiniteTalk 量化 (fp8 × 3 + int8 × 3 + LoRA × 2) | 8×19GB | ❌ | ✅ `/data/192.168.8.231/zj/.../InfiniteTalk/weights/InfiniteTalk/quant_models/` | ✅ `/nfs-data/InfiniteTalk/weights/InfiniteTalk/quant_models/` | 全量化 |
| InfiniteTalk T5 fp8 | 6.7GB | ❌ | ✅ 同上 | ✅ | 文本编码器量化 |
| **MuseV** motion UNet | 5.7GB | ❌ | ✅ `/data/192.168.8.231/zj/metahuman_work/MuseV/MuseV/motion/musev/unet/diffusion_pytorch_model.bin` | ❌ | MuseTalk 运动先验 |
| **MuseTalk** v15 UNet + musetalk | 3.4GB + 3.4GB | ❌ | ✅ `/data/192.168.8.231/zj/metahuman_work/OpenAvatarChat/models/musetalk/` | ❌ | 实时唇同步 |
| **SoulX-FlashHead-V10** Model_Lite | 6.1GB | ❌ | ✅ `/data/192.168.8.231/lbg/project/SoulX-FlashHead_V10/models/SoulX-FlashHead-1.3B/Model_Lite/` | ❌ | 头像生成 |
| SoulX-FlashHead-V10 Model_Pro | 6.0GB | ❌ | ✅ 同上 `Model_Pro/` | ❌ | 头像生成 |
| SoulX-FlashHead-V10 VAE_Wan | 485MB | ❌ | ✅ 同上 `VAE_Wan/Wan2.1_VAE.pth` | ❌ | Wan VAE 复用 |
| SoulX-FlashHead-V10 VAE_LTX | 1.7GB | ❌ | ✅ 同上 `VAE_LTX/diffusion_pytorch_model.safetensors` | ❌ | LTX VAE |
| **OpenAvatarChat** avatar_moxing14 frames.pkl | 4.4GB | ❌ | ✅ `/data/192.168.8.231/zj/metahuman_work/OpenAvatarChat/models/.../avatars/avatar_moxing14*/frames.pkl` | ❌ | 数字人资产 |
| **TEN_Turn_Detection** (4 shards) | ~19GB | ❌ | ✅ `/data/192.168.8.231/zj/metahuman_work/ten-turn-detection/TEN_Turn_Detection/` | ❌ | 检测模型 |
| **LTX** VAE (SoulX 用) | 1.7GB | ❌ | ✅ | ❌ | LTX-Video VAE |

---

## 🔤 大语言模型 / 多模态 LLM

| 模型 | 大小 | 192.168.100.201 | 192.168.100.230 | 192.168.8.231 /nfs-data | 备注 |
|---|---|---|---|---|---|
| **Qwen3.6-35B-A3B** (完整 26 shards) | ~94GB | ✅ `/data/feature/models/Qwen3.6-35B-A3B/` | ❌ | ❌ | MoE 主模型 |
| Qwen3.6-35B-A3B-NVFP4 (3 shards) | ~29GB | ✅ `/data/feature/models/Qwen3.6-35B-A3B-NVFP4/` + ✅ `/data/qianyong/tmp/` | ❌ | ❌ | NVFP4 量化 |
| Qwen3.6-35B-A3B-GPTQ-W4A16 (4 shards) | ~20GB | ✅ `/data/feature/models/` + ✅ `/data/qianyong/tmp/` | ❌ | ❌ | GPTQ 量化 |
| Qwen3.6-35B-A3B-NVFP4-DSPARK-v2 | 3.1GB | ✅ `/data/feature/models/Qwen3.6-35B-A3B-NVFP4-DSPARK-v2/` | ❌ | ❌ | DSPARK 版本 |
| **Qwen3.8-27B-NVFP4** | 22.6GB | ❌ | ❌ | ✅ `/nfs-data/metahuman_work/models/unsloth/` | NVFP4 量化 |
| Qwen3.8-27B-NVFP4 (Inferact shard 6/6) | 5.4GB | ❌ | ❌ | ✅ `/nfs-data/metahuman_work/models/Inferact/` | 推理分片 |
| **Qwen3-32B** w4a16 (4 shards) | ~19GB | ❌ | ✅ `/data/metahuman_work/` + ✅ `/data/lx_code/` | ❌ | AWQ |
| Qwen3-32B w4a16-chinese | ~19GB | ❌ | ✅ `/data/lx_code/llm_models/Qwen3-32B/Qwen3-32B-w4a16-chinese/` | ❌ | 中文特化 |
| Qwen3-32B Thinking-speculator.eagle3 | 3.4GB + opt 3.8GB | ✅ `/data/feature/models/` + ✅ `/data/qianyong/tmp/` | ❌ | ❌ | Speculative decoding |
| **Qwen3-14B** AWQ (2 shards) | ~10GB | ✅ `/data/metahuman_work/models/` + ✅ `/data/lbg/models/` | ✅ 同上路径 | ❌ | 多处重复 |
| Qwen3-14B 完整 8 shards | ~30GB | ❌ | ✅ `/data/lx_code/llm_models/confucius3-math-llm-14b/weights/Qwen3-14B/` | ❌ | |
| **Qwen3-8B** (5 shards) | ~19GB | ❌ | ❌ | ✅ `/nfs-data/metahuman_work/models/Qwen3-8B/` | |
| **Qwen2.5-7B-Instruct** AWQ | 4.0GB | ❌ | ✅ `/data/metahuman_work/qwen2.5/models/Qwen2.5-7B-Instruct-AWQ/` | ❌ | |
| **Confucius3-Math** (8 shards) | ~31GB | ❌ | ✅ `/data/metahuman_work/` + ✅ `/data/lx_code/llm_models/confucius3-math-llm-14b/` | ❌ | 数学推理 |
| **Phi-4-mini-reasoning** fp16 (2 shards) | ~10GB | ❌ | ✅ `/data/qianyong/Phi-4-mini-reasoning/` + ✅ `/data/lixiao/` | ❌ | reasoner |
| Phi-4-mini-reasoning-W4A16-by-LLMC | 4.1GB | ❌ | ✅ `/data/qianyong/` | ❌ | |
| Phi-4-mini-reasoning-INT8-by-BnB | ~4GB | ❌ | ✅ `/data/qianyong/` | ❌ | |
| Phi-4-mini-reasoning-NF4-by-BnB | ~2.9GB | ❌ | ✅ `/data/qianyong/` | ❌ | |
| Phi-4-mini-flash-fp8 (2 shards) | ~8.5GB | ❌ | ✅ `/data/lixiao/python_code/llm_phi4_mini/Phi-4-mini-flash-fp8/` | ❌ | Flash 推理 |
| **Hunyuan3D-Part** | 6.6GB + 7.4GB tar.gz | ✅ `/data/feature/models/Hunyuan3D-Part/model/` | ✅ `/data/lbg/models/Tencent-Hunyuan_Hunyuan3D-Part/model/` + tar.gz | ❌ | 3D 生成 |
| **MiniCPM-RobotManip** | 3.5GB | ✅ `/data/zhangping/robotic_arm/models/MiniCPM-RobotManip/` | ❌ | ❌ | 具身智能 |
| **SJY-1995 GeoTikzBridge-Base-8B** (4 shards) | ~20GB | ❌ | ✅ `/data/lbg/models/SJY-1995_GeoTikzBridge-Base-8B/` | ❌ | TikZ 生成 |
| **Gemma-3-12B** (5 shards) | ~24GB | ❌ | ❌ | ✅ `/nfs-data/aigc/models/gemma-3-12b/` | Google Gemma |
| **Qwen3-TTS-12Hz-1.7B-CustomVoice** | 682MB | ✅ `/data/aigc/comfyui-qwen-image21/models/Qwen3-TTS-Models/` | ❌ | ✅ `/nfs-data/aigc/comfyui-qwen-image21/models/Qwen3-TTS-Models/` | TTS |
| **Qwen3-ASR-1.7B** (2 shards) | ~8GB | ❌ | ✅ `/data/metahuman_work/qwen3-asr/Qwen3-ASR-1.7B/` | ❌ | ASR |
| **deepseek-math-7b-rl-prm-v0.1** (3 shards) | ~15GB | ❌ | ✅ `/data/metahuman_work/math_model_deployment/.../deepseek-math-7b-rl-prm-v0.1/` | ❌ | 数学推理奖励模型 |

---

## 🖼️ 图像 / 视频生成 ComfyUI 模型

| 模型 | 大小 | 192.168.100.201 | 192.168.100.230 | 192.168.8.231 /nfs-data | 备注 |
|---|---|---|---|---|---|
| **miniMAX-H3** ref2va pruned int8 convrot | 21GB | ✅ `/data/aigc/comfyui-h3/models/diffusion_models/` | ❌ | ✅ `/nfs-data/comfyui/storage-models/models/diffusion_models/` | MiniMax H3 视频 |
| miniMAX-H3 fl2va pruned int8 convrot | 21GB | ✅ 同上 | ❌ | ✅ 同上 | |
| miniMAX-H3 VAE fp16 | 5.2GB | ✅ `/data/aigc/comfyui-h3/models/vae/` | ❌ | ❌ | |
| miniMAX-H3 audio VAE fp32 | 605MB | ✅ `/data/aigc/comfyui-h3/models/vae/` | ❌ | ❌ | |
| miniMAX-H3 ref2v_turbo 4step LoRA bf16 | 1.9GB | ✅ `/data/aigc/comfyui-h3/models/loras/` | ❌ | ❌ | |
| miniMAX-H3 turbo LoRA v4 step600 EMA | 780MB | ✅ 同上 | ❌ | ❌ | |
| **krea2_turbo fp8 scaled** | 13GB | ❌ | ❌ | ✅ `/nfs-data/comfyui/storage-models/models/diffusion_models/` | Krea 2 视频 |
| **qwen_image_2.1** int8 convrot | 7.3GB | ✅ `/data/aigc/comfyui-qwen-image21/models/diffusion_models/qwen/` | ❌ | ✅ `/nfs-data/comfyui-qwen-image21/models/diffusion_models/qwen/` | Qwen 图像 2.1 |
| qwen_image_2.1 VAE bf16 | 675MB | ✅ `/data/aigc/comfyui-qwen-image21/models/vae/` | ❌ | ✅ 同上 | |
| **qwen3vl_8b** int8 convrot | 9.4GB | ✅ `/data/aigc/comfyui-qwen-image21/models/text_encoders/` | ❌ | ✅ `/nfs-data/aigc/comfyui-qwen-image21/models/text_encoders/` | Qwen3-VL 视觉 |
| **qwen3vl_32b** minimax_h3 nvfp4 awq | 15.7GB | ❌ | ❌ | ✅ `/nfs-data/comfyui/storage-models/models/text_encoders/` | MiniMax H3 T5 |
| **sd_xl_base_1.0** | 6.9GB | ❌ | ✅ `/data/metahuman_work/models/Stable-diffusion/` | ❌ | SDXL 1.0 |
| **IC-Light** SD15 fbc | 1.7GB | ✅ `/data/zhangping/ring/ComfyUI/models/unet/IC-Light/` | ❌ | ❌ | 光照控制 |
| **Hyper-FLUX.1-dev-8steps-LoRA** | 1.4GB | ✅ `/data/zhangping/ring/ComfyUI/models/loras/` | ❌ | ❌ | FLUX LoRA |
| **MelBandRoformer** | 912MB | ❌ | ✅ `/data/192.168.8.231/lbg/models/models/diffusion_models/` | ❌ | 音频生成 |
| **Sonic UNet** | 6.3GB | ❌ | ❌ | ✅ `/nfs-data/comfyui/storage-models/models/sonic/` | 音频驱动视频 |

---

## 🧠 其他模型 / 预训练权重

| 模型 | 大小 | 192.168.100.201 | 192.168.100.230 | 192.168.8.231 /nfs-data | 备注 |
|---|---|---|---|---|---|
| **whisper-large-v3** (flax + pytorch) | ~9GB | ❌ | ❌ | ✅ `/nfs-data/avatarops/models/LongCat-Video-Avatar-1.5/whisper-large-v3/` + ✅ `/nfs-data/chenna/audioDetect/models/whisper-large-v3/` | ASR |
| **chinese-wav2vec2-base** | 1.1GB | ❌ | ✅ `/data/192.168.8.231/lbg/models/models/transformers/TencentGameMate/` | ✅ `/nfs-data/InfiniteTalk/weights/chinese-wav2vec2-base/` | ASR |
| **DAViD** surface normals | 1.4GB | ❌ | ❌ | ❌ | **需下载**（深度估计） |
| **DepthAnything-V2-Large** | 1.3GB | ❌ | ❌ | ❌ | **需下载**（深度估计） |
| **BM1684X yoloe_top_outputs.npz** | 8.9GB | ❌ | ✅ `/data/liuc/release/qianyong/yoloe/workspace/` | ❌ | TPU YOLOE |
| **PaddleOCR-VL-1.6-adapted** | 1.9GB | ✅ `/data/chenna/models/` + ✅ 多处备份 | ❌ | ❌ | OCR 视觉语言 |
| **HunyuanImage-3.0** (32 shards) | ~170GB | ❌ | ❌ | ✅ `/nfs-data/feature/models/HunyuanImage-3.0/` | 混元图像 |
| **BM1684X bmodel** (SophGo) | — | ✅ `/data/qianyong/221.bak/liuc/ai-service-dev-env/.../BM1684X/` | ❌ | ❌ | 算能 TPU |
| **PP-FormulaNet-L** | 724MB + 561MB tar.gz | ❌ | ✅ `/data/chenna/Paddle-VL.bak/test_pp_formulat/` | ❌ | 公式识别 |

---

## 🐳 Docker Images（AI 相关，只列 >10GB）

| Image | 大小 | 192.168.100.201 | 192.168.100.230 | 192.168.8.231 /nfs-data |
|---|---|---|---|---|
| infinitetalk-v6.tar | 31GB | ❌ | ✅ `/data/192.168.8.231/docker-images/` | ❌ |
| infinitetalk-0.0.29.tar | 27GB | ❌ | ✅ 同上 | ❌ |
| mineru-vllm.tar | 26GB | ❌ | ✅ 同上 | ❌ |
| vllm-openai-v0.11.0.tar | 26GB | ❌ | ✅ 同上 + ✅ `/data/lixiao/install/` | ✅ `/nfs-data/lx_code/docker_images/` |
| cosyvoice-v1.6.tar + tar.gz | 18GB × 2 | ❌ | ✅ 两处 | ❌ |
| pytorch-2.9.tar | 17GB | ❌ | ✅ 同上 | ❌ |
| pytorch-2.8.0-cuda12.8 | 17GB | ❌ | ✅ 同上 | ❌ |
| pytorch-2.8.0-cuda12.6 | 14GB | ❌ | ✅ 同上 | ❌ |
| stable-diffusion-webui-service | 12GB | ❌ | ✅ 同上 | ❌ |
| vllm-openai-v0.26.0.tar | 8.9GB | ✅ `/data/qianyong/tmp/` | ❌ | ❌ |
| sophgo_v3.2_qianyong_1.tar | 9.2GB | ✅ `/data/` | ❌ | ❌ |
| DeepSeek-Prover-V1.5.tar | 14GB | ❌ | ✅ `/data/192.168.8.231/docker-images/zenking-ragflow.tar` (ragflow 含) | ❌ |
| dsj-20260825.tar.gz | 9.8GB | ❌ | ❌ | ✅ `/nfs-data/dsj/` |

---

## 📊 按服务器汇总

| 服务器 | IP | AI 模型 >500MB 数量 | 估算总大小 | 主要类别 |
|---|---|---|---|---|
| **201** | `192.168.100.201` | ~55 | ~450GB | MiniMax H3 + Qwen3 全套 + ID-V2V 部署中 |
| **230** | `192.168.100.230` | ~80 | ~1.2TB | Wan2.1 全家桶 + Avatar 全家桶 + ComfyUI + Docker 镜像 |
| **231** /nfs-data | `192.168.8.231` | ~100+ | ~3TB+ | InfiniteTalk + AvatarOps + HunyuanVideo + LongCat + Gemma + HunyuanImage-3.0 |

## 🔗 ID-V2V 依赖满足情况

| 组件 | 来源 | 状态 |
|---|---|---|
| idv2v.pth | **只能 hf-mirror**，201 上 curl -C - 续传 | 🔄 10.5GB / 72GB |
| Wan T5+VAE+CLIP+tok | 230 `/zj/metahuman_work/InfiniteTalk/` → 本地中转 scp → 201 | ✅ 到位 |
| SAM 2.1 | 201 `/data/chenna/BadgeGroove-OOD-Seg/` → symlink + configs 复制 | ✅ 到位 |
| insightface | 201 `/data/aigc/insightface/server/.models/` → symlink | ✅ 到位 |
| sam2 pip 包 | pip install sam2 | ✅ 已装 |
| **省掉的下载** | SAM3 gated 6.5GB + Wan 16GB + SAM2 1.2GB + insightface 0.3GB | **~24GB** |
