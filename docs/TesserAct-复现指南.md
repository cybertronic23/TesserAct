# TesserAct 复现指南

## 项目简介

**TesserAct** 是 ICCV 2025 论文提出的 4D 世界模型，输入图像和文本指令，生成 RGB、Depth、Normal 视频，重建 4D 场景并预测动作。

- 论文: [arXiv:2504.20995](https://arxiv.org/abs/2504.20995)
- GitHub: [UMass-Embodied-AGI/TesserAct](https://github.com/UMass-Embodied-AGI/TesserAct)
- 模型: [HuggingFace](https://huggingface.co/anyeZHY/tesseract)

---

## 硬件资源要求

### 地瓜云实例选择

| 实例规格 | 显存 | CPU | 内存 | 适用场景 | 推荐度 |
|---------|-----|-----|-----|---------|-------|
| `desktop-5880gpu12g-16c32g` | 12GB | 16核 | 32GB | ❌ 不够 | ✗ |
| `desktop-5880gpu24g-32c64g` | 24GB | 32核 | 64GB | ❌ 勉强 | ✗ |
| `desktop-5880gpu48g-64c128g` | 48GB | 64核 | 128GB | ✅ LoRA 微调 | ⭐推荐 |
| `desktop-5880gpu96g-64c256g` | 48GB×2 | 64核 | 256GB | ✅ LoRA + SFT | ⭐⭐最佳 |

### 推荐配置

- **初学者/LoRA 微调**: `desktop-5880gpu48g-64c128g`
- **全量训练/SFT**: `desktop-5880gpu96g-64c256g`

---

## 环境配置

### 1. 创建 conda 环境

```bash
conda create -n tesseract python=3.9
conda activate tesseract
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 安装 TesserAct

```bash
pip install -e .
```

---

## 代码和数据准备

### 1. 克隆仓库

```bash
git clone https://github.com/UMass-Embodied-AGI/TesserAct.git
cd TesserAct
```

### 2. 下载预训练模型

需要下载 CogVideoX-5b-I2V 基座模型（约 30GB）：

```bash
# 安装 git-lfs
git lfs install

# 克隆模型（国内可能较慢，建议配置镜像）
git clone https://huggingface.co/THUDM/CogVideoX-5b-I2V
```

### 3. 已发布的 TesserAct 模型

```
anyeZHY/tesseract/tesseract_v01e_rgbdn_sft  # RGB+Depth+Normal SFT 模型
anyeZHY/tesseract/tesseract_v01e_rgb_lora    # RGB-only LoRA 模型
```

---

## 数据准备

详细步骤请参考 [DATA.md](./DATA.md)。

### 必要数据

| 数据类型 | 说明 | 生成工具 |
|---------|------|---------|
| RGB 视频 | 原始视频数据 | - |
| Depth 深度图 | 每帧的深度信息 | Temporal Marigold |
| Normal 法向量图 | 每帧的表面法线 | Temporal Marigold / NormalCrafter |

### 数据目录结构

```
data/
├── bridge/
│   └── processed/
│       └── 1/
│           └── video/
│               ├── rgb.mp4
│               ├── depth.mp4
│               └── normal.mp4
```

---

## 训练方式对比

### LoRA vs 全量 SFT

| 特性 | LoRA 微调 | 全量 SFT |
|-----|----------|---------|
| GPU 显存 | ~30GB | ~80GB+ |
| 训练时间 | ~2天 | 数天 |
| 数据需求 | ~100 视频 | 更多数据 |
| 参数量 | 几百万 | 全部 (~30B) |
| 效果 | 稍差但够用 | 更好 |

### LoRA 原理简介

LoRA (Low-Rank Adaptation) 是一种高效微调技术：

```
传统微调: 更新整个模型 (~30B 参数)
LoRA:     只训练 A、B 两个小矩阵 (~几百万参数)

训练后: W' = W + BA  (B×A 是低秩矩阵)
```

### 关键参数

| 参数 | 建议值 | 说明 |
|-----|-------|-----|
| `rank` (r) | 8-32 | 越大效果越好，但显存需求更高 |
| `lora_alpha` | 2×rank | 缩放因子 |
| `target_modules` | q_proj, v_proj | 注意力层 |

---

## 开始训练

### LoRA 微调（推荐）

```bash
bash train_i2v_depth_normal_lora.sh
```

关键配置参数：
- `frame_length`: 49 帧
- `train_batch_size`: 1
- `max_train_steps`: 200000
- `rank`: 512
- `lora_alpha`: 512
- `learning_rate`: 5e-5

### 全量 SFT 训练

```bash
bash train_i2v_depth_normal_sft.sh
```

---

## 推理测试

### RGB + Depth + Normal 生成

```bash
python inference/inference_rgbdn_sft.py \
  --weights_path anyeZHY/tesseract/tesseract_v01e_rgbdn_sft \
  --image_path asset/images/fruit_vangogh.png \
  --prompt "pick up the apple google robot"
```

### RGB-only 生成

```bash
python inference/inference_rgb_lora.py \
  --weights_path anyeZHY/tesseract/tesseract_v01e_rgb_lora \
  --image_path asset/images/fruit_vangogh.png \
  --prompt "pick up the apple google robot"
```

### RGBDN LoRA 推理

```bash
python inference/inference_rgbdn_lora.py \
  --base_weights_path anyeZHY/tesseract/tesseract_v01e_rgbdn_sft \
  --lora_weights_path ./your_local_lora_weights \
  --image_path asset/images/fruit_vangogh.png \
  --prompt "pick up the apple google robot"
```

输出结果保存在 `results/` 目录。

---

## 点云渲染（可选）

使用 Blender 渲染点云：

1. 下载 [Blender 4.3+](https://www.blender.org/download/)
2. 安装 [PyBlend](https://github.com/anyeZHY/PyBlend)
3. 运行渲染脚本：

```bash
blender-4.4.3/blender -b -P scripts/rendering_points.py -- \
  --combined_video ./results/val_0_pick_up_the_apple_google_robot_0.mp4 \
  --render_output ./results/rendered_results
```

---

## 参考资料

### 官方文档

- [GitHub 仓库](https://github.com/UMass-Embodied-AGI/TesserAct)
- [项目主页](https://tesseractworld.github.io)
- [USAGE.MD](./doc/usage.md) - 详细使用指南

### LoRA 学习资料

- [Hugging Face LoRA 教程](https://huggingface.co/learn/llm-course/chapter11/4)
- [PEFT 库文档](https://huggingface.co/docs/peft)
- [LightningAI lit-gpt](https://github.com/Lightning-AI/lit-gpt/blob/main/tutorials/finetune_lora.md)

### 相关技术

- [Temporal Marigold](https://huggingface.co/docs/diffusers/en/using-diffusers/marigold_usage) - 深度估计
- [NormalCrafter](https://github.com/Binyr/NormalCrafter) - 法向量生成
- [Finetrainers](https://github.com/a-r-r-o-w/finetrainers) - 训练框架

---

## 常见问题

### Q: 显存不够怎么办？

A: 减小 `frame_length` 和 `train_batch_size`，启用 `gradient_checkpointing`

### Q: 训练很慢怎么办？

A: 使用 `desktop-5880gpu96g` 多卡实例，或减少训练步数

### Q: 数据如何生成 Depth/Normal？

A: 使用 Temporal Marigold 处理 RGB 视频生成深度图，然后使用 NormalCrafter 生成法向量

---

## 附录：完整配置示例

### train_i2v_depth_normal_lora.sh 关键配置

```bash
# 模型路径
MODEL_PATH="THUDM/CogVideoX-5b-I2V"

# 训练参数
frame_length=49
height=480
width=640
train_batch_size=1
max_train_steps=200000

# LoRA 参数
rank=512
lora_alpha=512

# 优化器
optimizer=adamw
learning_rate=5e-5
lr_scheduler=constant_with_warmup
lr_warmup_steps=200

# 显存优化
gradient_checkpointing
enable_slicing
enable_tiling
mixed_precision=bf16
```