# TesserAct LoRA 源码深度分析

> 本文档逐段解析 TesserAct 项目中与 LoRA 微调相关的全部代码，从训练到推理，覆盖每一个关键环节。

---

## 目录

1. [LoRA 基础概念回顾](#1-lora-基础概念回顾)
2. [项目中的 LoRA 相关文件](#2-项目中的-lora-相关文件)
3. [训练脚本详解](#3-训练脚本详解)
4. [训练核心代码逐段解析](#4-训练核心代码逐段解析)
5. [RGB-only LoRA 推理代码解析](#5-rgb-only-lora-推理代码解析)
6. [RGBDN LoRA 推理代码解析](#6-rgbdn-lora-推理代码解析)
7. [LoRA 参数配置全表](#7-lora-参数配置全表)
8. [LoRA 与 SFT 的关键差异](#8-lora-与-sft-的关键差异)
9. [常见问题和调优建议](#9-常见问题和调优建议)

---

## 1. LoRA 基础概念回顾

LoRA（Low-Rank Adaptation）是一种高效微调技术，核心思想是：**冻结预训练模型的全部权重，在模型层旁边插入可训练的低秩矩阵**。

对于线性层 `W`（权重矩阵），LoRA 将其前向计算修改为：

```
h = Wx + BAx
```

其中：
- `W`：原始权重矩阵（冻结，不更新）
- `B`：低秩矩阵（可训练），形状 `d × r`
- `A`：低秩矩阵（可训练），形状 `r × k`
- `r`：rank（秩），远小于 `d` 和 `k`

`r` 是关键超参数。`r=512` 意味着用两个 `512` 维的低秩矩阵来近似原始权重的更新量（Delta W）。训练完成后，可以将 `BA` 合并回 `W`（推理时零额外开销），也可以单独保存 LoRA 权重（方便分发）。

---

## 2. 项目中的 LoRA 相关文件

| 文件 | 类型 | 行数 | 用途 |
|------|------|------|------|
| `tesseract/i2v_depth_normal_lora.py` | 训练代码 | 1152 | LoRA 微调训练主程序 |
| `train_i2v_depth_normal_lora.sh` | Shell 脚本 | 106 | 启动训练的入口脚本 |
| `inference/inference_rgb_lora.py` | 推理代码 | 160 | RGB-only LoRA 模型推理 |
| `inference/inference_rgbdn_lora.py` | 推理代码 | 210 | RGB+Depth+Normal LoRA 推理 |
| `tesseract/args.py` | 参数定义 | 490 | 定义 `--rank`、`--lora_alpha` 等参数 |

---

## 3. 训练脚本详解

### 3.1 Shell 启动脚本 `train_i2v_depth_normal_lora.sh`

**文件位置：** `train_i2v_depth_normal_lora.sh`（106 行）

这个脚本是整个 LoRA 训练的入口。我们逐部分拆解：

#### 3.1.1 环境变量配置（第 1-15 行）

```bash
export TORCH_LOGS="+dynamo,recompiles,graph_breaks"
export TORCHDYNAMO_VERBOSE=1
export WANDB_MODE="offline"
export NCCL_P2P_DISABLE=1
export TORCH_NCCL_ENABLE_MONITORING=0
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
```

这些环境变量控制：
- **TORCH_LOGS / TORCHDYNAMO_VERBOSE**：PyTorch 的动态编译日志级别。`+dynamo,recompiles,graph_breaks` 会在每次 torchdynamo 重新编译或图断裂时输出日志，帮助诊断性能问题。训练稳定后可以去掉。
- **WANDB_MODE="offline"**：Weights & Biases 日志以离线模式记录（不实时上传），训练结束后再手动同步。
- **NCCL_P2P_DISABLE=1**：禁用 NCCL 的 P2P（点对点）通信。在单机多卡环境中，如果 GPU 之间没有 NVLink 直连，禁用 P2P 可以避免通信错误。
- **TOKENIZERS_PARALLELISM=false**：禁用 tokenizer 的并行处理，避免在多进程 DataLoader 中出现死锁。
- **OMP_NUM_THREADS=1**：OpenMP 线程数设为 1，避免与 PyTorch 的线程池冲突。

```bash
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_TIMEOUT=1800
export NCCL_SOCKET_TIMEOUT=1800
export TORCH_NCCL_BLOCKING_WAIT=1
```

这些是 NCCL 分布式通信的调试和优化配置：
- **NCCL_IB_DISABLE=1**：禁用 InfiniBand（IB），在有 IB 的集群中才需要启用。
- **NCCL_SOCKET_IFNAME=eth0**：指定 NCCL 使用的网络接口为 `eth0`。如果机器有多个网卡（如 `eth0`、`eth1`、`ib0`），需要正确指定。
- **NCCL_SOCKET_TIMEOUT=1800**：Socket 超时时间 1800 秒，大模型训练中避免因通信耗时过长而超时断开。

#### 3.1.2 分布式启动配置（第 17-22 行）

```bash
NUM_GPUS=$(nvidia-smi -L | wc -l)
PORT=$(shuf -i 20000-60000 -n 1)
MASTER_ADDR=$HOSTNAME
MASTER_PORT=$SLURM_JOB_ID
NNODES=$SLURM_JOB_NUM_NODES
NODE_RANK=$SLURM_NODEID
```

这里使用 `nvidia-smi -L | wc -l` 自动检测 GPU 数量。注意 `MASTER_PORT=$SLURM_JOB_ID` 使用了 SLURM 环境变量——这意味着它预期在 SLURM 调度系统下运行。如果不在 SLURM 环境，这些变量可能为空，需要手动设置 `NNODES=1` 和 `NODE_RANK=0`。

#### 3.1.3 训练超参数（第 24-40 行）

```bash
LEARNING_RATES=("5e-5")
LR_SCHEDULES=("constant_with_warmup")
OPTIMIZERS=("adamw")
MAX_TRAIN_STEPS=("200000")
frame_length=49
```

这些变量外面套了多层 `for` 循环（42-106 行），可以方便地做超参数搜索。比如把 `LEARNING_RATES` 改为 `("5e-5" "1e-4" "3e-5")`，就会依次用三个学习率各训练一次。

`frame_length=49` 是视频帧数。TesserAct 的 LoRA 训练使用 49 帧视频，相比 SFT 的全量训练，这属于标准配置。

#### 3.1.4 核心训练命令（第 48-98 行）

```bash
launcher_cmd="torchrun --master_addr=$MASTER_ADDR --node_rank=$NODE_RANK \
  --rdzv_backend static --nnodes $NNODES --nproc_per_node=$NUM_GPUS --rdzv_id=$NODE_RANK"

cmd="$launcher_cmd \
  tesseract/i2v_depth_normal_lora.py \
  --pretrained_model_name_or_path $MODEL_PATH \
  --dataset_file cache/samples_depth_normal.json \
  --data_root $DATA_ROOT \
  ...
  --rank 512 \
  --lora_alpha 512 \
  ..."
```

使用 `torchrun` 启动分布式训练。关键 LoRA 参数是 `--rank 512` 和 `--lora_alpha 512`。

训练命令使用了 `deepspeed` 配置（虽然不是通过 `deepspeed` 命令启动，但训练代码中会检测 DeepSpeed 配置）。

---

## 4. 训练核心代码逐段解析

**文件位置：** `tesseract/i2v_depth_normal_lora.py`（1152 行）

### 4.1 导入模块（第 1-72 行）

```python
from peft import LoraConfig, get_peft_model_state_dict, set_peft_model_state_dict
```

这是 HuggingFace PEFT（Parameter-Efficient Fine-Tuning）库的核心导入：
- **`LoraConfig`**：LoRA 的配置类，定义 rank、alpha、target_modules 等
- **`get_peft_model_state_dict`**：从模型中提取 LoRA 权重字典
- **`set_peft_model_state_dict`**：将 LoRA 权重字典加载回模型

其他导入：
```python
from diffusers.training_utils import cast_training_params
from diffusers.utils import export_to_video, load_image, convert_unet_state_dict_to_peft
```

注意 `convert_unet_state_dict_to_peft`——这个函数将 Diffusers 格式的 U-Net 状态字典转换为 PEFT 兼容的格式。因为 TesserAct 的 transformer 架构与 U-Net 不同，转换时实际上只是重命名 key。

### 4.2 模型加载与冻结（第 386-434 行）

```python
tokenizer = AutoTokenizer.from_pretrained(...)
text_encoder = T5EncoderModel.from_pretrained(...)
```

加载 T5 文本编码器（用于编码文字指令）：

```python
transformer = CogVideoXTransformer3DModel.from_pretrained_modify(
    "anyeZHY/tesseract",
    subfolder="tesseract_v01e_rgbdn_sft",
    torch_dtype=load_dtype,
)
```

这里是关键——LoRA 训练加载的是 **已经 SFT 微调过的 TesserAct 权重**（`tesseract_v01e_rgbdn_sft`），而不是基座模型 `CogVideoX-5b-I2V`。这意味着 LoRA 训练是在 SFT 权重的基础上进一步微调。

这在实践中意味着：如果你没有先训练 SFT 模型，LoRA 训练不会直接使用基座模型。你需要先有 SFT 权重。

```python
vae = AutoencoderKLCogVideoX.from_pretrained(...)
```

加载 VAE（变分自编码器），用于将视频帧压缩到潜空间：

```python
text_encoder.requires_grad_(False)
vae.requires_grad_(False)
transformer.requires_grad_(False)
```

**这是 LoRA 的核心——冻结所有原始权重：**
- `text_encoder` 冻结：文本编码器不更新
- `vae` 冻结：VAE 编码器/解码器不更新
- `transformer` 冻结：整个 transformer 的主干权重不更新
- 只有后面插入的 LoRA 参数会更新

### 4.3 LoRA 配置与注入（第 475-484 行）

这是整个 LoRA 训练最关键的部分：

```python
# now we will add new LoRA weights to the attention layers
target_modules = ["to_k", "to_q", "to_v", "to_out.0"]
target_modules.extend(["patch_embed.proj", "patch_embed.depth_proj", "patch_embed.normal_proj"])
transformer_lora_config = LoraConfig(
    r=args.rank,
    lora_alpha=args.lora_alpha,
    init_lora_weights=True,
    target_modules=target_modules,
)
transformer.add_adapter(transformer_lora_config)
```

#### target_modules 解析

LoRA 被注入到两类模块：

**第一类：自注意力层的 4 个线性投影**

| 模块名 | 对应操作 | 作用 |
|--------|---------|------|
| `to_k` | 生成 Key 矩阵 | 注意力机制中的 Key |
| `to_q` | 生成 Query 矩阵 | 注意力机制中的 Query |
| `to_v` | 生成 Value 矩阵 | 注意力机制中的 Value |
| `to_out.0` | 注意力输出投影 | 将多头注意力的输出投影回原始维度 |

这些是标准的 Transformer 注意力投影，在 CogVideoX 的 3D 注意力块中每个都有。注入 LoRA 后，`to_q` 的前向计算变为：

```
output = W_q * input + B_q * A_q * input
         ^^^^^^^^   ^^^^^^^^^^^^^^^^
         冻结       可训练 (LoRA)
```

**第二类：Patch Embedding 的 3 个投影**

| 模块名 | 对应操作 | 作用 |
|--------|---------|------|
| `patch_embed.proj` | RGB 图像的 Patch 投影 | 将图像块映射到嵌入空间 |
| `patch_embed.depth_proj` | 深度图的 Patch 投影 | 深度图独立编码 |
| `patch_embed.normal_proj` | 法向量图的 Patch 投影 | 法向量图独立编码 |

这三个投影层处理不同的输入模态（RGB、深度、法向量），注入 LoRA 后可以针对特定数据集的分布做调整。

**为什么不注入 MLP 层（`ff.net`）？**

PEFT 库的 `LoraConfig` 默认只注入线性层。如果 `target_modules` 中不包含 `ff.net` 之类的 MLP 层，那么 MLP 保持完全冻结。这个项目只注入了注意力投影和 Patch Embedding，大约覆盖了模型参数的 10-20%。

#### LoraConfig 参数

```python
r=args.rank,              # LoRA 秩，默认 64，训练脚本设为 512
lora_alpha=args.lora_alpha,  # 缩放因子，默认 64，训练脚本设为 512
init_lora_weights=True,   # 使用 Kaiming 均匀分布初始化 A 矩阵，B 矩阵初始化为 0
target_modules=[...],     # 要注入的目标模块名列表
```

**缩放机制：** 实际权重更新量为 `(lora_alpha / r) * BA`。当 `lora_alpha=512, r=512` 时，缩放因子为 1.0。如果 `lora_alpha=256, r=512`，缩放因子为 0.5。

**初始化方式：** `init_lora_weights=True` 使用默认的 Kaiming 初始化：
- A 矩阵用 `~Uniform(-1/sqrt(k), 1/sqrt(k))` 初始化
- B 矩阵全零初始化
- 这样训练开始时 BA = 0，不影响原始模型输出

#### transformer.add_adapter 内部发生了什么？

`add_adapter` 是 PEFT 库的方法，它会：
1. 遍历 `transformer` 的所有模块
2. 对名称匹配 `target_modules` 的 `nn.Linear` 层，在原层旁边插入 `nn.Linear` 的 LoRA 变体
3. 将 LoRA 变体的权重标记为 `requires_grad=True`
4. 将原始权重标记为 `requires_grad=False`

### 4.4 可训练参数分类（第 496-605 行）

LoRA 训练中并非所有可训练参数都是 LoRA 参数。代码将可训练参数分为三类：

```python
lora_params = [p for n, p in transformer.named_parameters() if "lora" in n and p.requires_grad]
dn_out_params = [p for n, p in transformer.named_parameters() if "dn_out" in n and p.requires_grad]
patch_embed_params = [
    p for n, p in transformer.named_parameters() if "lora" not in n and "dn_out" not in n and p.requires_grad
]
non_lora_params = dn_out_params + patch_embed_params
```

这三组参数分别使用不同的学习率：

```python
params_to_optimize = [
    {"params": lora_params, "lr": args.learning_rate},       # 标准学习率
    {"params": patch_embed_params, "lr": args.learning_rate}, # 标准学习率
    {"params": dn_out_params, "lr": args.learning_rate * 3},  # 3 倍学习率
]
```

**为什么 `dn_out_params` 用 3 倍学习率？**

`dn_out` 层是 TesserAct 自定义的输出投影层（Depth-Normal 输出头）。这些层是从零初始化的（不是预训练权重），且参数量少，因此用更高的学习率让它们更快收敛。这是一个在实践中验证有效的训练技巧。

### 4.5 保存钩子（第 487-512 行）

```python
def save_model_hook(models, weights, output_dir):
    if accelerator.is_main_process:
        transformer_requires_grad_state_dict = {}
        for model in models:
            if isinstance(unwrap_model(accelerator, model), type(unwrap_model(accelerator, transformer))):
                model = unwrap_model(accelerator, model)
                requires_grad_state_dict = {
                    k.replace("default.", ""): v for k, v in model.named_parameters() if v.requires_grad
                }
                transformer_requires_grad_state_dict.update(requires_grad_state_dict)
            if weights:
                weights.pop()
        CogVideoXImageToVideoPipeline.save_lora_weights(
            output_dir,
            transformer_lora_layers=transformer_requires_grad_state_dict,
        )
```

这个钩子重写了 Accelerate 的默认保存逻辑：
1. 只保存 `requires_grad=True` 的参数（即 LoRA 权重 + dn_out + patch_embed 可训练部分）
2. 使用 `CogVideoXImageToVideoPipeline.save_lora_weights` 以 Diffusers 标准的 LoRA 格式保存
3. 注意 `k.replace("default.", "")`：PEFT 库在模块名前加 `default.` 前缀，保存时去掉

**保存的文件结构：**

```
checkpoint-1000/
├── transformer.safetensors       # 只含可训练参数（~几 MB）
└── ...
```

这与保存完整模型不同，LoRA checkpoints 非常小（通常 5-50 MB，取决于 rank 和模块数）。

### 4.6 加载钩子（第 514-571 行）

加载钩子比保存更复杂：

```python
def load_model_hook(models, input_dir):
    ...
    # 如果是 DeepSpeed 模式，需要先重建模型
    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        transformer_ = CogVideoXTransformer3DModel.from_pretrained_modify(...)
        transformer_.add_adapter(transformer_lora_config)
    
    # 加载完整的状态字典
    state_dict = CogVideoXImageToVideoPipeline.lora_state_dict(input_dir)
    
    # 分离 LoRA 权重和基座权重
    # 第一步：加载 LoRA 权重
    lora_state_dict = {k: v for k, v in state_dict.items() if "lora" in k}
    lora_state_dict = convert_unet_state_dict_to_peft(lora_state_dict)
    set_peft_model_state_dict(transformer_, lora_state_dict, adapter_name="default")
    
    # 第二步：加载非 LoRA 的基座权重（dn_out, patch_embed 等）
    base_state_dict = {k: v for k, v in state_dict.items() if "lora" not in k}
    transformer_.load_state_dict(base_state_dict, strict=False)
```

这里的核心逻辑是 **双阶段加载**：
1. 先用 `set_peft_model_state_dict` 加载 LoRA 权重（PEFT 格式）
2. 再用 `load_state_dict(strict=False)` 加载非 LoRA 的可训练权重（dn_out、patch_embed 等）

`strict=False` 是因为状态字典中只包含可训练参数，不包含冻结参数的权重。

### 4.7 训练循环（第 782-1036 行）

训练循环的标准流程：

```
for epoch in range(first_epoch, args.num_train_epochs):
    for step, batch in enumerate(train_dataloader):
        with accelerator.accumulate(transformer):
            1. VAE 编码图像和视频（冻结 VAE）
            2. 用 T5 编码文本指令（冻结 text_encoder）
            3. 添加噪声（前向扩散过程）
            4. Transformer 预测噪声
            5. 计算 loss（RGB + Depth + Normal 三个分支）
            6. 反向传播（只更新 LoRA 参数）
            7. 梯度裁剪
            8. 优化器步进
```

#### Loss 计算（第 926-944 行）

```python
def loss_fn(pred, target, weight):
    return torch.mean((weight * (pred - target) ** 2).reshape(batch_size, -1), dim=1).mean()

rgb_loss = loss_fn(rgb_pred, rgb_target, weights)
depth_loss = loss_fn(depth_pred, depth_target, weights)
normal_loss = loss_fn(normal_pred, normal_target, weights)
loss = rgb_loss + depth_loss + normal_loss
```

这是标准的 MSE（均方误差）损失，三个模态各自计算后相加。

值得注意的是 **masked loss**（第 927-933 行）：

```python
mask = batch["masks"].to(accelerator.device, non_blocking=True)
C = target.shape[2]
rgb_mask, depth_mask, normal_mask = mask[:, 0], mask[:, 1], mask[:, 2]
target[depth_mask == 0, :, C // 3 : C // 3 * 2] = 0
model_pred[depth_mask == 0, :, C // 3 : C // 3 * 2] = 0
target[normal_mask == 0, :, C // 3 * 2 :] = 0
model_pred[normal_mask == 0, :, C // 3 * 2 :] = 0
```

如果某个样本缺少 depth 或 normal 标签（mask=0），对应的 loss 项被 mask 掉，不参与梯度的计算。这使得模型可以同时处理有完整标签和部分标签的数据。

---

## 5. RGB-only LoRA 推理代码解析

**文件位置：** `inference/inference_rgb_lora.py`（160 行）

这是最简化的 LoRA 推理流程，只生成 RGB 视频（不需要 depth 和 normal）。

### 5.1 Pipeline 加载（第 64-70 行）

```python
pipe = CogVideoXImageToVideoPipeline.from_pretrained(
    pretrained_model_path, torch_dtype=weight_dtype
).to("cuda")

del pipe.transformer.patch_embed.pos_embedding
pipe.transformer.patch_embed.use_learned_positional_embeddings = False
pipe.transformer.config.use_learned_positional_embeddings = False
```

这里直接使用标准的 Diffusers `CogVideoXImageToVideoPipeline`（不是 TesserAct 的自定义 Pipeline），因为 RGB-only LoRA 不需要 depth/normal 处理。

**为什么要删除 pos_embedding？**

TesserAct 的 SFT 训练移除了位置编码（`use_learned_positional_embeddings = False`），因为自定义的 Patch Embedding 使用 RoPE（旋转位置编码）替代。推理时也必须做同样的处理，否则会报 shape 不匹配的错误。

### 5.2 LoRA 权重加载（第 73-86 行）

```python
if lora_weights_path.count("/") == 2:
    # 从 HuggingFace Hub 下载
    list_of_files = lora_weights_path.split("/")
    repo_id = "/".join(list_of_files[:2])
    subfolder = list_of_files[2]
    lora_weights_path = snapshot_download(
        repo_id=repo_id,
        local_dir_use_symlinks=False,
    )
    lora_weights_path = os.path.join(lora_weights_path, subfolder)

pipe.load_lora_weights(lora_weights_path, adapter_name="cogvideox-lora")
pipe.set_adapters(["cogvideox-lora"], [1.0])
```

路径判断逻辑：如果路径包含 2 个斜杠（如 `anyeZHY/tesseract/tesseract_v01e_rgb_lora`），就认为它来自 HuggingFace Hub，用 `snapshot_download` 下载。

`load_lora_weights` 是从 Diffusers pipeline 继承的方法。它的内部逻辑：
1. 读取保存的 `transformer.safetensors`（只含 LoRA 权重）
2. 识别所有权重名称中的 LoRA 参数
3. 用 PEFT 的 `set_peft_model_state_dict` 注入到模型的对应层

`pipe.set_adapters(["cogvideox-lora"], [1.0])` 中的 `[1.0]` 是适配器缩放因子。设为 1.0 表示 LoRA 的贡献以 100% 强度生效。可以设为小于 1 的值来减弱 LoRA 的影响。

### 5.3 推理循环（第 96-124 行）

```python
for im_idx, (prompt, image_path) in enumerate(zip(validation_prompts, validation_images)):
    image = load_image(image_path)
    image = crop_and_resize_frames([np.array(image)], (height, width))[0]
    image = Image.fromarray(image)
    image = torch.from_numpy(np.array(image)).to("cuda") / 255.0
    image = image.permute(2, 0, 1).unsqueeze(0)
    
    for idx in range(num_validation_videos):
        result = pipe(
            image=image,
            prompt=prompt,
            guidance_scale=guidance_scale,
            use_dynamic_cfg=use_dynamic_cfg,
            height=height,
            width=width,
            num_inference_steps=50,
        )
```

与 SFT 推理不同，这里不需要 depth 和 normal 输入。模型直接根据 RGB 图像 + 文本指令生成 49 帧视频。

---

## 6. RGBDN LoRA 推理代码解析

**文件位置：** `inference/inference_rgbdn_lora.py`（210 行）

这个脚本比 RGB-only 版本更复杂，因为它需要 depth 和 normal 输入。

### 6.1 三阶段模型加载

**阶段 1：加载 Pipeline（第 58 行）**

```python
pipe = TesserActImageToDepthNormalVideoPipeline.from_pretrained(
    pretrained_model_path, torch_dtype=weight_dtype
).to(device)
```

使用 TesserAct 自定义 Pipeline，支持 RGB+Depth+Normal 四种输入。

**阶段 2：加载基座权重（第 66-84 行）**

```python
transformer = TesserActDepthNormal.from_pretrained_modify(
    base_weights_path, subfolder=subfolder
)
transformer.config.in_channels += 16 * 4  # 扩展输入通道数以支持 depth+normal
del transformer.patch_embed.pos_embedding
transformer.patch_embed.use_learned_positional_embeddings = False
pipe.transformer = transformer
```

这里加载的是 TesserAct SFT 权重（`tesseract_v01e_rgbdn_sft`），作为 LoRA 的基座模型。`in_channels += 16 * 4` 是因为 TesserAct 的 Patch Embedding 额外编码了 depth 和 normal 信息。

**阶段 3：加载 LoRA 权重（第 89-101 行）**

```python
if lora_weights_path.count("/") == 2:
    # 从 HuggingFace Hub 下载
    ...
pipe.load_lora_weights(lora_weights_path, adapter_name="cogvideox-lora")
pipe.set_adapters(["cogvideox-lora"], [1.0])
```

与 RGB-only 版本完全相同的加载方式，只是这里的 pipeline 是 `TesserActImageToDepthNormalVideoPipeline`。

### 6.2 输入数据处理（第 111-144 行）

```python
# 加载 RGB 图像
val_image = load_image(image_path)

# 加载深度图
depth_image = np.load(depth_path)
depth_image = 1 - depth_image

# 加载法向量图
normal_image = cv2.cvtColor(cv2.imread(normal_image), cv2.COLOR_BGR2RGB)

# 拼接为 9 通道输入 [H, W, 9]
image = torch.cat([val_image, depth_image, normal_image], dim=2)
image = image.permute(2, 0, 1).unsqueeze(0)  # [B, C, H, W]
```

为什么是 9 通道？RGB（3 通道）+ Depth（3 通道，重复 3 次）+ Normal（3 通道）= 9 通道。

---

## 7. LoRA 参数配置全表

### 7.1 可配置参数

| 参数 | 默认值 | 训练脚本设置 | 含义 |
|------|--------|------------|------|
| `--rank` | 64 | 512 | LoRA 矩阵的秩，越大表达能力越强、显存占用越高 |
| `--lora_alpha` | 64 | 512 | 缩放因子，`scaling = lora_alpha / rank` |
| 隐式 target_modules | 无 | `to_k, to_q, to_v, to_out.0, patch_embed.proj, depth_proj, normal_proj` | 注入的目标层 |

### 7.2 rank 参数的影响

| rank | 可训练参数量 | 显存增量 | 表达能力 | 推荐场景 |
|------|------------|---------|---------|---------|
| 8-16 | ~2M | ~0.5GB | 弱 | 数据极少（< 50 视频） |
| 32-64 | ~8M | ~1GB | 中 | 默认/起步 |
| 128-256 | ~32M | ~2GB | 较强 | 中等数据量 |
| 512 | ~128M | ~4GB | 强 | 项目推荐配置，数据充足 |

项目选择 r=512，说明对表达能力要求较高，同时 48GB 的 RTX5880 完全可以容纳这个开销。

### 7.3 优化器参数

```python
optimizer=adamw
learning_rate=5e-5
beta1=0.9
beta2=0.95
weight_decay=0.001
```

LoRA 参数使用标准 AdamW 优化器，与全量 SFT 相似。但权重衰减（weight_decay=0.001）对 LoRA 参数有正则化作用，防止过拟合。

---

## 8. LoRA 与 SFT 的关键差异

### 8.1 代码层面的差异

| 对比项 | LoRA (`i2v_depth_normal_lora.py`) | SFT (`i2v_depth_normal_sft.py`) |
|--------|----------------------------------|----------------------------------|
| 可训练参数 | 仅 LoRA 权重 | 全部 transformer 参数 |
| 模型冻结 | `transformer.requires_grad_(False)` 再 add_adapter | 无冻结 |
| 参数数量 | ~几百万（~128M @ rank=512） | ~50 亿 |
| 显存需求 | ~30GB | ~80GB+ |
| 保存的权重 | 仅 LoRA 参数（几 MB-GB） | 全部参数（~20GB） |
| 推理时的加载 | `load_lora_weights()` 注入 | 直接替换 transformer |

### 8.2 训练速度对比

```
LoRA:   每步 ~0.5-1 秒 (rank=512, batch=1, 单卡 48GB)
SFT:    每步 ~2-3 秒 (batch=1, 单卡 48GB 不够，需要多卡)

LoRA:   200000 步约 2 天
SFT:    40000 步约数天（需要多卡）
```

### 8.3 内存使用差异

LoRA 训练不需要存储优化器状态（如 Adam 的动量）给冻结参数，因此显著节省内存：

```
SFT 内存组成:
  - 模型参数: ~20GB
  - 梯度: ~20GB  
  - 优化器状态: ~40GB (Adam: 2×参数量)
  - 中间激活: ~5-10GB
  - Total: ~85-90GB

LoRA 内存组成:
  - 模型参数（冻结）: ~20GB
  - LoRA 参数: ~0.5GB
  - LoRA 梯度: ~0.5GB
  - LoRA 优化器状态: ~1GB
  - 中间激活: ~5-10GB
  - Total: ~27-32GB
```

---

## 9. 常见问题和调优建议

### 9.1 为什么训练脚本里的 rank=512 但默认是 64？

脚本中 `--rank 512` 是硬编码在 shell 命令里的，而 `args.py` 的默认值是 64。这是项目作者经过实验后推荐的配置。对于自己的数据集，可以先从 r=64 起步，如果欠拟合再增大 rank。

### 9.2 LoRA 权重合并和导出

训练完成后，LoRA 权重保存在 `checkpoint-{step}/transformer.safetensors` 中。要合并回原始模型用于推理：

```python
from diffusers import CogVideoXImageToVideoPipeline

# 加载基座模型
pipe = CogVideoXImageToVideoPipeline.from_pretrained("THUDM/CogVideoX-5b-I2V")

# 加载 LoRA 并合并
pipe.load_lora_weights("path/to/lora/weights")
pipe.fuse_lora()  # 将 LoRA 权重合并回原模型

# 保存合并后的模型
pipe.save_pretrained("merged-model")
```

`fuse_lora()` 将 `BA` 矩阵合并到原始 `W` 中：`W' = W + (alpha / r) * BA`。合并后推理速度与原始模型完全相同，没有额外开销。

### 9.3 多 adapter 切换

Diffusers Pipeline 的 `set_adapters` 支持加载多个 LoRA adapter 并切换：

```python
pipe.load_lora_weights("adapter1", adapter_name="my-dataset-v1")
pipe.load_lora_weights("adapter2", adapter_name="my-dataset-v2")

# 切换 adapter
pipe.set_adapters(["my-dataset-v1"], [1.0])
# 或者混合多个 adapter
pipe.set_adapters(["my-dataset-v1", "my-dataset-v2"], [0.7, 0.3])
```

这在对比不同微调版本的效果时非常有用，不需要重新加载模型。

### 9.4 什么时候应该用 LoRA 而不是 SFT？

**适合 LoRA 的场景：**
- 自定义数据集只有 50-200 个视频
- GPU 显存有限（<= 48GB）
- 只需要对模型的特定行为做小幅调整
- 需要快速迭代多个版本

**适合 SFT 的场景：**
- 有大规模数据集（1000+ 视频）
- 需要对模型做大幅度的行为改变
- 有多卡集群（80GB × 8+）

### 9.5 推理时 adapter 缩放因子的作用

```python
pipe.set_adapters(["cogvideox-lora"], [1.0])  # 正常强度
pipe.set_adapters(["cogvideox-lora"], [0.5])  # 减半 LoRA 影响
pipe.set_adapters(["cogvideox-lora"], [0.0])  # 完全禁用 LoRA
```

缩放因子小于 1 可以减弱 LoRA 的效果，在某些情况下可以改善泛化性（防止过拟合到训练数据的风格）。建议在生成时尝试 0.7-1.0 之间的值，观察效果差异。
