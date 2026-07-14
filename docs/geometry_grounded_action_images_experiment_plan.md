# Geometry-Grounded Action Images 实验设计计划

## 1. 目标

本文档整理基于 TesserAct 项目继续推进创新研究的实验计划。核心方向是：

> 将 TesserAct-style RGB-D-Normal / 4D 几何预测转换为 ActionImages-style 多视角动作热力图，再通过几何解码恢复 7-DoF 机器人动作。

暂定题目：

**Geometry-Grounded Action Images: Lightweight Action Decoding from 4D World Models**

中文可表述为：

**几何增强动作图像：从 4D 世界模型到机器人动作的轻量解码方法**

## 2. 核心假设

TesserAct 证明几何丰富的 RGB-D-Normal future 对动作推断有价值，但其动作接口主要是 point-cloud inverse dynamics。ActionImages 证明动作可以被表示成 pixel-grounded multiview heatmaps，但完整 video-action diffusion 训练成本很高。

本研究的核心假设是：

1. RGB-D-Normal / 4D geometry 可以作为 action-image prediction 的强输入信号。
2. action-image bottleneck 比直接 7-DoF regression 更可解释、更容易诊断失败。
3. 在低算力条件下，训练一个轻量 geometry-to-action-image decoder 比重训 world-action diffusion model 更可行。
4. 多视角 action heatmaps 可以利用相机几何稳定恢复 3D 语义动作点。

## 3. 总体 Pipeline

第一阶段不修改 TesserAct 主干，先做后处理轻量 decoder：

```text
RLBench GT RGB-D-N / camera / action
    -> ActionImages-style heatmap label generation
    -> lightweight geometry-to-action-image decoder
    -> predicted multiview RGB action heatmaps
    -> multiview geometric decoding
    -> 7-DoF action
```

第二阶段再接入 TesserAct 生成结果：

```text
initial image + instruction
    -> TesserAct RGB-D-N future
    -> trained action-image decoder
    -> action heatmaps
    -> 7-DoF action
```

第三阶段才考虑更重的模型集成：

```text
TesserAct backbone
    -> RGB + Depth + Normal + Action Image branch
```

## 4. 推荐 MVP

第一个里程碑：

> 在 RLBench 小子集上训练一个轻量 decoder，将 GT RGB-D-Normal clips 映射到 ActionImages-style action heatmaps，并解码回 7-DoF action。

该 MVP 不依赖真实机器人，也不需要训练 TesserAct / CogVideoX 主干。它要回答的问题是：

1. action-image bottleneck 在 GT geometry 输入下是否可行？
2. RGB-D-N 是否优于 RGB-only / RGB-D？
3. heatmap 解码误差主要来自 label projection、decoder prediction，还是 multiview fusion？
4. 该表示是否比 direct 7-DoF regression 更可解释？

## 5. 代码改造计划

建议在 TesserAct 仓库中新建一个实验包：

```text
tesseract_action_decoder/
    __init__.py
    datasets/
        rlbench_action_image_dataset.py
    geometry/
        action_projection.py
        heatmap_decode.py
        camera.py
    models/
        small_unet.py
        direct_regression.py
    train_action_image_decoder.py
    eval_action_image_decoder.py
    visualize_action_images.py
    configs/
        rlbench_rgb.yaml
        rlbench_rgbd.yaml
        rlbench_rgbdn.yaml
```

### 5.1 可复用代码

优先复用 ActionImages 仓库中的几何函数：

- `project_actions_7d_to_5d_torch_batch`
- `project_action_5d_to_rgb_torch`
- `fuse_multiview_heatmaps_to_7d_point_torch`
- `intrinsics_transform`

这些函数位于：

```text
../ActionImages/training/utils.py
```

RLBench 多视角数据读取可以参考：

```text
../ActionImages/training/dataset/rlbench.py
```

TesserAct 的 RGB-D-N 数据读取逻辑可以参考：

```text
tesseract/robodataset.py
```

### 5.2 需要新写的部分

1. RLBench action-image dataset adapter
   - 读取多视角 RGB 视频。
   - 读取或生成 depth / normal。
   - 读取 camera intrinsics / extrinsics。
   - 读取 7D / 8D action。
   - 调用 ActionImages 投影函数生成 heatmap labels。

2. Lightweight decoder
   - 第一版使用小型 2D CNN / U-Net。
   - 输入可以是 per-view RGB-D-N frame stack。
   - 输出为 per-view RGB heatmap。
   - 先做 per-frame 或 short-clip decoder，不急着上大 temporal transformer。

3. Evaluation script
   - heatmap peak error。
   - PCK。
   - 3D position error。
   - direction / orientation error。
   - gripper accuracy。
   - reprojection error。

4. Visualization script
   - 保存 predicted heatmap。
   - 保存 GT heatmap。
   - 保存 heatmap overlay on RGB。
   - 保存 decoded 3D action trajectory。

## 6. 实验分组

### 6.1 表征消融

| 实验 | 输入 | 输出 | 目的 |
| --- | --- | --- | --- |
| RGB-only | RGB | action heatmap | 基础视觉 baseline |
| RGB-D | RGB + depth | action heatmap | 验证 depth 是否提升空间定位 |
| RGB-D-N | RGB + depth + normal | action heatmap | 主方法 |
| RGB-D-N + camera rays | RGB-D-N + Plucker/camera ray | action heatmap | 验证相机几何编码收益 |

### 6.2 解码方式对比

| 实验 | 输入 | 输出 | 目的 |
| --- | --- | --- | --- |
| Direct regression | RGB-D-N | 7D action | 对比无 bottleneck 的动作回归 |
| Action-image bottleneck | RGB-D-N | heatmap -> 7D action | 主方法 |
| Oracle heatmap decode | GT heatmap | 7D action | 测几何解码上限 |

### 6.3 鲁棒性测试

| 扰动 | 目的 |
| --- | --- |
| depth noise | 模拟 TesserAct / depth estimator 误差 |
| normal noise | 测试 normal 对方向恢复的稳定性 |
| missing view | 测试多视角缺失 |
| random occlusion | 测试遮挡 |
| blur / compression | 模拟生成视频伪影 |

## 7. 指标

### 7.1 Heatmap 指标

- `peak_error_px`: 预测 peak 与 GT peak 的像素距离。
- `soft_argmax_error_px`: soft-argmax 坐标误差。
- `PCK@k`: peak 在 k 像素内的比例。
- `heatmap_mse`: 预测 heatmap 与 GT heatmap 的 MSE。

### 7.2 3D Action 指标

- `position_error`: gripper 3D 位置误差。
- `direction_error`: semantic direction vector angular error。
- `rotation_error`: 如果恢复完整姿态，则报告旋转误差。
- `gripper_accuracy`: 开合状态准确率。
- `trajectory_smoothness`: 相邻动作差分平滑性。

### 7.3 诊断指标

- `reprojection_error`: decoded 3D 点重新投影到各视角后的误差。
- `edge_peak_rate`: heatmap peak 落在图像边缘的比例。
- `view_consistency`: 多视角解码得分或重投影一致性。
- `latency_ms`: decoder 推理耗时。

## 8. 实验阶段

### 阶段 0：数据和几何闭环检查

目标：

```text
action -> multiview heatmap -> action
```

只测试投影和解码，不训练模型。

检查项：

1. RLBench action 格式是否统一。
2. camera extrinsics 是 camera-to-world 还是 world-to-camera。
3. gripper open / close 语义是否与 ActionImages 一致。
4. heatmap label 是否正确落在 gripper 附近。
5. oracle heatmap 解码误差是否足够小。

通过标准：

- GT heatmap 解码回 action 的 position error 在合理范围。
- 可视化 overlay 与原始视频一致。
- gripper 开合语义没有反。

### 阶段 1：GT RGB-D-N -> Action Heatmap

目标：

训练小型 decoder，验证 action-image bottleneck 是否成立。

实验：

1. RGB-only。
2. RGB-D。
3. RGB-D-N。
4. Direct 7D regression baseline。

通过标准：

- RGB-D-N 在 3D position / direction 指标上优于 RGB-only。
- action-image bottleneck 至少接近 direct regression，并提供更好的可视化诊断。

### 阶段 2：加入噪声和缺视角鲁棒性

目标：

验证几何输入和 action-image bottleneck 是否在不完美输入下稳定。

实验：

1. depth noise。
2. normal noise。
3. missing view。
4. random occlusion。
5. combined corruption。

通过标准：

- RGB-D-N decoder 在 moderate noise 下不显著崩溃。
- uncertainty / visibility 加权如果加入，应降低 missing view 下的误差。

### 阶段 3：TesserAct-generated RGB-D-N -> Action Heatmap

目标：

将输入从 GT geometry 替换为 TesserAct 生成的 RGB-D-N future。

实验：

1. GT RGB-D-N upper bound。
2. TesserAct-generated RGB-D-N。
3. TesserAct-generated + noise-trained decoder。

关键分析：

- world prediction quality 到 action error 的 gap。
- 哪些任务中 TesserAct 的 depth / normal 伪影最影响动作恢复。
- generated geometry 失败时，heatmap 是否提供可解释 failure pattern。

## 9. 初版模型建议

先使用小模型，避免过早引入复杂 transformer：

### 9.1 Per-view Small U-Net

输入：

```text
[B, V, T, C, H, W]
```

简化为：

```text
[B * V * T, C, H, W]
```

输出：

```text
[B * V * T, 3, H, W]
```

优点：

- 简单。
- 训练快。
- 容易 debug。
- 可直接测试 RGB / RGB-D / RGB-D-N 消融。

局限：

- 不使用时间上下文。
- 不显式建模多视角一致性。

### 9.2 Short-clip Temporal CNN

第二版可以加入短时序卷积：

```text
[B * V, C, T, H, W] -> [B * V, 3, T, H, W]
```

优点：

- 可利用轨迹连续性。
- 有助于 gripper state / action direction 平滑。

### 9.3 Multi-view Fusion Head

第三版再加入多视角融合：

```text
per-view features + camera ray encoding -> per-view heatmaps
```

这个版本更像论文主方法，但不建议一开始就做。

## 10. 关键风险

| 风险 | 说明 | 应对 |
| --- | --- | --- |
| RLBench 数据格式不统一 | action/camera/depth/normal 路径可能与两个仓库预期不同 | 先做数据探针脚本和 5 个样本可视化 |
| camera 坐标系错误 | c2w / w2c 混淆会导致 heatmap 完全错位 | 阶段 0 必须做 oracle projection-decode |
| gripper 语义反了 | open/close 在不同代码注释中可能不一致 | 通过样本动作和可视化确认 |
| normal 质量不稳定 | normal 估计可能影响方向预测 | 先做 RGB-D，再加 RGB-D-N |
| 只离线评估说服力不足 | 没有真实机器人执行 | 强化 oracle、robustness、failure visualization 和 RLBench proxy 指标 |

## 11. 我准备如何开始改代码

第一步先不训练模型，先做数据和几何闭环：

1. 创建 `tesseract_action_decoder/geometry/`。
2. 将 ActionImages 的投影与解码函数以最小依赖方式封装进本仓库，或通过本地路径导入。
3. 创建 `scripts/probe_rlbench_action_images.py`：
   - 读取一个 RLBench episode。
   - 加载 RGB、多视角相机、action。
   - 生成 GT action heatmap。
   - 解码 heatmap 回 action。
   - 输出 overlay 图和误差 JSON。
4. 先跑 3-5 个 episode，确认坐标系和 gripper 语义。

第二步做训练数据集：

1. 创建 `RLBenchActionImageDataset`。
2. 支持 `modalities=rgb|rgbd|rgbdn`。
3. 返回：

```python
{
    "inputs": Tensor,          # [V, T, C, H, W]
    "heatmaps": Tensor,        # [V, T, 3, H, W]
    "actions": Tensor,         # [T, 7]
    "extrinsics": Tensor,      # [V, T, 3, 4]
    "intrinsics": Tensor,      # [V, T, 3, 3]
    "metadata": dict,
}
```

第三步做 baseline：

1. `SmallUNetActionDecoder`。
2. `DirectActionRegressionBaseline`。
3. `train_action_image_decoder.py`。
4. `eval_action_image_decoder.py`。

## 12. 需要用户确认或提供的帮助

为了开始实现，最好先确认以下事项：

1. 是否已经下载 RLBench 数据，具体路径在哪里？
2. 数据里是否已经包含 depth / normal，还是需要用脚本生成？
3. 可用 GPU 情况：本地是否有 GPU，还是主要租云 GPU？
4. 希望第一版实验优先追求速度，还是尽量贴近论文设置？
5. 是否希望把新实验代码放在 TesserAct 仓库内，还是单独建一个新仓库/目录？

我的建议：

- 第一版代码直接放在 TesserAct 仓库内。
- 数据优先使用 ActionImages-RLBench 格式，因为它已经包含多视角、camera 和 action 的读取逻辑。
- 第一版分辨率降到 `128x128` 或 `160x160` 做快速闭环，确认方向成立后再上 `256x256`。
- 第一版只做 2 个视角和短片段，避免一开始被数据吞吐拖慢。

