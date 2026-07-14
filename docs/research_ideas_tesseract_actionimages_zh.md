# research_ideas_tesseract_actionimages_zh-研究笔记：在低算力条件下结合 TesserAct 与 ActionImages

## 1. 背景与动机

本文档整理我们围绕以下两篇工作进行的研究方向分析：

- TesserAct: Learning 4D Embodied World Models
- Action Images: End-to-End Policy Learning via Multiview Video Generation

这里的实际约束非常重要：可用训练资源有限，只能偶尔租用云 GPU，没有真实机器人平台可部署。因此，目标不应该是复现大规模视频扩散模型训练，而是寻找一个理论上站得住脚、实验上可以验证、并且能用公开数据和较低算力完成的方法创新方向。

## 2. 两篇论文分别贡献了什么

### TesserAct

TesserAct 是一个 4D embodied world model。给定输入图像和文本指令，它生成 RGB、Depth、Normal 视频。随后，这些 RGB-D-Normal 预测结果会被重建成 4D 场景表示，并通过 point-cloud inverse dynamics model 预测 7-DoF 机器人动作。

其核心 pipeline 是：

```
初始图像 + 指令
    -> RGB-D-Normal 视频生成
    -> 4D 场景 / 点云重建
    -> PointNet 风格 inverse dynamics model
    -> 7-DoF 动作
```

TesserAct 适合作为实验基础，原因是它的仓库相对完整，提供了训练和推理脚本、LoRA 相关代码、数据生成工具和较清楚的使用说明。相比 ActionImages，它更容易跑起来，也更容易在代码层面做修改。

### ActionImages

ActionImages 提出把机器人动作表示成图像。一个 7-DoF 动作会被转换成三个语义 3D 点：夹爪位置、normal 方向和 up 方向。这些点被投影到多个相机视角中，并渲染成 RGB Gaussian heatmaps。生成出的 action heatmaps 可以再通过多视角几何方法解码回连续动作。

其核心 pipeline 是：

```
7-DoF 动作
    -> 语义 3D 点
    -> 多视角投影
    -> RGB action heatmaps / action videos
    -> 多视角几何解码
    -> 7-DoF 动作
```

ActionImages 论文训练了一个统一的 video-action model，但公开仓库对于完整实验复现并不如 TesserAct 完整，而且完整训练所需算力也明显更高。因此，在低算力研究中最值得复用的部分不是大规模视频扩散训练，而是它提出的 action-image 表示和对应解码逻辑。

## 3. 关键观察

TesserAct 和 ActionImages 其实从不同方向处理相似问题：

- TesserAct 关心：几何丰富的未来预测能否帮助机器人动作推断？
- ActionImages 关心：机器人动作本身能否表示到与观察相同的像素/视频空间中？

这带来一个低算力研究机会：

> 使用 TesserAct 风格的 RGB-D-Normal 或 4D 几何预测作为输入，但不直接回归 7-DoF 动作，而是引入可解释的 ActionImages-style action-image bottleneck。
> 

对应 pipeline 可以变成：

```
RGB-D-Normal / 4D 世界预测
    -> 轻量 geometry-to-action-image decoder
    -> 多视角 pixel-grounded action heatmaps
    -> 几何动作解码
    -> 7-DoF 动作
```

这样可以避免训练大型 world-action diffusion model，同时结合两篇论文的核心思想：TesserAct 的几何世界建模，以及 ActionImages 的像素级动作表示。

## 4. 推荐研究方向

### 暂定题目

题目：**Geometry-Grounded Action Images: Lightweight Action Decoding from 4D World Models**

中文可以理解为：

**几何增强动作图像：从 4D 世界模型到机器人动作的轻量解码方法**。

### 核心研究问题

RGB-D-Normal 或 4D 几何世界预测能否被转换成 pixel-grounded action images，从而提供一个从 world model 到 robot action 的轻量、可解释桥梁？

### 主要假设

相比从点云或 RGB 特征直接回归 7-DoF 动作，引入 action-image 中间表示可能带来：

1. 更好的可解释性。
2. 对噪声或不完整未来预测更强的鲁棒性。
3. 在几何 world model 与视频空间动作表示之间建立自然桥梁。
4. 避免训练统一 video-action diffusion model 的高算力成本。

## 5. 两种集成策略

### 策略 A：后处理轻量 Decoder

这是推荐的第一阶段方向。

```
TesserAct / GT RGB-D-Normal 视频
    -> 小型 geometry-to-action-image decoder
    -> ActionImages-style heatmap
    -> 7-DoF 动作
```

这种方式保持 TesserAct 模型本身不变，只单独训练一个小 decoder，将 RGB-D-Normal 或重建几何转换成 action images。

优点：

- 工程风险低。
- 训练成本低。
- 可以先使用 ground-truth RGB-D-Normal 数据，而不依赖 TesserAct 生成质量。
- 容易定位失败原因。
- 适合单卡实验。

局限：

- 不是端到端。
- 仍然存在 world prediction 到 action prediction 的误差传递。
- 如果不强调 action-image bottleneck，容易被认为只是 inverse dynamics 的替代模块。

### 策略 B：TesserAct 多输出分支

这是可能的第二阶段扩展。

```
初始图像 + 指令
    -> TesserAct backbone
    -> RGB + Depth + Normal + Action Image 视频
    -> 7-DoF 动作
```

这种方式需要修改 TesserAct，让 action images 成为第四种输出模态。

优点：

- 更统一，更接近 ActionImages 的思想。
- 方法故事更强。
- 可以联合建模世界预测和动作预测。

局限：

- 工程成本高。
- 需要改模型结构、数据加载、loss、训练脚本和推理 pipeline。
- 需要更多调参和 GPU 资源。
- 失败时更难定位原因。

推荐推进顺序：

```
第一阶段：后处理轻量 decoder
第二阶段：如果第一阶段有效，再做小规模多输出分支原型
```

## 6. 建议方法：后处理轻量 Decoder

### 输入

decoder 的输入可以包括：

- 仅 RGB 视频。
- RGB-D 视频。
- RGB-D-Normal 视频。
- 重建点云或 pointmaps。
- 当前状态与预测未来状态组成的时序对。

### 输出

decoder 预测 ActionImages-style 多视角 action heatmaps：

- position heatmap。
- normal-direction heatmap。
- up-direction heatmap。
- 可选 gripper openness channel。
- 可选 uncertainty 或 visibility channel。

### Decoder 选择

建议从简单模型开始：

- 对堆叠 RGB-DN 帧使用 2D CNN 或 U-Net。
- 对短视频片段使用小型 temporal CNN。
- 只有当 CNN baseline 明显不足时，再考虑 Tiny ViT。

不建议一开始使用 diffusion 或大 transformer。第一阶段目标不是生成逼真视频，而是准确、鲁棒地预测 action images。

### 动作恢复

预测 heatmaps 后，用几何解码恢复 7-DoF 动作：

```
预测的多视角 heatmaps
    -> 2D 点提取
    -> ray casting / triangulation / multiview scoring
    -> 语义 3D 点
    -> 7-DoF 动作重建
```

这样可以直接与 ActionImages 的解码逻辑对齐，同时保持模型轻量。

## 7. 实验设计

### 数据集

建议优先使用 RLBench，因为它提供：

- 仿真 demonstrations。
- 相机参数。
- 7-DoF 动作。
- 多视角观察。
- 无需真实机器人部署即可评估。

TesserAct-style RGB-D-Normal 数据可以使用已有深度/法线估计工具生成或近似，具体取决于可用预处理脚本和算力。

### 阶段 1：表示与 Decoder 研究

先使用 ground-truth 数据：

```
GT RGB-D-Normal / geometry
    -> geometry-to-action-image decoder
    -> predicted action image
    -> 7-DoF action
```

Baseline：

1. RGB-only to action image。
2. RGB-D to action image。
3. RGB-D-Normal to action image。
4. 如果可行，加入 point-cloud IDM baseline。
5. 直接 7-DoF regression baseline。

指标：

- 2D heatmap peak error。
- 3D position error。
- orientation error。
- gripper state accuracy。
- temporal smoothness。
- missing views、heatmap noise、depth noise、occlusion 下的鲁棒性。

### 阶段 2：TesserAct 输出研究

使用 TesserAct 生成或推断出的 RGB-D-Normal：

```
TesserAct-generated RGB-D-Normal
    -> geometry-to-action-image decoder
    -> predicted action
```

将其与 ground-truth-input 版本比较，量化 world-model prediction quality 对 action recovery 的影响。

### 阶段 3：可选集成

如果后处理 decoder 结果明显有效，再考虑给 TesserAct 加一个小型 action branch：

```
RGB-D-Normal generation + action-image prediction
```

这应该作为原型验证，而不是第一阶段目标。

## 8. 为什么该方向适合低算力条件

ActionImages 最昂贵的部分是训练大型视频扩散 backbone。本方向避开了这一点。

主要训练目标是一个轻量 action-image decoder，可以在以下设置下完成：

- RLBench 小子集。
- 降低分辨率。
- 短视频片段。
- 单 GPU 或少量云 GPU。

该实验仍然有意义，因为主张并不是“训练一个更好的 world model”，而是：

> 几何丰富的世界预测可以转换成可解释的 action images，从而提供一个低算力的 4D world model 到 pixel-grounded robot control 的桥梁。
> 

## 9. 最接近的相关工作

| 工作 | 链接 | 主要贡献 / 创新点 | 核心 Pipeline | 与本方向的关系 |
| --- | --- | --- | --- | --- |
| TesserAct | arXiv, Project, GitHub | 提出 4D embodied world model，生成 RGB、Depth、Normal 视频，重建 4D 场景，并通过 inverse dynamics 预测动作。 | 初始图像 + 指令 -> RGB-D-Normal 视频 -> 4D 重建 / 点云 -> PointNet inverse dynamics -> 7-DoF 动作。 | 主要实验基础。本方向用可解释 action-image bottleneck 替代或增强其 PointNet 式直接动作回归。 |
| ActionImages | arXiv, Project, GitHub | 将 7-DoF 动作转换成多视角 pixel-grounded RGB Gaussian heatmaps，并训练统一 video-action model。 | 7-DoF 动作 -> 语义 3D 点 -> 多视角 heatmaps -> video-action generation -> 多视角解码 -> 7-DoF 动作。 | 提供 action-image 表示和解码灵感。本方向复用表示，而不训练完整大规模 video-action diffusion model。 |
| MVISTA-4D | arXiv | 构建 view-consistent 4D world model，支持 arbitrary-view RGBD 生成和 test-time action inference，缓解单视角和 inverse dynamics 病态问题。 | 单视角 RGBD -> 多视角 RGBD imagination -> back-projection 和 fusion -> test-time trajectory latent optimization -> residual inverse dynamics -> 动作。 | 思路接近：world prediction to action。区别是 MVISTA-4D 使用优化和 residual IDM，本方向使用 action images 作为显式中间表示。 |
| X-WAM | arXiv, Project | 统一 4D world synthesis 与 action execution，使用多视角 RGB-D 预测、3D 重建和 asynchronous denoising。 | Observation -> video diffusion prior -> multi-view RGB-D future + 3D reconstruction + asynchronous denoising action execution。 | 高算力统一 world-action model。本方向是低算力后处理替代路线，而不是完整 unified diffusion model。 |
| PointAction | arXiv | 使用 3D points 作为通用动作表示，将共享 video-to-point model 与 embodiment-specific point-to-action decoder 分离。 | Observation/video -> RGB + dynamic XYZ pointmaps -> point-to-action decoder -> executable controls。 | 模块化设计很接近。区别是 PointAction 使用 pointmaps 和 point-to-action decoding，本方向使用 RGB-DN 几何预测 pixel-grounded action images。 |
| GEM-4D | arXiv | 训练时通过 geometry branch 将几何结构蒸馏进 video world model，推理时移除该分支，并使用 inverse dynamics 得到控制。 | 初始观察 -> geometry-enhanced video world model training -> correspondence-consistent rollouts -> inverse dynamics -> 6-DoF trajectories。 | 同样强调几何有助于动作提取。区别是 GEM-4D 修改 world-model training，本方向保持 world model 固定，只训练小型 action-image decoder。 |
| GAF | arXiv | 提出 Gaussian Action Field，在 3D Gaussians 中加入 motion attributes，用于当前渲染、未来预测和 action query。 | 两张 RGB 输入 -> Gaussian scene representation -> motion-augmented future Gaussian field -> point cloud matching action query。 | 相关于 4D 几何动作表示。区别是 GAF 使用 3D Gaussian motion fields，本方向使用 TesserAct-style RGB-DN/4D 输出和 ActionImages-style heatmaps。 |
| LaWAM | arXiv | 使用 inverse-dynamics latent actions 和冻结视觉编码器特征构建 latent world action model，生成 latent visual subgoals。 | Visual encoder features -> inverse-dynamics latent action posterior -> latent future subgoal prediction -> action generation。 | 同属模块化 world-to-action 设计。区别是 LaWAM 在 latent feature space 中工作，本方向是 pixel-grounded 且 geometry-aware。 |
| RepWAM | arXiv | 提出 representation visual-action tokenizer，将 visual latents 与 latent action transitions 对齐，用于 world-action modeling。 | Visual observations -> representation visual-action tokenizer -> latent action tokens -> world/action modeling -> robot control adaptation。 | 相关于 latent action bottleneck。区别是本方向使用显式、可视化 action heatmaps，而不是 latent action tokens。 |
| DIAL | Project | 通过 latent world modeling 解耦 intent 与 action。VLM 预测 latent intent，轻量 inverse-dynamics policy 转成动作。 | Observation -> latent foresight / intent bottleneck -> lightweight inverse dynamics policy -> action chunks。 | 高层上也区分 world/intent 与 action。区别是 DIAL 偏 latent VLA，本方向是 RGB-DN geometry to action-image decoding。 |
| MoLA | arXiv | 使用多种 modality-aware inverse dynamics models 和 latent actions，将 imagined future frames 转成可执行动作。 | Video future prediction -> modality-specific inverse dynamics models -> latent actions -> diffusion/flow action head -> executable controls。 | 相关于 imagined futures to actions。区别是 MoLA 使用 latent-action mixtures，本方向使用可解释 action-image bottlenecks。 |
| Structured 4D Latent Predictive Model for Robot Planning | OpenReview PDF | 在结构化 3D/4D latent space 中建模 dynamics，并使用 goal-conditioned inverse dynamics 做 robot planning。 | 多视角图像 -> 3D latent state -> 4D latent future prediction -> 点云 / rendered views -> goal-conditioned inverse dynamics -> 动作。 | 相关于 4D latent planning 和 inverse dynamics。区别是本方向聚焦从 RGB-DN 几何预测 pixel-grounded action images。 |

## 10. 与已有工作的差异化

很多近期工作遵循类似模式：

```
World prediction / latent future / 4D geometry
    -> inverse dynamics or action decoder
    -> robot action
```

因此，本方向不能被包装成“又一个 inverse dynamics model”。需要强调以下差异点：

1. **Action image as an interpretable bottleneck**
    
    decoder 预测的是可视化 action heatmaps，而不仅仅是低维动作。
    
2. **Geometry-grounded action-image prediction**
    
    TesserAct-style RGB-D-Normal 或 4D geometry 为 action-image generation 提供显式空间结构。
    
3. **Low-compute bridge between two paradigms**
    
    结合 4D world models 的几何信息和 ActionImages 的 pixel-grounded action 表示，同时避免大规模 video-action model 训练。
    
4. **Robust and diagnosable action inference**
    
    在解码成 7-DoF 动作之前，可以先在 action heatmap 空间中检查失败原因。
    

## 11. 建议论文故事线

可以这样定位论文：

> TesserAct 证明 geometry-rich RGB-D-Normal world predictions 对机器人动作推断有帮助，但其 downstream inverse dynamics module 是从重建点云中直接回归动作。ActionImages 证明动作可以表示为 pixel-grounded multiview heatmaps，但需要大规模 video-action model 训练。我们结合这两者，提出一个轻量 geometry-to-action-image decoder，将 RGB-D-Normal 或 4D world predictions 转成可解释 action images，再解码为连续 7-DoF 动作。
> 

可能的贡献点：

1. 一个面向 4D world models 的 geometry-grounded action-image decoding framework。
2. 一个轻量 decoder，用于将 RGB-D-Normal 或 4D geometry 映射到 ActionImages-style multiview heatmaps。
3. 在 RLBench 上进行可控实验，证明相比直接 action regression 或 RGB-only decoding，该方法更具可解释性和鲁棒性。
4. 分析 depth、normal、多视角几何在 noise、occlusion、missing-view 条件下对 action reconstruction 的影响。

## 12. 实际下一步

1. 复现 TesserAct 的 inference/data path，确认 RGB-D-Normal 输入输出格式。
2. 在 RLBench 上复现 ActionImages 的 action projection 和 decoding 逻辑。
3. 构建离线数据集：

```
RGB / Depth / Normal / camera parameters / 7-DoF action
    -> ActionImages-style heatmap labels
```

1. 训练一个小型 RGB-DN-to-action-image decoder。
2. 与 RGB-only 和 direct-action baselines 比较。
3. 增加鲁棒性测试：noise、blur、missing view、depth noise、occlusion。
4. 只有当轻量 decoder 被验证有效后，再考虑集成 action branch 到 TesserAct。

## 13. 推荐第一个里程碑

第一个里程碑建议设为：

> 在 RLBench 上训练一个小型 decoder，将 GT RGB-D-Normal clips 映射到 ActionImages-style action heatmaps，再解码回 7-DoF 动作，并与 RGB-only 和 direct regression baselines 对比。
> 

这个里程碑不需要真实机器人，也不需要训练大型 diffusion model。它可以直接检验核心假设，并为后续是否扩展到 TesserAct 模型内部提供清晰判断。

## 14. 与持久 3D 记忆方向的资源和复杂度对比

后续调研中还发现了另一个可能方向：借鉴 **Learning 3D Persistent Embodied World Models**，在 TesserAct 这类 4D / RGB-D-Normal world model 上加入持久 3D memory。这个方向可以补充 TesserAct 的一个短板：TesserAct 能生成短 horizon 的 4D future，但不显式维护跨时间、跨视角、可长期更新的 3D scene memory。

不过，在资源紧张的情况下，**TesserAct + Action Image decoder** 仍然更适合作为主线；**TesserAct + persistent 3D memory** 更适合作为后续扩展或 ablation。

| 方向 | 算力 / 资源要求 | 实验设计难度 | 代码开发难度 | 推荐优先级 |
| --- | --- | --- | --- | --- |
| TesserAct + Action Image decoder | 低到中 | 中 | 中 | 高 |
| TesserAct + persistent 3D memory | 中到高 | 高 | 高 | 中低 |

### 14.1 Action Image decoder 方向为什么更轻量

该方向的核心是 decoder-level innovation，可以冻结 TesserAct，甚至第一阶段完全不依赖 TesserAct 预测结果，只使用 ground-truth RGB-D-Normal / point cloud 训练和验证 decoder：

```
GT or TesserAct RGB-D-Normal / 4D scene
    -> geometry-to-action-image decoder
    -> action heatmaps
    -> multi-view geometric decoding
    -> 7-DoF action
```

资源优势：

1. 不需要重训 video diffusion 或 world model 主干。
2. 可以使用小模型作为 decoder，例如 PointNet、Point Transformer、轻量 UNet heatmap head。
3. 可以先做离线 supervised learning，用 action error 和 heatmap reprojection error 评估。
4. 失败原因容易拆解：geometry 预测错、heatmap 预测错、几何解码错可以分开分析。
5. 与 TesserAct 原始 PointNet-IDM baseline 的对比清晰，论文故事容易成立。

主要工程工作：

1. 实现 ActionImages-style heatmap label generation。
2. 构建 RGB-D-Normal / point cloud 到 action heatmap 的轻量 decoder。
3. 实现 multi-view heatmap 到 7-DoF action 的几何解码。
4. 与 direct 7-DoF regression、RGB-only decoder、TesserAct PointNet-IDM 做对比。

### 14.2 Persistent 3D memory 方向为什么更重

Persistent 3D memory 方向改动的是 world model / memory 层，而不是单纯后处理 decoder。即使不修改 TesserAct 主干，也需要处理多步 rollout、坐标对齐、3D map 更新、噪声累积和长时序评估：

```
TesserAct RGB-D-Normal chunks
    -> point cloud / voxel / TSDF / feature map memory update
    -> memory-aware action decoder
    -> long-horizon action prediction
```

资源压力主要来自：

1. 需要做 multi-step 或 chunk-level rollouts。
2. 需要设计 3D memory 表示，例如 accumulated point cloud、voxel grid、TSDF、3D Gaussian 或 3D feature map。
3. 需要处理 TesserAct depth / normal 预测误差在 memory 中的累积。
4. 需要找有遮挡、长时序、多阶段依赖的任务，否则 memory 的优势不明显。
5. 如果进一步把 memory 反向输入 TesserAct diffusion 模型，就会变成 world model 主干改造，训练成本会显著上升。

因此，资源紧张时不建议一开始做 memory-conditioned TesserAct generation。更稳妥的低算力版本是：**不改 TesserAct 主模型，只把 memory 作为 action decoder 的额外输入**。

## 15. 推荐优先级和实验路线

综合资源、创新清晰度和实验可控性，建议按以下顺序推进：

### 15.1 第一优先级：TesserAct + ActionImages-style decoder

这是当前最推荐的主线。

核心研究问题：

> RGB-D-Normal 或 4D geometry world predictions 能否被转换成 pixel-grounded action images，从而提供一个轻量、可解释的 action decoding bridge？
> 

推荐实验顺序：

1. **GT geometry -> action images**
    
    使用 ground-truth RGB-D-Normal / point cloud 训练 decoder，验证 action-image bottleneck 本身是否可行。
    
2. **TesserAct-predicted geometry -> action images**
    
    用 TesserAct 预测的 RGB-D-Normal / 4D scene 替换 GT 输入，测试 world prediction noise 下的鲁棒性。
    
3. **与 baselines 对比**
    
    对比 direct 7-DoF regression、RGB-only decoder、TesserAct 原始 PointNet-IDM。
    
4. **鲁棒性分析**
    
    加入 depth noise、normal noise、missing view、occlusion、blur 等扰动，分析 action heatmap 的可解释失败模式。
    

### 15.2 第二优先级：轻量 persistent 3D memory ablation

如果第一优先级实验已经证明 action-image decoder 有效，可以增加一个轻量 memory ablation，而不是完整改造 TesserAct：

```
current RGB-D-Normal / point cloud
    vs
current RGB-D-Normal / point cloud + accumulated 3D memory
```

可检验的问题：

> Persistent 3D memory 是否能在遮挡或长时序任务中提升 action decoding？
> 

最低成本设计：

1. 使用 ground-truth depth / point cloud 累积一个 **GT memory upper bound**。
2. 将 memory 编码后作为 action decoder 的额外输入。
3. 只在有遮挡或历史依赖的任务上测试。
4. 如果 GT memory 有明显收益，再尝试使用 TesserAct-predicted geometry 构建 predicted memory。

不建议一开始做：

1. 训练 memory-conditioned diffusion world model。
2. 修改 TesserAct 主干，让其直接读取 persistent 3D map。
3. 设计复杂的 neural 3D feature memory 并端到端训练。

### 15.3 最终建议

如果目标是尽快做出一个可验证、低算力、论文故事清晰的创新点，应优先推进：

```
TesserAct-style RGB-D-Normal / 4D future
    -> lightweight geometry-to-action-image decoder
    -> interpretable multiview action heatmaps
    -> geometric decoding
    -> 7-DoF robot action
```

Persistent 3D memory 可以作为后续增强：

```
current 4D future + accumulated 3D memory
    -> memory-aware action-image decoder
    -> better decoding under occlusion / long-horizon settings
```

这样可以保证主创新足够聚焦，同时保留向长时序 world model 扩展的空间。