# FACTR Vision Model 输入输出与流程说明

## 1. 模型定位

本仓库里的 `agent: transformer_vit` 并不是通用意义上会生成文本的 Vision-Language Model，而是一个用于机器人模仿学习的视觉策略模型：

- 视觉部分：ViT 图像编码器，定义在 `factr/models/vit.py`
- 策略部分：ACT 风格的 Transformer action decoder，定义在 `factr/models/action_transformer.py`
- 输入：相机图像和低维机器人状态
- 输出：未来一段时间的连续动作序列

因此，更准确的叫法是 **视觉-状态到动作的 Transformer policy**。

## 2. 主要配置入口

默认训练配置在 `factr/cfg/train_bc.yaml`：

```yaml
defaults:
  - agent: transformer_vit
  - task: franka

ac_chunk: 100
img_chunk: 1
```

模型结构配置在 `factr/cfg/agent/transformer_vit.yaml`：

```yaml
_target_: factr.models.action_transformer.TransformerAgent
odim: ${task.obs_dim}
n_cams: ${task.n_cams}
use_obs: add_token
ac_dim: ${task.ac_dim}
ac_chunk: ${ac_chunk}
imgs_per_cam: ${img_chunk}
early_fusion: True
feat_norm: layer_norm
token_dim: 512
```

ViT 视觉编码器配置在 `factr/cfg/agent/features/vit_base.yaml`：

```yaml
_target_: factr.models.vit.load_vit
model:
  _target_: factr.models.vit.vit_base_patch16
  img_size: 224
  use_cls: True
```

任务维度示例在 `factr/cfg/task/single_franka.yaml`：

```yaml
obs_dim: 7
ac_dim: 7
cam_indexes: [0]
n_cams: ${len:${task.cam_indexes}}
```

注意：`task.ac_dim` 必须和处理后数据里的动作维度一致。比如 `process_data/cfg/default.yaml` 中的多个 `action_config` 会被拼接成最终 action，训练配置要与该拼接结果对齐。

## 3. 输入

### 3.1 图像输入 `imgs`

训练和推理时，模型接收一个按相机编号组织的字典：

```python
imgs = {
    "cam0": Tensor,
    "cam1": Tensor,
    ...
}
```

经过 `DataLoader` 后，每个相机张量通常是：

```text
[B, T, C, H, W]
```

含义：

| 维度 | 含义 |
| --- | --- |
| `B` | batch size |
| `T` | 图像时间步数量，对应 `img_chunk` |
| `C` | 图像通道数，通常为 3 |
| `H, W` | 图像尺寸，ViT 默认使用 224 x 224 |

图像来自 `RobobufReplayBuffer.__getitem__()`：

1. 根据 `cam_indexes` 读取 robobuf 中的相机图像。
2. `_get_imgs(step, cam_idx, past_frames)` 取当前帧和若干历史帧。
3. `_img_to_tensor()` 将图像从 numpy 转成 PyTorch tensor，并把格式从 `[T, H, W, C]` 转成 `[T, C, H, W]`。
4. transform 将图像 resize/augment/normalize 到 ViT 需要的格式。

默认预处理使用 ImageNet 归一化：

```python
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
```

### 3.2 低维状态输入 `obs`

低维状态输入是一个 tensor：

```text
obs: [B, obs_dim]
```

在 `RobobufReplayBuffer.__getitem__()` 中来自：

```python
o_t = step.obs.state
```

默认示例 `single_franka.yaml` 中：

```yaml
obs_dim: 7
```

根据 `process_data/cfg/default.yaml` 的示例，低维状态可能由以下 topic 拼接得到：

```yaml
obs_topics:
- /franka/right/obs_franka_torque
- /gripper/right/obs_gripper_pos
```

实际含义取决于数据处理配置和机器人数据记录方式。

### 3.3 训练标签 `actions`

训练时还需要监督动作：

```text
actions: [B, ac_chunk, ac_dim]
```

默认：

```yaml
ac_chunk: 100
```

含义是：模型不是只预测当前一步动作，而是预测从当前时刻开始的未来 `ac_chunk` 步动作序列。

在 `RobobufReplayBuffer` 中，动作构造逻辑是：

1. 从当前 step 开始，沿着 `next` 指针向后取动作。
2. 最多取 `ac_chunk` 个动作。
3. 如果轨迹到末尾不足 `ac_chunk`，用最后一个动作补齐。
4. 同时生成 `loss_mask`，真实存在的未来动作位置为 1，补齐位置为 0。

训练时还会把动作展平成：

```text
ac_flat: [B, ac_chunk * ac_dim]
mask_flat: [B, ac_chunk * ac_dim]
```

## 4. 输出

### 4.1 推理输出

推理调用：

```python
pred_actions = agent.get_actions(imgs, obs)
```

输出 shape：

```text
pred_actions: [B, ac_chunk, ac_dim]
```

含义：

| 维度 | 含义 |
| --- | --- |
| `B` | batch size |
| `ac_chunk` | 预测未来动作步数 |
| `ac_dim` | 每步动作维度 |

如果使用默认示例配置：

```text
pred_actions: [B, 100, 7]
```

每个 `pred_actions[b, t]` 是第 `b` 个样本在未来第 `t` 个动作步的连续控制命令。

### 4.2 训练输出

训练时 `TransformerAgent.forward()` 返回的是 loss：

```python
l1 = (F.l1_loss(ac_flat_hat, ac_flat, reduction="none") * mask_flat).mean()
```

也就是预测动作和真实动作之间的 masked L1 loss。

## 5. 从输入到输出的完整流程

### 5.1 数据读取流程

```text
robobuf trajectory
    |
    |-- step.obs.image(cam_idx)
    |       -> imgs["cam{i}"]: [T, C, H, W]
    |
    |-- step.obs.state
    |       -> obs: [obs_dim]
    |
    |-- step.action + next steps
            -> actions: [ac_chunk, ac_dim]
            -> loss_mask: [ac_chunk, ac_dim]
```

经过 `DataLoader` batch 后：

```text
imgs["cam{i}"]: [B, T, C, H, W]
obs:           [B, obs_dim]
actions:       [B, ac_chunk, ac_dim]
loss_mask:     [B, ac_chunk, ac_dim]
```

### 5.2 图像编码流程

代码入口：

```python
tokens = self.tokenize_obs(imgs, obs)
```

内部首先调用：

```python
tokens = self.embed(imgs)
```

对于每个相机：

1. 如果 `early_fusion=True` 且输入是 `[B, T, C, H, W]`，会把多个时间步沿通道维拼接：

   ```text
   [B, T, C, H, W] -> [B, T*C, H, W]
   ```

   默认 `img_chunk=1`，所以仍然等价于 `[B, 3, H, W]`。

2. ViT 将图像切成 patch，并取 cls token 作为图像表示。

3. 因为 `use_cls=True`，每张图像/每个相机输出 1 个 token。

ViT base 的内部 embedding 维度是 768，之后会投影到 `token_dim=512`：

```text
image tokens: [B, n_image_tokens, 768] -> [B, n_image_tokens, 512]
```

其中：

```text
n_image_tokens = n_cams * imgs_per_cam * features.n_tokens
```

默认 `n_cams=1`、`img_chunk=1`、`features.n_tokens=1`，所以图像 token 数为 1。

### 5.3 FACTR curriculum 处理

`BaseAgent.tokenize_obs()` 里会根据 `curriculum` 对视觉信息做课程扰动。

默认配置：

```yaml
curriculum:
  space: pixel
  operator: blur
  scheduler: linear
  start_scale: 5
  stop_scale: 0
```

如果 `space: pixel`：

```text
原始图像 -> gaussian blur 或 downsample -> ViT
```

如果 `space: latent`：

```text
ViT token -> gaussian_1d_smoothing 或 downsample_1d
```

训练早期扰动较强，随着 step 增加逐渐减弱到 `stop_scale`。如果 scheduler 是 `no`，则 scale 为 0，不做扰动。

### 5.4 状态 token 拼接

默认：

```yaml
use_obs: add_token
```

因此低维状态会经过一个线性层变成一个 token：

```text
obs:       [B, obs_dim]
obs token: [B, 1, token_dim]
```

然后拼接到图像 token 后：

```text
all tokens: [B, n_image_tokens + 1, token_dim]
```

默认情况下：

```text
all tokens: [B, 2, 512]
```

分别是：

1. `cam0` 的视觉 token
2. 低维状态 token

之后经过 `LayerNorm` 和 dropout。

### 5.5 Transformer action decoder

代码入口：

```python
action_tokens = self.transformer(tokens, self.ac_query.weight)
```

Transformer 包含：

- encoder：对输入 tokens 做 self-attention 编码
- decoder：用 action query 从 encoder memory 中解码未来动作 token

动作 query 来自：

```python
self.ac_query = nn.Embedding(ac_chunk, d_model)
```

也就是说，每一个未来动作时间步都有一个可学习 query：

```text
ac_query.weight: [ac_chunk, d_model]
```

默认：

```text
ac_query.weight: [100, 512]
```

流程：

```text
input tokens: [B, n_tokens, 512]
    |
    | transpose
    v
[n_tokens, B, 512]
    |
    | Transformer encoder
    v
memory: [n_tokens, B, 512]
    |
    | Transformer decoder with action queries
    v
action tokens: [B, ac_chunk, 512]
```

### 5.6 动作投影

最后一层线性投影：

```python
self.ac_proj = nn.Linear(d_model, ac_dim)
actions = self.ac_proj(action_tokens)
```

得到最终动作：

```text
actions: [B, ac_chunk, ac_dim]
```

默认：

```text
actions: [B, 100, 7]
```

## 6. 端到端流程图

```text
相机图像 imgs["cam{i}"]          低维状态 obs
        |                           |
        | transform/normalize        | Linear
        v                           v
   ViT image encoder             obs token
        |                           |
        +----------- concat --------+
                    |
                    v
          visual/state tokens
          [B, n_tokens, 512]
                    |
                    v
          Transformer encoder
                    |
                    v
              encoded memory
                    |
       action queries [ac_chunk, 512]
                    |
                    v
          Transformer decoder
                    |
                    v
          action tokens [B, ac_chunk, 512]
                    |
                    v
          Linear projection to ac_dim
                    |
                    v
          predicted actions
          [B, ac_chunk, ac_dim]
```

## 7. 训练与推理的差异

| 阶段 | 输入 | 输出 | 额外处理 |
| --- | --- | --- | --- |
| 训练 | `imgs`, `obs`, `actions`, `mask` | masked L1 loss | 使用真实未来动作监督 |
| 推理 | `imgs`, `obs` | `pred_actions` | 不需要真实动作和 mask |

训练代码路径：

```text
BCTask / DataLoader
    -> trainer.training_step()
    -> TransformerAgent.forward(imgs, obs, ac_flat, mask_flat)
    -> loss
```

推理代码路径：

```text
agent.get_actions(imgs, obs)
    -> tokenize_obs()
    -> transformer()
    -> ac_proj()
    -> pred_actions
```

## 8. 快速 mock 示例

仓库中已有 `scripts/mock_inference.py`，可以构造随机输入跑通一次前向过程：

```bash
python scripts/mock_inference.py --batch-size 2 --obs-dim 7 --action-dim 7 --action-chunk 100 --n-cams 1 --image-chunk 1
```

预期关键 shape：

```text
image[cam0] shape: (2, 1, 3, 224, 224)
obs shape:         (2, 7)
token shape:       (2, 2, 512)
actions shape:     (2, 100, 7)
```

