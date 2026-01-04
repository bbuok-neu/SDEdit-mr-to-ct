# SDEdit库配置分析与MR-to-CT实现指南

## 目录
1. [配置文件结构分析](#1-配置文件结构分析)
2. [运行图像翻译所需组件](#2-运行图像翻译所需组件)
3. [MR-to-CT医学图像合成可行性分析](#3-mr-to-ct医学图像合成可行性分析)
4. [实现方案与步骤](#4-实现方案与步骤)

---

## 1. 配置文件结构分析

SDEdit使用YAML格式的配置文件，位于`configs/`目录下。每个配置文件包含四个主要部分：

### 1.1 数据配置 (`data`)

```yaml
data:
    dataset: "LSUN"              # 数据集名称（LSUN, CelebA_HQ等）
    category: "bedroom"          # 数据集类别
    image_size: 256              # 图像尺寸（必须是正方形）
    channels: 3                  # 图像通道数（RGB=3，灰度=1）
    logit_transform: false       # 是否应用logit变换
    uniform_dequantization: false # 均匀去量化
    gaussian_dequantization: false # 高斯去量化
    random_flip: true            # 训练时随机翻转
    rescaled: true               # 是否重新缩放到[-1,1]
    num_workers: 32              # 数据加载器工作进程数
```

### 1.2 模型配置 (`model`)

```yaml
model:
    type: "simple"               # 模型类型
    in_channels: 3               # 输入通道数
    out_ch: 3                    # 输出通道数
    ch: 128                      # 基础通道数
    ch_mult: [1, 1, 2, 2, 4, 4]  # 通道倍数（决定UNet深度）
    num_res_blocks: 2            # 每个分辨率的残差块数量
    attn_resolutions: [16, ]     # 应用注意力的分辨率
    dropout: 0.0                 # Dropout率
    var_type: fixedsmall         # 方差类型（fixedsmall或fixedlarge）
    ema_rate: 0.999              # EMA衰减率
    ema: True                    # 是否使用EMA
    resamp_with_conv: True       # 是否使用卷积进行重采样
```

### 1.3 扩散过程配置 (`diffusion`)

```yaml
diffusion:
    beta_schedule: linear        # Beta调度方式（linear, cosine等）
    beta_start: 0.0001           # Beta起始值
    beta_end: 0.02               # Beta结束值
    num_diffusion_timesteps: 1000 # 扩散时间步数
```

### 1.4 采样配置 (`sampling`)

```yaml
sampling:
    batch_size: 8                # 采样批次大小
    last_only: True              # 是否只保存最后结果
```

---

## 2. 运行图像翻译所需组件

### 2.1 必需组件

| 组件 | 描述 | 位置/格式 |
|------|------|-----------|
| 配置文件 | YAML格式的模型和数据配置 | `configs/*.yml` |
| 预训练模型 | 已训练的扩散模型权重 | `.ckpt`文件 |
| 输入数据 | 图像和掩码 | `.pth`格式 `[mask, img]` |
| Python环境 | PyTorch及相关依赖 | `requirements.txt` |

### 2.2 命令行参数

```bash
python main.py \
    --config bedroom.yml \     # 配置文件名
    --exp ./runs/ \            # 实验输出目录
    --sample \                 # 启用采样模式
    -i images \                # 输出图像文件夹名
    --npy_name lsun_bedroom1 \ # 输入数据文件名（不含.pth）
    --sample_step 3 \          # 采样步数（生成多少个样本）
    --t 500 \                  # 噪声水平（控制编辑程度）
    --ni                       # 非交互模式
```

### 2.3 参数说明

- **`--t`（噪声水平）**: 
  - 值越大，编辑程度越大，生成结果越多样化
  - 值越小，保留原始结构越多
  - 推荐范围：300-500

- **`--sample_step`**: 
  - 迭代采样次数
  - 每次迭代都会优化结果

### 2.4 输入数据格式

```python
import torch

# 加载数据
[mask, img] = torch.load("colab_demo/your_data.pth")

# mask: 二值掩码张量，标记编辑区域
# - mask == 1: 保持原始内容
# - mask != 1: 需要编辑/生成的区域

# img: 图像张量
# - 形状: [C, H, W]
# - 值范围: [0, 1]
```

### 2.5 Mask设置说明

对于MR-to-CT全图翻译任务，mask应设置为**全0**，表示整个图像都需要被转换：

```python
import torch

# 全图翻译：mask全为0
# 这意味着整个MR图像都将被转换为CT风格
mask = torch.zeros(1, 256, 256)  # 对于灰度图像，通道数为1
# 或
mask = torch.zeros(3, 256, 256)  # 对于RGB模型

# 如果只想转换部分区域，可以设置：
# mask = torch.ones(1, 256, 256)  # 默认保持原样
# mask[:, 50:200, 50:200] = 0     # 只有这个区域会被转换
```

**mask的工作原理**：
- `mask == 1` 的区域：保持原始MR图像内容不变
- `mask != 1` (通常为0) 的区域：使用扩散模型生成CT风格内容
- 对于完整的MR-to-CT转换，使用全0的mask

---

## 3. MR-to-CT医学图像合成可行性分析

### 3.1 SDEdit的工作原理

SDEdit通过以下步骤工作：
1. 向输入图像添加噪声（扩散前向过程）
2. 使用预训练的扩散模型去噪（反向过程）
3. 生成与输入结构相似但更真实的图像

### 3.2 零样本MR-to-CT的挑战

**结论：SDEdit本身不能直接实现零样本MR-to-CT转换**

原因：
1. SDEdit需要在**目标域**（CT）上预训练的扩散模型
2. 现有预训练模型是针对自然图像（CelebA、LSUN等）
3. MR和CT图像的统计特性与自然图像差异很大

### 3.3 实现MR-to-CT的可行方案

#### 方案A：训练CT域的扩散模型（推荐）

```
步骤1: 收集CT图像数据集
步骤2: 使用DDPM/DDIM训练CT域的扩散模型
步骤3: 将MR图像作为输入，使用SDEdit进行转换
```

优点：
- 可以保持MR图像的结构信息
- 生成的CT图像更真实

#### 方案B：使用配对数据训练条件扩散模型

```
步骤1: 收集配对的MR-CT数据
步骤2: 训练条件扩散模型（以MR为条件生成CT）
步骤3: 进行推理转换
```

优点：
- 更精确的转换
- 不需要SDEdit的编辑框架

---

## 4. 实现方案与步骤

### 4.1 步骤一：准备CT数据集

建议的数据集：
- [AAPM Low Dose CT Grand Challenge](https://www.aapm.org/grandchallenge/lowdosect/)
- [CT Colonography Dataset](https://wiki.cancerimagingarchive.net/display/Public/CT+COLONOGRAPHY)
- [COVID-19 CT scans](https://github.com/UCSD-AI4H/COVID-CT)

数据预处理：
```python
# 将CT图像处理为256x256
# 归一化到[0, 1]范围
# 保存为PyTorch格式
```

### 4.2 步骤二：训练CT扩散模型

参考DDIM仓库进行训练：
https://github.com/ermongroup/ddim

训练命令示例：
```bash
python main.py --config ct_medical.yml --exp ct_experiment --doc ct_model
```

### 4.3 步骤三：使用SDEdit进行MR-to-CT转换

准备MR输入：
```python
import torch

# 加载MR图像
mr_image = load_and_preprocess_mr_image("path/to/mr.nii")

# 创建掩码（全部区域都需要转换）
mask = torch.zeros(3, 256, 256)  # 全为0表示所有区域都编辑

# 保存
torch.save([mask, mr_image], "colab_demo/mr_input.pth")
```

运行SDEdit：
```bash
python main.py \
    --config ct_medical.yml \
    --exp ./runs/ \
    --sample \
    -i ct_output \
    --npy_name mr_input \
    --sample_step 3 \
    --t 400 \
    --ni
```

### 4.4 配置文件模板

医学图像的推荐配置 (`configs/ct_medical.yml`)：

```yaml
data:
    dataset: "CT_Medical"
    category: "abdomen"
    image_size: 256
    channels: 1           # 医学图像通常是灰度的
    logit_transform: false
    uniform_dequantization: false
    gaussian_dequantization: false
    random_flip: false    # 医学图像通常不翻转
    rescaled: true
    num_workers: 8

model:
    type: "simple"
    in_channels: 1        # 灰度图像
    out_ch: 1
    ch: 128
    ch_mult: [1, 1, 2, 2, 4, 4]
    num_res_blocks: 2
    attn_resolutions: [16, ]
    dropout: 0.1          # 适当的dropout
    var_type: fixedsmall
    ema_rate: 0.9999
    ema: True
    resamp_with_conv: True

diffusion:
    beta_schedule: linear
    beta_start: 0.0001
    beta_end: 0.02
    num_diffusion_timesteps: 1000

sampling:
    batch_size: 4         # 医学图像可能需要更多内存
    last_only: True
```

---

## 5. DDIM训练配置与SDEdit兼容性

### 5.1 使用DDIM训练无条件生成模型

要使DDIM训练的checkpoint能被SDEdit直接使用，需要确保配置一致。

#### DDIM训练配置示例 (for CT)

在DDIM仓库中创建 `ct_medical.yml`：

```yaml
data:
    dataset: "CUSTOM"
    image_size: 256
    channels: 1               # 灰度医学图像
    logit_transform: false
    uniform_dequantization: false
    gaussian_dequantization: false
    random_flip: false        # 医学图像不翻转
    rescaled: true

model:
    type: "simple"
    in_channels: 1            # 必须与data.channels一致
    out_ch: 1
    ch: 128
    ch_mult: [1, 1, 2, 2, 4, 4]
    num_res_blocks: 2
    attn_resolutions: [16]
    dropout: 0.1
    var_type: fixedsmall
    ema_rate: 0.9999
    ema: true
    resamp_with_conv: true

diffusion:
    beta_schedule: linear     # 必须与SDEdit配置一致
    beta_start: 0.0001
    beta_end: 0.02
    num_diffusion_timesteps: 1000

training:
    batch_size: 8
    n_epochs: 100
    n_iters: 400000
    snapshot_freq: 10000
    log_freq: 100
    val_freq: 1000

sampling:
    batch_size: 4
    last_only: true

optim:
    weight_decay: 0.0
    optimizer: "Adam"
    lr: 0.0002
    beta1: 0.9
    amsgrad: false
    eps: 0.00000001
```

### 5.2 关键兼容性要求

确保SDEdit能加载DDIM训练的checkpoint，需要满足以下条件：

| 参数 | DDIM训练配置 | SDEdit配置 | 说明 |
|------|-------------|-----------|------|
| `model.in_channels` | 1 | 1 | 必须完全一致 |
| `model.out_ch` | 1 | 1 | 必须完全一致 |
| `model.ch` | 128 | 128 | 必须完全一致 |
| `model.ch_mult` | [1,1,2,2,4,4] | [1,1,2,2,4,4] | 必须完全一致 |
| `model.num_res_blocks` | 2 | 2 | 必须完全一致 |
| `model.attn_resolutions` | [16] | [16] | 必须完全一致 |
| `diffusion.beta_schedule` | linear | linear | 推荐保持一致 |
| `diffusion.num_diffusion_timesteps` | 1000 | 1000 | 推荐保持一致 |
| `data.image_size` | 256 | 256 | 必须完全一致 |

### 5.3 Checkpoint格式说明

DDIM保存的checkpoint通常有多种格式，SDEdit现在支持所有常见格式：

```python
# 格式1：直接的state_dict（键直接是层名称）
ckpt = model.state_dict()

# 格式2：包含'model'键
ckpt = {
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'epoch': epoch
}

# 格式3：包含'ema'键（DDIM默认使用EMA权重）
ckpt = {
    'ema': ema_helper.state_dict(),
    'model': model.state_dict(),
}

# 格式4：包含'state_dict'键
ckpt = {
    'state_dict': model.state_dict()
}
```

SDEdit会自动检测并加载正确的格式，按以下顺序尝试：
1. `ckpt['model']`
2. `ckpt['state_dict']`
3. `ckpt['ema']`
4. `ckpt['model_state_dict']`
5. 直接使用 `ckpt`（如果是state_dict格式）

还会自动处理DataParallel的 `module.` 前缀问题。

### 5.4 训练命令示例

```bash
# 在DDIM仓库中训练
cd ddim
python main.py --config ct_medical.yml --exp experiments/ct --doc ct_model

# 训练完成后，checkpoint保存在：
# experiments/ct/logs/ct_model/ckpt_*.pth
```

### 5.5 使用训练好的模型

**单个文件处理：**
```bash
python main.py \
    --config ct_medical.yml \
    --exp ./runs/ \
    --sample \
    -i ct_output \
    --npy_name /path/to/mr_input.pth \
    --ckpt /path/to/ddim/experiments/ct/logs/ct_model/ckpt_400000.pth \
    --sample_step 3 \
    --t 400 \
    --ni
```

**批量处理整个目录：**
```bash
# --npy_name 可以是一个包含多个.pth文件的目录
python main.py \
    --config ct_medical.yml \
    --exp ./runs/ \
    --sample \
    -i ct_output \
    --npy_name /path/to/sdedit_inputs \
    --ckpt /path/to/ddim/experiments/ct/logs/ct_model/ckpt_400000.pth \
    --sample_step 3 \
    --t 400 \
    --ni
```

输出文件会以输入文件名为前缀，例如：
- `image1_original_input.png`
- `image1_samples_0.png`
- `image2_original_input.png`
- `image2_samples_0.png`

---

## 6. 注意事项

### 6.1 医学图像处理建议

1. **灰度图像处理**：将配置中的`channels`和`in_channels`设为1
2. **不进行翻转增强**：医学图像的方向有意义
3. **适当的归一化**：使用合适的窗宽窗位

### 6.2 训练资源需求

- GPU: 推荐NVIDIA V100或A100（至少16GB显存）
- 训练时间: 约1-3天（取决于数据集大小）
- 存储: 至少50GB用于数据集和检查点

### 6.3 推荐的替代方案

如果不想从头训练，可以考虑：

1. **使用现有医学图像扩散模型**
   - [MedSegDiff](https://github.com/WuJunde/MedSegDiff)
   - [Medical Diffusion](https://github.com/FirasGit/medicaldiffusion)

2. **使用其他图像翻译方法**
   - CycleGAN
   - Pix2Pix（如果有配对数据）
   - UNIT

---

## 7. 总结

| 问题 | 回答 |
|------|------|
| SDEdit能否实现零样本MR-to-CT？ | ❌ 不能直接实现 |
| 是否需要重新训练模型？ | ✅ 需要在CT数据上训练扩散模型 |
| SDEdit框架是否适合医学图像？ | ✅ 适合，但需要修改配置和训练新模型 |
| DDIM训练的模型能用吗？ | ✅ 可以，确保配置参数一致即可 |
| 推荐方案 | 在CT数据集上训练扩散模型，然后使用SDEdit进行转换 |
