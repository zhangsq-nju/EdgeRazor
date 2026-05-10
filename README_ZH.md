<div align="center">
  <br/>
  <img src="https://raw.githubusercontent.com/zhangsq-nju/EdgeRazor/main/asset/Logo-full.png" alt="EdgeRazor Logo" width="60%">
  <h3>
    端侧 AI 的轻量化框架
  </h3>

  <p>
    <a href="https://arxiv.org/abs/2605.04062" target="_blank">
      <img src="https://img.shields.io/badge/arXiv-EdgeRazor-b31b1b?style=flat&logo=arxiv" alt="arXiv EdgeRazor">
    </a>
    <a href="https://huggingface.co/spaces/zhangsq-nju/EdgeRazor-Playground" target="_blank">
      <img src="https://img.shields.io/badge/HF-Playground-FFD21F?style=flat&logo=huggingface&logoColor=FFD21F" alt="Hugging Face Space">
    </a>
    <a href="https://huggingface.co/collections/zhangsq-nju/edgerazor-nbit" target="_blank">
      <img src="https://img.shields.io/badge/HF-Collection-FFD21F?style=flat&logo=huggingface&logoColor=FFD21F" alt="Hugging Face Collection">
    </a>
    <a href="https://github.com/zhangsq-nju/EdgeRazor/blob/main/README_ZH.md" target="_blank">
      <img src="https://img.shields.io/badge/README-ZH-blue?style=flat&logo=readme" alt="README ZH">
    </a>
    <a href="https://github.com/zhangsq-nju/EdgeRazor/blob/main/LICENSE">
      <img src="https://img.shields.io/badge/License-Apache_2.0-green?logo=opensourceinitiative&logoColor=green" alt="License: Apache 2.0">
    </a>
  </p>
  
  <h5>
    ✨ 如果您喜欢我们的项目，请给我们一个星 ⭐️ 以支持最新更新。
  </h5>

  <img src="https://raw.githubusercontent.com/zhangsq-nju/EdgeRazor/main/asset/Demo.gif" alt="EdgeRazor Demo" width="75%">
</div>

---

**EdgeRazor** 是一个针对端侧 AI 的轻量化框架，旨在训练出更小、更快，并可部署于多样化硬件平台的模型，覆盖从端侧设备到对时延敏感的云端场景。EdgeRazor 框架能够以**最小代码改动**将模型压缩技术**无缝集成**到现有全精度训练流程中，在保持优良任务性能的同时，实现低成本与高效率计算。

EdgeRazor 当前聚焦于通过可配置的量化感知蒸馏实现低比特 LLM 压缩。在**量化**方面，EdgeRazor 支持对权重（包括 embedding 与 lm_head 层）、激活值以及 KV cache 进行量化。量化位宽不仅包含统一的 1.58-bit 与 4-bit，还支持矩阵维度的混合精度配置，例如 2.79-bit（50% 4-bit + 50% 1.58-bit）和 1.88-bit（12.5% 4-bit + 87.5% 1.58-bit）。在**蒸馏**方面，EdgeRazor 提供 logits、特征与注意力蒸馏，并可在统一配置接口下灵活组合。

EdgeRazor 在多类模型上均取得了当前最先进表现，涵盖基础大模型、指令微调大模型与多模态大模型。以 W-A8-KV8 量化为例，**Qwen3-0.6B-EdgeRazor** 在 4-bit / 2.79-bit / 1.88-bit / 1.58-bit 下的平均分分别达到 **47.80** / **44.10** / **41.76** / **39.81**，对应压缩倍率分别为 **3.94×** / **5.05×** / **6.40×** / **7.03×**。相比之下，现有最佳方法在 4-bit / 3-bit / 2-bit 下仅达到 <u>45.74</u> / <u>37.38</u> / <u>30.49</u>，对应压缩倍率为 <u>2.21×</u> / <u>2.47×</u> / <u>2.78×</u>。

<p align="center">
  <img src="https://raw.githubusercontent.com/zhangsq-nju/EdgeRazor/main/asset/Architeacture.png" alt="EdgeRazor Architecture">
  <br> EdgeRazor 框架用于轻量化模型训练的流程图
</p>

## 最新消息

- 🔥 **[2026-04]**: 📄 论文-EdgeRazor 已在 arXiv 上发布：[2605.04062](https://arxiv.org/abs/2605.04062)！
- 🔥 **[2026-04]**: 🏅 [CCF 算法能力大赛决赛](https://cacc.ccf.org.cn/#/tzgg/%E9%80%9A%E7%9F%A5%E5%85%AC%E5%91%8A/6ce6fd51cffa62eb3859a8bb80af1040)在 AI 赛道使用 EdgRazor作为题目的工具库！人机共舞，智创未来！
- 🔥 **[2026-04]**: 🏆 由 EdgeRazor 框架训练的低比特大模型已在 Hugging Face 上发布！查看我们的 Hugging Face 模型集：[zhangsq-nju/edgerazor-nbit](https://huggingface.co/collections/zhangsq-nju/edgerazor-nbit)。
- 🔥 **[2026-04]**: 🛠️ 开源 EdgeRazor-V1 发布！现在可在各种模型的训练流程上无缝集成和定制！
- 🔥 **[2025-10]**: 📄 论文-TernaryCLIP 已在 arXiv 上发布：[https://arxiv.org/abs/2510.21879](https://arxiv.org/abs/2510.21879)！

## Contents

- [最新消息](#最新消息)
- [Contents](#contents)
- [上手指南](#上手指南)
  - [安装](#安装)
  - [用法](#用法)
  - [在 Docker 上部署低比特大模型](#在-docker-上部署低比特大模型)
  - [在 Playground 上开始](#在-playground-上开始)
- [主要技术](#主要技术)
- [实际应用](#实际应用)
- [模型列表](#模型列表)
  - [语言大模型](#语言大模型)
  - [多模态大模型](#多模态大模型)
- [待办事项](#待办事项)
- [引用](#引用)
- [贡献者列表](#贡献者列表)

## 上手指南

### 安装

- 从 PyPi 下载

```bash
pip install edgerazor
```

- 从 GitHub 下载（最新版本）

```bash
git clone https://github.com/zhangsq-nju/EdgeRazor.git && cd EdgeRazor
conda create -n edgerazor python=3.10.20 -y
conda activate edgerazor
pip install -e .
```

### 用法

安装完成后，您可以将 EdgeRazor 集成到现有训练流程中，构建轻量化模型。

1. 使用统一配置接口，可通过 [yaml](./example/configs/qad/qat_w4_a8_kd_fd.yaml)、[json](./example/configs/qad/qat_w4_a8_kd_fd.json) 或 [dict](./example/configs/qad/qat_w4_a8_kd_fd.py) 进行配置。

2. 将 EdgeRazor 无缝接入您的全精度模型训练流程，即刻开启轻量化之旅！

```python
# 初始化 EdgeRazor 以构建轻量化模型
edgerazor = EdgeRazor(config="/path/to/config.yaml")
student = edgerazor.quantize(student)
# 训练循环
student_outputs = student(inputs)
teacher_outputs = teacher(inputs)
# 计算损失
loss, loss_dict = edgerazor.compute_loss(student_outputs, teacher_outputs, labels)
```

### 在 Docker 上部署低比特大模型

您可以基于 EdgeRazor 训练得到的权重转化为轻量化模型。例如，可将 Qwen3-EdgeRazor-4bit 权重转换为 Q4_0 GGUF 格式。我们也在 [collection](https://huggingface.co/collections/zhangsq-nju/edgerazor-nbit) 中提供了开箱即用的量化模型，包括 [Qwen3-0.6B-EdgeRazor-GGUF](https://huggingface.co/zhangsq-nju/Qwen3-0.6B-EdgeRazor-GGUF) 和 [Qwen3-1.7B-EdgeRazor-GGUF](https://huggingface.co/zhangsq-nju/Qwen3-1.7B-EdgeRazor-GGUF)。

```bash
# 仅使用 CPU 来进行量化大模型的部署：
docker pull ghcr.io/ggml-org/llama.cpp:server
hf download zhangsq-nju/Qwen3-1.7B-EdgeRazor-GGUF Qwen3-1.7B-EdgeRazor-TQ2_0.gguf --local-dir /path/to/Qwen3-1.7B-EdgeRazor-GGUF
cd ./docker && bash local_server_tq2_0.sh
```

### 在 Playground 上开始

上线 CPU 友好的 EdgeRazor Playground！在端侧设备上尽情享受低比特大模型的魅力！

```bash
cd EdgeRazor/playground
pip install -r requirements.txt
python app.py
```

![EdgeRazor Playground Sreenshot](https://raw.githubusercontent.com/zhangsq-nju/EdgeRazor/main/asset/Playground.png)

## 主要技术

量化感知蒸馏 (Quantization-Aware Distillation, QAD)：

- 针对模型权重：可配置的矩阵维度的混合精度量化训练
- 蒸馏训练过程：可配置的从 16-bit 到 $n$-bit 模型的蒸馏
- 蒸馏训练方法：
  - 自适应的特征蒸馏 (Adaptive Feature Distillation, AFD)
  - 熵感知的 KL 散度 (Entropy-Aware KL Divergence, EAKLD)

![EdgeRazor Framework Overview](https://raw.githubusercontent.com/zhangsq-nju/EdgeRazor/main/asset/EdgeRazor-Framework.png)

## 实际应用

- 轻量化 ViT-S/16，[点击这里查看]((./example/vit/README.md))。
- 轻量化 ResNet-18，[点击这里查看](./example/resnet/README.md)。
- 轻量化 Qwen3-0.6B/1.7B，[点击这里查看](./example/edgerazor-llm/README.md)。
- 轻量化 MobileLLM-ParetoQ-350M-BF16，[点击这里查看](./example/edgerazor-llm/README.md)。
- 轻量化 Qwen2.5-Omni-7B，[点击这里查看](./example/qwen2.5-omni/README.md)。

## 模型列表

### 语言大模型

- 平均性能（Avg.）：使用 [lm-eval v0.4.9.1](https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.9.1) 及对应 [tasks](./src/eval/tasks/lm_eval/) 在多项任务上评测得到的平均分。
  - 指令大模型的评测列表：arc_easy, arc_challenge, hellaswag, boolq, social_iqa, openbookqa, piqa, winogrande, hendrycks_ethics, truthfulqa_mc2, mmlu, gsm8k, humaneval_instruct, ifeval。
  - 基础大模型的评测列表：arc_easy, arc_challenge, hellaswag, boolq, social_iqa, openbookqa, piqa, winogrande, hendrycks_ethics, truthfulqa_mc2, mmlu, gsm8k, humaneval。
  - 除了 gsm8k 是 5-shot 之外，其余的任务都采用 0-shot。

- Hub Link：我们提供原始量化后的权重；在兼容的情况下，也会将其转换为 GGUF（[llama.cpp](https://github.com/ggml-org/llama.cpp)）与 GPTQ（[GPTQModel](https://github.com/ModelCloud/GPTQModel)，推进中）格式。

| Model          | W-A-KV       | Group Size | Avg.  | Hub Link                                                                                                                                                                                                                                                                                                             |
| -------------- | ------------ | ---------- | ----- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Qwen3-0.6B     | W16-A16-KV16 | -          | 47.35 | [Base](https://huggingface.co/Qwen/Qwen3-0.6B)                                                                                                                                                                                                                                                                       |
| Qwen3-0.6B     | W4-A8-KV8    | 256        | 47.80 | [EdgeRazor](https://huggingface.co/zhangsq-nju/Qwen3-0.6B-EdgeRazor-4bit), [Q4_0](https://huggingface.co/zhangsq-nju/Qwen3-0.6B-EdgeRazor-GGUF/resolve/main/Qwen3-0.6B-EdgeRazor-Q4_0.gguf)                                                                                                                     |
| Qwen3-0.6B     | W2.79-A8-KV8 | 256        | 44.10 | [EdgeRazor](https://huggingface.co/zhangsq-nju/Qwen3-0.6B-EdgeRazor-2.79bit)                                                                                                                                                                                                                                         |
| Qwen3-0.6B     | W1.88-A8-KV8 | 256        | 41.76 | [EdgeRazor](https://huggingface.co/zhangsq-nju/Qwen3-0.6B-EdgeRazor-1.88bit)                                                                                                                                                                                                                                         |
| Qwen3-0.6B     | W1.58-A8-KV8 | 256        | 39.81 | [EdgeRazor](https://huggingface.co/zhangsq-nju/Qwen3-0.6B-EdgeRazor-1.58bit), [TQ1_0](https://huggingface.co/zhangsq-nju/Qwen3-0.6B-EdgeRazor-GGUF/resolve/main/Qwen3-0.6B-EdgeRazor-TQ1_0.gguf), [TQ2_0](https://huggingface.co/zhangsq-nju/Qwen3-0.6B-EdgeRazor-GGUF/resolve/main/Qwen3-0.6B-EdgeRazor-TQ2_0.gguf) |
| Qwen3-1.7B     | W16-A16-KV16 | -          | 58.65 | [Base](https://huggingface.co/Qwen/Qwen3-1.7B)                                                                                                                                                                                                                                                                       |
| Qwen3-1.7B     | W4-A8-KV8    | 256        | 58.57 | [EdgeRazor](https://huggingface.co/zhangsq-nju/Qwen3-1.7B-EdgeRazor-4bit), [Q4_0](https://huggingface.co/zhangsq-nju/Qwen3-1.7B-EdgeRazor-GGUF/resolve/main/Qwen3-1.7B-EdgeRazor-Q4_0.gguf)                                                                                                                          |
| Qwen3-1.7B     | W2.79-A8-KV8 | 256        | 53.00 | [EdgeRazor](https://huggingface.co/zhangsq-nju/Qwen3-1.7B-EdgeRazor-2.79bit)                                                                                                                                                                                                                                         |
| Qwen3-1.7B     | W1.88-A8-KV8 | 256        | 47.14 | [EdgeRazor](https://huggingface.co/zhangsq-nju/Qwen3-1.7B-EdgeRazor-1.88bit)                                                                                                                                                                                                                                         |
| Qwen3-1.7B     | W1.58-A8-KV8 | 256        | 43.91 | [EdgeRazor](https://huggingface.co/zhangsq-nju/Qwen3-1.7B-EdgeRazor-1.58bit), [TQ1_0](https://huggingface.co/zhangsq-nju/Qwen3-1.7B-EdgeRazor-GGUF/resolve/main/Qwen3-1.7B-EdgeRazor-TQ1_0.gguf), [TQ2_0](https://huggingface.co/zhangsq-nju/Qwen3-1.7B-EdgeRazor-GGUF/resolve/main/Qwen3-1.7B-EdgeRazor-TQ2_0.gguf) |
| MobileLLM-350M | W16-A16-KV16 | -          | 41.18 | [Base](https://huggingface.co/facebook/MobileLLM-ParetoQ-350M-BF16)                                                                                                                                                                                                                                                  |
| MobileLLM-350M | W4-A8-KV8    | 64         | 41.86 | [EdgeRazor](https://huggingface.co/zhangsq-nju/MobileLLM-350M-EdgeRazor-4bit)                                                                                                                                                                                                                                        |
| MobileLLM-350M | W2.79-A8-KV8 | 64         | 40.62 | [EdgeRazor](https://huggingface.co/zhangsq-nju/MobileLLM-350M-EdgeRazor-2.79bit)                                                                                                                                                                                                                                     |
| MobileLLM-350M | W1.88-A8-KV8 | 64         | 39.32 | [EdgeRazor](https://huggingface.co/zhangsq-nju/MobileLLM-350M-EdgeRazor-1.88bit)                                                                                                                                                                                                                                     |
| MobileLLM-350M | W1.58-A8-KV8 | 64         | 38.12 | [EdgeRazor](https://huggingface.co/zhangsq-nju/MobileLLM-350M-EdgeRazor-1.58bit)                                                                                                                                                                                                                                     |

### 多模态大模型

- Video-MME 与 MLVU 为视频理解任务，使用 [lmms-eval v0.5.0](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/v0.5) 及对应 [tasks](./src/eval/tasks/lmms-eval/) 进行评测。

| Model           | W-A-KV       | Group Size | Video-MME | MLVU  | Hub Link                                                                       |
| --------------- | ------------ | ---------- | --------- | ----- | ------------------------------------------------------------------------------ |
| Qwen2.5-Omni-7B | W16-A16-KV16 | -          | 62.81     | 48.01 | [Base](https://huggingface.co/Qwen/Qwen2.5-Omni-7B)                            |
| Qwen2.5-Omni-7B | W4-A16-KV16  | 32         | 62.22     | 48.82 | [EdgeRazor](https://huggingface.co/zhangsq-nju/Qwen2.5-Omni-7B-EdgeRazor-4bit) |

## 待办事项

EdgeRazor 正在持续不断发展！以下是即将推出的内容：

- [x] 一系列轻量化大模型的训练代码
- [ ] 升级针对最新依赖库的支持

有任何想法或建议吗？我们欢迎任何贡献和合作！请随时提交 issues 或 pull requests！🚀

## 引用

如果您觉得我们的论文和工具对您的研究有帮助，请考虑引用我们的论文 ✏️：

```
@article{zhangsh-ternaryclip,
  title={{EdgeRazor}: A Lightweight Framework for Large Language Models via Mixed-Precision Quantization-Aware Distillation},
  author={Shu-Hao Zhang and Le-Tong Huang and Xiang-Sheng Deng and Xin-Yi Zou and Chen Wu and Nan Li and Shao-Qun Zhang},
  year={2026},
  journal={arXiv preprint arXiv:2605.04062}
}

@article{zhangsh-ternaryclip,
  title={{TernaryCLIP}: Efficiently Compressing Vision-Language Models with Ternary Weights and Distilled Knowledge},
  author={Shu-Hao Zhang and Wei-Cheng Tang and Chen Wu and Peng Hu and Nan Li and Liang-Jie Zhang and Qi Zhang and Shao-Qun Zhang},
  year={2025},
  journal={arXiv preprint arXiv:2510.21879}
}
```

## 贡献者列表

这个项目由[机器学习与数据挖掘研究所（LAMDA）](https://www.lamda.nju.edu.cn)和[张绍群教授](https://www.lamda.nju.edu.cn/zhangsq)支持。[张书豪](https://github.com/zhsh9)是本项目的核心开发者和维护者，[邓翔升](https://github.com/deng-xiangsheng)和[黄乐彤](https://github.com/LT1923)是本项目的参与者和开发者。