<div align="center">
  <br/>
  <img src="https://raw.githubusercontent.com/zhangsq-nju/EdgeRazor/main/asset/Logo-full.png" alt="EdgeRazor Logo" width="60%">
  <h3>
    Lightweight Framework for Edge AI
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
    ✨ If you like our project, please give us a star ⭐️ for the latest update.
  </h5>

  <img src="https://raw.githubusercontent.com/zhangsq-nju/EdgeRazor/main/asset/Demo-PC.gif" alt="EdgeRazor Demo" width="75%">
</div>

---

**EdgeRazor** is a lightweight framework for edge AI, designed to train models that are smaller, faster, and deployable across diverse hardware, ranging from mobile and edge endpoints to latency-sensitive clouds. The EdgeRazor framework **seamlessly integrates** model compression techniques into existing full-precision training pipelines with **minimal code modification**, preserving promising task performance and enabling low-cost and high-efficiency computations.

EdgeRazor currently focuses on low-bit LLM compression via configurable quantization-aware distillation. In terms of **quantization**, EdgeRazor supports quantizing weights (including embedding and lm_head layers), activations, and KV cache. Quantized bit-widths include the uniform 1.58-bit and 4-bit, as well as matrix-wise mixed-precision, such as 2.79-bit (50% 4-bit + 50% 1.58-bit) and 1.88-bit (12.5% 4-bit + 87.5% 1.58-bit). In terms of **distillation**, EdgeRazor offers the logits, features, and attention distillation, all of which can be flexibly combined within a unified configuration interface.

EdgeRazor achieves the state-of-the-art performance across a range of models, including base LLMs, instruction-tuned LLMs, and multimodal LLMs. For W-A8-KV8 quantization, **Qwen3-0.6B-EdgeRazor** attains average scores of **47.80** / **44.10** / **41.76** / **39.81** at 4-bit / 2.79-bit / 1.88-bit / 1.58-bit, corresponding to compression ratios of **3.94×** / **5.05×** / **6.40×** / **7.03×**, respectively. In comparison, the best prior methods achieve <u>45.74</u> / <u>37.38</u> / <u>30.49</u> at 4-bit / 3-bit / 2-bit with compression ratios of <u>2.21×</u> / <u>2.47×</u> / <u>2.78×</u>.

<p align="center">
  <img src="https://raw.githubusercontent.com/zhangsq-nju/EdgeRazor/main/asset/Architeacture.png" alt="EdgeRazor Architecture">
  <br> Figure: The EdgeRazor framework with lightweight model training pipeline.
</p>

## News

- 🔥 **[2026-04]**: 📄 Paper-EdgeRazor is available on arXiv: [2605.04062](https://arxiv.org/abs/2605.04062)!
- 🔥 **[2026-04]**: 🚀 [EdgeRazor Playground](https://huggingface.co/spaces/zhangsq-nju/EdgeRazor-Playground) is launched and open-sourced! CPU-friendly! Have a try!
- 🔥 **[2026-04]**: 🏅 [CACC 2025 Final](https://cacc.ccf.org.cn/#/tzgg/%E9%80%9A%E7%9F%A5%E5%85%AC%E5%91%8A/6ce6fd51cffa62eb3859a8bb80af1040) (China Algorithm Capability Competition) apply EdgeRazor as a solution in the AI subject!
- 🔥 **[2026-04]**: 🏆 Low-bit LLMs by EdgeRazor is released! Check our Hugging Face collection: [zhangsq-nju/edgerazor-nbit](https://huggingface.co/collections/zhangsq-nju/edgerazor-nbit).
- 🔥 **[2026-04]**: 🛠️ Open-sourced EdgeRazor-V1 is released! Now configurable on diverse models for seamless integration and customization!
- 🔥 **[2025-10]**: 📄 Paper-TernaryCLIP is available on arXiv: [2510.21879](https://arxiv.org/abs/2510.21879)!

## Contents

- [News](#news)
- [Contents](#contents)
- [Quick Start](#quick-start)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Docker](#docker)
  - [Playground](#playground)
- [Main Techniques](#main-techniques)
- [Applications](#applications)
- [Model Zoo](#model-zoo)
  - [LLMs](#llms)
  - [MLLMs](#mllms)
- [Todo List](#todo-list)
- [Citation](#citation)
- [Contributor List](#contributor-list)

## Quick Start

### Installation

- Download from PyPi

```bash
pip install edgerazor
```

- Download from GitHub (latest version)

```bash
git clone https://github.com/zhangsq-nju/EdgeRazor.git && cd EdgeRazor
conda create -n edgerazor python=3.10.20 -y
conda activate edgerazor
pip install -e .
```

### Usage

After installation, you can integrate EdgeRazor into your existing training pipeline to build lightweight models.

1. Use unified configuration by [yaml](https://github.com/zhangsq-nju/EdgeRazor/tree/main/example/configs/qad/qat_w4_a8_kd_fd.yaml), [json](https://github.com/zhangsq-nju/EdgeRazor/tree/main/example/configs/qad/qat_w4_a8_kd_fd.json) or [dict](https://github.com/zhangsq-nju/EdgeRazor/tree/main/example/configs/qad/qat_w4_a8_kd_fd.py).

2. Seamlessly integrate EdgeRazor into your FULL-PRECISION model training and enjoy your lightweight journey!

```python
# Init EdgeRazor for lightweight model
edgerazor = EdgeRazor(config="/path/to/config.yaml")
student = edgerazor.quantize(student)
# Training loop
student_outputs = student(inputs)
teacher_outputs = teacher(inputs)
# Calculate loss
loss, loss_dict = edgerazor.compute_loss(student_outputs, teacher_outputs, labels)
```

### Docker

Lightweight models are available from checkpoints trained with EdgeRazor. For example, you can convert Qwen3-EdgeRazor-4bit checkpoints to Q4_0 GGUF models. We also provide ready-to-use quantized models in our [collection](https://huggingface.co/collections/zhangsq-nju/edgerazor-nbit), including [Qwen3-0.6B-EdgeRazor-GGUF](https://huggingface.co/zhangsq-nju/Qwen3-0.6B-EdgeRazor-GGUF) and [Qwen3-1.7B-EdgeRazor-GGUF](https://huggingface.co/zhangsq-nju/Qwen3-1.7B-EdgeRazor-GGUF).

```bash
# Serve quantized LLMs under CPU-only environments:
docker pull ghcr.io/ggml-org/llama.cpp:server
hf download zhangsq-nju/Qwen3-1.7B-EdgeRazor-GGUF Qwen3-1.7B-EdgeRazor-TQ2_0.gguf --local-dir /path/to/Qwen3-1.7B-EdgeRazor-GGUF
cd ./docker && bash local_server_tq2_0.sh
```

### Playground

EdgeRazor Playgound is CPU-friendly! Enjoy low-bit LLMs from EdgeRazor on your edge devices!

```bash
cd EdgeRazor/playground
pip install -r requirements.txt
python app.py
```

![EdgeRazor Playground Sreenshot](https://raw.githubusercontent.com/zhangsq-nju/EdgeRazor/main/asset/Playground.png)

## Main Techniques

Quantization-Aware Distillation (QAD): 

- Configurable mixed-precision quantization for weights
- Configurable knowledge distillation pipelines between 16-bit and $n$-bit models
- Knowledge distillation methods: Adaptive Feature Distillation (AFD), Entropy-Aware KL Divergence (EAKLD)

<p align="center">
  <img src="https://raw.githubusercontent.com/zhangsq-nju/EdgeRazor/main/asset/EdgeRazor-Framework.png" alt="EdgeRazor Framework Overview" width="75%">
  <br> Figure: Workflow of the EdgeRazor framework.
</p>

## Applications

- Lightweight ViT-S/16 for MNIST, check [here](https://github.com/zhangsq-nju/EdgeRazor/tree/main/example/vit/README.md).
- Lightweight ViT-S/16 for RetinaMNIST, check [here](https://github.com/zhangsq-nju/EdgeRazor/tree/main/example/medical_vit/README.md).
- Lightweight ResNet-18, check [here](https://github.com/zhangsq-nju/EdgeRazor/tree/main/example/resnet/README.md).
- Lightweight Qwen3-0.6B/1.7B, check [here](https://github.com/zhangsq-nju/EdgeRazor/tree/main/example/edgerazor-llm/README.md).
- Lightweight MobileLLM-ParetoQ-350M-BF16, check [here](https://github.com/zhangsq-nju/EdgeRazor/tree/main/example/edgerazor-llm/README.md).
- Lightweight Qwen2.5-Omni-7B, check [here](https://github.com/zhangsq-nju/EdgeRazor/tree/main/example/qwen2.5-omni/README.md).

## Model Zoo

### LLMs

- Average Performance (Avg.): average of performance scores in multiple tasks using [lm-eval v0.4.9.1](https://github.com/EleutherAI/lm-evaluation-harness/tree/v0.4.9.1) with [tasks](https://github.com/zhangsq-nju/EdgeRazor/tree/main/src/eval/tasks/lm_eval/).
  - Tasks for instruct LLMs: arc_easy, arc_challenge, hellaswag, boolq, social_iqa, openbookqa, piqa, winogrande, hendrycks_ethics, truthfulqa_mc2, mmlu, gsm8k, humaneval_instruct, ifeval.
  - Tasks for base LLMs: arc_easy, arc_challenge, hellaswag, boolq, social_iqa, openbookqa, piqa, winogrande, hendrycks_ethics, truthfulqa_mc2, mmlu, gsm8k, humaneval.
  - Except for 5-shot gsm8k, all other tasks are 0-shot.

- Hub Link: We provide the original quantized checkpoints. We also transfer the checkpoints into GGUF ([llama.cpp](https://github.com/ggml-org/llama.cpp)) and GPTQ ([GPTQModel](https://github.com/ModelCloud/GPTQModel), working in progress) formats if compatible.

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

### MLLMs

- Video-MME and MLVU are video understanding tasks using [lmms-eval v0.5.0](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/v0.5) with [tasks](https://github.com/zhangsq-nju/EdgeRazor/tree/main/src/eval/tasks/lmms-eval/).

| Model           | W-A-KV       | Group Size | Video-MME | MLVU  | Hub Link                                                                       |
| --------------- | ------------ | ---------- | --------- | ----- | ------------------------------------------------------------------------------ |
| Qwen2.5-Omni-7B | W16-A16-KV16 | -          | 62.81     | 48.01 | [Base](https://huggingface.co/Qwen/Qwen2.5-Omni-7B)                            |
| Qwen2.5-Omni-7B | W4-A16-KV16  | 32         | 62.22     | 48.82 | [EdgeRazor](https://huggingface.co/zhangsq-nju/Qwen2.5-Omni-7B-EdgeRazor-4bit) |

## Todo List

EdgeRazor is continuously evolving! Here's what's coming:

- [x] Lightweight MobileLLM, Qwen3, and Qwen2.5-Omni: training code
- [x] New API: EdgeRazorCausalLMTrainer for compressing text LLMs
- [ ] Upgrade to support the newest dependencies

Have ideas or suggestions? We welcome and appreciate any contributions and collaborations! Please feel free to submit issues or pull requests! 🚀

## Citation

If you find our papar and code useful in your research, please consider kindly citing our papers ✏️:

```
@article{zhangsh-edgerazor,
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

## Contributor List

This project was supported by [LAMDA](https://www.lamda.nju.edu.cn) and Assistant Professor [Shao-Qun Zhang](https://www.lamda.nju.edu.cn/zhangsq). [Shu-Hao Zhang](https://github.com/zhsh9) is the core developer and maintainer of EdgeRazor-V1. [Xiang-Sheng Deng](https://github.com/deng-xiangsheng) and [Le-Tong Huang](https://github.com/LT1923) jointly participated in the development of this project.