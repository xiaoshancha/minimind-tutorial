# 第 1 章 LLM 全景与 MiniMind 项目导览

> 对应代码：仓库根目录（README.md、requirements.txt）
>
> 本章目标：建立"从 0 训练一个大语言模型需要经历哪些阶段"的整体认知，熟悉 MiniMind 项目的目录结构，并完成开发环境搭建。学完本章，你应该能回答"一个 LLM 从零到上线要经过哪几个阶段、每个阶段解决什么问题"。

---

## 1.1 大语言模型究竟在做什么

大语言模型（Large Language Model, LLM）表面上看是"聊天机器人"，但从机器学习角度看，它本质上是一个**自回归（autoregressive）语言模型**：给定一串 token，预测下一个 token 是什么。

```text
输入:  "天空为什么是"  →  模型 →  输出: "蓝色的"（的概率分布）
```

模型的全部"知识"——语言规律、事实知识、推理能力——都以**参数**的形式存在。训练，就是调整这些参数，让模型预测下一个 token 越来越准。

这个视角极其重要，因为它意味着：

1. **LLM 的训练目标与你平时做的分类任务没有本质区别**——都是最小化交叉熵损失，只是类别数变成了词表大小（MiniMind 是 6400）；
2. **模型能力的天花板由数据决定**——模型只能学会数据里反复出现的规律；
3. **"对话"能力不是天生的**——预训练出来的模型只会"接龙"，是后续阶段教会它一问一答的。

## 1.2 LLM 完整训练流水线

一个工业级 LLM 从 0 到 1 通常经历以下阶段。MiniMind 完整复现了这条链路，这也是我们整本教程的叙事主线：

| 阶段 | 输入数据 | 训练目标 | 产出的能力 | 类比 |
|------|----------|----------|------------|------|
| **① 分词器训练** | 原始语料 | 学到一套文本↔token 的映射 | 编解码能力 | 先编一本"词典" |
| **② 预训练 Pretrain** | 海量无标注文本 | 预测下一个 token | 语言规律、世界知识 | 大量读书，学会"接龙" |
| **③ 监督微调 SFT** | 高质量问答对 | 预测对话中 assistant 的回答 | 对话、指令跟随 | 学会一问一答的礼貌 |
| **④ 偏好对齐 RL** | 偏好/奖励信号 | 让回答更符合人类偏好 | 价值观对齐、推理增强 | 学会"什么该说、怎么说更好" |
| **⑤ 推理部署** | — | — | 服务用户 | 毕业工作 |

几个重要事实：

- **阶段①是前置工程**：先有词典才能训练语言模型。词表大小直接影响参数量和编解码效率（第 3 章详述）。
- **阶段②消耗 90% 以上的算力**：GPT-3、Llama 的预训练需要上千张 GPU 训练数月。MiniMind 把模型缩到 64M 参数（GPT-3 的 1/2700），让个人显卡几小时就能复现——**代价是能力上限很低**。
- **阶段③、④是"调教"而非"教学"**：SFT 之后模型的知识总量没有增加，只是学会了如何把已有知识组织成回答。这也是为什么 SFT 数据要求"高质量"——低质量数据会污染模型。
- **没有阶段④，模型也能用**：MiniMind 的 "Zero" 路线（Pretrain + SFT）就能得到一个可对话的模型，只是回答质量、安全性、推理深度远不如经过 RL 的版本。

### 1.2.1 后训练（Post-training）的两条路径

第 9~11 章会深入两个概念，这里先建立直觉：

- **RLHF（基于人类反馈）**：让人类对模型的多个回答打分排序，模型学会"人类更喜欢什么"。
- **RLAIF（基于 AI 反馈）**：用 AI 模型或规则/程序来提供反馈信号（比如"答案的最终数值对不对"），可规模化，是 DeepSeek-R1 这类推理模型的核心训练方式。

## 1.3 MiniMind 项目简介

**MiniMind**（仓库 `jingyaogong/minimind`）是一个"用 PyTorch 从 0 实现 LLM 全流程"的开源项目，它有两个核心特点：

1. **极小的规模**：主线模型 `minimind-3` 只有约 **64M 参数**，词表 6400，8 层 Transformer。单张 RTX 3090 上，Pretrain + SFT 各 1 epoch 约 2~3 小时、成本约 3 元人民币。
2. **全链路 + 纯手写**：关键算法（RoPE、GQA、MoE、LoRA、DPO、PPO、GRPO、蒸馏）全部用原生 PyTorch 实现，不依赖 peft/trl 的高层封装。

> 为什么要用小模型学？因为大模型的训练成本让个人无法试错。64M 模型上能验证的每一个原理（梯度消失、学习率敏感、loss 曲线形态、过拟合），放到 7B 模型上依然成立。**学会了小模型的全部细节，你就理解了大模型的全部机制。**

### 1.3.1 结构特点（第 4 章详解）

MiniMind-3 对齐 Qwen3 生态的 Decoder-Only 结构：

- Pre-Norm + **RMSNorm**（替代 LayerNorm）
- **SwiGLU** 激活（替代 ReLU）
- **RoPE** 旋转位置编码（支持 YaRN 外推）
- **GQA** 分组查询注意力（8 个 Q 头、4 个 KV 头）
- 可选 **MoE** 版本（4 experts / top-1）

## 1.4 项目地图：每个文件是干什么的

在动手之前，先认识这个仓库。MiniMind 的目录结构非常规整，基本是一份"代码版教程"：

```text
minimind/
├── README.md                  # 项目文档（信息量极大，值得通读）
├── requirements.txt           # Python 依赖
├── dataset/
│   ├── lm_dataset.py          # ★ 全部数据集类（Pretrain/SFT/DPO/RLAIF/Agent）
│   └── dataset.md             # 数据集说明
├── model/
│   ├── model_minimind.py      # ★ 核心：MiniMind 模型（Dense + MoE）
│   ├── model_lora.py          # ★ LoRA 实现（纯手写）
│   ├── tokenizer.json         # BPE 分词器文件
│   └── tokenizer_config.json  # 分词器配置 + chat_template
├── trainer/
│   ├── trainer_utils.py       # 训练工具：学习率调度、checkpoint、DDP 初始化
│   ├── train_pretrain.py      # ★ 预训练
│   ├── train_full_sft.py      # ★ 全参数 SFT
│   ├── train_lora.py          # LoRA 微调
│   ├── train_dpo.py           # DPO 偏好对齐
│   ├── train_ppo.py           # PPO 强化学习
│   ├── train_grpo.py          # GRPO 强化学习
│   ├── train_agent.py         # Agentic RL（工具调用场景）
│   ├── train_distillation.py  # 白盒知识蒸馏
│   ├── train_tokenizer.py     # 自定义分词器训练
│   └── rollout_engine.py      # RL 的采样引擎（生成回复）
├── scripts/
│   ├── eval_toolcall.py       # 工具调用测试
│   ├── chat_api.py            # 简单 API 示例
│   ├── serve_openai_api.py    # OpenAI 兼容 API 服务
│   ├── web_demo.py            # Streamlit WebUI
│   └── convert_model.py       # 权重格式转换 / LoRA 合并
├── eval_llm.py                # ★ CLI 对话评测入口
└── images/                    # 文档图片（含模型结构图）
```

标 ★ 的文件是本教程的主线精读对象。建议读者一边学一边在仓库里打开对应文件对照。

### 1.4.1 学习路径建议

```text
第 3 章 Tokenizer → 第 4 章 模型 → 第 5 章 预训练 → 第 6 章 SFT → 第 7 章 推理
        ↓（可选进阶）
第 8~12 章：LoRA / DPO / PPO+GRPO / Agent / 蒸馏
        ↓
第 13 章 完整实战
```

主线（第 3~7 章）学完后，你已经完整理解了一个对话模型从 0 诞生的全部代码；进阶章节是在这个骨架上生长出工业界真实使用的方法。

## 1.5 环境搭建

### 1.5.1 硬件与软件要求

| 项目 | 最低要求 | 推荐 |
|------|----------|------|
| 操作系统 | Linux / Windows / macOS | Linux |
| 内存 | 16 GB | 32 GB |
| GPU | 无（CPU 可跑，但慢 20~50 倍） | RTX 3090 / 4090（24 GB） |
| CUDA | 11.8+（用 GPU 训练时） | 12.x |
| Python | 3.10+ | 3.10 |

> 说明：MiniMind 主线默认配置（768 维、8 层、64M 参数）在 CPU 上也能完成训练，只是单 epoch 需要数小时到一天。建议先用 CPU 跑通流程，再上 GPU。

### 1.5.2 创建虚拟环境并安装依赖

```bash
# 进入项目目录（若还未克隆）
git clone --depth 1 https://github.com/jingyaogong/minimind
cd minimind

# 创建虚拟环境（推荐 conda 或 venv）
conda create -n minimind python=3.10 -y
conda activate minimind

# 安装依赖（国内网络可加 -i https://mirrors.aliyun.com/pypi/simple）
pip install -r requirements.txt
```

> ⚠️ 若 `requirements.txt` 安装失败（例如个别包版本冲突），通常只需要核心依赖即可跑通主线：
>
> ```bash
> pip install torch transformers datasets tokenizers swanlab modelscope
> ```

### 1.5.3 验证环境

```python
import torch
from transformers import AutoTokenizer

# 1) 检查 CUDA
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")

# 2) 检查分词器能否正常加载
tok = AutoTokenizer.from_pretrained("./model")
print("词表大小:", len(tok))
print(tok.encode("你好，世界"))
```

预期输出示例：

```text
CUDA available: True
GPU: NVIDIA GeForce RTX 3090
词表大小: 6400
[1, 3166, 4902, 3112, 3350, 2]
```

`[1, ..., 2]` 中 `1` 是 `<|im_start|>`（bos），`2` 是 `<|im_end|>`（eos）。

## 1.6 第一次运行：快速体验已训练好的模型

不训练也能先玩起来。从 ModelScope 下载官方训练好的权重（约 130 MB）：

```bash
# 在项目根目录执行
modelscope download --model gongjy/minimind-3 --local_dir ./minimind-3
```

然后运行 CLI 对话：

```bash
python eval_llm.py --load_from ./minimind-3
```

进入交互后选择 `[1] 手动输入`，即可与模型对话：

```text
[0] 自动测试
[1] 手动输入
1
💬: 你是谁
🧠: 我是 MiniMind，一个小巧但有用的语言模型。我由 Jingyao Gong 开发...
```

> `eval_llm.py` 的完整参数（`--temperature`、`--top_p`、`--open_thinking` 等）在第 7 章详细讲解。

如果你暂时不想下载权重，也可以直接跳到第 5 章，从训练开始——训练完自然就有自己的权重了。

## 1.7 本章小结

- LLM 本质是**自回归模型**：预测下一个 token，训练目标就是交叉熵。
- 完整流水线：**Tokenizer → 预训练 → SFT → 偏好对齐 RL → 推理部署**，MiniMind 全链路覆盖。
- 预训练给能力，SFT 给形式，RL 给偏好；**没有 RL 也能对话**。
- MiniMind 是 64M 参数的"麻雀"，单卡数小时可复现，适合彻底学透原理。
- 项目目录结构规整：`model/`（结构）、`dataset/`（数据）、`trainer/`（训练）、`scripts/`（服务）、`eval_llm.py`（评测入口）。

## 1.8 思考题

1. 为什么说"模型的能力上限由数据决定"？如果预训练数据全是英文小说，模型的中文对话能力会怎样？
2. SFT 阶段能否"教会模型新知识"？请结合 1.2 节的类比说明。
3. 64M 参数 vs 175B 参数（GPT-3），训练成本差异约为多少倍？为什么小模型适合入门？
4. 打开 `requirements.txt`，找出与"训练"直接相关的包有哪些，与"推理服务"相关的有哪些。
5. （动手）验证你的环境中 `torch.cuda.is_available()` 的结果，并记录你的 GPU 型号与显存。

## 1.9 拓展阅读

- MiniMind README.md 开头的"项目介绍"与"快速开始"章节（原文，信息密度极高）
- 论文：《Language Models are Few-Shot Learners》（GPT-3, 2020）——理解大规模预训练的开山之作
- 论文：《Training language models to follow instructions with human feedback》（InstructGPT, 2022）——理解 SFT + RLHF 流水线为何如此设计
