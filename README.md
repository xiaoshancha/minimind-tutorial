# MiniMind 系统学习教程

> 在线阅读：https://tutorial.baimuyuan.online/minimind-tutorial/
>
> 基于 [jingyaogong/minimind](https://github.com/jingyaogong/minimind)（commit `393e387`）编写的教材级系统学习教程。
>
> 面向读者：具备深度学习基础与 Transformer 概念、未从 0 训练过 LLM 的学习者。
>
> 目标：学完后能**逐行读懂**一个 LLM 的全部核心训练代码，并亲手从 0 训练、部署自己的对话模型。

---

## 📚 学习路径

```text
第一部分 预备
  第 1 章  LLM 全景与项目导览        ← 建立整体认知、搭环境
  第 2 章  必备基础回顾              ← LLM 视角的数学与工程基础

第二部分 主线（必学）
  第 3 章  Tokenizer 与数据处理      ← BPE、模板、数据集类
  第 4 章  模型架构逐行精读          ← RMSNorm/RoPE/GQA/SwiGLU/MoE
  第 5 章  预训练                    ← 第一次训练！loss 曲线解读
  第 6 章  SFT 指令微调              ← 从"会接龙"到"会对话"
  第 7 章  推理与生成                ← 采样、KV Cache、部署三件套

第三部分 进阶（可选）
  第 8 章  LoRA 参数高效微调         ← 纯手写实现
  第 9 章  RLHF-DPO 偏好对齐          ← 数学推导 + 实现
  第 10 章 RLAIF：PPO 与 GRPO         ← 强化学习
  第 11 章 Agentic RL 与 Tool Call    ← 让模型会用工具
  第 12 章 知识蒸馏                  ← 大模型教小模型

第四部分 实战
  第 13 章 从 0 到 1 完整实战         ← 全流程串联（约 2 小时 3 元）
  第 14 章 项目扩展与进阶路线

附录
  附录 A  PyTorch 速查
  附录 B  LLM 术语表
```

## ⏱️ 预期投入

| 阶段 | 阅读时间 | 实验时间 |
|------|----------|----------|
| 主线（第 1~7 章） | 2~3 天 | 3~5 小时（GPU）/ 1 天（CPU） |
| 进阶（第 8~12 章） | 2~3 天 | 按需 |
| 实战（第 13 章） | 半天 | 2~4 小时（GPU） |

## ⚙️ 硬件要求

- 最低：16GB 内存 CPU（可跑通全部主线，速度慢 20~50 倍）
- 推荐：RTX 3090/4090（24GB），完整复现约 3 元成本

## 🔗 对应代码

教程的每章开头都标注了对应的源码文件。学习时请打开仓库对照阅读：

```
minimind/
├── model/model_minimind.py      ← 第 4 章
├── model/model_lora.py          ← 第 8 章
├── dataset/lm_dataset.py        ← 第 3 章
├── trainer/train_pretrain.py    ← 第 5 章
├── trainer/train_full_sft.py    ← 第 6 章
├── trainer/train_lora.py        ← 第 8 章
├── trainer/train_dpo.py         ← 第 9 章
├── trainer/train_grpo.py        ← 第 10 章
├── trainer/train_ppo.py         ← 第 10 章
├── trainer/rollout_engine.py    ← 第 10 章
├── trainer/train_agent.py       ← 第 11 章
├── trainer/train_distillation.py← 第 12 章
├── trainer/trainer_utils.py     ← 第 5 章
├── eval_llm.py                  ← 第 7 章
└── model/tokenizer_config.json  ← 第 3、11 章
```
