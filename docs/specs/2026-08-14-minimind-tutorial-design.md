# MiniMind 系统学习教程 — 设计文档

- 日期：2026-08-14
- 目标读者：具备深度学习基础与 Transformer 概念、未从 0 训练过 LLM 的学习者（中文）
- 目标仓库：https://github.com/jingyaogong/minimind （本地路径 /home/yq/learning/minimind，commit 393e387）
- 交付形式：`/home/yq/learning/minimind-tutorial/` 下按章节拆分的 Markdown 文件
- 深度：大学教材级别（数学推导 + 逐行代码精读 + 实验 + 思考题）
- 规模：预计每章 3000–5000 字，总 6–8 万字

## 学习路径（用户确认）

1. 懂 DL 基础 + Transformer 概念，没训过 LLM → 第 2 章快速回顾，不从零讲
2. 全链路但分主次 → 主线：Tokenizer / 模型 / 预训练 / SFT / 推理；拓展：LoRA / DPO / PPO/GRPO / Agentic RL / 蒸馏
3. 会动手跑 → 每章含实验，第 13 章完整实战串联
4. 章节 + 练习题 → 每章末尾思考题/练习题

## 章节大纲

**第一部分 预备**
- 第 1 章：LLM 全景与 MiniMind 项目导览（流水线总览、项目地图、环境搭建）
- 第 2 章：必备基础回顾（自回归、交叉熵/困惑度、Transformer 组件、显存概念）

**第二部分 主线**
- 第 3 章：Tokenizer 与数据处理（BPE、minimind_tokenizer、lm_dataset.py、chat_template）
- 第 4 章：模型架构（model_minimind.py：RMSNorm/RoPE/GQA/SwiGLU/MoE、参数量推导）
- 第 5 章：预训练（train_pretrain.py、因果 mask、next-token loss、实验）
- 第 6 章：SFT（train_full_sft.py、对话模板与 mask 策略、实验）
- 第 7 章：推理与生成（generate、采样、KV Cache、YaRN、CLI/WebUI/API）

**第三部分 进阶**
- 第 8 章：LoRA（model_lora.py + train_lora.py、实验）
- 第 9 章：RLHF-DPO（Bradley-Terry、train_dpo.py）
- 第 10 章：RLAIF（PPO、GRPO、rollout_engine.py）
- 第 11 章：Agentic RL 与 Tool Call（train_agent.py、open_thinking）
- 第 12 章：知识蒸馏（黑盒/白盒、train_distillation.py）

**第四部分 实战与拓展**
- 第 13 章：从 0 到 1 完整实战（数据→训练→评估→部署、断点续训、多卡）
- 第 14 章：项目扩展与进阶路线（MoE 深入、长上下文、llama.cpp/vllm/ollama 对接）

**附录**：A. PyTorch 速查；B. LLM 术语表

## 每章固定结构

学习目标 → 理论推导（含公式）→ 代码逐行精读 → 动手实验 → 小结 → 思考题/练习题 → 拓展阅读

## 写作要求

- 所有代码引用、文件路径、命令行参数必须与实际仓库核对（commit 393e387）
- 公式用 LaTeX 语法（KaTeX/MathJax 兼容）
- 每章独立成文，可单独阅读；章间有交叉引用
- 实验章节给出可执行命令与预期结果
