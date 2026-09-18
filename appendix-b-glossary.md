# 附录 B：LLM 术语表

> 按本教程出现章节标注，方便回查。

## B.1 基础概念

| 术语 | 中文 | 含义 | 章节 |
|------|------|------|------|
| Autoregressive | 自回归 | 用历史 token 预测下一个 token 的建模方式 | 1, 2 |
| Token | 词元 | 文本切分后的最小单元（词/子词/字节） | 3 |
| Vocabulary | 词表 | token 集合，大小 V 决定输出层维度 | 3 |
| Logits | 逻辑值 | 输出层未归一化的分数向量（V 维） | 2, 4 |
| Softmax | 软最大化 | logits → 概率分布 | 2 |
| Cross Entropy | 交叉熵 | 分类/语言建模的损失函数 | 2 |
| Perplexity (PPL) | 困惑度 | exp(交叉熵)，语言建模经典指标 | 2 |
| Teacher Forcing | 教师强制 | 训练时用真实 token 作条件而非模型输出 | 2 |
| KV Cache | 键值缓存 | 推理时缓存历史 K/V 免重算 | 4, 7 |
| GQA | 分组查询注意力 | 多 Q 头共享 KV 头，省显存 | 2, 4 |
| MHA | 多头注意力 | 标准多头注意力 | 2 |
| RoPE | 旋转位置编码 | 用旋转矩阵注入位置信息 | 2, 4 |
| YaRN | 外推算法 | 低频频率拉伸实现长度外推 | 4, 7 |
| RMSNorm | 均方根归一化 | 免均值中心化的 LayerNorm 变体 | 2, 4 |
| Pre-Norm | 预归一化 | 先归一化再子层的残差结构 | 2, 4 |
| SwiGLU | 门控线性单元 | SiLU 门控的 FFN 激活 | 2, 4 |
| MoE | 混合专家 | 多个 FFN 专家 + 路由器，稀疏激活 | 4, 14 |
| Flash Attention | 闪存注意力 | 分块计算的 O(1) 显存注意力内核 | 4 |
| Dense | 稠密 | 所有参数全部激活的常规模型 | 4 |

## B.2 训练相关

| 术语 | 中文 | 含义 | 章节 |
|------|------|------|------|
| Pretrain | 预训练 | 海量无标注文本上的 next-token 学习 | 5 |
| SFT | 有监督微调 | 高质量问答对上的对话能力训练 | 6 |
| LoRA | 低秩适配 | 低秩增量矩阵的参数高效微调 | 8 |
| PEFT | 参数高效微调 | LoRA/Adapter 等只更新少量参数的方法 | 8 |
| RLHF | 人类反馈强化学习 | 人类偏好信号驱动的强化学习 | 9, 10 |
| RLAIF | AI 反馈强化学习 | AI/规则信号驱动的强化学习 | 10 |
| RM | 奖励模型 | 给回答打分的模型 | 10 |
| Critic | 价值网络 | 估计状态价值的网络（PPO） | 10 |
| GAE | 广义优势估计 | 优势的递推估计方法 | 10 |
| PPO | 近端策略优化 | 裁剪式策略梯度算法 | 10 |
| GRPO | 组相对策略优化 | 组内归一化优势的无 Critic 算法 | 10 |
| DPO | 直接偏好优化 | 免 RM 免采样的偏好对齐 | 9 |
| KL 散度 | KL 散度 | 分布距离度量，RL 中的约束项 | 9, 10 |
| Importance Ratio | 重要性比值 | 新旧策略概率比，策略梯度修正 | 10 |
| Rollout | 采样展开 | RL 中模型生成回答的过程 | 10, 11 |
| RLVR | 可验证奖励 RL | 程序化验证答案的强化学习 | 11 |
| Agentic RL | 智能体强化学习 | 多轮工具调用场景的 RL | 11 |
| Tool Call | 工具调用 | 模型输出结构化函数调用 | 11 |
| Distillation | 知识蒸馏 | 学生模型学习教师分布 | 12 |
| KD | 知识蒸馏（缩写） | 同上 | 12 |
| Temperature (蒸馏) | 温度 | 软化分布的参数 T | 12 |
| Gradient Accumulation | 梯度累积 | 小 batch 多步累加等效大 batch | 2, 5 |
| AMP | 自动混合精度 | fp16/bf16 + fp32 混合训练 | 2, 5 |
| GradScaler | 梯度缩放器 | fp16 防下溢的 loss 缩放 | 2, 5 |
| Cosine Annealing | 余弦退火 | 学习率余弦衰减调度 | 2, 5 |
| Gradient Clipping | 梯度裁剪 | 限制梯度范数防爆炸 | 2, 5 |
| DDP | 分布式数据并行 | 多卡同步梯度训练 | 5 |
| Checkpoint | 检查点 | 训练状态快照（含优化器） | 5 |
| Resume | 续训 | 从检查点恢复训练 | 5 |
| Catastrophic Forgetting | 灾难性遗忘 | 新任务学习破坏旧知识 | 6 |
| Loss Spike | 损失尖峰 | 训练中 loss 突然跳高 | 5 |

## B.3 推理与部署

| 术语 | 中文 | 含义 | 章节 |
|------|------|------|------|
| Top-K 采样 | Top-K 采样 | 只从概率最高的 K 个 token 采样 | 7 |
| Top-P / Nucleus | 核采样 | 从累积概率达 P 的集合采样 | 7 |
| Repetition Penalty | 重复惩罚 | 压低已出现 token 概率 | 7 |
| Streamer | 流式输出 | token 逐块解码的实时输出 | 7 |
| Chat Template | 对话模板 | 消息 → 训练/推理文本的渲染规则 | 3, 6 |
| open_thinking | 思考开关 | 模板层注入/不注入思考标签 | 3, 7, 11 |
| Adaptive Thinking | 自适应思考 | 同一模型可切换思考/直答 | 11 |
| EOS/BOS/PAD | 结束/开始/填充标记 | 特殊 token | 3 |
| GGUF | GGUF 格式 | llama.cpp 的量化模型格式 | 13, 14 |

## B.4 数据与评估

| 术语 | 中文 | 含义 | 章节 |
|------|------|------|------|
| BPE | 字节对编码 | 子词切分算法 | 3 |
| ByteLevel | 字节级预切分 | 按 UTF-8 字节切分的预分词器 | 3 |
| JSONL | JSON 行格式 | 每行一个 JSON 对象的数据格式 | 3 |
| Label Mask | 标签掩码 | 只对特定位置计算损失 | 3, 6 |
| ignore_index | 忽略索引 | PyTorch CE 中跳过 -100 标签 | 2, 3 |
| -100 | -100 | 掩码位置的约定标签值 | 2, 3 |
| Chosen/Rejected | 偏好对 | 好/坏回答对（DPO 数据） | 9 |
| GT | 标准答案 | 任务真值（RLVR 验证用） | 11 |
| Data Augmentation | 数据增强 | 随机变换增加数据多样性 | 3, 6 |
| BPB | 每字节比特数 | 跨 tokenizer 的语言模型指标 | 3 |

## B.5 工程与生态

| 术语 | 中文 | 含义 | 章节 |
|------|------|------|------|
| DDP 解包 | — | model.module / _orig_mod 还原原模型 | 5, 8 |
| Monkey-patch | 猴子补丁 | 运行时替换方法（LoRA 注入） | 8 |
| Tied Embeddings | 权重共享 | embedding 与输出层共享参数 | 4 |
| Aux Loss | 辅助损失 | 附加约束损失（MoE 负载均衡） | 4 |
| Meta Device | 元设备 | 无数据权重的设备占位（transformers 5.x） | 4 |
| RANK/LOCAL_RANK | 进程编号 | torchrun 环境变量 | 5 |
| OpenAI 协议 | — | chat.completions 兼容 API | 7, 13 |
| SwanLab/WandB | 实验追踪 | 训练曲线可视化 | 5 |
