# 第 6 章 SFT 指令微调：从"会接龙"到"会对话"

> 对应代码：`trainer/train_full_sft.py`（173 行）、`dataset/lm_dataset.py` 的 `SFTDataset`
>
> 本章目标：理解有监督微调（Supervised Fine-Tuning, SFT）的原理与工程差异，逐行对比 SFT 与预训练两个训练脚本，动手训练出**第一个可对话的 MiniMind Zero 模型**。

---

## 6.1 SFT 在做什么

预训练出来的模型只会"接龙"。SFT 用**高质量的问答对**（`sft_t2t_mini.jsonl`，1.6GB）告诉模型：看到"用户提问"应该"这样回答"。

关键认知（第 1 章 1.2 节）：**SFT 不增加知识，只改变行为**。模型的知识全部来自预训练；SFT 教会的是：

1. **对话格式**：识别 user/assistant 角色边界，知道何时该说话、何时停（输出 `<|im_end|>`）；
2. **指令跟随**：听懂"请解释 X""写一个函数"这类要求并按要求产出；
3. **回答风格**：简洁、有条理、不胡言乱语（相对预训练"想到哪说到哪"）。

SFT 数据如果含 Tool Call、思考链样本，还会顺带教会工具调用与显式思考（第 11 章）。

## 6.2 SFT 与预训练的三处关键差异

对照两个脚本，`train_full_sft.py` 与 `train_pretrain.py` 的骨架几乎一样（九步流程、训练循环、断点续训、DDP），**差异只有三处**——这三处正是 SFT 的精髓：

| 差异点 | 预训练 | SFT | 原因 |
|--------|--------|-----|------|
| 数据集类 | `PretrainDataset`（全位置监督） | `SFTDataset`（仅 assistant 段监督） | 只学"怎么回答"，不学"怎么提问" |
| 学习率 | 5e-4 | **1e-5** | 模型已有好参数，大幅更新会破坏预训练知识（灾难性遗忘） |
| 初始权重 | 随机 | **`--from_weight pretrain`** | 在预训练基础上继续训练 |

### 6.2.1 为什么 SFT 学习率要低一个数量级？

预训练后的模型已经处于一个"好解"附近。SFT 用 1e-5（预训练的 1/50）做**局部细调**：

- 学习率太大 → 参数大步移动 → 把预训练学到的语言规律/知识冲掉（灾难性遗忘），表现是模型"变笨"或胡言乱语；
- 学习率太小 → 学不动对话格式，模型仍然不会问答；
- 1e-5 是 64M 小模型 SFT 的经验甜点值。

> 对比：LoRA（第 8 章）因为只更新少量低秩参数，学习率可以更激进（1e-4 量级），这也是 PEFT 的优势之一。

### 6.2.2 为什么只监督 assistant 段？

第 3 章 3.4.3 已详解 `SFTDataset.generate_labels` 的掩码算法，这里从训练目标的角度再论证一次：

- 输入序列 = `user 提问 + assistant 回答`，模型用整个序列预测下一个 token；
- 如果**所有位置都算 loss**：模型会花费大量梯度学习"如何提出好问题"——但推理时提问来自用户，不是模型；
- 更糟的是，模型会学习到"user 内容也是要生成的"——生成时它可能自己编造问题自问自答（复读机/独角戏现象）；
- 掩码后，模型只对 assistant 片段产生梯度，user 片段纯粹作为"条件上下文"输入。

**这是 LLM 对齐训练中最重要的工程细节之一**，所有对话模型（包括 ChatGPT 的训练）都遵循此策略。

## 6.3 训练脚本差异精读

### 6.3.1 参数默认值对比

```python
# train_full_sft.py（第 86~107 行）
parser.add_argument("--batch_size", type=int, default=16)          # 预训练 32
parser.add_argument("--learning_rate", type=float, default=1e-5)   # 预训练 5e-4
parser.add_argument("--accumulation_steps", type=int, default=1)   # 预训练 8
parser.add_argument("--max_seq_len", default=768)                  # 预训练 340
parser.add_argument('--from_weight', default='pretrain')           # 预训练 'none'
```

注意 `accumulation_steps=1`：有效 batch = 16×1 = 16，远小于预训练的 256——**SFT 数据量小、单样本价值高**，小 batch + 低学习率是"精雕细琢"模式。

### 6.3.2 数据流：模板 → 掩码 → 训练

```python
model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)   # 加载 pretrain_768.pth
train_ds = SFTDataset(args.data_path, tokenizer, max_length=args.max_seq_len)
```

`SFTDataset.__getitem__` 的完整链路（对照第 3 章 3.4.3）：

```text
jsonl 一行
   → pre_processing_chat：20% 概率加 system 消息（数据增强）
   → create_chat_prompt：apply_chat_template 渲染成文本（含 <think> 结构）
   → post_processing_chat：80% 概率移除空 think（数据增强）
   → tokenizer 编码 + 右侧 padding
   → generate_labels：扫描 <|im_start|>assistant 片段，其余置 -100
   → 返回 (input_ids, labels)
```

训练循环与预训练**完全相同**（loss 计算在 `MiniMindForCausalLM.forward` 中：`F.cross_entropy(..., ignore_index=-100)` 自动跳过掩码位置）——差异全部封装在数据层。**这是把"策略"下沉到数据集的经典设计**：训练代码零改动，行为完全不同。

## 6.4 动手实验：训练 MiniMind Zero

### 实验 6.1：SFT 训练

```bash
# 前提：已完成第 5 章预训练（存在 out/pretrain_768.pth）
# 下载 sft_t2t_mini.jsonl（1.6GB）放入 ./dataset/
cd trainer
python train_full_sft.py
```

预期日志：

```text
Model Params: 63.91M
Trainable Params: 63.910M
Epoch:[1/2](100/…), loss: 1.2500, logits_loss: 1.2500, ...
Epoch:[2/2](…/…), loss: 0.7300, ...
```

**loss 解读**：SFT 的 loss 起点远低于预训练（~1.3 vs ~4.5），且下降更快、最终更低（~0.7）——因为：

1. 只对 assistant 段算 loss（样本内大部分是"条件"，不参与损失）；
2. 模型已有语言能力，只需要学"对话模式"这个相对简单的任务；
3. 数据是高质量问答，分布集中。

> 若 SFT 的 loss 起点高于预训练终点太多（例如 3+），说明 `from_weight` 没生效或权重路径不对，检查 `out/pretrain_768.pth` 是否存在。

### 实验 6.2：测试对话能力

```bash
cd ..
python eval_llm.py --weight full_sft
```

对比第 5 章实验 5.2 的预训练输出，观察质变：

```text
💬: 解释什么是机器学习
🧠: 机器学习是人工智能的核心技术之一，通过算法让计算机从数据中学习规律，
    并持续改进预测或决策效果，常见应用包括推荐系统、图像识别、语音识别和自然语言处理。
```

- 输出**以回答句结尾**（不再无限接龙）；
- 格式规范：一问一答，不再自言自语；
- 偶尔仍有知识错误或重复——64M 模型的正常水平，也是后续 RL 阶段要改善的。

### 实验 6.3：对比"只预训练"与"预训练+SFT"（理解 SFT 的必要性）

```bash
# 用两个权重分别测同一批问题
python eval_llm.py --weight pretrain
python eval_llm.py --weight full_sft
```

记录两者对同一问题的回答差异，回答三个问题：格式差异是什么？内容差异是什么？哪个更像"助手"？

## 6.5 深入：灾难性遗忘与对策

SFT 的一个核心风险是**灾难性遗忘**（catastrophic forgetting）：新任务学习覆盖旧知识。对策在 MiniMind 及业界中的实践：

| 对策 | 原理 | MiniMind 中的体现 |
|------|------|-------------------|
| 低学习率 | 参数移动幅度小，旧知识扰动小 | SFT lr=1e-5 |
| 混合通用数据 | SFT 数据中掺入通用/预训练风格样本 | `sft_t2t` 数据混入 Tool Call、reasoning 等多类型 |
| LoRA/冻结部分层 | 只更新少量参数 | 第 8 章 |
| 少量多轮 | 每个 epoch 训练遍数少（1~2） | 默认 epochs=2 |
| 学习率退火 | 后期学习率极低，收敛不震荡 | 余弦退火 |

> 有趣的细节：MiniMind README 提到主线 SFT 数据达 14GB（`sft_t2t.jsonl`），规模已接近"mid training"——数据量大到一定程度，SFT 本身也在继续注入知识。这说明 SFT 与预训练的边界在实践中是连续的。

## 6.6 本章小结

- SFT = 用高质量问答对在预训练模型上做低学习率微调，只学 assistant 段；
- 与预训练的三处差异：数据（掩码策略）、学习率（1e-5）、初始权重（pretrain）；
- 掩码策略是 LLM 对齐工程的基石：模型只学习"回答"，不学习"提问"；
- 训练循环代码与预训练完全相同，差异封装在数据集类——值得学习的架构设计；
- SFT 后模型获得对话能力，但知识量不变；loss 起点 ~1.3、终点 ~0.7 是正常形态。

## 6.7 思考题

1. 如果 SFT 时把 `from_weight` 设为 `none`（随机初始化训练），会出现什么现象？请结合 loss 曲线推理。
2. SFT 数据中混入大量"简单问候"样本（如"你好→你好！"），对模型整体能力有什么影响？（提示：数据分布）
3. 为什么 SFT 的有效 batch（16）远小于预训练（256）是合理的？结合"样本价值密度"分析。
4. `post_processing_chat` 以 80% 概率移除空 think——如果概率改为 0（从不移除），推理时模型行为会有什么变化？
5. （动手）用 `--learning_rate 5e-4` 训练 SFT 500 步，与正常 lr 对比测试回答质量，验证 6.2.1 的论述。

## 6.8 拓展阅读

- 《Training language models to follow instructions with human feedback》（InstructGPT, 2022）——SFT 在 RLHF 流水线中的角色
- 《LIMA: Less Is More for Alignment》（2023）——"数据质量 > 数据量"的对齐研究
- MiniMind README"SFT"一节（含 14GB 主线数据的定位讨论）
