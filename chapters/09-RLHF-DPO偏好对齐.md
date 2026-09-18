# 第 9 章 RLHF-DPO：让模型学会"偏好"

> 对应代码：`trainer/train_dpo.py`（228 行）、`dataset/lm_dataset.py` 的 `DPODataset`
>
> 本章目标：理解偏好对齐（Alignment）的必要性与数学框架（Bradley-Terry 模型），推导 DPO 损失（从 RLHF 的 KL 约束优化出发，消去奖励模型），并逐行读懂 MiniMind 的 DPO 实现。

---

## 9.1 为什么需要偏好对齐

SFT 之后的模型能对话，但存在两个问题：

1. **回答质量参差**：SFT 数据是"标准答案"，但"标准"不等于"好"——模型可能给出冗长、绕圈、甚至有害的回答；
2. **无法表达"相对偏好"**：SFT 只能学"这个回答是对的"，学不了"这个回答比那个更好"。而人类对模型的期望本质上是排序式的（"更愿意看哪个回答"）。

**偏好对齐（Preference Alignment）**就是用"成对偏好"数据（chosen > rejected）继续训练，把模型的输出分布推向人类偏好方向。

### 9.1.1 RLHF 的两阶段路线

经典的 RLHF（InstructGPT）分三步：

1. **SFT**：得到初始模型 $\pi_{\text{SFT}}$；
2. **训练奖励模型（Reward Model, RM）**：用偏好对数据训练一个打分器 $r_\theta(x, y)$，学习"哪个回答更好"；
3. **PPO 强化学习**：用 RM 的分数作为奖励，优化策略模型 $\pi_\theta$（第 10 章）。

这条路线工程复杂：要训练 RM、要调 PPO 的众多超参、要维护 4 个模型（actor/ref/RM/critic）。

### 9.1.2 DPO：跳过奖励模型

**Direct Preference Optimization**（Rafailov et al., 2023）的核心发现：在**隐式奖励模型**的视角下，RLHF 的最优策略有闭式解，可以直接从偏好数据推导出损失函数——**不需要显式训练 RM，也不需要 PPO 采样**。

## 9.2 数学推导（教材级）

### 9.2.1 第一步：偏好建模（Bradley-Terry）

偏好数据形式：$(x, y_w, y_l)$——对提示 $x$，$y_w$（winner/chosen）比 $y_l$（loser/rejected）更被偏好。

Bradley-Terry 模型假设人类偏好服从逻辑斯蒂分布：

$$
P(y_w \succ y_l \mid x) = \sigma\left(r(x, y_w) - r(x, y_l)\right)
$$

其中 $\sigma$ 是 sigmoid，$r(x, y)$ 是隐式"质量分数"（reward）。这个假设把"排序"转化为"分数差"，是后续一切推导的地基。

### 9.2.2 第二步：RLHF 的优化目标

在 KL 约束下最大化奖励期望（限制策略不能偏离参考模型太远）：

$$
\max_{\pi_\theta} \; \mathbb{E}_{x, y \sim \pi_\theta(\cdot|x)} \left[ r(x, y) \right] - \beta \, \mathbb{D}_{\text{KL}}\left[\pi_\theta(\cdot|x) \,\|\, \pi_{\text{ref}}(\cdot|x)\right]
$$

这个带约束的优化有**解析解**（拉格朗日法）：

$$
\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left(\frac{r(x,y)}{\beta}\right)
$$

反解出隐式奖励：

$$
r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)
$$

### 9.2.3 第三步：代入 Bradley-Terry

把隐式奖励代入 BT 模型，**$Z(x)$ 项恰好抵消**（同一提示 x 下）：

$$
P(y_w \succ y_l) = \sigma\left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right)
$$

### 9.2.4 第四步：得到 DPO 损失

最大似然估计（最大化偏好对似然），加负号得到损失：

$$
\boxed{\, \mathcal{L}_{\text{DPO}}(\pi_\theta) = - \mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma\left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right] \,}
$$

**直觉解读**：

- 目标：提高 chosen 的概率、降低 rejected 的概率；
- 若策略已经"偏好 chosen 多于 ref 模型"（比值 > 1），损失小；否则损失大；
- $\beta$（MiniMind 默认 0.15）控制对参考模型的约束强度：$\beta$ 越大，策略越不敢偏离 ref；
- **全程只需 π_θ 和 π_ref 两个模型**，π_ref 冻结——这就是"免 RM、免采样"的来源。

## 9.3 DPODataset：偏好数据的准备（lm_dataset.py 第 122~192 行）

```python
class DPODataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=4096):
        ...
        self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', add_special_tokens=False).input_ids
        self.eos_id = tokenizer(f'{tokenizer.eos_token}\n', add_special_tokens=False).input_ids
        self.samples = load_dataset('json', data_files=file_path, split='train')

    def __getitem__(self, index):
        sample = self.samples[index]
        chosen_prompt = self.tokenizer.apply_chat_template(sample['chosen'], tokenize=False, add_generation_prompt=False)
        rejected_prompt = self.tokenizer.apply_chat_template(sample['rejected'], tokenize=False, add_generation_prompt=False)
        chosen_encoding = self.tokenizer(chosen_prompt, truncation=True, max_length=self.max_length, padding='max_length')
        rejected_encoding = self.tokenizer(rejected_prompt, truncation=True, max_length=self.max_length, padding='max_length')
        ...
        return {
            'x_chosen': ..., 'y_chosen': ..., 'mask_chosen': ...,
            'x_rejected': ..., 'y_rejected': ..., 'mask_rejected': ...
        }
```

要点：

1. 一条样本 = chosen（好回答）+ rejected（差回答），各自模板化、编码、掩码（`generate_loss_mask` 只标 assistant 段，第 3 章已讲）；
2. 返回 `x`（输入，[:-1]）与 `y`（标签，[1:]）——直接给模型前向用，且**不经过模型 forward 的 loss 计算**（DPO 需要手工取 logprob，见下）；
3. `max_length=4096`：偏好数据通常较长，DPO 阶段序列预算比 SFT 宽裕。

## 9.4 train_dpo.py 逐行精读

### 9.4.1 核心辅助函数（第 25~50 行）

```python
def logits_to_log_probs(logits, labels):
    # logits: (B, seq, V), labels: (B, seq) → log_probs: (B, seq)
    log_probs = F.log_softmax(logits, dim=2)
    log_probs_per_token = torch.gather(log_probs, dim=2, index=labels.unsqueeze(2)).squeeze(-1)
    return log_probs_per_token
```

`torch.gather`：按 labels 取每个位置真实 token 的 log 概率。**注意这里的 labels 是输入序列本身右移**——`y = x[:, 1:]`，logits 对应的 `[:-1]` 切片在训练循环里完成。

```python
def dpo_loss(ref_log_probs, policy_log_probs, mask, beta):
    ref_log_probs = (ref_log_probs * mask).sum(dim=1)       # 序列求和 → 整句对数概率
    policy_log_probs = (policy_log_probs * mask).sum(dim=1)
    batch_size = ref_log_probs.shape[0]
    chosen_ref = ref_log_probs[:batch_size // 2]            # 前半是 chosen，后半是 rejected
    reject_ref = ref_log_probs[batch_size // 2:]
    chosen_policy = policy_log_probs[:batch_size // 2]
    reject_policy = policy_log_probs[batch_size // 2:]
    pi_logratios = chosen_policy - reject_policy
    ref_logratios = chosen_ref - reject_ref
    logits = pi_logratios - ref_logratios                   # = β 前的核心量
    loss = -F.logsigmoid(beta * logits)
    return loss.mean()
```

**逐行对照公式 9.2.4**：

- `chosen_policy - reject_policy` = $\log \frac{\pi_\theta(y_w|x)}{\pi_\theta(y_l|x)}$；
- `chosen_ref - reject_ref` = $\log \frac{\pi_{\text{ref}}(y_w|x)}{\pi_{\text{ref}}(y_l|x)}$；
- `logits` = 两者的差（隐式奖励差）；`-log sigmoid(β·logits)` = 负对数似然；
- **掩码只在序列求和时使用**（`mask` 是 assistant 段掩码）——padding 和 user 段不贡献 logprob 和；
- 拼接技巧：`torch.cat([x_chosen, x_rejected])` 一次前向算两份，`[:B/2]`/`[B/2:]` 切分，省一次前向。

### 9.4.2 训练循环（第 53~128 行）

```python
with autocast_ctx:
    with torch.no_grad():
        ref_outputs = ref_model(x)                     # 参考模型：冻结，无梯度
        ref_logits = ref_outputs.logits
    ref_log_probs = logits_to_log_probs(ref_logits, y)

    outputs = model(x)                                 # 策略模型：正常前向
    logits = outputs.logits
    policy_log_probs = logits_to_log_probs(logits, y)

    dpo_loss_val = dpo_loss(ref_log_probs, policy_log_probs, mask, beta=beta)
    loss = dpo_loss_val + outputs.aux_loss
```

双模型结构：

- `model`（策略 π_θ）：训练，更新参数；
- `ref_model`（参考 π_ref）：**从同一权重初始化、全程冻结**（`requires_grad_(False)` + `torch.no_grad()`），提供"偏离基准"；
- 注意：DPO 训练**没有生成采样**（不同于 PPO/GRPO），用的是**静态偏好数据**——成本接近普通 SFT。

### 9.4.3 参数与工程细节（第 131~156 行）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--learning_rate` | **4e-8** | 比 SFT(1e-5) 低两个数量级！ |
| `--beta` | 0.15 | KL 约束强度 |
| `--batch_size` | 4 | 很小（显存敏感：序列长 + 双模型） |
| `--epochs` | 1 | 只训一遍，防过拟合 |
| `--from_weight` | `full_sft` | 在 SFT 模型上做对齐 |

**为什么 DPO 学习率低到 4e-8？**

- DPO 的梯度方向是"推开 rejected、拉近 chosen"，对分布细节极其敏感；
- 高学习率会迅速把策略推离 ref（KL 爆炸），表现是输出崩坏/退化；
- 经验：DPO 是"极度温和的修正"，宁可多训几轮低 lr，不可一次大步长。这是**全教程学习率最低的环节**。

### 9.4.4 对比 PPO 的成本优势

| 环节 | DPO | PPO（第 10 章） |
|------|-----|-----------------|
| 奖励信号 | 静态偏好对（免费） | 需 RM/规则在线打分 |
| 生成采样 | 无 | 每步 rollout |
| 模型数 | 2（策略+ref） | 4（actor+ref+RM+critic） |
| 训练稳定性 | 高 | 低（需大量调参） |
| 效果上限 | 受静态数据限制 | 可在线探索，更强 |

## 9.5 动手实验

### 实验 9.1：DPO 训练

```bash
# 前提：已有 full_sft 权重 + dpo.jsonl 数据（53MB，下载放入 dataset/）
cd trainer
python train_dpo.py
```

预期日志（注意 dpo_loss 与 lr 的量级）：

```text
Epoch:[1/1](100/…), loss: 0.6800, dpo_loss: 0.6800, aux_loss: 0.0000, learning_rate: 0.00000004, ...
```

loss 解读：

- 起点 ≈ log 2 ≈ 0.693（sigmoid 输入为 0 时的损失——**初始化时策略 = ref，logits 恰好为 0**，这是 DPO 的天然锚点，可用来检查代码正确性）；
- 下降缓慢（lr 极小）：从 0.693 降到 0.6x 量级即算有效；
- 若 loss 快速降到 0.3 以下，警惕过拟合（KL 爆炸）。

### 实验 9.2：对比 SFT 与 DPO 的回答

```bash
python eval_llm.py --weight full_sft
python eval_llm.py --weight dpo
```

用同一批问题对比，重点观察：回答长度、礼貌度、是否拒绝不当请求、是否更简洁。

## 9.6 本章小结

- 偏好对齐解决"SFT 只学标准答案、学不了相对偏好"的问题；
- RLHF 三步走（SFT→RM→PPO）；DPO 通过 Bradley-Terry + 隐式奖励推导，**免 RM、免采样**；
- DPO 损失 = 最大化 chosen/rejected 的策略比值 vs 参考比值的 sigmoid 似然；
- 实现三件套：双模型（策略 + 冻结 ref）、logprob 掩码求和、batch 拼接一次前向；
- 学习率 4e-8 是"温和修正"哲学的极致体现；初始 loss ≈ ln 2 是正确性锚点。

## 9.7 思考题

1. 推导 DPO 时，为什么 $Z(x)$（配分函数）会在代入 BT 模型后抵消？这个抵消在数学上依赖什么条件？
2. 如果 ref 模型与策略模型**不**从同一权重初始化，DPO 会发生什么？
3. β=0.15 与 β=2.0 分别对应什么样的"约束心态"？训练曲线会有什么差异？
4. DPO 数据中 chosen 和 rejected 的**长度差异**很大时（如 chosen 300 词、rejected 10 词），损失会偏向哪边？有什么隐患？
5. （动手）验证初始化时 dpo_loss ≈ ln 2：在训练前打印第一个 batch 的 loss。
6. DPO 与 SFT 的 loss 都涉及 assistant 段掩码，但 DPO 为什么要**整句求和**而不是逐位置平均？（提示：BT 模型的对象是完整回答）

## 9.8 拓展阅读

- 《Direct Preference Optimization: Your Language Model is Secretly a Reward Model》（Rafailov et al., 2023）——DPO 原始论文（推导非常清晰，建议精读附录）
- 《Training language models to follow instructions with human feedback》（InstructGPT, 2022）——RLHF 范式原文
- 《A General Theoretical Paradigm to Understand Learning from Human Preferences》（Azar et al., 2023）——IPO，DPO 的改良派
- MiniMind README 的 RL 数据一节（dpo.jsonl 的 chosen/rejected 格式）
