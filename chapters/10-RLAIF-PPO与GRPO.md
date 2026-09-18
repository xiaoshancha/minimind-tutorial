# 第 10 章 RLAIF：PPO 与 GRPO 强化学习

> 对应代码：`trainer/train_ppo.py`（451 行）、`trainer/train_grpo.py`（334 行）、`trainer/rollout_engine.py`（224 行）、`trainer/trainer_utils.py` 的 `LMForRewardModel`
>
> 本章目标：理解 RLAIF（基于 AI 反馈的强化学习）的整体框架——Rollout（采样）→ Reward（打分）→ Policy Update（更新）；从原理推导 PPO 的 actor-critic 与 GRPO 的分组优势，对照代码理解二者的异同；理解 rollout_engine 的可插拔设计。

---

## 10.1 RLAIF 总览：一个循环

与 DPO（静态偏好数据）不同，RLAIF 是**在线学习**：模型自己生成回答 → 自动评估 → 用评估结果更新自己。

```text
┌─────────────────────────────────────────────────────┐
│  RLAIF 训练循环（每步）                              │
│                                                     │
│  ① Rollout：策略模型对 prompt 生成 N 个回答           │
│      ↓                                              │
│  ② Reward：AI/规则给每个回答打分                     │
│      ↓                                              │
│  ③ Advantage：组内相对优势（或 GAE 优势）             │
│      ↓                                              │
│  ④ Update：策略梯度更新（PPO 裁剪 / GRPO）           │
│      ↓                                              │
│  └──────── 回到 ①（用更新后的模型）─────────────────┘ │
└─────────────────────────────────────────────────────┘
```

**"AI 反馈"体现在 ②**：MiniMind 的奖励来源组合（`calculate_rewards`）：

| 奖励项 | 类型 | 说明 |
|--------|------|------|
| 长度分 | 规则 | 20~800 字符给 +0.5，否则 -0.5（鼓励完整回答） |
| 思考分 | 规则 | 有 `<think>` 且长度 20~300 给 +1.0；思考闭合（只出现一次 `</think>`）+0.25 |
| 重复惩罚 | 规则 | 3-gram 重复率惩罚（`rep_penalty`） |
| **RM 模型分** | AI 模型 | `LMForRewardModel`：调用外部奖励模型打分，截断到 [-3, 3] |

这里没有人类参与——全部信号可自动化，这就是 RLAIF 与 RLHF 的本质区别。

## 10.2 先读懂公共设施：rollout_engine.py

### 10.2.1 可插拔的采样引擎（第 51~61 行）

```python
class RolloutEngine(ABC):
    @abstractmethod
    def rollout(self, prompt_ids, attention_mask, num_generations, max_new_tokens, temperature) -> RolloutResult: ...
    @abstractmethod
    def update_policy(self, model): ...
```

两个实现：

- **`TorchRolloutEngine`**（第 64~95 行）：直接用策略模型 `generate` 采样（第 7 章逻辑），适合小模型单机；
- **`SGLangRolloutEngine`**（第 99~205 行）：通过 HTTP 调用 SGLang 推理服务采样（`return_logprob` 直接返回 logprob，可处理大模型），训练中定期 `update_weights_from_disk` 同步策略权重。

**这个抽象的价值**：采样是 RL 最贵的环节（每步要生成 num_generations 条完整回答），把它从训练循环里解耦——训练代码不变，只换引擎（`--rollout_engine torch|sglang`）。

### 10.2.2 逐 token logprob 的计算（第 24~36 行）

```python
def compute_per_token_logps(model, input_ids, n_keep, attention_mask=None):
    unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
    logits = unwrapped(input_ids, attention_mask=attention_mask, logits_to_keep=n_keep + 1).logits[:, :-1, :]
    per_token_logps = []
    for logits_row, ids_row in zip(logits, input_ids[:, -n_keep:]):
        per_token_logps.append(
            torch.gather(logits_row.log_softmax(dim=-1), 1, ids_row.unsqueeze(1)).squeeze(1)
        )
    return torch.stack(per_token_logps)
```

- **`logits_to_keep=n_keep+1`**：第 4 章埋的伏笔在这里兑现——只需要"生成部分"的 logits（+1 是因为预测下一个 token 需要多看一位），**省掉对 prompt 部分的全量 logits 计算**，显存大幅下降；
- 这些 logprob 是"采样时的旧策略概率"（`old_per_token_logps`），后续更新要做重要性采样比值。

### 10.2.3 RolloutResult 数据结构（第 40~47 行）

```python
@dataclass
class RolloutResult:
    output_ids: Tensor        # 完整序列 (prompt + 生成)
    completion_ids: Tensor    # 仅生成部分
    per_token_logps: Tensor   # 旧策略下每个生成 token 的 logprob
    completions: List[str]    # 解码后的文本
    prompt_lens: Tensor       # 每个样本的 prompt 长度
    completion_mask: Tensor   # 生成部分的掩码（1=有效，0=padding）
```

一个数据结构把"采样结果 + 旧概率 + 对齐信息"打包，训练循环开箱即用——**这是 RL 工程里"一次采样、多处使用"的规范做法**。

## 10.3 GRPO：无 Critic 的组相对策略优化

`train_grpo.py` 是当前更推荐、更简洁的实现（DeepSeek-R1 系列采用），先讲它。

### 10.3.1 原理：组内相对优势

GRPO（Shao et al., 2024）的洞见：**不需要价值网络（Critic）估计绝对优势，用"同组多次采样的相对表现"代替**。

对每个 prompt 采样 $G$ 个回答（MiniMind `num_generations=6`），组内归一化：

$$
A_i = \frac{r_i - \text{mean}(r_1, \dots, r_G)}{\text{std}(r_1, \dots, r_G) + \epsilon}
$$

直觉：**这个回答在同组的 6 个回答里是"偏上"还是"偏下"**。优点：

- 省掉 Critic 模型（PPO 最不稳定的部分）；
- 自动适应奖励量纲（无需学习 reward 的绝对尺度）。

对应代码（第 121~124 行）：

```python
grouped_rewards = rewards.view(-1, args.num_generations)      # (B, G)
mean_r = grouped_rewards.mean(dim=1).repeat_interleave(args.num_generations)
std_r = grouped_rewards.std(dim=1, unbiased=False).repeat_interleave(args.num_generations)
advantages = (rewards - mean_r) / (std_r + 1e-4)
```

### 10.3.2 损失函数（第 132~143 行）

```python
kl_div = ref_per_token_logps - per_token_logps
per_token_kl = torch.exp(kl_div) - kl_div - 1                  # KL 惩罚（无偏估计）
ratio = torch.exp(per_token_logps - old_per_token_logps)       # 重要性采样比值
if args.loss_type == "cispo":                                  # CISPO（新实验损失）
    clamped_ratio = torch.clamp(ratio, max=args.epsilon_high).detach()
    per_token_loss = -(clamped_ratio * advantages.unsqueeze(1) * per_token_logps - args.beta * per_token_kl)
else:                                                          # 标准 GRPO
    clipped_ratio = torch.clamp(ratio, 1 - args.epsilon, 1 + args.epsilon)
    per_token_loss1 = ratio * advantages.unsqueeze(1)
    per_token_loss2 = clipped_ratio * advantages.unsqueeze(1)
    per_token_loss = -(torch.min(per_token_loss1, per_token_loss2) - args.beta * per_token_kl)
policy_loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1).clamp(min=1)).mean()
```

**标准 GRPO 分支**与 PPO 的裁剪公式同源（`min(ratio, clip(ratio)) × advantage`），区别只在优势来源。**KL 惩罚用的是无偏估计**：

$$
\text{KL} \approx e^{\log \frac{\pi_{\text{ref}}}{\pi_\theta}} - \log \frac{\pi_{\text{ref}}}{\pi_\theta} - 1
$$

这个形式（而非 $\frac{1}{2}(\log r)^2$ 的近似 KL）方差更小、是真实 KL 的无偏估计，是 GRPO/RLHF 实现的新标准。

**CISPO 分支**：只做**单边裁剪**（`clamp(max=epsilon_high)`，无下限），且直接对 `per_token_logps` 加权——一种避免裁剪偏差的实验性损失（README 提到 CISPO 是项目新引入的算法）。学习时可只关注 GRPO 分支，CISPO 作了解。

### 10.3.3 完整数据流（第 72~105 行）

```python
prompt_inputs = tokenizer(prompts, return_tensors="pt", padding=True,
                          padding_side="left", add_special_tokens=False)   # ★ 左侧 padding！
rollout_result = rollout_engine.rollout(...)                                # 采样
rewards = calculate_rewards(prompts, completions, reward_model)             # 打分
...
res = model_unwrapped(outputs, attention_mask=full_mask)                    # 新策略前向
per_token_logps = F.log_softmax(res.logits[:, :-1, :], dim=-1).gather(2, outputs[:, 1:].unsqueeze(-1)).squeeze(-1).gather(1, logp_pos)
```

关键工程细节：

1. **`padding_side="left"`**：生成时右侧必须对齐当前生成位置，prompt 统一左侧 padding——与 SFT 训练（右侧 padding）相反，这是 RL 采样特有的要求；
2. **`logp_pos = prompt_lens - 1 + arange(R)`**：从全序列 logprob 里**只取生成部分**的位置（因为 prompt 部分的 logits 只用于生成第一个 token，不参与损失）；
3. `old_per_token_logps` 来自 rollout 引擎（旧策略），`per_token_logps` 来自当前前向（新策略）——两者的比值构成 importance ratio；
4. 早期停止相关：`scheduler = CosineAnnealingLR(optimizer, T_max=..., eta_min=lr/10)` 用官方调度器（与 pretrain 的手写余弦等价，但按"优化器步数"调度）。

## 10.4 PPO：Actor-Critic 强化学习

`train_ppo.py` 在 GRPO 的基础上多了 **Critic（价值网络）**。

### 10.4.1 CriticModel（第 36~48 行）

```python
class CriticModel(MiniMindForCausalLM):
    def __init__(self, params):
        super().__init__(params)
        self.value_head = nn.Linear(params.hidden_size, 1)      # 把 lm_head 换成标量价值头

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        hidden_states = self.model.norm(outputs[0])
        values = self.value_head(hidden_states).squeeze(-1)     # (B, T)：每个位置的价值
        return values
```

- 复用 MiniMindForCausalLM 的主干，只把输出头换成单维线性层；
- **从 base 权重初始化**（`load_state_dict(state_dict, strict=False)`——value_head 是新增层，strict=False 允许缺失）；
- 训练目标：预测"每个 token 位置的期望回报"，用于 GAE 优势估计。

### 10.4.2 GAE 优势估计（第 129~150 行）

```python
with torch.no_grad():  # rollout 阶段全部关梯度
    values_seq = critic_for_rollout(input_ids=gen_out, attention_mask=full_mask)
    old_resp_values = values_seq.gather(1, logp_pos) * resp_value_mask
    ...
    token_rewards = torch.zeros_like(old_resp_logp)
    token_rewards[torch.arange(B)[valid_resp], last_idx[valid_resp]] += rewards[valid_resp]  # 只在末尾放外部奖励
    ...
    for t in reversed(range(gen_len)):      # 从后往前递推
        nv = old_resp_values[:, t + 1] if t < gen_len - 1 else 0.0
        delta = token_rewards[:, t] + args.gamma * nv - old_resp_values[:, t]
        lastgaelam = delta + args.gamma * args.lam * lastgaelam
        advs_rev.append(lastgaelam)
    advantages = torch.stack(advs_rev[::-1], dim=1)
    returns = advantages + old_resp_values
```

GAE（Generalized Advantage Estimation）的递推式：

$$
\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t), \quad A_t = \delta_t + \gamma\lambda\, A_{t+1}
$$

- $\gamma$（默认 1.0）：折扣因子，这里设 1 表示"所有位置同等重要"（语言生成场景常见）；
- $\lambda$（默认 0.95）：偏差-方差权衡，越大越依赖长期估计；
- 奖励只在**回答末尾**放一次（`token_rewards[..., last_idx] += rewards`）——中间位置奖励为 0，价值函数负责把奖励"传播"到每个 token 位置；
- 最后做**优势标准化**（第 148~150 行）：减均值除标准差，稳定训练。

### 10.4.3 PPO 更新循环（第 152~235 行）

```python
for ppo_epoch in range(args.ppo_update_iters):        # 同一批数据重复利用 2 次
    b_inds = torch.randperm(B, device=args.device)
    for i in range(0, B, mb_size):                    # mini-batch 更新
        ...
        mb_resp_logp = F.log_softmax(res.logits[:, :-1], dim=-1)...  # 新策略
        log_ratio = mb_resp_logp - old_resp_logp[inds]
        ratio = torch.exp(log_ratio)
        clipfrac = (((ratio - 1.0).abs() > args.clip_epsilon).float() * mask).sum() / mask.sum()
        policy_loss = max(-adv*ratio, -adv*clip(ratio))  # 裁剪目标
        value_loss = 0.5 * max((V-return)², (clip(V)-return)²)   # 价值裁剪
        kl_ref_penalty = (exp(ref-policy) - (ref-policy) - 1)    # 参考 KL
        if stop_ppo:
            loss = (policy_loss + args.vf_coef * value_loss + aux_loss) * 0.0   # 早停但保 DDP 通信闭环
        else:
            loss = (policy_loss + args.vf_coef * value_loss + aux_loss) / args.accumulation_steps
        loss.backward()
```

PPO 经典组件在代码中的位置：

| 组件 | 公式 | 代码位置 |
|------|------|----------|
| 重要性比值 | $r_t = \pi_\theta/\pi_{\theta_{old}}$ | `ratio = exp(log_ratio)` |
| 裁剪目标 | $\min(r_t A_t, \text{clip}(r_t, 1\pm\epsilon) A_t)$ | `torch.max(-adv*ratio, -adv*clipped)` |
| 价值裁剪 | $\max((V-\hat R)^2, (\text{clip}(V)-\hat R)^2)$ | value_loss |
| 参考 KL | 无偏 KL 估计 | `kl_ref_penalty` |
| 早停 | KL 超阈值停止本批更新 | `early_stop_kl=0.25` |

**早停的 DDP 细节**（第 196~223 行）：KL 超限时把 loss 乘 0（参数不更新），但**仍然执行 backward**——保证 DDP 的梯度同步通信不被破坏，避免多卡死锁。这是分布式 RL 的高阶工程细节。

### 10.4.4 PPO 与 GRPO 的对比

| 维度 | PPO | GRPO |
|------|-----|------|
| 优势估计 | Critic 价值网络 + GAE | 组内均值归一化 |
| 模型数 | 4（actor/ref/RM/critic） | 3（actor/ref/RM） |
| 超参数 | γ、λ、cliprange_value、vf_coef... | 少得多 |
| 显存 | +critic | 省一个模型 |
| 稳定性 | 经典但脆弱 | 更鲁棒，DeepSeek-R1 验证 |
| MiniMind 默认 | `--save_weight ppo_actor` | `--loss_type grpo/cispo` |

> 训练脚本都在 `trainer/` 下、数据 `rlaif.jsonl`（24MB），**都需要外部 RM**（`--reward_model_path` 默认指向 internlm2-1.8b-reward）。实践上 GRPO 因简洁稳定是首选。

## 10.5 动手实验

### 实验 10.1：GRPO 训练（推荐入门）

```bash
# 前提：full_sft 权重 + rlaif.jsonl + 外部奖励模型（internlm2-1_8b-reward 或自行替换）
cd trainer
python train_grpo.py --debug_mode --log_interval 5 --save_interval 10
```

用 `--debug_mode` 观察每步采样的 prompt、回答与 reward，理解"模型在 RL 中如何被反馈引导"。关键日志指标：

```text
Reward: 1.1234, KL_ref: 0.0542, Adv Std: 0.8231, Adv Mean: 0.0012, Actor Loss: ..., Avg Response Len: 87.32
```

- **Reward 上升**：模型在变"好"（按奖励定义）；
- **KL_ref 适度**：太大说明偏离 ref 太快（β 不够），太小说明没学到东西；
- **Adv Std ≈ 1**：组内归一化正常；
- **Avg Response Len**：观察模型是否学会了"凑长度"——奖励设计的反馈信号。

### 实验 10.2：PPO 训练（对照）

```bash
python train_ppo.py --debug_mode --log_interval 5 --save_interval 10
```

对照日志中的 `Critic Loss`、`ClipFrac`（裁剪比例，过高说明步子太大）与 GRPO 的差异。

### 实验 10.3：RL 前后对比

```bash
python eval_llm.py --weight full_sft
python eval_llm.py --weight grpo        # 或 ppo_actor
```

观察：回答是否更长/更有条理/更符合思考链格式（`<think>` 结构）。**注意**：RL 让模型学会的是"按奖励定义变好"，如果奖励里没有"事实正确性"，模型可能只是学会了"话多"——这正是奖励设计（reward hacking）的经典教训。

## 10.6 本章小结

- RLAIF = 在线循环：Rollout → Reward → Advantage → Update，反馈全部来自 AI/规则；
- rollout_engine 抽象：采样后端可插拔（torch 原生 / SGLang），`logits_to_keep` 省显存，RolloutResult 打包旧概率；
- GRPO：组内归一化优势（省 Critic），`num_generations=6` 组内比较，无偏 KL 惩罚，DeepSeek-R1 同款思路；
- PPO：Critic + GAE 递推优势 + 裁剪目标 + 价值裁剪 + 参考 KL；早停保持 DDP 通信闭环；
- 奖励设计决定训练方向：长度分、思考分、重复惩罚、RM 分各司其职，也埋下 reward hacking 的隐患；
- 学习率 3e-7 量级——RL 阶段是所有训练中最温和的。

## 10.7 思考题

1. GRPO 为什么可以用"组内相对"代替"绝对优势"？如果组内 6 个回答全部很差，会发生什么？
2. PPO 中 Critic 预测的 $V(s_t)$ 与"token 位置"是什么关系？为什么奖励只放在末尾、靠 GAE 向前传播？
3. `padding_side="left"` 在 RL 采样中为什么是必须的？如果用了右侧 padding 会怎样？
4. `logits_to_keep=n_keep+1` 为什么能省显存？省了多少？（提示：只算生成部分 logits）
5. KL 惩罚项用 `exp(x) - x - 1` 而不是 $x^2/2$，各自的统计性质是什么？
6. （动手）修改 `rep_penalty` 的 n-gram 大小（3→5），观察训练中 Avg Response Len 的变化，解释原因。
7. 为什么 RL 学习率（3e-7）比 DPO（4e-8）高但比 SFT（1e-5）低？结合"采样噪声"与"分布漂移"分析。

## 10.8 拓展阅读

- 《DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models》（Shao et al., 2024）——GRPO 原始论文
- 《Proximal Policy Optimization Algorithms》（Schulman et al., 2017）——PPO 论文
- 《High-Dimensional Continuous Control Using Generalized Advantage Estimation》（Schulman et al., 2016）——GAE
- 《Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters》（2024）——测试时计算与 RL 的关系
- MiniMind README 的"PO 算法的统一视角"一节（所有策略优化算法的统一目标函数框架，强烈建议阅读）
