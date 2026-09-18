# 第 8 章 LoRA：参数高效微调（纯手写实现）

> 对应代码：`model/model_lora.py`（65 行）、`trainer/train_lora.py`（186 行）
>
> 本章目标：理解低秩适配（Low-Rank Adaptation, LoRA）的数学原理，读懂 MiniMind 不依赖 peft 的纯手写 LoRA 实现（monkey-patch 注入、冻结主干、低秩合并），并动手完成一次垂域/自我认知微调。

---

## 8.1 动机：为什么需要 LoRA

第 6 章的全参数 SFT 有两个痛点：

1. **成本高**：64M 模型全参微调需要保存和更新全部参数 + 优化器状态（约 1.5~3 GB 显存）；换成 7B 模型就是百 GB 量级，个人无法负担；
2. **每个任务存一份完整模型**：医疗模型、法律模型、客服模型……各存一份 7B 权重，存储和分发成本爆炸。

LoRA（Hu et al., 2021）的洞见：**模型在微调时，权重的变化 ΔW 是"低秩"的**——真正需要的自由度远小于矩阵本身的大小。因此可以用两个小矩阵近似：

$$
W' = W_0 + \Delta W = W_0 + \frac{\alpha}{r} B A
$$

其中 $W_0 \in \mathbb{R}^{d \times d}$ 冻结不动，$A \in \mathbb{R}^{r \times d}$、$B \in \mathbb{R}^{d \times r}$ 是待训练的低秩矩阵，$r \ll d$。

- 参数量：从 $d^2$ 降到 $2dr$（r=16 时，768² → 2×768×16，省 **24 倍**）；
- 训练时只需给 A、B 计算梯度和优化器状态；
- **推理时零额外开销**：可以把 BA 合并回 W（`merge_lora`），或者运行时加算（`forward` 补丁）；
- 多个任务可以共享同一个基模，各挂各的 LoRA（第 7 章 eval_llm.py 的 `--lora_weight` 就是这么用的）。

### 8.1.1 初始化技巧：为什么 B 置零、A 随机？

```python
self.A.weight.data.normal_(mean=0.0, std=0.02)   # A 高斯初始化
self.B.weight.data.zero_()                        # B 全零初始化
```

微调开始瞬间必须有 $BA = 0 \Rightarrow W' = W_0$——**模型起点与基模完全一致**，训练是"从基模出发的平滑偏离"。如果 A、B 都随机初始化，模型一开局就被随机扰动，会破坏基模能力。这是 LoRA 论文的关键设计（与残差连接的初始化哲学一致）。

## 8.2 纯手写 LoRA：model_lora.py 逐行精读

### 8.2.1 LoRA 模块本体（第 6~18 行）

```python
class LoRA(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super().__init__()
        self.rank = rank
        self.A = nn.Linear(in_features, rank, bias=False)   # (r, d)
        self.B = nn.Linear(rank, out_features, bias=False)  # (d, r)
        self.A.weight.data.normal_(mean=0.0, std=0.02)
        self.B.weight.data.zero_()

    def forward(self, x):
        return self.B(self.A(x))    # 低秩路径：d → r → d
```

注意 `in_features == out_features` 才有意义（方阵），这是 `apply_lora` 的筛选条件。

### 8.2.2 注入机制：monkey-patch forward（第 21~32 行）

这是整个实现最精巧的部分——**不修改原模型任何代码，动态替换 forward**：

```python
def apply_lora(model, rank=16):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module.in_features == module.out_features:
            lora = LoRA(module.in_features, module.out_features, rank=rank).to(model.device)
            setattr(module, "lora", lora)                 # 把 LoRA 挂到 Linear 上
            original_forward = module.forward

            def forward_with_lora(x, layer1=original_forward, layer2=lora):
                return layer1(x) + layer2(x)              # 原路径 + 低秩路径

            module.forward = forward_with_lora
```

三个细节值得展开：

1. **闭包陷阱的规避**：`forward_with_lora(x, layer1=original_forward, layer2=lora)` 用**默认参数绑定**捕获原 forward 和 LoRA 模块。若直接引用外层变量 `module`/`lora`，循环结束后所有补丁会共享最后一个模块——这是 Python 闭包晚绑定的经典坑，作者用默认参数做了早绑定；
2. **作用范围**：只处理**方阵** Linear（`in_features == out_features`）——q/k/v/o 投影和 gate/up/down 都是方阵，恰好全覆盖；embedding/lm_head 是 6400×768 非方阵，跳过（LoRA 论文同样不动 embedding）；
3. **为什么可行**：PyTorch 的 `module.forward` 只是属性，替换后 `module(x)` 调用新逻辑，原权重、state_dict、DDP 全部不受影响。

### 8.2.3 保存 / 加载 / 合并（第 35~65 行）

```python
def save_lora(model, path):
    state_dict = {}
    for name, module in raw_model.named_modules():
        if hasattr(module, 'lora'):
            lora_state = {f'{clean_name}.lora.{k}': v.cpu().half() for k, v in module.lora.state_dict().items()}
            state_dict.update(lora_state)
    torch.save(state_dict, path)
```

只保存 `xxx.lora.A/B` 权重（每个方阵 Linear 两个小矩阵），文件极小（r=16 时 16 个模块 × 24576 参数 × 2 字节 ≈ **0.8 MB**，对比基模 fp16 权重约 128 MB）。

```python
def merge_lora(model, lora_path, save_path):
    load_lora(model, lora_path)
    state_dict = {k: v for k, v in raw_model.state_dict().items() if '.lora.' not in k}
    for name, module in raw_model.named_modules():
        if isinstance(module, nn.Linear) and '.lora.' not in name:
            state_dict[f'{name}.weight'] = module.weight.data.clone()
            if hasattr(module, 'lora'):
                state_dict[f'{name}.weight'] += (module.lora.B.weight.data @ module.lora.A.weight.data)  # ΔW = BA
    torch.save(state_dict, save_path)
```

合并数学：$W' = W_0 + B \cdot A$（注意 $\alpha/r$ 缩放已包含在训练中——MiniMind 直接以学习率缩放，等价于固定 $\alpha=1$ 的实现）。合并后的权重就是"完整微调后的模型"，可用第 4 章的标准加载流程（`convert_model.py` 也提供此功能）。

### 8.2.4 缩放因子 α/r 去哪了？

标准 LoRA 有超参数 $\alpha$（`lora_alpha`），MiniMind 的简化实现**没有显式 α**，等效于 $\alpha = r$。这带来的实际差异：更新幅度 = BA 的原始乘积。对小模型 + 低学习率（1e-4）实践上完全够用——**这是"删繁就简"的教学取舍**，理解时知道标准写法即可。

## 8.3 train_lora.py 与全参 SFT 的差异

对比第 6 章，train_lora.py 的差异集中在**参数冻结**（第 128~152 行）：

```python
model, tokenizer = init_model(lm_config, args.from_weight, device=args.device)  # 默认从 full_sft 起训
apply_lora(model)

# 冻结非 LoRA 参数，收集 LoRA 参数
lora_params = []
for name, param in model.named_parameters():
    if 'lora' in name:
        param.requires_grad = True
        lora_params.append(param)
    else:
        param.requires_grad = False
...
optimizer = optim.AdamW(lora_params, lr=args.learning_rate)   # 优化器只管 LoRA 参数
```

关键点：

1. **优化器只接收 `lora_params`**——backward 时 PyTorch 只为 `requires_grad=True` 的参数计算梯度（非 LoRA 参数梯度为 None，不占优化器状态）；
2. 学习率 1e-4（比全参 SFT 高 10 倍）：**低秩子空间容量小，可以承受更大的步长**；
3. 默认 `from_weight='full_sft'`：在对话模型上做垂域适配（数据 `lora_medical.jsonl`）；
4. `use_compile` 自动关闭（第 165~166 行）：**torch.compile 对 monkey-patch 的 forward 不兼容**——这是手写方案的真实约束，README 与代码都明确说明；
5. 训练循环与全参 SFT 相同（损失、调度、AMP、断点续训），只把 `clip_grad_norm_` 的目标换成 `lora_params`；
6. 保存时只存 LoRA 权重 + 完整 checkpoint（断点续训恢复 LoRA 状态）。

> 注意 `save_lora` 里 `raw_model = getattr(model, '_orig_mod', model)`：DDP 包装的模型要先解包（`.module`），torch.compile 包装的模型要先取 `_orig_mod`——与第 5 章保存逻辑同样的解包套路。

## 8.4 动手实验：自我认知微调

### 实验 8.1：准备垂域数据并训练

```bash
# 1) 准备数据：仿照 README 的 lora_identity 格式，创建 dataset/lora_identity.jsonl
#    {"conversations": [{"role": "user", "content": "你是谁"}, {"role": "assistant", "content": "我是 MiniMind，由 ... 开发 ..."}]}
#    数据量：几十到几百条即可见效

# 2) 训练（CPU 也能较快完成）
cd trainer
python train_lora.py --lora_name lora_identity --data_path ../dataset/lora_identity.jsonl --epochs 10
```

预期日志：

```text
LLM 总参数量: 63.910 M
LoRA 参数量: 0.393 M          ← 只训练 ~0.6% 的参数！
LoRA 参数占比: 0.62%
Epoch:[1/10](10/…), loss: 0.9500, ...
```

**验证参数量**（对照 8.2.2 的"只处理方阵"条件，逐层数一遍）：r=16 时，每个方阵 Linear 的 LoRA 参数 = 768×16×2 = 24576。哪些是方阵？

- 每层 Attention：`q_proj`（768→768）✓、`o_proj`（768→768）✓；**k_proj/v_proj 是 768→384（GQA，非方阵）✗**；
- 每层 FFN：`gate/up`（768→2432）、`down`（2432→768）**全部非方阵 ✗**；
- RMSNorm 不是 Linear ✗；lm_head（768→6400）✗。

所以每层仅 2 个模块 × 8 层 = **16 个 LoRA**，16 × 24576 ≈ **0.39M**，与日志打印一致。这也揭示了一个设计事实：**GQA 的 KV 投影与 FFN 不参与 LoRA**——LoRA 主要作用在注意力最"方"的部分。

### 实验 8.2：测试 LoRA 效果

```bash
cd ..
# 加载 base 模型 + LoRA 权重（eval_llm.py 自动完成 apply_lora + load_lora）
python eval_llm.py --weight full_sft --lora_weight lora_identity
```

对比测试：

```bash
# 不加 LoRA 的 base 模型
python eval_llm.py --weight full_sft
```

提问"你是谁"，对比两个模型的回答差异：**base 模型不认识"MiniMind"身份，加了 identity LoRA 后能按定制身份回答**——这就是"垂域适配"的最小演示。

### 实验 8.3：LoRA 合并（可选）

```bash
cd scripts && python convert_model.py   # 使用 convert_merge_base_lora 合并基模 + LoRA
```

合并后的权重 ≈ 全参微调模型，可直接 `--weight <新名字>` 加载，无需再挂 LoRA。

## 8.5 本章小结

- LoRA 假设微调增量 ΔW 低秩：$W' = W_0 + BA$，训练成本降一个数量级；
- 初始化纪律：A 随机、B 置零，保证起点 = 基模；
- MiniMind 用 **monkey-patch forward** 实现注入：闭包早绑定、只处理方阵 Linear、不碰原模型代码；
- 冻结主干 → 优化器只管 LoRA 参数；lr 可提到 1e-4；torch.compile 与 patch 方案不兼容；
- 保存只存小权重（~3MB），可运行期叠加、可合并回主模型；
- 业务价值：一个基模 + N 个 LoRA = N 个垂域模型。

## 8.6 思考题

1. 为什么 LoRA 只作用于方阵 Linear？如果对 6400×768 的 embedding 做 LoRA，会有什么问题？
2. `forward_with_lora` 的默认参数绑定 `layer1=original_forward` 解决了什么问题？如果改成闭包引用会发生什么？
3. LoRA 训练时，冻结参数的前向仍然会计算吗？梯度呢？（提示：requires_grad 的作用范围）
4. r（秩）从 16 调到 64，训练参数量、表达能力、过拟合风险各如何变化？
5. （动手）验证：训练前后，`module.weight`（主干权重）完全不变，只有 `lora.A/B` 变化。
6. LoRA 与全参 SFT 的效果差距在什么场景下最大？（提示：任务与基模能力的距离）

## 8.7 拓展阅读

- 《LoRA: Low-Rank Adaptation of Large Language Models》（Hu et al., 2021）——原始论文
- 《QLoRA: Efficient Finetuning of Quantized LLMs》（Dettmers et al., 2023）——LoRA + 量化，4bit 微调 65B 的工程实践
- peft 库的 LoRA 实现（对比 MiniMind 手写版，理解工业封装做了哪些事）
- MiniMind README"LoRA"一节（垂域与自我认知数据示例）
