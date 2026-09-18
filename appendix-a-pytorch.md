# 附录 A：PyTorch 速查（本教程用到的 API）

> 用途：学习本教程时随手查阅。只收录 MiniMind 代码中出现过的 API，按使用场景分类。

## A.1 张量基础

```python
torch.tensor([1, 2, 3])                 # 从列表创建
torch.zeros(B, T, dtype=torch.long)     # 全零
torch.ones_like(x)                      # 形状同 x 的全一
torch.arange(0, 10)                     # [0,1,...,9]
torch.randperm(n)                       # 0..n-1 的随机排列
torch.cat([a, b], dim=1)                # 沿 dim 拼接
torch.stack([a, b], dim=1)              # 新增一维堆叠
x.view(B, T, -1)                        # 变形（-1 自动推断）
x.unsqueeze(1) / x.squeeze(-1)          # 增维 / 去维
x.transpose(1, 2)                       # 交换两个维度
x.clone() / x.detach()                  # 拷贝 / 脱离计算图
x.contiguous()                          # 内存连续化（gather/view 前常用）
x.repeat(n, 1)                          # 复制（拷贝数据）
x.expand(B, T, n, D)                    # 广播视图（不拷贝）
x.is_inference()                        # 是否 inference tensor（MiniMind 兼容性检查）
```

## A.2 形状操作与索引

```python
x[..., -1, :]             # 取最后一维的最后一个位置（生成用 logits）
x[:, :-1, :]              # 去掉最后一个位置（logits/labels 对齐）
x[:, 1:]                  # 去掉第一个位置
x[mask]                   # 布尔索引
x.index_add_(0, idx, v)   # 原地按 idx 累加（MoE 聚合）
torch.gather(x, dim=2, index=idx)   # 按索引取元素（logprob 提取的核心）
```

## A.3 数学函数

```python
torch.rsqrt(x)            # 1/sqrt(x)（RMSNorm）
x.pow(2).mean(-1)         # 均方（RMSNorm）
torch.softmax(x, dim=-1)
torch.log_softmax(x, dim=-1)
torch.logsigmoid(x)       # log(sigmoid(x))（DPO 损失）
F.cross_entropy(logits, labels, ignore_index=-100)   # 交叉熵（LLM 主损失）
F.kl_div(log_p, p, reduction='batchmean')            # KL 散度（蒸馏）
F.scaled_dot_product_attention(q, k, v, is_causal=True)  # Flash Attention 内核
torch.topk(x, k, dim=-1)  # 前 k 大（top-k 采样、MoE 路由）
torch.multinomial(probs, num_samples=1)  # 按分布采样
torch.clamp(x, min, max)  # 裁剪（PPO clip）
torch.where(cond, a, b)   # 条件选择（重复惩罚）
torch.unique(x)           # 去重（重复惩罚）
torch.outer(a, b)         # 外积（RoPE 频率表）
```

## A.4 神经网络模块

```python
nn.Linear(in, out, bias=False)
nn.Embedding(vocab, dim)
nn.ModuleList([...])          # 层列表（8 个 block）
nn.Parameter(torch.ones(dim)) # 可学习参数（RMSNorm 的 γ）
nn.Dropout(p)
module.state_dict()           # 参数快照
module.load_state_dict(sd, strict=False)  # strict=False 允许缺失（critic value_head）
module.named_parameters()     # (name, param) 迭代（冻结/统计）
module.named_modules()        # 递归遍历子模块（LoRA 注入）
module.train() / module.eval()
module.requires_grad_(False)  # 冻结整个模型
module.to(device) / model.half()
```

## A.5 优化器与调度

```python
optim.AdamW(params, lr=1e-5)
optimizer.step() / optimizer.zero_grad(set_to_none=True)
scheduler = CosineAnnealingLR(opt, T_max=N, eta_min=lr/10)   # 官方余弦
torch.nn.utils.clip_grad_norm_(params, max_norm)             # 梯度裁剪
```

## A.6 混合精度

```python
torch.cuda.amp.autocast(dtype=torch.bfloat16)   # 前向/反向上下文
torch.cuda.amp.GradScaler(enabled=True)          # fp16 专用
scaler.scale(loss).backward()
scaler.unscale_(optimizer)                        # 裁剪前必须先 unscale
scaler.step(optimizer); scaler.update()
```

## A.7 数据加载

```python
from torch.utils.data import Dataset, DataLoader, DistributedSampler
class MyDS(Dataset): __len__/__getitem__     # 自定义数据集
DataLoader(ds, batch_size=32, num_workers=8, pin_memory=True)
DistributedSampler(ds); sampler.set_epoch(epoch)
padding_side="left"    # tokenizer：RL 采样必须左侧 padding
```

## A.8 分布式

```python
import torch.distributed as dist
dist.init_process_group(backend="nccl")
dist.get_rank() / dist.get_world_size()
dist.barrier() / dist.destroy_process_group()
torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
model.module          # DDP 解包
dist.all_reduce(t, op=dist.ReduceOp.AVG)   # 跨卡平均（PPO 早停判断）
# 启动：torchrun --nproc_per_node N train_xxx.py
```

## A.9 Transformers 集成

```python
AutoTokenizer.from_pretrained(path)
tokenizer.encode/decode/apply_chat_template(..., open_thinking=..., tools=...)
PreTrainedModel / GenerationMixin / PretrainedConfig   # MiniMind 的继承基类
AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)
TextStreamer(tokenizer, skip_prompt=True)               # 流式输出
```

## A.10 调试技巧

```python
x.item()                     # 0 维张量转 Python 标量（打印 loss）
x.shape / x.dtype / x.device # 形状检查三连
torch.autograd.set_detect_anomaly(True)   # NaN 梯度定位（慢，仅调试）
import pdb; pdb.set_trace()  # 断点
# 随机权重 loss 锚点：cross_entropy ≈ ln(vocab_size) ≈ 8.76（vocab=6400）
```
