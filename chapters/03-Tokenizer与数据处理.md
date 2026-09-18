# 第 3 章 Tokenizer 与数据处理

> 对应代码：`trainer/train_tokenizer.py`、`dataset/lm_dataset.py`、`model/tokenizer_config.json`、`model/tokenizer.json`
>
> 本章目标：理解"文本 ↔ token"之间的桥梁——BPE 算法的原理与训练流程、MiniMind 自定义 tokenizer 的设计（6400 词表、特殊 token 体系、chat_template 渲染规则），以及预训练/SFT 数据集类的实现细节。

---

## 3.1 为什么需要 Tokenizer

神经网络吃的是数字，不是字符。Tokenizer 承担三个职责：

1. **编码（encode）**：文本 → token id 序列（喂给模型）；
2. **解码（decode）**：token id 序列 → 文本（模型输出还原）；
3. **词表管理**：决定模型输出层的类别数（$V$）。

可选方案对比：

| 方案 | 原理 | 问题 |
|------|------|------|
| 字符级 | 每个字符一个 token | 序列太长，训练效率低；无法捕获词根等语言规律 |
| 词级 | 每个词一个 token | 词表爆炸（英文词汇百万级），OOV（未登录词）问题 |
| **子词级（BPE 等）** | 常用词/词根整块，生僻部分拆开 | 平衡序列长度与词表大小，**业界标准** |

MiniMind 的词表只有 **6400**，而 GPT-3 是 50257、Qwen2 是 151643。为什么敢这么小？因为词表大小直接决定 embedding 层和输出层的参数量（第 4 章会算：6400 × 768 × 2 ≈ 9.8M 参数，占 64M 模型的 15%）。对小模型来说，**词表精简是合理的取舍**——代价是编解码效率略低（中文约 1.5~1.7 字符/token，英文 4~5 字符/token），但对模型能力影响不大。

## 3.2 BPE 算法：从原理到实现

**Byte Pair Encoding（BPE）**，字节对编码，1994 年由 Gage 提出用于数据压缩，后被 GPT-2 引入 NLP。

### 3.2.1 训练流程（迭代合并）

BPE 训练是一个**自底向上的合并过程**：

1. 把语料按字节/字符切分（MiniMind 用 ByteLevel，即把 Unicode 文本按字节切，天然支持任何语言）；
2. 统计所有相邻字节对（byte pair）的出现频率；
3. 选择**频率最高**的一对合并成一个新 token；
4. 重复步骤 2~3，直到词表达到目标大小。

```text
语料: ["低", "低", "低级", "低级词"]（示意）

初始: 低 / 低 / 低 级 / 低 级 词
统计: ("低","级") 出现 2 次 → 合并
结果: 低 / 低 / 低级 / 低级 词
统计: ("低级","词") 出现 1 次 → 合并
结果: 低 / 低 / 低级 / 低级词
```

直觉：**高频共现的子序列会被优先吸收成独立 token**，所以常用词（"的"、"了"、"人工智能"）会成为整体 token，生僻词则退化为字节级拼接（OOV 问题被彻底消灭——任何文本都能编码，代价是多几个 token）。

### 3.2.2 MiniMind 的 tokenizer 训练脚本

`trainer/train_tokenizer.py` 完整展示了用 HuggingFace `tokenizers` 库训练 BPE 的流程（仅 168 行）：

```python
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
```

- `models.BPE()`：BPE 模型，词表文件见 `model/tokenizer.json`；
- `pre_tokenizers.ByteLevel(add_prefix_space=False)`：**预分词器**。ByteLevel 把所有文本先编码为 UTF-8 字节再按字节切分。`add_prefix_space=False` 表示英文单词前不强制加空格（对中文友好）。

**特殊 token 体系**（第 28~42 行）——这是 MiniMind 功能设计的核心：

```python
special_tokens_list = [      # 21 个"真"特殊 token（模型视其为单一 token，不可拆）
    "<|endoftext|>", "<|im_start|>", "<|im_end|>",
    "<|object_ref_start|>", ..., "<tts_text_bos_single>"
]
additional_tokens_list = [   # 6 个"功能" token（普通 token，可参与 BPE 合并）
    "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>",
    "<think>", "</think>"
]
num_buffer = special_tokens_num - len(special_tokens_list + additional_tokens_list)  # 36-27=9
buffer_tokens = [f"<|buffer{i}|>" for i in range(1, num_buffer + 1)]  # <|buffer1|>~<|buffer9|>
```

值得讲解的设计点：

1. **总预留 36 个 token 位**：21 特殊 + 6 功能 + 9 buffer。buffer token 是**预留的空位**，未来要加新能力（比如新的模态标记）时不用重新训练 tokenizer——这正是 tokenizer "不可轻动"特性的工程化应对。
2. `<|im_start|>` / `<|im_end|>` 是 **ChatML 格式**的角色边界标记（`im` = instruction message），类似 OpenAI 的 `<|im_start|>system`。MiniMind 的 bos=eos=分别是它们（token id 1、2），`<|endoftext|>` 兼任 pad 与 unk（id 0）。
3. `<think>` / `</think>` 是**思考链标记**（第 11 章详述）；`<tool_call>` 等是工具调用标记（第 11 章）。它们被设为 `special: false`（普通 token），意味着它们可以参与 BPE 合并、允许被拆解——代码里第 60~64 行专门做了这个后处理：

```python
for token_info in tokenizer_data.get('added_tokens', []):
    if token_info['content'] not in special_tokens_list:
        token_info['special'] = False
```

4. 训练数据只取 `sft_t2t_mini.jsonl` 的前 10000 行（`get_texts` 中 `i >= 10000: break`）——**用对话数据训 tokenizer**，保证词表偏向对话场景的用词分布。

> ⚠️ 文件头部的注释很重要：**不建议重新训练 tokenizer**。词表和切分规则一变，模型权重、数据格式、推理接口、第三方生态全部要跟着变。MiniMind 自带 tokenizer 就是官方主线，个人学习时直接使用即可。

### 3.2.3 一个需要理解的行为差异

同一文本在不同 tokenizer 下的 token 数不同。MiniMind README 给过示例：

```text
中文："白日依山尽"（5 字符）→ ["白日","依","山","尽"]（4 tokens）
英文："The sun sets in the west"（24 字符）→ ["The ","sun ","sets ","in ","the","west"]（6 tokens）
```

这解释了为什么 `max_seq_len` 的语义是 **tokens 而非字符数**，以及为什么 `pretrain_t2t_mini.jsonl` 推荐 `max_seq_len≈768`——这些数字背后是数据分布的平均长度统计。

## 3.3 chat_template：文本 → 对话模板

预训练模型只会"接龙"，对话能力来自**把对话历史渲染成一段固定格式的文本**。`model/tokenizer_config.json` 中的 `chat_template`（Jinja2 模板）就是这套渲染规则。

### 3.3.1 基础渲染

对消息列表：

```json
[
  {"role": "user", "content": "你好"},
  {"role": "assistant", "content": "你好！"}
]
```

渲染结果（`apply_chat_template(..., add_generation_prompt=True)`）：

```text
<|im_start|>user
你好<|im_end|>
<|im_start|>assistant
<think>

</think>

你好！<|im_end|>
```

模板逻辑拆解（对照 tokenizer_config.json 中的 Jinja2）：

- 每个角色消息包在 `<|im_start|>角色\n...<|im_end|>\n` 中；
- **assistant 消息前固定注入 `<think>\n\n</think>\n\n` 结构**——这是"自适应思考"（adaptive thinking）的模板层实现：训练时让模型见过"空思考 + 显式思考"两种形态，推理时通过 `open_thinking` 开关动态决定注入 `<think>\n`（开）还是空 `<think>\n\n</think>\n\n`（关）。第 11 章详细展开；
- `add_generation_prompt=True` 时末尾加上 `<|im_start|>assistant\n`，让模型从"assistant 应该说话了"的位置开始续写。

### 3.3.2 工具调用渲染（第 11 章的伏笔）

当消息带 `tools` 时，模板会渲染出一段工具说明（`# Tools ...` + `<tools>...</tools>`），assistant 的工具调用渲染为：

```text
<tool_call>
{"name": "get_current_time", "arguments": {"timezone": "Asia/Shanghai"}}
</tool_call>
```

工具结果则渲染为 `<tool_response>...</tool_response>`。这套 XML 式标记让模型"学会"结构化输出，第 11 章会看到它的完整机制。

### 3.3.3 模板中的小技巧

- `message.reasoning_content` 字段：显式思考内容；若消息只有 content 且含 `<think>...`，模板会自动拆分（`content.split('</think>')`）；
- tool 消息的边界处理：连续多条 tool 消息合并进同一个 `<|im_start|>user` 块（`loop.first or messages[loop.index0-1].role != "tool"` 条件控制）；
- `multi_step_tool` 逻辑：从后往前找"真正的最后一个用户问题"，用于多轮工具调用场景的定位（Agentic RL 依赖它）。

## 3.4 数据格式与数据集类精读

### 3.4.1 数据文件格式

**预训练** `pretrain_t2t_mini.jsonl`——纯文本，每行一条：

```jsonl
{"text": "如何才能摆脱拖延症？治愈拖延症并不容易，但以下建议可能有所帮助。"}
{"text": "清晨的阳光透过窗帘洒进房间，桌上的书页被风轻轻翻动。"}
```

**SFT** `sft_t2t_mini.jsonl`——多轮对话，每行一个完整对话：

```jsonl
{"conversations": [
  {"role": "user", "content": "你好"},
  {"role": "assistant", "content": "你好！"}
]}
```

（tool call 样本在此基础上加 `system.tools`、`assistant.tool_calls`、`tool` 角色，见第 11 章。）

> 全部数据在 ModelScope/HuggingFace 的 `gongjy/minimind_dataset` 下载，放入 `./dataset/` 目录。快速复现只需 `pretrain_t2t_mini.jsonl`（1.2GB）+ `sft_t2t_mini.jsonl`（1.6GB）两个文件。

### 3.4.2 PretrainDataset：最简单的监督信号

`dataset/lm_dataset.py` 第 37~55 行，仅 19 行，但包含预训练数据的全部要点：

```python
class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        ...
        self.samples = load_dataset('json', data_files=data_path, split='train')

    def __getitem__(self, index):
        sample = self.samples[index]
        tokens = self.tokenizer(str(sample['text']), add_special_tokens=False,
                                max_length=self.max_length - 2, truncation=True).input_ids
        tokens = [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]
        input_ids = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        labels = input_ids.clone()
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        return input_ids, labels
```

逐行讲解：

1. `add_special_tokens=False`：文本本身不加特殊 token（手动加，避免重复）；
2. `max_length - 2`：留 2 个位置给 bos/eos；
3. `[bos] + tokens + [eos]`：完整文本包在 `1 ... 2` 之间——**为什么预训练也加 bos？** 让模型学习"一句话从哪里开始"；
4. 右侧 padding 到 `max_length`：一个 batch 内所有样本等长才能张量化（**padding 在右侧**，不破坏因果顺序）；
5. `labels = input_ids.clone()`：预训练是**全位置监督**（每个 token 都要预测下一个）；
6. **padding 位置 label 置 -100**：`ignore_index`，PyTorch 交叉熵自动跳过，不产生梯度。这是全篇教程反复出现的核心套路。

注意：`max_length` 默认 512，但 `train_pretrain.py` 传入的是 `--max_seq_len`（默认 340，mini 数据推荐 768）。**序列越长，单样本显存越大、训练越慢**，这就是"截断"的工程代价。

### 3.4.3 SFTDataset：只学"assistant 的话"

`dataset/lm_dataset.py` 第 58~119 行。核心差异在于 **label 掩码策略**：SFT 只让模型学习 assistant 回复，不学 user 提问（提问是条件，无需预测）。否则模型会退化成一个"复读机"。

实现分三步：

**① 模板化**（第 71~86 行）：

```python
def create_chat_prompt(self, conversations):
    messages = []
    tools = None
    for message in conversations:
        message = dict(message)
        if message.get("role") == "system" and message.get("tools"):
            tools = json.loads(message["tools"]) ...
        if message.get("tool_calls") and isinstance(message["tool_calls"], str):
            message["tool_calls"] = json.loads(message["tool_calls"]) ...
        messages.append(message)
    return self.tokenizer.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=False, tools=tools)
```

- 把 `tools`、`tool_calls` 的 JSON 字符串解析成对象，供模板使用；
- `apply_chat_template(tokenize=False)`：先渲染成纯文本，**不直接 tokenize**——为什么？因为 label 掩码需要按文本结构（assistant 片段）定位，先有文本再统一 tokenize 更可控。

**② 概率性预处理**（第 9~35 行）：

```python
def pre_processing_chat(conversations, add_system_ratio=0.2):
    if any(conv.get('tools') for conv in conversations): return conversations  # tool 数据完整保留
    SYSTEM_PROMPTS = ["你是一个知识丰富的AI...", ..., "You are minimind, a small but useful language model."]
    if conversations[0].get('role') != 'system':
        if random.random() < add_system_ratio:
            return [{'role': 'system', 'content': random.choice(SYSTEM_PROMPTS)}] + conversations
    return conversations
```

- 20% 概率给对话加一条 system 消息——**数据增强**：让模型见过"带/不带 system"两种输入形态，推理时才能对 system 提示词鲁棒；
- 中英双语 system prompt 随机选取；
- tool call 数据**跳过该处理**（结构敏感，不能动）。

**③ label 掩码**（第 88~104 行）——本文件最精巧的部分：

```python
def generate_labels(self, input_ids):
    labels = [-100] * len(input_ids)
    i = 0
    while i < len(input_ids):
        if input_ids[i:i + len(self.bos_id)] == self.bos_id:   # 命中"assistant 开始"标记
            start = i + len(self.bos_id)
            end = start
            while end < len(input_ids):
                if input_ids[end:end + len(self.eos_id)] == self.eos_id:  # 找到"assistant 结束"
                    break
                end += 1
            for j in range(start, min(end + len(self.eos_id), self.max_length)):
                labels[j] = input_ids[j]
            i = end + len(self.eos_id) if end < len(input_ids) else len(input_ids)
        else:
            i += 1
    return labels
```

关键思想：`self.bos_id = tokenizer(f'{tokenizer.bos_token}assistant\n', ...)`——**"assistant 回复开始"的 token 序列**（`<|im_start|>assistant\n<think>\n` 的起始片段）。算法从左到右扫描，每当命中这个片段，就把**其后直到 `</think>` 之后、`<|im_end|>` 为止**的 token 标为学习对象，其余全部 `-100`。

> 注意细节：`bos_id` 是 `<|im_start|>assistant\n` 的子串而非完整 `<think>` 前缀——这样能同时命中"空思考"和"带思考"两种模板形态（二者在 `<|im_start|>assistant\n` 之后才分叉）。

**`post_processing_chat`**（第 31~35 行）还有一处数据增强：80% 概率移除空的 `<think>\n\n</think>\n\n` 片段——与模板渲染的注入逻辑形成对称，让模型既见过空思考也见过无思考标记的样本。

### 3.4.4 其余数据集类（第 11~12 章用到，先建立印象）

| 类 | 用途 | 特点 |
|----|------|------|
| `DPODataset` | DPO 偏好学习 | 同时返回 chosen/rejected 两套 `(x, y, mask)`，mask 只标 assistant 段 |
| `RLAIFDataset` | PPO/GRPO 采样 | 返回 `prompt`（含 `open_thinking` 概率注入），`answer` 留空待 rollout |
| `AgentRLDataset` | Agentic RL | 解析 tools、返回 messages + 标准答案 `gt` |

## 3.5 动手实验

### 实验 3.1：观察 tokenizer 行为

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("./model")

# 1) 基础编解码
text = "人工智能是计算机科学的一个分支"
ids = tok.encode(text)
print("ids:", ids)
print("decode:", tok.decode(ids))

# 2) 逐 token 查看
for i in ids:
    print(f"{i:5d} -> {tok.convert_ids_to_tokens(i)!r}")

# 3) 特殊 token 映射
print("bos:", tok.bos_token_id, "| eos:", tok.eos_token_id, "| pad:", tok.pad_token_id)
```

### 实验 3.2：观察 chat_template 渲染

```python
messages = [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！"},
    {"role": "user", "content": "你叫什么"},
]
# 不带生成提示
print(tok.apply_chat_template(messages, tokenize=False))
print("-" * 60)
# 带生成提示 + 开思考
print(tok.apply_chat_template(messages, tokenize=False,
                              add_generation_prompt=True, open_thinking=True))
```

观察输出中 `<|im_start|>user\n...`、`<think>\n\n</think>` 的结构，对照 3.3 节。

### 实验 3.3：观察 SFT 掩码效果

```python
import torch
from dataset.lm_dataset import SFTDataset

ds = SFTDataset("./dataset/sft_t2t_mini.jsonl", tok, max_length=256)  # 需先下载数据
ids, labels = ds[0]
# 统计：有多少位置被 mask 掉
masked = (labels == -100).sum().item()
print(f"总长度 {len(ids)}，mask 掉 {masked} 个位置（{masked/len(ids):.1%}）")
# 解码看 assistant 片段是否保留
for j in range(len(ids)):
    if labels[j] != -100:
        print(tok.decode([ids[j]]), end="")
```

预期：被 mask 的是 user 提问与 padding，保留的是 assistant 回复——这正是模型唯一要学习的内容。

## 3.6 本章小结

- Tokenizer 是文本与数字的桥梁；BPE 通过高频合并构建子词词表，天然解决 OOV；
- MiniMind 词表 6400：小词表压缩 embedding/输出层参数，是 64M 模型的合理取舍；
- 特殊 token 体系（36 个预留位）承载了对话、思考、工具、多模态扩展等能力；
- chat_template 是"对话 → 训练文本"的渲染规则，`<think>` 结构与 `open_thinking` 开关都在模板层实现；
- 预训练全位置监督；SFT 只监督 assistant 片段（`-100` 掩码）；padding 一律掩码；
- 数据增强（随机 system、随机移除空 think）提高模型对输入形态的鲁棒性。

## 3.7 思考题

1. 为什么 BPE 选择"频率最高"的相邻对合并，而不是随机合并？试想合并顺序会影响什么。
2. `PretrainDataset` 中如果 `text` 比 `max_length` 还长，会发生什么？`truncation=True` 截断在哪一侧？
3. SFT 的 label 掩码里，为什么 `bos_id` 用 `'<|im_start|>assistant\n'` 而不是完整的 `<think>` 前缀？
4. 如果让模型也学习 user 提问（不掩码），会发生什么？请从"复读机现象"角度推理。
5. （动手）打印一个 SFT 样本的完整输入文本与 label 掩码，画出"user 段 vs assistant 段"的分界。
6. 为什么 MiniMind 用 6400 词表而 Qwen2 用 151643？分别从参数量、编解码效率、模型能力三个角度分析。

## 3.8 拓展阅读

- 《Neural Machine Translation of Rare Words with Subword Units》（Sennrich et al., 2016）：BPE 进入 NLP 的开山之作
- HuggingFace 官方文档：`tokenizers` 库与 `apply_chat_template` 教程（chat_template 是当前各模型"提示词格式"的标准实现方式）
- MiniMind README "Tokenizer" 一节（词表对比表与 BPB 指标的讨论）
