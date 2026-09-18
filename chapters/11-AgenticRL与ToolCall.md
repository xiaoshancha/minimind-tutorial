# 第 11 章 Agentic RL 与 Tool Call：让模型"使用工具"

> 对应代码：`trainer/train_agent.py`（492 行）、`dataset/lm_dataset.py` 的 `AgentRLDataset`、`scripts/eval_toolcall.py`、`model/tokenizer_config.json` 的 chat_template
>
> 本章目标：理解工具调用（Tool Calling）的数据格式与模板机制，掌握 Agentic RL 的多轮交互循环（rollout → 执行工具 → 观察 → 继续生成），读懂基于工具执行结果的奖励设计（可验证奖励 RLVR），并理解"自适应思考"（open_thinking）的模板层实现。

---

## 11.1 从"纯文本模型"到"会用工具的模型"

对话模型只会"说话"；工具调用模型能**输出结构化的函数调用**，由外部系统执行后把结果喂回模型，模型再基于结果继续回答：

```text
用户: 帮我算一下 256 × 37
模型: <tool_call>{"name": "calculate_math", "arguments": {"expression": "256 * 37"}}</tool_call>
系统: 执行工具 → {"result": "9472"}
模型: 256 乘以 37 等于 9472。
```

这个能力让 LLM 从"知识回忆器"升级为"能行动的执行者"（Agent）。MiniMind 的处理思路是**把工具调用能力并入主线 SFT 数据**（`sft_t2t_mini.jsonl` 已混入约 10 万条 qwen3-4b 合成的 tool call 样本），再用 Agentic RL 强化多轮工具使用策略。

## 11.2 Tool Call 的数据格式与模板机制

### 11.2.1 OpenAI 风格数据（SFT 中的形态）

```jsonl
{
  "conversations": [
    {"role": "system", "content": "# Tools ...", "tools": "[...]"},              // 工具定义挂在 system 上
    {"role": "user", "content": "帮我算一下 256 乘以 37 等于多少"},
    {"role": "assistant", "content": "", "tool_calls": "[{\"name\":\"calculate_math\",\"arguments\":{\"expression\":\"256 * 37\"}}]"},
    {"role": "tool", "content": "{\"result\":\"9472\"}"},                         // 工具结果
    {"role": "assistant", "content": "256 乘以 37 等于 9472。"}
  ]
}
```

### 11.2.2 chat_template 的渲染规则（tokenizer_config.json）

**工具定义**（`tools` 参数传入时渲染成 XML 说明，第 3 章 3.3.2）：

```text
<|im_start|>system
你是一个知识丰富的AI...
# Tools

You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "calculate_math", "description": "计算数学表达式", ...}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call><|im_end|>
```

**模型输出**（`assistant.tool_calls` 渲染）：

```text
<|im_start|>assistant
<think>

</think>

<tool_call>
{"name": "calculate_math", "arguments": {"expression": "256 * 37"}}
</tool_call><|im_end|>
```

**工具结果**（`role: tool` 渲染，连续多条合并进一个 user 块）：

```text
<|im_start|>user
<tool_response>
{"result": "9472"}
</tool_response><|im_end|>
```

为什么用 XML 标记而不是纯 JSON？**结构化 + 可解析 + 可流式**：模型只需学习"输出 `<tool_call>` 包裹的 JSON"，解析用正则即可完成（`parse_tool_calls`）。训练时这些标记是普通 token（第 3 章：`special: false`），模型自然学会生成。

## 11.3 Agentic RL：train_agent.py 精读

与第 10 章的 GRPO 相比，Agentic RL 的核心区别：**多轮交互**——模型输出工具调用后不立即结束，而是"执行工具 → 观察结果 → 继续生成"，直到不再调用工具或达到轮数上限。

### 11.3.1 模拟工具环境（第 40~95 行）

MiniMind 定义了 6 个模拟工具（`calculate_math`、`unit_converter`、`get_current_weather`、`get_current_time`、`get_exchange_rate`、`translate_text`），配好静态数据（天气、时间、汇率、翻译表）与执行函数：

```python
MOCK_RESULTS = {
    "calculate_math": lambda args: {"result": str(eval(str(args.get("expression", "0")).replace("^", "**")...))},
    "get_current_weather": lambda args: (lambda w: {"city": args.get("location"), "temperature": w[0], ...})(WEATHER_DATA.get(args.get("location"), ("22°C", "晴"))),
    ...
}
```

```python
def execute_tool(name, args):
    fn = MOCK_RESULTS.get(name)
    if not fn: return None
    try:
        signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError()))  # 1 秒超时保护
        signal.alarm(1)
        return fn(args)
    except:
        return None
```

工程细节：`SIGALRM` 给工具执行加 **1 秒超时**（防止 eval 恶意/死循环表达式），`CHECK_ARGS` 做参数校验。真实系统中这里换成真正的 API 调用即可——**模拟环境的意义是让训练可复现、成本为零**。

### 11.3.2 多轮 Rollout（第 98~157 行）

```python
def rollout_single(rollout_engine, tokenizer, messages, tools, max_turns=3, max_new_tokens=256, ...):
    for turn in range(max_turns):
        context = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=tools, open_thinking=open_thinking)
        rollout_result = rollout_engine.rollout(...)
        new_text = rollout_result.completions[0]
        response_ids.extend(new_ids); response_old_logps.extend(new_logps)
        calls = parse_tool_calls(new_text)          # 解析 <tool_call> 标记
        if not calls: break                          # 没有工具调用 → 回答完成，结束
        messages.append({"role": "assistant", "content": new_text})
        for call in calls:
            result = execute_tool(name, raw)         # 执行工具
            messages.append({"role": "tool", "content": result_str})
        # 关键：把"工具结果"模板也纳入训练序列（掩码为 0）
        observe_context = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=not unfinished, tools=tools, open_thinking=open_thinking)
        obs_delta = observe_ids[current_len:]
        response_ids.extend(obs_delta)               # 观察段计入序列
        response_mask.extend([0] * len(obs_delta))   # 但不算模型输出（掩码 0）
        response_old_logps.extend([0.0] * len(obs_delta))
    return final_output, final_context, prompt_ids, response_ids, response_mask, response_old_logps, ...
```

多轮 Rollout 的数据流：

```text
turn 1: [prompt] → 生成 → <tool_call>...  → 解析 → 执行工具
             ↓（append assistant 消息 + tool 消息）
turn 2: [prompt + tool_call + tool_response] → 生成 → 最终回答 → 无 tool_call → 结束
```

**掩码设计**：模型输出的 token（每轮的生成部分）掩码为 1（参与训练）；工具执行结果（`<tool_response>` 内容）掩码为 0——**模型只学"如何调用工具"，不学"工具返回什么"**（那是环境的事）。这与 SFT 中"只学 assistant 段"是同一哲学。

### 11.3.3 可验证奖励（RLVR）：第 182~239 行

Agentic RL 的奖励与第 10 章的通用奖励相比，增加了**任务真值验证**：

```python
def validate_gt_in_text(text, gt_list):
    # 检查最终回答中是否包含标准答案（支持字符串包含与数值容差匹配）
    return {g for g in gt_list if (str(g) in text) or (数字匹配 < 1e-6 容差)}

def calculate_rewards(prompts, completions, gt_batch, tools_batch, num_gen, ...):
    ...
    # -------- 有工具调用分支：执行结果奖励 --------
    tool_gap = abs(valid_call_count - len(gt)) + max(0, len(tool_calls) - valid_call_count)
    reward += 0.5 if tool_gap == 0 else -0.5 * tool_gap      # 工具调用次数对齐
    final_text = ... (去掉 think/tool_call 的最终回答)
    verified = validate_gt_in_text(final_text, gt)           # 标准答案验证
    if gt: reward += 2.5 * len(verified) / len(gt)           # ★ GT 命中分（大头）
    if unfinished: reward -= 0.5                             # 未完成惩罚
    reward -= rep_penalty(final_text)
    rewards[idx] = max(min(reward, 3.0), -3.0)               # 总分 clip
```

这是 **RLVR（Reinforcement Learning with Verifiable Rewards）** 的典型设计——奖励来自"程序化验证"而非模型打分：

| 奖励项 | 分数 | 验证什么 |
|--------|------|----------|
| 工具调用对齐 | ±0.5/次 | 调用的工具是否合法（在工具列表、参数完整）、次数是否与标准一致 |
| **GT 命中** | 2.5×命中率 | **最终回答是否包含标准答案**（数字容差/字符串包含） |
| 思考格式 | ±0.5~0.25 | think 长度与闭合 |
| 未完成 | -0.5 | 达到轮数上限仍未回答 |
| 重复惩罚 | -0.5 上限 | 3-gram 重复 |

**为什么 GT 分是大头（2.5）**？它直接把训练目标锚定到"任务是否做对"，而不是"话是否说得好"。这让模型学会：该调用工具就调用、调用后把结果组织成最终答案。`agent_rl_math.jsonl` 就是专门为 RLVR 准备的数学数据（带最终数值答案）。

### 11.3.4 训练循环与损失（第 242~332 行）

与 GRPO 完全同构：`rollout_batch` 批量采样（每 prompt 生成 num_gen 条多轮轨迹）→ `calculate_rewards` → 组内优势 → 裁剪/无偏 KL 损失。差异仅在数据组装：

```python
packed_samples = []
for p, r, m, old_lp in zip(prompt_ids_batch, response_ids_batch, response_masks_batch, response_old_logps_batch):
    ids = p + r                                  # prompt + 多轮生成拼接
    mask = [0] * len(p) + m                      # prompt 段掩码 0
    ...
    if len(ids) > args.max_total_len:            # 截断保护
        ids = ids[-args.max_total_len:]
```

**拼接（packing）**：把多轮轨迹拼成一条长序列训练（类似预训练的序列拼接思想），避免多轮分段 padding 浪费。

## 11.4 自适应思考：open_thinking 机制

第 3 章埋的线在这里收束。MiniMind 不单独训练"思考模型"，而是**模板层动态开关**：

| 开关 | 模板注入 | 模型行为 |
|------|----------|----------|
| `open_thinking=0` | `<think>\n\n</think>\n\n`（空） | 直接回答 |
| `open_thinking=1` | `<think>\n`（起始标签） | 先输出显式思考再回答 |

训练时（`thinking_ratio=0.9`，RLAIFDataset）：90% 概率注入真实 `<think>` 引导模型产出思考链，10% 注入空 think——**让模型见过"该想时想、该直答时直答"的混合分布**。推理时同一个权重，切换开关即可换行为（第 7 章实验 7.x 已体验）。

`<think>` 标记本身是普通 token，所以思考链的生成完全由数据驱动，不需要特殊代码路径。`eval_llm.py`/`web_demo.py`/`serve_openai_api.py` 三端都支持这个开关（`chat_template_kwargs: {"open_thinking": True}`）。

> 已知局限（README 明确说明）：目前数据里"reasoning 与 tool call 同时出现"的联合蒸馏样本不足，所以同时开启 Tool Call 与显式思考时，模型不太能稳定地两全。

## 11.5 动手实验

### 实验 11.1：测试 SFT 模型的 Tool Call 能力

```bash
# full_sft 权重已混入 tool call 数据，直接用 eval_toolcall.py 测试
cd scripts
python eval_toolcall.py --weight full_sft
```

预期交互（工具选择 + 执行 + 回答）：

```text
💬: 现在几点了？
🧠: <tool_call>{"name": "get_current_time", "arguments": {"timezone": "Asia/Shanghai"}}</tool_call>
📞 [Tool Calling]: get_current_time
✅ [Tool Called]: {"datetime": "2026-03-15 17:18:22", "timezone": "Asia/Shanghai"}
🧠: 现在是2026年3月15日17时18分22秒。
```

### 实验 11.2：open_thinking 对比

```bash
cd ..
python eval_llm.py --weight full_sft --open_thinking 0    # 直答
python eval_llm.py --weight full_sft --open_thinking 1    # 思考
```

对比同一问题的回答结构（如"9.11 和 9.9 哪个大"这类需要推理的问题）。

### 实验 11.3：Agentic RL 训练（可选，需要 agent_rl.jsonl）

```bash
cd trainer
python train_agent.py --debug_mode --log_interval 5 --save_interval 10
```

用 `--debug_mode` 观察多轮轨迹与奖励明细，重点看：模型是否学会"先调用工具、再根据结果回答"的完整模式（对比实验 11.1 的 SFT 直接能力）。

## 11.6 本章小结

- Tool Call 能力 = 数据格式（OpenAI 风格）+ 模板渲染（XML 标记）+ 解析执行（正则 + 模拟环境）；
- Agentic RL = GRPO 框架 + 多轮 rollout（工具执行结果回填上下文）+ RLVR 奖励（工具对齐 + GT 验证）；
- 掩码哲学延续：模型只学"自己该输出的"，工具结果和 prompt 都不参与监督；
- 模拟工具环境（SIGALRM 超时、参数校验、静态数据）让训练零成本可复现；
- 自适应思考是模板层开关，`<think>` 是普通 token，训练混合分布、推理动态切换；
- Agentic RL 之后模型"会用工具且用得好"——这是通往通用 Agent 的工业界主流路径。

## 11.7 思考题

1. 为什么工具定义渲染成 XML 说明而不是直接给 JSON schema？两种方式对模型学习有什么不同影响？
2. `response_mask` 中工具结果段掩码为 0，如果改为 1（让模型学习工具结果格式）会发生什么？
3. RLVR 的 GT 验证为什么用"数字容差 1e-6 + 字符串包含"双模式？直接字符串匹配有什么缺陷？
4. `signal.alarm(1)` 超时保护解决了什么问题？在真实工具调用系统中对应的工程实践是什么？
5. 如果工具环境与训练时不一致（如天气数据更新了城市列表），模型表现会如何？这暴露了模拟环境的什么局限？
6. （动手）在 `eval_toolcall.py` 里给模型一个不在 TOOLS 列表中的工具名，观察模型的失败模式。
7. open_thinking 为什么用"混合训练"而不是"两个模型各训一个"？数据量视角怎么分析？

## 11.8 拓展阅读

- 《Toolformer: Language Models Can Teach Themselves to Use Tools》（Schramowski et al., 2023）——工具调用训练的开创工作
- 《Training Verifiers to Solve Math Word Problems》（Cobbe et al., 2021）——GSM8K 与验证器（RLVR 前身）
- DeepSeek-R1 技术报告——RLVR + GRPO 在大模型上的完整实践（MiniMind 的 Agentic RL 是其缩小版）
- MiniMind README"工具调用 & 自适应思考"一节
