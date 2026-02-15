# RLVR for Agents — Foolproof Training Plan

## What You're Building

A pipeline that takes an SFT agent (Qwen 3B fine-tuned for function calling) and makes it significantly more reliable at completing multi-step tasks through GRPO reinforcement learning. You already have every piece — this plan connects them.

## What You Already Have

- [x] Agent framework with agentic loop, tool execution, memory
- [x] Qwen 3B fine-tuned for chat + function calling (SFT baseline)
- [x] GRPO implementation from raw PyTorch
- [x] GAIA benchmark evaluation
- [x] Tools: web search, code execution, file handling

## The 8 Sub-Components You'll Master

Each one is independently valuable and hireable. Together they make you one of very few people who've touched the full RLVR stack.

### 1. Verifier Design
How do you programmatically check if an agent's output is correct? Exact match, numeric tolerance, regex extraction, code execution sandboxes, symbolic equivalence (SymPy), LLM-as-a-Judge for soft verification. The verifier IS the reward signal. Bad verifier = bad training.

### 2. Environment Design
The agent's simulated world. Tool execution sandboxes, state management across steps, observation formatting (how tool results feed back to the model), episode termination logic, determinism vs stochasticity in tool results during training.

### 3. Trajectory Tokenization & Masking
The most underrated problem. Representing multi-turn conversations as token sequences, deciding which tokens get gradients (model outputs) vs which are environment (tool results, system prompts), handling variable-length trajectories in a batch, padding, truncation, attention masks, chat template formatting. Get this wrong and you're training on garbage.

### 4. Rollout Generation (Inference at Scale)
The pipeline bottleneck. Batched inference across GPUs, KV-cache management across turns, temperature/sampling for diverse rollouts, tool execution parallelism, result caching (8 rollouts searching "GDP of Japan" should cache), async orchestration. This is I/O bound, not compute bound.

### 5. Advantage Estimation
Turning raw rewards into useful training signal. Group-relative normalization (GRPO's core), handling degenerate groups (all succeed or all fail = zero advantage = no learning), reward shaping without one component dominating, reward normalization across task difficulties, variance reduction.

### 6. Policy Optimization
The RL update. Clipped surrogate objective, KL penalty against reference model (SFT baseline), learning rate scheduling for RL (very different from SFT), gradient accumulation across variable-length trajectories, mini-batch updates within a single group.

### 7. Training Stability & Diagnostics
What to look at when things go wrong. Reward curves, KL divergence tracking, policy entropy, gradient norms, reward hacking detection, mode collapse detection, validation-based early stopping. The difference between "can run GRPO" and "can make GRPO work reliably."

### 8. Evaluation & Analysis
Proving it works and understanding why. Held-out test sets, per-category breakdowns, trajectory comparison (SFT vs RLVR side by side), ablation design, generalization testing (GAIA), statistical significance.

### How They Map to Lab Roles

| Sub-component | Who hires for this |
|---|---|
| Verifier design | Research Scientists (reward design is active research) |
| Environment design | Agent Infrastructure teams |
| Trajectory tokenization | Post-training teams (daily work) |
| Rollout generation | Inference/Systems teams |
| Advantage estimation | RL Research teams |
| Policy optimization | Post-training teams |
| Training diagnostics | Everyone who ships models |
| Evaluation & analysis | Research Scientists, Alignment teams |

---

## Distributed Training Strategy

### Why you need it

The model (3B) fits on a single GPU. The bottleneck is rollout generation: 500 tasks x 8 rollouts x ~5 steps x ~1.5s per step = ~30,000 seconds (~8 hours) per epoch on 1 GPU. With 5 epochs, that's 40+ hours just for rollouts in the main training run.

### Architecture: Distributed Rollouts, Centralized Training

```
GPU 0: rollouts for tasks 0-124    ─┐
GPU 1: rollouts for tasks 125-249  ─┤── gather scored trajectories ──→ GPU 0: GRPO update
GPU 2: rollouts for tasks 250-374  ─┤                                      │
GPU 3: rollouts for tasks 375-499  ─┘                    broadcast weights ←┘
                                                                │
                                          all GPUs: next epoch ←┘
```

- **Rollout phase:** Data-parallel inference. Each GPU runs a copy of the model, generates rollouts for its shard of tasks independently. No communication needed during rollout generation.
- **Training phase:** Single GPU computes GRPO update on all gathered trajectories. 3B model + optimizer fits easily on one A100.
- **Sync:** `torch.distributed.broadcast` updated weights to all rollout workers.

### What distributed concepts you'll use

| Concept | Where it shows up |
|---|---|
| Data parallelism (DDP) | Rollout generation across GPUs |
| Parameter broadcasting | Sync updated weights to rollout workers |
| Async worker orchestration | Rollout workers run independently |
| Mixed inference + training | GPUs switch between generate and train mode |
| `all_gather` | Collecting trajectories from all workers |
| FSDP | Only if you scale to 7B+ (your existing experience applies) |

### Scaling path

- **3B (start here):** 4x A100 node. DDP rollouts, single-GPU training. No FSDP needed.
- **7B (if 3B works):** 4-8x A100. FSDP for training, DDP for rollouts. Your existing FSDP code from the reasoning LLM project transfers directly.

---

## Cost Estimate

### Compute breakdown (main training run at 3B, 4x A100)

| Phase | GPU-hours | Wall clock (4 GPUs) |
|---|---|---|
| SFT baseline measurement | 5 | 1.5 hrs |
| Rollout generation (5 epochs) | 42 | 10.5 hrs |
| GRPO training updates | 3 | 0.75 hrs |
| Validation evaluations | 5 | 1.25 hrs |
| **Main run total** | **~55** | **~14 hrs** |

### Full project compute

| Component | GPU-hours |
|---|---|
| Main training run | 55 |
| Ablations (5 experiments, ~60% each) | 165 |
| Final evaluation + GAIA | 10 |
| Debugging and iteration (always 1-2x planned) | 120 |
| **Total** | **~350** |

### Dollar cost

| Provider | $/hr per A100 | Total cost |
|---|---|---|
| RunPod spot | $0.74 | **~$260** |
| Lambda Labs | $1.10 | **~$385** |
| RunPod on-demand | $1.64 | **~$575** |
| University cluster | free | **$0** |

### Other costs

| Item | Cost |
|---|---|
| Task generation (GPT-4o API) | $20-50 |
| Web search API for tool execution | $20-50 (or use free APIs + caching) |
| Weights & Biases logging | free tier |
| **Total non-compute** | **~$50-100** |

### Realistic total budget: $300-700

Use RunPod spot instances for best value. Cache tool results aggressively to reduce both API costs and rollout time. Start with Category A (math) tasks only to debug the pipeline cheaply before scaling to the full task suite.

---

## The Full Pipeline

```
SFT Agent (baseline) → Build Task Environment → Collect Rollouts → Score with Verifiable Rewards → GRPO Update → Better Agent
                                                      ↑                                                    |
                                                      └────────────────────────────────────────────────────┘
                                                                         repeat
```

---

## Phase 1: The SFT Baseline (Week 1–2)

**Goal:** Have a working agent with measurable performance numbers. This is your "before" photo.

### Step 1: Lock the SFT agent

Your Qwen 3B fine-tuned for function calling is the baseline. Run it through your agent framework on a fixed set of tasks. Don't touch this model again — it's the control.

### Step 2: Build a task suite with verifiable answers

You need 500–1000 tasks where the answer is programmatically checkable. Three categories:

**Category A: Tool-use math (easiest)**
- "What is the population of France divided by the population of Germany? Round to 2 decimal places."
- Requires: search tool (get populations) + calculator tool (divide)
- Verification: exact numerical match
- Generate 200 of these with GPT-4 — vary the countries, operations, and complexity

**Category B: Multi-hop information retrieval (medium)**
- "Who directed the movie that won Best Picture in 2023, and what year was that director born?"
- Requires: search tool (find movie) + search tool (find director) + search tool (find birth year)
- Verification: exact match against known answers
- Use existing datasets: HotpotQA, 2WikiMultiHopQA, or generate with GPT-4 + verify answers

**Category C: Code tasks (hardest)**
- "Write a Python function that returns the nth Fibonacci number. Test it with n=10."
- Requires: code execution tool
- Verification: code runs + output matches expected value
- Use: MBPP (Mostly Basic Python Problems) or HumanEval, simplified

Split: 70% train, 15% validation, 15% test. Never touch test until final evaluation.

### Step 3: Measure the baseline

Run the SFT agent on all tasks. Log everything:
- Task success rate (overall and per category)
- Average number of steps per task
- Tool call failure rate (malformed calls, wrong tool)
- Average tokens consumed per task

**Expected baseline:** Probably 40–60% success rate. This is normal. SFT agents are mediocre. That's the whole point.

### Deliverable: A table showing SFT baseline performance across all categories.

---

## Phase 2: The Rollout Engine (Week 2–3)

**Goal:** A system that runs the agent on a task G times, collects full trajectories, and scores each one.

### What a rollout looks like

```
Task: "What is the GDP of Japan divided by GDP of India?"

Rollout 1 (success):
  Step 1: Agent calls search("GDP of Japan 2024")         → $4.2 trillion
  Step 2: Agent calls search("GDP of India 2024")         → $3.7 trillion
  Step 3: Agent calls calculator("4.2 / 3.7")             → 1.135
  Step 4: Agent returns "1.14"
  Reward: 1.0 ✓ (matches ground truth)

Rollout 2 (failure):
  Step 1: Agent calls search("Japan India GDP comparison") → gets a paragraph
  Step 2: Agent returns "Japan has higher GDP"
  Reward: 0.0 ✗ (not a number, didn't answer the question)

Rollout 3 (failure):
  Step 1: Agent calls calculator("GDP Japan / GDP India")  → error, not numbers
  Step 2: Agent returns "I couldn't calculate this"
  Reward: 0.0 ✗ (gave up)
```

### The rollout collector

For each task in the training set:
1. Run the agent G times (start with G=8, scale to 16 later)
2. Each run uses temperature > 0 (0.7–0.8) so you get diverse trajectories
3. Collect the full trajectory: every model output, every tool call, every tool result
4. Score each trajectory with the verifiable reward

### Reward function design

```python
def compute_reward(trajectory, ground_truth):
    # Core reward: did you get the right answer?
    if agent_answer_matches(trajectory.final_answer, ground_truth):
        correctness = 1.0
    else:
        correctness = 0.0

    # Format reward: were all tool calls valid?
    valid_calls = sum(1 for call in trajectory.tool_calls if call.parsed_successfully)
    format_score = valid_calls / max(len(trajectory.tool_calls), 1)

    # Efficiency bonus: fewer steps = better (only if correct)
    if correctness == 1.0:
        step_penalty = -0.02 * len(trajectory.steps)  # small penalty per step
    else:
        step_penalty = 0.0  # don't reward efficiency on wrong answers

    return correctness + 0.1 * format_score + step_penalty
```

Why this works:
- Correctness dominates (0 or 1). The model learns to get the answer right first.
- Format reward is small (0–0.1). Nudges toward valid tool calls without overwhelming the correctness signal.
- Step penalty is tiny (–0.02 per step). Only kicks in on correct answers — don't reward a fast wrong answer.

### Key implementation detail: tokenizing trajectories

The tricky part is that agent trajectories interleave model tokens and tool results. For GRPO, you only compute the policy gradient on the model's tokens, not the tool outputs (those are environment observations, not model actions).

```
[model tokens: "I'll search for Japan's GDP"]  ← gradient here
[tool result: "Japan GDP is $4.2T"]              ← no gradient (environment)
[model tokens: "Now I'll search India's GDP"]   ← gradient here
[tool result: "India GDP is $3.7T"]              ← no gradient (environment)
[model tokens: "Let me calculate 4.2/3.7"]      ← gradient here
[tool result: "1.135"]                            ← no gradient (environment)
[model tokens: "The answer is 1.14"]             ← gradient here
```

Mask out tool result tokens in the loss computation. Only update on tokens the model actually generated.

### Deliverable: A rollout engine that produces scored trajectories ready for GRPO.

---

## Phase 3: GRPO Training (Week 3–5)

**Goal:** Train the agent with GRPO. Make it better.

### You already have GRPO. Here's what changes for agents:

**Your math GRPO:**
- Input: math problem (short)
- Output: chain-of-thought + answer (single generation)
- Reward: correct/incorrect
- One forward pass per rollout

**Agent GRPO:**
- Input: task description
- Output: multi-turn trajectory (multiple generations interleaved with tool results)
- Reward: task completion + format + efficiency
- Multiple forward passes per rollout (one per agent step)

### The training loop

```python
for epoch in range(num_epochs):
    for task, ground_truth in training_tasks:

        # 1. Collect G rollouts
        trajectories = []
        for g in range(G):
            traj = run_agent(model, task, tools, temperature=0.7)
            traj.reward = compute_reward(traj, ground_truth)
            trajectories.append(traj)

        # 2. Compute group-relative advantages
        rewards = [t.reward for t in trajectories]
        mean_r = mean(rewards)
        std_r = std(rewards) + 1e-8
        advantages = [(r - mean_r) / std_r for r in rewards]

        # 3. For each trajectory, compute policy gradient on model tokens only
        for traj, advantage in zip(trajectories, advantages):
            model_token_ids = extract_model_tokens(traj)  # mask out tool results
            log_probs = compute_log_probs(model, traj)     # only on model tokens

            # Clipped policy gradient (same as your math GRPO)
            ratio = exp(log_probs - old_log_probs)
            clipped = clip(ratio, 1-eps, 1+eps)
            loss = -min(ratio * advantage, clipped * advantage)

            # KL penalty against the SFT baseline
            kl = compute_kl(model, sft_model, model_token_ids)
            total_loss = loss + beta * kl

        # 4. Gradient step
        optimizer.step()

    # 5. Evaluate on validation set after each epoch
    val_success_rate = evaluate(model, val_tasks)
    log(epoch, val_success_rate)
```

### Hyperparameters to start with

| Param | Value | Why |
|-------|-------|-----|
| G (group size) | 8 | Enough diversity, not too expensive |
| Temperature | 0.7 | Diverse rollouts without garbage |
| Learning rate | 1e-6 | Small — RL training is unstable |
| KL beta | 0.05 | Prevent drifting too far from SFT |
| Clip epsilon | 0.2 | Standard PPO/GRPO clip |
| Max steps per rollout | 10 | Cap to prevent infinite loops |
| Epochs | 3–5 | More than this usually overfits |

### What to watch for during training

**Good signs:**
- Validation success rate increasing
- Average steps per successful task decreasing (model getting more efficient)
- Tool call failure rate decreasing
- Reward variance within groups decreasing (model becoming more consistent)

**Bad signs:**
- Reward hacking: model finds a shortcut that scores high but doesn't actually solve the task. Check manually.
- Mode collapse: all G rollouts become identical. Increase temperature or KL penalty.
- Training instability: loss oscillating wildly. Reduce learning rate.
- Validation success decreasing while training success increases. You're overfitting. Stop.

### Deliverable: A trained agent that measurably outperforms the SFT baseline.

---

## Phase 4: Ablations (Week 5–7)

**Goal:** Understand WHY it works. This is what separates an engineer from a researcher.

### Ablation 1: Reward components

| Experiment | Reward | Question |
|-----------|--------|----------|
| correctness only | 0/1 binary | Is the format/efficiency reward even needed? |
| correctness + format | add format score | Does penalizing bad tool calls help? |
| correctness + efficiency | add step penalty | Does the model learn to take fewer steps? |
| full reward | all three | Is the combination better than any single component? |

### Ablation 2: Group size

| G | Expected behavior |
|---|-------------------|
| 4 | Noisy advantages, faster training, less stable |
| 8 | Balanced (your default) |
| 16 | Better advantage estimates, 2x more expensive |
| 32 | Probably diminishing returns |

Plot: success rate vs training compute for each G. Find the sweet spot.

### Ablation 3: KL penalty

| Beta | Expected behavior |
|------|-------------------|
| 0.0 | No constraint — model drifts freely, likely reward hacking |
| 0.01 | Light constraint |
| 0.05 | Your default |
| 0.2 | Heavy constraint — model barely moves from SFT |

This one matters a lot. Too low = reward hacking. Too high = model can't improve.

### Ablation 4: Task difficulty

Split your tasks into easy (1–2 tool calls needed), medium (3–4), hard (5+). Train on each difficulty separately and combined. Does training on easy tasks transfer to hard tasks? Does mixing difficulties in training help?

### Ablation 5: SFT vs RLVR

The most important comparison:
- SFT-only agent (your baseline)
- RLVR-trained agent (your method)
- SFT with more data (control — what if you just had more SFT examples?)

If RLVR beats "SFT with more data," that's a strong result. It means RL discovers strategies that supervised learning can't teach from demonstrations alone.

### Deliverable: Tables and plots showing what works and why. This IS the paper.

---

## Phase 5: Evaluation and Writeup (Week 7–9)

### Final evaluation on held-out test set

Run both models (SFT baseline vs RLVR-trained) on the test set. Report:

| Metric | SFT Baseline | RLVR Agent |
|--------|-------------|------------|
| Overall success rate | X% | Y% |
| Category A (math) | | |
| Category B (multi-hop) | | |
| Category C (code) | | |
| Avg steps (successful tasks) | | |
| Tool call failure rate | | |
| Avg tokens per task | | |

### Qualitative analysis

Pick 10 tasks where SFT fails and RLVR succeeds. Show the trajectories side by side. Explain what the RLVR agent learned to do differently:
- Does it retry after failures?
- Does it plan better?
- Does it use tools more efficiently?
- Does it verify its answer before submitting?

Pick 5 tasks where RLVR still fails. Analyze why. What would it take to fix these?

### Also evaluate on GAIA

Run both agents on GAIA benchmark (your existing eval). This is an external benchmark you didn't train on — shows generalization.

### The writeup structure

```
Title: "Training Reliable Agents with Reinforcement Learning from Verifiable Rewards"

Abstract: 100 words

1. Introduction
   - Agents are unreliable. SFT teaches format, not strategy.
   - RLVR can teach models to reason about tool use.

2. Method
   - Agent architecture (your framework)
   - SFT baseline (Qwen 3B + function calling fine-tuning)
   - Task environment and verifiable rewards
   - GRPO training on agent trajectories
   - Token masking for tool results

3. Experiments
   - Task suite description
   - Baseline results
   - RLVR results
   - Ablations (reward components, group size, KL, difficulty)

4. Analysis
   - What does the RLVR agent learn? (qualitative trajectory comparison)
   - Failure analysis
   - Generalization to GAIA

5. Related Work
   - RLVR (DeepSeek-R1, OpenAI o1)
   - Agent training (Anthropic tool use, Toolformer, FireAct)
   - Agent evaluation (GAIA, AgentBench, WebArena)

6. Conclusion
```

### Deliverable: arXiv preprint + GitHub repo with full training code.

---

## Week-by-Week Timeline

| Week | What | Output |
|------|------|--------|
| 1 | Build task suite (500+ tasks), verify answers | tasks.json |
| 2 | Measure SFT baseline, build rollout engine | baseline_results.json |
| 3 | First GRPO training run, debug the loop | first trained checkpoint |
| 4 | Iterate on reward function, hyperparams | stable training |
| 5 | Run ablation 1–3 (reward, group size, KL) | ablation tables |
| 6 | Run ablation 4–5 (difficulty, SFT vs RLVR) | more tables |
| 7 | Final evaluation on test set + GAIA | final numbers |
| 8 | Qualitative analysis + trajectory comparisons | figures |
| 9 | Write paper, clean repo | arXiv submission |
| 10–12 | Interview prep, apply to labs | job |

---

## What Can Go Wrong (and What to Do)

**"RLVR doesn't improve over SFT"**
- Check reward signal. Is it too sparse? Add format reward.
- Check KL penalty. Too high = model can't move.
- Check rollout diversity. All same = no learning signal. Increase temperature.
- Check task difficulty. Start with easy tasks. If RLVR can't beat SFT on 2-step tasks, something is fundamentally broken.

**"Training is too slow"**
- Rollouts are expensive because each one = multiple LLM calls + tool execution.
- Reduce max steps from 10 to 5 initially.
- Use smaller G (4 instead of 8).
- Mock tool execution where possible (cache search results).
- Start with category A (math) only — simplest tasks, fastest iterations.

**"Model starts reward hacking"**
- Example: model always outputs the most common answer in training set.
- Fix: Increase KL penalty. Add diversity bonus. Check if training tasks are too similar.
- Example: model learns to call calculator with the answer directly instead of reasoning.
- Fix: This is actually fine if it gets the right answer. Check if it generalizes.

**"I can't generate enough tasks"**
- Use GPT-4o to generate tasks. Verify answers programmatically.
- Use existing datasets: HotpotQA, TriviaQA, GSM8K (wrapped with tool requirement), MBPP.
- 500 tasks is enough to start. You can add more if training is working.

**"Qwen 3B is too small for complex agent tasks"**
- Start with easy tasks (1–2 tool calls). Even 3B can learn these.
- If it works on easy tasks, that's already a publishable result.
- Scale to 7B later if you have the GPU budget.
