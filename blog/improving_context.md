# Context in LLMs: What Determines It, What It Costs, and What Actually Works

---

## 1. What Determines Context Length

When people say a model "supports 128K context," that sounds like a single setting you can crank up. It's not. Context length is determined by three things working together: how the model knows token order, how much attention costs, and how much memory inference eats.

### Constraint 1: Positional Encoding Scheme

Transformers have no inherent notion of order. Self-attention is permutation-invariant. Positional encodings inject order. The scheme you pick dictates the ceiling.

| Scheme | How It Works | Limit |
|--------|-------------|-------|
| **Learned Absolute** (original GPT) | Learns an embedding vector for positions 0..N-1 | Hard ceiling at N. Position 4097 literally doesn't exist. |
| **Sinusoidal** (original Transformer) | Fixed sin/cos at different frequencies | Theoretically infinite, but untested positions degrade |
| **RoPE** (LLaMA, Qwen, most modern LLMs) | Encodes position as rotation angle in embedding space | Theoretically extrapolable, but rotations at unseen angles are out-of-distribution |
| **ALiBi** (BLOOM) | Adds linear bias to attention scores based on distance | Better extrapolation, but penalizes long-range attention by design |

#### RoPE: The Dominant Approach (and Its Limitation)

Most modern models use **Rotary Position Embedding (RoPE)**. For each dimension pair (i), the rotation angle at position m is:

```
θ_i(m) = m × base^(-2i/d)

where base = 10000 (typically), d = head dimension
```

**The problem**: If you train on positions 0 to 4096, the model has only ever seen rotation angles in that range. Position 8000 produces rotations the model has never encountered. It's like teaching someone a 12-hour clock then asking them to read a 24-hour clock. The mechanism is the same but the values are foreign. This is a **distribution shift**, and the model's output degrades unpredictably.

### Constraint 2: Attention's Quadratic Cost

Self-attention computes pairwise scores between **every** token pair:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V

QK^T matrix is n × n where n = sequence length
Memory:  O(n²)
Compute: O(n² × d)
```

| From → To | Token increase | Attention cost increase |
|-----------|---------------|----------------------|
| 4K → 8K | 2x | **4x** |
| 4K → 32K | 8x | **64x** |
| 4K → 128K | 32x | **1,024x** |
| 4K → 1M | 250x | **62,500x** |

This is the fundamental reason you can't "just make context longer." The cost grows with the **square** of the length.

### Constraint 3: KV Cache at Inference

At inference, you store Key and Value vectors for every past token, every layer. For a 70B model with GQA (8 KV heads):

```
Per token KV cost:
  = num_layers × 2(K,V) × num_kv_heads × head_dim × 2 bytes(fp16)
  = 80 × 2 × 8 × 128 × 2
  ≈ 327 KB per token

At different context lengths:
   4K context:  →  1.3 GB   (manageable)
  32K context:  → 10.5 GB   (one GPU)
 128K context:  →  42 GB    (needs multiple GPUs JUST for KV cache)
   1M context:  → 327 GB    (impossible on current hardware for one request)
```

This is **per request**. A serving system handling 100 concurrent users at 128K context needs 4.2 TB just for KV cache, before any model weights.

### Why You Can't "Just Extend" the Context

#### Problem 1: Training Distribution Mismatch (the deepest reason)

The model learns attention patterns from training data. If most training documents are 2-8K tokens:
- It learns "the answer is usually within a few thousand tokens of the question"
- It learns attention heads that specialize in **local** patterns
- It **never practices** retrieving a fact from 50K tokens away

Even if you architecturally support 128K, the model hasn't learned **when and how** to attend across that distance. This is a learned behavior problem, not an architecture problem.

#### Problem 2: Lost in the Middle (Liu et al., 2023)

Even models with "long context" show a U-shaped retrieval curve:

```
Retrieval accuracy by position of relevant info:
Beginning: ████████████████████ 95%
Middle:    ████████             40%  ← catastrophic drop
End:       ███████████████████  90%
```

Softmax attention distributes probability mass. Over long sequences, the middle tokens get starved, neither benefiting from primacy nor recency bias.

#### Problem 3: Effective vs. Claimed Context

A model "supporting" 128K and **effectively using** 128K are different things:

```
Claimed:  128K context
Needle retrieval at 128K: ~60%
Needle retrieval at 4K:   ~99%
```

The spec sheet number is the architectural limit. The effective limit is much lower.

### The Fundamental Trilemma

```
        Long Context
           /\
          /  \
         /    \
        /      \
Quality -------- Efficiency
```

Pick at most two:
- **Long + Quality** = massive compute (full attention, long training)
- **Long + Efficient** = sparse/approximate attention (loses quality)
- **Quality + Efficient** = short context (what most models do best)

---

## 2. Strategies for Extending and Managing Context

There are two fundamentally different approaches: **make the window bigger** (architectural) or **make the model smarter within the window it has** (behavioral). Both have their place.

### Architectural Strategies: Making the Window Bigger

| Technique | How It Works | Trade-off |
|-----------|-------------|-----------|
| **Position Interpolation** | Scale positions down: pos' = pos × (L_train / L_target) | Reduces positional resolution. Nearby tokens look more similar. |
| **NTK-Aware Scaling** | Scale the RoPE frequency base instead of positions | Better than naive interpolation. Preserves local resolution. Still needs fine-tuning. |
| **YaRN** | Different scaling for different frequency dimensions. High-freq minimal, low-freq aggressive. Plus attention temp scaling. | Best extrapolation quality. Most complex. Still needs some training. |
| **ALiBi** | Linear distance bias on attention scores, no learned positions. | Good extrapolation, but linearly penalizes distance by design. |
| **Sliding Window** (Mistral) | Each token only attends to last W tokens (e.g., W=4096). | O(n×W) instead of O(n²). But cannot retrieve beyond window. |
| **Ring Attention** | Split sequence across GPUs, pass KV in a ring. | Solves memory. Does NOT solve the learning problem. |
| **Sparse Attention** (Longformer, BigBird) | Local + global + random attention patterns. O(n) cost. | Some token pairs never directly attend, information lost. Replaced by Flash Attention. |
| **Flash Attention** | Exact same math, tiled computation fits in GPU SRAM. O(n) memory, O(n²) compute. | **No quality loss.** Solves memory. Does NOT reduce compute. What everyone uses. |
| **GQA** (Grouped Query Attention) | Share KV heads across query heads (64 query → 8 KV heads). | Reduces KV cache by 8x. Minimal quality loss. Standard in modern models. |

**Key distinction:** Flash Attention is NOT sparse attention. Sparse drops connections (quality loss). Flash computes exact full attention but tiles for memory efficiency (no quality loss). Don't confuse them.

### Behavioral Strategies: Making the Model Smarter Within Its Window

Instead of making the window bigger, teach the model to **use its existing window more intelligently**.

#### Strategy 1: Surgical Retrieval. Read functions, not files
```
Dumb:   Read all of user_service.py (2000 lines)      → ~8K tokens
Smart:  Read lines 145-178 (the one relevant function) → ~200 tokens
                                                         40x reduction
```

#### Strategy 2: Context Eviction. Proactively discard stale information
```
Turn 3:  Read models/user.py to understand schema     → 800 tokens in context
Turn 5:  Already used that info to write the fix

Current: Keep the full file verbatim forever           → 800 tokens wasted
Smart:   Compress to "User model: id, email_address, created_at"  → 20 tokens
                                                         40x reduction
```

#### Strategy 3: Structured Memory. Notes over verbatim copies
```
Optimized (structured scratchpad):
  {
    "file": "models/user.py",
    "facts": ["email_address: str", "uses SQLAlchemy", "has created_at"],
    "re_read_lines": "145-150"    ← pointer to re-read if needed
  }
  → 30 tokens, and exact lines are one tool call away
```

#### Strategy 4: Plan Before Reading. Avoid wrong leads entirely
```
Dumb:   "auth broken" → reads 6 files → discovers only 2 mattered
        Cost: 12K tokens, 8K wasted

Smart:  "auth broken" → reads stack trace (200 tokens) → reads 2 files
        Cost: 4K tokens, 0 wasted
```

#### Strategy 5: Compaction as a Tool
Give the agent a `compact()` tool and let it **learn when to use it** through RLVR. The model learns through trial and error: compact too early and you lose critical details. Compact too late and context fills up with noise.

### The Core Principle

```
Unlimited context → lazy agent → dump everything → hope attention finds signal
                                                    (it often doesn't)

Tight context     → disciplined agent → precise retrieval → every token earns its place
                                                             (attention concentrated on signal)
```

A **100K agent that's selective will outperform a 1M agent that's sloppy**.

---

## 3. At Which Stage Can You Do This?

LLM development has four stages. Each offers different knobs for improving context, at very different cost levels.

```
Stage 1: Pretraining            → set context capacity, learn attention patterns
Stage 2: Mid-Training           → extend context with RoPE scaling
Stage 3: Post-Training (SFT/RL) → teach smart context management behaviors
Stage 4: Agent Scaffolding       → system-level context management
```

### Stage 1: Pretraining
- Choose positional encoding, attention architecture, progressive length training
- Who: Only frontier labs. Cost: $10M-1B+. Not realistic for individuals.

### Stage 2: Mid-Training (Context Extension)
- Take model pretrained at 4K-8K, extend to 32K-128K with RoPE scaling
- Examples: Llama 2 (4K) → Code Llama (100K), Mistral (8K) → Mistral-128K
- Cost: $10K-1M. Realistic only at the small end for individuals.

### Stage 3: Post-Training (SFT and RL)
- SFT on multi-turn agent trajectories. RLVR to teach when to compact, extract, and manage context.
- Cost: $500-1200. **This is the sweet spot for individuals.**

### Stage 4: Agent Scaffolding
- Auto compaction, prompt caching, selective reading, conversation summarization
- Cost: $0 (engineering time only). Accessible to anyone.

| Stage | What It Does | Cost | Who Does It |
|-------|-------------|------|-------------|
| Pretraining | Sets context capacity | $10M-1B+ | Frontier labs only |
| Mid-Training | Extends context window | $10K-1M | Labs, startups |
| Post-Training | Teaches smart context behaviors | $500-1200 | Anyone with GPUs |
| Agent Scaffolding | System-level context mgmt | $0 | Anyone |

---

## 4. Honest Cost Analysis

### Cost of Training with Long Context

| Context Length | Cost per Sample (vs 4K) | What You Need |
|---------------|------------------------|---------------|
| 4K | 1x | Standard GPU |
| 32K | 64x | Flash Attention, multi-GPU |
| 128K | 1,024x | Multi-node, Ring Attention |
| 1M | 62,500x | Frontier lab budget |

### Cost of Inference with Long Context

| Context | Cost per Turn | 30-Turn Session |
|---------|-------------|-----------------|
| 4K | ~$0.01 | ~$0.30 |
| 32K | ~$0.08 | ~$2.40 |
| 100K | ~$0.25 | ~$7.50 |
| 200K | ~$0.50 | ~$15.00 |
| 1M | ~$2.50 | ~$75.00 |

### Cost of Self-Hosted Inference

```
Qwen 3B on A100 ($1.10/hr):
  4K context:   ~2700 tasks/hour  → $0.0004/task
  32K context:  ~500 tasks/hour   → $0.002/task

70B on 4x A100 ($4.40/hr):
  4K context:   ~200 tasks/hour   → $0.022/task
  32K context:  ~50 tasks/hour    → $0.088/task
  128K context: ~10 tasks/hour    → $0.440/task
```

**The honest conclusion:** Past 100K tokens, you're paying exponentially more for linearly diminishing returns. Invest in smarter context management, not bigger windows.

---

## 5. What Reasoning Over Long Context Actually Means

With the mechanics and costs established, let's address what everyone is actually chasing: making models **reason across** long context, not just hold it.

### Long-Horizon Tasks
Long-horizon tasks require many sequential steps before reaching a final outcome. Error compounding, state tracking, planning under uncertainty, and credit assignment make these hard.

### "Holds" vs. "Reasons Across"
- **Holds**: Can regurgitate text from position 60K if asked
- **Reasons across**: While generating at position 120K, attention actively pulls from position 60K because it's relevant, without being told to look there

### What "Reasoning Across" Looks Like in Code
- **Cross-file causal tracing**: Connecting a None return in file A to a crash in file B, 13,000 tokens apart
- **Pattern recognition**: Inferring codebase conventions from reading 8-10 files
- **Multi-step coherence**: Step 5's test must be consistent with decisions from steps 1-4
- **Relevance detection at scale**: Picking the 4 relevant items out of 800 at every generation step

### How Models Address Long-Horizon Reasoning
- Chain-of-thought, Tree/graph search, RL (GRPO, PPO), Process reward models, Decomposition
- Key insight: SFT imitates, RL discovers. The gap between chaining 5 vs 50 steps is the frontier.

### How Agentic Tools Put It All Together
No vector DB. No knowledge graph. The model is simultaneously the retriever, planner, and reasoner. Context built iteratively. Old context compressed. Ground truth file reads, not embeddings.

---

## 6. Where the Field Is Heading

```
Phase 1 (2023): "Make context bigger"     → brute force, GPT-4 128K, Gemini 1M
Phase 2 (2024): "Make context smarter"    → better attention training, YaRN, RoPE scaling
Phase 3 (2025+): "Make agents leaner"     → use less context more effectively
```

The endgame is not "2M context." It's an agent that navigates a million-file codebase using a 50K context window, because it knows exactly where to look and what to remember.

**Bottom Line:** Context length is determined by architecture. It can be extended at different stages, at vastly different costs. The most impactful and accessible approach is to make the model smarter within its existing window, through post-training (SFT/RLVR) and agent-level strategies.
