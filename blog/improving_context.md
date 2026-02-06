# Improving Context: Reasoning & Long Context in LLMs

---

## 1. Reasoning Over Long-Horizon Tasks

**Long-horizon tasks** are tasks that require many sequential steps, decisions, or intermediate sub-goals before reaching a final outcome. "Reasoning over" them means the model must plan, track state, and make coherent decisions across that entire chain.

### Concrete Examples

| Domain | Short-Horizon | Long-Horizon |
|--------|--------------|--------------|
| Coding | "Fix this typo" | "Build a REST API with auth, DB, tests, and deploy it" |
| Math | "What is 2+3?" | "Prove this theorem using 15 intermediate lemmas" |
| Agents | "Search the web" | "Research a topic, synthesize findings, write a report, iterate on feedback" |

### Why It's Hard for LLMs

1. **Error compounding** - A small mistake in step 3 of 20 derails everything downstream. Errors accumulate multiplicatively.
2. **State tracking** - The model must remember what it has done, what remains, and what intermediate results it produced, across a long context.
3. **Planning under uncertainty** - Early decisions constrain later options. The model needs to reason about consequences several steps ahead.
4. **Credit assignment** - When the final answer is wrong, it's hard to identify *which* step failed.

### How Models Address This

- **Chain-of-thought (CoT)** - Explicit step-by-step reasoning reduces the per-step difficulty.
- **Tree/graph search (e.g., ToT, GoT)** - Explore multiple reasoning paths, backtrack when stuck.
- **Reinforcement learning (GRPO, PPO on reasoning traces)** - Train models to get reward for correct *final* answers, forcing them to learn robust multi-step strategies. This is what DeepSeek-R1 and OpenAI's o1/o3 do.
- **Process reward models (PRMs)** - Give reward at *each step*, not just the final answer, so the model learns which intermediate steps are good.
- **Decomposition** - Break the long task into sub-tasks, solve each, then compose.

### Why This Matters

- Fine-tuning a model with **outcome-based RL** (like GRPO) teaches it to solve long-horizon tasks by trial and error over reasoning traces.
- **Distillation** from a strong reasoning model (e.g., DeepSeek-R1 into a smaller model) transfers long-horizon reasoning ability cheaply.
- The key insight: **supervised fine-tuning alone struggles with long-horizon reasoning** because it only imitates. RL lets the model *discover* better strategies through exploration.

This is exactly the frontier of "reasoning models" right now - the gap between models that can chain 5 steps vs. 50 steps reliably.

---

## 2. How Context Length Is Decided — Positional Encodings, Attention Cost & KV Cache

Context length is not a single knob. It's jointly determined by three interacting constraints.

### Constraint 1: Positional Encoding Scheme

Transformers have no inherent notion of order — self-attention is permutation-invariant. Positional encodings inject order. The scheme you pick dictates the ceiling.

| Scheme | How It Works | Limit |
|--------|-------------|-------|
| **Learned Absolute** (original GPT) | Learns an embedding vector for positions 0..N-1 | Hard ceiling at N. Position 4097 literally doesn't exist. |
| **Sinusoidal** (original Transformer) | Fixed sin/cos at different frequencies | Theoretically infinite, but untested positions degrade |
| **RoPE** (LLaMA, Qwen, most modern LLMs) | Encodes position as rotation angle in embedding space | Theoretically extrapolable, but rotations at unseen angles are out-of-distribution |
| **ALiBi** (BLOOM) | Adds linear bias to attention scores based on distance | Better extrapolation, but penalizes long-range attention by design |

#### RoPE — The Dominant Approach (and Its Limitation)

Most modern models use **Rotary Position Embedding (RoPE)**. For each dimension pair (i), the rotation angle at position m is:

```
θ_i(m) = m × base^(-2i/d)

where base = 10000 (typically), d = head dimension
```

**The problem**: If you train on positions 0 to 4096, the model has only ever seen rotation angles in that range. Position 8000 produces rotations the model has never encountered. It's like teaching someone a 12-hour clock then asking them to read a 24-hour clock — the mechanism is the same but the values are foreign. This is a **distribution shift**, and the model's output degrades unpredictably.

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

This is **per request**. A serving system handling 100 concurrent users at 128K context needs 4.2 TB just for KV cache — before any model weights.

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

Softmax attention distributes probability mass. Over long sequences, the middle tokens get starved — neither benefiting from primacy nor recency bias.

#### Problem 3: Effective vs. Claimed Context

A model "supporting" 128K and **effectively using** 128K are different things:

```
Claimed:  128K context
Needle retrieval at 128K: ~60%
Needle retrieval at 4K:   ~99%
```

The spec sheet number is the architectural limit. The effective limit is much lower.

### Context Extension Techniques — What Works and What Breaks

| Technique | How It Works | Trade-off |
|-----------|-------------|-----------|
| **Position Interpolation** | Scale positions down: pos' = pos × (L_train / L_target). So position 8000 → 4000 (within training range). | Reduces positional resolution. Nearby tokens look more similar. Fine distinctions blur. |
| **NTK-Aware Scaling** | Scale the RoPE frequency base instead of positions: base' = base × scale_factor | Better than naive interpolation. Preserves local resolution. Still needs fine-tuning on longer data. |
| **YaRN** | Combines NTK scaling with attention scaling. | Best extrapolation without full retraining. But still an approximation. |
| **ALiBi** | Linear distance bias on attention scores — no learned positions at all. | Good extrapolation, but linearly penalizes distance, limiting long-range attention by design. |
| **Sliding Window** (Mistral) | Each token only attends to last W tokens (e.g., W=4096). | O(n×W) instead of O(n²). But literally cannot retrieve beyond window. Stacking layers creates indirect receptive field, but it's lossy. |
| **Ring Attention** | Split sequence across GPUs, pass KV in a ring. | Solves memory. Does NOT solve the learning problem — model still must learn long-range patterns. |
| **Sparse Attention** (Longformer, BigBird) | Local + global + random attention patterns. | O(n) cost. But some token pairs never directly attend to each other — information is lost. |
| **Continue pre-training on longer sequences** | Actually train on progressively longer data: 4K → 16K → 64K → 128K. | Actually works. But extremely expensive (quadratic cost at each stage) and needs long documents with real dependencies. |

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

The current frontier (Claude, GPT-4, Gemini) pushes all three using progressive training, GQA for KV efficiency, and massive compute budgets.

---

## 3. What Does "Reasoning Across Long Context for Coding" Actually Mean?

### A Codebase Is Not a Document

When you read a novel, information flows linearly. When you read a codebase, information is a **graph**:

```
routes/auth.py  ──imports──►  services/user.py  ──imports──►  models/user.py
      │                              │                              │
      │                              ▼                              │
      │                       utils/hashing.py                      │
      │                                                             ▼
      └──────reads───────►  config/settings.py  ◄──reads──  db/migrations/003.py
                                     ▲
                                     │
                              tests/test_auth.py
```

A bug in `routes/auth.py` might exist because `models/user.py` changed a field name, which broke `services/user.py`, which returns `None` instead of raising, which causes a silent failure in the route. The **relevant information is scattered across 5 files that have no linear relationship to each other**.

This is fundamentally different from "read a long document and answer questions." This is "hold a graph in your head and trace paths through it."

### What "In Context" Means for an Agentic Coding Tool

An agentic coding tool runs an **agentic loop**, not a single prompt. Here's what actually happens when you say "fix the auth bug":

```
Turn 1: Model reads the error → decides it needs routes/auth.py → calls Read tool
Turn 2: Sees the route calls user_service.get_user() → calls Grep to find that function
Turn 3: Reads services/user.py → sees it queries User model → calls Read on models/user.py
Turn 4: Reads the model → notices field was renamed in a migration → reads migration
Turn 5: Now has all 5 files in context → reasons across ALL of them → produces fix
```

By turn 5, the context window contains:
- The original error message
- 5 source files
- The grep results
- The model's own reasoning from turns 1-4
- Maybe 30-50K tokens total

**"Reasoning across long context" means: at turn 5, the model is simultaneously attending to information from ALL of these sources to produce a coherent fix.** It's not just looking at the file it's currently editing — it's cross-referencing the migration it saw in turn 4 with the model definition from turn 3 with the service call from turn 2.

### What "Reasoning Across" Actually Requires

There are specific cognitive operations that "reasoning across context" implies:

#### Cross-File Causal Tracing

```python
# File A (in context at position ~2K)
def get_user(id):
    return db.query(User).filter(User.user_id == id).first()  # returns None if not found

# File B (in context at position ~15K)
def login(request):
    user = get_user(request.id)
    token = generate_token(user.email)  # ← crashes: NoneType has no attribute 'email'
```

The model must connect a `None` return in one file to an unguarded attribute access in another file. These two code fragments might be **13,000 tokens apart** in the context. The model's attention mechanism must literally assign high attention weight from the `user.email` token to the `return ... .first()` token across that gap.

**This is what "lost in the middle" kills.** A weaker model sees File B, generates a fix like `if user is None: return error` — which is correct but shallow. A model reasoning across full context also notices that `get_user` should probably raise `UserNotFoundError` instead of returning `None`, because 6 other call sites (also in context from earlier grep results) all assume it returns a valid user.

#### Pattern Recognition Across the Codebase

After reading 8-10 files, the model should recognize:
- "This codebase uses the repository pattern"
- "Errors are custom exceptions caught by middleware, not return codes"
- "Every endpoint has a corresponding Pydantic schema in `schemas/`"
- "Tests use factory_boy fixtures, not raw object creation"

This is **not** stored in any single file. It emerges from reasoning across the **aggregate** context. When the model then generates new code, it should follow these patterns — not because it was told to, but because it **inferred** them from the context.

A model that can't reason across long context will generate code that is locally correct but **stylistically alien** to the codebase.

#### Consistency Maintenance Across Edits

Say the model needs to rename `user_id` to `account_id` across the codebase. It:
- Finds 23 occurrences across 11 files
- Must change ALL 23, not 20
- Must change them in type definitions, function signatures, database queries, test assertions, API schemas, and migration files
- Must NOT change the string `"user_id"` in a comment explaining the old schema

This requires **holding all 23 occurrences in working memory simultaneously** and reasoning about each one's semantic role. Miss one and you've introduced a bug. Change a wrong one and you've introduced a different bug.

#### Multi-Step Plan Coherence

```
Step 1: Add new column to User model        → produces migration
Step 2: Update UserSchema to include field   → must match the column type from step 1
Step 3: Update create_user service           → must use the schema from step 2
Step 4: Update API endpoint                  → must match the service signature from step 3
Step 5: Add test                             → must test the endpoint from step 4 with
                                                the schema from step 2 using the model from step 1
```

By step 5, the model is writing test code that must be **simultaneously consistent with decisions made in steps 1-4**, all of which are in the context from earlier turns. The test creates a User (step 1's schema), sends it through the API (step 4's route), validates the response (step 2's schema), and checks the DB (step 1's model).

If the model "forgets" that it used `account_id` in step 1 and writes `user_id` in the step 5 test, **the entire chain breaks**.

### The Difference Between "Holds" and "Reasons Across"

This is the crux. Many models support 128K tokens. The question is what happens to information at various positions:

```
Model that HOLDS long context:
  ✓ Can regurgitate text from position 60K if asked "what was in file X?"
  ✗ Does NOT spontaneously use info from position 60K when generating code at position 120K

Model that REASONS ACROSS long context:
  ✓ While generating code at position 120K, attention heads actively pull
    information from position 60K because it's relevant
  ✓ Does this without being explicitly told "look at file X"
  ✓ Does this even when the connection is implicit (same variable name,
    compatible type signature, related business logic)
```

The difference is whether the model's **attention patterns during generation** actually reach back and meaningfully integrate distant information, versus just having it present but functionally ignored.

### Why This Is So Hard (The Non-Obvious Part)

The real difficulty isn't memory — it's **relevance detection at scale**.

At 100K tokens of context, there might be 500 function definitions, 200 class attributes, 50 config values, and 100 test assertions. When generating one line of code, maybe 3-4 of those are relevant. The model must:

1. **Not attend to** the 796 irrelevant items (noise suppression)
2. **Strongly attend to** the 4 relevant items (signal detection)
3. Do this **for every token it generates**

This is a needle-in-a-haystack problem **at every single generation step**, not just once. And the "needle" changes for every token — the relevant context for choosing a function name is different from the relevant context for choosing its arguments.

```
Generating: token = generate_token(user.?????)

To decide this next token, the model must:
- Recall User model definition (position 8K): field is called "email_address" not "email"
- Recall generate_token signature (position 22K): first param is type str
- Recall project convention (positions 5K, 12K, 31K): always uses model attributes directly
- Ignore 99.5% of everything else in context

→ Output: "email_address"
```

### Summary: What "Reasons Across Long Context for Coding" Claims

1. **The agentic loop** gathers the right files into context (tool use + search strategy)
2. **The model** attends to information across the full accumulated context — not just the most recent file
3. **Cross-references** are made between distant parts of the context without explicit prompting
4. **Generated code** is consistent with patterns, types, names, and conventions observed anywhere in the context, not just locally
5. **Multi-step edits** maintain coherence from first change to last
6. **Causal chains** spanning multiple files are traced correctly

It's the combination of the agentic scaffolding (getting the right info into context) and the model's actual attention quality over that context (using the info once it's there) that makes it work.

### Implications for Fine-Tuning & Evaluation

- **Evaluation is hard.** You can't test "long context reasoning for code" with a simple needle-in-a-haystack test. You need:
  - Multi-file edit consistency benchmarks
  - Cross-file bug localization tasks
  - "Apply this pattern codebase-wide" tasks
  - Multi-step refactoring that must maintain correctness at each step
- **Post-training (RLHF/RL) matters** — supervised fine-tuning teaches the model what good code looks like locally, but RL teaches it to **maintain coherence across the full context**.

---

## 4. Why 200K Context and Not 1M? — And Why 100K Is Enough

### The Problem: Bigger Context ≠ Better Reasoning

| Context Length | Needle Retrieval | Reasoning Quality |
|---------------|-----------------|-------------------|
| 4K | ~99% | excellent |
| 32K | ~97% | very good |
| 100K | ~90% | good |
| 200K | ~85% | acceptable |
| 500K | ~60% | degraded |
| 1M | ~40% | significantly degraded |

Going from 200K to 1M doesn't give 5x useful context — it gives ~800K tokens the model **barely attends to**. Three forces make larger context impractical:

**Cost**: At 1M context, a 30-turn coding session costs ~$36 vs ~$7 at 200K (5x). Most of those extra tokens are noise.

**Latency**: Attention is O(n²). Doubling context = 4x compute. 200K→1M = **25x attention cost**. Prefill jumps from ~3-5s to ~15-30s per turn, making an agentic loop unusable.

**Training**: Training on 1M-length sequences costs 62,500x more per sample than 4K. And finding millions of high-quality 1M-token documents with genuine long-range dependencies is nearly impossible.

### The Solution: Smart Agents Beat Big Context

#### Problem: Current agents waste most of their context

```
Typical 56K token session breakdown:
  System prompt + instructions:       3K   ✓ needed
  Relevant source files:             7.5K  ✓ needed
  Grep results:                       3K   ~60% useful
  Irrelevant file (wrong lead):      1.5K  ✗ wasted
  Model's stale reasoning history:   12K   ✗ redundant after acting on it
  Tool call formatting overhead:      4K   ✗ pure overhead
  Old conversation turns (verbatim): 25K   ✗ mostly stale
  ─────────────────────────────────────────
  Total:                             56K
  Actually needed for the fix:       ~12K  ← only 21% is signal
```

#### Solution 1: Surgical Retrieval — Read functions, not files

```
Dumb:   Read all of user_service.py (2000 lines)      → ~8K tokens
Smart:  Read lines 145-178 (the one relevant function) → ~200 tokens
                                                         40x reduction
```

**Why it works**: The agent already knows (from the stack trace or grep) which function matters. Reading the whole file dumps 1800 lines of noise into context that competes for attention with the 30 lines that matter.

#### Solution 2: Context Eviction — Proactively discard stale information

```
Turn 3:  Read models/user.py to understand schema     → 800 tokens in context
Turn 5:  Already used that info to write the fix

Current: Keep the full file verbatim forever           → 800 tokens wasted
Smart:   Compress to "User model: id, email_address, created_at"  → 20 tokens
                                                         40x reduction
```

**Why it works**: After you've acted on information, you rarely need it verbatim. A summary preserves the decision-relevant facts while freeing context for new information. Current agents only compress when hitting limits — smart agents compress proactively.

#### Solution 3: Structured Memory — Notes over verbatim copies

```
Current (raw context):
  Full file stored in conversation history  → 2000 tokens per file
  Degrades as it drifts further from attention window

Optimized (structured scratchpad):
  {
    "file": "models/user.py",
    "facts": ["email_address: str", "uses SQLAlchemy", "has created_at"],
    "re_read_lines": "145-150"    ← pointer to re-read if needed
  }
  → 30 tokens, and exact lines are one tool call away
```

**Why it works**: Gives the agent an external "notebook" instead of relying on attention over raw tokens. The model decides what's important, writes it down concisely, and can re-fetch details on demand.

#### Solution 4: Plan Before Reading — Avoid wrong leads entirely

```
Dumb:   "auth broken" → reads auth.py, user.py, session.py, middleware.py,
        config.py, database.py → discovers only auth.py + user.py mattered
        Cost: 6 file reads × ~2K = 12K tokens, 4 were wasted

Smart:  "auth broken" → reads stack trace (200 tokens) → thinks "crash is in
        login() calling get_user(), need those two files only" → reads 2 files
        Cost: 2 file reads × ~2K = 4K tokens, 0 wasted
```

**Why it works**: 200 tokens of planning saves 8K tokens of unnecessary reads. The model spends attention budget thinking about what to retrieve rather than dumping everything and hoping attention sorts it out.

### The Core Principle

```
Unlimited context → lazy agent → dump everything → hope attention finds signal
                                                    (it often doesn't)

Tight context     → disciplined agent → precise retrieval → every token earns its place
                                                             (attention concentrated on signal)
```

A **100K agent that's selective will outperform a 1M agent that's sloppy** because:
- Higher signal-to-noise ratio → attention works better on what's there
- Less "lost in the middle" because there's less middle
- Faster per-turn → more iterations possible → better refinement
- Cheaper per turn → can afford longer sessions

This mirrors **human working memory**: we hold ~7 items. That constraint forces us to organize, prioritize, and abstract — producing better reasoning than trying to hold everything at once.

### Where the Field Is Heading

```
Phase 1 (2023): "Make context bigger"     → brute force, GPT-4 128K, Gemini 1M
Phase 2 (2024): "Make context smarter"    → better attention training, YaRN, RoPE scaling
Phase 3 (2025+): "Make agents leaner"     → use less context more effectively
                  ├── Surgical retrieval   (read functions, not files)
                  ├── Active eviction      (discard stale context proactively)
                  ├── Structured memory    (notes > verbatim copies)
                  ├── Plan-then-retrieve   (think before reading)
                  └── Parallel targeted reads (speculative tool use)
```

**The endgame**: An agent that navigates a million-file codebase using a 50K context window, because it knows exactly where to look and what to remember. The next breakthrough won't be "2M context" — it'll be "same quality at 64K with 10x less cost and 5x less latency."

---

## 5. How Agentic Coding Tools Achieve Long-Context Reasoning (Claude Code as Example)

No vector database. No knowledge graph. No external memory. Just a flat conversation where the **model is simultaneously the retriever, the planner, and the reasoner**.

### The Architecture: Everything Is a Conversation

```
Msg 1:  [system]     System prompt + project instructions         (~3K tokens)
Msg 2:  [user]       "Fix the auth bug"                          (~50 tokens)
Msg 3:  [assistant]  Thinking + tool_call: Grep("auth error")    (~200 tokens)
Msg 4:  [tool]       Grep results: routes/auth.py:47             (~500 tokens)
Msg 5:  [assistant]  Thinking + tool_call: Read("routes/auth.py")
Msg 6:  [tool]       File contents                                (~2K tokens)
Msg 7:  [assistant]  Thinking + tool_call: Read("services/user.py")
Msg 8:  [tool]       File contents                                (~1.5K tokens)
Msg 9:  [assistant]  Thinking + tool_call: Edit(...)
Msg 10: [tool]       Edit successful
Msg 11: [assistant]  "Fixed. Here's what I changed..."
```

At message 11, the model sees **ALL of messages 1-10**. "Long context reasoning" = at this point, attention reaches back across every previous message simultaneously to produce a coherent fix.

### Mechanism 1: The Model IS the Retriever

```
RAG pipeline:
  Query → Embedding model → Vector search → Top-K chunks → LLM generates
  Problem: Embedding model and LLM are DIFFERENT models.
           Retriever doesn't know what the generator will need.
           Selection is based on surface similarity, not causal reasoning.

Agentic approach:
  Question → Claude thinks "crash is in login(), which calls get_user()" → Grep for get_user
           → Claude reads result, thinks "defined in user.py, need to check return type" → Read
           → The SAME model choosing what to retrieve will use what it retrieves.
```

The critical difference:
```
RAG retriever:   "This text is 0.87 similar to the query"     (statistical)
Agentic retriever: "The crash is in login() calling get_user(),
                    I need to read where get_user is defined"   (causal reasoning)
```

No embedding approximation. No similarity threshold. The model reasons about code dependencies and retrieves based on **understanding**, not pattern matching.

### Mechanism 2: Iterative Context Building

The agent doesn't load everything at once. Understanding is built turn by turn:

```
Turn 1: Read error       → "login endpoint crashes with NoneType"
Turn 2: Read auth.py     → "user.email fails because user is None"
Turn 3: Grep get_user    → "get_user is in services/user.py, returns .first()"
Turn 4: Read user.py     → ".first() returns None when no match, no error raised"
Turn 5: Read tests       → "tests expect UserNotFoundError — but it's never raised"
```

Each turn refines understanding. By turn 5, the model has a complete causal chain in context AND has been building its mental model incrementally. This is fundamentally better than dumping 5 files cold — the **order of discovery** helps the model organize information.

### Mechanism 3: Automatic Context Compression

When context approaches 200K, older messages get summarized:

```
Before (approaching limit):
  Messages 1-30: Full verbatim content                    (~180K tokens)

After compression:
  Messages 1-15: Summarized by a fast model               (~8K tokens)
    "Investigated auth bug. get_user() returns None instead of raising.
     Fixed routes/auth.py and services/user.py. User asked to update tests."
  Messages 16-30: Full verbatim content                    (~90K tokens)
  Total: ~98K — back under budget
```

This mirrors what attention does naturally (recent = high attention, old = low attention), but makes it **explicit and honest** instead of pretending the model attends well to 180K tokens.

### Mechanism 4: Ground Truth Over Approximation

Every tool result is **actual file content**, not an embedding or summary:

```
Vector RAG:     Retrieves chunk that's 0.89 similar. Maybe correct. Maybe stale index.
Claude Code:    Reads the actual file right now. Always current. Exact syntax.
```

For code, this matters enormously — a single wrong character (`.email` vs `.email_address`) breaks everything. Embeddings can't distinguish these; direct file reads can.

### What Attention Does During Generation

When generating an edit, the model's attention simultaneously reaches:

```
Position ~500    (system prompt):   "write safe, secure code"        → error handling style
Position ~1200   (user message):    "fix the auth bug"               → intent
Position ~5000   (grep results):    "get_user at user.py:152"        → function location
Position ~8000   (user.py source):  "return ...first()"              → knows it returns Optional
Position ~12000  (auth.py source):  "user.email"                     → the crash point
Position ~14000  (test file):       "assert raises(UserNotFoundError)"→ expected behavior

All positions attended to simultaneously at every generated token.
```

A model "good at long context" = attention heads at position 15000 successfully assign high weight to relevant tokens at positions 500, 5000, 8000, 12000, 14000 — without being told which positions to look at.

### The Design Tradeoff

```
What this approach gives up:
  ✗ No persistence across sessions (re-explores each time)
  ✗ No pre-indexed codebase (must search fresh)
  ✗ Each tool call has latency (not instant retrieval)

What this approach gains:
  ✓ Zero setup — works on any codebase immediately
  ✓ Always-fresh reads — no stale index
  ✓ 100% ground truth — actual file contents, not approximations
  ✓ Retrieval guided by reasoning, not statistical similarity
  ✓ Simple architecture — no external systems to maintain
```

### Summary

```
How agentic coding tools achieve long-context reasoning:

1. ARCHITECTURE:   Flat conversation. No hidden memory. No vector DB.
2. RETRIEVAL:      The model IS the retriever — reasons about what to read.
3. BUILDING:       Context built iteratively, each turn refining understanding.
4. COMPRESSION:    Old context summarized when approaching limits.
5. GROUND TRUTH:   Tool results are actual file contents, not approximations.
6. ATTENTION:      At generation, model attends across ALL accumulated context.

Works because the model is good enough to:
  plan retrieval → execute it → reason across all results simultaneously
```

---

## Key Concepts Reference

| Concept | One-Line Summary |
|---------|-----------------|
| Long-horizon reasoning | Planning and executing across many dependent steps without error compounding |
| Lost in the middle | Models attend to start/end of context but lose information in the middle |
| Effective vs. claimed context | Architectural support ≠ actual retrieval/reasoning ability at that length |
| Cross-file causal tracing | Connecting cause in file A to effect in file B across large token distances |
| Holds vs. reasons across | Storing tokens ≠ actively using them during generation |
| Relevance detection at scale | Picking the 4 relevant items out of 800 at every generation step |
| Process reward models | Rewarding each reasoning step, not just the final answer |
| Agentic context accumulation | Iteratively gathering the right information into context via tool use |
| Surgical retrieval | Reading only the relevant function/lines instead of entire files |
| Context eviction | Proactively discarding stale information to keep context tight |
| Structured memory | Storing extracted facts + line pointers instead of verbatim file content |
| Plan-then-retrieve | Spending tokens thinking about what to read before reading anything |
| Signal-to-noise ratio | The % of context tokens that actually contribute to the current generation |
| O(n²) attention cost | Why doubling context = 4x compute, making 1M context 25x costlier than 200K |
| RoPE (Rotary Position Embedding) | Encodes position as rotation angle; works within training range, degrades outside it |
| Position interpolation | Scale positions down to fit trained range; trades resolution for extrapolation |
| YaRN / NTK scaling | Scale RoPE frequency base instead of positions; better local resolution, still needs fine-tuning |
| KV cache | Stored Key/Value vectors per token per layer; 327 KB/token for 70B model, main memory bottleneck at inference |
| GQA (Grouped Query Attention) | Share KV heads across query heads to reduce KV cache size (e.g., 64 query → 8 KV heads) |
| Training distribution mismatch | Model never practiced attending to position 50K if trained on 4K docs — learned behavior gap |
| Context trilemma | Long + Quality + Efficient — pick at most two |
| Model-as-retriever | The generating model also decides what to retrieve, replacing separate embedding-based search |
| Iterative context building | Understanding built turn-by-turn; order of discovery helps the model organize information |
| Automatic context compression | Summarize old turns when approaching limits; honest version of attention decay |
