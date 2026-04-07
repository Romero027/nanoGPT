# nanoGPT Summary

All code lives in a single file: `model.py` (~330 lines). The default config (GPT-2 124M) uses these dimensions:

```python
@dataclass
class GPTConfig:
    block_size: int = 1024 # Context Window
    vocab_size: int = 50304 # Number of tokens in the tokenizer vocab
    n_layer: int = 12 # Number of transformer block stacked
    n_head: int = 12 # Number of attention head
    n_embd: int = 768 # The hidden dimension of the transformer -> every token becomes a vector of size 768
    dropout: float = 0.0 # Randomly zeros some activations, used to prevent overfitting
    bias: bool = True # Whether linear layers include bias terms 
```

Throughout this plan, we use **B=batch, T=sequence length (up to 1024), C=768, nh=12 heads, hs=64 (head size = C/nh)**.

---

## Block 1: The High-Level Transformer Block

The `Block` class is the single repeating unit stacked 12 times. It is only 4 lines of logic:

```python
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        # x: (B (batch size), T (sequence length), C （embedding dimension)) -> output: (B, T, C)
        # Residual connections around both sub-layers for stable gradient flow
        # Attention internally changes shapes but returns the same shape as input 
        x = x + self.attn(self.ln_1(x))  # (B, T, C) -> LN(B, T, C)->(B, T, C) -> attn -> (B, T, C), then add residual
        x = x + self.mlp(self.ln_2(x))   # (B, T, C) -> LN(B, T, C)->(B, T, C) -> MLP -> (B, T, C), then add residual
        return x
```

### 1.1 LayerNorm (the "norm")

LayerNorm normalizes each token's embedding vector independently: for a given token, it computes the mean and variance across the C (embedding) dimension, subtracts the mean, and divides by the standard deviation. This re-centers and re-scales the activations so they don't drift to extreme values as they flow through many layers. After normalizing, it applies a learnable affine transform (scale `weight` and optional `bias`) so the network can still represent whatever distribution it needs.

See `model.py` lines 18–27. nanoGPT uses **pre-norm** (norm *before* attention/MLP, not after). This is the GPT-2 / modern convention. The custom `LayerNorm` exists only to support `bias=False`:

```python
class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
```

**Key questions to answer:**

- What does LayerNorm normalize over? — The last dimension, i.e., the 768-dim embedding for each token independently.
- Why pre-norm instead of post-norm? — Training stability: gradients flow more cleanly through the residual path.

### 1.2 CausalSelfAttention (the "attention")

See `model.py` lines 29–76. This is the heart of the transformer. We dissect it thoroughly in Block 2. For now, understand its role: each token attends to all *previous* tokens (causal = can't look ahead) and produces a weighted combination of their values.

### 1.3 MLP (the "feed-forward")

See `model.py` lines 78–92.

```python
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
```

A two-layer feed-forward network applied to each token position independently:

| Step | Operation | Shape |
|------|-----------|-------|
| Input | `x` | `(B, T, 768)` |
| Up-project | `c_fc` | `(B, T, 768)` → `(B, T, 3072)` |
| Activation | GELU | `(B, T, 3072)` (same) |
| Down-project | `c_proj` | `(B, T, 3072)` → `(B, T, 768)` |
| Output | after dropout | `(B, T, 768)` |

**Key questions to answer:**

- Why expand to 4x then contract? — The expansion gives the network more capacity to learn nonlinear functions before projecting back.
- Why GELU instead of ReLU? — Smoother activation, better training dynamics. Standard in GPT-2 and later.

### 1.4 Residual Connections

Look again at `Block.forward`:

```python
x = x + self.attn(self.ln_1(x))   # residual around attention
x = x + self.mlp(self.ln_2(x))    # residual around MLP
```

The pattern is `x = x + sublayer(norm(x))`. The input `x` is added back to the output of each sublayer. This creates a "gradient highway" — during backprop, gradients can flow directly through the `+ x` path without being attenuated by the sublayer's weights.

**Key questions to answer:**

- What happens if you remove the residual? — Deep transformers become untrainable (vanishing gradients).
- The full model applies a *final* LayerNorm after all blocks (line 182). Why? — Stabilize the representation before the output projection.

### 1.5 How Blocks Compose into the Full Model

```python
self.transformer = nn.ModuleDict(dict(
    wte = nn.Embedding(config.vocab_size, config.n_embd),
    wpe = nn.Embedding(config.block_size, config.n_embd),
    drop = nn.Dropout(config.dropout),
    h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
    ln_f = LayerNorm(config.n_embd, bias=config.bias),
))
```

The forward pass (lines 170–193) shows the full pipeline:

```
tokens (B,T) → tok_emb + pos_emb → dropout → [Block × 12] → final LayerNorm → lm_head → logits
```

Note the **weight tying** at line 138: `self.transformer.wte.weight = self.lm_head.weight` — the input embedding and output projection share the same weight matrix.

---

## Block 2: Tracing One Attention Forward Pass with Shapes

Now we dissect `CausalSelfAttention.forward` line by line, using concrete shapes from GPT-2 defaults (B=batch, T=seq_len, C=768, nh=12, hs=64).

### 2.1 Input Shape

```python
def forward(self, x):
    B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)
```

| Variable | Meaning | Shape |
|----------|---------|-------|
| `x` | Input (after LayerNorm) | `(B, T, 768)` |

### 2.2 Q/K/V Projection (Single Fused Linear)

One big linear projects from 768 to 2304 (= 3 × 768), then splits into three tensors:

```python
# weight shape: (2304, 768)
self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

# in forward:
q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
```

| Step | Operation | Shape |
|------|-----------|-------|
| Fused projection | `self.c_attn(x)` | `(B, T, 768)` → `(B, T, 2304)` |
| Split | `.split(768, dim=2)` | 3 tensors, each `(B, T, 768)` |

This is a performance optimization — one big matmul instead of three separate ones for Q, K, V.

### 2.3 Head Split (Reshape + Transpose)

```python
k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
```

Each of Q, K, V goes through:

| Step | Operation | Shape |
|------|-----------|-------|
| Start | `q` after split | `(B, T, 768)` |
| View | `.view(B, T, 12, 64)` | `(B, T, 12, 64)` — expose the 12 heads |
| Transpose | `.transpose(1, 2)` | `(B, 12, T, 64)` — heads become a batch dim |

After this, each of q, k, v has shape **`(B, 12, T, 64)`** — each head independently attends over the sequence with 64-dimensional keys/queries/values.

### 2.4 Attention Scores

Two code paths exist — Flash Attention (fast, opaque) and manual (instructive). The manual path:

```python
att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
att = F.softmax(att, dim=-1)
att = self.attn_dropout(att)
y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
```

Traced step by step:

| Step | Expression | Shape | Notes |
|------|-----------|-------|-------|
| **QKᵀ** | `q @ k.transpose(-2,-1)` | `(B,12,T,64) @ (B,12,64,T)` = `(B,12,T,T)` | Raw attention scores |
| **Scale** | `× (1/√64)` = `× 0.125` | `(B,12,T,T)` | Prevents softmax saturation |
| **Causal mask** | `.masked_fill(mask==0, -inf)` | `(B,12,T,T)` | Upper triangle → -inf (can't attend to future) |
| **Softmax** | `F.softmax(dim=-1)` | `(B,12,T,T)` | Each row sums to 1 (attention weights) |
| **Dropout** | `attn_dropout` | `(B,12,T,T)` | Regularization during training |
| **Weighted sum** | `att @ v` | `(B,12,T,T) @ (B,12,T,64)` = `(B,12,T,64)` | Output per head |

The causal mask is a lower-triangular matrix created at init:

```python
self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                    .view(1, 1, config.block_size, config.block_size))
```

This ensures token *i* can only attend to tokens 0..i (autoregressive property).

### 2.5 Recombine Heads + Output Projection

```python
y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side
y = self.resid_dropout(self.c_proj(y))
return y
```

| Step | Operation | Shape |
|------|-----------|-------|
| Start | `y` from attention | `(B, 12, T, 64)` |
| Transpose | `.transpose(1, 2)` | `(B, T, 12, 64)` |
| Concatenate | `.view(B, T, 768)` | `(B, T, 768)` — all 12 heads glued back together |
| Output projection | `c_proj(y)` | `(B, T, 768)` → `(B, T, 768)` |
| Dropout + return | | `(B, T, 768)` |

The output projection (`c_proj`) mixes information across heads. Note the special initialization (line 145): `c_proj` weights in both attention and MLP are initialized with std scaled by `1/√(2 × n_layer)`. This keeps the variance of the residual stream stable as it accumulates contributions from many layers (2 residual additions per block × 12 blocks = 24 total additions).

---

## Summary: Data Flow Through One Block

```
x: (B, T, 768)
│
├──► LayerNorm ──► CausalSelfAttention ──► + (residual)
│    (B,T,768)     (B,T,768)──►QKV split──►head split──►scores──►recombine──►proj
│                  (B,T,768) out                                              │
│◄────────────────────────────────────────────────────────────────────────────┘
│
├──► LayerNorm ──► MLP ──► + (residual)
│    (B,T,768)     768──►3072──►GELU──►768
│◄──────────────────────────────────────┘
│
x: (B, T, 768)   # same shape out as in
```

---

## Block 3: The Full GPT Model — Init, Forward, and Weight Tying

The `GPT` class (`model.py` lines 129–204) wraps everything together. Its `__init__` builds the full architecture, and its `forward` runs the end-to-end pipeline from token IDs to logits.

### 3.1 Model Architecture (`__init__`)

```python
self.transformer = nn.ModuleDict(dict(
    wte = nn.Embedding(config.vocab_size, config.n_embd),       # token embeddings
    wpe = nn.Embedding(config.block_size, config.n_embd),       # position embeddings
    drop = nn.Dropout(config.dropout),
    h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),  # 12 transformer blocks
    ln_f = LayerNorm(config.n_embd, bias=config.bias),          # final layer norm
))
self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # output projection
```

| Component | Shape / Size | Purpose |
|-----------|-------------|---------|
| `wte` | `(50304, 768)` | Maps each token ID to a 768-dim vector |
| `wpe` | `(1024, 768)` | Maps each position (0–1023) to a 768-dim vector |
| `h` | 12 × `Block` | The repeated transformer blocks |
| `ln_f` | `(768,)` weight | Final LayerNorm before output projection |
| `lm_head` | `(768, 50304)` | Projects embeddings back to vocabulary logits |

### 3.2 Weight Tying

```python
self.transformer.wte.weight = self.lm_head.weight
```

The input token embedding matrix and the output projection matrix are **the same tensor**. This is "weight tying" — it reduces parameter count (saves 768 × 50304 ≈ 38.6M parameters) and enforces the constraint that tokens with similar embeddings should also produce similar output distributions. This is why `get_num_params` does not subtract `wte` from the count: the token embeddings are "used" as the final projection, so they earn their place.

### 3.3 Weight Initialization

Two-stage initialization (lines 152–156):

```python
# Stage 1: apply standard init to all modules
self.apply(self._init_weights)

# Stage 2: special scaled init for residual projection layers
for pn, p in self.named_parameters():
    if pn.endswith('c_proj.weight'):
        torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
```

**Stage 1** (`_init_weights`): All `nn.Linear` weights get N(0, 0.02), biases get zeros. All `nn.Embedding` weights get N(0, 0.02).

**Stage 2**: The `c_proj` layers (one in attention, one in MLP) are the projections that feed *into* the residual stream. Each block adds two such contributions, so after 12 blocks there are 24 residual additions. Scaling their init by `1/√(2 × 12) ≈ 0.204` keeps the residual stream's variance from growing with depth. Without this, the signal variance would grow linearly with the number of layers, destabilizing training.

### 3.4 Forward Pass

```python
def forward(self, idx, targets=None):
    b, t = idx.size()
    pos = torch.arange(0, t, dtype=torch.long, device=device)

    tok_emb = self.transformer.wte(idx)           # (B, T) -> (B, T, C)
    pos_emb = self.transformer.wpe(pos)            # (T,)   -> (T, C), broadcasts to (B, T, C)
    x = self.transformer.drop(tok_emb + pos_emb)   # (B, T, C)
    for block in self.transformer.h:
        x = block(x)                               # (B, T, C) -> (B, T, C)
    x = self.transformer.ln_f(x)                   # (B, T, C)

    if targets is not None:
        logits = self.lm_head(x)                   # (B, T, C) -> (B, T, V)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
    else:
        logits = self.lm_head(x[:, [-1], :])       # (B, 1, C) -> (B, 1, V)
        loss = None
    return logits, loss
```

| Step | Operation | Shape |
|------|-----------|-------|
| Input | Token IDs | `(B, T)` |
| Token embed | `wte(idx)` | `(B, T, 768)` |
| Position embed | `wpe(pos)` | `(T, 768)` → broadcast `(B, T, 768)` |
| Sum + dropout | `drop(tok_emb + pos_emb)` | `(B, T, 768)` |
| 12 Blocks | `block(x)` × 12 | `(B, T, 768)` |
| Final norm | `ln_f(x)` | `(B, T, 768)` |
| Output (train) | `lm_head(x)` | `(B, T, 50304)` |
| Output (infer) | `lm_head(x[:, [-1], :])` | `(B, 1, 50304)` |

**Inference optimization**: When generating, only the last token's logits matter (we predict the *next* token). So at inference time, only the last position is projected through `lm_head`, avoiding a large unnecessary matmul over all T positions.

**Loss computation**: `cross_entropy` expects `(N, C)` input and `(N,)` targets, so logits are flattened from `(B, T, V)` to `(B*T, V)` and targets from `(B, T)` to `(B*T,)`. The `ignore_index=-1` allows masking out specific positions from the loss calculation.

**Key questions to answer:**

- Why add token and position embeddings (rather than concatenate)? — Addition keeps dimensionality at C. The model learns to encode both identity and position within the same vector space. This is the standard approach from the original Transformer.
- What is `vocab_size=50304` instead of 50257? — 50257 is GPT-2's actual vocabulary. Padding to 50304 (nearest multiple of 64) makes the `lm_head` matrix multiplication more hardware-friendly on GPUs, which operate most efficiently on dimensions that are multiples of 64.

---

## Block 4: Loading Pretrained Weights

`from_pretrained` (lines 218–272) loads official OpenAI GPT-2 weights via HuggingFace and copies them into nanoGPT's model structure.

### 4.1 The Key Challenge: Conv1D vs Linear

OpenAI's GPT-2 implementation uses `Conv1D` layers, which store weights **transposed** relative to PyTorch's `nn.Linear`:

| Layer type | Weight shape | Matmul |
|-----------|-------------|--------|
| `nn.Linear` | `(out, in)` | `x @ W.T` |
| `Conv1D` | `(in, out)` | `x @ W` |

So for four specific weight matrices, the code transposes during loading:

```python
transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
for k in sd_keys_hf:
    if any(k.endswith(w) for w in transposed):
        assert sd_hf[k].shape[::-1] == sd[k].shape
        with torch.no_grad():
            sd[k].copy_(sd_hf[k].t())
    else:
        assert sd_hf[k].shape == sd[k].shape
        with torch.no_grad():
            sd[k].copy_(sd_hf[k])
```

### 4.2 Buffer Filtering

Both nanoGPT and HuggingFace store the causal mask as a buffer (not a learnable parameter). These are filtered out before comparing keys:

```python
sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]           # nanoGPT's mask
sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')]  # HF's mask
sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')]         # HF's mask
```

### 4.3 `crop_block_size` — Model Surgery

If you load a pretrained model with block_size=1024 but want a shorter context:

```python
def crop_block_size(self, block_size):
    self.config.block_size = block_size
    self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
    for block in self.transformer.h:
        if hasattr(block.attn, 'bias'):
            block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]
```

This truncates the position embedding table and the causal mask. The position embeddings for positions 0 to `block_size-1` are preserved — those weights were already trained. You just lose the ability to attend to positions beyond the new limit.

---

## Block 5: Optimizer Configuration and MFU

### 5.1 `configure_optimizers` — Weight Decay Groups

Not all parameters should receive weight decay. The optimizer separates them into two groups based on tensor dimensionality:

```python
decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]     # weight matrices
nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]    # biases, LayerNorm params
optim_groups = [
    {'params': decay_params, 'weight_decay': weight_decay},
    {'params': nodecay_params, 'weight_decay': 0.0}
]
```

| Group | Condition | Examples | Weight decay? |
|-------|-----------|----------|---------------|
| Decay | `dim >= 2` | All `nn.Linear` weights, `nn.Embedding` weights | Yes (`0.1`) |
| No decay | `dim < 1` | Biases (1D), LayerNorm `weight`/`bias` (1D) | No |

**Why?** Weight decay penalizes large weight values, acting as L2 regularization on the weight matrices. Biases and normalization parameters are low-dimensional and don't benefit from this — decaying them can hurt training.

The code also uses **fused AdamW** when available on CUDA, which fuses the optimizer step into a single kernel for better GPU throughput:

```python
fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
use_fused = fused_available and device_type == 'cuda'
optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
```

### 5.2 `estimate_mfu` — Model FLOPs Utilization

MFU measures what fraction of theoretical GPU peak FLOPS the model actually achieves:

```python
def estimate_mfu(self, fwdbwd_per_iter, dt):
    N = self.get_num_params()
    L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
    flops_per_token = 6*N + 12*L*H*Q*T
    flops_per_fwdbwd = flops_per_token * T
    flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
    flops_achieved = flops_per_iter * (1.0/dt)
    mfu = flops_achieved / 312e12       # A100 bfloat16 peak = 312 TFLOPS
    return mfu
```

The formula (from the PaLM paper) breaks FLOPS per token into:
- **`6N`**: Forward and backward passes through all parameters (each param does ~2 FLOPs per token forward, ~4 backward → 6 total).
- **`12·L·H·Q·T`**: The attention score computation. Each of the L layers computes Q·K^T (and its backward), which is an `(H, T, Q) × (H, Q, T)` matmul. This term accounts for the quadratic cost of attention in sequence length T.

**Key questions to answer:**

- Why is MFU useful? — It tells you how efficiently you're using the hardware. A typical well-optimized training run achieves ~30-60% MFU. Much lower means something is bottlenecking (data loading, communication, memory).
- Why 312 TFLOPS? — That's the theoretical peak for bfloat16 tensor core operations on an A100 GPU.

---

## Block 6: Text Generation

The `generate` method (lines 317–341) is the autoregressive decoding loop used at inference time:

```python
@torch.no_grad()
def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
        logits, _ = self(idx_cond)
        logits = logits[:, -1, :] / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx
```

Step-by-step for each generated token:

| Step | Operation | Detail |
|------|-----------|--------|
| **Context crop** | `idx[:, -block_size:]` | If the sequence exceeds 1024 tokens, only the last 1024 are kept as context (sliding window) |
| **Forward** | `self(idx_cond)` | Runs the full model; returns logits of shape `(B, 1, V)` thanks to the inference optimization |
| **Temperature** | `logits / temperature` | Divides logits before softmax. `T<1` → sharper (more deterministic), `T>1` → flatter (more random), `T=1` → no change |
| **Top-k filter** | Keep only top k logits | All logits below the k-th largest are set to `-inf`, zeroing their probability after softmax |
| **Sample** | `multinomial(probs, 1)` | Randomly draws one token from the filtered probability distribution |
| **Append** | `cat(idx, idx_next)` | Appends the new token to the sequence; next iteration conditions on the extended sequence |

**Temperature intuition**: Softmax converts logits to probabilities via `p_i = exp(z_i/T) / Σ exp(z_j/T)`. Dividing by a small T amplifies differences between logits, making the distribution peakier. Dividing by a large T compresses differences, making it more uniform.

**Top-k intuition**: Even with temperature, rare tokens can still be sampled. Top-k hard-cuts the distribution to only the k most likely candidates, preventing the model from producing highly improbable tokens.

**Key questions to answer:**

- Why `@torch.no_grad()`? — No gradients needed during generation. This saves memory and computation by not building the computational graph.
- Why is generation slow? — Each token requires a full forward pass through 12 blocks. The T × T attention computation is repeated for every new token. KV-caching (not implemented here) is the standard optimization.

---

## Block 7: The Training Loop (`train.py`)

`train.py` (~337 lines) is a self-contained training script that handles data loading, optimization, checkpointing, evaluation, and distributed training. We trace the key subsystems.

### 7.1 Configuration

All hyperparameters are defined as module-level variables (lines 33–74), then overridden via `configurator.py` which supports both command-line args and config files:

```python
exec(open('configurator.py').read())  # overrides from command line or config file
```

Default config targets GPT-2 124M on OpenWebText:

| Hyperparameter | Value | Purpose |
|---------------|-------|---------|
| `batch_size` | 12 | Micro-batch size per GPU |
| `block_size` | 1024 | Context window length |
| `gradient_accumulation_steps` | 40 (5×8) | Simulate larger batch via accumulated gradients |
| `learning_rate` | 6e-4 | Peak learning rate |
| `max_iters` | 600,000 | Total training steps |
| `weight_decay` | 0.1 | L2 regularization on weight matrices |
| `grad_clip` | 1.0 | Max gradient norm |
| `warmup_iters` | 2,000 | Linear LR warmup steps |
| `min_lr` | 6e-5 | LR floor (10× smaller than peak) |

**Effective batch size**: `gradient_accumulation_steps × ddp_world_size × batch_size × block_size` = 40 × 1 × 12 × 1024 ≈ **491,520 tokens per iteration** (single GPU). With 8 GPUs via DDP, gradient accumulation drops to 5 per process, keeping the same effective batch.

### 7.2 Data Loading (`get_batch`)

```python
def get_batch(split):
    data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x, y
```

| Aspect | Detail |
|--------|--------|
| **Storage** | Training data is a flat binary file of uint16 token IDs. `np.memmap` reads it without loading the entire file into RAM. |
| **Sampling** | `batch_size` random starting indices are drawn. Each gives a contiguous chunk of `block_size` tokens. |
| **Input vs Target** | `x = data[i:i+1024]`, `y = data[i+1:i+1025]` — the target is shifted by one. At every position, the model predicts the next token. |
| **GPU transfer** | `pin_memory()` + `non_blocking=True` enables asynchronous CPU→GPU transfer, overlapping data movement with compute. |
| **memmap re-creation** | A new `memmap` object is created every call to avoid a known memory leak with long-lived memmaps. |

### 7.3 Learning Rate Schedule

Cosine decay with linear warmup (lines 231–242):

```python
def get_lr(it):
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)      # linear warmup
    if it > lr_decay_iters:
        return min_lr                                               # floor
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))          # cosine: 1 → 0
    return min_lr + coeff * (learning_rate - min_lr)                # lerp between max and min
```

The schedule has three phases:

```
LR
6e-4 ┤          ╭────╮
     │         ╱      ╲
     │        ╱        ╲  cosine decay
     │       ╱          ╲
     │      ╱            ╲
     │ warmup              ╲
6e-5 ┤╱                      ╰──────── floor
     └──────────────────────────────── iteration
     0    2k                  600k
```

**Why warmup?** Early in training, the model's parameters and Adam's running statistics are poorly calibrated. A large initial learning rate can cause divergent updates. Warmup ramps the LR linearly from near-zero, allowing the optimizer to build up reasonable moment estimates before applying full-strength updates.

**Why cosine decay?** Gradually reduces the LR so the model can settle into a sharper minimum. Empirically, cosine decay tends to outperform step decay for transformers.

### 7.4 The Core Training Step

The inner loop (lines 292–314) implements gradient accumulation with mixed-precision training:

```python
for micro_step in range(gradient_accumulation_steps):
    if ddp:
        model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
    with ctx:   # torch.amp.autocast for mixed precision
        logits, loss = model(X, Y)
        loss = loss / gradient_accumulation_steps
    X, Y = get_batch('train')           # prefetch next batch
    scaler.scale(loss).backward()       # backward with gradient scaling

if grad_clip != 0.0:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

scaler.step(optimizer)
scaler.update()
optimizer.zero_grad(set_to_none=True)
```

**Gradient accumulation**: Instead of one big batch, run `gradient_accumulation_steps` micro-batches, each computing `loss / gradient_accumulation_steps`. The gradients accumulate additively across micro-steps, and the optimizer steps once at the end. This simulates a large batch size without requiring the GPU memory for it all at once.

**Mixed precision (`torch.amp.autocast`)**: Forward and backward passes run in bfloat16 (or float16), which is ~2× faster and uses ~2× less memory. The `GradScaler` prevents underflow when using float16 by scaling the loss before backward (bfloat16 has sufficient dynamic range and technically doesn't need scaling, but the code handles both).

**Gradient clipping**: After unscaling, gradients are clipped to max norm 1.0. This prevents exploding gradients from destabilizing training — particularly important for transformers which can produce large gradient spikes.

**`set_to_none=True`**: After stepping, gradients are set to `None` rather than zeroed. This saves memory because PyTorch doesn't need to keep the gradient tensors allocated.

### 7.5 Evaluation and Checkpointing

```python
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
```

Every `eval_interval` (2000) steps, the model is evaluated on 200 random batches from both train and val splits. The averaged loss gives a noise-reduced estimate of model performance. The model is set to `eval()` mode (disabling dropout) during evaluation, then back to `train()`.

Checkpoints save the full training state — model weights, optimizer state, iteration number, best validation loss, and the model architecture args — so training can be resumed exactly:

```python
checkpoint = {
    'model': raw_model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'model_args': model_args,
    'iter_num': iter_num,
    'best_val_loss': best_val_loss,
    'config': config,
}
```

### 7.6 Distributed Data Parallel (DDP)

DDP (lines 82–101, 210–212) enables multi-GPU training by replicating the model on each GPU and synchronizing gradients:

```python
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    device = f'cuda:{ddp_local_rank}'
    # each GPU gets a different seed and a proportionally smaller gradient_accumulation_steps
    gradient_accumulation_steps //= ddp_world_size
```

| Concept | Detail |
|---------|--------|
| **Data parallelism** | Each GPU processes different batches (via different random seeds). Gradients are averaged across GPUs via all-reduce before the optimizer step. |
| **Scaling accumulation** | With N GPUs, each GPU does `gradient_accumulation_steps / N` micro-steps. The all-reduce averages the gradients, so the effective batch size stays the same. |
| **Sync optimization** | `model.require_backward_grad_sync` is set to `False` for all micro-steps except the last one. This avoids redundant gradient communication on intermediate accumulation steps. |
| **`master_process`** | Only rank 0 does logging, evaluation, and checkpoint saving to avoid duplicated I/O. |

### 7.7 `torch.compile`

```python
if compile:
    model = torch.compile(model)
```

PyTorch 2.0's compiler fuses operations, eliminates overhead, and generates optimized GPU kernels. This can give a significant speedup (often 20–40%) with no code changes to the model.

---

## Block 8: Sampling Script (`sample.py`)

`sample.py` is a short (~90 line) standalone script for generating text from a trained model. It supports two initialization modes:

| Mode | `init_from` | Source |
|------|------------|--------|
| Resume | `'resume'` | Load from `out_dir/ckpt.pt` (your own checkpoint) |
| Pretrained | `'gpt2'`, `'gpt2-xl'`, etc. | Load from HuggingFace via `from_pretrained` |

The tokenizer is also flexible — if a `meta.pkl` exists for the dataset (char-level models), it uses that encoder/decoder. Otherwise, it falls back to GPT-2's BPE tokenizer via `tiktoken`.

The generation loop is straightforward:

```python
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
```

Default sampling parameters: `temperature=0.8` (slightly sharper than uniform), `top_k=200` (keeps top 200 candidates), `max_new_tokens=500`.

---

## Summary: Full Data Flow (Training)

```
train.bin (on disk)
│
├─► get_batch() ─► random slices ─► x: (B, T)  y: (B, T) shifted by 1
│                                     │
│     ┌───────────────────────────────┘
│     ▼
│   wte(x) + wpe(pos) ─► dropout ─► [Block × 12] ─► ln_f ─► lm_head ─► logits: (B, T, V)
│                                                                            │
│                                                       cross_entropy(logits, y) ─► loss
│                                                                                     │
│                                    ◄─── loss.backward() ───────────────────────────┘
│                                    │
│                         gradient accumulation (repeat micro_steps times)
│                                    │
│                         grad_clip ─► optimizer.step() ─► zero_grad
│                                    │
│                         LR schedule: warmup ─► cosine decay ─► min_lr
│
└─► every 2000 steps: evaluate on train/val, save checkpoint
```

---

## Summary: Full Data Flow (Generation)

```
prompt tokens: (1, t)
│
▼
┌──────────────────────────────────────────────────────────┐
│ Loop max_new_tokens times:                               │
│   crop to last 1024 tokens if needed                     │
│   ─► forward pass ─► logits for last position: (1, V)    │
│   ─► ÷ temperature                                       │
│   ─► top-k filter                                        │
│   ─► softmax ─► probabilities                            │
│   ─► multinomial sample ─► 1 new token                   │
│   ─► append to sequence                                  │
└──────────────────────────────────────────────────────────┘
│
▼
generated sequence: (1, t + max_new_tokens)
```

---

## Suggested Exercises

1. **Count parameters**: For one `Block`, manually compute the parameter count. Check with `sum(p.numel() for p in Block(GPTConfig()).parameters())`. (Expected: ~7M per block for GPT-2 124M.)

2. **Visualize the causal mask**: Run `torch.tril(torch.ones(8,8))` and understand why setting the upper triangle to `-inf` before softmax makes those positions contribute zero weight.

3. **Trace shapes on paper**: Pick B=1, T=4, C=8, nh=2, hs=4 and work through every tensor shape in `CausalSelfAttention.forward` by hand.

4. **Trace a training step**: Follow `get_batch` → forward → loss → backward → optimizer step through the code. Identify where gradient accumulation divides the loss and where gradients are synchronized in DDP.

5. **Parameter grouping**: Run `configure_optimizers` and verify which parameters get weight decay and which don't. Check that the counts match: how many decayed vs non-decayed parameter tensors are there, and how many total parameters in each group?

6. **Temperature experiment**: Load a pretrained GPT-2 and generate text with temperature=0.1 vs 1.0 vs 2.0. Observe how the output shifts from repetitive/deterministic to creative to incoherent.

7. **Understand weight tying**: Verify that `model.transformer.wte.weight is model.lm_head.weight` returns `True`. Think about what it means for the gradient: a single backward pass updates this shared matrix from *both* the embedding loss signal and the output projection loss signal.

8. **Compute effective batch size**: For a 4-GPU DDP setup with the default config, work out `tokens_per_iter`. Verify that reducing `gradient_accumulation_steps` proportionally keeps it constant.
