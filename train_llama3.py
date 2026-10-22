"""
Reference code for LLaMA-3.1 training and inference.
Will save the model weights into files, to be read from C as initialization.

This code differs from GPT-2 very slightly, there are three main differences:
1) RoPE: LLaMA uses a different positional encoding scheme called Relative Positional Encoding (RoPE).
2) GQA: Grouped Query Attention (GQA) is used to reduce the number of attention heads.
3) SwiGLU: Swish-Gated Linear Unit (SwiGLU) is used as the activation function in the MLP.

References:
# 1) https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/api/tokenizer.py
# 2) https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/api/model.py
# 3) https://github.com/meta-llama/llama3/blob/11817d47e1ba7a4959b025eb1ca308572e0e3963/llama/generation.py

Example launches to only benchmark the speed of bfloat16 compiled GPU training:
TODO: add the actual commands
"""

ಆಮದು argparse
ಆಮದು os
ಆಮದು math
ಆಮದು glob
ಆಮದು inspect
ತಃ contextlib ಆಮದು nullcontext
ತಃ dataclasses ಆಮದು dataclass
ತಃ pathlib ಆಮದು ಮಾರ್ಗ_2
ಆಮದು time
ತಃ typing ಆಮದು (
    Abstract_ಗಣ,
    Collection,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

ಆಮದು numpy ಇವ np
ಆಮದು torch
ಆಮದು torch.nn ಇವ nn
ತಃ torch.nn ಆಮದು functional ಇವ F
ಆಮದು torch._inductor.config ಇವ config
ತಃ torch.nn.parallel ಆಮದು Distributed_ದತ್ತಾಂಶ_Parallel ಇವ DDP
ತಃ torch.distributed ಆಮದು init_process_ಗುಂಪು, destroy_process_ಗುಂಪು
ತಃ torch.distributed.optim ಆಮದು ZeroRedundancyOptimizer
ಆಮದು torch.distributed ಇವ dist

ಆಮದು tiktoken
ತಃ tiktoken.ತುಂಬು ಆಮದು ತುಂಬು_tiktoken_bpe

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the LLaMA 3.x model

# Used in Grouped Query Attention (GQA), broadcasts the key and value tensors
ವ್ಯಾಖ್ಯಾನ repeat_kv(x: torch.Tensor, n_rep: ಪೂರ್ಣಾಂಕ) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, ತಲೆ_dim = x.shape
    ಆದರೆ n_rep == 1:
        ಹಿಂತಿರುಗಿ x
    ಹಿಂತಿರುಗಿ (
        x[:, :, :, ಏನೂಇಲ್ಲ, :]
        .expand(bs, slen, n_kv_heads, n_rep, ತಲೆ_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, ತಲೆ_dim)
    )

# -----------------------------------------------------------------------------
# RoPE related

ವ್ಯಾಖ್ಯಾನ reshape_for_ಪ್ರಸಾರ(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    ಪ್ರತಿಜ್ಞಾ 0 <= 1 < ndim
    ಪ್ರತಿಜ್ಞಾ freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d ಆದರೆ i == 1 ಅಥವಾ i == ndim - 1 ಇಲ್ಲದಿದ್ದರೆ 1 ಗಾಗಿ i, d ಒಳಗೆ enumerate(x.shape)]
    ಹಿಂತಿರುಗಿ freqs_cis.view(*shape)

ವ್ಯಾಖ್ಯಾನ apply_scaling(freqs: torch.Tensor):
    # Values obtained from grid search
    scale_ಗುಣಕ = 8
    ಕೆಳಗಿನ_freq_ಗುಣಕ = 1
    ಮೇಲಿನ_freq_ಗುಣಕ = 4
    old_context_len = 8192  # original llama3 length

    ಕೆಳಗಿನ_freq_wavelen = old_context_len / ಕೆಳಗಿನ_freq_ಗುಣಕ
    ಮೇಲಿನ_freq_wavelen = old_context_len / ಮೇಲಿನ_freq_ಗುಣಕ
    new_freqs = []
    ಗಾಗಿ freq ಒಳಗೆ freqs:
        wavelen = 2 * math.pi / freq
        ಆದರೆ wavelen < ಮೇಲಿನ_freq_wavelen:
            new_freqs.append(freq)
        ಅಥವಾಆದರೆ wavelen > ಕೆಳಗಿನ_freq_wavelen:
            new_freqs.append(freq / scale_ಗುಣಕ)
        ಇಲ್ಲದಿದ್ದರೆ:
            ಪ್ರತಿಜ್ಞಾ ಕೆಳಗಿನ_freq_wavelen != ಮೇಲಿನ_freq_wavelen
            smooth = (old_context_len / wavelen - ಕೆಳಗಿನ_freq_ಗುಣಕ) / (
                ಮೇಲಿನ_freq_ಗುಣಕ - ಕೆಳಗಿನ_freq_ಗುಣಕ
            )
            new_freqs.append((1 - smooth) * freq / scale_ಗುಣಕ + smooth * freq)
    ಹಿಂತಿರುಗಿ torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)

ವ್ಯಾಖ್ಯಾನ apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.ದಶಮಾಂಶ().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.ದಶಮಾಂಶ().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_ಪ್ರಸಾರ(freqs_cis, xq_)
    xq_ಹೊರಹೋಗು = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_ಹೊರಹೋಗು = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    ಹಿಂತಿರುಗಿ xq_ಹೊರಹೋಗು.type_as(xq), xk_ಹೊರಹೋಗು.type_as(xk)

ವ್ಯಾಖ್ಯಾನ precompute_freqs_cis(
    dim: ಪೂರ್ಣಾಂಕ, ಕೊನೆ: ಪೂರ್ಣಾಂಕ, theta: ದಶಮಾಂಶ = 10000.0, use_scaled: bool = ಸುಳ್ಳು
):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].ದಶಮಾಂಶ() / dim))
    t = torch.arange(ಕೊನೆ, device=freqs.device, dtype=torch.float32)
    ಆದರೆ use_scaled:
        freqs = apply_scaling(freqs)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    ಹಿಂತಿರುಗಿ freqs_cis

# -----------------------------------------------------------------------------
# LLaMA building blocks

# LLaMA reference code explicitly implemented RMSNorm so we copy pasted it
# (https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/api/model.py)
# we could also use nn.RMSNorm, it has slightly different numeric properties, but equivalent
ವರ್ಗ RMSNorm(torch.nn.Module):
    ವ್ಯಾಖ್ಯಾನ __init__(self, dim: ಪೂರ್ಣಾಂಕ, eps: ದಶಮಾಂಶ = 1e-6):
        ಪರ().__init__()
        self.eps = eps
        self.ತೂಕ = nn.Parameter(torch.ones(dim))

    ವ್ಯಾಖ್ಯಾನ _norm(self, x):
        ಹಿಂತಿರುಗಿ x * torch.rsqrt(x.pow(2).ಸರಾಸರಿ(-1, keepdim=ನಿಜ) + self.eps)

    ವ್ಯಾಖ್ಯಾನ forward(self, x):
        output = self._norm(x.ದಶಮಾಂಶ()).type_as(x)
        ಹಿಂತಿರುಗಿ output * self.ತೂಕ

ವರ್ಗ CausalSelfAttention(nn.Module):

    ವ್ಯಾಖ್ಯಾನ __init__(self, config):
        ಪರ().__init__()
        ಪ್ರತಿಜ್ಞಾ config.n_embd % config.n_ತಲೆ == 0

        self.n_ತಲೆ = config.n_ತಲೆ
        self.n_kv_ತಲೆ = config.n_kv_ತಲೆ
        self.n_rep = self.n_ತಲೆ // self.n_kv_ತಲೆ
        self.hd = config.n_embd // config.n_ತಲೆ
        self.use_kv = config.use_kv
        self.flash = config.flash

        self.c_attn = nn.ರೇಖೀಯ(config.n_embd, (config.n_ತಲೆ + 2 * config.n_kv_ತಲೆ) * self.hd, bias=ಸುಳ್ಳು)  # key, query, value projections
        self.c_proj = nn.ರೇಖೀಯ(config.n_embd, config.n_embd, bias=ಸುಳ್ಳು)  # output projection

        # static KV cache - we could alternatively allocate it outside of the model and just pass it in when needed
        ಆದರೆ self.use_kv:
            self.cache_k = torch.zeros((config.max_gen_batch_ಗಾತ್ರ, config.ಖಂಡ_ಗಾತ್ರ, config.n_kv_ತಲೆ, self.hd))
            self.cache_v = torch.zeros((config.max_gen_batch_ಗಾತ್ರ, config.ಖಂಡ_ಗಾತ್ರ, config.n_kv_ತಲೆ, self.hd))

    ವ್ಯಾಖ್ಯಾನ forward(self, x, freqs_cis=ಏನೂಇಲ್ಲ, ಆರಂಭ_pos=ಏನೂಇಲ್ಲ, ಮುಖವಾಡ=ಏನೂಇಲ್ಲ):
        B, T, C = x.ಗಾತ್ರ_2() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.ವಿಭಜಿಸು([self.n_ತಲೆ * self.hd, self.n_kv_ತಲೆ * self.hd, self.n_kv_ತಲೆ * self.hd], dim=-1)
        q, k, v = map(ಅನಾಮಿಕ t: t.view(B, T, -1, self.hd), (q, k, v))  # (B, T, NH, HD)

        q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)  # rotate QK (rope)  <-- 1. difference compared to GPT-2

        ಆದರೆ self.use_kv ಮತ್ತು ಅಲ್ಲ self.training ಮತ್ತು ಆರಂಭ_pos >= 0:  # use kv-caching during inference
            self.cache_k[:B, ಆರಂಭ_pos : ಆರಂಭ_pos + T] = k
            self.cache_v[:B, ಆರಂಭ_pos : ಆರಂಭ_pos + T] = v
            k = self.cache_k[:B, : ಆರಂಭ_pos + T]
            v = self.cache_v[:B, : ಆರಂಭ_pos + T]

        k = repeat_kv(k, self.n_rep)  # GQA <-- 2. difference compared to GPT-2
        v = repeat_kv(v, self.n_rep)

        q, k, v = map(ಅನಾಮಿಕ t: t.transpose(1, 2), (q, k, v))  # (B, NH, T, HD)

        ಆದರೆ self.flash:
            # flashattention
            # if T == 1 no need to mask, otherwise the function complains
            # scaled_dot_product_attention expects a mask where value of True indicates that the element should take part in attention
            # our mask is the opposite, so we need to invert it
            y = F.scaled_dot_ಗುಣಲಬ್ಧ_attention(q, k, v, ಮುಖವಾಡ == 0 ಆದರೆ T > 1 ಇಲ್ಲದಿದ್ದರೆ ಏನೂಇಲ್ಲ)
        ಇಲ್ಲದಿದ್ದರೆ:
            # manual implementation of attention
            # this materializes the large (T,T) matrix for all the queries and keys
            scores = (q @ k.transpose(-2, -1)) * (1.0 / math.ವರ್ಗಮೂಲ(self.hd))
            ಆದರೆ ಮುಖವಾಡ ಆಗಿರುವ ಅಲ್ಲ ಏನೂಇಲ್ಲ:
                scores.masked_fill_(ಮುಖವಾಡ, torch.finfo(scores.dtype).min)
            att = F.softmax(scores.ದಶಮಾಂಶ(), dim=-1).type_as(q)
            y = att @ v # (B, NH, T, T) x (B, NH, T, HD) -> (B, NH, T, HD)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        ಹಿಂತಿರುಗಿ y

ವರ್ಗ MLP(nn.Module):

    ವ್ಯಾಖ್ಯಾನ __init__(self, config):
        ಪರ().__init__()
        hidden_dim = 4 * config.n_embd
        hidden_dim = ಪೂರ್ಣಾಂಕ(2 * hidden_dim / 3)
        # custom dim factor multiplier
        ಆದರೆ config.ffn_dim_multiplier ಆಗಿರುವ ಅಲ್ಲ ಏನೂಇಲ್ಲ:
            hidden_dim = ಪೂರ್ಣಾಂಕ(config.ffn_dim_multiplier * hidden_dim)
        hidden_dim = config.ಅಪವರ್ತ್ಯ_ದ * ((hidden_dim + config.ಅಪವರ್ತ್ಯ_ದ - 1) // config.ಅಪವರ್ತ್ಯ_ದ)
        self.c_fc = nn.ರೇಖೀಯ(config.n_embd, hidden_dim, bias=ಸುಳ್ಳು)
        self.c_fc2 = nn.ರೇಖೀಯ(config.n_embd, hidden_dim, bias=ಸುಳ್ಳು)
        self.c_proj = nn.ರೇಖೀಯ(hidden_dim, config.n_embd, bias=ಸುಳ್ಳು)

    ವ್ಯಾಖ್ಯಾನ forward(self, x):
        # SwiGLU self.c_proj(F.silu(self.c_fc2(x)) * self.c_fc(x))  <-- 3. difference compared to GPT-2
        x1 = self.c_fc(x)
        x2 = self.c_fc2(x)
        x2 = F.silu(x2)
        x = x1 * x2
        x = self.c_proj(x)
        ಹಿಂತಿರುಗಿ x

ವರ್ಗ ಖಂಡ(nn.Module):

    ವ್ಯಾಖ್ಯಾನ __init__(self, config):
        ಪರ().__init__()
        self.ln_1 = RMSNorm(config.n_embd, config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd, config.norm_eps)
        self.mlp = MLP(config)

    ವ್ಯಾಖ್ಯಾನ forward(self, x, freqs_cis=ಏನೂಇಲ್ಲ, ಆರಂಭ_pos=ಏನೂಇಲ್ಲ, ಮುಖವಾಡ=ಏನೂಇಲ್ಲ):
        x = x + self.attn(self.ln_1(x), freqs_cis, ಆರಂಭ_pos, ಮುಖವಾಡ)
        x = x + self.mlp(self.ln_2(x))
        ಹಿಂತಿರುಗಿ x

# -----------------------------------------------------------------------------
# The main LLaMA 3.1 model

@dataclass
ವರ್ಗ LlamaConfig:
    ಆವೃತ್ತಿ: str = "3.1"
    ಖಂಡ_ಗಾತ್ರ: ಪೂರ್ಣಾಂಕ = 8192
    vocab_ಗಾತ್ರ: ಪೂರ್ಣಾಂಕ = 128256
    n_layer: ಪೂರ್ಣಾಂಕ = 32
    n_ತಲೆ: ಪೂರ್ಣಾಂಕ = 32
    n_kv_ತಲೆ: ಪೂರ್ಣಾಂಕ = 8
    n_embd: ಪೂರ್ಣಾಂಕ = 4096
    ffn_dim_multiplier: ದಶಮಾಂಶ = 1.3
    ಅಪವರ್ತ್ಯ_ದ: ಪೂರ್ಣಾಂಕ = 1024
    norm_eps: ದಶಮಾಂಶ = 1e-5
    rope_theta: ದಶಮಾಂಶ = 500000.0
    use_scaled_rope: bool = ನಿಜ
    max_gen_batch_ಗಾತ್ರ: ಪೂರ್ಣಾಂಕ = 4
    use_kv: bool = ನಿಜ
    flash: bool = ಸುಳ್ಳು  # use flashattention?

    ವ್ಯಾಖ್ಯಾನ __init__(self, **kwargs):
        ಗಾಗಿ k, v ಒಳಗೆ kwargs.ವಸ್ತುಗಳು():
            ಆದರೆ hasattr(self, k):
                setattr(self, k, v)
        ಪ್ರತಿಜ್ಞಾ self.n_kv_ತಲೆ <= self.n_ತಲೆ
        ಪ್ರತಿಜ್ಞಾ self.n_ತಲೆ % self.n_kv_ತಲೆ == 0
        ಪ್ರತಿಜ್ಞಾ self.n_embd % self.n_ತಲೆ == 0

ವರ್ಗ LLaMA(nn.Module):

    ವ್ಯಾಖ್ಯಾನ __init__(self, config):
        ಪರ().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_ಗಾತ್ರ, config.n_embd),
            h = nn.ModuleList([ಖಂಡ(config) ಗಾಗಿ _ ಒಳಗೆ ಶ್ರೇಣಿ(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd, config.norm_eps),
        ))
        self.lm_ತಲೆ = nn.ರೇಖೀಯ(config.n_embd, config.vocab_ಗಾತ್ರ, bias=ಸುಳ್ಳು)

        # init all weights, use a torch rng object to be very careful
        self.init_rng = torch.Generator()
        self.init_rng.manual_ಬೀಜ(42)

        self.freqs_cis = precompute_freqs_cis(
            config.n_embd // config.n_ತಲೆ,
            config.ಖಂಡ_ಗಾತ್ರ * 2,
            config.rope_theta,
            config.use_scaled_rope,
        )

    ವ್ಯಾಖ್ಯಾನ forward(self, idx, targets=ಏನೂಇಲ್ಲ, return_logits=ನಿಜ, ಆರಂಭ_pos=0):
        _, t = idx.ಗಾತ್ರ_2()
        ಪ್ರತಿಜ್ಞಾ t <= self.config.ಖಂಡ_ಗಾತ್ರ, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # forward the LLaMA model itself
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        freqs_cis = self.freqs_cis[ಆರಂಭ_pos:ಆರಂಭ_pos+t]

        ಮುಖವಾಡ = torch.triu(torch.ones((t, t), device=ಮುಂದಿನ_2(self.parameters()).device, dtype=torch.bool), diagonal=1)

        ಗಾಗಿ i, ಖಂಡ_2 ಒಳಗೆ enumerate(self.transformer.h):
            x = ಖಂಡ_2(x, freqs_cis, ಆರಂಭ_pos, ಮುಖವಾಡ)
        x = self.transformer.ln_f(x)

        ಆದರೆ targets ಆಗಿರುವ ಅಲ್ಲ ಏನೂಇಲ್ಲ:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_ತಲೆ(x).ದಶಮಾಂಶ()
            loss = F.cross_entropy(logits.view(-1, logits.ಗಾತ್ರ_2(-1)), targets.view(-1), ignore_ಸೂಚ್ಯಂಕ=-1)
        ಇಲ್ಲದಿದ್ದರೆ:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_ತಲೆ(x[:, [-1], :]).ದಶಮಾಂಶ() # note: using list [-1] to preserve the time dim
            loss = ಏನೂಇಲ್ಲ

        # there are performance reasons why not returning logits is prudent, if not needed
        ಆದರೆ ಅಲ್ಲ return_logits:
            logits = ಏನೂಇಲ್ಲ

        ಹಿಂತಿರುಗಿ logits, loss

    @staticmethod
    ವ್ಯಾಖ್ಯಾನ adapt_llama_state_dict_ಕೀಲಿಗಳು(checkpoint, config: LlamaConfig):
        # Modify key names from Meta's LLaMA to our LLaMA
        # our key names are derived from GPT-2's key names
        checkpoint['transformer.wte.weight'] = checkpoint.ತೆಗೆ_2('tok_embeddings.weight')

        ಗಾಗಿ i ಒಳಗೆ ಶ್ರೇಣಿ(config.n_layer):
            ಗಾಗಿ name ಒಳಗೆ ['attention_norm', 'ffn_norm']:
                old_ಕೀಲಿ = f'layers.{i}.{name}.weight'  # e.g. layers.x.attention_norm.weight -> transformer.h.x.ln_1.weight
                new_ಕೀಲಿ = f'transformer.h.{i}.ln_{1 if name == "attention_norm" else 2}.weight'
                checkpoint[new_ಕೀಲಿ] = checkpoint.ತೆಗೆ_2(old_ಕೀಲಿ)

        ಗಾಗಿ i ಒಳಗೆ ಶ್ರೇಣಿ(config.n_layer):
            ಗಾಗಿ name ಒಳಗೆ ['attention.wq', 'attention.wk', 'attention.wv']:
                old_ಕೀಲಿ = f'layers.{i}.{name}.weight'
                new_ಕೀಲಿ = f'transformer.h.{i}.attn.c_attn.weight'
                ಆದರೆ name == 'attention.wq':
                    checkpoint[new_ಕೀಲಿ] = checkpoint.ತೆಗೆ_2(old_ಕೀಲಿ)
                ಇಲ್ಲದಿದ್ದರೆ:  # merge 3 weights into transformer.h.x.attn.c_attn.weight
                    checkpoint[new_ಕೀಲಿ] = torch.cat((checkpoint[new_ಕೀಲಿ], checkpoint.ತೆಗೆ_2(old_ಕೀಲಿ)), dim=0)
            old_ಕೀಲಿ = f'layers.{i}.attention.wo.weight'
            new_ಕೀಲಿ = f'transformer.h.{i}.attn.c_proj.weight'
            checkpoint[new_ಕೀಲಿ] = checkpoint.ತೆಗೆ_2(old_ಕೀಲಿ)

        ffn_map = {'w1': 'c_fc2', 'w2': 'c_proj', 'w3': 'c_fc'}
        ಗಾಗಿ i ಒಳಗೆ ಶ್ರೇಣಿ(config.n_layer):
            ಗಾಗಿ name ಒಳಗೆ ['feed_forward.w1', 'feed_forward.w2', 'feed_forward.w3']:
                old_ಕೀಲಿ = f'layers.{i}.{name}.weight'
                new_ಕೀಲಿ = f'transformer.h.{i}.mlp.{ffn_map[name.split(".")[-1]]}.weight'
                checkpoint[new_ಕೀಲಿ] = checkpoint.ತೆಗೆ_2(old_ಕೀಲಿ)

        checkpoint['transformer.ln_f.weight'] = checkpoint.ತೆಗೆ_2('norm.weight')
        checkpoint['lm_head.weight'] = checkpoint.ತೆಗೆ_2('output.weight')

        ಹಿಂತಿರುಗಿ checkpoint

    @staticmethod
    ವ್ಯಾಖ್ಯಾನ adapt_llama_state_dict_ಕೀಲಿಗಳು_hf(checkpoint, config: LlamaConfig):
        # Modify key names from HuggingFace's LLaMA to our LLaMA
        # our key names are derived from GPT-2's key names
        checkpoint['transformer.wte.weight'] = checkpoint.ತೆಗೆ_2('model.embed_tokens.weight')

        # We need to unpermute K and V because HF script permuted the original Meta-LLaMA weights
        # see: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py
        ವ್ಯಾಖ್ಯಾನ unpermute(w, n_heads, dim1, dim2):
            ಹಿಂತಿರುಗಿ w.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)

        ಗಾಗಿ i ಒಳಗೆ ಶ್ರೇಣಿ(config.n_layer):
            ಗಾಗಿ name ಒಳಗೆ ['input_layernorm', 'post_attention_layernorm']:
                old_ಕೀಲಿ = f'model.layers.{i}.{name}.weight'  # e.g. layers.x.attention_norm.weight -> transformer.h.x.ln_1.weight
                new_ಕೀಲಿ = f'transformer.h.{i}.ln_{1 if name == "input_layernorm" else 2}.weight'
                checkpoint[new_ಕೀಲಿ] = checkpoint.ತೆಗೆ_2(old_ಕೀಲಿ)

        ಗಾಗಿ i ಒಳಗೆ ಶ್ರೇಣಿ(config.n_layer):
            ಗಾಗಿ name ಒಳಗೆ ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj']:
                old_ಕೀಲಿ = f'model.layers.{i}.{name}.weight'
                new_ಕೀಲಿ = f'transformer.h.{i}.attn.c_attn.weight'
                ಆದರೆ name == 'self_attn.q_proj':
                    checkpoint[new_ಕೀಲಿ] = unpermute(checkpoint.ತೆಗೆ_2(old_ಕೀಲಿ), config.n_ತಲೆ, config.n_embd, config.n_embd)
                ಇಲ್ಲದಿದ್ದರೆ:  # merge 3 weights into transformer.h.x.attn.c_attn.weight
                    tensor = checkpoint.ತೆಗೆ_2(old_ಕೀಲಿ)
                    ಆದರೆ name == 'self_attn.k_proj':
                        tensor = unpermute(tensor, config.n_kv_ತಲೆ, config.n_kv_ತಲೆ * (config.n_embd // config.n_ತಲೆ), config.n_embd)
                    checkpoint[new_ಕೀಲಿ] = torch.cat((checkpoint[new_ಕೀಲಿ], tensor), dim=0)
            old_ಕೀಲಿ = f'model.layers.{i}.self_attn.o_proj.weight'
            new_ಕೀಲಿ = f'transformer.h.{i}.attn.c_proj.weight'
            checkpoint[new_ಕೀಲಿ] = checkpoint.ತೆಗೆ_2(old_ಕೀಲಿ)

        ffn_map = {'gate_proj': 'c_fc2', 'down_proj': 'c_proj', 'up_proj': 'c_fc'}
        ಗಾಗಿ i ಒಳಗೆ ಶ್ರೇಣಿ(config.n_layer):
            ಗಾಗಿ name ಒಳಗೆ ['gate_proj', 'down_proj', 'up_proj']:
                old_ಕೀಲಿ = f'model.layers.{i}.mlp.{name}.weight'
                new_ಕೀಲಿ = f'transformer.h.{i}.mlp.{ffn_map[name]}.weight'
                checkpoint[new_ಕೀಲಿ] = checkpoint.ತೆಗೆ_2(old_ಕೀಲಿ)

        checkpoint['transformer.ln_f.weight'] = checkpoint.ತೆಗೆ_2('model.norm.weight')

        ಹಿಂತಿರುಗಿ checkpoint

    @classmethod
    ವ್ಯಾಖ್ಯಾನ from_pretrained_llama3_hf(cls, model_id):
        """Loads pretrained LLaMA model weights from HuggingFace"""
        ತಃ transformers ಆಮದು AutoModelForCausalLM, AutoTokenizer
        ಪ್ರತಿಜ್ಞಾ model_id == "meta-llama/Meta-Llama-3.1-8B", "Only the 8B-base model is supported for now"
        model_ವಾದಗಳು = LlamaConfig()

        model = AutoModelForCausalLM.from_pretrained(model_id)
        checkpoint = LLaMA.adapt_llama_state_dict_ಕೀಲಿಗಳು_hf(model.state_dict(), model_ವಾದಗಳು)

        original_default_type = torch.get_default_dtype()  # save the default type
        torch.ಗಣ_default_tensor_type(torch.cuda.BFloat16Tensor)  # much faster loading
        model = LLaMA(model_ವಾದಗಳು)
        model.ತುಂಬು_state_dict(checkpoint, strict=ಸುಳ್ಳು)
        torch.ಗಣ_default_tensor_type(torch.tensor([], dtype=original_default_type, device="cpu").type())  # restore default type

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_id = 128004  # this is the pad token id for LLaMA 3.1 base, we need to set this explicitly as our generate func expects it
        tokenizer.stop_ಟೋಕನುಗಳು = [tokenizer.eos_token_id]
        model.tokenizer = tokenizer
        ಹಿಂತಿರುಗಿ model

    @classmethod
    ವ್ಯಾಖ್ಯಾನ from_pretrained_llama3_meta(cls, ckpt_dir, tokenizer_ಮಾರ್ಗ):
        """Loads pretrained LLaMA model weights from a checkpoint directory"""
        model_ವಾದಗಳು = LlamaConfig()

        ckpt_ಮಾರ್ಗ = sorted(ಮಾರ್ಗ_2(ckpt_dir).glob("*.pth"))[0]
        checkpoint = torch.ತುಂಬು(ckpt_ಮಾರ್ಗ, map_location="cpu", weights_only=ನಿಜ)
        checkpoint = LLaMA.adapt_llama_state_dict_ಕೀಲಿಗಳು(checkpoint, model_ವಾದಗಳು)

        original_default_type = torch.get_default_dtype()  # save the default type
        torch.ಗಣ_default_tensor_type(torch.cuda.BFloat16Tensor)  # much faster loading
        model = LLaMA(model_ವಾದಗಳು)
        model.ತುಂಬು_state_dict(checkpoint, strict=ಸುಳ್ಳು)
        torch.ಗಣ_default_tensor_type(torch.tensor([], dtype=original_default_type, device="cpu").type())  # restore default type

        tokenizer = Tokenizer(model_ಮಾರ್ಗ=tokenizer_ಮಾರ್ಗ)
        model.tokenizer = tokenizer
        ಹಿಂತಿರುಗಿ model

    ವ್ಯಾಖ್ಯಾನ configure_optimizers(self, ತೂಕ_decay, learning_ದರ, betas, device_type, zero_stage):
        # start with all of the candidate parameters
        param_dict = {pn: p ಗಾಗಿ pn, p ಒಳಗೆ self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p ಗಾಗಿ pn, p ಒಳಗೆ param_dict.ವಸ್ತುಗಳು() ಆದರೆ p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p ಗಾಗಿ n, p ಒಳಗೆ param_dict.ವಸ್ತುಗಳು() ಆದರೆ p.dim() >= 2]
        nodecay_params = [p ಗಾಗಿ n, p ಒಳಗೆ param_dict.ವಸ್ತುಗಳು() ಆದರೆ p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': ತೂಕ_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        ಅಂಕೆ_decay_params = ಮೊತ್ತ(p.numel() ಗಾಗಿ p ಒಳಗೆ decay_params)
        ಅಂಕೆ_nodecay_params = ಮೊತ್ತ(p.numel() ಗಾಗಿ p ಒಳಗೆ nodecay_params)
        print0(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print0(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_ಲಭ್ಯ = 'fused' ಒಳಗೆ inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_ಲಭ್ಯ ಮತ್ತು device_type == 'cuda'
        print0(f"using fused AdamW: {use_fused}")
        ಆದರೆ zero_stage == 1:
            print0("using ZeroRedundancyOptimizer")
            optimizer = ZeroRedundancyOptimizer(**optim_groups[0], optimizer_class=torch.optim.AdamW,
                                                lr=learning_ದರ, betas=betas, fused=use_fused)
            optimizer.ಕೂಡಿಸು_param_ಗುಂಪು(optim_groups[1])
        ಇಲ್ಲದಿದ್ದರೆ:
            print0("using regular AdamW")
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_ದರ, betas=betas, fused=use_fused)
        ಹಿಂತಿರುಗಿ optimizer

    @torch.inference_ಬಹುಲಕ()
    ವ್ಯಾಖ್ಯಾನ generate(
        self,
        prompt_ಟೋಕನುಗಳು: List[List[ಪೂರ್ಣಾಂಕ]],
        max_gen_len: ಪೂರ್ಣಾಂಕ,
        temperature: ದಶಮಾಂಶ = 0.6,
        ತುದಿ_p: ದಶಮಾಂಶ = 0.9,
        echo: bool = ಸುಳ್ಳು,
    ) -> Tuple[List[List[ಪೂರ್ಣಾಂಕ]], Optional[List[List[ದಶಮಾಂಶ]]]]:
        """
        Generate text sequences based on provided prompts using the language generation model.

        Args:
            prompt_tokens (List[List[int]]): List of tokenized prompts, where each prompt is represented as a list of integers.
            max_gen_len (int): Maximum length of the generated text sequence.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            Tuple[List[List[int]], Optional[List[List[float]]]]: A tuple containing generated token sequences.

        Note:
            This method uses the provided prompts as a basis for generating text. It employs nucleus sampling to produce text with controlled randomness.

        """
        bsz = ಉದ್ದ(prompt_ಟೋಕನುಗಳು)
        ಪ್ರತಿಜ್ಞಾ bsz <= self.config.max_gen_batch_ಗಾತ್ರ, f"Batch size {bsz} exceeds the maximum generation batch size {self.config.max_gen_batch_size}"
        device = ಮುಂದಿನ_2(self.parameters()).device

        min_prompt_len = min(ಉದ್ದ(t) ಗಾಗಿ t ಒಳಗೆ prompt_ಟೋಕನುಗಳು)
        max_prompt_len = max(ಉದ್ದ(t) ಗಾಗಿ t ಒಳಗೆ prompt_ಟೋಕನುಗಳು)
        ಪ್ರತಿಜ್ಞಾ max_prompt_len <= self.config.ಖಂಡ_ಗಾತ್ರ, f"Prompt length {max_prompt_len} exceeds the maximum block size {self.config.block_size}"
        ಒಟ್ಟು_len = min(self.config.ಖಂಡ_ಗಾತ್ರ, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_id
        ಟೋಕನುಗಳು = torch.ತುಂಬಿದ((bsz, ಒಟ್ಟು_len), pad_id, dtype=torch.ದೀರ್ಘ, device=device)
        ಗಾಗಿ idx, t ಒಳಗೆ enumerate(prompt_ಟೋಕನುಗಳು):
            ಟೋಕನುಗಳು[idx, : ಉದ್ದ(t)] = torch.tensor(t, dtype=torch.ದೀರ್ಘ, device=device)

        prev_pos = 0
        eos_reached = torch.tensor([ಸುಳ್ಳು] * bsz, device=device)
        ಒಳಹರಿವು_ಪಠ್ಯ_ಮುಖವಾಡ = ಟೋಕನುಗಳು != pad_id

        ಆದರೆ min_prompt_len == ಒಟ್ಟು_len:
            logits, _ = self.forward(ಟೋಕನುಗಳು, ಆರಂಭ_pos=prev_pos)

        stop_ಟೋಕನುಗಳು = torch.tensor(list(self.tokenizer.stop_ಟೋಕನುಗಳು)).ಗಮ್ಯ(device)

        ಗಾಗಿ cur_pos ಒಳಗೆ ಶ್ರೇಣಿ(min_prompt_len, ಒಟ್ಟು_len):
            logits, _ = self.forward(ಟೋಕನುಗಳು[:, prev_pos:cur_pos], ಆರಂಭ_pos=prev_pos)
            ಆದರೆ temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                ಮುಂದಿನ_token = sample_ತುದಿ_p(probs, ತುದಿ_p)
            ಇಲ್ಲದಿದ್ದರೆ:
                ಮುಂದಿನ_token = torch.argmax(logits[:, -1], dim=-1)

            ಮುಂದಿನ_token = ಮುಂದಿನ_token.reshape(-1)
            # only replace token if prompt has already been generated
            ಮುಂದಿನ_token = torch.where(ಒಳಹರಿವು_ಪಠ್ಯ_ಮುಖವಾಡ[:, cur_pos], ಟೋಕನುಗಳು[:, cur_pos], ಮುಂದಿನ_token)
            ಟೋಕನುಗಳು[:, cur_pos] = ಮುಂದಿನ_token
            eos_reached |= ~ಒಳಹರಿವು_ಪಠ್ಯ_ಮುಖವಾಡ[:, cur_pos] & torch.isin(ಮುಂದಿನ_token, stop_ಟೋಕನುಗಳು)
            prev_pos = cur_pos
            ಆದರೆ all(eos_reached):
                ನಿಲ್ಲಿಸು

        ಹೊರಹೋಗು_ಟೋಕನುಗಳು = []
        ಗಾಗಿ i, toks ಒಳಗೆ enumerate(ಟೋಕನುಗಳು.tolist()):
            # cut to max gen len
            ಆರಂಭ = 0 ಆದರೆ echo ಇಲ್ಲದಿದ್ದರೆ ಉದ್ದ(prompt_ಟೋಕನುಗಳು[i])
            toks = toks[ಆರಂಭ : ಉದ್ದ(prompt_ಟೋಕನುಗಳು[i]) + max_gen_len]
            # cut to after eos tok if any
            ಗಾಗಿ stop_token ಒಳಗೆ self.tokenizer.stop_ಟೋಕನುಗಳು:
                ಪ್ರಯತ್ನಿಸು:
                    eos_idx = toks.ಸೂಚ್ಯಂಕ(stop_token)
                    toks = toks[:eos_idx]
                ಅಪವಾದ ಮೌಲ್ಯ_Error:
                    ದಾಟು
            ಹೊರಹೋಗು_ಟೋಕನುಗಳು.append(toks)
        ಹಿಂತಿರುಗಿ ಹೊರಹೋಗು_ಟೋಕನುಗಳು

# -----------------------------------------------------------------------------
# sampling utils

ವ್ಯಾಖ್ಯಾನ sample_ತುದಿ_p(probs, p):
    """
    Perform top-p (nucleus) sampling on a probability distribution.

    Args:
        probs (torch.Tensor): Probability distribution tensor.
        p (float): Probability threshold for top-p sampling.

    Returns:
        torch.Tensor: Sampled token indices.

    Note:
        Top-p sampling selects the smallest set of tokens whose cumulative probability mass
        exceeds the threshold p. The distribution is renormalized based on the selected tokens.
    """
    probs_ವಿಂಗಡಿಸು, probs_idx = torch.ವಿಂಗಡಿಸು(probs, dim=-1, descending=ನಿಜ)
    probs_ಮೊತ್ತ = torch.cumsum(probs_ವಿಂಗಡಿಸು, dim=-1)
    ಮುಖವಾಡ = probs_ಮೊತ್ತ - probs_ವಿಂಗಡಿಸು > p
    probs_ವಿಂಗಡಿಸು[ಮುಖವಾಡ] = 0.0
    probs_ವಿಂಗಡಿಸು.div_(probs_ವಿಂಗಡಿಸು.ಮೊತ್ತ(dim=-1, keepdim=ನಿಜ))
    ಮುಂದಿನ_token = torch.multinomial(probs_ವಿಂಗಡಿಸು, ಅಂಕೆ_samples=1)
    ಮುಂದಿನ_token = torch.gather(probs_idx, -1, ಮುಂದಿನ_token)
    ಹಿಂತಿರುಗಿ ಮುಂದಿನ_token

# -----------------------------------------------------------------------------
# Llama 3.1 Tokenizer

# The tiktoken tokenizer can handle <=400k chars without
# pyo3_runtime.PanicException.
TIKTOKEN_MAX_ENCODE_CHARS = 400_000

# https://github.com/openai/tiktoken/issues/195
# Here we iterate over subsequences and split if we exceed the limit
# of max consecutive non-whitespace or whitespace characters.
MAX_NO_WHITESPACES_CHARS = 25_000


ವರ್ಗ Tokenizer:
    """
    Tokenizing and encoding/decoding text using the Tiktoken tokenizer.
    """

    special_ಟೋಕನುಗಳು_2: Dict[str, ಪೂರ್ಣಾಂಕ]

    ಅಂಕೆ_reserved_special_ಟೋಕನುಗಳು = 256

    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: E501

    ವ್ಯಾಖ್ಯಾನ __init__(self, model_ಮಾರ್ಗ: str):
        """
        Initializes the Tokenizer with a Tiktoken model.

        Args:
            model_path (str): The path to the Tiktoken model file.
        """
        ಪ್ರತಿಜ್ಞಾ os.ಮಾರ್ಗ.isfile(model_ಮಾರ್ಗ), model_ಮಾರ್ಗ

        mergeable_ranks = ತುಂಬು_tiktoken_bpe(model_ಮಾರ್ಗ)
        ಅಂಕೆ_ತಳ_ಟೋಕನುಗಳು = ಉದ್ದ(mergeable_ranks)
        special_ಟೋಕನುಗಳು_2 = [
            "<|begin_of_text|>",
            "<|end_of_text|>",
            "<|reserved_special_token_0|>",
            "<|reserved_special_token_1|>",
            "<|finetune_right_pad_id|>",
            "<|step_id|>",
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|eom_id|>",  # end of message
            "<|eot_id|>",  # end of turn
            "<|python_tag|>",
        ]
        reserved_ಟೋಕನುಗಳು = [
            f"<|reserved_special_token_{2 + i}|>"
            ಗಾಗಿ i ಒಳಗೆ ಶ್ರೇಣಿ(self.ಅಂಕೆ_reserved_special_ಟೋಕನುಗಳು - ಉದ್ದ(special_ಟೋಕನುಗಳು_2))
        ]
        special_ಟೋಕನುಗಳು_2 = special_ಟೋಕನುಗಳು_2 + reserved_ಟೋಕನುಗಳು

        self.special_ಟೋಕನುಗಳು_2 = {
            token: ಅಂಕೆ_ತಳ_ಟೋಕನುಗಳು + i ಗಾಗಿ i, token ಒಳಗೆ enumerate(special_ಟೋಕನುಗಳು_2)
        }
        self.model = tiktoken.Encoding(
            name=ಮಾರ್ಗ_2(model_ಮಾರ್ಗ).name,
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_ಟೋಕನುಗಳು_2=self.special_ಟೋಕನುಗಳು_2,
        )

        self.n_ಪದಗಳು: ಪೂರ್ಣಾಂಕ = ಅಂಕೆ_ತಳ_ಟೋಕನುಗಳು + ಉದ್ದ(special_ಟೋಕನುಗಳು_2)
        # BOS / EOS token IDs
        self.bos_id: ಪೂರ್ಣಾಂಕ = self.special_ಟೋಕನುಗಳು_2["<|begin_of_text|>"]
        self.eos_id: ಪೂರ್ಣಾಂಕ = self.special_ಟೋಕನುಗಳು_2["<|end_of_text|>"]
        self.eot_id: ಪೂರ್ಣಾಂಕ = self.special_ಟೋಕನುಗಳು_2["<|eot_id|>"]
        self.eom_id: ಪೂರ್ಣಾಂಕ = self.special_ಟೋಕನುಗಳು_2["<|eom_id|>"]
        self.python_ಟ್ಯಾಗ್_id = self.special_ಟೋಕನುಗಳು_2["<|python_tag|>"]
        self.pad_id: ಪೂರ್ಣಾಂಕ = self.special_ಟೋಕನುಗಳು_2["<|finetune_right_pad_id|>"]
        # hardcoded stop tokens for the base model
        self.stop_ಟೋಕನುಗಳು = [
            self.special_ಟೋಕನುಗಳು_2["<|begin_of_text|>"],
            self.special_ಟೋಕನುಗಳು_2["<|end_of_text|>"],
        ]

    ವ್ಯಾಖ್ಯಾನ encode(
        self,
        s: str,
        *,
        bos: bool,
        eos: bool,
        allowed_special: Optional[Union[Literal["all"], Abstract_ಗಣ[str]]] = ಏನೂಇಲ್ಲ,
        disallowed_special: Union[Literal["all"], Collection[str]] = (),
    ) -> List[ಪೂರ್ಣಾಂಕ]:
        """
        Encodes a string into a list of token IDs.

        Args:
            s (str): The input string to be encoded.
            bos (bool): Whether to prepend the beginning-of-sequence token.
            eos (bool): Whether to append the end-of-sequence token.
            allowed_tokens ("all"|set[str]): allowed special tokens in string
            disallowed_tokens ("all"|set[str]): special tokens that raise an error when in string

        Returns:
            list[int]: A list of token IDs.

        By default, setting disallowed_special=() encodes a string by ignoring
        special tokens. Specifically:
        - Setting `disallowed_special` to () will cause all text corresponding
          to special tokens to be encoded as natural text (insteading of raising
          an error).
        - Setting `allowed_special` to "all" will treat all text corresponding
          to special tokens to be encoded as special tokens.
        """
        ಆದರೆ allowed_special ಆಗಿರುವ ಏನೂಇಲ್ಲ:
            allowed_special = ಗಣ()
        ಪ್ರತಿಜ್ಞಾ type(s) ಆಗಿರುವ str

        substrs = (
            substr
            ಗಾಗಿ i ಒಳಗೆ ಶ್ರೇಣಿ(0, ಉದ್ದ(s), TIKTOKEN_MAX_ENCODE_CHARS)
            ಗಾಗಿ substr ಒಳಗೆ self.ವಿಭಜಿಸು_whitespaces_or_nonwhitespaces(
                s[i : i + TIKTOKEN_MAX_ENCODE_CHARS], MAX_NO_WHITESPACES_CHARS
            )
        )
        t: List[ಪೂರ್ಣಾಂಕ] = []
        ಗಾಗಿ substr ಒಳಗೆ substrs:
            t.extend(
                self.model.encode(
                    substr,
                    allowed_special=allowed_special,
                    disallowed_special=disallowed_special,
                )
            )
        ಆದರೆ bos:
            t.ಸೇರಿಸು(0, self.bos_id)
        ಆದರೆ eos:
            t.append(self.eos_id)
        ಹಿಂತಿರುಗಿ t

    ವ್ಯಾಖ್ಯಾನ decode(self, t: Sequence[ಪೂರ್ಣಾಂಕ]) -> str:
        # Typecast is safe here. Tiktoken doesn't do anything list-related with the sequence.
        ಹಿಂತಿರುಗಿ self.model.decode(cast(List[ಪೂರ್ಣಾಂಕ], t))

    @staticmethod
    ವ್ಯಾಖ್ಯಾನ ವಿಭಜಿಸು_whitespaces_or_nonwhitespaces(
        s: str, max_consecutive_ತುಂಡು_len: ಪೂರ್ಣಾಂಕ
    ) -> Iterator[str]:
        """
        Splits the string `s` so that each substring contains no more than `max_consecutive_slice_len`
        consecutive whitespaces or consecutive non-whitespaces.
        """
        current_ತುಂಡು_len = 0
        current_ತುಂಡು_is_space = s[0].isspace() ಆದರೆ ಉದ್ದ(s) > 0 ಇಲ್ಲದಿದ್ದರೆ ಸುಳ್ಳು
        ತುಂಡು_ಆರಂಭ = 0

        ಗಾಗಿ i ಒಳಗೆ ಶ್ರೇಣಿ(ಉದ್ದ(s)):
            is_now_space = s[i].isspace()

            ಆದರೆ current_ತುಂಡು_is_space ^ is_now_space:
                current_ತುಂಡು_len = 1
                current_ತುಂಡು_is_space = is_now_space
            ಇಲ್ಲದಿದ್ದರೆ:
                current_ತುಂಡು_len += 1
                ಆದರೆ current_ತುಂಡು_len > max_consecutive_ತುಂಡು_len:
                    ನೀಡು s[ತುಂಡು_ಆರಂಭ:i]
                    ತುಂಡು_ಆರಂಭ = i
                    current_ತುಂಡು_len = 1
        ನೀಡು s[ತುಂಡು_ಆರಂಭ:]

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

ವ್ಯಾಖ್ಯಾನ peek_ದತ್ತಾಂಶ_shard(filename):
    # only reads the header, returns header data
    ಜೊತೆ open(filename, "rb") ಇವ f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        ಶೀರ್ಷಿಕೆ = np.frombuffer(f.ಪಠ(256*4), dtype=np.int32)
    ಆದರೆ ಶೀರ್ಷಿಕೆ[0] != 20240801:
        ಮುದ್ರಿಸು("ERROR: magic number mismatch in the data .bin file!")
        exit(1)
    ಪ್ರತಿಜ್ಞಾ ಶೀರ್ಷಿಕೆ[1] == 7, "unsupported version"
    ntok = ಶೀರ್ಷಿಕೆ[2] # number of tokens (claimed)
    ಹಿಂತಿರುಗಿ ntok # for now just return the number of tokens

ವ್ಯಾಖ್ಯಾನ ತುಂಬು_ದತ್ತಾಂಶ_shard(filename):
    ಜೊತೆ open(filename, "rb") ಇವ f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        ಶೀರ್ಷಿಕೆ = np.frombuffer(f.ಪಠ(256*4), dtype=np.int32)
        ಪ್ರತಿಜ್ಞಾ ಶೀರ್ಷಿಕೆ[0] == 20240801, "magic number mismatch in the data .bin file"
        ಪ್ರತಿಜ್ಞಾ ಶೀರ್ಷಿಕೆ[1] == 7, "unsupported version"
        ntok = ಶೀರ್ಷಿಕೆ[2] # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        ಟೋಕನುಗಳು = np.frombuffer(f.ಪಠ(), dtype=np.uint32)
    ಪ್ರತಿಜ್ಞಾ ಉದ್ದ(ಟೋಕನುಗಳು) == ntok, "number of tokens read does not match header?"
    ಹಿಂತಿರುಗಿ ಟೋಕನುಗಳು

ವರ್ಗ Distributed_Sharded_ದತ್ತಾಂಶ_Loader:
    """
    This DataLoader is both:
    - distributed (works correctly in case of multiple processes in DDP)
    - sharded (supports datasets that are broken up into multiple data shards)
    It is not *permuted*, meaning that it itearates over the data in the order
    of the dataset on disk, so the user should make sure to shuffle their examples
    during the creation of their data shards for best performance.
    """
    ವ್ಯಾಖ್ಯಾನ __init__(self, filename_ನಮೂನೆ, B, T, process_rank, ಅಂಕೆ_processes):
        self.process_rank = process_rank
        self.ಅಂಕೆ_processes = ಅಂಕೆ_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_ನಮೂನೆ))
        ಪ್ರತಿಜ್ಞಾ ಉದ್ದ(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_ಒಟ್ಟು = 0
        ಗಾಗಿ fname ಒಳಗೆ self.files:
            shard_ntok = peek_ದತ್ತಾಂಶ_shard(fname)
            ಪ್ರತಿಜ್ಞಾ shard_ntok >= ಅಂಕೆ_processes * B * T + 1
            ntok_ಒಟ್ಟು += shard_ntok
        self.ntok_ಒಟ್ಟು = ntok_ಒಟ್ಟು
        print0(f"DataLoader: total number of tokens: {ntok_total:,} across {len(self.files)} files")

        # kick things off
        self.current_shard = ಏನೂಇಲ್ಲ
        self.reset()

    ವ್ಯಾಖ್ಯಾನ reset(self):
        # we're being a bit clever here: if we already had shard 0 loaded,
        # then don't do the work to reload it, just reset the pointer
        ಆದರೆ self.current_shard != 0:
            self.current_shard = 0
            self.ಟೋಕನುಗಳು = ತುಂಬು_ದತ್ತಾಂಶ_shard(self.files[self.current_shard])
        self.current_position = self.process_rank * self.B * self.T

    ವ್ಯಾಖ್ಯಾನ advance(self): # advance to next data shard
        self.current_shard = (self.current_shard + 1) % ಉದ್ದ(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.ಟೋಕನುಗಳು = ತುಂಬು_ದತ್ತಾಂಶ_shard(self.files[self.current_shard])

    ವ್ಯಾಖ್ಯಾನ ಮುಂದಿನ_batch(self):
        B = self.B
        T = self.T
        buf = self.ಟೋಕನುಗಳು[self.current_position : self.current_position+B*T+1]
        buf = torch.tensor(buf, dtype=torch.ದೀರ್ಘ)
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the start pointer in current shard
        self.current_position += B * T * self.ಅಂಕೆ_processes
        # if loading the next batch would be out of bounds advance the shard
        ಆದರೆ self.current_position + (B * T * self.ಅಂಕೆ_processes + 1) > ಉದ್ದ(self.ಟೋಕನುಗಳು):
            self.advance()
        ಹಿಂತಿರುಗಿ x, y

# -----------------------------------------------------------------------------
# Python -> C bridge utilities for saving params/grads/activations to .bin files

ವ್ಯಾಖ್ಯಾನ write_fp32(tensor, file):
    t = tensor.detach().cpu().ಗಮ್ಯ(torch.float32)
    b = t.numpy().tobytes()
    file.ಲಿಖ(b)

ವ್ಯಾಖ್ಯಾನ write_bf16(tensor, file):
    t = tensor.detach().cpu().ಗಮ್ಯ(torch.bfloat16)
    # numpy doesn't have bf16 datatype so we have to trick it
    t = t.view(torch.int16) # trick: reinterpret as int16
    b = t.numpy().tobytes()
    file.ಲಿಖ(b)

ವ್ಯಾಖ್ಯಾನ write_tensors(model_tensors, L, file, dtype):
    # writes LLaMA 3 model's weights to a binary file
    ಪ್ರತಿಜ್ಞಾ dtype ಒಳಗೆ {"float32", "bfloat16"}
    write_fun = write_fp32 ಆದರೆ dtype == "float32" ಇಲ್ಲದಿದ್ದರೆ write_bf16
    write_fun(model_tensors["transformer.wte.weight"], file) # (V, C)
    ಗಾಗಿ i ಒಳಗೆ ಶ್ರೇಣಿ(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_1.weight"], file)
    ಗಾಗಿ i ಒಳಗೆ ಶ್ರೇಣಿ(L): # (L, 3C, C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_attn.weight"], file)
    ಗಾಗಿ i ಒಳಗೆ ಶ್ರೇಣಿ(L): # (L, C, C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_proj.weight"], file)
    ಗಾಗಿ i ಒಳಗೆ ಶ್ರೇಣಿ(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_2.weight"], file)
    ಗಾಗಿ i ಒಳಗೆ ಶ್ರೇಣಿ(L): # (L, 4C, C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_fc.weight"], file)
    ಗಾಗಿ i ಒಳಗೆ ಶ್ರೇಣಿ(L): # (L, 4C, C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_fc2.weight"], file)
    ಗಾಗಿ i ಒಳಗೆ ಶ್ರೇಣಿ(L): # (L, C, 4C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_proj.weight"], file)
    write_fun(model_tensors["transformer.ln_f.weight"], file) # (C, )
    write_fun(model_tensors["lm_head.weight"], file) # (V, C)

ವ್ಯಾಖ್ಯಾನ write_model(model, filename, dtype):
    # everything we need to instantiate the model
    # 1) header is: version int, LLaMAConfig ints, padding to 1024 bytes
    ಪ್ರತಿಜ್ಞಾ dtype ಒಳಗೆ {"float32", "bfloat16"}
    ಆವೃತ್ತಿ = {
        "float32": 3, # 3: all tensors are fp32
        "bfloat16": 5, # 5: all tensors are bf16
    }[dtype]
    ಶೀರ್ಷಿಕೆ = torch.zeros(256, dtype=torch.int32)
    ಶೀರ್ಷಿಕೆ[0] = 20240803 # magic
    ಶೀರ್ಷಿಕೆ[1] = ಆವೃತ್ತಿ # checkpoint version
    ಶೀರ್ಷಿಕೆ[2] = model.config.ಖಂಡ_ಗಾತ್ರ
    ಶೀರ್ಷಿಕೆ[3] = model.config.vocab_ಗಾತ್ರ
    ಶೀರ್ಷಿಕೆ[4] = model.config.n_layer
    ಶೀರ್ಷಿಕೆ[5] = model.config.n_ತಲೆ
    ಶೀರ್ಷಿಕೆ[6] = model.config.n_kv_ತಲೆ
    ಶೀರ್ಷಿಕೆ[7] = model.config.n_embd
    ಶೀರ್ಷಿಕೆ[8] = model.config.ffn_dim_multiplier
    ಶೀರ್ಷಿಕೆ[9] = model.config.ಅಪವರ್ತ್ಯ_ದ
    ಶೀರ್ಷಿಕೆ[10] = model.config.norm_eps
    ಶೀರ್ಷಿಕೆ[11] = model.config.rope_theta
    ಶೀರ್ಷಿಕೆ[12] = model.config.use_scaled_rope
    ಶೀರ್ಷಿಕೆ[13] = model.config.max_gen_batch_ಗಾತ್ರ
    ಶೀರ್ಷಿಕೆ[14] = ಪೂರ್ಣಾಂಕ(model.config.ಆವೃತ್ತಿ.ವಿಭಜಿಸು('.')[0]) # major version
    ಶೀರ್ಷಿಕೆ[15] = ಪೂರ್ಣಾಂಕ(model.config.ಆವೃತ್ತಿ.ವಿಭಜಿಸು('.')[1]) # minor version
    # 2) the parameters follow the header
    params = {name: param.cpu() ಗಾಗಿ name, param ಒಳಗೆ model.named_parameters()}
    # now write to file
    ಜೊತೆ open(filename, "wb") ಇವ file:
        file.ಲಿಖ(ಶೀರ್ಷಿಕೆ.numpy().tobytes()) #ಶೀರ್ಷಿಕೆ
        write_tensors(params, model.config.n_layer, file, dtype) # params
    ಮುದ್ರಿಸು(f"wrote {filename}")

ವ್ಯಾಖ್ಯಾನ write_state(model, x, y, logits, loss, filename):
    # the state is used for debugging.
    # it contains information about the input, logits, loss, and the parameter gradients
    # this can be used for checking the computation correctness in C
    ಶೀರ್ಷಿಕೆ = torch.zeros(256, dtype=torch.int32)
    ಶೀರ್ಷಿಕೆ[0] = 20240803 # magic
    ಶೀರ್ಷಿಕೆ[1] = x.ಗಾತ್ರ_2(0) # batch size of the batch, B
    ಶೀರ್ಷಿಕೆ[2] = x.ಗಾತ್ರ_2(1) # temporal extent of the batch, T
    grads = {name: param.grad.cpu() ಗಾಗಿ name, param ಒಳಗೆ model.named_parameters()}
    ಜೊತೆ open(filename, "wb") ಇವ file:
        #ಶೀರ್ಷಿಕೆ
        file.ಲಿಖ(ಶೀರ್ಷಿಕೆ.numpy().tobytes())
        # input x
        file.ಲಿಖ(x.cpu().numpy().astype("int32").tobytes()) # (B, T)
        # targets y
        file.ಲಿಖ(y.cpu().numpy().astype("int32").tobytes()) # (B, T)
        # logits (result of the model forward pass)
        write_fp32(logits.cpu(), file)
        # loss (single float, result of the cross entropy loss)
        write_fp32(loss.cpu(), file)
        # gradients
        write_tensors(grads, model.config.n_layer, file, "float32")
    ಮುದ್ರಿಸು(f"wrote {filename}")

# -----------------------------------------------------------------------------
# int main

ವ್ಯಾಖ್ಯಾನ print0(*ವಾದಗಳು, **kwargs):
    # modified print that only prints from the master process
    # if this is not a distributed run, it's just a print
    ಆದರೆ ಪೂರ್ಣಾಂಕ(os.environ.get("RANK", 0)) == 0:
        ಮುದ್ರಿಸು(*ವಾದಗಳು, **kwargs)

ಆದರೆ __name__ == "__main__":
    print0(f"Running pytorch {torch.version.__version__}")

    # default settings will overfit a tiny batch of data
    # and save model weights and debug state to disk on the first iteration
    parser = argparse.ವಾದ_Parser()
    parser.ಕೂಡಿಸು_ವಾದ("--use_hf", type=ಪೂರ್ಣಾಂಕ, ಪೂರ್ವನಿಯೋಜಿತ=1, help="use HuggingFace (default) or use Meta's model")
    parser.ಕೂಡಿಸು_ವಾದ("--ckpt_dir", type=str, ಪೂರ್ವನಿಯೋಜಿತ=ಏನೂಇಲ್ಲ, help="path to llama3 model checkpoint (needed if use_hf=0)")
    parser.ಕೂಡಿಸು_ವಾದ("--tokenizer_path", type=str, ಪೂರ್ವನಿಯೋಜಿತ=ಏನೂಇಲ್ಲ, help="path to llama3 tokenizer (needed if use_hf=0)")
    # file system input / output
    parser.ಕೂಡಿಸು_ವಾದ("--input_bin", type=str, ಪೂರ್ವನಿಯೋಜಿತ="dev/data/tinyshakespeare/tiny_shakespeare_val.bin", help="input .bin to train on")
    parser.ಕೂಡಿಸು_ವಾದ("--input_val_bin", type=str, ಪೂರ್ವನಿಯೋಜಿತ="", help="input .bin to eval validation loss on")
    parser.ಕೂಡಿಸು_ವಾದ("--output_dir", type=str, ಪೂರ್ವನಿಯೋಜಿತ="", help="output directory to which to write logs and checkpoints")
    parser.ಕೂಡಿಸು_ವಾದ("--model", type=str, ಪೂರ್ವನಿಯೋಜಿತ="meta-llama/Meta-Llama-3.1-8B", help="chose the llama model")
    # token layout for each step of the optimization
    parser.ಕೂಡಿಸು_ವಾದ("--batch_size", type=ಪೂರ್ಣಾಂಕ, ಪೂರ್ವನಿಯೋಜಿತ=4, help="batch size, in units of #batch dimensions")
    parser.ಕೂಡಿಸು_ವಾದ("--sequence_length", type=ಪೂರ್ಣಾಂಕ, ಪೂರ್ವನಿಯೋಜಿತ=64, help="sequence length")
    parser.ಕೂಡಿಸು_ವಾದ("--total_batch_size", type=ಪೂರ್ಣಾಂಕ, ಪೂರ್ವನಿಯೋಜಿತ=256, help="total desired batch size, in units of #tokens")
    # workload (number of steps)
    parser.ಕೂಡಿಸು_ವಾದ("--num_iterations", type=ಪೂರ್ಣಾಂಕ, ಪೂರ್ವನಿಯೋಜಿತ=10, help="number of iterations to run")
    parser.ಕೂಡಿಸು_ವಾದ("--inference_only", type=ಪೂರ್ಣಾಂಕ, ಪೂರ್ವನಿಯೋಜಿತ=0, help="only run inference")
    # optimization
    parser.ಕೂಡಿಸು_ವಾದ("--learning_rate", type=ದಶಮಾಂಶ, ಪೂರ್ವನಿಯೋಜಿತ=1e-5, help="learning rate warmup iterations")
    parser.ಕೂಡಿಸು_ವಾದ("--warmup_iters", type=ಪೂರ್ಣಾಂಕ, ಪೂರ್ವನಿಯೋಜಿತ=0, help="learning rate warmup iterations")
    parser.ಕೂಡಿಸು_ವಾದ("--learning_rate_decay_frac", type=ದಶಮಾಂಶ, ಪೂರ್ವನಿಯೋಜಿತ=1.0, help="learning rate warmup iterations")
    parser.ಕೂಡಿಸು_ವಾದ("--weight_decay", type=ದಶಮಾಂಶ, ಪೂರ್ವನಿಯೋಜಿತ=0.0, help="weight decay")
    parser.ಕೂಡಿಸು_ವಾದ("--grad_clip", type=ದಶಮಾಂಶ, ಪೂರ್ವನಿಯೋಜಿತ=1.0, help="maximum gradient magnitude")
    # evaluation
    parser.ಕೂಡಿಸು_ವಾದ("--val_loss_every", type=ಪೂರ್ಣಾಂಕ, ಪೂರ್ವನಿಯೋಜಿತ=0, help="every how mant steps to evaluate val loss?")
    parser.ಕೂಡಿಸು_ವಾದ("--val_max_steps", type=ಪೂರ್ಣಾಂಕ, ಪೂರ್ವನಿಯೋಜಿತ=20, help="how many batches of val to average?")
    parser.ಕೂಡಿಸು_ವಾದ("--sample_every", type=ಪೂರ್ಣಾಂಕ, ಪೂರ್ವನಿಯೋಜಿತ=0, help="how often to sample from the model?")
    # debugging
    parser.ಕೂಡಿಸು_ವಾದ("--overfit_single_batch", type=ಪೂರ್ಣಾಂಕ, ಪೂರ್ವನಿಯೋಜಿತ=1, help="overfit just one batch of data")
    # numerics
    parser.ಕೂಡಿಸು_ವಾದ("--tensorcores", type=ಪೂರ್ಣಾಂಕ, ಪೂರ್ವನಿಯೋಜಿತ=0, help="use tensorcores")
    # memory management
    parser.ಕೂಡಿಸು_ವಾದ("--device", type=str, ಪೂರ್ವನಿಯೋಜಿತ="", help="by default we autodetect, or set it here")
    parser.ಕೂಡಿಸು_ವಾದ("--compile", type=ಪೂರ್ಣಾಂಕ, ಪೂರ್ವನಿಯೋಜಿತ=0, help="torch.compile the model")
    parser.ಕೂಡಿಸು_ವಾದ("--dtype", type=str, ಪೂರ್ವನಿಯೋಜಿತ="bfloat16", help="float32|float16|bfloat16")
    parser.ಕೂಡಿಸು_ವಾದ("--zero_stage", type=ಪೂರ್ಣಾಂಕ, ಪೂರ್ವನಿಯೋಜಿತ=0, help="zero redundancy optimizer stage (0/1/2/3)")
    # python -> C bridge
    parser.ಕೂಡಿಸು_ವಾದ("--write_tensors", type=ಪೂರ್ಣಾಂಕ, ಪೂರ್ವನಿಯೋಜಿತ=0, help="write tensors to disk")
    ವಾದಗಳು = parser.parse_ವಾದಗಳು()

    # args error checking and convenience variables
    B, T = ವಾದಗಳು.batch_ಗಾತ್ರ, ವಾದಗಳು.sequence_ಉದ್ದಳತೆ
    ಪ್ರತಿಜ್ಞಾ 1 <= T <= 8192, "sequence length must be between 1 and 8192"
    ಪ್ರತಿಜ್ಞಾ ವಾದಗಳು.dtype ಒಳಗೆ {"float32", "float16", "bfloat16"}
    ಪ್ರತಿಜ್ಞಾ ವಾದಗಳು.model ಒಳಗೆ {"meta-llama/Meta-Llama-3.1-8B"}  # only 8B base model supported for now

    # create the logging directory if it does not exist
    logfile = ಏನೂಇಲ್ಲ
    ಆದರೆ ವಾದಗಳು.output_dir:
        os.makedirs(ವಾದಗಳು.output_dir, exist_ok=ನಿಜ)
        logfile = os.ಮಾರ್ಗ.join(ವಾದಗಳು.output_dir, "main.log")
        # create the log file "main.log" inside it, and wipe it clean
        ಜೊತೆ open(logfile, "w") ಇವ f:
            ದಾಟು

    # set up DDP (distributed data parallel). torchrun sets this env variable
    ddp = ಪೂರ್ಣಾಂಕ(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    ಆದರೆ ddp:
        # use of DDP atm demands CUDA, we set the device appropriately according to rank
        ಪ್ರತಿಜ್ಞಾ torch.cuda.is_ಲಭ್ಯ(), "for now i think we need CUDA for DDP"
        init_process_ಗುಂಪು(backend='nccl')
        ddp_rank = ಪೂರ್ಣಾಂಕ(os.environ['RANK'])
        ddp_local_rank = ಪೂರ್ಣಾಂಕ(os.environ['LOCAL_RANK'])
        ddp_world_ಗಾತ್ರ = ಪೂರ್ಣಾಂಕ(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.ಗಣ_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        ಬೀಜ_ಸ್ಥಾನಾಂತರ = 0 # each process gets the exact same seed
        zero_stage = ವಾದಗಳು.zero_stage
    ಇಲ್ಲದಿದ್ದರೆ:
        ddp_rank = 0
        ddp_local_rank = 0
        zero_stage = 0
        ddp_world_ಗಾತ್ರ = 1
        master_process = ನಿಜ
        ಬೀಜ_ಸ್ಥಾನಾಂತರ = 0
        # select the device
        ಆದರೆ ವಾದಗಳು.device:
            # provided explicitly by the user
            device = ವಾದಗಳು.device
        ಇಲ್ಲದಿದ್ದರೆ:
            # attempt to autodetect the device
            device = "cpu"
            ಆದರೆ torch.cuda.is_ಲಭ್ಯ():
                device = "cuda"
            ಅಥವಾಆದರೆ hasattr(torch.backends, "mps") ಮತ್ತು torch.backends.mps.is_ಲಭ್ಯ():
                device = "mps"
    device_type = 'cuda' ಆದರೆ 'cuda' ಒಳಗೆ device ಇಲ್ಲದಿದ್ದರೆ 'cpu'
    ಪ್ರತಿಜ್ಞಾ device_type ಒಳಗೆ {'cuda'}, "GPU required to run LLaMA 3"  # we need to load LLaMA as bf16 on CUDA
    ಮುದ್ರಿಸು(f"using device: {device}")

    # calculate gradient accumulation from the desired total batch size and the current run configuration
    ಟೋಕನುಗಳು_ಪ್ರತಿ_fwdbwd = B * T * ddp_world_ಗಾತ್ರ
    ಪ್ರತಿಜ್ಞಾ ವಾದಗಳು.ಒಟ್ಟು_batch_ಗಾತ್ರ % ಟೋಕನುಗಳು_ಪ್ರತಿ_fwdbwd == 0
    grad_accum_ಹೆಜ್ಜೆಗಳು = ವಾದಗಳು.ಒಟ್ಟು_batch_ಗಾತ್ರ // ಟೋಕನುಗಳು_ಪ್ರತಿ_fwdbwd
    print0(f"total desired batch size: {args.total_batch_size}")
    print0(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    # set up a context manager following the desired dtype and device
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[ವಾದಗಳು.dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) ಆದರೆ (device_type == "cuda") ಇಲ್ಲದಿದ್ದರೆ nullcontext()

    # rng / reproducibility
    torch.manual_ಬೀಜ(42)
    ಆದರೆ torch.cuda.is_ಲಭ್ಯ():
        torch.cuda.manual_ಬೀಜ(42)

    # set the torch precision mode to use TensorFloat32 (TF32) for matmuls
    # docs https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    ಆದರೆ ವಾದಗಳು.tensorcores:
        torch.ಗಣ_float32_matmul_precision('high')

    # init the model
    ಆದರೆ ವಾದಗಳು.use_hf:
        model = LLaMA.from_pretrained_llama3_hf(ವಾದಗಳು.model)
    ಇಲ್ಲದಿದ್ದರೆ:  # use Meta's checkpoint
        ಪ್ರತಿಜ್ಞಾ ವಾದಗಳು.ckpt_dir ಆಗಿರುವ ಅಲ್ಲ ಏನೂಇಲ್ಲ ಮತ್ತು os.ಮಾರ್ಗ.exists(ವಾದಗಳು.ckpt_dir), f"llama3 ckpt dir {args.ckpt_dir} does not exist"
        ಪ್ರತಿಜ್ಞಾ ವಾದಗಳು.tokenizer_ಮಾರ್ಗ ಆಗಿರುವ ಅಲ್ಲ ಏನೂಇಲ್ಲ ಮತ್ತು os.ಮಾರ್ಗ.exists(ವಾದಗಳು.tokenizer_ಮಾರ್ಗ), f"llama3 tokenizer path {args.tokenizer_path} does not exist"
        model = LLaMA.from_pretrained_llama3_meta(ವಾದಗಳು.ckpt_dir, ವಾದಗಳು.tokenizer_ಮಾರ್ಗ)

    model.train()
    ಆದರೆ ವಾದಗಳು.compile:
        ಆದರೆ hasattr(config, "coordinate_descent_tuning"):
            config.coordinate_descent_tuning = ನಿಜ # suggested by @Chillee
        print0("compiling the model...")
        model = torch.compile(model)

    # -------------------------------------------------------------------------
    # Our own version of a simple DistributedDataLoader

    # load tokens
    train_loader = Distributed_Sharded_ದತ್ತಾಂಶ_Loader(ವಾದಗಳು.ಒಳಹರಿವು_bin, B, T, ddp_rank, ddp_world_ಗಾತ್ರ)
    val_loader = ಏನೂಇಲ್ಲ
    ಆದರೆ ವಾದಗಳು.ಒಳಹರಿವು_val_bin:
        val_loader = Distributed_Sharded_ದತ್ತಾಂಶ_Loader(ವಾದಗಳು.ಒಳಹರಿವು_val_bin, B, T, ddp_rank, ddp_world_ಗಾತ್ರ)

    # -------------------------------------------------------------------------
    # PyTorch -> C bridge: save some weights and state for C to load later as reference

    # do one forward pass to generate ground truth for our C tests
    ಆದರೆ master_process ಮತ್ತು ವಾದಗಳು.write_tensors ಮತ್ತು (ಅಲ್ಲ ವಾದಗಳು.inference_only):
        x, y = train_loader.ಮುಂದಿನ_batch()
        x, y = x.ಗಮ್ಯ(device), y.ಗಮ್ಯ(device)
        logits, loss = model(x, y)
        loss.backward()
        # save model params, in bfloat16
        model_ಗಮ್ಯ_ಗಾತ್ರ = {"meta-llama/Meta-Llama-3.1-8B": "8B"}
        model_ಗಾತ್ರ_str = model_ಗಮ್ಯ_ಗಾತ್ರ[ವಾದಗಳು.model] # e.g. "8B"
        write_model(model, os.ಮಾರ್ಗ.join(ವಾದಗಳು.output_dir, f"llama3.1_{model_size_str}_bf16.bin"), dtype="bfloat16")
        # save x, y, logits, loss, and parameter gradients, for debugging C
        # always store these in fp32 to have an accurate reference (?)
        write_state(model, x, y, logits, loss, os.ಮಾರ್ಗ.join(ವಾದಗಳು.output_dir, f"llama3_{model_size_str}_debug_state.bin"))
        # reset the train_loader for the optimization below
        train_loader.reset()

    # -------------------------------------------------------------------------
    # main training loop

    # here we wrap model into DDP container
    ಆದರೆ ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module ಆದರೆ ddp ಇಲ್ಲದಿದ್ದರೆ model # always contains the "raw" unwrapped model

    # init the optimizer
    optimizer = raw_model.configure_optimizers(ತೂಕ_decay=ವಾದಗಳು.ತೂಕ_decay,
                                               learning_ದರ=ವಾದಗಳು.learning_ದರ, betas=(0.9, 0.95),
                                               device_type=device, zero_stage=zero_stage)

    # learning rate decay scheduler (cosine with warmup)
    ವ್ಯಾಖ್ಯಾನ get_lr(it):
        min_lr = ವಾದಗಳು.learning_ದರ * ವಾದಗಳು.learning_ದರ_decay_frac
        # 1) linear warmup for warmup_iters steps
        ಆದರೆ it < ವಾದಗಳು.warmup_iters:
            ಹಿಂತಿರುಗಿ ವಾದಗಳು.learning_ದರ * (it+1) / ವಾದಗಳು.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        ಆದರೆ it > ವಾದಗಳು.ಅಂಕೆ_iterations:
            ಹಿಂತಿರುಗಿ min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ಅನುಪಾತ = (it - ವಾದಗಳು.warmup_iters) / (ವಾದಗಳು.ಅಂಕೆ_iterations - ವಾದಗಳು.warmup_iters)
        ಪ್ರತಿಜ್ಞಾ 0 <= decay_ಅನುಪಾತ <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ಅನುಪಾತ)) # coeff starts at 1 and goes to 0
        ಹಿಂತಿರುಗಿ min_lr + coeff * (ವಾದಗಳು.learning_ದರ - min_lr)

    ಆದರೆ device == "cuda":
        torch.cuda.reset_peak_ಸ್ಮರಣೆ_stats()
    timings = []
    norm = -1.0   # dummy value to print in inference-only mode
    ಗಾಗಿ ಹೆಜ್ಜೆ ಒಳಗೆ ಶ್ರೇಣಿ(ವಾದಗಳು.ಅಂಕೆ_iterations + 1):
        t0 = time.time()
        ಕೊನೆಯ_ಹೆಜ್ಜೆ = (ಹೆಜ್ಜೆ == ವಾದಗಳು.ಅಂಕೆ_iterations)

        # once in a while evaluate the validation dataset
        ಆದರೆ (ವಾದಗಳು.val_loss_every > 0 \
            ಮತ್ತು (ಹೆಜ್ಜೆ % ವಾದಗಳು.val_loss_every == 0 ಅಥವಾ ಕೊನೆಯ_ಹೆಜ್ಜೆ)) \
            ಮತ್ತು (val_loader ಆಗಿರುವ ಅಲ್ಲ ಏನೂಇಲ್ಲ):
            model.eval()
            val_loader.reset()
            ಜೊತೆ torch.no_grad():
                val_loss = 0.0
                ಗಾಗಿ _ ಒಳಗೆ ಶ್ರೇಣಿ(ವಾದಗಳು.val_max_ಹೆಜ್ಜೆಗಳು):
                    x, y = val_loader.ಮುಂದಿನ_batch()
                    x, y = x.ಗಮ್ಯ(device), y.ಗಮ್ಯ(device)
                    _, loss = model(x, y, return_logits=ಸುಳ್ಳು)
                    val_loss += loss.ಅಂಶ()
                val_loss /= ವಾದಗಳು.val_max_ಹೆಜ್ಜೆಗಳು
            # log to console and to file
            print0(f"val loss {val_loss}")
            ಆದರೆ master_process ಮತ್ತು logfile ಆಗಿರುವ ಅಲ್ಲ ಏನೂಇಲ್ಲ:
                ಜೊತೆ open(logfile, "a") ಇವ f:
                    f.ಲಿಖ("s:%d tel:%f\n" % (ಹೆಜ್ಜೆ, val_loss))

        # once in a while perform model inference on the master process
        ಆದರೆ (ವಾದಗಳು.sample_every > 0 \
            ಮತ್ತು (ಹೆಜ್ಜೆ % ವಾದಗಳು.sample_every == 0 ಅಥವಾ ಕೊನೆಯ_ಹೆಜ್ಜೆ)) \
            ಮತ್ತು master_process:
            model.eval()
            prompts: List[str] = [
        "Clearly, the meaning of life is",
        "Simply put, the theory of relativity states that",
        """The repo llm.c on GitHub is""",
        """Translate English to French:

        sea otter => loutre de mer
        peppermint => menthe poivrée
        plush girafe => girafe peluche
        cheese =>""",
            ]
            ಆದರೆ ವಾದಗಳು.use_hf:
                prompt_ಟೋಕನುಗಳು = [model.tokenizer(x).ಒಳಹರಿವು_ids ಗಾಗಿ x ಒಳಗೆ prompts]
            ಇಲ್ಲದಿದ್ದರೆ:  # Meta
                prompt_ಟೋಕನುಗಳು = [model.tokenizer.encode(x, bos=ನಿಜ, eos=ಸುಳ್ಳು) ಗಾಗಿ x ಒಳಗೆ prompts]

            generation_ಟೋಕನುಗಳು = model.generate(prompt_ಟೋಕನುಗಳು, max_gen_len=64, temperature=0.6, ತುದಿ_p=0.9, echo=ಸುಳ್ಳು)
            results = [{"generation": model.tokenizer.decode(t)} ಗಾಗಿ t ಒಳಗೆ generation_ಟೋಕನುಗಳು]
            ಗಾಗಿ prompt, ಫಲಿತಾಂಶ ಒಳಗೆ zip(prompts, results):
                ಮುದ್ರಿಸು(prompt, ಕೊನೆ="")
                ಮುದ್ರಿಸು(f"{result['generation']}")
                ಮುದ್ರಿಸು("\n==================================\n")

        # bit confusing: we want to make sure to eval and sample on 0th iteration
        # but also after the very last iteration. so we loop for step <= num_iterations
        # instead of just < num_iterations (one extra due to <=), only to do
        # the validation/sampling one last time, and then we break right here as we're done.
        ಆದರೆ ಕೊನೆಯ_ಹೆಜ್ಜೆ:
            ನಿಲ್ಲಿಸು

        # --------------- TRAINING SECTION BEGIN -----------------
        model.train()
        optimizer.zero_grad(ಗಣ_ಗಮ್ಯ_none=ನಿಜ)
        # if we are trying to overfit a single batch, we reset the loader here
        ಆದರೆ ವಾದಗಳು.overfit_single_batch:
            train_loader.reset()
        # micro-batch loop where we do gradient accumulation to reach desired total batch size
        lossf = 0.0 # for getting the mean loss (as simple float) over the accumulation steps
        ಗಾಗಿ micro_ಹೆಜ್ಜೆ ಒಳಗೆ ಶ್ರೇಣಿ(grad_accum_ಹೆಜ್ಜೆಗಳು):
            # fetch a batch
            x, y = train_loader.ಮುಂದಿನ_batch()
            x, y = x.ಗಮ್ಯ(device), y.ಗಮ್ಯ(device)
            ಆದರೆ ddp:
                # we want only the last micro-step to sync grads in a DDP model
                # the official way to do this is with model.no_sync(), but that is a
                # context manager that bloats the code, so we just toggle this variable
                model.require_backward_grad_sync = (micro_ಹೆಜ್ಜೆ == grad_accum_ಹೆಜ್ಜೆಗಳು - 1)
            # forward pass
            ಜೊತೆ ctx:
                _, loss = model(x, y, return_logits=ಸುಳ್ಳು)
                # we have to scale the loss to account for gradient accumulation,
                # because the gradients just add on each successive backward().
                # addition of gradients corresponds to a SUM in the objective, but
                # instead of a SUM we want MEAN, so we scale the loss here
                loss = loss / grad_accum_ಹೆಜ್ಜೆಗಳು
                lossf += loss.detach() # keep track of the mean loss
            # backward pass
            ಆದರೆ ಅಲ್ಲ ವಾದಗಳು.inference_only:
                loss.backward()
        ಆದರೆ ddp:
            dist.all_reduce(lossf, op=dist.ReduceOp.AVG)
        lossf = lossf.ಅಂಶ()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), ವಾದಗಳು.grad_clip)
        # determine and set the learning rate for this iteration
        lr = get_lr(ಹೆಜ್ಜೆ)
        ಗಾಗಿ param_ಗುಂಪು ಒಳಗೆ optimizer.param_groups:
            param_ಗುಂಪು['lr'] = lr
        # step the optimizer
        optimizer.ಹೆಜ್ಜೆ()
        # --------------- TRAINING SECTION END -------------------
        # everything that follows now is just diagnostics, prints, logging, etc.

        # wait on the CPU for all device work to end so we get accurate per-iteration timings below
        ಆದರೆ device == "mps":
            torch.mps.synchronize()
        ಅಥವಾಆದರೆ device == "cuda":
            torch.cuda.synchronize()
        # time and print
        t1 = time.time()
        # the 0th iteration is often an outlier (much slower) => skip logging it
        ಟೋಕನುಗಳು_ಪ್ರತಿ_ಎರಡನೇ = grad_accum_ಹೆಜ್ಜೆಗಳು * ddp_world_ಗಾತ್ರ * B * T / (t1-t0)
        print0(f"step {step+1:4d}/{args.num_iterations} | train loss {lossf:.6f} | norm {norm:.4f} | lr {lr:.2e} | ({(t1-t0)*1000:.2f} ms | {tokens_per_second:.0f} tok/s)")
        # log to logile
        ಆದರೆ master_process ಮತ್ತು logfile ಆಗಿರುವ ಅಲ್ಲ ಏನೂಇಲ್ಲ:
            ಜೊತೆ open(logfile, "a") ಇವ f:
                f.ಲಿಖ("s:%d trl:%f\n" % (ಹೆಜ್ಜೆ, lossf))

        # keep track of smooth timings, last 20 iterations
        ಆದರೆ ಹೆಜ್ಜೆ > 0 ಮತ್ತು ಹೆಜ್ಜೆ > ವಾದಗಳು.ಅಂಕೆ_iterations - 20:
            timings.append(t1-t0)

    # print the average of the last 20 timings, to get something smooth-ish
    timings = timings[-20:]
    print0(f"final {len(timings)} iters avg: {np.mean(timings)*1000:.3f}ms")
    print0(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

    # -------------------------------------------------------------------------
    # clean up nice
    ಆದರೆ ddp:
        destroy_process_ಗುಂಪು()
