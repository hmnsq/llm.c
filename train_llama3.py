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

ਆਯਾਤ argparse
ਆਯਾਤ os
ਆਯਾਤ math
ਆਯਾਤ glob
ਆਯਾਤ inspect
ਤਃ contextlib ਆਯਾਤ nullcontext
ਤਃ dataclasses ਆਯਾਤ dataclass
ਤਃ pathlib ਆਯਾਤ ਰਸਤਾ_2
ਆਯਾਤ time
ਤਃ typing ਆਯਾਤ (
    Abstract_ਸੈੱਟ,
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

ਆਯਾਤ numpy ਇਵ np
ਆਯਾਤ torch
ਆਯਾਤ torch.nn ਇਵ nn
ਤਃ torch.nn ਆਯਾਤ functional ਇਵ F
ਆਯਾਤ torch._inductor.config ਇਵ config
ਤਃ torch.nn.parallel ਆਯਾਤ Distributed_ਅੰਕੜੇ_Parallel ਇਵ DDP
ਤਃ torch.distributed ਆਯਾਤ init_process_ਸਮੂਹ, destroy_process_ਸਮੂਹ
ਤਃ torch.distributed.optim ਆਯਾਤ ZeroRedundancyOptimizer
ਆਯਾਤ torch.distributed ਇਵ dist

ਆਯਾਤ tiktoken
ਤਃ tiktoken.ਲੱਦੋ ਆਯਾਤ ਲੱਦੋ_tiktoken_bpe

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the LLaMA 3.x model

# Used in Grouped Query Attention (GQA), broadcasts the key and value tensors
ਪਰਿਭਾਸ਼ਾ repeat_kv(x: torch.Tensor, n_rep: ਪੂਰਨ) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, ਸਿਰ_dim = x.shape
    ਜੇ n_rep == 1:
        ਵਾਪਸ x
    ਵਾਪਸ (
        x[:, :, :, ਕੁਝ_ਨਹੀਂ, :]
        .expand(bs, slen, n_kv_heads, n_rep, ਸਿਰ_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, ਸਿਰ_dim)
    )

# -----------------------------------------------------------------------------
# RoPE related

ਪਰਿਭਾਸ਼ਾ reshape_for_ਪ੍ਰਸਾਰਣ(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    ਪ੍ਰਤਿਜ੍ਞਾ 0 <= 1 < ndim
    ਪ੍ਰਤਿਜ੍ਞਾ freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d ਜੇ i == 1 ਜਾਂ i == ndim - 1 ਨਹੀਂਤਾਂ 1 ਲਈ i, d ਵਿੱਚ enumerate(x.shape)]
    ਵਾਪਸ freqs_cis.view(*shape)

ਪਰਿਭਾਸ਼ਾ apply_scaling(freqs: torch.Tensor):
    # Values obtained from grid search
    scale_ਗੁਣਕ = 8
    ਨੀਵਾਂ_freq_ਗੁਣਕ = 1
    ਉੱਚਾ_freq_ਗੁਣਕ = 4
    old_context_len = 8192  # original llama3 length

    ਨੀਵਾਂ_freq_wavelen = old_context_len / ਨੀਵਾਂ_freq_ਗੁਣਕ
    ਉੱਚਾ_freq_wavelen = old_context_len / ਉੱਚਾ_freq_ਗੁਣਕ
    new_freqs = []
    ਲਈ freq ਵਿੱਚ freqs:
        wavelen = 2 * math.pi / freq
        ਜੇ wavelen < ਉੱਚਾ_freq_wavelen:
            new_freqs.append(freq)
        ਜਾਂਜੇ wavelen > ਨੀਵਾਂ_freq_wavelen:
            new_freqs.append(freq / scale_ਗੁਣਕ)
        ਨਹੀਂਤਾਂ:
            ਪ੍ਰਤਿਜ੍ਞਾ ਨੀਵਾਂ_freq_wavelen != ਉੱਚਾ_freq_wavelen
            smooth = (old_context_len / wavelen - ਨੀਵਾਂ_freq_ਗੁਣਕ) / (
                ਉੱਚਾ_freq_ਗੁਣਕ - ਨੀਵਾਂ_freq_ਗੁਣਕ
            )
            new_freqs.append((1 - smooth) * freq / scale_ਗੁਣਕ + smooth * freq)
    ਵਾਪਸ torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)

ਪਰਿਭਾਸ਼ਾ apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.ਦਸ਼ਮਲਵ().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.ਦਸ਼ਮਲਵ().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_ਪ੍ਰਸਾਰਣ(freqs_cis, xq_)
    xq_ਬਾਹਰਲਾ = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_ਬਾਹਰਲਾ = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    ਵਾਪਸ xq_ਬਾਹਰਲਾ.type_as(xq), xk_ਬਾਹਰਲਾ.type_as(xk)

ਪਰਿਭਾਸ਼ਾ precompute_freqs_cis(
    dim: ਪੂਰਨ, ਸਿਰਾ: ਪੂਰਨ, theta: ਦਸ਼ਮਲਵ = 10000.0, use_scaled: bool = ਝੂਠ
):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].ਦਸ਼ਮਲਵ() / dim))
    t = torch.arange(ਸਿਰਾ, device=freqs.device, dtype=torch.float32)
    ਜੇ use_scaled:
        freqs = apply_scaling(freqs)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    ਵਾਪਸ freqs_cis

# -----------------------------------------------------------------------------
# LLaMA building blocks

# LLaMA reference code explicitly implemented RMSNorm so we copy pasted it
# (https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/api/model.py)
# we could also use nn.RMSNorm, it has slightly different numeric properties, but equivalent
ਵਰਗ RMSNorm(torch.nn.Module):
    ਪਰਿਭਾਸ਼ਾ __init__(self, dim: ਪੂਰਨ, eps: ਦਸ਼ਮਲਵ = 1e-6):
        ਪਰਮ().__init__()
        self.eps = eps
        self.ਭਾਰ = nn.Parameter(torch.ones(dim))

    ਪਰਿਭਾਸ਼ਾ _norm(self, x):
        ਵਾਪਸ x * torch.rsqrt(x.pow(2).ਔਸਤ(-1, keepdim=ਸੱਚ) + self.eps)

    ਪਰਿਭਾਸ਼ਾ forward(self, x):
        output = self._norm(x.ਦਸ਼ਮਲਵ()).type_as(x)
        ਵਾਪਸ output * self.ਭਾਰ

ਵਰਗ CausalSelfAttention(nn.Module):

    ਪਰਿਭਾਸ਼ਾ __init__(self, config):
        ਪਰਮ().__init__()
        ਪ੍ਰਤਿਜ੍ਞਾ config.n_embd % config.n_ਸਿਰ == 0

        self.n_ਸਿਰ = config.n_ਸਿਰ
        self.n_kv_ਸਿਰ = config.n_kv_ਸਿਰ
        self.n_rep = self.n_ਸਿਰ // self.n_kv_ਸਿਰ
        self.hd = config.n_embd // config.n_ਸਿਰ
        self.use_kv = config.use_kv
        self.flash = config.flash

        self.c_attn = nn.ਰੇਖਿਕ(config.n_embd, (config.n_ਸਿਰ + 2 * config.n_kv_ਸਿਰ) * self.hd, bias=ਝੂਠ)  # key, query, value projections
        self.c_proj = nn.ਰੇਖਿਕ(config.n_embd, config.n_embd, bias=ਝੂਠ)  # output projection

        # static KV cache - we could alternatively allocate it outside of the model and just pass it in when needed
        ਜੇ self.use_kv:
            self.cache_k = torch.zeros((config.max_gen_batch_ਮਾਪ, config.ਖੰਡ_ਮਾਪ, config.n_kv_ਸਿਰ, self.hd))
            self.cache_v = torch.zeros((config.max_gen_batch_ਮਾਪ, config.ਖੰਡ_ਮਾਪ, config.n_kv_ਸਿਰ, self.hd))

    ਪਰਿਭਾਸ਼ਾ forward(self, x, freqs_cis=ਕੁਝ_ਨਹੀਂ, ਸ਼ੁਰੂ_pos=ਕੁਝ_ਨਹੀਂ, ਨਕਾਬ=ਕੁਝ_ਨਹੀਂ):
        B, T, C = x.ਮਾਪ() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.ਵੰਡੋ([self.n_ਸਿਰ * self.hd, self.n_kv_ਸਿਰ * self.hd, self.n_kv_ਸਿਰ * self.hd], dim=-1)
        q, k, v = map(ਅਨਾਮ t: t.view(B, T, -1, self.hd), (q, k, v))  # (B, T, NH, HD)

        q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)  # rotate QK (rope)  <-- 1. difference compared to GPT-2

        ਜੇ self.use_kv ਅਤੇ ਨਹੀਂ self.training ਅਤੇ ਸ਼ੁਰੂ_pos >= 0:  # use kv-caching during inference
            self.cache_k[:B, ਸ਼ੁਰੂ_pos : ਸ਼ੁਰੂ_pos + T] = k
            self.cache_v[:B, ਸ਼ੁਰੂ_pos : ਸ਼ੁਰੂ_pos + T] = v
            k = self.cache_k[:B, : ਸ਼ੁਰੂ_pos + T]
            v = self.cache_v[:B, : ਸ਼ੁਰੂ_pos + T]

        k = repeat_kv(k, self.n_rep)  # GQA <-- 2. difference compared to GPT-2
        v = repeat_kv(v, self.n_rep)

        q, k, v = map(ਅਨਾਮ t: t.transpose(1, 2), (q, k, v))  # (B, NH, T, HD)

        ਜੇ self.flash:
            # flashattention
            # if T == 1 no need to mask, otherwise the function complains
            # scaled_dot_product_attention expects a mask where value of True indicates that the element should take part in attention
            # our mask is the opposite, so we need to invert it
            y = F.scaled_dot_ਗੁਣਨਫਲ_attention(q, k, v, ਨਕਾਬ == 0 ਜੇ T > 1 ਨਹੀਂਤਾਂ ਕੁਝ_ਨਹੀਂ)
        ਨਹੀਂਤਾਂ:
            # manual implementation of attention
            # this materializes the large (T,T) matrix for all the queries and keys
            scores = (q @ k.transpose(-2, -1)) * (1.0 / math.ਵਰਗਮੂਲ(self.hd))
            ਜੇ ਨਕਾਬ ਹੈ ਨਹੀਂ ਕੁਝ_ਨਹੀਂ:
                scores.masked_fill_(ਨਕਾਬ, torch.finfo(scores.dtype).min)
            att = F.softmax(scores.ਦਸ਼ਮਲਵ(), dim=-1).type_as(q)
            y = att @ v # (B, NH, T, T) x (B, NH, T, HD) -> (B, NH, T, HD)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        ਵਾਪਸ y

ਵਰਗ MLP(nn.Module):

    ਪਰਿਭਾਸ਼ਾ __init__(self, config):
        ਪਰਮ().__init__()
        hidden_dim = 4 * config.n_embd
        hidden_dim = ਪੂਰਨ(2 * hidden_dim / 3)
        # custom dim factor multiplier
        ਜੇ config.ffn_dim_multiplier ਹੈ ਨਹੀਂ ਕੁਝ_ਨਹੀਂ:
            hidden_dim = ਪੂਰਨ(config.ffn_dim_multiplier * hidden_dim)
        hidden_dim = config.ਗੁਣਜ_ਦਾ * ((hidden_dim + config.ਗੁਣਜ_ਦਾ - 1) // config.ਗੁਣਜ_ਦਾ)
        self.c_fc = nn.ਰੇਖਿਕ(config.n_embd, hidden_dim, bias=ਝੂਠ)
        self.c_fc2 = nn.ਰੇਖਿਕ(config.n_embd, hidden_dim, bias=ਝੂਠ)
        self.c_proj = nn.ਰੇਖਿਕ(hidden_dim, config.n_embd, bias=ਝੂਠ)

    ਪਰਿਭਾਸ਼ਾ forward(self, x):
        # SwiGLU self.c_proj(F.silu(self.c_fc2(x)) * self.c_fc(x))  <-- 3. difference compared to GPT-2
        x1 = self.c_fc(x)
        x2 = self.c_fc2(x)
        x2 = F.silu(x2)
        x = x1 * x2
        x = self.c_proj(x)
        ਵਾਪਸ x

ਵਰਗ ਖੰਡ(nn.Module):

    ਪਰਿਭਾਸ਼ਾ __init__(self, config):
        ਪਰਮ().__init__()
        self.ln_1 = RMSNorm(config.n_embd, config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd, config.norm_eps)
        self.mlp = MLP(config)

    ਪਰਿਭਾਸ਼ਾ forward(self, x, freqs_cis=ਕੁਝ_ਨਹੀਂ, ਸ਼ੁਰੂ_pos=ਕੁਝ_ਨਹੀਂ, ਨਕਾਬ=ਕੁਝ_ਨਹੀਂ):
        x = x + self.attn(self.ln_1(x), freqs_cis, ਸ਼ੁਰੂ_pos, ਨਕਾਬ)
        x = x + self.mlp(self.ln_2(x))
        ਵਾਪਸ x

# -----------------------------------------------------------------------------
# The main LLaMA 3.1 model

@dataclass
ਵਰਗ LlamaConfig:
    ਸੰਸਕਰਣ: str = "3.1"
    ਖੰਡ_ਮਾਪ: ਪੂਰਨ = 8192
    vocab_ਮਾਪ: ਪੂਰਨ = 128256
    n_layer: ਪੂਰਨ = 32
    n_ਸਿਰ: ਪੂਰਨ = 32
    n_kv_ਸਿਰ: ਪੂਰਨ = 8
    n_embd: ਪੂਰਨ = 4096
    ffn_dim_multiplier: ਦਸ਼ਮਲਵ = 1.3
    ਗੁਣਜ_ਦਾ: ਪੂਰਨ = 1024
    norm_eps: ਦਸ਼ਮਲਵ = 1e-5
    rope_theta: ਦਸ਼ਮਲਵ = 500000.0
    use_scaled_rope: bool = ਸੱਚ
    max_gen_batch_ਮਾਪ: ਪੂਰਨ = 4
    use_kv: bool = ਸੱਚ
    flash: bool = ਝੂਠ  # use flashattention?

    ਪਰਿਭਾਸ਼ਾ __init__(self, **kwargs):
        ਲਈ k, v ਵਿੱਚ kwargs.ਚੀਜ਼ਾਂ():
            ਜੇ hasattr(self, k):
                setattr(self, k, v)
        ਪ੍ਰਤਿਜ੍ਞਾ self.n_kv_ਸਿਰ <= self.n_ਸਿਰ
        ਪ੍ਰਤਿਜ੍ਞਾ self.n_ਸਿਰ % self.n_kv_ਸਿਰ == 0
        ਪ੍ਰਤਿਜ੍ਞਾ self.n_embd % self.n_ਸਿਰ == 0

ਵਰਗ LLaMA(nn.Module):

    ਪਰਿਭਾਸ਼ਾ __init__(self, config):
        ਪਰਮ().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_ਮਾਪ, config.n_embd),
            h = nn.ModuleList([ਖੰਡ(config) ਲਈ _ ਵਿੱਚ ਸੀਮਾ(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd, config.norm_eps),
        ))
        self.lm_ਸਿਰ = nn.ਰੇਖਿਕ(config.n_embd, config.vocab_ਮਾਪ, bias=ਝੂਠ)

        # init all weights, use a torch rng object to be very careful
        self.init_rng = torch.Generator()
        self.init_rng.manual_ਬੀਜ(42)

        self.freqs_cis = precompute_freqs_cis(
            config.n_embd // config.n_ਸਿਰ,
            config.ਖੰਡ_ਮਾਪ * 2,
            config.rope_theta,
            config.use_scaled_rope,
        )

    ਪਰਿਭਾਸ਼ਾ forward(self, idx, targets=ਕੁਝ_ਨਹੀਂ, return_logits=ਸੱਚ, ਸ਼ੁਰੂ_pos=0):
        _, t = idx.ਮਾਪ()
        ਪ੍ਰਤਿਜ੍ਞਾ t <= self.config.ਖੰਡ_ਮਾਪ, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # forward the LLaMA model itself
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        freqs_cis = self.freqs_cis[ਸ਼ੁਰੂ_pos:ਸ਼ੁਰੂ_pos+t]

        ਨਕਾਬ = torch.triu(torch.ones((t, t), device=ਅਗਲਾ_2(self.parameters()).device, dtype=torch.bool), diagonal=1)

        ਲਈ i, ਖੰਡ_2 ਵਿੱਚ enumerate(self.transformer.h):
            x = ਖੰਡ_2(x, freqs_cis, ਸ਼ੁਰੂ_pos, ਨਕਾਬ)
        x = self.transformer.ln_f(x)

        ਜੇ targets ਹੈ ਨਹੀਂ ਕੁਝ_ਨਹੀਂ:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_ਸਿਰ(x).ਦਸ਼ਮਲਵ()
            loss = F.cross_entropy(logits.view(-1, logits.ਮਾਪ(-1)), targets.view(-1), ignore_ਸੂਚਕਾਂਕ=-1)
        ਨਹੀਂਤਾਂ:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_ਸਿਰ(x[:, [-1], :]).ਦਸ਼ਮਲਵ() # note: using list [-1] to preserve the time dim
            loss = ਕੁਝ_ਨਹੀਂ

        # there are performance reasons why not returning logits is prudent, if not needed
        ਜੇ ਨਹੀਂ return_logits:
            logits = ਕੁਝ_ਨਹੀਂ

        ਵਾਪਸ logits, loss

    @staticmethod
    ਪਰਿਭਾਸ਼ਾ adapt_llama_state_dict_ਕੁੰਜੀਆਂ(checkpoint, config: LlamaConfig):
        # Modify key names from Meta's LLaMA to our LLaMA
        # our key names are derived from GPT-2's key names
        checkpoint['transformer.wte.weight'] = checkpoint.ਕੱਢੋ('tok_embeddings.weight')

        ਲਈ i ਵਿੱਚ ਸੀਮਾ(config.n_layer):
            ਲਈ name ਵਿੱਚ ['attention_norm', 'ffn_norm']:
                old_ਕੁੰਜੀ = f'layers.{i}.{name}.weight'  # e.g. layers.x.attention_norm.weight -> transformer.h.x.ln_1.weight
                new_ਕੁੰਜੀ = f'transformer.h.{i}.ln_{1 if name == "attention_norm" else 2}.weight'
                checkpoint[new_ਕੁੰਜੀ] = checkpoint.ਕੱਢੋ(old_ਕੁੰਜੀ)

        ਲਈ i ਵਿੱਚ ਸੀਮਾ(config.n_layer):
            ਲਈ name ਵਿੱਚ ['attention.wq', 'attention.wk', 'attention.wv']:
                old_ਕੁੰਜੀ = f'layers.{i}.{name}.weight'
                new_ਕੁੰਜੀ = f'transformer.h.{i}.attn.c_attn.weight'
                ਜੇ name == 'attention.wq':
                    checkpoint[new_ਕੁੰਜੀ] = checkpoint.ਕੱਢੋ(old_ਕੁੰਜੀ)
                ਨਹੀਂਤਾਂ:  # merge 3 weights into transformer.h.x.attn.c_attn.weight
                    checkpoint[new_ਕੁੰਜੀ] = torch.cat((checkpoint[new_ਕੁੰਜੀ], checkpoint.ਕੱਢੋ(old_ਕੁੰਜੀ)), dim=0)
            old_ਕੁੰਜੀ = f'layers.{i}.attention.wo.weight'
            new_ਕੁੰਜੀ = f'transformer.h.{i}.attn.c_proj.weight'
            checkpoint[new_ਕੁੰਜੀ] = checkpoint.ਕੱਢੋ(old_ਕੁੰਜੀ)

        ffn_map = {'w1': 'c_fc2', 'w2': 'c_proj', 'w3': 'c_fc'}
        ਲਈ i ਵਿੱਚ ਸੀਮਾ(config.n_layer):
            ਲਈ name ਵਿੱਚ ['feed_forward.w1', 'feed_forward.w2', 'feed_forward.w3']:
                old_ਕੁੰਜੀ = f'layers.{i}.{name}.weight'
                new_ਕੁੰਜੀ = f'transformer.h.{i}.mlp.{ffn_map[name.split(".")[-1]]}.weight'
                checkpoint[new_ਕੁੰਜੀ] = checkpoint.ਕੱਢੋ(old_ਕੁੰਜੀ)

        checkpoint['transformer.ln_f.weight'] = checkpoint.ਕੱਢੋ('norm.weight')
        checkpoint['lm_head.weight'] = checkpoint.ਕੱਢੋ('output.weight')

        ਵਾਪਸ checkpoint

    @staticmethod
    ਪਰਿਭਾਸ਼ਾ adapt_llama_state_dict_ਕੁੰਜੀਆਂ_hf(checkpoint, config: LlamaConfig):
        # Modify key names from HuggingFace's LLaMA to our LLaMA
        # our key names are derived from GPT-2's key names
        checkpoint['transformer.wte.weight'] = checkpoint.ਕੱਢੋ('model.embed_tokens.weight')

        # We need to unpermute K and V because HF script permuted the original Meta-LLaMA weights
        # see: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py
        ਪਰਿਭਾਸ਼ਾ unpermute(w, n_heads, dim1, dim2):
            ਵਾਪਸ w.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)

        ਲਈ i ਵਿੱਚ ਸੀਮਾ(config.n_layer):
            ਲਈ name ਵਿੱਚ ['input_layernorm', 'post_attention_layernorm']:
                old_ਕੁੰਜੀ = f'model.layers.{i}.{name}.weight'  # e.g. layers.x.attention_norm.weight -> transformer.h.x.ln_1.weight
                new_ਕੁੰਜੀ = f'transformer.h.{i}.ln_{1 if name == "input_layernorm" else 2}.weight'
                checkpoint[new_ਕੁੰਜੀ] = checkpoint.ਕੱਢੋ(old_ਕੁੰਜੀ)

        ਲਈ i ਵਿੱਚ ਸੀਮਾ(config.n_layer):
            ਲਈ name ਵਿੱਚ ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj']:
                old_ਕੁੰਜੀ = f'model.layers.{i}.{name}.weight'
                new_ਕੁੰਜੀ = f'transformer.h.{i}.attn.c_attn.weight'
                ਜੇ name == 'self_attn.q_proj':
                    checkpoint[new_ਕੁੰਜੀ] = unpermute(checkpoint.ਕੱਢੋ(old_ਕੁੰਜੀ), config.n_ਸਿਰ, config.n_embd, config.n_embd)
                ਨਹੀਂਤਾਂ:  # merge 3 weights into transformer.h.x.attn.c_attn.weight
                    tensor = checkpoint.ਕੱਢੋ(old_ਕੁੰਜੀ)
                    ਜੇ name == 'self_attn.k_proj':
                        tensor = unpermute(tensor, config.n_kv_ਸਿਰ, config.n_kv_ਸਿਰ * (config.n_embd // config.n_ਸਿਰ), config.n_embd)
                    checkpoint[new_ਕੁੰਜੀ] = torch.cat((checkpoint[new_ਕੁੰਜੀ], tensor), dim=0)
            old_ਕੁੰਜੀ = f'model.layers.{i}.self_attn.o_proj.weight'
            new_ਕੁੰਜੀ = f'transformer.h.{i}.attn.c_proj.weight'
            checkpoint[new_ਕੁੰਜੀ] = checkpoint.ਕੱਢੋ(old_ਕੁੰਜੀ)

        ffn_map = {'gate_proj': 'c_fc2', 'down_proj': 'c_proj', 'up_proj': 'c_fc'}
        ਲਈ i ਵਿੱਚ ਸੀਮਾ(config.n_layer):
            ਲਈ name ਵਿੱਚ ['gate_proj', 'down_proj', 'up_proj']:
                old_ਕੁੰਜੀ = f'model.layers.{i}.mlp.{name}.weight'
                new_ਕੁੰਜੀ = f'transformer.h.{i}.mlp.{ffn_map[name]}.weight'
                checkpoint[new_ਕੁੰਜੀ] = checkpoint.ਕੱਢੋ(old_ਕੁੰਜੀ)

        checkpoint['transformer.ln_f.weight'] = checkpoint.ਕੱਢੋ('model.norm.weight')

        ਵਾਪਸ checkpoint

    @classmethod
    ਪਰਿਭਾਸ਼ਾ from_pretrained_llama3_hf(cls, model_id):
        """Loads pretrained LLaMA model weights from HuggingFace"""
        ਤਃ transformers ਆਯਾਤ AutoModelForCausalLM, AutoTokenizer
        ਪ੍ਰਤਿਜ੍ਞਾ model_id == "meta-llama/Meta-Llama-3.1-8B", "Only the 8B-base model is supported for now"
        model_ਦਲੀਲਾਂ = LlamaConfig()

        model = AutoModelForCausalLM.from_pretrained(model_id)
        checkpoint = LLaMA.adapt_llama_state_dict_ਕੁੰਜੀਆਂ_hf(model.state_dict(), model_ਦਲੀਲਾਂ)

        original_default_type = torch.get_default_dtype()  # save the default type
        torch.ਸੈੱਟ_default_tensor_type(torch.cuda.BFloat16Tensor)  # much faster loading
        model = LLaMA(model_ਦਲੀਲਾਂ)
        model.ਲੱਦੋ_state_dict(checkpoint, strict=ਝੂਠ)
        torch.ਸੈੱਟ_default_tensor_type(torch.tensor([], dtype=original_default_type, device="cpu").type())  # restore default type

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_id = 128004  # this is the pad token id for LLaMA 3.1 base, we need to set this explicitly as our generate func expects it
        tokenizer.stop_ਟੋਕਨ = [tokenizer.eos_token_id]
        model.tokenizer = tokenizer
        ਵਾਪਸ model

    @classmethod
    ਪਰਿਭਾਸ਼ਾ from_pretrained_llama3_meta(cls, ckpt_dir, tokenizer_ਰਸਤਾ):
        """Loads pretrained LLaMA model weights from a checkpoint directory"""
        model_ਦਲੀਲਾਂ = LlamaConfig()

        ckpt_ਰਸਤਾ = sorted(ਰਸਤਾ_2(ckpt_dir).glob("*.pth"))[0]
        checkpoint = torch.ਲੱਦੋ(ckpt_ਰਸਤਾ, map_location="cpu", weights_only=ਸੱਚ)
        checkpoint = LLaMA.adapt_llama_state_dict_ਕੁੰਜੀਆਂ(checkpoint, model_ਦਲੀਲਾਂ)

        original_default_type = torch.get_default_dtype()  # save the default type
        torch.ਸੈੱਟ_default_tensor_type(torch.cuda.BFloat16Tensor)  # much faster loading
        model = LLaMA(model_ਦਲੀਲਾਂ)
        model.ਲੱਦੋ_state_dict(checkpoint, strict=ਝੂਠ)
        torch.ਸੈੱਟ_default_tensor_type(torch.tensor([], dtype=original_default_type, device="cpu").type())  # restore default type

        tokenizer = Tokenizer(model_ਰਸਤਾ=tokenizer_ਰਸਤਾ)
        model.tokenizer = tokenizer
        ਵਾਪਸ model

    ਪਰਿਭਾਸ਼ਾ configure_optimizers(self, ਭਾਰ_decay, learning_ਦਰ, betas, device_type, zero_stage):
        # start with all of the candidate parameters
        param_dict = {pn: p ਲਈ pn, p ਵਿੱਚ self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p ਲਈ pn, p ਵਿੱਚ param_dict.ਚੀਜ਼ਾਂ() ਜੇ p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p ਲਈ n, p ਵਿੱਚ param_dict.ਚੀਜ਼ਾਂ() ਜੇ p.dim() >= 2]
        nodecay_params = [p ਲਈ n, p ਵਿੱਚ param_dict.ਚੀਜ਼ਾਂ() ਜੇ p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': ਭਾਰ_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        ਅੰਕ_decay_params = ਜੋੜ(p.numel() ਲਈ p ਵਿੱਚ decay_params)
        ਅੰਕ_nodecay_params = ਜੋੜ(p.numel() ਲਈ p ਵਿੱਚ nodecay_params)
        print0(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print0(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_ਉਪਲਬਧ = 'fused' ਵਿੱਚ inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_ਉਪਲਬਧ ਅਤੇ device_type == 'cuda'
        print0(f"using fused AdamW: {use_fused}")
        ਜੇ zero_stage == 1:
            print0("using ZeroRedundancyOptimizer")
            optimizer = ZeroRedundancyOptimizer(**optim_groups[0], optimizer_class=torch.optim.AdamW,
                                                lr=learning_ਦਰ, betas=betas, fused=use_fused)
            optimizer.ਜੋੜੋ_param_ਸਮੂਹ(optim_groups[1])
        ਨਹੀਂਤਾਂ:
            print0("using regular AdamW")
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_ਦਰ, betas=betas, fused=use_fused)
        ਵਾਪਸ optimizer

    @torch.inference_ਬਹੁਲਕ()
    ਪਰਿਭਾਸ਼ਾ generate(
        self,
        prompt_ਟੋਕਨ: List[List[ਪੂਰਨ]],
        max_gen_len: ਪੂਰਨ,
        temperature: ਦਸ਼ਮਲਵ = 0.6,
        ਸਿਖਰ_p: ਦਸ਼ਮਲਵ = 0.9,
        echo: bool = ਝੂਠ,
    ) -> Tuple[List[List[ਪੂਰਨ]], Optional[List[List[ਦਸ਼ਮਲਵ]]]]:
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
        bsz = ਲੰਬਾਈ(prompt_ਟੋਕਨ)
        ਪ੍ਰਤਿਜ੍ਞਾ bsz <= self.config.max_gen_batch_ਮਾਪ, f"Batch size {bsz} exceeds the maximum generation batch size {self.config.max_gen_batch_size}"
        device = ਅਗਲਾ_2(self.parameters()).device

        min_prompt_len = min(ਲੰਬਾਈ(t) ਲਈ t ਵਿੱਚ prompt_ਟੋਕਨ)
        max_prompt_len = max(ਲੰਬਾਈ(t) ਲਈ t ਵਿੱਚ prompt_ਟੋਕਨ)
        ਪ੍ਰਤਿਜ੍ਞਾ max_prompt_len <= self.config.ਖੰਡ_ਮਾਪ, f"Prompt length {max_prompt_len} exceeds the maximum block size {self.config.block_size}"
        ਕੁੱਲ_len = min(self.config.ਖੰਡ_ਮਾਪ, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_id
        ਟੋਕਨ = torch.ਭਰਿਆ((bsz, ਕੁੱਲ_len), pad_id, dtype=torch.ਲੰਮਾ, device=device)
        ਲਈ idx, t ਵਿੱਚ enumerate(prompt_ਟੋਕਨ):
            ਟੋਕਨ[idx, : ਲੰਬਾਈ(t)] = torch.tensor(t, dtype=torch.ਲੰਮਾ, device=device)

        prev_pos = 0
        eos_reached = torch.tensor([ਝੂਠ] * bsz, device=device)
        ਨਿਵੇਸ਼_ਪਾਠ_ਨਕਾਬ = ਟੋਕਨ != pad_id

        ਜੇ min_prompt_len == ਕੁੱਲ_len:
            logits, _ = self.forward(ਟੋਕਨ, ਸ਼ੁਰੂ_pos=prev_pos)

        stop_ਟੋਕਨ = torch.tensor(list(self.tokenizer.stop_ਟੋਕਨ)).ਮੰਜ਼ਿਲ(device)

        ਲਈ cur_pos ਵਿੱਚ ਸੀਮਾ(min_prompt_len, ਕੁੱਲ_len):
            logits, _ = self.forward(ਟੋਕਨ[:, prev_pos:cur_pos], ਸ਼ੁਰੂ_pos=prev_pos)
            ਜੇ temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                ਅਗਲਾ_token = sample_ਸਿਖਰ_p(probs, ਸਿਖਰ_p)
            ਨਹੀਂਤਾਂ:
                ਅਗਲਾ_token = torch.argmax(logits[:, -1], dim=-1)

            ਅਗਲਾ_token = ਅਗਲਾ_token.reshape(-1)
            # only replace token if prompt has already been generated
            ਅਗਲਾ_token = torch.where(ਨਿਵੇਸ਼_ਪਾਠ_ਨਕਾਬ[:, cur_pos], ਟੋਕਨ[:, cur_pos], ਅਗਲਾ_token)
            ਟੋਕਨ[:, cur_pos] = ਅਗਲਾ_token
            eos_reached |= ~ਨਿਵੇਸ਼_ਪਾਠ_ਨਕਾਬ[:, cur_pos] & torch.isin(ਅਗਲਾ_token, stop_ਟੋਕਨ)
            prev_pos = cur_pos
            ਜੇ all(eos_reached):
                ਰੋਕੋ

        ਬਾਹਰਲਾ_ਟੋਕਨ = []
        ਲਈ i, toks ਵਿੱਚ enumerate(ਟੋਕਨ.tolist()):
            # cut to max gen len
            ਸ਼ੁਰੂ = 0 ਜੇ echo ਨਹੀਂਤਾਂ ਲੰਬਾਈ(prompt_ਟੋਕਨ[i])
            toks = toks[ਸ਼ੁਰੂ : ਲੰਬਾਈ(prompt_ਟੋਕਨ[i]) + max_gen_len]
            # cut to after eos tok if any
            ਲਈ stop_token ਵਿੱਚ self.tokenizer.stop_ਟੋਕਨ:
                ਕੋਸ਼ਿਸ਼:
                    eos_idx = toks.ਸੂਚਕਾਂਕ(stop_token)
                    toks = toks[:eos_idx]
                ਅਪਵਾਦ ਮੁੱਲ_Error:
                    ਲੰਘੋ
            ਬਾਹਰਲਾ_ਟੋਕਨ.append(toks)
        ਵਾਪਸ ਬਾਹਰਲਾ_ਟੋਕਨ

# -----------------------------------------------------------------------------
# sampling utils

ਪਰਿਭਾਸ਼ਾ sample_ਸਿਖਰ_p(probs, p):
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
    probs_ਛਾਂਟੋ, probs_idx = torch.ਛਾਂਟੋ(probs, dim=-1, descending=ਸੱਚ)
    probs_ਜੋੜ = torch.cumsum(probs_ਛਾਂਟੋ, dim=-1)
    ਨਕਾਬ = probs_ਜੋੜ - probs_ਛਾਂਟੋ > p
    probs_ਛਾਂਟੋ[ਨਕਾਬ] = 0.0
    probs_ਛਾਂਟੋ.div_(probs_ਛਾਂਟੋ.ਜੋੜ(dim=-1, keepdim=ਸੱਚ))
    ਅਗਲਾ_token = torch.multinomial(probs_ਛਾਂਟੋ, ਅੰਕ_samples=1)
    ਅਗਲਾ_token = torch.gather(probs_idx, -1, ਅਗਲਾ_token)
    ਵਾਪਸ ਅਗਲਾ_token

# -----------------------------------------------------------------------------
# Llama 3.1 Tokenizer

# The tiktoken tokenizer can handle <=400k chars without
# pyo3_runtime.PanicException.
TIKTOKEN_MAX_ENCODE_CHARS = 400_000

# https://github.com/openai/tiktoken/issues/195
# Here we iterate over subsequences and split if we exceed the limit
# of max consecutive non-whitespace or whitespace characters.
MAX_NO_WHITESPACES_CHARS = 25_000


ਵਰਗ Tokenizer:
    """
    Tokenizing and encoding/decoding text using the Tiktoken tokenizer.
    """

    special_ਟੋਕਨ_2: Dict[str, ਪੂਰਨ]

    ਅੰਕ_reserved_special_ਟੋਕਨ = 256

    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: E501

    ਪਰਿਭਾਸ਼ਾ __init__(self, model_ਰਸਤਾ: str):
        """
        Initializes the Tokenizer with a Tiktoken model.

        Args:
            model_path (str): The path to the Tiktoken model file.
        """
        ਪ੍ਰਤਿਜ੍ਞਾ os.ਰਸਤਾ.isfile(model_ਰਸਤਾ), model_ਰਸਤਾ

        mergeable_ranks = ਲੱਦੋ_tiktoken_bpe(model_ਰਸਤਾ)
        ਅੰਕ_ਅਧਾਰ_ਟੋਕਨ = ਲੰਬਾਈ(mergeable_ranks)
        special_ਟੋਕਨ_2 = [
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
        reserved_ਟੋਕਨ = [
            f"<|reserved_special_token_{2 + i}|>"
            ਲਈ i ਵਿੱਚ ਸੀਮਾ(self.ਅੰਕ_reserved_special_ਟੋਕਨ - ਲੰਬਾਈ(special_ਟੋਕਨ_2))
        ]
        special_ਟੋਕਨ_2 = special_ਟੋਕਨ_2 + reserved_ਟੋਕਨ

        self.special_ਟੋਕਨ_2 = {
            token: ਅੰਕ_ਅਧਾਰ_ਟੋਕਨ + i ਲਈ i, token ਵਿੱਚ enumerate(special_ਟੋਕਨ_2)
        }
        self.model = tiktoken.Encoding(
            name=ਰਸਤਾ_2(model_ਰਸਤਾ).name,
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_ਟੋਕਨ_2=self.special_ਟੋਕਨ_2,
        )

        self.n_ਸ਼ਬਦ: ਪੂਰਨ = ਅੰਕ_ਅਧਾਰ_ਟੋਕਨ + ਲੰਬਾਈ(special_ਟੋਕਨ_2)
        # BOS / EOS token IDs
        self.bos_id: ਪੂਰਨ = self.special_ਟੋਕਨ_2["<|begin_of_text|>"]
        self.eos_id: ਪੂਰਨ = self.special_ਟੋਕਨ_2["<|end_of_text|>"]
        self.eot_id: ਪੂਰਨ = self.special_ਟੋਕਨ_2["<|eot_id|>"]
        self.eom_id: ਪੂਰਨ = self.special_ਟੋਕਨ_2["<|eom_id|>"]
        self.python_ਟੈਗ_id = self.special_ਟੋਕਨ_2["<|python_tag|>"]
        self.pad_id: ਪੂਰਨ = self.special_ਟੋਕਨ_2["<|finetune_right_pad_id|>"]
        # hardcoded stop tokens for the base model
        self.stop_ਟੋਕਨ = [
            self.special_ਟੋਕਨ_2["<|begin_of_text|>"],
            self.special_ਟੋਕਨ_2["<|end_of_text|>"],
        ]

    ਪਰਿਭਾਸ਼ਾ encode(
        self,
        s: str,
        *,
        bos: bool,
        eos: bool,
        allowed_special: Optional[Union[Literal["all"], Abstract_ਸੈੱਟ[str]]] = ਕੁਝ_ਨਹੀਂ,
        disallowed_special: Union[Literal["all"], Collection[str]] = (),
    ) -> List[ਪੂਰਨ]:
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
        ਜੇ allowed_special ਹੈ ਕੁਝ_ਨਹੀਂ:
            allowed_special = ਸੈੱਟ()
        ਪ੍ਰਤਿਜ੍ਞਾ type(s) ਹੈ str

        substrs = (
            substr
            ਲਈ i ਵਿੱਚ ਸੀਮਾ(0, ਲੰਬਾਈ(s), TIKTOKEN_MAX_ENCODE_CHARS)
            ਲਈ substr ਵਿੱਚ self.ਵੰਡੋ_whitespaces_or_nonwhitespaces(
                s[i : i + TIKTOKEN_MAX_ENCODE_CHARS], MAX_NO_WHITESPACES_CHARS
            )
        )
        t: List[ਪੂਰਨ] = []
        ਲਈ substr ਵਿੱਚ substrs:
            t.extend(
                self.model.encode(
                    substr,
                    allowed_special=allowed_special,
                    disallowed_special=disallowed_special,
                )
            )
        ਜੇ bos:
            t.ਪਾਓ(0, self.bos_id)
        ਜੇ eos:
            t.append(self.eos_id)
        ਵਾਪਸ t

    ਪਰਿਭਾਸ਼ਾ decode(self, t: Sequence[ਪੂਰਨ]) -> str:
        # Typecast is safe here. Tiktoken doesn't do anything list-related with the sequence.
        ਵਾਪਸ self.model.decode(cast(List[ਪੂਰਨ], t))

    @staticmethod
    ਪਰਿਭਾਸ਼ਾ ਵੰਡੋ_whitespaces_or_nonwhitespaces(
        s: str, max_consecutive_ਟੁਕੜਾ_len: ਪੂਰਨ
    ) -> Iterator[str]:
        """
        Splits the string `s` so that each substring contains no more than `max_consecutive_slice_len`
        consecutive whitespaces or consecutive non-whitespaces.
        """
        current_ਟੁਕੜਾ_len = 0
        current_ਟੁਕੜਾ_is_space = s[0].isspace() ਜੇ ਲੰਬਾਈ(s) > 0 ਨਹੀਂਤਾਂ ਝੂਠ
        ਟੁਕੜਾ_ਸ਼ੁਰੂ = 0

        ਲਈ i ਵਿੱਚ ਸੀਮਾ(ਲੰਬਾਈ(s)):
            is_now_space = s[i].isspace()

            ਜੇ current_ਟੁਕੜਾ_is_space ^ is_now_space:
                current_ਟੁਕੜਾ_len = 1
                current_ਟੁਕੜਾ_is_space = is_now_space
            ਨਹੀਂਤਾਂ:
                current_ਟੁਕੜਾ_len += 1
                ਜੇ current_ਟੁਕੜਾ_len > max_consecutive_ਟੁਕੜਾ_len:
                    ਦਿਓ s[ਟੁਕੜਾ_ਸ਼ੁਰੂ:i]
                    ਟੁਕੜਾ_ਸ਼ੁਰੂ = i
                    current_ਟੁਕੜਾ_len = 1
        ਦਿਓ s[ਟੁਕੜਾ_ਸ਼ੁਰੂ:]

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

ਪਰਿਭਾਸ਼ਾ peek_ਅੰਕੜੇ_shard(filename):
    # only reads the header, returns header data
    ਨਾਲ open(filename, "rb") ਇਵ f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        ਸਿਰਲੇਖ = np.frombuffer(f.ਪਠ(256*4), dtype=np.int32)
    ਜੇ ਸਿਰਲੇਖ[0] != 20240801:
        ਛਾਪੋ("ERROR: magic number mismatch in the data .bin file!")
        exit(1)
    ਪ੍ਰਤਿਜ੍ਞਾ ਸਿਰਲੇਖ[1] == 7, "unsupported version"
    ntok = ਸਿਰਲੇਖ[2] # number of tokens (claimed)
    ਵਾਪਸ ntok # for now just return the number of tokens

ਪਰਿਭਾਸ਼ਾ ਲੱਦੋ_ਅੰਕੜੇ_shard(filename):
    ਨਾਲ open(filename, "rb") ਇਵ f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        ਸਿਰਲੇਖ = np.frombuffer(f.ਪਠ(256*4), dtype=np.int32)
        ਪ੍ਰਤਿਜ੍ਞਾ ਸਿਰਲੇਖ[0] == 20240801, "magic number mismatch in the data .bin file"
        ਪ੍ਰਤਿਜ੍ਞਾ ਸਿਰਲੇਖ[1] == 7, "unsupported version"
        ntok = ਸਿਰਲੇਖ[2] # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        ਟੋਕਨ = np.frombuffer(f.ਪਠ(), dtype=np.uint32)
    ਪ੍ਰਤਿਜ੍ਞਾ ਲੰਬਾਈ(ਟੋਕਨ) == ntok, "number of tokens read does not match header?"
    ਵਾਪਸ ਟੋਕਨ

ਵਰਗ Distributed_Sharded_ਅੰਕੜੇ_Loader:
    """
    This DataLoader is both:
    - distributed (works correctly in case of multiple processes in DDP)
    - sharded (supports datasets that are broken up into multiple data shards)
    It is not *permuted*, meaning that it itearates over the data in the order
    of the dataset on disk, so the user should make sure to shuffle their examples
    during the creation of their data shards for best performance.
    """
    ਪਰਿਭਾਸ਼ਾ __init__(self, filename_ਨਮੂਨਾ, B, T, process_rank, ਅੰਕ_processes):
        self.process_rank = process_rank
        self.ਅੰਕ_processes = ਅੰਕ_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_ਨਮੂਨਾ))
        ਪ੍ਰਤਿਜ੍ਞਾ ਲੰਬਾਈ(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_ਕੁੱਲ = 0
        ਲਈ fname ਵਿੱਚ self.files:
            shard_ntok = peek_ਅੰਕੜੇ_shard(fname)
            ਪ੍ਰਤਿਜ੍ਞਾ shard_ntok >= ਅੰਕ_processes * B * T + 1
            ntok_ਕੁੱਲ += shard_ntok
        self.ntok_ਕੁੱਲ = ntok_ਕੁੱਲ
        print0(f"DataLoader: total number of tokens: {ntok_total:,} across {len(self.files)} files")

        # kick things off
        self.current_shard = ਕੁਝ_ਨਹੀਂ
        self.reset()

    ਪਰਿਭਾਸ਼ਾ reset(self):
        # we're being a bit clever here: if we already had shard 0 loaded,
        # then don't do the work to reload it, just reset the pointer
        ਜੇ self.current_shard != 0:
            self.current_shard = 0
            self.ਟੋਕਨ = ਲੱਦੋ_ਅੰਕੜੇ_shard(self.files[self.current_shard])
        self.current_position = self.process_rank * self.B * self.T

    ਪਰਿਭਾਸ਼ਾ advance(self): # advance to next data shard
        self.current_shard = (self.current_shard + 1) % ਲੰਬਾਈ(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.ਟੋਕਨ = ਲੱਦੋ_ਅੰਕੜੇ_shard(self.files[self.current_shard])

    ਪਰਿਭਾਸ਼ਾ ਅਗਲਾ_batch(self):
        B = self.B
        T = self.T
        buf = self.ਟੋਕਨ[self.current_position : self.current_position+B*T+1]
        buf = torch.tensor(buf, dtype=torch.ਲੰਮਾ)
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the start pointer in current shard
        self.current_position += B * T * self.ਅੰਕ_processes
        # if loading the next batch would be out of bounds advance the shard
        ਜੇ self.current_position + (B * T * self.ਅੰਕ_processes + 1) > ਲੰਬਾਈ(self.ਟੋਕਨ):
            self.advance()
        ਵਾਪਸ x, y

# -----------------------------------------------------------------------------
# Python -> C bridge utilities for saving params/grads/activations to .bin files

ਪਰਿਭਾਸ਼ਾ write_fp32(tensor, file):
    t = tensor.detach().cpu().ਮੰਜ਼ਿਲ(torch.float32)
    b = t.numpy().tobytes()
    file.ਲਿਖ(b)

ਪਰਿਭਾਸ਼ਾ write_bf16(tensor, file):
    t = tensor.detach().cpu().ਮੰਜ਼ਿਲ(torch.bfloat16)
    # numpy doesn't have bf16 datatype so we have to trick it
    t = t.view(torch.int16) # trick: reinterpret as int16
    b = t.numpy().tobytes()
    file.ਲਿਖ(b)

ਪਰਿਭਾਸ਼ਾ write_tensors(model_tensors, L, file, dtype):
    # writes LLaMA 3 model's weights to a binary file
    ਪ੍ਰਤਿਜ੍ਞਾ dtype ਵਿੱਚ {"float32", "bfloat16"}
    write_fun = write_fp32 ਜੇ dtype == "float32" ਨਹੀਂਤਾਂ write_bf16
    write_fun(model_tensors["transformer.wte.weight"], file) # (V, C)
    ਲਈ i ਵਿੱਚ ਸੀਮਾ(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_1.weight"], file)
    ਲਈ i ਵਿੱਚ ਸੀਮਾ(L): # (L, 3C, C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_attn.weight"], file)
    ਲਈ i ਵਿੱਚ ਸੀਮਾ(L): # (L, C, C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_proj.weight"], file)
    ਲਈ i ਵਿੱਚ ਸੀਮਾ(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_2.weight"], file)
    ਲਈ i ਵਿੱਚ ਸੀਮਾ(L): # (L, 4C, C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_fc.weight"], file)
    ਲਈ i ਵਿੱਚ ਸੀਮਾ(L): # (L, 4C, C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_fc2.weight"], file)
    ਲਈ i ਵਿੱਚ ਸੀਮਾ(L): # (L, C, 4C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_proj.weight"], file)
    write_fun(model_tensors["transformer.ln_f.weight"], file) # (C, )
    write_fun(model_tensors["lm_head.weight"], file) # (V, C)

ਪਰਿਭਾਸ਼ਾ write_model(model, filename, dtype):
    # everything we need to instantiate the model
    # 1) header is: version int, LLaMAConfig ints, padding to 1024 bytes
    ਪ੍ਰਤਿਜ੍ਞਾ dtype ਵਿੱਚ {"float32", "bfloat16"}
    ਸੰਸਕਰਣ = {
        "float32": 3, # 3: all tensors are fp32
        "bfloat16": 5, # 5: all tensors are bf16
    }[dtype]
    ਸਿਰਲੇਖ = torch.zeros(256, dtype=torch.int32)
    ਸਿਰਲੇਖ[0] = 20240803 # magic
    ਸਿਰਲੇਖ[1] = ਸੰਸਕਰਣ # checkpoint version
    ਸਿਰਲੇਖ[2] = model.config.ਖੰਡ_ਮਾਪ
    ਸਿਰਲੇਖ[3] = model.config.vocab_ਮਾਪ
    ਸਿਰਲੇਖ[4] = model.config.n_layer
    ਸਿਰਲੇਖ[5] = model.config.n_ਸਿਰ
    ਸਿਰਲੇਖ[6] = model.config.n_kv_ਸਿਰ
    ਸਿਰਲੇਖ[7] = model.config.n_embd
    ਸਿਰਲੇਖ[8] = model.config.ffn_dim_multiplier
    ਸਿਰਲੇਖ[9] = model.config.ਗੁਣਜ_ਦਾ
    ਸਿਰਲੇਖ[10] = model.config.norm_eps
    ਸਿਰਲੇਖ[11] = model.config.rope_theta
    ਸਿਰਲੇਖ[12] = model.config.use_scaled_rope
    ਸਿਰਲੇਖ[13] = model.config.max_gen_batch_ਮਾਪ
    ਸਿਰਲੇਖ[14] = ਪੂਰਨ(model.config.ਸੰਸਕਰਣ.ਵੰਡੋ('.')[0]) # major version
    ਸਿਰਲੇਖ[15] = ਪੂਰਨ(model.config.ਸੰਸਕਰਣ.ਵੰਡੋ('.')[1]) # minor version
    # 2) the parameters follow the header
    params = {name: param.cpu() ਲਈ name, param ਵਿੱਚ model.named_parameters()}
    # now write to file
    ਨਾਲ open(filename, "wb") ਇਵ file:
        file.ਲਿਖ(ਸਿਰਲੇਖ.numpy().tobytes()) #ਸਿਰਲੇਖ
        write_tensors(params, model.config.n_layer, file, dtype) # params
    ਛਾਪੋ(f"wrote {filename}")

ਪਰਿਭਾਸ਼ਾ write_state(model, x, y, logits, loss, filename):
    # the state is used for debugging.
    # it contains information about the input, logits, loss, and the parameter gradients
    # this can be used for checking the computation correctness in C
    ਸਿਰਲੇਖ = torch.zeros(256, dtype=torch.int32)
    ਸਿਰਲੇਖ[0] = 20240803 # magic
    ਸਿਰਲੇਖ[1] = x.ਮਾਪ(0) # batch size of the batch, B
    ਸਿਰਲੇਖ[2] = x.ਮਾਪ(1) # temporal extent of the batch, T
    grads = {name: param.grad.cpu() ਲਈ name, param ਵਿੱਚ model.named_parameters()}
    ਨਾਲ open(filename, "wb") ਇਵ file:
        #ਸਿਰਲੇਖ
        file.ਲਿਖ(ਸਿਰਲੇਖ.numpy().tobytes())
        # input x
        file.ਲਿਖ(x.cpu().numpy().astype("int32").tobytes()) # (B, T)
        # targets y
        file.ਲਿਖ(y.cpu().numpy().astype("int32").tobytes()) # (B, T)
        # logits (result of the model forward pass)
        write_fp32(logits.cpu(), file)
        # loss (single float, result of the cross entropy loss)
        write_fp32(loss.cpu(), file)
        # gradients
        write_tensors(grads, model.config.n_layer, file, "float32")
    ਛਾਪੋ(f"wrote {filename}")

# -----------------------------------------------------------------------------
# int main

ਪਰਿਭਾਸ਼ਾ print0(*ਦਲੀਲਾਂ, **kwargs):
    # modified print that only prints from the master process
    # if this is not a distributed run, it's just a print
    ਜੇ ਪੂਰਨ(os.environ.get("RANK", 0)) == 0:
        ਛਾਪੋ(*ਦਲੀਲਾਂ, **kwargs)

ਜੇ __name__ == "__main__":
    print0(f"Running pytorch {torch.version.__version__}")

    # default settings will overfit a tiny batch of data
    # and save model weights and debug state to disk on the first iteration
    parser = argparse.ਦਲੀਲ_Parser()
    parser.ਜੋੜੋ_ਦਲੀਲ("--use_hf", type=ਪੂਰਨ, ਮੂਲ=1, help="use HuggingFace (default) or use Meta's model")
    parser.ਜੋੜੋ_ਦਲੀਲ("--ckpt_dir", type=str, ਮੂਲ=ਕੁਝ_ਨਹੀਂ, help="path to llama3 model checkpoint (needed if use_hf=0)")
    parser.ਜੋੜੋ_ਦਲੀਲ("--tokenizer_path", type=str, ਮੂਲ=ਕੁਝ_ਨਹੀਂ, help="path to llama3 tokenizer (needed if use_hf=0)")
    # file system input / output
    parser.ਜੋੜੋ_ਦਲੀਲ("--input_bin", type=str, ਮੂਲ="dev/data/tinyshakespeare/tiny_shakespeare_val.bin", help="input .bin to train on")
    parser.ਜੋੜੋ_ਦਲੀਲ("--input_val_bin", type=str, ਮੂਲ="", help="input .bin to eval validation loss on")
    parser.ਜੋੜੋ_ਦਲੀਲ("--output_dir", type=str, ਮੂਲ="", help="output directory to which to write logs and checkpoints")
    parser.ਜੋੜੋ_ਦਲੀਲ("--model", type=str, ਮੂਲ="meta-llama/Meta-Llama-3.1-8B", help="chose the llama model")
    # token layout for each step of the optimization
    parser.ਜੋੜੋ_ਦਲੀਲ("--batch_size", type=ਪੂਰਨ, ਮੂਲ=4, help="batch size, in units of #batch dimensions")
    parser.ਜੋੜੋ_ਦਲੀਲ("--sequence_length", type=ਪੂਰਨ, ਮੂਲ=64, help="sequence length")
    parser.ਜੋੜੋ_ਦਲੀਲ("--total_batch_size", type=ਪੂਰਨ, ਮੂਲ=256, help="total desired batch size, in units of #tokens")
    # workload (number of steps)
    parser.ਜੋੜੋ_ਦਲੀਲ("--num_iterations", type=ਪੂਰਨ, ਮੂਲ=10, help="number of iterations to run")
    parser.ਜੋੜੋ_ਦਲੀਲ("--inference_only", type=ਪੂਰਨ, ਮੂਲ=0, help="only run inference")
    # optimization
    parser.ਜੋੜੋ_ਦਲੀਲ("--learning_rate", type=ਦਸ਼ਮਲਵ, ਮੂਲ=1e-5, help="learning rate warmup iterations")
    parser.ਜੋੜੋ_ਦਲੀਲ("--warmup_iters", type=ਪੂਰਨ, ਮੂਲ=0, help="learning rate warmup iterations")
    parser.ਜੋੜੋ_ਦਲੀਲ("--learning_rate_decay_frac", type=ਦਸ਼ਮਲਵ, ਮੂਲ=1.0, help="learning rate warmup iterations")
    parser.ਜੋੜੋ_ਦਲੀਲ("--weight_decay", type=ਦਸ਼ਮਲਵ, ਮੂਲ=0.0, help="weight decay")
    parser.ਜੋੜੋ_ਦਲੀਲ("--grad_clip", type=ਦਸ਼ਮਲਵ, ਮੂਲ=1.0, help="maximum gradient magnitude")
    # evaluation
    parser.ਜੋੜੋ_ਦਲੀਲ("--val_loss_every", type=ਪੂਰਨ, ਮੂਲ=0, help="every how mant steps to evaluate val loss?")
    parser.ਜੋੜੋ_ਦਲੀਲ("--val_max_steps", type=ਪੂਰਨ, ਮੂਲ=20, help="how many batches of val to average?")
    parser.ਜੋੜੋ_ਦਲੀਲ("--sample_every", type=ਪੂਰਨ, ਮੂਲ=0, help="how often to sample from the model?")
    # debugging
    parser.ਜੋੜੋ_ਦਲੀਲ("--overfit_single_batch", type=ਪੂਰਨ, ਮੂਲ=1, help="overfit just one batch of data")
    # numerics
    parser.ਜੋੜੋ_ਦਲੀਲ("--tensorcores", type=ਪੂਰਨ, ਮੂਲ=0, help="use tensorcores")
    # memory management
    parser.ਜੋੜੋ_ਦਲੀਲ("--device", type=str, ਮੂਲ="", help="by default we autodetect, or set it here")
    parser.ਜੋੜੋ_ਦਲੀਲ("--compile", type=ਪੂਰਨ, ਮੂਲ=0, help="torch.compile the model")
    parser.ਜੋੜੋ_ਦਲੀਲ("--dtype", type=str, ਮੂਲ="bfloat16", help="float32|float16|bfloat16")
    parser.ਜੋੜੋ_ਦਲੀਲ("--zero_stage", type=ਪੂਰਨ, ਮੂਲ=0, help="zero redundancy optimizer stage (0/1/2/3)")
    # python -> C bridge
    parser.ਜੋੜੋ_ਦਲੀਲ("--write_tensors", type=ਪੂਰਨ, ਮੂਲ=0, help="write tensors to disk")
    ਦਲੀਲਾਂ = parser.parse_ਦਲੀਲਾਂ()

    # args error checking and convenience variables
    B, T = ਦਲੀਲਾਂ.batch_ਮਾਪ, ਦਲੀਲਾਂ.sequence_ਲੰਮਾਈ
    ਪ੍ਰਤਿਜ੍ਞਾ 1 <= T <= 8192, "sequence length must be between 1 and 8192"
    ਪ੍ਰਤਿਜ੍ਞਾ ਦਲੀਲਾਂ.dtype ਵਿੱਚ {"float32", "float16", "bfloat16"}
    ਪ੍ਰਤਿਜ੍ਞਾ ਦਲੀਲਾਂ.model ਵਿੱਚ {"meta-llama/Meta-Llama-3.1-8B"}  # only 8B base model supported for now

    # create the logging directory if it does not exist
    logfile = ਕੁਝ_ਨਹੀਂ
    ਜੇ ਦਲੀਲਾਂ.output_dir:
        os.makedirs(ਦਲੀਲਾਂ.output_dir, exist_ok=ਸੱਚ)
        logfile = os.ਰਸਤਾ.join(ਦਲੀਲਾਂ.output_dir, "main.log")
        # create the log file "main.log" inside it, and wipe it clean
        ਨਾਲ open(logfile, "w") ਇਵ f:
            ਲੰਘੋ

    # set up DDP (distributed data parallel). torchrun sets this env variable
    ddp = ਪੂਰਨ(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    ਜੇ ddp:
        # use of DDP atm demands CUDA, we set the device appropriately according to rank
        ਪ੍ਰਤਿਜ੍ਞਾ torch.cuda.is_ਉਪਲਬਧ(), "for now i think we need CUDA for DDP"
        init_process_ਸਮੂਹ(backend='nccl')
        ddp_rank = ਪੂਰਨ(os.environ['RANK'])
        ddp_local_rank = ਪੂਰਨ(os.environ['LOCAL_RANK'])
        ddp_world_ਮਾਪ = ਪੂਰਨ(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.ਸੈੱਟ_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        ਬੀਜ_ਵਿੱਥ = 0 # each process gets the exact same seed
        zero_stage = ਦਲੀਲਾਂ.zero_stage
    ਨਹੀਂਤਾਂ:
        ddp_rank = 0
        ddp_local_rank = 0
        zero_stage = 0
        ddp_world_ਮਾਪ = 1
        master_process = ਸੱਚ
        ਬੀਜ_ਵਿੱਥ = 0
        # select the device
        ਜੇ ਦਲੀਲਾਂ.device:
            # provided explicitly by the user
            device = ਦਲੀਲਾਂ.device
        ਨਹੀਂਤਾਂ:
            # attempt to autodetect the device
            device = "cpu"
            ਜੇ torch.cuda.is_ਉਪਲਬਧ():
                device = "cuda"
            ਜਾਂਜੇ hasattr(torch.backends, "mps") ਅਤੇ torch.backends.mps.is_ਉਪਲਬਧ():
                device = "mps"
    device_type = 'cuda' ਜੇ 'cuda' ਵਿੱਚ device ਨਹੀਂਤਾਂ 'cpu'
    ਪ੍ਰਤਿਜ੍ਞਾ device_type ਵਿੱਚ {'cuda'}, "GPU required to run LLaMA 3"  # we need to load LLaMA as bf16 on CUDA
    ਛਾਪੋ(f"using device: {device}")

    # calculate gradient accumulation from the desired total batch size and the current run configuration
    ਟੋਕਨ_ਪ੍ਰਤੀ_fwdbwd = B * T * ddp_world_ਮਾਪ
    ਪ੍ਰਤਿਜ੍ਞਾ ਦਲੀਲਾਂ.ਕੁੱਲ_batch_ਮਾਪ % ਟੋਕਨ_ਪ੍ਰਤੀ_fwdbwd == 0
    grad_accum_ਕਦਮਾਂ = ਦਲੀਲਾਂ.ਕੁੱਲ_batch_ਮਾਪ // ਟੋਕਨ_ਪ੍ਰਤੀ_fwdbwd
    print0(f"total desired batch size: {args.total_batch_size}")
    print0(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    # set up a context manager following the desired dtype and device
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[ਦਲੀਲਾਂ.dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) ਜੇ (device_type == "cuda") ਨਹੀਂਤਾਂ nullcontext()

    # rng / reproducibility
    torch.manual_ਬੀਜ(42)
    ਜੇ torch.cuda.is_ਉਪਲਬਧ():
        torch.cuda.manual_ਬੀਜ(42)

    # set the torch precision mode to use TensorFloat32 (TF32) for matmuls
    # docs https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    ਜੇ ਦਲੀਲਾਂ.tensorcores:
        torch.ਸੈੱਟ_float32_matmul_precision('high')

    # init the model
    ਜੇ ਦਲੀਲਾਂ.use_hf:
        model = LLaMA.from_pretrained_llama3_hf(ਦਲੀਲਾਂ.model)
    ਨਹੀਂਤਾਂ:  # use Meta's checkpoint
        ਪ੍ਰਤਿਜ੍ਞਾ ਦਲੀਲਾਂ.ckpt_dir ਹੈ ਨਹੀਂ ਕੁਝ_ਨਹੀਂ ਅਤੇ os.ਰਸਤਾ.exists(ਦਲੀਲਾਂ.ckpt_dir), f"llama3 ckpt dir {args.ckpt_dir} does not exist"
        ਪ੍ਰਤਿਜ੍ਞਾ ਦਲੀਲਾਂ.tokenizer_ਰਸਤਾ ਹੈ ਨਹੀਂ ਕੁਝ_ਨਹੀਂ ਅਤੇ os.ਰਸਤਾ.exists(ਦਲੀਲਾਂ.tokenizer_ਰਸਤਾ), f"llama3 tokenizer path {args.tokenizer_path} does not exist"
        model = LLaMA.from_pretrained_llama3_meta(ਦਲੀਲਾਂ.ckpt_dir, ਦਲੀਲਾਂ.tokenizer_ਰਸਤਾ)

    model.train()
    ਜੇ ਦਲੀਲਾਂ.compile:
        ਜੇ hasattr(config, "coordinate_descent_tuning"):
            config.coordinate_descent_tuning = ਸੱਚ # suggested by @Chillee
        print0("compiling the model...")
        model = torch.compile(model)

    # -------------------------------------------------------------------------
    # Our own version of a simple DistributedDataLoader

    # load tokens
    train_loader = Distributed_Sharded_ਅੰਕੜੇ_Loader(ਦਲੀਲਾਂ.ਨਿਵੇਸ਼_bin, B, T, ddp_rank, ddp_world_ਮਾਪ)
    val_loader = ਕੁਝ_ਨਹੀਂ
    ਜੇ ਦਲੀਲਾਂ.ਨਿਵੇਸ਼_val_bin:
        val_loader = Distributed_Sharded_ਅੰਕੜੇ_Loader(ਦਲੀਲਾਂ.ਨਿਵੇਸ਼_val_bin, B, T, ddp_rank, ddp_world_ਮਾਪ)

    # -------------------------------------------------------------------------
    # PyTorch -> C bridge: save some weights and state for C to load later as reference

    # do one forward pass to generate ground truth for our C tests
    ਜੇ master_process ਅਤੇ ਦਲੀਲਾਂ.write_tensors ਅਤੇ (ਨਹੀਂ ਦਲੀਲਾਂ.inference_only):
        x, y = train_loader.ਅਗਲਾ_batch()
        x, y = x.ਮੰਜ਼ਿਲ(device), y.ਮੰਜ਼ਿਲ(device)
        logits, loss = model(x, y)
        loss.backward()
        # save model params, in bfloat16
        model_ਮੰਜ਼ਿਲ_ਮਾਪ = {"meta-llama/Meta-Llama-3.1-8B": "8B"}
        model_ਮਾਪ_str = model_ਮੰਜ਼ਿਲ_ਮਾਪ[ਦਲੀਲਾਂ.model] # e.g. "8B"
        write_model(model, os.ਰਸਤਾ.join(ਦਲੀਲਾਂ.output_dir, f"llama3.1_{model_size_str}_bf16.bin"), dtype="bfloat16")
        # save x, y, logits, loss, and parameter gradients, for debugging C
        # always store these in fp32 to have an accurate reference (?)
        write_state(model, x, y, logits, loss, os.ਰਸਤਾ.join(ਦਲੀਲਾਂ.output_dir, f"llama3_{model_size_str}_debug_state.bin"))
        # reset the train_loader for the optimization below
        train_loader.reset()

    # -------------------------------------------------------------------------
    # main training loop

    # here we wrap model into DDP container
    ਜੇ ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module ਜੇ ddp ਨਹੀਂਤਾਂ model # always contains the "raw" unwrapped model

    # init the optimizer
    optimizer = raw_model.configure_optimizers(ਭਾਰ_decay=ਦਲੀਲਾਂ.ਭਾਰ_decay,
                                               learning_ਦਰ=ਦਲੀਲਾਂ.learning_ਦਰ, betas=(0.9, 0.95),
                                               device_type=device, zero_stage=zero_stage)

    # learning rate decay scheduler (cosine with warmup)
    ਪਰਿਭਾਸ਼ਾ get_lr(it):
        min_lr = ਦਲੀਲਾਂ.learning_ਦਰ * ਦਲੀਲਾਂ.learning_ਦਰ_decay_frac
        # 1) linear warmup for warmup_iters steps
        ਜੇ it < ਦਲੀਲਾਂ.warmup_iters:
            ਵਾਪਸ ਦਲੀਲਾਂ.learning_ਦਰ * (it+1) / ਦਲੀਲਾਂ.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        ਜੇ it > ਦਲੀਲਾਂ.ਅੰਕ_iterations:
            ਵਾਪਸ min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ਅਨੁਪਾਤ = (it - ਦਲੀਲਾਂ.warmup_iters) / (ਦਲੀਲਾਂ.ਅੰਕ_iterations - ਦਲੀਲਾਂ.warmup_iters)
        ਪ੍ਰਤਿਜ੍ਞਾ 0 <= decay_ਅਨੁਪਾਤ <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ਅਨੁਪਾਤ)) # coeff starts at 1 and goes to 0
        ਵਾਪਸ min_lr + coeff * (ਦਲੀਲਾਂ.learning_ਦਰ - min_lr)

    ਜੇ device == "cuda":
        torch.cuda.reset_peak_ਸਮ੍ਰਿਤੀ_stats()
    timings = []
    norm = -1.0   # dummy value to print in inference-only mode
    ਲਈ ਕਦਮ ਵਿੱਚ ਸੀਮਾ(ਦਲੀਲਾਂ.ਅੰਕ_iterations + 1):
        t0 = time.time()
        ਆਖਰੀ_ਕਦਮ = (ਕਦਮ == ਦਲੀਲਾਂ.ਅੰਕ_iterations)

        # once in a while evaluate the validation dataset
        ਜੇ (ਦਲੀਲਾਂ.val_loss_every > 0 \
            ਅਤੇ (ਕਦਮ % ਦਲੀਲਾਂ.val_loss_every == 0 ਜਾਂ ਆਖਰੀ_ਕਦਮ)) \
            ਅਤੇ (val_loader ਹੈ ਨਹੀਂ ਕੁਝ_ਨਹੀਂ):
            model.eval()
            val_loader.reset()
            ਨਾਲ torch.no_grad():
                val_loss = 0.0
                ਲਈ _ ਵਿੱਚ ਸੀਮਾ(ਦਲੀਲਾਂ.val_max_ਕਦਮਾਂ):
                    x, y = val_loader.ਅਗਲਾ_batch()
                    x, y = x.ਮੰਜ਼ਿਲ(device), y.ਮੰਜ਼ਿਲ(device)
                    _, loss = model(x, y, return_logits=ਝੂਠ)
                    val_loss += loss.ਵਸਤੂ_2()
                val_loss /= ਦਲੀਲਾਂ.val_max_ਕਦਮਾਂ
            # log to console and to file
            print0(f"val loss {val_loss}")
            ਜੇ master_process ਅਤੇ logfile ਹੈ ਨਹੀਂ ਕੁਝ_ਨਹੀਂ:
                ਨਾਲ open(logfile, "a") ਇਵ f:
                    f.ਲਿਖ("s:%d tel:%f\n" % (ਕਦਮ, val_loss))

        # once in a while perform model inference on the master process
        ਜੇ (ਦਲੀਲਾਂ.sample_every > 0 \
            ਅਤੇ (ਕਦਮ % ਦਲੀਲਾਂ.sample_every == 0 ਜਾਂ ਆਖਰੀ_ਕਦਮ)) \
            ਅਤੇ master_process:
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
            ਜੇ ਦਲੀਲਾਂ.use_hf:
                prompt_ਟੋਕਨ = [model.tokenizer(x).ਨਿਵੇਸ਼_ids ਲਈ x ਵਿੱਚ prompts]
            ਨਹੀਂਤਾਂ:  # Meta
                prompt_ਟੋਕਨ = [model.tokenizer.encode(x, bos=ਸੱਚ, eos=ਝੂਠ) ਲਈ x ਵਿੱਚ prompts]

            generation_ਟੋਕਨ = model.generate(prompt_ਟੋਕਨ, max_gen_len=64, temperature=0.6, ਸਿਖਰ_p=0.9, echo=ਝੂਠ)
            results = [{"generation": model.tokenizer.decode(t)} ਲਈ t ਵਿੱਚ generation_ਟੋਕਨ]
            ਲਈ prompt, ਨਤੀਜਾ ਵਿੱਚ zip(prompts, results):
                ਛਾਪੋ(prompt, ਸਿਰਾ="")
                ਛਾਪੋ(f"{result['generation']}")
                ਛਾਪੋ("\n==================================\n")

        # bit confusing: we want to make sure to eval and sample on 0th iteration
        # but also after the very last iteration. so we loop for step <= num_iterations
        # instead of just < num_iterations (one extra due to <=), only to do
        # the validation/sampling one last time, and then we break right here as we're done.
        ਜੇ ਆਖਰੀ_ਕਦਮ:
            ਰੋਕੋ

        # --------------- TRAINING SECTION BEGIN -----------------
        model.train()
        optimizer.zero_grad(ਸੈੱਟ_ਮੰਜ਼ਿਲ_none=ਸੱਚ)
        # if we are trying to overfit a single batch, we reset the loader here
        ਜੇ ਦਲੀਲਾਂ.overfit_single_batch:
            train_loader.reset()
        # micro-batch loop where we do gradient accumulation to reach desired total batch size
        lossf = 0.0 # for getting the mean loss (as simple float) over the accumulation steps
        ਲਈ micro_ਕਦਮ ਵਿੱਚ ਸੀਮਾ(grad_accum_ਕਦਮਾਂ):
            # fetch a batch
            x, y = train_loader.ਅਗਲਾ_batch()
            x, y = x.ਮੰਜ਼ਿਲ(device), y.ਮੰਜ਼ਿਲ(device)
            ਜੇ ddp:
                # we want only the last micro-step to sync grads in a DDP model
                # the official way to do this is with model.no_sync(), but that is a
                # context manager that bloats the code, so we just toggle this variable
                model.require_backward_grad_sync = (micro_ਕਦਮ == grad_accum_ਕਦਮਾਂ - 1)
            # forward pass
            ਨਾਲ ctx:
                _, loss = model(x, y, return_logits=ਝੂਠ)
                # we have to scale the loss to account for gradient accumulation,
                # because the gradients just add on each successive backward().
                # addition of gradients corresponds to a SUM in the objective, but
                # instead of a SUM we want MEAN, so we scale the loss here
                loss = loss / grad_accum_ਕਦਮਾਂ
                lossf += loss.detach() # keep track of the mean loss
            # backward pass
            ਜੇ ਨਹੀਂ ਦਲੀਲਾਂ.inference_only:
                loss.backward()
        ਜੇ ddp:
            dist.all_reduce(lossf, op=dist.ReduceOp.AVG)
        lossf = lossf.ਵਸਤੂ_2()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), ਦਲੀਲਾਂ.grad_clip)
        # determine and set the learning rate for this iteration
        lr = get_lr(ਕਦਮ)
        ਲਈ param_ਸਮੂਹ ਵਿੱਚ optimizer.param_groups:
            param_ਸਮੂਹ['lr'] = lr
        # step the optimizer
        optimizer.ਕਦਮ()
        # --------------- TRAINING SECTION END -------------------
        # everything that follows now is just diagnostics, prints, logging, etc.

        # wait on the CPU for all device work to end so we get accurate per-iteration timings below
        ਜੇ device == "mps":
            torch.mps.synchronize()
        ਜਾਂਜੇ device == "cuda":
            torch.cuda.synchronize()
        # time and print
        t1 = time.time()
        # the 0th iteration is often an outlier (much slower) => skip logging it
        ਟੋਕਨ_ਪ੍ਰਤੀ_ਦੂਜਾ = grad_accum_ਕਦਮਾਂ * ddp_world_ਮਾਪ * B * T / (t1-t0)
        print0(f"step {step+1:4d}/{args.num_iterations} | train loss {lossf:.6f} | norm {norm:.4f} | lr {lr:.2e} | ({(t1-t0)*1000:.2f} ms | {tokens_per_second:.0f} tok/s)")
        # log to logile
        ਜੇ master_process ਅਤੇ logfile ਹੈ ਨਹੀਂ ਕੁਝ_ਨਹੀਂ:
            ਨਾਲ open(logfile, "a") ਇਵ f:
                f.ਲਿਖ("s:%d trl:%f\n" % (ਕਦਮ, lossf))

        # keep track of smooth timings, last 20 iterations
        ਜੇ ਕਦਮ > 0 ਅਤੇ ਕਦਮ > ਦਲੀਲਾਂ.ਅੰਕ_iterations - 20:
            timings.append(t1-t0)

    # print the average of the last 20 timings, to get something smooth-ish
    timings = timings[-20:]
    print0(f"final {len(timings)} iters avg: {np.mean(timings)*1000:.3f}ms")
    print0(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

    # -------------------------------------------------------------------------
    # clean up nice
    ਜੇ ddp:
        destroy_process_ਸਮੂਹ()
