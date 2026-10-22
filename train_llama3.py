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

ಆನಯನ argparse
ಆನಯನ os
ಆನಯನ math
ಆನಯನ glob
ಆನಯನ inspect
ತಃ contextlib ಆನಯನ nullcontext
ತಃ dataclasses ಆನಯನ dataclass
ತಃ pathlib ಆನಯನ Path
ಆನಯನ time
ತಃ typing ಆನಯನ (
    AbstractSet,
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

ಆನಯನ numpy ಇವ np
ಆನಯನ torch
ಆನಯನ torch.nn ಇವ nn
ತಃ torch.nn ಆನಯನ functional ಇವ F
ಆನಯನ torch._inductor.config ಇವ config
ತಃ torch.nn.parallel ಆನಯನ DistributedDataParallel ಇವ DDP
ತಃ torch.distributed ಆನಯನ init_process_group, destroy_process_group
ತಃ torch.distributed.optim ಆನಯನ ZeroRedundancyOptimizer
ಆನಯನ torch.distributed ಇವ dist

ಆನಯನ tiktoken
ತಃ tiktoken.load ಆನಯನ load_tiktoken_bpe

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the LLaMA 3.x model

# Used in Grouped Query Attention (GQA), broadcasts the key and value tensors
def repeat_kv(x: torch.Tensor, n_rep: ಪೂರ್ಣಾಙ್ಕ) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    ಯದಿ n_rep == 1:
        ಪ್ರತ್ಯಾಗಮ x
    ಪ್ರತ್ಯಾಗಮ (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

# -----------------------------------------------------------------------------
# RoPE related

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    ಪ್ರತಿಜ್ಞಾ 0 <= 1 < ndim
    ಪ್ರತಿಜ್ಞಾ freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d ಯದಿ i == 1 ವಾ i == ndim - 1 ಅನ್ಯಥಾ 1 ಪ್ರತಿ i, d ಅನ್ತಃ enumerate(x.shape)]
    ಪ್ರತ್ಯಾಗಮ freqs_cis.view(*shape)

def apply_scaling(freqs: torch.Tensor):
    # Values obtained from grid search
    scale_factor = 8
    ಕೆಳಗಿನ_freq_factor = 1
    ಮೇಲಿನ_freq_factor = 4
    old_context_len = 8192  # original llama3 length

    ಕೆಳಗಿನ_freq_wavelen = old_context_len / ಕೆಳಗಿನ_freq_factor
    ಮೇಲಿನ_freq_wavelen = old_context_len / ಮೇಲಿನ_freq_factor
    new_freqs = []
    ಪ್ರತಿ freq ಅನ್ತಃ freqs:
        wavelen = 2 * math.pi / freq
        ಯದಿ wavelen < ಮೇಲಿನ_freq_wavelen:
            new_freqs.append(freq)
        ನೋಚೇದ್ಯದಿ wavelen > ಕೆಳಗಿನ_freq_wavelen:
            new_freqs.append(freq / scale_factor)
        ಅನ್ಯಥಾ:
            ಪ್ರತಿಜ್ಞಾ ಕೆಳಗಿನ_freq_wavelen != ಮೇಲಿನ_freq_wavelen
            smooth = (old_context_len / wavelen - ಕೆಳಗಿನ_freq_factor) / (
                ಮೇಲಿನ_freq_factor - ಕೆಳಗಿನ_freq_factor
            )
            new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
    ಪ್ರತ್ಯಾಗಮ torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.ಭಿನ್ನ().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.ಭಿನ್ನ().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    ಪ್ರತ್ಯಾಗಮ xq_out.type_as(xq), xk_out.type_as(xk)

def precompute_freqs_cis(
    dim: ಪೂರ್ಣಾಙ್ಕ, end: ಪೂರ್ಣಾಙ್ಕ, theta: ಭಿನ್ನ = 10000.0, use_scaled: bool = ಅಸತ್ಯ
):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].ಭಿನ್ನ() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    ಯದಿ use_scaled:
        freqs = apply_scaling(freqs)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    ಪ್ರತ್ಯಾಗಮ freqs_cis

# -----------------------------------------------------------------------------
# LLaMA building blocks

# LLaMA reference code explicitly implemented RMSNorm so we copy pasted it
# (https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/api/model.py)
# we could also use nn.RMSNorm, it has slightly different numeric properties, but equivalent
ವರ್ಗ RMSNorm(torch.nn.Module):
    def __init__(self, dim: ಪೂರ್ಣಾಙ್ಕ, eps: ಭಿನ್ನ = 1e-6):
        ಪರ().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        ಪ್ರತ್ಯಾಗಮ x * torch.rsqrt(x.pow(2).mean(-1, keepdim=ಸತ್ಯ) + self.eps)

    def forward(self, x):
        output = self._norm(x.ಭಿನ್ನ()).type_as(x)
        ಪ್ರತ್ಯಾಗಮ output * self.weight

ವರ್ಗ CausalSelfAttention(nn.Module):

    def __init__(self, config):
        ಪರ().__init__()
        ಪ್ರತಿಜ್ಞಾ config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_rep = self.n_head // self.n_kv_head
        self.hd = config.n_embd // config.n_head
        self.use_kv = config.use_kv
        self.flash = config.flash

        self.c_attn = nn.Linear(config.n_embd, (config.n_head + 2 * config.n_kv_head) * self.hd, bias=ಅಸತ್ಯ)  # key, query, value projections
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=ಅಸತ್ಯ)  # output projection

        # static KV cache - we could alternatively allocate it outside of the model and just pass it in when needed
        ಯದಿ self.use_kv:
            self.cache_k = torch.zeros((config.max_gen_batch_size, config.block_size, config.n_kv_head, self.hd))
            self.cache_v = torch.zeros((config.max_gen_batch_size, config.block_size, config.n_kv_head, self.hd))

    def forward(self, x, freqs_cis=None, start_pos=None, mask=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split([self.n_head * self.hd, self.n_kv_head * self.hd, self.n_kv_head * self.hd], dim=-1)
        q, k, v = map(ಅನಾಮನ್ t: t.view(B, T, -1, self.hd), (q, k, v))  # (B, T, NH, HD)

        q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)  # rotate QK (rope)  <-- 1. difference compared to GPT-2

        ಯದಿ self.use_kv ಚ ನ self.training ಚ start_pos >= 0:  # use kv-caching during inference
            self.cache_k[:B, start_pos : start_pos + T] = k
            self.cache_v[:B, start_pos : start_pos + T] = v
            k = self.cache_k[:B, : start_pos + T]
            v = self.cache_v[:B, : start_pos + T]

        k = repeat_kv(k, self.n_rep)  # GQA <-- 2. difference compared to GPT-2
        v = repeat_kv(v, self.n_rep)

        q, k, v = map(ಅನಾಮನ್ t: t.transpose(1, 2), (q, k, v))  # (B, NH, T, HD)

        ಯದಿ self.flash:
            # flashattention
            # if T == 1 no need to mask, otherwise the function complains
            # scaled_dot_product_attention expects a mask where value of True indicates that the element should take part in attention
            # our mask is the opposite, so we need to invert it
            y = F.scaled_dot_product_attention(q, k, v, mask == 0 ಯದಿ T > 1 ಅನ್ಯಥಾ None)
        ಅನ್ಯಥಾ:
            # manual implementation of attention
            # this materializes the large (T,T) matrix for all the queries and keys
            scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.hd))
            ಯದಿ mask is ನ None:
                scores.masked_fill_(mask, torch.finfo(scores.dtype).min)
            att = F.softmax(scores.ಭಿನ್ನ(), dim=-1).type_as(q)
            y = att @ v # (B, NH, T, T) x (B, NH, T, HD) -> (B, NH, T, HD)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        ಪ್ರತ್ಯಾಗಮ y

ವರ್ಗ MLP(nn.Module):

    def __init__(self, config):
        ಪರ().__init__()
        hidden_dim = 4 * config.n_embd
        hidden_dim = ಪೂರ್ಣಾಙ್ಕ(2 * hidden_dim / 3)
        # custom dim factor multiplier
        ಯದಿ config.ffn_dim_multiplier is ನ None:
            hidden_dim = ಪೂರ್ಣಾಙ್ಕ(config.ffn_dim_multiplier * hidden_dim)
        hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)
        self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=ಅಸತ್ಯ)
        self.c_fc2 = nn.Linear(config.n_embd, hidden_dim, bias=ಅಸತ್ಯ)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=ಅಸತ್ಯ)

    def forward(self, x):
        # SwiGLU self.c_proj(F.silu(self.c_fc2(x)) * self.c_fc(x))  <-- 3. difference compared to GPT-2
        x1 = self.c_fc(x)
        x2 = self.c_fc2(x)
        x2 = F.silu(x2)
        x = x1 * x2
        x = self.c_proj(x)
        ಪ್ರತ್ಯಾಗಮ x

ವರ್ಗ Block(nn.Module):

    def __init__(self, config):
        ಪರ().__init__()
        self.ln_1 = RMSNorm(config.n_embd, config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd, config.norm_eps)
        self.mlp = MLP(config)

    def forward(self, x, freqs_cis=None, start_pos=None, mask=None):
        x = x + self.attn(self.ln_1(x), freqs_cis, start_pos, mask)
        x = x + self.mlp(self.ln_2(x))
        ಪ್ರತ್ಯಾಗಮ x

# -----------------------------------------------------------------------------
# The main LLaMA 3.1 model

@dataclass
ವರ್ಗ LlamaConfig:
    version: str = "3.1"
    block_size: ಪೂರ್ಣಾಙ್ಕ = 8192
    vocab_size: ಪೂರ್ಣಾಙ್ಕ = 128256
    n_layer: ಪೂರ್ಣಾಙ್ಕ = 32
    n_head: ಪೂರ್ಣಾಙ್ಕ = 32
    n_kv_head: ಪೂರ್ಣಾಙ್ಕ = 8
    n_embd: ಪೂರ್ಣಾಙ್ಕ = 4096
    ffn_dim_multiplier: ಭಿನ್ನ = 1.3
    multiple_of: ಪೂರ್ಣಾಙ್ಕ = 1024
    norm_eps: ಭಿನ್ನ = 1e-5
    rope_theta: ಭಿನ್ನ = 500000.0
    use_scaled_rope: bool = ಸತ್ಯ
    max_gen_batch_size: ಪೂರ್ಣಾಙ್ಕ = 4
    use_kv: bool = ಸತ್ಯ
    flash: bool = ಅಸತ್ಯ  # use flashattention?

    def __init__(self, **kwargs):
        ಪ್ರತಿ k, v ಅನ್ತಃ kwargs.ವಸ್ತುಗಳು():
            ಯದಿ hasattr(self, k):
                setattr(self, k, v)
        ಪ್ರತಿಜ್ಞಾ self.n_kv_head <= self.n_head
        ಪ್ರತಿಜ್ಞಾ self.n_head % self.n_kv_head == 0
        ಪ್ರತಿಜ್ಞಾ self.n_embd % self.n_head == 0

ವರ್ಗ LLaMA(nn.Module):

    def __init__(self, config):
        ಪರ().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) ಪ್ರತಿ _ ಅನ್ತಃ range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd, config.norm_eps),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=ಅಸತ್ಯ)

        # init all weights, use a torch rng object to be very careful
        self.init_rng = torch.Generator()
        self.init_rng.manual_seed(42)

        self.freqs_cis = precompute_freqs_cis(
            config.n_embd // config.n_head,
            config.block_size * 2,
            config.rope_theta,
            config.use_scaled_rope,
        )

    def forward(self, idx, targets=None, return_logits=ಸತ್ಯ, start_pos=0):
        _, t = idx.size()
        ಪ್ರತಿಜ್ಞಾ t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # forward the LLaMA model itself
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        freqs_cis = self.freqs_cis[start_pos:start_pos+t]

        mask = torch.triu(torch.ones((t, t), device=next(self.parameters()).device, dtype=torch.bool), diagonal=1)

        ಪ್ರತಿ i, block ಅನ್ತಃ enumerate(self.transformer.h):
            x = block(x, freqs_cis, start_pos, mask)
        x = self.transformer.ln_f(x)

        ಯದಿ targets is ನ None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x).ಭಿನ್ನ()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        ಅನ್ಯಥಾ:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]).ಭಿನ್ನ() # note: using list [-1] to preserve the time dim
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        ಯದಿ ನ return_logits:
            logits = None

        ಪ್ರತ್ಯಾಗಮ logits, loss

    @staticmethod
    def adapt_llama_state_dict_ಕೀಲಿಗಳು(checkpoint, config: LlamaConfig):
        # Modify key names from Meta's LLaMA to our LLaMA
        # our key names are derived from GPT-2's key names
        checkpoint['transformer.wte.weight'] = checkpoint.ತೆಗೆ('tok_embeddings.weight')

        ಪ್ರತಿ i ಅನ್ತಃ range(config.n_layer):
            ಪ್ರತಿ name ಅನ್ತಃ ['attention_norm', 'ffn_norm']:
                old_ಕೀಲಿ = f'layers.{i}.{name}.weight'  # e.g. layers.x.attention_norm.weight -> transformer.h.x.ln_1.weight
                new_ಕೀಲಿ = f'transformer.h.{i}.ln_{1 if name == "attention_norm" else 2}.weight'
                checkpoint[new_ಕೀಲಿ] = checkpoint.ತೆಗೆ(old_ಕೀಲಿ)

        ಪ್ರತಿ i ಅನ್ತಃ range(config.n_layer):
            ಪ್ರತಿ name ಅನ್ತಃ ['attention.wq', 'attention.wk', 'attention.wv']:
                old_ಕೀಲಿ = f'layers.{i}.{name}.weight'
                new_ಕೀಲಿ = f'transformer.h.{i}.attn.c_attn.weight'
                ಯದಿ name == 'attention.wq':
                    checkpoint[new_ಕೀಲಿ] = checkpoint.ತೆಗೆ(old_ಕೀಲಿ)
                ಅನ್ಯಥಾ:  # merge 3 weights into transformer.h.x.attn.c_attn.weight
                    checkpoint[new_ಕೀಲಿ] = torch.cat((checkpoint[new_ಕೀಲಿ], checkpoint.ತೆಗೆ(old_ಕೀಲಿ)), dim=0)
            old_ಕೀಲಿ = f'layers.{i}.attention.wo.weight'
            new_ಕೀಲಿ = f'transformer.h.{i}.attn.c_proj.weight'
            checkpoint[new_ಕೀಲಿ] = checkpoint.ತೆಗೆ(old_ಕೀಲಿ)

        ffn_map = {'w1': 'c_fc2', 'w2': 'c_proj', 'w3': 'c_fc'}
        ಪ್ರತಿ i ಅನ್ತಃ range(config.n_layer):
            ಪ್ರತಿ name ಅನ್ತಃ ['feed_forward.w1', 'feed_forward.w2', 'feed_forward.w3']:
                old_ಕೀಲಿ = f'layers.{i}.{name}.weight'
                new_ಕೀಲಿ = f'transformer.h.{i}.mlp.{ffn_map[name.split(".")[-1]]}.weight'
                checkpoint[new_ಕೀಲಿ] = checkpoint.ತೆಗೆ(old_ಕೀಲಿ)

        checkpoint['transformer.ln_f.weight'] = checkpoint.ತೆಗೆ('norm.weight')
        checkpoint['lm_head.weight'] = checkpoint.ತೆಗೆ('output.weight')

        ಪ್ರತ್ಯಾಗಮ checkpoint

    @staticmethod
    def adapt_llama_state_dict_ಕೀಲಿಗಳು_hf(checkpoint, config: LlamaConfig):
        # Modify key names from HuggingFace's LLaMA to our LLaMA
        # our key names are derived from GPT-2's key names
        checkpoint['transformer.wte.weight'] = checkpoint.ತೆಗೆ('model.embed_tokens.weight')

        # We need to unpermute K and V because HF script permuted the original Meta-LLaMA weights
        # see: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py
        def unpermute(w, n_heads, dim1, dim2):
            ಪ್ರತ್ಯಾಗಮ w.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)

        ಪ್ರತಿ i ಅನ್ತಃ range(config.n_layer):
            ಪ್ರತಿ name ಅನ್ತಃ ['input_layernorm', 'post_attention_layernorm']:
                old_ಕೀಲಿ = f'model.layers.{i}.{name}.weight'  # e.g. layers.x.attention_norm.weight -> transformer.h.x.ln_1.weight
                new_ಕೀಲಿ = f'transformer.h.{i}.ln_{1 if name == "input_layernorm" else 2}.weight'
                checkpoint[new_ಕೀಲಿ] = checkpoint.ತೆಗೆ(old_ಕೀಲಿ)

        ಪ್ರತಿ i ಅನ್ತಃ range(config.n_layer):
            ಪ್ರತಿ name ಅನ್ತಃ ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj']:
                old_ಕೀಲಿ = f'model.layers.{i}.{name}.weight'
                new_ಕೀಲಿ = f'transformer.h.{i}.attn.c_attn.weight'
                ಯದಿ name == 'self_attn.q_proj':
                    checkpoint[new_ಕೀಲಿ] = unpermute(checkpoint.ತೆಗೆ(old_ಕೀಲಿ), config.n_head, config.n_embd, config.n_embd)
                ಅನ್ಯಥಾ:  # merge 3 weights into transformer.h.x.attn.c_attn.weight
                    tensor = checkpoint.ತೆಗೆ(old_ಕೀಲಿ)
                    ಯದಿ name == 'self_attn.k_proj':
                        tensor = unpermute(tensor, config.n_kv_head, config.n_kv_head * (config.n_embd // config.n_head), config.n_embd)
                    checkpoint[new_ಕೀಲಿ] = torch.cat((checkpoint[new_ಕೀಲಿ], tensor), dim=0)
            old_ಕೀಲಿ = f'model.layers.{i}.self_attn.o_proj.weight'
            new_ಕೀಲಿ = f'transformer.h.{i}.attn.c_proj.weight'
            checkpoint[new_ಕೀಲಿ] = checkpoint.ತೆಗೆ(old_ಕೀಲಿ)

        ffn_map = {'gate_proj': 'c_fc2', 'down_proj': 'c_proj', 'up_proj': 'c_fc'}
        ಪ್ರತಿ i ಅನ್ತಃ range(config.n_layer):
            ಪ್ರತಿ name ಅನ್ತಃ ['gate_proj', 'down_proj', 'up_proj']:
                old_ಕೀಲಿ = f'model.layers.{i}.mlp.{name}.weight'
                new_ಕೀಲಿ = f'transformer.h.{i}.mlp.{ffn_map[name]}.weight'
                checkpoint[new_ಕೀಲಿ] = checkpoint.ತೆಗೆ(old_ಕೀಲಿ)

        checkpoint['transformer.ln_f.weight'] = checkpoint.ತೆಗೆ('model.norm.weight')

        ಪ್ರತ್ಯಾಗಮ checkpoint

    @classmethod
    def from_pretrained_llama3_hf(cls, model_id):
        """Loads pretrained LLaMA model weights from HuggingFace"""
        ತಃ transformers ಆನಯನ AutoModelForCausalLM, AutoTokenizer
        ಪ್ರತಿಜ್ಞಾ model_id == "meta-llama/Meta-Llama-3.1-8B", "Only the 8B-base model is supported for now"
        model_args = LlamaConfig()

        model = AutoModelForCausalLM.from_pretrained(model_id)
        checkpoint = LLaMA.adapt_llama_state_dict_ಕೀಲಿಗಳು_hf(model.state_dict(), model_args)

        original_default_type = torch.get_default_dtype()  # save the default type
        torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)  # much faster loading
        model = LLaMA(model_args)
        model.load_state_dict(checkpoint, strict=ಅಸತ್ಯ)
        torch.set_default_tensor_type(torch.tensor([], dtype=original_default_type, device="cpu").type())  # restore default type

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_id = 128004  # this is the pad token id for LLaMA 3.1 base, we need to set this explicitly as our generate func expects it
        tokenizer.stop_tokens = [tokenizer.eos_token_id]
        model.tokenizer = tokenizer
        ಪ್ರತ್ಯಾಗಮ model

    @classmethod
    def from_pretrained_llama3_meta(cls, ckpt_dir, tokenizer_path):
        """Loads pretrained LLaMA model weights from a checkpoint directory"""
        model_args = LlamaConfig()

        ckpt_path = sorted(Path(ckpt_dir).glob("*.pth"))[0]
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=ಸತ್ಯ)
        checkpoint = LLaMA.adapt_llama_state_dict_ಕೀಲಿಗಳು(checkpoint, model_args)

        original_default_type = torch.get_default_dtype()  # save the default type
        torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)  # much faster loading
        model = LLaMA(model_args)
        model.load_state_dict(checkpoint, strict=ಅಸತ್ಯ)
        torch.set_default_tensor_type(torch.tensor([], dtype=original_default_type, device="cpu").type())  # restore default type

        tokenizer = Tokenizer(model_path=tokenizer_path)
        model.tokenizer = tokenizer
        ಪ್ರತ್ಯಾಗಮ model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, zero_stage):
        # start with all of the candidate parameters
        param_dict = {pn: p ಪ್ರತಿ pn, p ಅನ್ತಃ self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p ಪ್ರತಿ pn, p ಅನ್ತಃ param_dict.ವಸ್ತುಗಳು() ಯದಿ p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p ಪ್ರತಿ n, p ಅನ್ತಃ param_dict.ವಸ್ತುಗಳು() ಯದಿ p.dim() >= 2]
        nodecay_params = [p ಪ್ರತಿ n, p ಅನ್ತಃ param_dict.ವಸ್ತುಗಳು() ಯದಿ p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        ಅಂಕೆ_decay_params = sum(p.numel() ಪ್ರತಿ p ಅನ್ತಃ decay_params)
        ಅಂಕೆ_nodecay_params = sum(p.numel() ಪ್ರತಿ p ಅನ್ತಃ nodecay_params)
        print0(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print0(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' ಅನ್ತಃ inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available ಚ device_type == 'cuda'
        print0(f"using fused AdamW: {use_fused}")
        ಯದಿ zero_stage == 1:
            print0("using ZeroRedundancyOptimizer")
            optimizer = ZeroRedundancyOptimizer(**optim_groups[0], optimizer_class=torch.optim.AdamW,
                                                lr=learning_rate, betas=betas, fused=use_fused)
            optimizer.add_param_group(optim_groups[1])
        ಅನ್ಯಥಾ:
            print0("using regular AdamW")
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=use_fused)
        ಪ್ರತ್ಯಾಗಮ optimizer

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[List[ಪೂರ್ಣಾಙ್ಕ]],
        max_gen_len: ಪೂರ್ಣಾಙ್ಕ,
        temperature: ಭಿನ್ನ = 0.6,
        ತುದಿ_p: ಭಿನ್ನ = 0.9,
        echo: bool = ಅಸತ್ಯ,
    ) -> Tuple[List[List[ಪೂರ್ಣಾಙ್ಕ]], Optional[List[List[ಭಿನ್ನ]]]]:
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
        bsz = len(prompt_tokens)
        ಪ್ರತಿಜ್ಞಾ bsz <= self.config.max_gen_batch_size, f"Batch size {bsz} exceeds the maximum generation batch size {self.config.max_gen_batch_size}"
        device = next(self.parameters()).device

        min_prompt_len = min(len(t) ಪ್ರತಿ t ಅನ್ತಃ prompt_tokens)
        max_prompt_len = max(len(t) ಪ್ರತಿ t ಅನ್ತಃ prompt_tokens)
        ಪ್ರತಿಜ್ಞಾ max_prompt_len <= self.config.block_size, f"Prompt length {max_prompt_len} exceeds the maximum block size {self.config.block_size}"
        total_len = min(self.config.block_size, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device=device)
        ಪ್ರತಿ idx, t ಅನ್ತಃ enumerate(prompt_tokens):
            tokens[idx, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)

        prev_pos = 0
        eos_reached = torch.tensor([ಅಸತ್ಯ] * bsz, device=device)
        ಒಳಹರಿವು_text_mask = tokens != pad_id

        ಯದಿ min_prompt_len == total_len:
            logits, _ = self.forward(tokens, start_pos=prev_pos)

        stop_tokens = torch.tensor(list(self.tokenizer.stop_tokens)).to(device)

        ಪ್ರತಿ cur_pos ಅನ್ತಃ range(min_prompt_len, total_len):
            logits, _ = self.forward(tokens[:, prev_pos:cur_pos], start_pos=prev_pos)
            ಯದಿ temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_ತುದಿ_p(probs, ತುದಿ_p)
            ಅನ್ಯಥಾ:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(ಒಳಹರಿವು_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token
            eos_reached |= ~ಒಳಹರಿವು_text_mask[:, cur_pos] & torch.isin(next_token, stop_tokens)
            prev_pos = cur_pos
            ಯದಿ all(eos_reached):
                ಭಙ್ಗ

        out_tokens = []
        ಪ್ರತಿ i, toks ಅನ್ತಃ enumerate(tokens.tolist()):
            # cut to max gen len
            start = 0 ಯದಿ echo ಅನ್ಯಥಾ len(prompt_tokens[i])
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
            # cut to after eos tok if any
            ಪ್ರತಿ stop_token ಅನ್ತಃ self.tokenizer.stop_tokens:
                ಯತ್ನ:
                    eos_idx = toks.index(stop_token)
                    toks = toks[:eos_idx]
                ಅಪವಾದ ಮೌಲ್ಯ_Error:
                    pass
            out_tokens.append(toks)
        ಪ್ರತ್ಯಾಗಮ out_tokens

# -----------------------------------------------------------------------------
# sampling utils

def sample_ತುದಿ_p(probs, p):
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
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=ಸತ್ಯ)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=ಸತ್ಯ))
    next_token = torch.multinomial(probs_sort, ಅಂಕೆ_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    ಪ್ರತ್ಯಾಗಮ next_token

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

    special_tokens: Dict[str, ಪೂರ್ಣಾಙ್ಕ]

    ಅಂಕೆ_reserved_special_tokens = 256

    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: E501

    def __init__(self, model_path: str):
        """
        Initializes the Tokenizer with a Tiktoken model.

        Args:
            model_path (str): The path to the Tiktoken model file.
        """
        ಪ್ರತಿಜ್ಞಾ os.path.isfile(model_path), model_path

        mergeable_ranks = load_tiktoken_bpe(model_path)
        ಅಂಕೆ_base_tokens = len(mergeable_ranks)
        special_tokens = [
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
        reserved_tokens = [
            f"<|reserved_special_token_{2 + i}|>"
            ಪ್ರತಿ i ಅನ್ತಃ range(self.ಅಂಕೆ_reserved_special_tokens - len(special_tokens))
        ]
        special_tokens = special_tokens + reserved_tokens

        self.special_tokens = {
            token: ಅಂಕೆ_base_tokens + i ಪ್ರತಿ i, token ಅನ್ತಃ enumerate(special_tokens)
        }
        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )

        self.n_ಪದಗಳು: ಪೂರ್ಣಾಙ್ಕ = ಅಂಕೆ_base_tokens + len(special_tokens)
        # BOS / EOS token IDs
        self.bos_id: ಪೂರ್ಣಾಙ್ಕ = self.special_tokens["<|begin_of_text|>"]
        self.eos_id: ಪೂರ್ಣಾಙ್ಕ = self.special_tokens["<|end_of_text|>"]
        self.eot_id: ಪೂರ್ಣಾಙ್ಕ = self.special_tokens["<|eot_id|>"]
        self.eom_id: ಪೂರ್ಣಾಙ್ಕ = self.special_tokens["<|eom_id|>"]
        self.python_tag_id = self.special_tokens["<|python_tag|>"]
        self.pad_id: ಪೂರ್ಣಾಙ್ಕ = self.special_tokens["<|finetune_right_pad_id|>"]
        # hardcoded stop tokens for the base model
        self.stop_tokens = [
            self.special_tokens["<|begin_of_text|>"],
            self.special_tokens["<|end_of_text|>"],
        ]

    def encode(
        self,
        s: str,
        *,
        bos: bool,
        eos: bool,
        allowed_special: Optional[Union[Literal["all"], AbstractSet[str]]] = None,
        disallowed_special: Union[Literal["all"], Collection[str]] = (),
    ) -> List[ಪೂರ್ಣಾಙ್ಕ]:
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
        ಯದಿ allowed_special is None:
            allowed_special = set()
        ಪ್ರತಿಜ್ಞಾ type(s) is str

        substrs = (
            substr
            ಪ್ರತಿ i ಅನ್ತಃ range(0, len(s), TIKTOKEN_MAX_ENCODE_CHARS)
            ಪ್ರತಿ substr ಅನ್ತಃ self._split_whitespaces_or_nonwhitespaces(
                s[i : i + TIKTOKEN_MAX_ENCODE_CHARS], MAX_NO_WHITESPACES_CHARS
            )
        )
        t: List[ಪೂರ್ಣಾಙ್ಕ] = []
        ಪ್ರತಿ substr ಅನ್ತಃ substrs:
            t.extend(
                self.model.encode(
                    substr,
                    allowed_special=allowed_special,
                    disallowed_special=disallowed_special,
                )
            )
        ಯದಿ bos:
            t.ಸೇರಿಸು(0, self.bos_id)
        ಯದಿ eos:
            t.append(self.eos_id)
        ಪ್ರತ್ಯಾಗಮ t

    def decode(self, t: Sequence[ಪೂರ್ಣಾಙ್ಕ]) -> str:
        # Typecast is safe here. Tiktoken doesn't do anything list-related with the sequence.
        ಪ್ರತ್ಯಾಗಮ self.model.decode(cast(List[ಪೂರ್ಣಾಙ್ಕ], t))

    @staticmethod
    def _split_whitespaces_or_nonwhitespaces(
        s: str, max_consecutive_slice_len: ಪೂರ್ಣಾಙ್ಕ
    ) -> Iterator[str]:
        """
        Splits the string `s` so that each substring contains no more than `max_consecutive_slice_len`
        consecutive whitespaces or consecutive non-whitespaces.
        """
        current_slice_len = 0
        current_slice_is_space = s[0].isspace() ಯದಿ len(s) > 0 ಅನ್ಯಥಾ ಅಸತ್ಯ
        slice_start = 0

        ಪ್ರತಿ i ಅನ್ತಃ range(len(s)):
            is_now_space = s[i].isspace()

            ಯದಿ current_slice_is_space ^ is_now_space:
                current_slice_len = 1
                current_slice_is_space = is_now_space
            ಅನ್ಯಥಾ:
                current_slice_len += 1
                ಯದಿ current_slice_len > max_consecutive_slice_len:
                    ದಾ s[slice_start:i]
                    slice_start = i
                    current_slice_len = 1
        ದಾ s[slice_start:]

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") ಇವ f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.ಪಠ(256*4), dtype=np.int32)
    ಯದಿ header[0] != 20240801:
        print("ERROR: magic number mismatch in the data .bin file!")
        exit(1)
    ಪ್ರತಿಜ್ಞಾ header[1] == 7, "unsupported version"
    ntok = header[2] # number of tokens (claimed)
    ಪ್ರತ್ಯಾಗಮ ntok # for now just return the number of tokens

def _load_data_shard(filename):
    with open(filename, "rb") ಇವ f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.ಪಠ(256*4), dtype=np.int32)
        ಪ್ರತಿಜ್ಞಾ header[0] == 20240801, "magic number mismatch in the data .bin file"
        ಪ್ರತಿಜ್ಞಾ header[1] == 7, "unsupported version"
        ntok = header[2] # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.ಪಠ(), dtype=np.uint32)
    ಪ್ರತಿಜ್ಞಾ len(tokens) == ntok, "number of tokens read does not match header?"
    ಪ್ರತ್ಯಾಗಮ tokens

ವರ್ಗ DistributedShardedDataLoader:
    """
    This DataLoader is both:
    - distributed (works correctly in case of multiple processes in DDP)
    - sharded (supports datasets that are broken up into multiple data shards)
    It is not *permuted*, meaning that it itearates over the data in the order
    of the dataset on disk, so the user should make sure to shuffle their examples
    during the creation of their data shards for best performance.
    """
    def __init__(self, filename_pattern, B, T, process_rank, ಅಂಕೆ_processes):
        self.process_rank = process_rank
        self.ಅಂಕೆ_processes = ಅಂಕೆ_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        ಪ್ರತಿಜ್ಞಾ len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        ಪ್ರತಿ fname ಅನ್ತಃ self.files:
            shard_ntok = _peek_data_shard(fname)
            ಪ್ರತಿಜ್ಞಾ shard_ntok >= ಅಂಕೆ_processes * B * T + 1
            ntok_total += shard_ntok
        self.ntok_total = ntok_total
        print0(f"DataLoader: total number of tokens: {ntok_total:,} across {len(self.files)} files")

        # kick things off
        self.current_shard = None
        self.reset()

    def reset(self):
        # we're being a bit clever here: if we already had shard 0 loaded,
        # then don't do the work to reload it, just reset the pointer
        ಯದಿ self.current_shard != 0:
            self.current_shard = 0
            self.tokens = _load_data_shard(self.files[self.current_shard])
        self.current_position = self.process_rank * self.B * self.T

    def advance(self): # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = _load_data_shard(self.files[self.current_shard])

    def next_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        buf = torch.tensor(buf, dtype=torch.long)
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the start pointer in current shard
        self.current_position += B * T * self.ಅಂಕೆ_processes
        # if loading the next batch would be out of bounds advance the shard
        ಯದಿ self.current_position + (B * T * self.ಅಂಕೆ_processes + 1) > len(self.tokens):
            self.advance()
        ಪ್ರತ್ಯಾಗಮ x, y

# -----------------------------------------------------------------------------
# Python -> C bridge utilities for saving params/grads/activations to .bin files

def write_fp32(tensor, file):
    t = tensor.detach().cpu().to(torch.float32)
    b = t.numpy().tobytes()
    file.ಲಿಖ(b)

def write_bf16(tensor, file):
    t = tensor.detach().cpu().to(torch.bfloat16)
    # numpy doesn't have bf16 datatype so we have to trick it
    t = t.view(torch.int16) # trick: reinterpret as int16
    b = t.numpy().tobytes()
    file.ಲಿಖ(b)

def write_tensors(model_tensors, L, file, dtype):
    # writes LLaMA 3 model's weights to a binary file
    ಪ್ರತಿಜ್ಞಾ dtype ಅನ್ತಃ {"float32", "bfloat16"}
    write_fun = write_fp32 ಯದಿ dtype == "float32" ಅನ್ಯಥಾ write_bf16
    write_fun(model_tensors["transformer.wte.weight"], file) # (V, C)
    ಪ್ರತಿ i ಅನ್ತಃ range(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_1.weight"], file)
    ಪ್ರತಿ i ಅನ್ತಃ range(L): # (L, 3C, C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_attn.weight"], file)
    ಪ್ರತಿ i ಅನ್ತಃ range(L): # (L, C, C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_proj.weight"], file)
    ಪ್ರತಿ i ಅನ್ತಃ range(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_2.weight"], file)
    ಪ್ರತಿ i ಅನ್ತಃ range(L): # (L, 4C, C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_fc.weight"], file)
    ಪ್ರತಿ i ಅನ್ತಃ range(L): # (L, 4C, C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_fc2.weight"], file)
    ಪ್ರತಿ i ಅನ್ತಃ range(L): # (L, C, 4C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_proj.weight"], file)
    write_fun(model_tensors["transformer.ln_f.weight"], file) # (C, )
    write_fun(model_tensors["lm_head.weight"], file) # (V, C)

def write_model(model, filename, dtype):
    # everything we need to instantiate the model
    # 1) header is: version int, LLaMAConfig ints, padding to 1024 bytes
    ಪ್ರತಿಜ್ಞಾ dtype ಅನ್ತಃ {"float32", "bfloat16"}
    version = {
        "float32": 3, # 3: all tensors are fp32
        "bfloat16": 5, # 5: all tensors are bf16
    }[dtype]
    header = torch.zeros(256, dtype=torch.int32)
    header[0] = 20240803 # magic
    header[1] = version # checkpoint version
    header[2] = model.config.block_size
    header[3] = model.config.vocab_size
    header[4] = model.config.n_layer
    header[5] = model.config.n_head
    header[6] = model.config.n_kv_head
    header[7] = model.config.n_embd
    header[8] = model.config.ffn_dim_multiplier
    header[9] = model.config.multiple_of
    header[10] = model.config.norm_eps
    header[11] = model.config.rope_theta
    header[12] = model.config.use_scaled_rope
    header[13] = model.config.max_gen_batch_size
    header[14] = ಪೂರ್ಣಾಙ್ಕ(model.config.version.split('.')[0]) # major version
    header[15] = ಪೂರ್ಣಾಙ್ಕ(model.config.version.split('.')[1]) # minor version
    # 2) the parameters follow the header
    params = {name: param.cpu() ಪ್ರತಿ name, param ಅನ್ತಃ model.named_parameters()}
    # now write to file
    with open(filename, "wb") ಇವ file:
        file.ಲಿಖ(header.numpy().tobytes()) # header
        write_tensors(params, model.config.n_layer, file, dtype) # params
    print(f"wrote {filename}")

def write_state(model, x, y, logits, loss, filename):
    # the state is used for debugging.
    # it contains information about the input, logits, loss, and the parameter gradients
    # this can be used for checking the computation correctness in C
    header = torch.zeros(256, dtype=torch.int32)
    header[0] = 20240803 # magic
    header[1] = x.size(0) # batch size of the batch, B
    header[2] = x.size(1) # temporal extent of the batch, T
    grads = {name: param.grad.cpu() ಪ್ರತಿ name, param ಅನ್ತಃ model.named_parameters()}
    with open(filename, "wb") ಇವ file:
        # header
        file.ಲಿಖ(header.numpy().tobytes())
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
    print(f"wrote {filename}")

# -----------------------------------------------------------------------------
# int main

def print0(*args, **kwargs):
    # modified print that only prints from the master process
    # if this is not a distributed run, it's just a print
    ಯದಿ ಪೂರ್ಣಾಙ್ಕ(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)

ಯದಿ __name__ == "__main__":
    print0(f"Running pytorch {torch.version.__version__}")

    # default settings will overfit a tiny batch of data
    # and save model weights and debug state to disk on the first iteration
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_hf", type=ಪೂರ್ಣಾಙ್ಕ, ಸ್ವಭಾವ=1, help="use HuggingFace (default) or use Meta's model")
    parser.add_argument("--ckpt_dir", type=str, ಸ್ವಭಾವ=None, help="path to llama3 model checkpoint (needed if use_hf=0)")
    parser.add_argument("--tokenizer_path", type=str, ಸ್ವಭಾವ=None, help="path to llama3 tokenizer (needed if use_hf=0)")
    # file system input / output
    parser.add_argument("--input_bin", type=str, ಸ್ವಭಾವ="dev/data/tinyshakespeare/tiny_shakespeare_val.bin", help="input .bin to train on")
    parser.add_argument("--input_val_bin", type=str, ಸ್ವಭಾವ="", help="input .bin to eval validation loss on")
    parser.add_argument("--output_dir", type=str, ಸ್ವಭಾವ="", help="output directory to which to write logs and checkpoints")
    parser.add_argument("--model", type=str, ಸ್ವಭಾವ="meta-llama/Meta-Llama-3.1-8B", help="chose the llama model")
    # token layout for each step of the optimization
    parser.add_argument("--batch_size", type=ಪೂರ್ಣಾಙ್ಕ, ಸ್ವಭಾವ=4, help="batch size, in units of #batch dimensions")
    parser.add_argument("--sequence_length", type=ಪೂರ್ಣಾಙ್ಕ, ಸ್ವಭಾವ=64, help="sequence length")
    parser.add_argument("--total_batch_size", type=ಪೂರ್ಣಾಙ್ಕ, ಸ್ವಭಾವ=256, help="total desired batch size, in units of #tokens")
    # workload (number of steps)
    parser.add_argument("--num_iterations", type=ಪೂರ್ಣಾಙ್ಕ, ಸ್ವಭಾವ=10, help="number of iterations to run")
    parser.add_argument("--inference_only", type=ಪೂರ್ಣಾಙ್ಕ, ಸ್ವಭಾವ=0, help="only run inference")
    # optimization
    parser.add_argument("--learning_rate", type=ಭಿನ್ನ, ಸ್ವಭಾವ=1e-5, help="learning rate warmup iterations")
    parser.add_argument("--warmup_iters", type=ಪೂರ್ಣಾಙ್ಕ, ಸ್ವಭಾವ=0, help="learning rate warmup iterations")
    parser.add_argument("--learning_rate_decay_frac", type=ಭಿನ್ನ, ಸ್ವಭಾವ=1.0, help="learning rate warmup iterations")
    parser.add_argument("--weight_decay", type=ಭಿನ್ನ, ಸ್ವಭಾವ=0.0, help="weight decay")
    parser.add_argument("--grad_clip", type=ಭಿನ್ನ, ಸ್ವಭಾವ=1.0, help="maximum gradient magnitude")
    # evaluation
    parser.add_argument("--val_loss_every", type=ಪೂರ್ಣಾಙ್ಕ, ಸ್ವಭಾವ=0, help="every how mant steps to evaluate val loss?")
    parser.add_argument("--val_max_steps", type=ಪೂರ್ಣಾಙ್ಕ, ಸ್ವಭಾವ=20, help="how many batches of val to average?")
    parser.add_argument("--sample_every", type=ಪೂರ್ಣಾಙ್ಕ, ಸ್ವಭಾವ=0, help="how often to sample from the model?")
    # debugging
    parser.add_argument("--overfit_single_batch", type=ಪೂರ್ಣಾಙ್ಕ, ಸ್ವಭಾವ=1, help="overfit just one batch of data")
    # numerics
    parser.add_argument("--tensorcores", type=ಪೂರ್ಣಾಙ್ಕ, ಸ್ವಭಾವ=0, help="use tensorcores")
    # memory management
    parser.add_argument("--device", type=str, ಸ್ವಭಾವ="", help="by default we autodetect, or set it here")
    parser.add_argument("--compile", type=ಪೂರ್ಣಾಙ್ಕ, ಸ್ವಭಾವ=0, help="torch.compile the model")
    parser.add_argument("--dtype", type=str, ಸ್ವಭಾವ="bfloat16", help="float32|float16|bfloat16")
    parser.add_argument("--zero_stage", type=ಪೂರ್ಣಾಙ್ಕ, ಸ್ವಭಾವ=0, help="zero redundancy optimizer stage (0/1/2/3)")
    # python -> C bridge
    parser.add_argument("--write_tensors", type=ಪೂರ್ಣಾಙ್ಕ, ಸ್ವಭಾವ=0, help="write tensors to disk")
    args = parser.parse_args()

    # args error checking and convenience variables
    B, T = args.batch_size, args.sequence_length
    ಪ್ರತಿಜ್ಞಾ 1 <= T <= 8192, "sequence length must be between 1 and 8192"
    ಪ್ರತಿಜ್ಞಾ args.dtype ಅನ್ತಃ {"float32", "float16", "bfloat16"}
    ಪ್ರತಿಜ್ಞಾ args.model ಅನ್ತಃ {"meta-llama/Meta-Llama-3.1-8B"}  # only 8B base model supported for now

    # create the logging directory if it does not exist
    logfile = None
    ಯದಿ args.output_dir:
        os.makedirs(args.output_dir, exist_ok=ಸತ್ಯ)
        logfile = os.path.join(args.output_dir, "main.log")
        # create the log file "main.log" inside it, and wipe it clean
        with open(logfile, "w") ಇವ f:
            pass

    # set up DDP (distributed data parallel). torchrun sets this env variable
    ddp = ಪೂರ್ಣಾಙ್ಕ(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    ಯದಿ ddp:
        # use of DDP atm demands CUDA, we set the device appropriately according to rank
        ಪ್ರತಿಜ್ಞಾ torch.cuda.is_available(), "for now i think we need CUDA for DDP"
        init_process_group(backend='nccl')
        ddp_rank = ಪೂರ್ಣಾಙ್ಕ(os.environ['RANK'])
        ddp_local_rank = ಪೂರ್ಣಾಙ್ಕ(os.environ['LOCAL_RANK'])
        ddp_world_size = ಪೂರ್ಣಾಙ್ಕ(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        seed_offset = 0 # each process gets the exact same seed
        zero_stage = args.zero_stage
    ಅನ್ಯಥಾ:
        ddp_rank = 0
        ddp_local_rank = 0
        zero_stage = 0
        ddp_world_size = 1
        master_process = ಸತ್ಯ
        seed_offset = 0
        # select the device
        ಯದಿ args.device:
            # provided explicitly by the user
            device = args.device
        ಅನ್ಯಥಾ:
            # attempt to autodetect the device
            device = "cpu"
            ಯದಿ torch.cuda.is_available():
                device = "cuda"
            ನೋಚೇದ್ಯದಿ hasattr(torch.backends, "mps") ಚ torch.backends.mps.is_available():
                device = "mps"
    device_type = 'cuda' ಯದಿ 'cuda' ಅನ್ತಃ device ಅನ್ಯಥಾ 'cpu'
    ಪ್ರತಿಜ್ಞಾ device_type ಅನ್ತಃ {'cuda'}, "GPU required to run LLaMA 3"  # we need to load LLaMA as bf16 on CUDA
    print(f"using device: {device}")

    # calculate gradient accumulation from the desired total batch size and the current run configuration
    tokens_per_fwdbwd = B * T * ddp_world_size
    ಪ್ರತಿಜ್ಞಾ args.total_batch_size % tokens_per_fwdbwd == 0
    grad_accum_steps = args.total_batch_size // tokens_per_fwdbwd
    print0(f"total desired batch size: {args.total_batch_size}")
    print0(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    # set up a context manager following the desired dtype and device
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) ಯದಿ (device_type == "cuda") ಅನ್ಯಥಾ nullcontext()

    # rng / reproducibility
    torch.manual_seed(42)
    ಯದಿ torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # set the torch precision mode to use TensorFloat32 (TF32) for matmuls
    # docs https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    ಯದಿ args.tensorcores:
        torch.set_float32_matmul_precision('high')

    # init the model
    ಯದಿ args.use_hf:
        model = LLaMA.from_pretrained_llama3_hf(args.model)
    ಅನ್ಯಥಾ:  # use Meta's checkpoint
        ಪ್ರತಿಜ್ಞಾ args.ckpt_dir is ನ None ಚ os.path.exists(args.ckpt_dir), f"llama3 ckpt dir {args.ckpt_dir} does not exist"
        ಪ್ರತಿಜ್ಞಾ args.tokenizer_path is ನ None ಚ os.path.exists(args.tokenizer_path), f"llama3 tokenizer path {args.tokenizer_path} does not exist"
        model = LLaMA.from_pretrained_llama3_meta(args.ckpt_dir, args.tokenizer_path)

    model.train()
    ಯದಿ args.compile:
        ಯದಿ hasattr(config, "coordinate_descent_tuning"):
            config.coordinate_descent_tuning = ಸತ್ಯ # suggested by @Chillee
        print0("compiling the model...")
        model = torch.compile(model)

    # -------------------------------------------------------------------------
    # Our own version of a simple DistributedDataLoader

    # load tokens
    train_loader = DistributedShardedDataLoader(args.ಒಳಹರಿವು_bin, B, T, ddp_rank, ddp_world_size)
    val_loader = None
    ಯದಿ args.ಒಳಹರಿವು_val_bin:
        val_loader = DistributedShardedDataLoader(args.ಒಳಹರಿವು_val_bin, B, T, ddp_rank, ddp_world_size)

    # -------------------------------------------------------------------------
    # PyTorch -> C bridge: save some weights and state for C to load later as reference

    # do one forward pass to generate ground truth for our C tests
    ಯದಿ master_process ಚ args.write_tensors ಚ (ನ args.inference_only):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, y)
        loss.backward()
        # save model params, in bfloat16
        model_to_size = {"meta-llama/Meta-Llama-3.1-8B": "8B"}
        model_size_str = model_to_size[args.model] # e.g. "8B"
        write_model(model, os.path.join(args.output_dir, f"llama3.1_{model_size_str}_bf16.bin"), dtype="bfloat16")
        # save x, y, logits, loss, and parameter gradients, for debugging C
        # always store these in fp32 to have an accurate reference (?)
        write_state(model, x, y, logits, loss, os.path.join(args.output_dir, f"llama3_{model_size_str}_debug_state.bin"))
        # reset the train_loader for the optimization below
        train_loader.reset()

    # -------------------------------------------------------------------------
    # main training loop

    # here we wrap model into DDP container
    ಯದಿ ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module ಯದಿ ddp ಅನ್ಯಥಾ model # always contains the "raw" unwrapped model

    # init the optimizer
    optimizer = raw_model.configure_optimizers(weight_decay=args.weight_decay,
                                               learning_rate=args.learning_rate, betas=(0.9, 0.95),
                                               device_type=device, zero_stage=zero_stage)

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        min_lr = args.learning_rate * args.learning_rate_decay_frac
        # 1) linear warmup for warmup_iters steps
        ಯದಿ it < args.warmup_iters:
            ಪ್ರತ್ಯಾಗಮ args.learning_rate * (it+1) / args.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        ಯದಿ it > args.ಅಂಕೆ_iterations:
            ಪ್ರತ್ಯಾಗಮ min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - args.warmup_iters) / (args.ಅಂಕೆ_iterations - args.warmup_iters)
        ಪ್ರತಿಜ್ಞಾ 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
        ಪ್ರತ್ಯಾಗಮ min_lr + coeff * (args.learning_rate - min_lr)

    ಯದಿ device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    timings = []
    norm = -1.0   # dummy value to print in inference-only mode
    ಪ್ರತಿ step ಅನ್ತಃ range(args.ಅಂಕೆ_iterations + 1):
        t0 = time.time()
        last_step = (step == args.ಅಂಕೆ_iterations)

        # once in a while evaluate the validation dataset
        ಯದಿ (args.val_loss_every > 0 \
            ಚ (step % args.val_loss_every == 0 ವಾ last_step)) \
            ಚ (val_loader is ನ None):
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss = 0.0
                ಪ್ರತಿ _ ಅನ್ತಃ range(args.val_max_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    _, loss = model(x, y, return_logits=ಅಸತ್ಯ)
                    val_loss += loss.item()
                val_loss /= args.val_max_steps
            # log to console and to file
            print0(f"val loss {val_loss}")
            ಯದಿ master_process ಚ logfile is ನ None:
                with open(logfile, "a") ಇವ f:
                    f.ಲಿಖ("s:%d tel:%f\n" % (step, val_loss))

        # once in a while perform model inference on the master process
        ಯದಿ (args.sample_every > 0 \
            ಚ (step % args.sample_every == 0 ವಾ last_step)) \
            ಚ master_process:
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
            ಯದಿ args.use_hf:
                prompt_tokens = [model.tokenizer(x).ಒಳಹರಿವು_ids ಪ್ರತಿ x ಅನ್ತಃ prompts]
            ಅನ್ಯಥಾ:  # Meta
                prompt_tokens = [model.tokenizer.encode(x, bos=ಸತ್ಯ, eos=ಅಸತ್ಯ) ಪ್ರತಿ x ಅನ್ತಃ prompts]

            generation_tokens = model.generate(prompt_tokens, max_gen_len=64, temperature=0.6, ತುದಿ_p=0.9, echo=ಅಸತ್ಯ)
            results = [{"generation": model.tokenizer.decode(t)} ಪ್ರತಿ t ಅನ್ತಃ generation_tokens]
            ಪ್ರತಿ prompt, result ಅನ್ತಃ zip(prompts, results):
                print(prompt, end="")
                print(f"{result['generation']}")
                print("\n==================================\n")

        # bit confusing: we want to make sure to eval and sample on 0th iteration
        # but also after the very last iteration. so we loop for step <= num_iterations
        # instead of just < num_iterations (one extra due to <=), only to do
        # the validation/sampling one last time, and then we break right here as we're done.
        ಯದಿ last_step:
            ಭಙ್ಗ

        # --------------- TRAINING SECTION BEGIN -----------------
        model.train()
        optimizer.zero_grad(set_to_none=ಸತ್ಯ)
        # if we are trying to overfit a single batch, we reset the loader here
        ಯದಿ args.overfit_single_batch:
            train_loader.reset()
        # micro-batch loop where we do gradient accumulation to reach desired total batch size
        lossf = 0.0 # for getting the mean loss (as simple float) over the accumulation steps
        ಪ್ರತಿ micro_step ಅನ್ತಃ range(grad_accum_steps):
            # fetch a batch
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            ಯದಿ ddp:
                # we want only the last micro-step to sync grads in a DDP model
                # the official way to do this is with model.no_sync(), but that is a
                # context manager that bloats the code, so we just toggle this variable
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            # forward pass
            with ctx:
                _, loss = model(x, y, return_logits=ಅಸತ್ಯ)
                # we have to scale the loss to account for gradient accumulation,
                # because the gradients just add on each successive backward().
                # addition of gradients corresponds to a SUM in the objective, but
                # instead of a SUM we want MEAN, so we scale the loss here
                loss = loss / grad_accum_steps
                lossf += loss.detach() # keep track of the mean loss
            # backward pass
            ಯದಿ ನ args.inference_only:
                loss.backward()
        ಯದಿ ddp:
            dist.all_reduce(lossf, op=dist.ReduceOp.AVG)
        lossf = lossf.item()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        # determine and set the learning rate for this iteration
        lr = get_lr(step)
        ಪ್ರತಿ param_group ಅನ್ತಃ optimizer.param_groups:
            param_group['lr'] = lr
        # step the optimizer
        optimizer.step()
        # --------------- TRAINING SECTION END -------------------
        # everything that follows now is just diagnostics, prints, logging, etc.

        # wait on the CPU for all device work to end so we get accurate per-iteration timings below
        ಯದಿ device == "mps":
            torch.mps.synchronize()
        ನೋಚೇದ್ಯದಿ device == "cuda":
            torch.cuda.synchronize()
        # time and print
        t1 = time.time()
        # the 0th iteration is often an outlier (much slower) => skip logging it
        tokens_per_second = grad_accum_steps * ddp_world_size * B * T / (t1-t0)
        print0(f"step {step+1:4d}/{args.num_iterations} | train loss {lossf:.6f} | norm {norm:.4f} | lr {lr:.2e} | ({(t1-t0)*1000:.2f} ms | {tokens_per_second:.0f} tok/s)")
        # log to logile
        ಯದಿ master_process ಚ logfile is ನ None:
            with open(logfile, "a") ಇವ f:
                f.ಲಿಖ("s:%d trl:%f\n" % (step, lossf))

        # keep track of smooth timings, last 20 iterations
        ಯದಿ step > 0 ಚ step > args.ಅಂಕೆ_iterations - 20:
            timings.append(t1-t0)

    # print the average of the last 20 timings, to get something smooth-ish
    timings = timings[-20:]
    print0(f"final {len(timings)} iters avg: {np.mean(timings)*1000:.3f}ms")
    print0(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

    # -------------------------------------------------------------------------
    # clean up nice
    ಯದಿ ddp:
        destroy_process_group()
