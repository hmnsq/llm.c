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

ଆମଦାନୀ argparse
ଆମଦାନୀ os
ଆମଦାନୀ math
ଆମଦାନୀ glob
ଆମଦାନୀ inspect
ତଃ contextlib ଆମଦାନୀ nullcontext
ତଃ dataclasses ଆମଦାନୀ dataclass
ତଃ pathlib ଆମଦାନୀ Path
ଆମଦାନୀ time
ତଃ typing ଆମଦାନୀ (
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

ଆମଦାନୀ numpy ଇଵ np
ଆମଦାନୀ torch
ଆମଦାନୀ torch.nn ଇଵ nn
ତଃ torch.nn ଆମଦାନୀ functional ଇଵ F
ଆମଦାନୀ torch._inductor.config ଇଵ config
ତଃ torch.nn.parallel ଆମଦାନୀ DistributedDataParallel ଇଵ DDP
ତଃ torch.distributed ଆମଦାନୀ init_process_group, destroy_process_group
ତଃ torch.distributed.optim ଆମଦାନୀ ZeroRedundancyOptimizer
ଆମଦାନୀ torch.distributed ଇଵ dist

ଆମଦାନୀ tiktoken
ତଃ tiktoken.load ଆମଦାନୀ load_tiktoken_bpe

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the LLaMA 3.x model

# Used in Grouped Query Attention (GQA), broadcasts the key and value tensors
def repeat_kv(x: torch.Tensor, n_rep: ପୂର୍ଣ୍ଣସଂଖ୍ୟା) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    ଯଦି n_rep == 1:
        ଫେରନ୍ତୁ x
    ଫେରନ୍ତୁ (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

# -----------------------------------------------------------------------------
# RoPE related

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    ପ୍ରତିଜ୍ଞା 0 <= 1 < ndim
    ପ୍ରତିଜ୍ଞା freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d ଯଦି i == 1 କିମ୍ବା i == ndim - 1 ଅନ୍ୟଥା 1 ପାଇଁ i, d ଭିତରେ enumerate(x.shape)]
    ଫେରନ୍ତୁ freqs_cis.view(*shape)

def apply_scaling(freqs: torch.Tensor):
    # Values obtained from grid search
    scale_factor = 8
    ତଳ_freq_factor = 1
    ଉପର_freq_factor = 4
    old_context_len = 8192  # original llama3 length

    ତଳ_freq_wavelen = old_context_len / ତଳ_freq_factor
    ଉପର_freq_wavelen = old_context_len / ଉପର_freq_factor
    new_freqs = []
    ପାଇଁ freq ଭିତରେ freqs:
        wavelen = 2 * math.pi / freq
        ଯଦି wavelen < ଉପର_freq_wavelen:
            new_freqs.append(freq)
        ନଚେତଯଦି wavelen > ତଳ_freq_wavelen:
            new_freqs.append(freq / scale_factor)
        ଅନ୍ୟଥା:
            ପ୍ରତିଜ୍ଞା ତଳ_freq_wavelen != ଉପର_freq_wavelen
            smooth = (old_context_len / wavelen - ତଳ_freq_factor) / (
                ଉପର_freq_factor - ତଳ_freq_factor
            )
            new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
    ଫେରନ୍ତୁ torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.ଦଶମିକ().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.ଦଶମିକ().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    ଫେରନ୍ତୁ xq_out.type_as(xq), xk_out.type_as(xk)

def precompute_freqs_cis(
    dim: ପୂର୍ଣ୍ଣସଂଖ୍ୟା, end: ପୂର୍ଣ୍ଣସଂଖ୍ୟା, theta: ଦଶମିକ = 10000.0, use_scaled: bool = ମିଥ୍ୟା
):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].ଦଶମିକ() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    ଯଦି use_scaled:
        freqs = apply_scaling(freqs)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    ଫେରନ୍ତୁ freqs_cis

# -----------------------------------------------------------------------------
# LLaMA building blocks

# LLaMA reference code explicitly implemented RMSNorm so we copy pasted it
# (https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/api/model.py)
# we could also use nn.RMSNorm, it has slightly different numeric properties, but equivalent
ଶ୍ରେଣୀ RMSNorm(torch.nn.Module):
    def __init__(self, dim: ପୂର୍ଣ୍ଣସଂଖ୍ୟା, eps: ଦଶମିକ = 1e-6):
        ପରମ().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        ଫେରନ୍ତୁ x * torch.rsqrt(x.pow(2).mean(-1, keepdim=ସତ୍ୟ) + self.eps)

    def forward(self, x):
        output = self._norm(x.ଦଶମିକ()).type_as(x)
        ଫେରନ୍ତୁ output * self.weight

ଶ୍ରେଣୀ CausalSelfAttention(nn.Module):

    def __init__(self, config):
        ପରମ().__init__()
        ପ୍ରତିଜ୍ଞା config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_rep = self.n_head // self.n_kv_head
        self.hd = config.n_embd // config.n_head
        self.use_kv = config.use_kv
        self.flash = config.flash

        self.c_attn = nn.Linear(config.n_embd, (config.n_head + 2 * config.n_kv_head) * self.hd, bias=ମିଥ୍ୟା)  # key, query, value projections
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=ମିଥ୍ୟା)  # output projection

        # static KV cache - we could alternatively allocate it outside of the model and just pass it in when needed
        ଯଦି self.use_kv:
            self.cache_k = torch.zeros((config.max_gen_batch_size, config.block_size, config.n_kv_head, self.hd))
            self.cache_v = torch.zeros((config.max_gen_batch_size, config.block_size, config.n_kv_head, self.hd))

    def forward(self, x, freqs_cis=None, start_pos=None, mask=None):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split([self.n_head * self.hd, self.n_kv_head * self.hd, self.n_kv_head * self.hd], dim=-1)
        q, k, v = map(ଅନାମ t: t.view(B, T, -1, self.hd), (q, k, v))  # (B, T, NH, HD)

        q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)  # rotate QK (rope)  <-- 1. difference compared to GPT-2

        ଯଦି self.use_kv ଏବଂ ନୁହେଁ self.training ଏବଂ start_pos >= 0:  # use kv-caching during inference
            self.cache_k[:B, start_pos : start_pos + T] = k
            self.cache_v[:B, start_pos : start_pos + T] = v
            k = self.cache_k[:B, : start_pos + T]
            v = self.cache_v[:B, : start_pos + T]

        k = repeat_kv(k, self.n_rep)  # GQA <-- 2. difference compared to GPT-2
        v = repeat_kv(v, self.n_rep)

        q, k, v = map(ଅନାମ t: t.transpose(1, 2), (q, k, v))  # (B, NH, T, HD)

        ଯଦି self.flash:
            # flashattention
            # if T == 1 no need to mask, otherwise the function complains
            # scaled_dot_product_attention expects a mask where value of True indicates that the element should take part in attention
            # our mask is the opposite, so we need to invert it
            y = F.scaled_dot_product_attention(q, k, v, mask == 0 ଯଦି T > 1 ଅନ୍ୟଥା None)
        ଅନ୍ୟଥା:
            # manual implementation of attention
            # this materializes the large (T,T) matrix for all the queries and keys
            scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.hd))
            ଯଦି mask is ନୁହେଁ None:
                scores.masked_fill_(mask, torch.finfo(scores.dtype).min)
            att = F.softmax(scores.ଦଶମିକ(), dim=-1).type_as(q)
            y = att @ v # (B, NH, T, T) x (B, NH, T, HD) -> (B, NH, T, HD)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        ଫେରନ୍ତୁ y

ଶ୍ରେଣୀ MLP(nn.Module):

    def __init__(self, config):
        ପରମ().__init__()
        hidden_dim = 4 * config.n_embd
        hidden_dim = ପୂର୍ଣ୍ଣସଂଖ୍ୟା(2 * hidden_dim / 3)
        # custom dim factor multiplier
        ଯଦି config.ffn_dim_multiplier is ନୁହେଁ None:
            hidden_dim = ପୂର୍ଣ୍ଣସଂଖ୍ୟା(config.ffn_dim_multiplier * hidden_dim)
        hidden_dim = config.multiple_of * ((hidden_dim + config.multiple_of - 1) // config.multiple_of)
        self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=ମିଥ୍ୟା)
        self.c_fc2 = nn.Linear(config.n_embd, hidden_dim, bias=ମିଥ୍ୟା)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=ମିଥ୍ୟା)

    def forward(self, x):
        # SwiGLU self.c_proj(F.silu(self.c_fc2(x)) * self.c_fc(x))  <-- 3. difference compared to GPT-2
        x1 = self.c_fc(x)
        x2 = self.c_fc2(x)
        x2 = F.silu(x2)
        x = x1 * x2
        x = self.c_proj(x)
        ଫେରନ୍ତୁ x

ଶ୍ରେଣୀ Block(nn.Module):

    def __init__(self, config):
        ପରମ().__init__()
        self.ln_1 = RMSNorm(config.n_embd, config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd, config.norm_eps)
        self.mlp = MLP(config)

    def forward(self, x, freqs_cis=None, start_pos=None, mask=None):
        x = x + self.attn(self.ln_1(x), freqs_cis, start_pos, mask)
        x = x + self.mlp(self.ln_2(x))
        ଫେରନ୍ତୁ x

# -----------------------------------------------------------------------------
# The main LLaMA 3.1 model

@dataclass
ଶ୍ରେଣୀ LlamaConfig:
    version: str = "3.1"
    block_size: ପୂର୍ଣ୍ଣସଂଖ୍ୟା = 8192
    vocab_size: ପୂର୍ଣ୍ଣସଂଖ୍ୟା = 128256
    n_layer: ପୂର୍ଣ୍ଣସଂଖ୍ୟା = 32
    n_head: ପୂର୍ଣ୍ଣସଂଖ୍ୟା = 32
    n_kv_head: ପୂର୍ଣ୍ଣସଂଖ୍ୟା = 8
    n_embd: ପୂର୍ଣ୍ଣସଂଖ୍ୟା = 4096
    ffn_dim_multiplier: ଦଶମିକ = 1.3
    multiple_of: ପୂର୍ଣ୍ଣସଂଖ୍ୟା = 1024
    norm_eps: ଦଶମିକ = 1e-5
    rope_theta: ଦଶମିକ = 500000.0
    use_scaled_rope: bool = ସତ୍ୟ
    max_gen_batch_size: ପୂର୍ଣ୍ଣସଂଖ୍ୟା = 4
    use_kv: bool = ସତ୍ୟ
    flash: bool = ମିଥ୍ୟା  # use flashattention?

    def __init__(self, **kwargs):
        ପାଇଁ k, v ଭିତରେ kwargs.ବସ୍ତୁଗୁଡ଼ିକ():
            ଯଦି hasattr(self, k):
                setattr(self, k, v)
        ପ୍ରତିଜ୍ଞା self.n_kv_head <= self.n_head
        ପ୍ରତିଜ୍ଞା self.n_head % self.n_kv_head == 0
        ପ୍ରତିଜ୍ଞା self.n_embd % self.n_head == 0

ଶ୍ରେଣୀ LLaMA(nn.Module):

    def __init__(self, config):
        ପରମ().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) ପାଇଁ _ ଭିତରେ range(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd, config.norm_eps),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=ମିଥ୍ୟା)

        # init all weights, use a torch rng object to be very careful
        self.init_rng = torch.Generator()
        self.init_rng.manual_seed(42)

        self.freqs_cis = precompute_freqs_cis(
            config.n_embd // config.n_head,
            config.block_size * 2,
            config.rope_theta,
            config.use_scaled_rope,
        )

    def forward(self, idx, targets=None, return_logits=ସତ୍ୟ, start_pos=0):
        _, t = idx.size()
        ପ୍ରତିଜ୍ଞା t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # forward the LLaMA model itself
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        freqs_cis = self.freqs_cis[start_pos:start_pos+t]

        mask = torch.triu(torch.ones((t, t), device=next(self.parameters()).device, dtype=torch.bool), diagonal=1)

        ପାଇଁ i, block ଭିତରେ enumerate(self.transformer.h):
            x = block(x, freqs_cis, start_pos, mask)
        x = self.transformer.ln_f(x)

        ଯଦି targets is ନୁହେଁ None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x).ଦଶମିକ()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        ଅନ୍ୟଥା:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]).ଦଶମିକ() # note: using list [-1] to preserve the time dim
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        ଯଦି ନୁହେଁ return_logits:
            logits = None

        ଫେରନ୍ତୁ logits, loss

    @staticmethod
    def adapt_llama_state_dict_ଚାବିଗୁଡ଼ିକ(checkpoint, config: LlamaConfig):
        # Modify key names from Meta's LLaMA to our LLaMA
        # our key names are derived from GPT-2's key names
        checkpoint['transformer.wte.weight'] = checkpoint.କାଢ଼('tok_embeddings.weight')

        ପାଇଁ i ଭିତରେ range(config.n_layer):
            ପାଇଁ name ଭିତରେ ['attention_norm', 'ffn_norm']:
                old_ଚାବି = f'layers.{i}.{name}.weight'  # e.g. layers.x.attention_norm.weight -> transformer.h.x.ln_1.weight
                new_ଚାବି = f'transformer.h.{i}.ln_{1 if name == "attention_norm" else 2}.weight'
                checkpoint[new_ଚାବି] = checkpoint.କାଢ଼(old_ଚାବି)

        ପାଇଁ i ଭିତରେ range(config.n_layer):
            ପାଇଁ name ଭିତରେ ['attention.wq', 'attention.wk', 'attention.wv']:
                old_ଚାବି = f'layers.{i}.{name}.weight'
                new_ଚାବି = f'transformer.h.{i}.attn.c_attn.weight'
                ଯଦି name == 'attention.wq':
                    checkpoint[new_ଚାବି] = checkpoint.କାଢ଼(old_ଚାବି)
                ଅନ୍ୟଥା:  # merge 3 weights into transformer.h.x.attn.c_attn.weight
                    checkpoint[new_ଚାବି] = torch.cat((checkpoint[new_ଚାବି], checkpoint.କାଢ଼(old_ଚାବି)), dim=0)
            old_ଚାବି = f'layers.{i}.attention.wo.weight'
            new_ଚାବି = f'transformer.h.{i}.attn.c_proj.weight'
            checkpoint[new_ଚାବି] = checkpoint.କାଢ଼(old_ଚାବି)

        ffn_map = {'w1': 'c_fc2', 'w2': 'c_proj', 'w3': 'c_fc'}
        ପାଇଁ i ଭିତରେ range(config.n_layer):
            ପାଇଁ name ଭିତରେ ['feed_forward.w1', 'feed_forward.w2', 'feed_forward.w3']:
                old_ଚାବି = f'layers.{i}.{name}.weight'
                new_ଚାବି = f'transformer.h.{i}.mlp.{ffn_map[name.split(".")[-1]]}.weight'
                checkpoint[new_ଚାବି] = checkpoint.କାଢ଼(old_ଚାବି)

        checkpoint['transformer.ln_f.weight'] = checkpoint.କାଢ଼('norm.weight')
        checkpoint['lm_head.weight'] = checkpoint.କାଢ଼('output.weight')

        ଫେରନ୍ତୁ checkpoint

    @staticmethod
    def adapt_llama_state_dict_ଚାବିଗୁଡ଼ିକ_hf(checkpoint, config: LlamaConfig):
        # Modify key names from HuggingFace's LLaMA to our LLaMA
        # our key names are derived from GPT-2's key names
        checkpoint['transformer.wte.weight'] = checkpoint.କାଢ଼('model.embed_tokens.weight')

        # We need to unpermute K and V because HF script permuted the original Meta-LLaMA weights
        # see: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py
        def unpermute(w, n_heads, dim1, dim2):
            ଫେରନ୍ତୁ w.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)

        ପାଇଁ i ଭିତରେ range(config.n_layer):
            ପାଇଁ name ଭିତରେ ['input_layernorm', 'post_attention_layernorm']:
                old_ଚାବି = f'model.layers.{i}.{name}.weight'  # e.g. layers.x.attention_norm.weight -> transformer.h.x.ln_1.weight
                new_ଚାବି = f'transformer.h.{i}.ln_{1 if name == "input_layernorm" else 2}.weight'
                checkpoint[new_ଚାବି] = checkpoint.କାଢ଼(old_ଚାବି)

        ପାଇଁ i ଭିତରେ range(config.n_layer):
            ପାଇଁ name ଭିତରେ ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj']:
                old_ଚାବି = f'model.layers.{i}.{name}.weight'
                new_ଚାବି = f'transformer.h.{i}.attn.c_attn.weight'
                ଯଦି name == 'self_attn.q_proj':
                    checkpoint[new_ଚାବି] = unpermute(checkpoint.କାଢ଼(old_ଚାବି), config.n_head, config.n_embd, config.n_embd)
                ଅନ୍ୟଥା:  # merge 3 weights into transformer.h.x.attn.c_attn.weight
                    tensor = checkpoint.କାଢ଼(old_ଚାବି)
                    ଯଦି name == 'self_attn.k_proj':
                        tensor = unpermute(tensor, config.n_kv_head, config.n_kv_head * (config.n_embd // config.n_head), config.n_embd)
                    checkpoint[new_ଚାବି] = torch.cat((checkpoint[new_ଚାବି], tensor), dim=0)
            old_ଚାବି = f'model.layers.{i}.self_attn.o_proj.weight'
            new_ଚାବି = f'transformer.h.{i}.attn.c_proj.weight'
            checkpoint[new_ଚାବି] = checkpoint.କାଢ଼(old_ଚାବି)

        ffn_map = {'gate_proj': 'c_fc2', 'down_proj': 'c_proj', 'up_proj': 'c_fc'}
        ପାଇଁ i ଭିତରେ range(config.n_layer):
            ପାଇଁ name ଭିତରେ ['gate_proj', 'down_proj', 'up_proj']:
                old_ଚାବି = f'model.layers.{i}.mlp.{name}.weight'
                new_ଚାବି = f'transformer.h.{i}.mlp.{ffn_map[name]}.weight'
                checkpoint[new_ଚାବି] = checkpoint.କାଢ଼(old_ଚାବି)

        checkpoint['transformer.ln_f.weight'] = checkpoint.କାଢ଼('model.norm.weight')

        ଫେରନ୍ତୁ checkpoint

    @classmethod
    def from_pretrained_llama3_hf(cls, model_id):
        """Loads pretrained LLaMA model weights from HuggingFace"""
        ତଃ transformers ଆମଦାନୀ AutoModelForCausalLM, AutoTokenizer
        ପ୍ରତିଜ୍ଞା model_id == "meta-llama/Meta-Llama-3.1-8B", "Only the 8B-base model is supported for now"
        model_args = LlamaConfig()

        model = AutoModelForCausalLM.from_pretrained(model_id)
        checkpoint = LLaMA.adapt_llama_state_dict_ଚାବିଗୁଡ଼ିକ_hf(model.state_dict(), model_args)

        original_default_type = torch.get_default_dtype()  # save the default type
        torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)  # much faster loading
        model = LLaMA(model_args)
        model.load_state_dict(checkpoint, strict=ମିଥ୍ୟା)
        torch.set_default_tensor_type(torch.tensor([], dtype=original_default_type, device="cpu").type())  # restore default type

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_id = 128004  # this is the pad token id for LLaMA 3.1 base, we need to set this explicitly as our generate func expects it
        tokenizer.stop_tokens = [tokenizer.eos_token_id]
        model.tokenizer = tokenizer
        ଫେରନ୍ତୁ model

    @classmethod
    def from_pretrained_llama3_meta(cls, ckpt_dir, tokenizer_path):
        """Loads pretrained LLaMA model weights from a checkpoint directory"""
        model_args = LlamaConfig()

        ckpt_path = sorted(Path(ckpt_dir).glob("*.pth"))[0]
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=ସତ୍ୟ)
        checkpoint = LLaMA.adapt_llama_state_dict_ଚାବିଗୁଡ଼ିକ(checkpoint, model_args)

        original_default_type = torch.get_default_dtype()  # save the default type
        torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)  # much faster loading
        model = LLaMA(model_args)
        model.load_state_dict(checkpoint, strict=ମିଥ୍ୟା)
        torch.set_default_tensor_type(torch.tensor([], dtype=original_default_type, device="cpu").type())  # restore default type

        tokenizer = Tokenizer(model_path=tokenizer_path)
        model.tokenizer = tokenizer
        ଫେରନ୍ତୁ model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, zero_stage):
        # start with all of the candidate parameters
        param_dict = {pn: p ପାଇଁ pn, p ଭିତରେ self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p ପାଇଁ pn, p ଭିତରେ param_dict.ବସ୍ତୁଗୁଡ଼ିକ() ଯଦି p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p ପାଇଁ n, p ଭିତରେ param_dict.ବସ୍ତୁଗୁଡ଼ିକ() ଯଦି p.dim() >= 2]
        nodecay_params = [p ପାଇଁ n, p ଭିତରେ param_dict.ବସ୍ତୁଗୁଡ଼ିକ() ଯଦି p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        ଅଙ୍କ_decay_params = sum(p.numel() ପାଇଁ p ଭିତରେ decay_params)
        ଅଙ୍କ_nodecay_params = sum(p.numel() ପାଇଁ p ଭିତରେ nodecay_params)
        print0(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print0(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' ଭିତରେ inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available ଏବଂ device_type == 'cuda'
        print0(f"using fused AdamW: {use_fused}")
        ଯଦି zero_stage == 1:
            print0("using ZeroRedundancyOptimizer")
            optimizer = ZeroRedundancyOptimizer(**optim_groups[0], optimizer_class=torch.optim.AdamW,
                                                lr=learning_rate, betas=betas, fused=use_fused)
            optimizer.add_param_group(optim_groups[1])
        ଅନ୍ୟଥା:
            print0("using regular AdamW")
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=use_fused)
        ଫେରନ୍ତୁ optimizer

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[List[ପୂର୍ଣ୍ଣସଂଖ୍ୟା]],
        max_gen_len: ପୂର୍ଣ୍ଣସଂଖ୍ୟା,
        temperature: ଦଶମିକ = 0.6,
        ଶୀର୍ଷ_p: ଦଶମିକ = 0.9,
        echo: bool = ମିଥ୍ୟା,
    ) -> Tuple[List[List[ପୂର୍ଣ୍ଣସଂଖ୍ୟା]], Optional[List[List[ଦଶମିକ]]]]:
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
        ପ୍ରତିଜ୍ଞା bsz <= self.config.max_gen_batch_size, f"Batch size {bsz} exceeds the maximum generation batch size {self.config.max_gen_batch_size}"
        device = next(self.parameters()).device

        min_prompt_len = min(len(t) ପାଇଁ t ଭିତରେ prompt_tokens)
        max_prompt_len = max(len(t) ପାଇଁ t ଭିତରେ prompt_tokens)
        ପ୍ରତିଜ୍ଞା max_prompt_len <= self.config.block_size, f"Prompt length {max_prompt_len} exceeds the maximum block size {self.config.block_size}"
        total_len = min(self.config.block_size, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_id
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device=device)
        ପାଇଁ idx, t ଭିତରେ enumerate(prompt_tokens):
            tokens[idx, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)

        prev_pos = 0
        eos_reached = torch.tensor([ମିଥ୍ୟା] * bsz, device=device)
        ନିବେଶ_text_mask = tokens != pad_id

        ଯଦି min_prompt_len == total_len:
            logits, _ = self.forward(tokens, start_pos=prev_pos)

        stop_tokens = torch.tensor(list(self.tokenizer.stop_tokens)).to(device)

        ପାଇଁ cur_pos ଭିତରେ range(min_prompt_len, total_len):
            logits, _ = self.forward(tokens[:, prev_pos:cur_pos], start_pos=prev_pos)
            ଯଦି temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_ଶୀର୍ଷ_p(probs, ଶୀର୍ଷ_p)
            ଅନ୍ୟଥା:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(ନିବେଶ_text_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token
            eos_reached |= ~ନିବେଶ_text_mask[:, cur_pos] & torch.isin(next_token, stop_tokens)
            prev_pos = cur_pos
            ଯଦି all(eos_reached):
                ଭାଙ୍ଗ

        out_tokens = []
        ପାଇଁ i, toks ଭିତରେ enumerate(tokens.tolist()):
            # cut to max gen len
            start = 0 ଯଦି echo ଅନ୍ୟଥା len(prompt_tokens[i])
            toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
            # cut to after eos tok if any
            ପାଇଁ stop_token ଭିତରେ self.tokenizer.stop_tokens:
                ଚେଷ୍ଟା:
                    eos_idx = toks.index(stop_token)
                    toks = toks[:eos_idx]
                ବ୍ୟତିକ୍ରମ ମୂଲ୍ୟ_Error:
                    pass
            out_tokens.append(toks)
        ଫେରନ୍ତୁ out_tokens

# -----------------------------------------------------------------------------
# sampling utils

def sample_ଶୀର୍ଷ_p(probs, p):
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
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=ସତ୍ୟ)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=ସତ୍ୟ))
    next_token = torch.multinomial(probs_sort, ଅଙ୍କ_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    ଫେରନ୍ତୁ next_token

# -----------------------------------------------------------------------------
# Llama 3.1 Tokenizer

# The tiktoken tokenizer can handle <=400k chars without
# pyo3_runtime.PanicException.
TIKTOKEN_MAX_ENCODE_CHARS = 400_000

# https://github.com/openai/tiktoken/issues/195
# Here we iterate over subsequences and split if we exceed the limit
# of max consecutive non-whitespace or whitespace characters.
MAX_NO_WHITESPACES_CHARS = 25_000


ଶ୍ରେଣୀ Tokenizer:
    """
    Tokenizing and encoding/decoding text using the Tiktoken tokenizer.
    """

    special_tokens: Dict[str, ପୂର୍ଣ୍ଣସଂଖ୍ୟା]

    ଅଙ୍କ_reserved_special_tokens = 256

    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: E501

    def __init__(self, model_path: str):
        """
        Initializes the Tokenizer with a Tiktoken model.

        Args:
            model_path (str): The path to the Tiktoken model file.
        """
        ପ୍ରତିଜ୍ଞା os.path.isfile(model_path), model_path

        mergeable_ranks = load_tiktoken_bpe(model_path)
        ଅଙ୍କ_base_tokens = len(mergeable_ranks)
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
            ପାଇଁ i ଭିତରେ range(self.ଅଙ୍କ_reserved_special_tokens - len(special_tokens))
        ]
        special_tokens = special_tokens + reserved_tokens

        self.special_tokens = {
            token: ଅଙ୍କ_base_tokens + i ପାଇଁ i, token ଭିତରେ enumerate(special_tokens)
        }
        self.model = tiktoken.Encoding(
            name=Path(model_path).name,
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_tokens=self.special_tokens,
        )

        self.n_ଶବ୍ଦଗୁଡ଼ିକ: ପୂର୍ଣ୍ଣସଂଖ୍ୟା = ଅଙ୍କ_base_tokens + len(special_tokens)
        # BOS / EOS token IDs
        self.bos_id: ପୂର୍ଣ୍ଣସଂଖ୍ୟା = self.special_tokens["<|begin_of_text|>"]
        self.eos_id: ପୂର୍ଣ୍ଣସଂଖ୍ୟା = self.special_tokens["<|end_of_text|>"]
        self.eot_id: ପୂର୍ଣ୍ଣସଂଖ୍ୟା = self.special_tokens["<|eot_id|>"]
        self.eom_id: ପୂର୍ଣ୍ଣସଂଖ୍ୟା = self.special_tokens["<|eom_id|>"]
        self.python_tag_id = self.special_tokens["<|python_tag|>"]
        self.pad_id: ପୂର୍ଣ୍ଣସଂଖ୍ୟା = self.special_tokens["<|finetune_right_pad_id|>"]
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
    ) -> List[ପୂର୍ଣ୍ଣସଂଖ୍ୟା]:
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
        ଯଦି allowed_special is None:
            allowed_special = set()
        ପ୍ରତିଜ୍ଞା type(s) is str

        substrs = (
            substr
            ପାଇଁ i ଭିତରେ range(0, len(s), TIKTOKEN_MAX_ENCODE_CHARS)
            ପାଇଁ substr ଭିତରେ self._split_whitespaces_or_nonwhitespaces(
                s[i : i + TIKTOKEN_MAX_ENCODE_CHARS], MAX_NO_WHITESPACES_CHARS
            )
        )
        t: List[ପୂର୍ଣ୍ଣସଂଖ୍ୟା] = []
        ପାଇଁ substr ଭିତରେ substrs:
            t.extend(
                self.model.encode(
                    substr,
                    allowed_special=allowed_special,
                    disallowed_special=disallowed_special,
                )
            )
        ଯଦି bos:
            t.ଭର୍ତ୍ତି(0, self.bos_id)
        ଯଦି eos:
            t.append(self.eos_id)
        ଫେରନ୍ତୁ t

    def decode(self, t: Sequence[ପୂର୍ଣ୍ଣସଂଖ୍ୟା]) -> str:
        # Typecast is safe here. Tiktoken doesn't do anything list-related with the sequence.
        ଫେରନ୍ତୁ self.model.decode(cast(List[ପୂର୍ଣ୍ଣସଂଖ୍ୟା], t))

    @staticmethod
    def _split_whitespaces_or_nonwhitespaces(
        s: str, max_consecutive_slice_len: ପୂର୍ଣ୍ଣସଂଖ୍ୟା
    ) -> Iterator[str]:
        """
        Splits the string `s` so that each substring contains no more than `max_consecutive_slice_len`
        consecutive whitespaces or consecutive non-whitespaces.
        """
        current_slice_len = 0
        current_slice_is_space = s[0].isspace() ଯଦି len(s) > 0 ଅନ୍ୟଥା ମିଥ୍ୟା
        slice_start = 0

        ପାଇଁ i ଭିତରେ range(len(s)):
            is_now_space = s[i].isspace()

            ଯଦି current_slice_is_space ^ is_now_space:
                current_slice_len = 1
                current_slice_is_space = is_now_space
            ଅନ୍ୟଥା:
                current_slice_len += 1
                ଯଦି current_slice_len > max_consecutive_slice_len:
                    ଦିଅ s[slice_start:i]
                    slice_start = i
                    current_slice_len = 1
        ଦିଅ s[slice_start:]

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") ଇଵ f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.ପଠ(256*4), dtype=np.int32)
    ଯଦି header[0] != 20240801:
        print("ERROR: magic number mismatch in the data .bin file!")
        exit(1)
    ପ୍ରତିଜ୍ଞା header[1] == 7, "unsupported version"
    ntok = header[2] # number of tokens (claimed)
    ଫେରନ୍ତୁ ntok # for now just return the number of tokens

def _load_data_shard(filename):
    with open(filename, "rb") ଇଵ f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.ପଠ(256*4), dtype=np.int32)
        ପ୍ରତିଜ୍ଞା header[0] == 20240801, "magic number mismatch in the data .bin file"
        ପ୍ରତିଜ୍ଞା header[1] == 7, "unsupported version"
        ntok = header[2] # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.ପଠ(), dtype=np.uint32)
    ପ୍ରତିଜ୍ଞା len(tokens) == ntok, "number of tokens read does not match header?"
    ଫେରନ୍ତୁ tokens

ଶ୍ରେଣୀ DistributedShardedDataLoader:
    """
    This DataLoader is both:
    - distributed (works correctly in case of multiple processes in DDP)
    - sharded (supports datasets that are broken up into multiple data shards)
    It is not *permuted*, meaning that it itearates over the data in the order
    of the dataset on disk, so the user should make sure to shuffle their examples
    during the creation of their data shards for best performance.
    """
    def __init__(self, filename_pattern, B, T, process_rank, ଅଙ୍କ_processes):
        self.process_rank = process_rank
        self.ଅଙ୍କ_processes = ଅଙ୍କ_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        ପ୍ରତିଜ୍ଞା len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        ପାଇଁ fname ଭିତରେ self.files:
            shard_ntok = _peek_data_shard(fname)
            ପ୍ରତିଜ୍ଞା shard_ntok >= ଅଙ୍କ_processes * B * T + 1
            ntok_total += shard_ntok
        self.ntok_total = ntok_total
        print0(f"DataLoader: total number of tokens: {ntok_total:,} across {len(self.files)} files")

        # kick things off
        self.current_shard = None
        self.reset()

    def reset(self):
        # we're being a bit clever here: if we already had shard 0 loaded,
        # then don't do the work to reload it, just reset the pointer
        ଯଦି self.current_shard != 0:
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
        self.current_position += B * T * self.ଅଙ୍କ_processes
        # if loading the next batch would be out of bounds advance the shard
        ଯଦି self.current_position + (B * T * self.ଅଙ୍କ_processes + 1) > len(self.tokens):
            self.advance()
        ଫେରନ୍ତୁ x, y

# -----------------------------------------------------------------------------
# Python -> C bridge utilities for saving params/grads/activations to .bin files

def write_fp32(tensor, file):
    t = tensor.detach().cpu().to(torch.float32)
    b = t.numpy().tobytes()
    file.ଲିଖ(b)

def write_bf16(tensor, file):
    t = tensor.detach().cpu().to(torch.bfloat16)
    # numpy doesn't have bf16 datatype so we have to trick it
    t = t.view(torch.int16) # trick: reinterpret as int16
    b = t.numpy().tobytes()
    file.ଲିଖ(b)

def write_tensors(model_tensors, L, file, dtype):
    # writes LLaMA 3 model's weights to a binary file
    ପ୍ରତିଜ୍ଞା dtype ଭିତରେ {"float32", "bfloat16"}
    write_fun = write_fp32 ଯଦି dtype == "float32" ଅନ୍ୟଥା write_bf16
    write_fun(model_tensors["transformer.wte.weight"], file) # (V, C)
    ପାଇଁ i ଭିତରେ range(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_1.weight"], file)
    ପାଇଁ i ଭିତରେ range(L): # (L, 3C, C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_attn.weight"], file)
    ପାଇଁ i ଭିତରେ range(L): # (L, C, C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_proj.weight"], file)
    ପାଇଁ i ଭିତରେ range(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_2.weight"], file)
    ପାଇଁ i ଭିତରେ range(L): # (L, 4C, C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_fc.weight"], file)
    ପାଇଁ i ଭିତରେ range(L): # (L, 4C, C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_fc2.weight"], file)
    ପାଇଁ i ଭିତରେ range(L): # (L, C, 4C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_proj.weight"], file)
    write_fun(model_tensors["transformer.ln_f.weight"], file) # (C, )
    write_fun(model_tensors["lm_head.weight"], file) # (V, C)

def write_model(model, filename, dtype):
    # everything we need to instantiate the model
    # 1) header is: version int, LLaMAConfig ints, padding to 1024 bytes
    ପ୍ରତିଜ୍ଞା dtype ଭିତରେ {"float32", "bfloat16"}
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
    header[14] = ପୂର୍ଣ୍ଣସଂଖ୍ୟା(model.config.version.split('.')[0]) # major version
    header[15] = ପୂର୍ଣ୍ଣସଂଖ୍ୟା(model.config.version.split('.')[1]) # minor version
    # 2) the parameters follow the header
    params = {name: param.cpu() ପାଇଁ name, param ଭିତରେ model.named_parameters()}
    # now write to file
    with open(filename, "wb") ଇଵ file:
        file.ଲିଖ(header.numpy().tobytes()) # header
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
    grads = {name: param.grad.cpu() ପାଇଁ name, param ଭିତରେ model.named_parameters()}
    with open(filename, "wb") ଇଵ file:
        # header
        file.ଲିଖ(header.numpy().tobytes())
        # input x
        file.ଲିଖ(x.cpu().numpy().astype("int32").tobytes()) # (B, T)
        # targets y
        file.ଲିଖ(y.cpu().numpy().astype("int32").tobytes()) # (B, T)
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
    ଯଦି ପୂର୍ଣ୍ଣସଂଖ୍ୟା(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)

ଯଦି __name__ == "__main__":
    print0(f"Running pytorch {torch.version.__version__}")

    # default settings will overfit a tiny batch of data
    # and save model weights and debug state to disk on the first iteration
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_hf", type=ପୂର୍ଣ୍ଣସଂଖ୍ୟା, ପୂର୍ବନିର୍ଦ୍ଧାରିତ=1, help="use HuggingFace (default) or use Meta's model")
    parser.add_argument("--ckpt_dir", type=str, ପୂର୍ବନିର୍ଦ୍ଧାରିତ=None, help="path to llama3 model checkpoint (needed if use_hf=0)")
    parser.add_argument("--tokenizer_path", type=str, ପୂର୍ବନିର୍ଦ୍ଧାରିତ=None, help="path to llama3 tokenizer (needed if use_hf=0)")
    # file system input / output
    parser.add_argument("--input_bin", type=str, ପୂର୍ବନିର୍ଦ୍ଧାରିତ="dev/data/tinyshakespeare/tiny_shakespeare_val.bin", help="input .bin to train on")
    parser.add_argument("--input_val_bin", type=str, ପୂର୍ବନିର୍ଦ୍ଧାରିତ="", help="input .bin to eval validation loss on")
    parser.add_argument("--output_dir", type=str, ପୂର୍ବନିର୍ଦ୍ଧାରିତ="", help="output directory to which to write logs and checkpoints")
    parser.add_argument("--model", type=str, ପୂର୍ବନିର୍ଦ୍ଧାରିତ="meta-llama/Meta-Llama-3.1-8B", help="chose the llama model")
    # token layout for each step of the optimization
    parser.add_argument("--batch_size", type=ପୂର୍ଣ୍ଣସଂଖ୍ୟା, ପୂର୍ବନିର୍ଦ୍ଧାରିତ=4, help="batch size, in units of #batch dimensions")
    parser.add_argument("--sequence_length", type=ପୂର୍ଣ୍ଣସଂଖ୍ୟା, ପୂର୍ବନିର୍ଦ୍ଧାରିତ=64, help="sequence length")
    parser.add_argument("--total_batch_size", type=ପୂର୍ଣ୍ଣସଂଖ୍ୟା, ପୂର୍ବନିର୍ଦ୍ଧାରିତ=256, help="total desired batch size, in units of #tokens")
    # workload (number of steps)
    parser.add_argument("--num_iterations", type=ପୂର୍ଣ୍ଣସଂଖ୍ୟା, ପୂର୍ବନିର୍ଦ୍ଧାରିତ=10, help="number of iterations to run")
    parser.add_argument("--inference_only", type=ପୂର୍ଣ୍ଣସଂଖ୍ୟା, ପୂର୍ବନିର୍ଦ୍ଧାରିତ=0, help="only run inference")
    # optimization
    parser.add_argument("--learning_rate", type=ଦଶମିକ, ପୂର୍ବନିର୍ଦ୍ଧାରିତ=1e-5, help="learning rate warmup iterations")
    parser.add_argument("--warmup_iters", type=ପୂର୍ଣ୍ଣସଂଖ୍ୟା, ପୂର୍ବନିର୍ଦ୍ଧାରିତ=0, help="learning rate warmup iterations")
    parser.add_argument("--learning_rate_decay_frac", type=ଦଶମିକ, ପୂର୍ବନିର୍ଦ୍ଧାରିତ=1.0, help="learning rate warmup iterations")
    parser.add_argument("--weight_decay", type=ଦଶମିକ, ପୂର୍ବନିର୍ଦ୍ଧାରିତ=0.0, help="weight decay")
    parser.add_argument("--grad_clip", type=ଦଶମିକ, ପୂର୍ବନିର୍ଦ୍ଧାରିତ=1.0, help="maximum gradient magnitude")
    # evaluation
    parser.add_argument("--val_loss_every", type=ପୂର୍ଣ୍ଣସଂଖ୍ୟା, ପୂର୍ବନିର୍ଦ୍ଧାରିତ=0, help="every how mant steps to evaluate val loss?")
    parser.add_argument("--val_max_steps", type=ପୂର୍ଣ୍ଣସଂଖ୍ୟା, ପୂର୍ବନିର୍ଦ୍ଧାରିତ=20, help="how many batches of val to average?")
    parser.add_argument("--sample_every", type=ପୂର୍ଣ୍ଣସଂଖ୍ୟା, ପୂର୍ବନିର୍ଦ୍ଧାରିତ=0, help="how often to sample from the model?")
    # debugging
    parser.add_argument("--overfit_single_batch", type=ପୂର୍ଣ୍ଣସଂଖ୍ୟା, ପୂର୍ବନିର୍ଦ୍ଧାରିତ=1, help="overfit just one batch of data")
    # numerics
    parser.add_argument("--tensorcores", type=ପୂର୍ଣ୍ଣସଂଖ୍ୟା, ପୂର୍ବନିର୍ଦ୍ଧାରିତ=0, help="use tensorcores")
    # memory management
    parser.add_argument("--device", type=str, ପୂର୍ବନିର୍ଦ୍ଧାରିତ="", help="by default we autodetect, or set it here")
    parser.add_argument("--compile", type=ପୂର୍ଣ୍ଣସଂଖ୍ୟା, ପୂର୍ବନିର୍ଦ୍ଧାରିତ=0, help="torch.compile the model")
    parser.add_argument("--dtype", type=str, ପୂର୍ବନିର୍ଦ୍ଧାରିତ="bfloat16", help="float32|float16|bfloat16")
    parser.add_argument("--zero_stage", type=ପୂର୍ଣ୍ଣସଂଖ୍ୟା, ପୂର୍ବନିର୍ଦ୍ଧାରିତ=0, help="zero redundancy optimizer stage (0/1/2/3)")
    # python -> C bridge
    parser.add_argument("--write_tensors", type=ପୂର୍ଣ୍ଣସଂଖ୍ୟା, ପୂର୍ବନିର୍ଦ୍ଧାରିତ=0, help="write tensors to disk")
    args = parser.parse_args()

    # args error checking and convenience variables
    B, T = args.batch_size, args.sequence_length
    ପ୍ରତିଜ୍ଞା 1 <= T <= 8192, "sequence length must be between 1 and 8192"
    ପ୍ରତିଜ୍ଞା args.dtype ଭିତରେ {"float32", "float16", "bfloat16"}
    ପ୍ରତିଜ୍ଞା args.model ଭିତରେ {"meta-llama/Meta-Llama-3.1-8B"}  # only 8B base model supported for now

    # create the logging directory if it does not exist
    logfile = None
    ଯଦି args.output_dir:
        os.makedirs(args.output_dir, exist_ok=ସତ୍ୟ)
        logfile = os.path.join(args.output_dir, "main.log")
        # create the log file "main.log" inside it, and wipe it clean
        with open(logfile, "w") ଇଵ f:
            pass

    # set up DDP (distributed data parallel). torchrun sets this env variable
    ddp = ପୂର୍ଣ୍ଣସଂଖ୍ୟା(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    ଯଦି ddp:
        # use of DDP atm demands CUDA, we set the device appropriately according to rank
        ପ୍ରତିଜ୍ଞା torch.cuda.is_available(), "for now i think we need CUDA for DDP"
        init_process_group(backend='nccl')
        ddp_rank = ପୂର୍ଣ୍ଣସଂଖ୍ୟା(os.environ['RANK'])
        ddp_local_rank = ପୂର୍ଣ୍ଣସଂଖ୍ୟା(os.environ['LOCAL_RANK'])
        ddp_world_size = ପୂର୍ଣ୍ଣସଂଖ୍ୟା(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        seed_offset = 0 # each process gets the exact same seed
        zero_stage = args.zero_stage
    ଅନ୍ୟଥା:
        ddp_rank = 0
        ddp_local_rank = 0
        zero_stage = 0
        ddp_world_size = 1
        master_process = ସତ୍ୟ
        seed_offset = 0
        # select the device
        ଯଦି args.device:
            # provided explicitly by the user
            device = args.device
        ଅନ୍ୟଥା:
            # attempt to autodetect the device
            device = "cpu"
            ଯଦି torch.cuda.is_available():
                device = "cuda"
            ନଚେତଯଦି hasattr(torch.backends, "mps") ଏବଂ torch.backends.mps.is_available():
                device = "mps"
    device_type = 'cuda' ଯଦି 'cuda' ଭିତରେ device ଅନ୍ୟଥା 'cpu'
    ପ୍ରତିଜ୍ଞା device_type ଭିତରେ {'cuda'}, "GPU required to run LLaMA 3"  # we need to load LLaMA as bf16 on CUDA
    print(f"using device: {device}")

    # calculate gradient accumulation from the desired total batch size and the current run configuration
    tokens_per_fwdbwd = B * T * ddp_world_size
    ପ୍ରତିଜ୍ଞା args.total_batch_size % tokens_per_fwdbwd == 0
    grad_accum_steps = args.total_batch_size // tokens_per_fwdbwd
    print0(f"total desired batch size: {args.total_batch_size}")
    print0(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    # set up a context manager following the desired dtype and device
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) ଯଦି (device_type == "cuda") ଅନ୍ୟଥା nullcontext()

    # rng / reproducibility
    torch.manual_seed(42)
    ଯଦି torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # set the torch precision mode to use TensorFloat32 (TF32) for matmuls
    # docs https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    ଯଦି args.tensorcores:
        torch.set_float32_matmul_precision('high')

    # init the model
    ଯଦି args.use_hf:
        model = LLaMA.from_pretrained_llama3_hf(args.model)
    ଅନ୍ୟଥା:  # use Meta's checkpoint
        ପ୍ରତିଜ୍ଞା args.ckpt_dir is ନୁହେଁ None ଏବଂ os.path.exists(args.ckpt_dir), f"llama3 ckpt dir {args.ckpt_dir} does not exist"
        ପ୍ରତିଜ୍ଞା args.tokenizer_path is ନୁହେଁ None ଏବଂ os.path.exists(args.tokenizer_path), f"llama3 tokenizer path {args.tokenizer_path} does not exist"
        model = LLaMA.from_pretrained_llama3_meta(args.ckpt_dir, args.tokenizer_path)

    model.train()
    ଯଦି args.compile:
        ଯଦି hasattr(config, "coordinate_descent_tuning"):
            config.coordinate_descent_tuning = ସତ୍ୟ # suggested by @Chillee
        print0("compiling the model...")
        model = torch.compile(model)

    # -------------------------------------------------------------------------
    # Our own version of a simple DistributedDataLoader

    # load tokens
    train_loader = DistributedShardedDataLoader(args.ନିବେଶ_bin, B, T, ddp_rank, ddp_world_size)
    val_loader = None
    ଯଦି args.ନିବେଶ_val_bin:
        val_loader = DistributedShardedDataLoader(args.ନିବେଶ_val_bin, B, T, ddp_rank, ddp_world_size)

    # -------------------------------------------------------------------------
    # PyTorch -> C bridge: save some weights and state for C to load later as reference

    # do one forward pass to generate ground truth for our C tests
    ଯଦି master_process ଏବଂ args.write_tensors ଏବଂ (ନୁହେଁ args.inference_only):
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
    ଯଦି ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module ଯଦି ddp ଅନ୍ୟଥା model # always contains the "raw" unwrapped model

    # init the optimizer
    optimizer = raw_model.configure_optimizers(weight_decay=args.weight_decay,
                                               learning_rate=args.learning_rate, betas=(0.9, 0.95),
                                               device_type=device, zero_stage=zero_stage)

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        min_lr = args.learning_rate * args.learning_rate_decay_frac
        # 1) linear warmup for warmup_iters steps
        ଯଦି it < args.warmup_iters:
            ଫେରନ୍ତୁ args.learning_rate * (it+1) / args.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        ଯଦି it > args.ଅଙ୍କ_iterations:
            ଫେରନ୍ତୁ min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - args.warmup_iters) / (args.ଅଙ୍କ_iterations - args.warmup_iters)
        ପ୍ରତିଜ୍ଞା 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
        ଫେରନ୍ତୁ min_lr + coeff * (args.learning_rate - min_lr)

    ଯଦି device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    timings = []
    norm = -1.0   # dummy value to print in inference-only mode
    ପାଇଁ step ଭିତରେ range(args.ଅଙ୍କ_iterations + 1):
        t0 = time.time()
        last_step = (step == args.ଅଙ୍କ_iterations)

        # once in a while evaluate the validation dataset
        ଯଦି (args.val_loss_every > 0 \
            ଏବଂ (step % args.val_loss_every == 0 କିମ୍ବା last_step)) \
            ଏବଂ (val_loader is ନୁହେଁ None):
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss = 0.0
                ପାଇଁ _ ଭିତରେ range(args.val_max_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    _, loss = model(x, y, return_logits=ମିଥ୍ୟା)
                    val_loss += loss.item()
                val_loss /= args.val_max_steps
            # log to console and to file
            print0(f"val loss {val_loss}")
            ଯଦି master_process ଏବଂ logfile is ନୁହେଁ None:
                with open(logfile, "a") ଇଵ f:
                    f.ଲିଖ("s:%d tel:%f\n" % (step, val_loss))

        # once in a while perform model inference on the master process
        ଯଦି (args.sample_every > 0 \
            ଏବଂ (step % args.sample_every == 0 କିମ୍ବା last_step)) \
            ଏବଂ master_process:
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
            ଯଦି args.use_hf:
                prompt_tokens = [model.tokenizer(x).ନିବେଶ_ids ପାଇଁ x ଭିତରେ prompts]
            ଅନ୍ୟଥା:  # Meta
                prompt_tokens = [model.tokenizer.encode(x, bos=ସତ୍ୟ, eos=ମିଥ୍ୟା) ପାଇଁ x ଭିତରେ prompts]

            generation_tokens = model.generate(prompt_tokens, max_gen_len=64, temperature=0.6, ଶୀର୍ଷ_p=0.9, echo=ମିଥ୍ୟା)
            results = [{"generation": model.tokenizer.decode(t)} ପାଇଁ t ଭିତରେ generation_tokens]
            ପାଇଁ prompt, result ଭିତରେ zip(prompts, results):
                print(prompt, end="")
                print(f"{result['generation']}")
                print("\n==================================\n")

        # bit confusing: we want to make sure to eval and sample on 0th iteration
        # but also after the very last iteration. so we loop for step <= num_iterations
        # instead of just < num_iterations (one extra due to <=), only to do
        # the validation/sampling one last time, and then we break right here as we're done.
        ଯଦି last_step:
            ଭାଙ୍ଗ

        # --------------- TRAINING SECTION BEGIN -----------------
        model.train()
        optimizer.zero_grad(set_to_none=ସତ୍ୟ)
        # if we are trying to overfit a single batch, we reset the loader here
        ଯଦି args.overfit_single_batch:
            train_loader.reset()
        # micro-batch loop where we do gradient accumulation to reach desired total batch size
        lossf = 0.0 # for getting the mean loss (as simple float) over the accumulation steps
        ପାଇଁ micro_step ଭିତରେ range(grad_accum_steps):
            # fetch a batch
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            ଯଦି ddp:
                # we want only the last micro-step to sync grads in a DDP model
                # the official way to do this is with model.no_sync(), but that is a
                # context manager that bloats the code, so we just toggle this variable
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            # forward pass
            with ctx:
                _, loss = model(x, y, return_logits=ମିଥ୍ୟା)
                # we have to scale the loss to account for gradient accumulation,
                # because the gradients just add on each successive backward().
                # addition of gradients corresponds to a SUM in the objective, but
                # instead of a SUM we want MEAN, so we scale the loss here
                loss = loss / grad_accum_steps
                lossf += loss.detach() # keep track of the mean loss
            # backward pass
            ଯଦି ନୁହେଁ args.inference_only:
                loss.backward()
        ଯଦି ddp:
            dist.all_reduce(lossf, op=dist.ReduceOp.AVG)
        lossf = lossf.item()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        # determine and set the learning rate for this iteration
        lr = get_lr(step)
        ପାଇଁ param_group ଭିତରେ optimizer.param_groups:
            param_group['lr'] = lr
        # step the optimizer
        optimizer.step()
        # --------------- TRAINING SECTION END -------------------
        # everything that follows now is just diagnostics, prints, logging, etc.

        # wait on the CPU for all device work to end so we get accurate per-iteration timings below
        ଯଦି device == "mps":
            torch.mps.synchronize()
        ନଚେତଯଦି device == "cuda":
            torch.cuda.synchronize()
        # time and print
        t1 = time.time()
        # the 0th iteration is often an outlier (much slower) => skip logging it
        tokens_per_second = grad_accum_steps * ddp_world_size * B * T / (t1-t0)
        print0(f"step {step+1:4d}/{args.num_iterations} | train loss {lossf:.6f} | norm {norm:.4f} | lr {lr:.2e} | ({(t1-t0)*1000:.2f} ms | {tokens_per_second:.0f} tok/s)")
        # log to logile
        ଯଦି master_process ଏବଂ logfile is ନୁହେଁ None:
            with open(logfile, "a") ଇଵ f:
                f.ଲିଖ("s:%d trl:%f\n" % (step, lossf))

        # keep track of smooth timings, last 20 iterations
        ଯଦି step > 0 ଏବଂ step > args.ଅଙ୍କ_iterations - 20:
            timings.append(t1-t0)

    # print the average of the last 20 timings, to get something smooth-ish
    timings = timings[-20:]
    print0(f"final {len(timings)} iters avg: {np.mean(timings)*1000:.3f}ms")
    print0(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

    # -------------------------------------------------------------------------
    # clean up nice
    ଯଦି ddp:
        destroy_process_group()
