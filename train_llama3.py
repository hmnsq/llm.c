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

درآمد argparse
درآمد os
درآمد math
درآمد glob
درآمد inspect
سے contextlib درآمد nullcontext
سے dataclasses درآمد dataclass
سے pathlib درآمد راستہ_2
درآمد time
سے typing درآمد (
    Abstract_سیٹ,
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

درآمد numpy بطور np
درآمد torch
درآمد torch.nn بطور nn
سے torch.nn درآمد functional بطور F
درآمد torch._inductor.config بطور config
سے torch.nn.parallel درآمد Distributed_ڈیٹا_Parallel بطور DDP
سے torch.distributed درآمد init_process_گروہ, destroy_process_گروہ
سے torch.distributed.optim درآمد ZeroRedundancyOptimizer
درآمد torch.distributed بطور dist

درآمد tiktoken
سے tiktoken.بھرو درآمد بھرو_tiktoken_bpe

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the LLaMA 3.x model

# Used in Grouped Query Attention (GQA), broadcasts the key and value tensors
تعریف repeat_kv(x: torch.Tensor, n_rep: صحیح) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, سر_dim = x.shape
    اگر n_rep == 1:
        واپس x
    واپس (
        x[:, :, :, کچھ_نہیں, :]
        .expand(bs, slen, n_kv_heads, n_rep, سر_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, سر_dim)
    )

# -----------------------------------------------------------------------------
# RoPE related

تعریف reshape_for_نشر(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    أكد 0 <= 1 < ndim
    أكد freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d اگر i == 1 یا i == ndim - 1 ورنہ 1 برائے i, d میں enumerate(x.shape)]
    واپس freqs_cis.view(*shape)

تعریف apply_scaling(freqs: torch.Tensor):
    # Values obtained from grid search
    scale_ضربی_عدد = 8
    کم_freq_ضربی_عدد = 1
    بلند_freq_ضربی_عدد = 4
    old_context_len = 8192  # original llama3 length

    کم_freq_wavelen = old_context_len / کم_freq_ضربی_عدد
    بلند_freq_wavelen = old_context_len / بلند_freq_ضربی_عدد
    new_freqs = []
    برائے freq میں freqs:
        wavelen = 2 * math.pi / freq
        اگر wavelen < بلند_freq_wavelen:
            new_freqs.append(freq)
        وگرنہ_اگر wavelen > کم_freq_wavelen:
            new_freqs.append(freq / scale_ضربی_عدد)
        ورنہ:
            أكد کم_freq_wavelen != بلند_freq_wavelen
            smooth = (old_context_len / wavelen - کم_freq_ضربی_عدد) / (
                بلند_freq_ضربی_عدد - کم_freq_ضربی_عدد
            )
            new_freqs.append((1 - smooth) * freq / scale_ضربی_عدد + smooth * freq)
    واپس torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)

تعریف apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.اعشاری().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.اعشاری().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_نشر(freqs_cis, xq_)
    xq_برآمد = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_برآمد = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    واپس xq_برآمد.type_as(xq), xk_برآمد.type_as(xk)

تعریف precompute_freqs_cis(
    dim: صحیح, انتہا: صحیح, theta: اعشاری = 10000.0, use_scaled: bool = جھوٹ
):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].اعشاری() / dim))
    t = torch.arange(انتہا, device=freqs.device, dtype=torch.float32)
    اگر use_scaled:
        freqs = apply_scaling(freqs)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    واپس freqs_cis

# -----------------------------------------------------------------------------
# LLaMA building blocks

# LLaMA reference code explicitly implemented RMSNorm so we copy pasted it
# (https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/api/model.py)
# we could also use nn.RMSNorm, it has slightly different numeric properties, but equivalent
جماعت RMSNorm(torch.nn.Module):
    تعریف __init__(self, dim: صحیح, eps: اعشاری = 1e-6):
        بالا().__init__()
        self.eps = eps
        self.وزن = nn.Parameter(torch.ones(dim))

    تعریف _norm(self, x):
        واپس x * torch.rsqrt(x.pow(2).اوسط(-1, keepdim=سچ) + self.eps)

    تعریف forward(self, x):
        output = self._norm(x.اعشاری()).type_as(x)
        واپس output * self.وزن

جماعت CausalSelfAttention(nn.Module):

    تعریف __init__(self, config):
        بالا().__init__()
        أكد config.n_embd % config.n_سر == 0

        self.n_سر = config.n_سر
        self.n_kv_سر = config.n_kv_سر
        self.n_rep = self.n_سر // self.n_kv_سر
        self.hd = config.n_embd // config.n_سر
        self.use_kv = config.use_kv
        self.flash = config.flash

        self.c_attn = nn.خطی(config.n_embd, (config.n_سر + 2 * config.n_kv_سر) * self.hd, bias=جھوٹ)  # key, query, value projections
        self.c_proj = nn.خطی(config.n_embd, config.n_embd, bias=جھوٹ)  # output projection

        # static KV cache - we could alternatively allocate it outside of the model and just pass it in when needed
        اگر self.use_kv:
            self.cache_k = torch.zeros((config.max_gen_batch_سائز, config.بلاک_سائز, config.n_kv_سر, self.hd))
            self.cache_v = torch.zeros((config.max_gen_batch_سائز, config.بلاک_سائز, config.n_kv_سر, self.hd))

    تعریف forward(self, x, freqs_cis=کچھ_نہیں, آغاز_pos=کچھ_نہیں, نقاب=کچھ_نہیں):
        B, T, C = x.سائز() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.بانٹو([self.n_سر * self.hd, self.n_kv_سر * self.hd, self.n_kv_سر * self.hd], dim=-1)
        q, k, v = map(بےنام t: t.view(B, T, -1, self.hd), (q, k, v))  # (B, T, NH, HD)

        q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)  # rotate QK (rope)  <-- 1. difference compared to GPT-2

        اگر self.use_kv اور نہیں self.training اور آغاز_pos >= 0:  # use kv-caching during inference
            self.cache_k[:B, آغاز_pos : آغاز_pos + T] = k
            self.cache_v[:B, آغاز_pos : آغاز_pos + T] = v
            k = self.cache_k[:B, : آغاز_pos + T]
            v = self.cache_v[:B, : آغاز_pos + T]

        k = repeat_kv(k, self.n_rep)  # GQA <-- 2. difference compared to GPT-2
        v = repeat_kv(v, self.n_rep)

        q, k, v = map(بےنام t: t.transpose(1, 2), (q, k, v))  # (B, NH, T, HD)

        اگر self.flash:
            # flashattention
            # if T == 1 no need to mask, otherwise the function complains
            # scaled_dot_product_attention expects a mask where value of True indicates that the element should take part in attention
            # our mask is the opposite, so we need to invert it
            y = F.scaled_dot_حاصل_ضرب_attention(q, k, v, نقاب == 0 اگر T > 1 ورنہ کچھ_نہیں)
        ورنہ:
            # manual implementation of attention
            # this materializes the large (T,T) matrix for all the queries and keys
            scores = (q @ k.transpose(-2, -1)) * (1.0 / math.جذر(self.hd))
            اگر نقاب ہے نہیں کچھ_نہیں:
                scores.masked_fill_(نقاب, torch.finfo(scores.dtype).min)
            att = F.softmax(scores.اعشاری(), dim=-1).type_as(q)
            y = att @ v # (B, NH, T, T) x (B, NH, T, HD) -> (B, NH, T, HD)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        واپس y

جماعت MLP(nn.Module):

    تعریف __init__(self, config):
        بالا().__init__()
        hidden_dim = 4 * config.n_embd
        hidden_dim = صحیح(2 * hidden_dim / 3)
        # custom dim factor multiplier
        اگر config.ffn_dim_multiplier ہے نہیں کچھ_نہیں:
            hidden_dim = صحیح(config.ffn_dim_multiplier * hidden_dim)
        hidden_dim = config.مضرب_کا * ((hidden_dim + config.مضرب_کا - 1) // config.مضرب_کا)
        self.c_fc = nn.خطی(config.n_embd, hidden_dim, bias=جھوٹ)
        self.c_fc2 = nn.خطی(config.n_embd, hidden_dim, bias=جھوٹ)
        self.c_proj = nn.خطی(hidden_dim, config.n_embd, bias=جھوٹ)

    تعریف forward(self, x):
        # SwiGLU self.c_proj(F.silu(self.c_fc2(x)) * self.c_fc(x))  <-- 3. difference compared to GPT-2
        x1 = self.c_fc(x)
        x2 = self.c_fc2(x)
        x2 = F.silu(x2)
        x = x1 * x2
        x = self.c_proj(x)
        واپس x

جماعت بلاک(nn.Module):

    تعریف __init__(self, config):
        بالا().__init__()
        self.ln_1 = RMSNorm(config.n_embd, config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd, config.norm_eps)
        self.mlp = MLP(config)

    تعریف forward(self, x, freqs_cis=کچھ_نہیں, آغاز_pos=کچھ_نہیں, نقاب=کچھ_نہیں):
        x = x + self.attn(self.ln_1(x), freqs_cis, آغاز_pos, نقاب)
        x = x + self.mlp(self.ln_2(x))
        واپس x

# -----------------------------------------------------------------------------
# The main LLaMA 3.1 model

@dataclass
جماعت LlamaConfig:
    نسخہ: str = "3.1"
    بلاک_سائز: صحیح = 8192
    vocab_سائز: صحیح = 128256
    n_layer: صحیح = 32
    n_سر: صحیح = 32
    n_kv_سر: صحیح = 8
    n_embd: صحیح = 4096
    ffn_dim_multiplier: اعشاری = 1.3
    مضرب_کا: صحیح = 1024
    norm_eps: اعشاری = 1e-5
    rope_theta: اعشاری = 500000.0
    use_scaled_rope: bool = سچ
    max_gen_batch_سائز: صحیح = 4
    use_kv: bool = سچ
    flash: bool = جھوٹ  # use flashattention?

    تعریف __init__(self, **kwargs):
        برائے k, v میں kwargs.اشیاء():
            اگر hasattr(self, k):
                setattr(self, k, v)
        أكد self.n_kv_سر <= self.n_سر
        أكد self.n_سر % self.n_kv_سر == 0
        أكد self.n_embd % self.n_سر == 0

جماعت LLaMA(nn.Module):

    تعریف __init__(self, config):
        بالا().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_سائز, config.n_embd),
            h = nn.ModuleList([بلاک(config) برائے _ میں سلسلہ(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd, config.norm_eps),
        ))
        self.lm_سر = nn.خطی(config.n_embd, config.vocab_سائز, bias=جھوٹ)

        # init all weights, use a torch rng object to be very careful
        self.init_rng = torch.Generator()
        self.init_rng.manual_بیج(42)

        self.freqs_cis = precompute_freqs_cis(
            config.n_embd // config.n_سر,
            config.بلاک_سائز * 2,
            config.rope_theta,
            config.use_scaled_rope,
        )

    تعریف forward(self, idx, targets=کچھ_نہیں, return_logits=سچ, آغاز_pos=0):
        _, t = idx.سائز()
        أكد t <= self.config.بلاک_سائز, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # forward the LLaMA model itself
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        freqs_cis = self.freqs_cis[آغاز_pos:آغاز_pos+t]

        نقاب = torch.triu(torch.ones((t, t), device=اگلا_2(self.parameters()).device, dtype=torch.bool), diagonal=1)

        برائے i, بلاک_2 میں enumerate(self.transformer.h):
            x = بلاک_2(x, freqs_cis, آغاز_pos, نقاب)
        x = self.transformer.ln_f(x)

        اگر targets ہے نہیں کچھ_نہیں:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_سر(x).اعشاری()
            loss = F.cross_entropy(logits.view(-1, logits.سائز(-1)), targets.view(-1), ignore_اشاریہ=-1)
        ورنہ:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_سر(x[:, [-1], :]).اعشاری() # note: using list [-1] to preserve the time dim
            loss = کچھ_نہیں

        # there are performance reasons why not returning logits is prudent, if not needed
        اگر نہیں return_logits:
            logits = کچھ_نہیں

        واپس logits, loss

    @staticmethod
    تعریف adapt_llama_state_dict_کلیدیں(checkpoint, config: LlamaConfig):
        # Modify key names from Meta's LLaMA to our LLaMA
        # our key names are derived from GPT-2's key names
        checkpoint['transformer.wte.weight'] = checkpoint.نکالو('tok_embeddings.weight')

        برائے i میں سلسلہ(config.n_layer):
            برائے name میں ['attention_norm', 'ffn_norm']:
                old_کلید = f'layers.{i}.{name}.weight'  # e.g. layers.x.attention_norm.weight -> transformer.h.x.ln_1.weight
                new_کلید = f'transformer.h.{i}.ln_{1 if name == "attention_norm" else 2}.weight'
                checkpoint[new_کلید] = checkpoint.نکالو(old_کلید)

        برائے i میں سلسلہ(config.n_layer):
            برائے name میں ['attention.wq', 'attention.wk', 'attention.wv']:
                old_کلید = f'layers.{i}.{name}.weight'
                new_کلید = f'transformer.h.{i}.attn.c_attn.weight'
                اگر name == 'attention.wq':
                    checkpoint[new_کلید] = checkpoint.نکالو(old_کلید)
                ورنہ:  # merge 3 weights into transformer.h.x.attn.c_attn.weight
                    checkpoint[new_کلید] = torch.cat((checkpoint[new_کلید], checkpoint.نکالو(old_کلید)), dim=0)
            old_کلید = f'layers.{i}.attention.wo.weight'
            new_کلید = f'transformer.h.{i}.attn.c_proj.weight'
            checkpoint[new_کلید] = checkpoint.نکالو(old_کلید)

        ffn_map = {'w1': 'c_fc2', 'w2': 'c_proj', 'w3': 'c_fc'}
        برائے i میں سلسلہ(config.n_layer):
            برائے name میں ['feed_forward.w1', 'feed_forward.w2', 'feed_forward.w3']:
                old_کلید = f'layers.{i}.{name}.weight'
                new_کلید = f'transformer.h.{i}.mlp.{ffn_map[name.split(".")[-1]]}.weight'
                checkpoint[new_کلید] = checkpoint.نکالو(old_کلید)

        checkpoint['transformer.ln_f.weight'] = checkpoint.نکالو('norm.weight')
        checkpoint['lm_head.weight'] = checkpoint.نکالو('output.weight')

        واپس checkpoint

    @staticmethod
    تعریف adapt_llama_state_dict_کلیدیں_hf(checkpoint, config: LlamaConfig):
        # Modify key names from HuggingFace's LLaMA to our LLaMA
        # our key names are derived from GPT-2's key names
        checkpoint['transformer.wte.weight'] = checkpoint.نکالو('model.embed_tokens.weight')

        # We need to unpermute K and V because HF script permuted the original Meta-LLaMA weights
        # see: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py
        تعریف unpermute(w, n_heads, dim1, dim2):
            واپس w.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)

        برائے i میں سلسلہ(config.n_layer):
            برائے name میں ['input_layernorm', 'post_attention_layernorm']:
                old_کلید = f'model.layers.{i}.{name}.weight'  # e.g. layers.x.attention_norm.weight -> transformer.h.x.ln_1.weight
                new_کلید = f'transformer.h.{i}.ln_{1 if name == "input_layernorm" else 2}.weight'
                checkpoint[new_کلید] = checkpoint.نکالو(old_کلید)

        برائے i میں سلسلہ(config.n_layer):
            برائے name میں ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj']:
                old_کلید = f'model.layers.{i}.{name}.weight'
                new_کلید = f'transformer.h.{i}.attn.c_attn.weight'
                اگر name == 'self_attn.q_proj':
                    checkpoint[new_کلید] = unpermute(checkpoint.نکالو(old_کلید), config.n_سر, config.n_embd, config.n_embd)
                ورنہ:  # merge 3 weights into transformer.h.x.attn.c_attn.weight
                    tensor = checkpoint.نکالو(old_کلید)
                    اگر name == 'self_attn.k_proj':
                        tensor = unpermute(tensor, config.n_kv_سر, config.n_kv_سر * (config.n_embd // config.n_سر), config.n_embd)
                    checkpoint[new_کلید] = torch.cat((checkpoint[new_کلید], tensor), dim=0)
            old_کلید = f'model.layers.{i}.self_attn.o_proj.weight'
            new_کلید = f'transformer.h.{i}.attn.c_proj.weight'
            checkpoint[new_کلید] = checkpoint.نکالو(old_کلید)

        ffn_map = {'gate_proj': 'c_fc2', 'down_proj': 'c_proj', 'up_proj': 'c_fc'}
        برائے i میں سلسلہ(config.n_layer):
            برائے name میں ['gate_proj', 'down_proj', 'up_proj']:
                old_کلید = f'model.layers.{i}.mlp.{name}.weight'
                new_کلید = f'transformer.h.{i}.mlp.{ffn_map[name]}.weight'
                checkpoint[new_کلید] = checkpoint.نکالو(old_کلید)

        checkpoint['transformer.ln_f.weight'] = checkpoint.نکالو('model.norm.weight')

        واپس checkpoint

    @classmethod
    تعریف from_pretrained_llama3_hf(cls, model_id):
        """Loads pretrained LLaMA model weights from HuggingFace"""
        سے transformers درآمد AutoModelForCausalLM, AutoTokenizer
        أكد model_id == "meta-llama/Meta-Llama-3.1-8B", "Only the 8B-base model is supported for now"
        model_دلائل = LlamaConfig()

        model = AutoModelForCausalLM.from_pretrained(model_id)
        checkpoint = LLaMA.adapt_llama_state_dict_کلیدیں_hf(model.state_dict(), model_دلائل)

        original_default_type = torch.get_default_dtype()  # save the default type
        torch.سیٹ_default_tensor_type(torch.cuda.BFloat16Tensor)  # much faster loading
        model = LLaMA(model_دلائل)
        model.بھرو_state_dict(checkpoint, strict=جھوٹ)
        torch.سیٹ_default_tensor_type(torch.tensor([], dtype=original_default_type, device="cpu").type())  # restore default type

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_id = 128004  # this is the pad token id for LLaMA 3.1 base, we need to set this explicitly as our generate func expects it
        tokenizer.stop_علامات = [tokenizer.eos_token_id]
        model.tokenizer = tokenizer
        واپس model

    @classmethod
    تعریف from_pretrained_llama3_meta(cls, ckpt_dir, tokenizer_راستہ):
        """Loads pretrained LLaMA model weights from a checkpoint directory"""
        model_دلائل = LlamaConfig()

        ckpt_راستہ = sorted(راستہ_2(ckpt_dir).glob("*.pth"))[0]
        checkpoint = torch.بھرو(ckpt_راستہ, map_location="cpu", weights_only=سچ)
        checkpoint = LLaMA.adapt_llama_state_dict_کلیدیں(checkpoint, model_دلائل)

        original_default_type = torch.get_default_dtype()  # save the default type
        torch.سیٹ_default_tensor_type(torch.cuda.BFloat16Tensor)  # much faster loading
        model = LLaMA(model_دلائل)
        model.بھرو_state_dict(checkpoint, strict=جھوٹ)
        torch.سیٹ_default_tensor_type(torch.tensor([], dtype=original_default_type, device="cpu").type())  # restore default type

        tokenizer = Tokenizer(model_راستہ=tokenizer_راستہ)
        model.tokenizer = tokenizer
        واپس model

    تعریف configure_optimizers(self, وزن_decay, learning_شرح, betas, device_type, zero_stage):
        # start with all of the candidate parameters
        param_dict = {pn: p برائے pn, p میں self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p برائے pn, p میں param_dict.اشیاء() اگر p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p برائے n, p میں param_dict.اشیاء() اگر p.dim() >= 2]
        nodecay_params = [p برائے n, p میں param_dict.اشیاء() اگر p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': وزن_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        عدد_decay_params = مجموع(p.numel() برائے p میں decay_params)
        عدد_nodecay_params = مجموع(p.numel() برائے p میں nodecay_params)
        print0(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print0(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_دستیاب = 'fused' میں inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_دستیاب اور device_type == 'cuda'
        print0(f"using fused AdamW: {use_fused}")
        اگر zero_stage == 1:
            print0("using ZeroRedundancyOptimizer")
            optimizer = ZeroRedundancyOptimizer(**optim_groups[0], optimizer_class=torch.optim.AdamW,
                                                lr=learning_شرح, betas=betas, fused=use_fused)
            optimizer.جوڑو_param_گروہ(optim_groups[1])
        ورنہ:
            print0("using regular AdamW")
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_شرح, betas=betas, fused=use_fused)
        واپس optimizer

    @torch.inference_منوال()
    تعریف generate(
        self,
        prompt_علامات: List[List[صحیح]],
        max_gen_len: صحیح,
        temperature: اعشاری = 0.6,
        چوٹی_p: اعشاری = 0.9,
        echo: bool = جھوٹ,
    ) -> Tuple[List[List[صحیح]], Optional[List[List[اعشاری]]]]:
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
        bsz = لمبائی(prompt_علامات)
        أكد bsz <= self.config.max_gen_batch_سائز, f"Batch size {bsz} exceeds the maximum generation batch size {self.config.max_gen_batch_size}"
        device = اگلا_2(self.parameters()).device

        min_prompt_len = min(لمبائی(t) برائے t میں prompt_علامات)
        max_prompt_len = max(لمبائی(t) برائے t میں prompt_علامات)
        أكد max_prompt_len <= self.config.بلاک_سائز, f"Prompt length {max_prompt_len} exceeds the maximum block size {self.config.block_size}"
        کل_len = min(self.config.بلاک_سائز, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_id
        علامات = torch.بھرا((bsz, کل_len), pad_id, dtype=torch.طویل, device=device)
        برائے idx, t میں enumerate(prompt_علامات):
            علامات[idx, : لمبائی(t)] = torch.tensor(t, dtype=torch.طویل, device=device)

        پچھلا_pos = 0
        eos_reached = torch.tensor([جھوٹ] * bsz, device=device)
        داخلہ_متن_نقاب = علامات != pad_id

        اگر min_prompt_len == کل_len:
            logits, _ = self.forward(علامات, آغاز_pos=پچھلا_pos)

        stop_علامات = torch.tensor(list(self.tokenizer.stop_علامات)).مقصد(device)

        برائے cur_pos میں سلسلہ(min_prompt_len, کل_len):
            logits, _ = self.forward(علامات[:, پچھلا_pos:cur_pos], آغاز_pos=پچھلا_pos)
            اگر temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                اگلا_token = sample_چوٹی_p(probs, چوٹی_p)
            ورنہ:
                اگلا_token = torch.argmax(logits[:, -1], dim=-1)

            اگلا_token = اگلا_token.reshape(-1)
            # only replace token if prompt has already been generated
            اگلا_token = torch.where(داخلہ_متن_نقاب[:, cur_pos], علامات[:, cur_pos], اگلا_token)
            علامات[:, cur_pos] = اگلا_token
            eos_reached |= ~داخلہ_متن_نقاب[:, cur_pos] & torch.isin(اگلا_token, stop_علامات)
            پچھلا_pos = cur_pos
            اگر all(eos_reached):
                توڑو

        برآمد_علامات = []
        برائے i, toks میں enumerate(علامات.tolist()):
            # cut to max gen len
            آغاز = 0 اگر echo ورنہ لمبائی(prompt_علامات[i])
            toks = toks[آغاز : لمبائی(prompt_علامات[i]) + max_gen_len]
            # cut to after eos tok if any
            برائے stop_token میں self.tokenizer.stop_علامات:
                کوشش:
                    eos_idx = toks.اشاریہ(stop_token)
                    toks = toks[:eos_idx]
                استثنا قدر_Error:
                    گزرو
            برآمد_علامات.append(toks)
        واپس برآمد_علامات

# -----------------------------------------------------------------------------
# sampling utils

تعریف sample_چوٹی_p(probs, p):
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
    probs_ترتیب_دو, probs_idx = torch.ترتیب_دو(probs, dim=-1, descending=سچ)
    probs_مجموع = torch.cumsum(probs_ترتیب_دو, dim=-1)
    نقاب = probs_مجموع - probs_ترتیب_دو > p
    probs_ترتیب_دو[نقاب] = 0.0
    probs_ترتیب_دو.div_(probs_ترتیب_دو.مجموع(dim=-1, keepdim=سچ))
    اگلا_token = torch.multinomial(probs_ترتیب_دو, عدد_samples=1)
    اگلا_token = torch.gather(probs_idx, -1, اگلا_token)
    واپس اگلا_token

# -----------------------------------------------------------------------------
# Llama 3.1 Tokenizer

# The tiktoken tokenizer can handle <=400k chars without
# pyo3_runtime.PanicException.
TIKTOKEN_MAX_ENCODE_CHARS = 400_000

# https://github.com/openai/tiktoken/issues/195
# Here we iterate over subsequences and split if we exceed the limit
# of max consecutive non-whitespace or whitespace characters.
MAX_NO_WHITESPACES_CHARS = 25_000


جماعت Tokenizer:
    """
    Tokenizing and encoding/decoding text using the Tiktoken tokenizer.
    """

    special_علامات_2: Dict[str, صحیح]

    عدد_reserved_special_علامات = 256

    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: E501

    تعریف __init__(self, model_راستہ: str):
        """
        Initializes the Tokenizer with a Tiktoken model.

        Args:
            model_path (str): The path to the Tiktoken model file.
        """
        أكد os.راستہ.isfile(model_راستہ), model_راستہ

        mergeable_ranks = بھرو_tiktoken_bpe(model_راستہ)
        عدد_بنیاد_علامات = لمبائی(mergeable_ranks)
        special_علامات_2 = [
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
        reserved_علامات = [
            f"<|reserved_special_token_{2 + i}|>"
            برائے i میں سلسلہ(self.عدد_reserved_special_علامات - لمبائی(special_علامات_2))
        ]
        special_علامات_2 = special_علامات_2 + reserved_علامات

        self.special_علامات_2 = {
            token: عدد_بنیاد_علامات + i برائے i, token میں enumerate(special_علامات_2)
        }
        self.model = tiktoken.Encoding(
            name=راستہ_2(model_راستہ).name,
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_علامات_2=self.special_علامات_2,
        )

        self.n_الفاظ: صحیح = عدد_بنیاد_علامات + لمبائی(special_علامات_2)
        # BOS / EOS token IDs
        self.bos_id: صحیح = self.special_علامات_2["<|begin_of_text|>"]
        self.eos_id: صحیح = self.special_علامات_2["<|end_of_text|>"]
        self.eot_id: صحیح = self.special_علامات_2["<|eot_id|>"]
        self.eom_id: صحیح = self.special_علامات_2["<|eom_id|>"]
        self.python_لیبل_id = self.special_علامات_2["<|python_tag|>"]
        self.pad_id: صحیح = self.special_علامات_2["<|finetune_right_pad_id|>"]
        # hardcoded stop tokens for the base model
        self.stop_علامات = [
            self.special_علامات_2["<|begin_of_text|>"],
            self.special_علامات_2["<|end_of_text|>"],
        ]

    تعریف encode(
        self,
        s: str,
        *,
        bos: bool,
        eos: bool,
        allowed_special: Optional[Union[Literal["all"], Abstract_سیٹ[str]]] = کچھ_نہیں,
        disallowed_special: Union[Literal["all"], Collection[str]] = (),
    ) -> List[صحیح]:
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
        اگر allowed_special ہے کچھ_نہیں:
            allowed_special = سیٹ()
        أكد type(s) ہے str

        substrs = (
            substr
            برائے i میں سلسلہ(0, لمبائی(s), TIKTOKEN_MAX_ENCODE_CHARS)
            برائے substr میں self.بانٹو_whitespaces_or_nonwhitespaces(
                s[i : i + TIKTOKEN_MAX_ENCODE_CHARS], MAX_NO_WHITESPACES_CHARS
            )
        )
        t: List[صحیح] = []
        برائے substr میں substrs:
            t.extend(
                self.model.encode(
                    substr,
                    allowed_special=allowed_special,
                    disallowed_special=disallowed_special,
                )
            )
        اگر bos:
            t.داخل_کرو(0, self.bos_id)
        اگر eos:
            t.append(self.eos_id)
        واپس t

    تعریف decode(self, t: Sequence[صحیح]) -> str:
        # Typecast is safe here. Tiktoken doesn't do anything list-related with the sequence.
        واپس self.model.decode(cast(List[صحیح], t))

    @staticmethod
    تعریف بانٹو_whitespaces_or_nonwhitespaces(
        s: str, max_consecutive_قاش_len: صحیح
    ) -> Iterator[str]:
        """
        Splits the string `s` so that each substring contains no more than `max_consecutive_slice_len`
        consecutive whitespaces or consecutive non-whitespaces.
        """
        current_قاش_len = 0
        current_قاش_is_space = s[0].isspace() اگر لمبائی(s) > 0 ورنہ جھوٹ
        قاش_آغاز = 0

        برائے i میں سلسلہ(لمبائی(s)):
            is_now_space = s[i].isspace()

            اگر current_قاش_is_space ^ is_now_space:
                current_قاش_len = 1
                current_قاش_is_space = is_now_space
            ورنہ:
                current_قاش_len += 1
                اگر current_قاش_len > max_consecutive_قاش_len:
                    دو s[قاش_آغاز:i]
                    قاش_آغاز = i
                    current_قاش_len = 1
        دو s[قاش_آغاز:]

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

تعریف peek_ڈیٹا_shard(filename):
    # only reads the header, returns header data
    ساتھ open(filename, "rb") بطور f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        سرخی = np.frombuffer(f.اقرأ(256*4), dtype=np.int32)
    اگر سرخی[0] != 20240801:
        چھاپو("ERROR: magic number mismatch in the data .bin file!")
        exit(1)
    أكد سرخی[1] == 7, "unsupported version"
    ntok = سرخی[2] # number of tokens (claimed)
    واپس ntok # for now just return the number of tokens

تعریف بھرو_ڈیٹا_shard(filename):
    ساتھ open(filename, "rb") بطور f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        سرخی = np.frombuffer(f.اقرأ(256*4), dtype=np.int32)
        أكد سرخی[0] == 20240801, "magic number mismatch in the data .bin file"
        أكد سرخی[1] == 7, "unsupported version"
        ntok = سرخی[2] # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        علامات = np.frombuffer(f.اقرأ(), dtype=np.uint32)
    أكد لمبائی(علامات) == ntok, "number of tokens read does not match header?"
    واپس علامات

جماعت Distributed_Sharded_ڈیٹا_Loader:
    """
    This DataLoader is both:
    - distributed (works correctly in case of multiple processes in DDP)
    - sharded (supports datasets that are broken up into multiple data shards)
    It is not *permuted*, meaning that it itearates over the data in the order
    of the dataset on disk, so the user should make sure to shuffle their examples
    during the creation of their data shards for best performance.
    """
    تعریف __init__(self, filename_نمونہ, B, T, process_rank, عدد_processes):
        self.process_rank = process_rank
        self.عدد_processes = عدد_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_نمونہ))
        أكد لمبائی(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_کل = 0
        برائے fname میں self.files:
            shard_ntok = peek_ڈیٹا_shard(fname)
            أكد shard_ntok >= عدد_processes * B * T + 1
            ntok_کل += shard_ntok
        self.ntok_کل = ntok_کل
        print0(f"DataLoader: total number of tokens: {ntok_total:,} across {len(self.files)} files")

        # kick things off
        self.current_shard = کچھ_نہیں
        self.reset()

    تعریف reset(self):
        # we're being a bit clever here: if we already had shard 0 loaded,
        # then don't do the work to reload it, just reset the pointer
        اگر self.current_shard != 0:
            self.current_shard = 0
            self.علامات = بھرو_ڈیٹا_shard(self.files[self.current_shard])
        self.current_position = self.process_rank * self.B * self.T

    تعریف advance(self): # advance to next data shard
        self.current_shard = (self.current_shard + 1) % لمبائی(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.علامات = بھرو_ڈیٹا_shard(self.files[self.current_shard])

    تعریف اگلا_batch(self):
        B = self.B
        T = self.T
        buf = self.علامات[self.current_position : self.current_position+B*T+1]
        buf = torch.tensor(buf, dtype=torch.طویل)
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the start pointer in current shard
        self.current_position += B * T * self.عدد_processes
        # if loading the next batch would be out of bounds advance the shard
        اگر self.current_position + (B * T * self.عدد_processes + 1) > لمبائی(self.علامات):
            self.advance()
        واپس x, y

# -----------------------------------------------------------------------------
# Python -> C bridge utilities for saving params/grads/activations to .bin files

تعریف write_fp32(tensor, file):
    t = tensor.detach().cpu().مقصد(torch.float32)
    b = t.numpy().tobytes()
    file.اكتب(b)

تعریف write_bf16(tensor, file):
    t = tensor.detach().cpu().مقصد(torch.bfloat16)
    # numpy doesn't have bf16 datatype so we have to trick it
    t = t.view(torch.int16) # trick: reinterpret as int16
    b = t.numpy().tobytes()
    file.اكتب(b)

تعریف write_tensors(model_tensors, L, file, dtype):
    # writes LLaMA 3 model's weights to a binary file
    أكد dtype میں {"float32", "bfloat16"}
    write_fun = write_fp32 اگر dtype == "float32" ورنہ write_bf16
    write_fun(model_tensors["transformer.wte.weight"], file) # (V, C)
    برائے i میں سلسلہ(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_1.weight"], file)
    برائے i میں سلسلہ(L): # (L, 3C, C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_attn.weight"], file)
    برائے i میں سلسلہ(L): # (L, C, C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_proj.weight"], file)
    برائے i میں سلسلہ(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_2.weight"], file)
    برائے i میں سلسلہ(L): # (L, 4C, C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_fc.weight"], file)
    برائے i میں سلسلہ(L): # (L, 4C, C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_fc2.weight"], file)
    برائے i میں سلسلہ(L): # (L, C, 4C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_proj.weight"], file)
    write_fun(model_tensors["transformer.ln_f.weight"], file) # (C, )
    write_fun(model_tensors["lm_head.weight"], file) # (V, C)

تعریف write_model(model, filename, dtype):
    # everything we need to instantiate the model
    # 1) header is: version int, LLaMAConfig ints, padding to 1024 bytes
    أكد dtype میں {"float32", "bfloat16"}
    نسخہ = {
        "float32": 3, # 3: all tensors are fp32
        "bfloat16": 5, # 5: all tensors are bf16
    }[dtype]
    سرخی = torch.zeros(256, dtype=torch.int32)
    سرخی[0] = 20240803 # magic
    سرخی[1] = نسخہ # checkpoint version
    سرخی[2] = model.config.بلاک_سائز
    سرخی[3] = model.config.vocab_سائز
    سرخی[4] = model.config.n_layer
    سرخی[5] = model.config.n_سر
    سرخی[6] = model.config.n_kv_سر
    سرخی[7] = model.config.n_embd
    سرخی[8] = model.config.ffn_dim_multiplier
    سرخی[9] = model.config.مضرب_کا
    سرخی[10] = model.config.norm_eps
    سرخی[11] = model.config.rope_theta
    سرخی[12] = model.config.use_scaled_rope
    سرخی[13] = model.config.max_gen_batch_سائز
    سرخی[14] = صحیح(model.config.نسخہ.بانٹو('.')[0]) # major version
    سرخی[15] = صحیح(model.config.نسخہ.بانٹو('.')[1]) # minor version
    # 2) the parameters follow the header
    params = {name: param.cpu() برائے name, param میں model.named_parameters()}
    # now write to file
    ساتھ open(filename, "wb") بطور file:
        file.اكتب(سرخی.numpy().tobytes()) #سرخی
        write_tensors(params, model.config.n_layer, file, dtype) # params
    چھاپو(f"wrote {filename}")

تعریف write_state(model, x, y, logits, loss, filename):
    # the state is used for debugging.
    # it contains information about the input, logits, loss, and the parameter gradients
    # this can be used for checking the computation correctness in C
    سرخی = torch.zeros(256, dtype=torch.int32)
    سرخی[0] = 20240803 # magic
    سرخی[1] = x.سائز(0) # batch size of the batch, B
    سرخی[2] = x.سائز(1) # temporal extent of the batch, T
    grads = {name: param.grad.cpu() برائے name, param میں model.named_parameters()}
    ساتھ open(filename, "wb") بطور file:
        #سرخی
        file.اكتب(سرخی.numpy().tobytes())
        # input x
        file.اكتب(x.cpu().numpy().astype("int32").tobytes()) # (B, T)
        # targets y
        file.اكتب(y.cpu().numpy().astype("int32").tobytes()) # (B, T)
        # logits (result of the model forward pass)
        write_fp32(logits.cpu(), file)
        # loss (single float, result of the cross entropy loss)
        write_fp32(loss.cpu(), file)
        # gradients
        write_tensors(grads, model.config.n_layer, file, "float32")
    چھاپو(f"wrote {filename}")

# -----------------------------------------------------------------------------
# int main

تعریف print0(*دلائل, **kwargs):
    # modified print that only prints from the master process
    # if this is not a distributed run, it's just a print
    اگر صحیح(os.environ.get("RANK", 0)) == 0:
        چھاپو(*دلائل, **kwargs)

اگر __name__ == "__main__":
    print0(f"Running pytorch {torch.version.__version__}")

    # default settings will overfit a tiny batch of data
    # and save model weights and debug state to disk on the first iteration
    parser = argparse.دلیل_Parser()
    parser.جوڑو_دلیل("--use_hf", type=صحیح, طےشدہ=1, help="use HuggingFace (default) or use Meta's model")
    parser.جوڑو_دلیل("--ckpt_dir", type=str, طےشدہ=کچھ_نہیں, help="path to llama3 model checkpoint (needed if use_hf=0)")
    parser.جوڑو_دلیل("--tokenizer_path", type=str, طےشدہ=کچھ_نہیں, help="path to llama3 tokenizer (needed if use_hf=0)")
    # file system input / output
    parser.جوڑو_دلیل("--input_bin", type=str, طےشدہ="dev/data/tinyshakespeare/tiny_shakespeare_val.bin", help="input .bin to train on")
    parser.جوڑو_دلیل("--input_val_bin", type=str, طےشدہ="", help="input .bin to eval validation loss on")
    parser.جوڑو_دلیل("--output_dir", type=str, طےشدہ="", help="output directory to which to write logs and checkpoints")
    parser.جوڑو_دلیل("--model", type=str, طےشدہ="meta-llama/Meta-Llama-3.1-8B", help="chose the llama model")
    # token layout for each step of the optimization
    parser.جوڑو_دلیل("--batch_size", type=صحیح, طےشدہ=4, help="batch size, in units of #batch dimensions")
    parser.جوڑو_دلیل("--sequence_length", type=صحیح, طےشدہ=64, help="sequence length")
    parser.جوڑو_دلیل("--total_batch_size", type=صحیح, طےشدہ=256, help="total desired batch size, in units of #tokens")
    # workload (number of steps)
    parser.جوڑو_دلیل("--num_iterations", type=صحیح, طےشدہ=10, help="number of iterations to run")
    parser.جوڑو_دلیل("--inference_only", type=صحیح, طےشدہ=0, help="only run inference")
    # optimization
    parser.جوڑو_دلیل("--learning_rate", type=اعشاری, طےشدہ=1e-5, help="learning rate warmup iterations")
    parser.جوڑو_دلیل("--warmup_iters", type=صحیح, طےشدہ=0, help="learning rate warmup iterations")
    parser.جوڑو_دلیل("--learning_rate_decay_frac", type=اعشاری, طےشدہ=1.0, help="learning rate warmup iterations")
    parser.جوڑو_دلیل("--weight_decay", type=اعشاری, طےشدہ=0.0, help="weight decay")
    parser.جوڑو_دلیل("--grad_clip", type=اعشاری, طےشدہ=1.0, help="maximum gradient magnitude")
    # evaluation
    parser.جوڑو_دلیل("--val_loss_every", type=صحیح, طےشدہ=0, help="every how mant steps to evaluate val loss?")
    parser.جوڑو_دلیل("--val_max_steps", type=صحیح, طےشدہ=20, help="how many batches of val to average?")
    parser.جوڑو_دلیل("--sample_every", type=صحیح, طےشدہ=0, help="how often to sample from the model?")
    # debugging
    parser.جوڑو_دلیل("--overfit_single_batch", type=صحیح, طےشدہ=1, help="overfit just one batch of data")
    # numerics
    parser.جوڑو_دلیل("--tensorcores", type=صحیح, طےشدہ=0, help="use tensorcores")
    # memory management
    parser.جوڑو_دلیل("--device", type=str, طےشدہ="", help="by default we autodetect, or set it here")
    parser.جوڑو_دلیل("--compile", type=صحیح, طےشدہ=0, help="torch.compile the model")
    parser.جوڑو_دلیل("--dtype", type=str, طےشدہ="bfloat16", help="float32|float16|bfloat16")
    parser.جوڑو_دلیل("--zero_stage", type=صحیح, طےشدہ=0, help="zero redundancy optimizer stage (0/1/2/3)")
    # python -> C bridge
    parser.جوڑو_دلیل("--write_tensors", type=صحیح, طےشدہ=0, help="write tensors to disk")
    دلائل = parser.parse_دلائل()

    # args error checking and convenience variables
    B, T = دلائل.batch_سائز, دلائل.sequence_لمبائی
    أكد 1 <= T <= 8192, "sequence length must be between 1 and 8192"
    أكد دلائل.dtype میں {"float32", "float16", "bfloat16"}
    أكد دلائل.model میں {"meta-llama/Meta-Llama-3.1-8B"}  # only 8B base model supported for now

    # create the logging directory if it does not exist
    logfile = کچھ_نہیں
    اگر دلائل.output_dir:
        os.makedirs(دلائل.output_dir, exist_ok=سچ)
        logfile = os.راستہ.join(دلائل.output_dir, "main.log")
        # create the log file "main.log" inside it, and wipe it clean
        ساتھ open(logfile, "w") بطور f:
            گزرو

    # set up DDP (distributed data parallel). torchrun sets this env variable
    ddp = صحیح(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    اگر ddp:
        # use of DDP atm demands CUDA, we set the device appropriately according to rank
        أكد torch.cuda.is_دستیاب(), "for now i think we need CUDA for DDP"
        init_process_گروہ(backend='nccl')
        ddp_rank = صحیح(os.environ['RANK'])
        ddp_local_rank = صحیح(os.environ['LOCAL_RANK'])
        ddp_world_سائز = صحیح(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.سیٹ_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        بیج_انحراف = 0 # each process gets the exact same seed
        zero_stage = دلائل.zero_stage
    ورنہ:
        ddp_rank = 0
        ddp_local_rank = 0
        zero_stage = 0
        ddp_world_سائز = 1
        master_process = سچ
        بیج_انحراف = 0
        # select the device
        اگر دلائل.device:
            # provided explicitly by the user
            device = دلائل.device
        ورنہ:
            # attempt to autodetect the device
            device = "cpu"
            اگر torch.cuda.is_دستیاب():
                device = "cuda"
            وگرنہ_اگر hasattr(torch.backends, "mps") اور torch.backends.mps.is_دستیاب():
                device = "mps"
    device_type = 'cuda' اگر 'cuda' میں device ورنہ 'cpu'
    أكد device_type میں {'cuda'}, "GPU required to run LLaMA 3"  # we need to load LLaMA as bf16 on CUDA
    چھاپو(f"using device: {device}")

    # calculate gradient accumulation from the desired total batch size and the current run configuration
    علامات_فی_fwdbwd = B * T * ddp_world_سائز
    أكد دلائل.کل_batch_سائز % علامات_فی_fwdbwd == 0
    grad_accum_اقدام = دلائل.کل_batch_سائز // علامات_فی_fwdbwd
    print0(f"total desired batch size: {args.total_batch_size}")
    print0(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    # set up a context manager following the desired dtype and device
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[دلائل.dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) اگر (device_type == "cuda") ورنہ nullcontext()

    # rng / reproducibility
    torch.manual_بیج(42)
    اگر torch.cuda.is_دستیاب():
        torch.cuda.manual_بیج(42)

    # set the torch precision mode to use TensorFloat32 (TF32) for matmuls
    # docs https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    اگر دلائل.tensorcores:
        torch.سیٹ_float32_matmul_precision('high')

    # init the model
    اگر دلائل.use_hf:
        model = LLaMA.from_pretrained_llama3_hf(دلائل.model)
    ورنہ:  # use Meta's checkpoint
        أكد دلائل.ckpt_dir ہے نہیں کچھ_نہیں اور os.راستہ.exists(دلائل.ckpt_dir), f"llama3 ckpt dir {args.ckpt_dir} does not exist"
        أكد دلائل.tokenizer_راستہ ہے نہیں کچھ_نہیں اور os.راستہ.exists(دلائل.tokenizer_راستہ), f"llama3 tokenizer path {args.tokenizer_path} does not exist"
        model = LLaMA.from_pretrained_llama3_meta(دلائل.ckpt_dir, دلائل.tokenizer_راستہ)

    model.train()
    اگر دلائل.compile:
        اگر hasattr(config, "coordinate_descent_tuning"):
            config.coordinate_descent_tuning = سچ # suggested by @Chillee
        print0("compiling the model...")
        model = torch.compile(model)

    # -------------------------------------------------------------------------
    # Our own version of a simple DistributedDataLoader

    # load tokens
    train_loader = Distributed_Sharded_ڈیٹا_Loader(دلائل.داخلہ_bin, B, T, ddp_rank, ddp_world_سائز)
    val_loader = کچھ_نہیں
    اگر دلائل.داخلہ_val_bin:
        val_loader = Distributed_Sharded_ڈیٹا_Loader(دلائل.داخلہ_val_bin, B, T, ddp_rank, ddp_world_سائز)

    # -------------------------------------------------------------------------
    # PyTorch -> C bridge: save some weights and state for C to load later as reference

    # do one forward pass to generate ground truth for our C tests
    اگر master_process اور دلائل.write_tensors اور (نہیں دلائل.inference_only):
        x, y = train_loader.اگلا_batch()
        x, y = x.مقصد(device), y.مقصد(device)
        logits, loss = model(x, y)
        loss.backward()
        # save model params, in bfloat16
        model_مقصد_سائز = {"meta-llama/Meta-Llama-3.1-8B": "8B"}
        model_سائز_str = model_مقصد_سائز[دلائل.model] # e.g. "8B"
        write_model(model, os.راستہ.join(دلائل.output_dir, f"llama3.1_{model_size_str}_bf16.bin"), dtype="bfloat16")
        # save x, y, logits, loss, and parameter gradients, for debugging C
        # always store these in fp32 to have an accurate reference (?)
        write_state(model, x, y, logits, loss, os.راستہ.join(دلائل.output_dir, f"llama3_{model_size_str}_debug_state.bin"))
        # reset the train_loader for the optimization below
        train_loader.reset()

    # -------------------------------------------------------------------------
    # main training loop

    # here we wrap model into DDP container
    اگر ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module اگر ddp ورنہ model # always contains the "raw" unwrapped model

    # init the optimizer
    optimizer = raw_model.configure_optimizers(وزن_decay=دلائل.وزن_decay,
                                               learning_شرح=دلائل.learning_شرح, betas=(0.9, 0.95),
                                               device_type=device, zero_stage=zero_stage)

    # learning rate decay scheduler (cosine with warmup)
    تعریف get_lr(it):
        min_lr = دلائل.learning_شرح * دلائل.learning_شرح_decay_frac
        # 1) linear warmup for warmup_iters steps
        اگر it < دلائل.warmup_iters:
            واپس دلائل.learning_شرح * (it+1) / دلائل.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        اگر it > دلائل.عدد_iterations:
            واپس min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_تناسب = (it - دلائل.warmup_iters) / (دلائل.عدد_iterations - دلائل.warmup_iters)
        أكد 0 <= decay_تناسب <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_تناسب)) # coeff starts at 1 and goes to 0
        واپس min_lr + coeff * (دلائل.learning_شرح - min_lr)

    اگر device == "cuda":
        torch.cuda.reset_peak_حافظہ_stats()
    timings = []
    norm = -1.0   # dummy value to print in inference-only mode
    برائے قدم میں سلسلہ(دلائل.عدد_iterations + 1):
        t0 = time.time()
        آخری_قدم = (قدم == دلائل.عدد_iterations)

        # once in a while evaluate the validation dataset
        اگر (دلائل.val_loss_every > 0 \
            اور (قدم % دلائل.val_loss_every == 0 یا آخری_قدم)) \
            اور (val_loader ہے نہیں کچھ_نہیں):
            model.eval()
            val_loader.reset()
            ساتھ torch.no_grad():
                val_loss = 0.0
                برائے _ میں سلسلہ(دلائل.val_max_اقدام):
                    x, y = val_loader.اگلا_batch()
                    x, y = x.مقصد(device), y.مقصد(device)
                    _, loss = model(x, y, return_logits=جھوٹ)
                    val_loss += loss.شے_2()
                val_loss /= دلائل.val_max_اقدام
            # log to console and to file
            print0(f"val loss {val_loss}")
            اگر master_process اور logfile ہے نہیں کچھ_نہیں:
                ساتھ open(logfile, "a") بطور f:
                    f.اكتب("s:%d tel:%f\n" % (قدم, val_loss))

        # once in a while perform model inference on the master process
        اگر (دلائل.sample_every > 0 \
            اور (قدم % دلائل.sample_every == 0 یا آخری_قدم)) \
            اور master_process:
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
            اگر دلائل.use_hf:
                prompt_علامات = [model.tokenizer(x).داخلہ_ids برائے x میں prompts]
            ورنہ:  # Meta
                prompt_علامات = [model.tokenizer.encode(x, bos=سچ, eos=جھوٹ) برائے x میں prompts]

            generation_علامات = model.generate(prompt_علامات, max_gen_len=64, temperature=0.6, چوٹی_p=0.9, echo=جھوٹ)
            results = [{"generation": model.tokenizer.decode(t)} برائے t میں generation_علامات]
            برائے prompt, نتیجہ میں zip(prompts, results):
                چھاپو(prompt, انتہا="")
                چھاپو(f"{result['generation']}")
                چھاپو("\n==================================\n")

        # bit confusing: we want to make sure to eval and sample on 0th iteration
        # but also after the very last iteration. so we loop for step <= num_iterations
        # instead of just < num_iterations (one extra due to <=), only to do
        # the validation/sampling one last time, and then we break right here as we're done.
        اگر آخری_قدم:
            توڑو

        # --------------- TRAINING SECTION BEGIN -----------------
        model.train()
        optimizer.zero_grad(سیٹ_مقصد_none=سچ)
        # if we are trying to overfit a single batch, we reset the loader here
        اگر دلائل.overfit_single_batch:
            train_loader.reset()
        # micro-batch loop where we do gradient accumulation to reach desired total batch size
        lossf = 0.0 # for getting the mean loss (as simple float) over the accumulation steps
        برائے micro_قدم میں سلسلہ(grad_accum_اقدام):
            # fetch a batch
            x, y = train_loader.اگلا_batch()
            x, y = x.مقصد(device), y.مقصد(device)
            اگر ddp:
                # we want only the last micro-step to sync grads in a DDP model
                # the official way to do this is with model.no_sync(), but that is a
                # context manager that bloats the code, so we just toggle this variable
                model.require_backward_grad_sync = (micro_قدم == grad_accum_اقدام - 1)
            # forward pass
            ساتھ ctx:
                _, loss = model(x, y, return_logits=جھوٹ)
                # we have to scale the loss to account for gradient accumulation,
                # because the gradients just add on each successive backward().
                # addition of gradients corresponds to a SUM in the objective, but
                # instead of a SUM we want MEAN, so we scale the loss here
                loss = loss / grad_accum_اقدام
                lossf += loss.detach() # keep track of the mean loss
            # backward pass
            اگر نہیں دلائل.inference_only:
                loss.backward()
        اگر ddp:
            dist.all_reduce(lossf, op=dist.ReduceOp.AVG)
        lossf = lossf.شے_2()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), دلائل.grad_clip)
        # determine and set the learning rate for this iteration
        lr = get_lr(قدم)
        برائے param_گروہ میں optimizer.param_groups:
            param_گروہ['lr'] = lr
        # step the optimizer
        optimizer.قدم()
        # --------------- TRAINING SECTION END -------------------
        # everything that follows now is just diagnostics, prints, logging, etc.

        # wait on the CPU for all device work to end so we get accurate per-iteration timings below
        اگر device == "mps":
            torch.mps.synchronize()
        وگرنہ_اگر device == "cuda":
            torch.cuda.synchronize()
        # time and print
        t1 = time.time()
        # the 0th iteration is often an outlier (much slower) => skip logging it
        علامات_فی_دوسرا = grad_accum_اقدام * ddp_world_سائز * B * T / (t1-t0)
        print0(f"step {step+1:4d}/{args.num_iterations} | train loss {lossf:.6f} | norm {norm:.4f} | lr {lr:.2e} | ({(t1-t0)*1000:.2f} ms | {tokens_per_second:.0f} tok/s)")
        # log to logile
        اگر master_process اور logfile ہے نہیں کچھ_نہیں:
            ساتھ open(logfile, "a") بطور f:
                f.اكتب("s:%d trl:%f\n" % (قدم, lossf))

        # keep track of smooth timings, last 20 iterations
        اگر قدم > 0 اور قدم > دلائل.عدد_iterations - 20:
            timings.append(t1-t0)

    # print the average of the last 20 timings, to get something smooth-ish
    timings = timings[-20:]
    print0(f"final {len(timings)} iters avg: {np.mean(timings)*1000:.3f}ms")
    print0(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

    # -------------------------------------------------------------------------
    # clean up nice
    اگر ddp:
        destroy_process_گروہ()
