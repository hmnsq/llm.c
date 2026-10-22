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

استورد argparse
استورد os
استورد math
استورد glob
استورد inspect
من contextlib استورد nullcontext
من dataclasses استورد dataclass
من pathlib استورد مسیر_2
استورد time
من typing استورد (
    Abstract_مجموعه,
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

استورد numpy كـ np
استورد torch
استورد torch.nn كـ nn
من torch.nn استورد functional كـ F
استورد torch._inductor.config كـ config
من torch.nn.parallel استورد Distributed_داده_Parallel كـ DDP
من torch.distributed استورد init_process_گروه, destroy_process_گروه
من torch.distributed.optim استورد ZeroRedundancyOptimizer
استورد torch.distributed كـ dist

استورد tiktoken
من tiktoken.بارگیری استورد بارگیری_tiktoken_bpe

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the LLaMA 3.x model

# Used in Grouped Query Attention (GQA), broadcasts the key and value tensors
تعریف repeat_kv(x: torch.Tensor, n_rep: صحيح) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, سر_dim = x.shape
    إذا n_rep == 1:
        أرجع x
    أرجع (
        x[:, :, :, هیچ, :]
        .expand(bs, slen, n_kv_heads, n_rep, سر_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, سر_dim)
    )

# -----------------------------------------------------------------------------
# RoPE related

تعریف reshape_for_پخش(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    أكد 0 <= 1 < ndim
    أكد freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d إذا i == 1 أو i == ndim - 1 وإلا 1 لكل i, d في enumerate(x.shape)]
    أرجع freqs_cis.view(*shape)

تعریف apply_scaling(freqs: torch.Tensor):
    # Values obtained from grid search
    scale_ضریب = 8
    کم_freq_ضریب = 1
    بالا_freq_ضریب = 4
    old_context_len = 8192  # original llama3 length

    کم_freq_wavelen = old_context_len / کم_freq_ضریب
    بالا_freq_wavelen = old_context_len / بالا_freq_ضریب
    new_freqs = []
    لكل freq في freqs:
        wavelen = 2 * math.pi / freq
        إذا wavelen < بالا_freq_wavelen:
            new_freqs.append(freq)
        وإلاإذا wavelen > کم_freq_wavelen:
            new_freqs.append(freq / scale_ضریب)
        وإلا:
            أكد کم_freq_wavelen != بالا_freq_wavelen
            smooth = (old_context_len / wavelen - کم_freq_ضریب) / (
                بالا_freq_ضریب - کم_freq_ضریب
            )
            new_freqs.append((1 - smooth) * freq / scale_ضریب + smooth * freq)
    أرجع torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)

تعریف apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.عائم().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.عائم().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_پخش(freqs_cis, xq_)
    xq_بیرون = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_بیرون = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    أرجع xq_بیرون.type_as(xq), xk_بیرون.type_as(xk)

تعریف precompute_freqs_cis(
    dim: صحيح, انتها: صحيح, theta: عائم = 10000.0, use_scaled: bool = خطأ
):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].عائم() / dim))
    t = torch.arange(انتها, device=freqs.device, dtype=torch.float32)
    إذا use_scaled:
        freqs = apply_scaling(freqs)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    أرجع freqs_cis

# -----------------------------------------------------------------------------
# LLaMA building blocks

# LLaMA reference code explicitly implemented RMSNorm so we copy pasted it
# (https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/api/model.py)
# we could also use nn.RMSNorm, it has slightly different numeric properties, but equivalent
صنف RMSNorm(torch.nn.Module):
    تعریف __init__(self, dim: صحيح, eps: عائم = 1e-6):
        أعلى().__init__()
        self.eps = eps
        self.وزن = nn.Parameter(torch.ones(dim))

    تعریف _norm(self, x):
        أرجع x * torch.rsqrt(x.pow(2).میانگین(-1, keepdim=درست) + self.eps)

    تعریف forward(self, x):
        output = self._norm(x.عائم()).type_as(x)
        أرجع output * self.وزن

صنف CausalSelfAttention(nn.Module):

    تعریف __init__(self, config):
        أعلى().__init__()
        أكد config.n_embd % config.n_سر == 0

        self.n_سر = config.n_سر
        self.n_kv_سر = config.n_kv_سر
        self.n_rep = self.n_سر // self.n_kv_سر
        self.hd = config.n_embd // config.n_سر
        self.use_kv = config.use_kv
        self.flash = config.flash

        self.c_attn = nn.خطی(config.n_embd, (config.n_سر + 2 * config.n_kv_سر) * self.hd, bias=خطأ)  # key, query, value projections
        self.c_proj = nn.خطی(config.n_embd, config.n_embd, bias=خطأ)  # output projection

        # static KV cache - we could alternatively allocate it outside of the model and just pass it in when needed
        إذا self.use_kv:
            self.cache_k = torch.zeros((config.max_gen_batch_اندازه, config.بلوک_اندازه, config.n_kv_سر, self.hd))
            self.cache_v = torch.zeros((config.max_gen_batch_اندازه, config.بلوک_اندازه, config.n_kv_سر, self.hd))

    تعریف forward(self, x, freqs_cis=هیچ, شروع_pos=هیچ, ماسک=هیچ):
        B, T, C = x.اندازه() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.تقسیم([self.n_سر * self.hd, self.n_kv_سر * self.hd, self.n_kv_سر * self.hd], dim=-1)
        q, k, v = map(مجهول t: t.view(B, T, -1, self.hd), (q, k, v))  # (B, T, NH, HD)

        q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)  # rotate QK (rope)  <-- 1. difference compared to GPT-2

        إذا self.use_kv و ليس self.training و شروع_pos >= 0:  # use kv-caching during inference
            self.cache_k[:B, شروع_pos : شروع_pos + T] = k
            self.cache_v[:B, شروع_pos : شروع_pos + T] = v
            k = self.cache_k[:B, : شروع_pos + T]
            v = self.cache_v[:B, : شروع_pos + T]

        k = repeat_kv(k, self.n_rep)  # GQA <-- 2. difference compared to GPT-2
        v = repeat_kv(v, self.n_rep)

        q, k, v = map(مجهول t: t.transpose(1, 2), (q, k, v))  # (B, NH, T, HD)

        إذا self.flash:
            # flashattention
            # if T == 1 no need to mask, otherwise the function complains
            # scaled_dot_product_attention expects a mask where value of True indicates that the element should take part in attention
            # our mask is the opposite, so we need to invert it
            y = F.scaled_dot_حاصلضرب_attention(q, k, v, ماسک == 0 إذا T > 1 وإلا هیچ)
        وإلا:
            # manual implementation of attention
            # this materializes the large (T,T) matrix for all the queries and keys
            scores = (q @ k.transpose(-2, -1)) * (1.0 / math.جذر(self.hd))
            إذا ماسک است ليس هیچ:
                scores.masked_fill_(ماسک, torch.finfo(scores.dtype).min)
            att = F.softmax(scores.عائم(), dim=-1).type_as(q)
            y = att @ v # (B, NH, T, T) x (B, NH, T, HD) -> (B, NH, T, HD)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        أرجع y

صنف MLP(nn.Module):

    تعریف __init__(self, config):
        أعلى().__init__()
        hidden_dim = 4 * config.n_embd
        hidden_dim = صحيح(2 * hidden_dim / 3)
        # custom dim factor multiplier
        إذا config.ffn_dim_multiplier است ليس هیچ:
            hidden_dim = صحيح(config.ffn_dim_multiplier * hidden_dim)
        hidden_dim = config.مضرب_از * ((hidden_dim + config.مضرب_از - 1) // config.مضرب_از)
        self.c_fc = nn.خطی(config.n_embd, hidden_dim, bias=خطأ)
        self.c_fc2 = nn.خطی(config.n_embd, hidden_dim, bias=خطأ)
        self.c_proj = nn.خطی(hidden_dim, config.n_embd, bias=خطأ)

    تعریف forward(self, x):
        # SwiGLU self.c_proj(F.silu(self.c_fc2(x)) * self.c_fc(x))  <-- 3. difference compared to GPT-2
        x1 = self.c_fc(x)
        x2 = self.c_fc2(x)
        x2 = F.silu(x2)
        x = x1 * x2
        x = self.c_proj(x)
        أرجع x

صنف بلوک(nn.Module):

    تعریف __init__(self, config):
        أعلى().__init__()
        self.ln_1 = RMSNorm(config.n_embd, config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd, config.norm_eps)
        self.mlp = MLP(config)

    تعریف forward(self, x, freqs_cis=هیچ, شروع_pos=هیچ, ماسک=هیچ):
        x = x + self.attn(self.ln_1(x), freqs_cis, شروع_pos, ماسک)
        x = x + self.mlp(self.ln_2(x))
        أرجع x

# -----------------------------------------------------------------------------
# The main LLaMA 3.1 model

@dataclass
صنف LlamaConfig:
    نسخه: str = "3.1"
    بلوک_اندازه: صحيح = 8192
    vocab_اندازه: صحيح = 128256
    n_layer: صحيح = 32
    n_سر: صحيح = 32
    n_kv_سر: صحيح = 8
    n_embd: صحيح = 4096
    ffn_dim_multiplier: عائم = 1.3
    مضرب_از: صحيح = 1024
    norm_eps: عائم = 1e-5
    rope_theta: عائم = 500000.0
    use_scaled_rope: bool = درست
    max_gen_batch_اندازه: صحيح = 4
    use_kv: bool = درست
    flash: bool = خطأ  # use flashattention?

    تعریف __init__(self, **kwargs):
        لكل k, v في kwargs.اقلام():
            إذا hasattr(self, k):
                setattr(self, k, v)
        أكد self.n_kv_سر <= self.n_سر
        أكد self.n_سر % self.n_kv_سر == 0
        أكد self.n_embd % self.n_سر == 0

صنف LLaMA(nn.Module):

    تعریف __init__(self, config):
        أعلى().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_اندازه, config.n_embd),
            h = nn.ModuleList([بلوک(config) لكل _ في دامنه(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd, config.norm_eps),
        ))
        self.lm_سر = nn.خطی(config.n_embd, config.vocab_اندازه, bias=خطأ)

        # init all weights, use a torch rng object to be very careful
        self.init_rng = torch.Generator()
        self.init_rng.manual_دانه(42)

        self.freqs_cis = precompute_freqs_cis(
            config.n_embd // config.n_سر,
            config.بلوک_اندازه * 2,
            config.rope_theta,
            config.use_scaled_rope,
        )

    تعریف forward(self, idx, targets=هیچ, return_logits=درست, شروع_pos=0):
        _, t = idx.اندازه()
        أكد t <= self.config.بلوک_اندازه, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # forward the LLaMA model itself
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        freqs_cis = self.freqs_cis[شروع_pos:شروع_pos+t]

        ماسک = torch.triu(torch.ones((t, t), device=بعدی_2(self.parameters()).device, dtype=torch.bool), diagonal=1)

        لكل i, بلوک_2 في enumerate(self.transformer.h):
            x = بلوک_2(x, freqs_cis, شروع_pos, ماسک)
        x = self.transformer.ln_f(x)

        إذا targets است ليس هیچ:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_سر(x).عائم()
            loss = F.cross_entropy(logits.view(-1, logits.اندازه(-1)), targets.view(-1), ignore_شاخص=-1)
        وإلا:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_سر(x[:, [-1], :]).عائم() # note: using list [-1] to preserve the time dim
            loss = هیچ

        # there are performance reasons why not returning logits is prudent, if not needed
        إذا ليس return_logits:
            logits = هیچ

        أرجع logits, loss

    @staticmethod
    تعریف adapt_llama_state_dict_کلیدها(checkpoint, config: LlamaConfig):
        # Modify key names from Meta's LLaMA to our LLaMA
        # our key names are derived from GPT-2's key names
        checkpoint['transformer.wte.weight'] = checkpoint.برداشت('tok_embeddings.weight')

        لكل i في دامنه(config.n_layer):
            لكل name في ['attention_norm', 'ffn_norm']:
                old_کلید = f'layers.{i}.{name}.weight'  # e.g. layers.x.attention_norm.weight -> transformer.h.x.ln_1.weight
                new_کلید = f'transformer.h.{i}.ln_{1 if name == "attention_norm" else 2}.weight'
                checkpoint[new_کلید] = checkpoint.برداشت(old_کلید)

        لكل i في دامنه(config.n_layer):
            لكل name في ['attention.wq', 'attention.wk', 'attention.wv']:
                old_کلید = f'layers.{i}.{name}.weight'
                new_کلید = f'transformer.h.{i}.attn.c_attn.weight'
                إذا name == 'attention.wq':
                    checkpoint[new_کلید] = checkpoint.برداشت(old_کلید)
                وإلا:  # merge 3 weights into transformer.h.x.attn.c_attn.weight
                    checkpoint[new_کلید] = torch.cat((checkpoint[new_کلید], checkpoint.برداشت(old_کلید)), dim=0)
            old_کلید = f'layers.{i}.attention.wo.weight'
            new_کلید = f'transformer.h.{i}.attn.c_proj.weight'
            checkpoint[new_کلید] = checkpoint.برداشت(old_کلید)

        ffn_map = {'w1': 'c_fc2', 'w2': 'c_proj', 'w3': 'c_fc'}
        لكل i في دامنه(config.n_layer):
            لكل name في ['feed_forward.w1', 'feed_forward.w2', 'feed_forward.w3']:
                old_کلید = f'layers.{i}.{name}.weight'
                new_کلید = f'transformer.h.{i}.mlp.{ffn_map[name.split(".")[-1]]}.weight'
                checkpoint[new_کلید] = checkpoint.برداشت(old_کلید)

        checkpoint['transformer.ln_f.weight'] = checkpoint.برداشت('norm.weight')
        checkpoint['lm_head.weight'] = checkpoint.برداشت('output.weight')

        أرجع checkpoint

    @staticmethod
    تعریف adapt_llama_state_dict_کلیدها_hf(checkpoint, config: LlamaConfig):
        # Modify key names from HuggingFace's LLaMA to our LLaMA
        # our key names are derived from GPT-2's key names
        checkpoint['transformer.wte.weight'] = checkpoint.برداشت('model.embed_tokens.weight')

        # We need to unpermute K and V because HF script permuted the original Meta-LLaMA weights
        # see: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py
        تعریف unpermute(w, n_heads, dim1, dim2):
            أرجع w.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)

        لكل i في دامنه(config.n_layer):
            لكل name في ['input_layernorm', 'post_attention_layernorm']:
                old_کلید = f'model.layers.{i}.{name}.weight'  # e.g. layers.x.attention_norm.weight -> transformer.h.x.ln_1.weight
                new_کلید = f'transformer.h.{i}.ln_{1 if name == "input_layernorm" else 2}.weight'
                checkpoint[new_کلید] = checkpoint.برداشت(old_کلید)

        لكل i في دامنه(config.n_layer):
            لكل name في ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj']:
                old_کلید = f'model.layers.{i}.{name}.weight'
                new_کلید = f'transformer.h.{i}.attn.c_attn.weight'
                إذا name == 'self_attn.q_proj':
                    checkpoint[new_کلید] = unpermute(checkpoint.برداشت(old_کلید), config.n_سر, config.n_embd, config.n_embd)
                وإلا:  # merge 3 weights into transformer.h.x.attn.c_attn.weight
                    tensor = checkpoint.برداشت(old_کلید)
                    إذا name == 'self_attn.k_proj':
                        tensor = unpermute(tensor, config.n_kv_سر, config.n_kv_سر * (config.n_embd // config.n_سر), config.n_embd)
                    checkpoint[new_کلید] = torch.cat((checkpoint[new_کلید], tensor), dim=0)
            old_کلید = f'model.layers.{i}.self_attn.o_proj.weight'
            new_کلید = f'transformer.h.{i}.attn.c_proj.weight'
            checkpoint[new_کلید] = checkpoint.برداشت(old_کلید)

        ffn_map = {'gate_proj': 'c_fc2', 'down_proj': 'c_proj', 'up_proj': 'c_fc'}
        لكل i في دامنه(config.n_layer):
            لكل name في ['gate_proj', 'down_proj', 'up_proj']:
                old_کلید = f'model.layers.{i}.mlp.{name}.weight'
                new_کلید = f'transformer.h.{i}.mlp.{ffn_map[name]}.weight'
                checkpoint[new_کلید] = checkpoint.برداشت(old_کلید)

        checkpoint['transformer.ln_f.weight'] = checkpoint.برداشت('model.norm.weight')

        أرجع checkpoint

    @classmethod
    تعریف from_pretrained_llama3_hf(cls, model_id):
        """Loads pretrained LLaMA model weights from HuggingFace"""
        من transformers استورد AutoModelForCausalLM, AutoTokenizer
        أكد model_id == "meta-llama/Meta-Llama-3.1-8B", "Only the 8B-base model is supported for now"
        model_آرگومانها = LlamaConfig()

        model = AutoModelForCausalLM.from_pretrained(model_id)
        checkpoint = LLaMA.adapt_llama_state_dict_کلیدها_hf(model.state_dict(), model_آرگومانها)

        original_default_type = torch.get_default_dtype()  # save the default type
        torch.مجموعه_default_tensor_type(torch.cuda.BFloat16Tensor)  # much faster loading
        model = LLaMA(model_آرگومانها)
        model.بارگیری_state_dict(checkpoint, strict=خطأ)
        torch.مجموعه_default_tensor_type(torch.tensor([], dtype=original_default_type, device="cpu").type())  # restore default type

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_id = 128004  # this is the pad token id for LLaMA 3.1 base, we need to set this explicitly as our generate func expects it
        tokenizer.stop_نشانهها = [tokenizer.eos_token_id]
        model.tokenizer = tokenizer
        أرجع model

    @classmethod
    تعریف from_pretrained_llama3_meta(cls, ckpt_dir, tokenizer_مسیر):
        """Loads pretrained LLaMA model weights from a checkpoint directory"""
        model_آرگومانها = LlamaConfig()

        ckpt_مسیر = sorted(مسیر_2(ckpt_dir).glob("*.pth"))[0]
        checkpoint = torch.بارگیری(ckpt_مسیر, map_location="cpu", weights_only=درست)
        checkpoint = LLaMA.adapt_llama_state_dict_کلیدها(checkpoint, model_آرگومانها)

        original_default_type = torch.get_default_dtype()  # save the default type
        torch.مجموعه_default_tensor_type(torch.cuda.BFloat16Tensor)  # much faster loading
        model = LLaMA(model_آرگومانها)
        model.بارگیری_state_dict(checkpoint, strict=خطأ)
        torch.مجموعه_default_tensor_type(torch.tensor([], dtype=original_default_type, device="cpu").type())  # restore default type

        tokenizer = Tokenizer(model_مسیر=tokenizer_مسیر)
        model.tokenizer = tokenizer
        أرجع model

    تعریف configure_optimizers(self, وزن_decay, learning_نرخ, betas, device_type, zero_stage):
        # start with all of the candidate parameters
        param_dict = {pn: p لكل pn, p في self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p لكل pn, p في param_dict.اقلام() إذا p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p لكل n, p في param_dict.اقلام() إذا p.dim() >= 2]
        nodecay_params = [p لكل n, p في param_dict.اقلام() إذا p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': وزن_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        رقم_decay_params = جمع(p.numel() لكل p في decay_params)
        رقم_nodecay_params = جمع(p.numel() لكل p في nodecay_params)
        print0(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print0(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_موجود = 'fused' في inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_موجود و device_type == 'cuda'
        print0(f"using fused AdamW: {use_fused}")
        إذا zero_stage == 1:
            print0("using ZeroRedundancyOptimizer")
            optimizer = ZeroRedundancyOptimizer(**optim_groups[0], optimizer_class=torch.optim.AdamW,
                                                lr=learning_نرخ, betas=betas, fused=use_fused)
            optimizer.جمع_کن_param_گروه(optim_groups[1])
        وإلا:
            print0("using regular AdamW")
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_نرخ, betas=betas, fused=use_fused)
        أرجع optimizer

    @torch.inference_نما()
    تعریف generate(
        self,
        prompt_نشانهها: List[List[صحيح]],
        max_gen_len: صحيح,
        temperature: عائم = 0.6,
        بالاترین_p: عائم = 0.9,
        echo: bool = خطأ,
    ) -> Tuple[List[List[صحيح]], Optional[List[List[عائم]]]]:
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
        bsz = طول(prompt_نشانهها)
        أكد bsz <= self.config.max_gen_batch_اندازه, f"Batch size {bsz} exceeds the maximum generation batch size {self.config.max_gen_batch_size}"
        device = بعدی_2(self.parameters()).device

        min_prompt_len = min(طول(t) لكل t في prompt_نشانهها)
        max_prompt_len = max(طول(t) لكل t في prompt_نشانهها)
        أكد max_prompt_len <= self.config.بلوک_اندازه, f"Prompt length {max_prompt_len} exceeds the maximum block size {self.config.block_size}"
        مجموع_len = min(self.config.بلوک_اندازه, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_id
        نشانهها = torch.پر((bsz, مجموع_len), pad_id, dtype=torch.دراز, device=device)
        لكل idx, t في enumerate(prompt_نشانهها):
            نشانهها[idx, : طول(t)] = torch.tensor(t, dtype=torch.دراز, device=device)

        prev_pos = 0
        eos_reached = torch.tensor([خطأ] * bsz, device=device)
        درون_داد_متن_ماسک = نشانهها != pad_id

        إذا min_prompt_len == مجموع_len:
            logits, _ = self.forward(نشانهها, شروع_pos=prev_pos)

        stop_نشانهها = torch.tensor(list(self.tokenizer.stop_نشانهها)).مقصد(device)

        لكل cur_pos في دامنه(min_prompt_len, مجموع_len):
            logits, _ = self.forward(نشانهها[:, prev_pos:cur_pos], شروع_pos=prev_pos)
            إذا temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                بعدی_token = sample_بالاترین_p(probs, بالاترین_p)
            وإلا:
                بعدی_token = torch.argmax(logits[:, -1], dim=-1)

            بعدی_token = بعدی_token.reshape(-1)
            # only replace token if prompt has already been generated
            بعدی_token = torch.where(درون_داد_متن_ماسک[:, cur_pos], نشانهها[:, cur_pos], بعدی_token)
            نشانهها[:, cur_pos] = بعدی_token
            eos_reached |= ~درون_داد_متن_ماسک[:, cur_pos] & torch.isin(بعدی_token, stop_نشانهها)
            prev_pos = cur_pos
            إذا all(eos_reached):
                اكسر

        بیرون_نشانهها = []
        لكل i, toks في enumerate(نشانهها.tolist()):
            # cut to max gen len
            شروع = 0 إذا echo وإلا طول(prompt_نشانهها[i])
            toks = toks[شروع : طول(prompt_نشانهها[i]) + max_gen_len]
            # cut to after eos tok if any
            لكل stop_token في self.tokenizer.stop_نشانهها:
                حاول:
                    eos_idx = toks.شاخص(stop_token)
                    toks = toks[:eos_idx]
                استثناء مقدار_Error:
                    بگذر
            بیرون_نشانهها.append(toks)
        أرجع بیرون_نشانهها

# -----------------------------------------------------------------------------
# sampling utils

تعریف sample_بالاترین_p(probs, p):
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
    probs_مرتب_سازی, probs_idx = torch.مرتب_سازی(probs, dim=-1, descending=درست)
    probs_جمع = torch.cumsum(probs_مرتب_سازی, dim=-1)
    ماسک = probs_جمع - probs_مرتب_سازی > p
    probs_مرتب_سازی[ماسک] = 0.0
    probs_مرتب_سازی.div_(probs_مرتب_سازی.جمع(dim=-1, keepdim=درست))
    بعدی_token = torch.multinomial(probs_مرتب_سازی, رقم_samples=1)
    بعدی_token = torch.gather(probs_idx, -1, بعدی_token)
    أرجع بعدی_token

# -----------------------------------------------------------------------------
# Llama 3.1 Tokenizer

# The tiktoken tokenizer can handle <=400k chars without
# pyo3_runtime.PanicException.
TIKTOKEN_MAX_ENCODE_CHARS = 400_000

# https://github.com/openai/tiktoken/issues/195
# Here we iterate over subsequences and split if we exceed the limit
# of max consecutive non-whitespace or whitespace characters.
MAX_NO_WHITESPACES_CHARS = 25_000


صنف Tokenizer:
    """
    Tokenizing and encoding/decoding text using the Tiktoken tokenizer.
    """

    special_نشانهها_2: Dict[str, صحيح]

    رقم_reserved_special_نشانهها = 256

    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: E501

    تعریف __init__(self, model_مسیر: str):
        """
        Initializes the Tokenizer with a Tiktoken model.

        Args:
            model_path (str): The path to the Tiktoken model file.
        """
        أكد os.مسیر.isfile(model_مسیر), model_مسیر

        mergeable_ranks = بارگیری_tiktoken_bpe(model_مسیر)
        رقم_پایه_نشانهها = طول(mergeable_ranks)
        special_نشانهها_2 = [
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
        reserved_نشانهها = [
            f"<|reserved_special_token_{2 + i}|>"
            لكل i في دامنه(self.رقم_reserved_special_نشانهها - طول(special_نشانهها_2))
        ]
        special_نشانهها_2 = special_نشانهها_2 + reserved_نشانهها

        self.special_نشانهها_2 = {
            token: رقم_پایه_نشانهها + i لكل i, token في enumerate(special_نشانهها_2)
        }
        self.model = tiktoken.Encoding(
            name=مسیر_2(model_مسیر).name,
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_نشانهها_2=self.special_نشانهها_2,
        )

        self.n_واژگان: صحيح = رقم_پایه_نشانهها + طول(special_نشانهها_2)
        # BOS / EOS token IDs
        self.bos_id: صحيح = self.special_نشانهها_2["<|begin_of_text|>"]
        self.eos_id: صحيح = self.special_نشانهها_2["<|end_of_text|>"]
        self.eot_id: صحيح = self.special_نشانهها_2["<|eot_id|>"]
        self.eom_id: صحيح = self.special_نشانهها_2["<|eom_id|>"]
        self.python_برچسب_id = self.special_نشانهها_2["<|python_tag|>"]
        self.pad_id: صحيح = self.special_نشانهها_2["<|finetune_right_pad_id|>"]
        # hardcoded stop tokens for the base model
        self.stop_نشانهها = [
            self.special_نشانهها_2["<|begin_of_text|>"],
            self.special_نشانهها_2["<|end_of_text|>"],
        ]

    تعریف encode(
        self,
        s: str,
        *,
        bos: bool,
        eos: bool,
        allowed_special: Optional[Union[Literal["all"], Abstract_مجموعه[str]]] = هیچ,
        disallowed_special: Union[Literal["all"], Collection[str]] = (),
    ) -> List[صحيح]:
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
        إذا allowed_special است هیچ:
            allowed_special = مجموعه()
        أكد type(s) است str

        substrs = (
            substr
            لكل i في دامنه(0, طول(s), TIKTOKEN_MAX_ENCODE_CHARS)
            لكل substr في self.تقسیم_whitespaces_or_nonwhitespaces(
                s[i : i + TIKTOKEN_MAX_ENCODE_CHARS], MAX_NO_WHITESPACES_CHARS
            )
        )
        t: List[صحيح] = []
        لكل substr في substrs:
            t.extend(
                self.model.encode(
                    substr,
                    allowed_special=allowed_special,
                    disallowed_special=disallowed_special,
                )
            )
        إذا bos:
            t.درج(0, self.bos_id)
        إذا eos:
            t.append(self.eos_id)
        أرجع t

    تعریف decode(self, t: Sequence[صحيح]) -> str:
        # Typecast is safe here. Tiktoken doesn't do anything list-related with the sequence.
        أرجع self.model.decode(cast(List[صحيح], t))

    @staticmethod
    تعریف تقسیم_whitespaces_or_nonwhitespaces(
        s: str, max_consecutive_برش_len: صحيح
    ) -> Iterator[str]:
        """
        Splits the string `s` so that each substring contains no more than `max_consecutive_slice_len`
        consecutive whitespaces or consecutive non-whitespaces.
        """
        current_برش_len = 0
        current_برش_is_space = s[0].isspace() إذا طول(s) > 0 وإلا خطأ
        برش_شروع = 0

        لكل i في دامنه(طول(s)):
            is_now_space = s[i].isspace()

            إذا current_برش_is_space ^ is_now_space:
                current_برش_len = 1
                current_برش_is_space = is_now_space
            وإلا:
                current_برش_len += 1
                إذا current_برش_len > max_consecutive_برش_len:
                    أنتج s[برش_شروع:i]
                    برش_شروع = i
                    current_برش_len = 1
        أنتج s[برش_شروع:]

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

تعریف peek_داده_shard(filename):
    # only reads the header, returns header data
    با open(filename, "rb") كـ f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        سرآیند = np.frombuffer(f.اقرأ(256*4), dtype=np.int32)
    إذا سرآیند[0] != 20240801:
        چاپ("ERROR: magic number mismatch in the data .bin file!")
        exit(1)
    أكد سرآیند[1] == 7, "unsupported version"
    ntok = سرآیند[2] # number of tokens (claimed)
    أرجع ntok # for now just return the number of tokens

تعریف بارگیری_داده_shard(filename):
    با open(filename, "rb") كـ f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        سرآیند = np.frombuffer(f.اقرأ(256*4), dtype=np.int32)
        أكد سرآیند[0] == 20240801, "magic number mismatch in the data .bin file"
        أكد سرآیند[1] == 7, "unsupported version"
        ntok = سرآیند[2] # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        نشانهها = np.frombuffer(f.اقرأ(), dtype=np.uint32)
    أكد طول(نشانهها) == ntok, "number of tokens read does not match header?"
    أرجع نشانهها

صنف Distributed_Sharded_داده_Loader:
    """
    This DataLoader is both:
    - distributed (works correctly in case of multiple processes in DDP)
    - sharded (supports datasets that are broken up into multiple data shards)
    It is not *permuted*, meaning that it itearates over the data in the order
    of the dataset on disk, so the user should make sure to shuffle their examples
    during the creation of their data shards for best performance.
    """
    تعریف __init__(self, filename_الگو, B, T, process_rank, رقم_processes):
        self.process_rank = process_rank
        self.رقم_processes = رقم_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_الگو))
        أكد طول(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_مجموع = 0
        لكل fname في self.files:
            shard_ntok = peek_داده_shard(fname)
            أكد shard_ntok >= رقم_processes * B * T + 1
            ntok_مجموع += shard_ntok
        self.ntok_مجموع = ntok_مجموع
        print0(f"DataLoader: total number of tokens: {ntok_total:,} across {len(self.files)} files")

        # kick things off
        self.current_shard = هیچ
        self.reset()

    تعریف reset(self):
        # we're being a bit clever here: if we already had shard 0 loaded,
        # then don't do the work to reload it, just reset the pointer
        إذا self.current_shard != 0:
            self.current_shard = 0
            self.نشانهها = بارگیری_داده_shard(self.files[self.current_shard])
        self.current_position = self.process_rank * self.B * self.T

    تعریف advance(self): # advance to next data shard
        self.current_shard = (self.current_shard + 1) % طول(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.نشانهها = بارگیری_داده_shard(self.files[self.current_shard])

    تعریف بعدی_batch(self):
        B = self.B
        T = self.T
        buf = self.نشانهها[self.current_position : self.current_position+B*T+1]
        buf = torch.tensor(buf, dtype=torch.دراز)
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the start pointer in current shard
        self.current_position += B * T * self.رقم_processes
        # if loading the next batch would be out of bounds advance the shard
        إذا self.current_position + (B * T * self.رقم_processes + 1) > طول(self.نشانهها):
            self.advance()
        أرجع x, y

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
    أكد dtype في {"float32", "bfloat16"}
    write_fun = write_fp32 إذا dtype == "float32" وإلا write_bf16
    write_fun(model_tensors["transformer.wte.weight"], file) # (V, C)
    لكل i في دامنه(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_1.weight"], file)
    لكل i في دامنه(L): # (L, 3C, C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_attn.weight"], file)
    لكل i في دامنه(L): # (L, C, C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_proj.weight"], file)
    لكل i في دامنه(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_2.weight"], file)
    لكل i في دامنه(L): # (L, 4C, C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_fc.weight"], file)
    لكل i في دامنه(L): # (L, 4C, C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_fc2.weight"], file)
    لكل i في دامنه(L): # (L, C, 4C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_proj.weight"], file)
    write_fun(model_tensors["transformer.ln_f.weight"], file) # (C, )
    write_fun(model_tensors["lm_head.weight"], file) # (V, C)

تعریف write_model(model, filename, dtype):
    # everything we need to instantiate the model
    # 1) header is: version int, LLaMAConfig ints, padding to 1024 bytes
    أكد dtype في {"float32", "bfloat16"}
    نسخه = {
        "float32": 3, # 3: all tensors are fp32
        "bfloat16": 5, # 5: all tensors are bf16
    }[dtype]
    سرآیند = torch.zeros(256, dtype=torch.int32)
    سرآیند[0] = 20240803 # magic
    سرآیند[1] = نسخه # checkpoint version
    سرآیند[2] = model.config.بلوک_اندازه
    سرآیند[3] = model.config.vocab_اندازه
    سرآیند[4] = model.config.n_layer
    سرآیند[5] = model.config.n_سر
    سرآیند[6] = model.config.n_kv_سر
    سرآیند[7] = model.config.n_embd
    سرآیند[8] = model.config.ffn_dim_multiplier
    سرآیند[9] = model.config.مضرب_از
    سرآیند[10] = model.config.norm_eps
    سرآیند[11] = model.config.rope_theta
    سرآیند[12] = model.config.use_scaled_rope
    سرآیند[13] = model.config.max_gen_batch_اندازه
    سرآیند[14] = صحيح(model.config.نسخه.تقسیم('.')[0]) # major version
    سرآیند[15] = صحيح(model.config.نسخه.تقسیم('.')[1]) # minor version
    # 2) the parameters follow the header
    params = {name: param.cpu() لكل name, param في model.named_parameters()}
    # now write to file
    با open(filename, "wb") كـ file:
        file.اكتب(سرآیند.numpy().tobytes()) #سرآیند
        write_tensors(params, model.config.n_layer, file, dtype) # params
    چاپ(f"wrote {filename}")

تعریف write_state(model, x, y, logits, loss, filename):
    # the state is used for debugging.
    # it contains information about the input, logits, loss, and the parameter gradients
    # this can be used for checking the computation correctness in C
    سرآیند = torch.zeros(256, dtype=torch.int32)
    سرآیند[0] = 20240803 # magic
    سرآیند[1] = x.اندازه(0) # batch size of the batch, B
    سرآیند[2] = x.اندازه(1) # temporal extent of the batch, T
    grads = {name: param.grad.cpu() لكل name, param في model.named_parameters()}
    با open(filename, "wb") كـ file:
        #سرآیند
        file.اكتب(سرآیند.numpy().tobytes())
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
    چاپ(f"wrote {filename}")

# -----------------------------------------------------------------------------
# int main

تعریف print0(*آرگومانها, **kwargs):
    # modified print that only prints from the master process
    # if this is not a distributed run, it's just a print
    إذا صحيح(os.environ.get("RANK", 0)) == 0:
        چاپ(*آرگومانها, **kwargs)

إذا __name__ == "__main__":
    print0(f"Running pytorch {torch.version.__version__}")

    # default settings will overfit a tiny batch of data
    # and save model weights and debug state to disk on the first iteration
    parser = argparse.آرگومان_Parser()
    parser.جمع_کن_آرگومان("--use_hf", type=صحيح, افتراضي=1, help="use HuggingFace (default) or use Meta's model")
    parser.جمع_کن_آرگومان("--ckpt_dir", type=str, افتراضي=هیچ, help="path to llama3 model checkpoint (needed if use_hf=0)")
    parser.جمع_کن_آرگومان("--tokenizer_path", type=str, افتراضي=هیچ, help="path to llama3 tokenizer (needed if use_hf=0)")
    # file system input / output
    parser.جمع_کن_آرگومان("--input_bin", type=str, افتراضي="dev/data/tinyshakespeare/tiny_shakespeare_val.bin", help="input .bin to train on")
    parser.جمع_کن_آرگومان("--input_val_bin", type=str, افتراضي="", help="input .bin to eval validation loss on")
    parser.جمع_کن_آرگومان("--output_dir", type=str, افتراضي="", help="output directory to which to write logs and checkpoints")
    parser.جمع_کن_آرگومان("--model", type=str, افتراضي="meta-llama/Meta-Llama-3.1-8B", help="chose the llama model")
    # token layout for each step of the optimization
    parser.جمع_کن_آرگومان("--batch_size", type=صحيح, افتراضي=4, help="batch size, in units of #batch dimensions")
    parser.جمع_کن_آرگومان("--sequence_length", type=صحيح, افتراضي=64, help="sequence length")
    parser.جمع_کن_آرگومان("--total_batch_size", type=صحيح, افتراضي=256, help="total desired batch size, in units of #tokens")
    # workload (number of steps)
    parser.جمع_کن_آرگومان("--num_iterations", type=صحيح, افتراضي=10, help="number of iterations to run")
    parser.جمع_کن_آرگومان("--inference_only", type=صحيح, افتراضي=0, help="only run inference")
    # optimization
    parser.جمع_کن_آرگومان("--learning_rate", type=عائم, افتراضي=1e-5, help="learning rate warmup iterations")
    parser.جمع_کن_آرگومان("--warmup_iters", type=صحيح, افتراضي=0, help="learning rate warmup iterations")
    parser.جمع_کن_آرگومان("--learning_rate_decay_frac", type=عائم, افتراضي=1.0, help="learning rate warmup iterations")
    parser.جمع_کن_آرگومان("--weight_decay", type=عائم, افتراضي=0.0, help="weight decay")
    parser.جمع_کن_آرگومان("--grad_clip", type=عائم, افتراضي=1.0, help="maximum gradient magnitude")
    # evaluation
    parser.جمع_کن_آرگومان("--val_loss_every", type=صحيح, افتراضي=0, help="every how mant steps to evaluate val loss?")
    parser.جمع_کن_آرگومان("--val_max_steps", type=صحيح, افتراضي=20, help="how many batches of val to average?")
    parser.جمع_کن_آرگومان("--sample_every", type=صحيح, افتراضي=0, help="how often to sample from the model?")
    # debugging
    parser.جمع_کن_آرگومان("--overfit_single_batch", type=صحيح, افتراضي=1, help="overfit just one batch of data")
    # numerics
    parser.جمع_کن_آرگومان("--tensorcores", type=صحيح, افتراضي=0, help="use tensorcores")
    # memory management
    parser.جمع_کن_آرگومان("--device", type=str, افتراضي="", help="by default we autodetect, or set it here")
    parser.جمع_کن_آرگومان("--compile", type=صحيح, افتراضي=0, help="torch.compile the model")
    parser.جمع_کن_آرگومان("--dtype", type=str, افتراضي="bfloat16", help="float32|float16|bfloat16")
    parser.جمع_کن_آرگومان("--zero_stage", type=صحيح, افتراضي=0, help="zero redundancy optimizer stage (0/1/2/3)")
    # python -> C bridge
    parser.جمع_کن_آرگومان("--write_tensors", type=صحيح, افتراضي=0, help="write tensors to disk")
    آرگومانها = parser.parse_آرگومانها()

    # args error checking and convenience variables
    B, T = آرگومانها.batch_اندازه, آرگومانها.sequence_درازی
    أكد 1 <= T <= 8192, "sequence length must be between 1 and 8192"
    أكد آرگومانها.dtype في {"float32", "float16", "bfloat16"}
    أكد آرگومانها.model في {"meta-llama/Meta-Llama-3.1-8B"}  # only 8B base model supported for now

    # create the logging directory if it does not exist
    logfile = هیچ
    إذا آرگومانها.output_dir:
        os.makedirs(آرگومانها.output_dir, exist_ok=درست)
        logfile = os.مسیر.join(آرگومانها.output_dir, "main.log")
        # create the log file "main.log" inside it, and wipe it clean
        با open(logfile, "w") كـ f:
            بگذر

    # set up DDP (distributed data parallel). torchrun sets this env variable
    ddp = صحيح(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    إذا ddp:
        # use of DDP atm demands CUDA, we set the device appropriately according to rank
        أكد torch.cuda.is_موجود(), "for now i think we need CUDA for DDP"
        init_process_گروه(backend='nccl')
        ddp_rank = صحيح(os.environ['RANK'])
        ddp_local_rank = صحيح(os.environ['LOCAL_RANK'])
        ddp_world_اندازه = صحيح(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.مجموعه_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        دانه_جابه_جایی = 0 # each process gets the exact same seed
        zero_stage = آرگومانها.zero_stage
    وإلا:
        ddp_rank = 0
        ddp_local_rank = 0
        zero_stage = 0
        ddp_world_اندازه = 1
        master_process = درست
        دانه_جابه_جایی = 0
        # select the device
        إذا آرگومانها.device:
            # provided explicitly by the user
            device = آرگومانها.device
        وإلا:
            # attempt to autodetect the device
            device = "cpu"
            إذا torch.cuda.is_موجود():
                device = "cuda"
            وإلاإذا hasattr(torch.backends, "mps") و torch.backends.mps.is_موجود():
                device = "mps"
    device_type = 'cuda' إذا 'cuda' في device وإلا 'cpu'
    أكد device_type في {'cuda'}, "GPU required to run LLaMA 3"  # we need to load LLaMA as bf16 on CUDA
    چاپ(f"using device: {device}")

    # calculate gradient accumulation from the desired total batch size and the current run configuration
    نشانهها_در_هر_fwdbwd = B * T * ddp_world_اندازه
    أكد آرگومانها.مجموع_batch_اندازه % نشانهها_در_هر_fwdbwd == 0
    grad_accum_گامها = آرگومانها.مجموع_batch_اندازه // نشانهها_در_هر_fwdbwd
    print0(f"total desired batch size: {args.total_batch_size}")
    print0(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    # set up a context manager following the desired dtype and device
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[آرگومانها.dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) إذا (device_type == "cuda") وإلا nullcontext()

    # rng / reproducibility
    torch.manual_دانه(42)
    إذا torch.cuda.is_موجود():
        torch.cuda.manual_دانه(42)

    # set the torch precision mode to use TensorFloat32 (TF32) for matmuls
    # docs https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    إذا آرگومانها.tensorcores:
        torch.مجموعه_float32_matmul_precision('high')

    # init the model
    إذا آرگومانها.use_hf:
        model = LLaMA.from_pretrained_llama3_hf(آرگومانها.model)
    وإلا:  # use Meta's checkpoint
        أكد آرگومانها.ckpt_dir است ليس هیچ و os.مسیر.exists(آرگومانها.ckpt_dir), f"llama3 ckpt dir {args.ckpt_dir} does not exist"
        أكد آرگومانها.tokenizer_مسیر است ليس هیچ و os.مسیر.exists(آرگومانها.tokenizer_مسیر), f"llama3 tokenizer path {args.tokenizer_path} does not exist"
        model = LLaMA.from_pretrained_llama3_meta(آرگومانها.ckpt_dir, آرگومانها.tokenizer_مسیر)

    model.train()
    إذا آرگومانها.compile:
        إذا hasattr(config, "coordinate_descent_tuning"):
            config.coordinate_descent_tuning = درست # suggested by @Chillee
        print0("compiling the model...")
        model = torch.compile(model)

    # -------------------------------------------------------------------------
    # Our own version of a simple DistributedDataLoader

    # load tokens
    train_loader = Distributed_Sharded_داده_Loader(آرگومانها.درون_داد_bin, B, T, ddp_rank, ddp_world_اندازه)
    val_loader = هیچ
    إذا آرگومانها.درون_داد_val_bin:
        val_loader = Distributed_Sharded_داده_Loader(آرگومانها.درون_داد_val_bin, B, T, ddp_rank, ddp_world_اندازه)

    # -------------------------------------------------------------------------
    # PyTorch -> C bridge: save some weights and state for C to load later as reference

    # do one forward pass to generate ground truth for our C tests
    إذا master_process و آرگومانها.write_tensors و (ليس آرگومانها.inference_only):
        x, y = train_loader.بعدی_batch()
        x, y = x.مقصد(device), y.مقصد(device)
        logits, loss = model(x, y)
        loss.backward()
        # save model params, in bfloat16
        model_مقصد_اندازه = {"meta-llama/Meta-Llama-3.1-8B": "8B"}
        model_اندازه_str = model_مقصد_اندازه[آرگومانها.model] # e.g. "8B"
        write_model(model, os.مسیر.join(آرگومانها.output_dir, f"llama3.1_{model_size_str}_bf16.bin"), dtype="bfloat16")
        # save x, y, logits, loss, and parameter gradients, for debugging C
        # always store these in fp32 to have an accurate reference (?)
        write_state(model, x, y, logits, loss, os.مسیر.join(آرگومانها.output_dir, f"llama3_{model_size_str}_debug_state.bin"))
        # reset the train_loader for the optimization below
        train_loader.reset()

    # -------------------------------------------------------------------------
    # main training loop

    # here we wrap model into DDP container
    إذا ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module إذا ddp وإلا model # always contains the "raw" unwrapped model

    # init the optimizer
    optimizer = raw_model.configure_optimizers(وزن_decay=آرگومانها.وزن_decay,
                                               learning_نرخ=آرگومانها.learning_نرخ, betas=(0.9, 0.95),
                                               device_type=device, zero_stage=zero_stage)

    # learning rate decay scheduler (cosine with warmup)
    تعریف get_lr(it):
        min_lr = آرگومانها.learning_نرخ * آرگومانها.learning_نرخ_decay_frac
        # 1) linear warmup for warmup_iters steps
        إذا it < آرگومانها.warmup_iters:
            أرجع آرگومانها.learning_نرخ * (it+1) / آرگومانها.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        إذا it > آرگومانها.رقم_iterations:
            أرجع min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_نسبت = (it - آرگومانها.warmup_iters) / (آرگومانها.رقم_iterations - آرگومانها.warmup_iters)
        أكد 0 <= decay_نسبت <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_نسبت)) # coeff starts at 1 and goes to 0
        أرجع min_lr + coeff * (آرگومانها.learning_نرخ - min_lr)

    إذا device == "cuda":
        torch.cuda.reset_peak_حافظه_stats()
    timings = []
    norm = -1.0   # dummy value to print in inference-only mode
    لكل گام في دامنه(آرگومانها.رقم_iterations + 1):
        t0 = time.time()
        آخرین_گام = (گام == آرگومانها.رقم_iterations)

        # once in a while evaluate the validation dataset
        إذا (آرگومانها.val_loss_every > 0 \
            و (گام % آرگومانها.val_loss_every == 0 أو آخرین_گام)) \
            و (val_loader است ليس هیچ):
            model.eval()
            val_loader.reset()
            با torch.no_grad():
                val_loss = 0.0
                لكل _ في دامنه(آرگومانها.val_max_گامها):
                    x, y = val_loader.بعدی_batch()
                    x, y = x.مقصد(device), y.مقصد(device)
                    _, loss = model(x, y, return_logits=خطأ)
                    val_loss += loss.قلم()
                val_loss /= آرگومانها.val_max_گامها
            # log to console and to file
            print0(f"val loss {val_loss}")
            إذا master_process و logfile است ليس هیچ:
                با open(logfile, "a") كـ f:
                    f.اكتب("s:%d tel:%f\n" % (گام, val_loss))

        # once in a while perform model inference on the master process
        إذا (آرگومانها.sample_every > 0 \
            و (گام % آرگومانها.sample_every == 0 أو آخرین_گام)) \
            و master_process:
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
            إذا آرگومانها.use_hf:
                prompt_نشانهها = [model.tokenizer(x).درون_داد_ids لكل x في prompts]
            وإلا:  # Meta
                prompt_نشانهها = [model.tokenizer.encode(x, bos=درست, eos=خطأ) لكل x في prompts]

            generation_نشانهها = model.generate(prompt_نشانهها, max_gen_len=64, temperature=0.6, بالاترین_p=0.9, echo=خطأ)
            results = [{"generation": model.tokenizer.decode(t)} لكل t في generation_نشانهها]
            لكل prompt, نتیجه في zip(prompts, results):
                چاپ(prompt, انتها="")
                چاپ(f"{result['generation']}")
                چاپ("\n==================================\n")

        # bit confusing: we want to make sure to eval and sample on 0th iteration
        # but also after the very last iteration. so we loop for step <= num_iterations
        # instead of just < num_iterations (one extra due to <=), only to do
        # the validation/sampling one last time, and then we break right here as we're done.
        إذا آخرین_گام:
            اكسر

        # --------------- TRAINING SECTION BEGIN -----------------
        model.train()
        optimizer.zero_grad(مجموعه_مقصد_none=درست)
        # if we are trying to overfit a single batch, we reset the loader here
        إذا آرگومانها.overfit_single_batch:
            train_loader.reset()
        # micro-batch loop where we do gradient accumulation to reach desired total batch size
        lossf = 0.0 # for getting the mean loss (as simple float) over the accumulation steps
        لكل micro_گام في دامنه(grad_accum_گامها):
            # fetch a batch
            x, y = train_loader.بعدی_batch()
            x, y = x.مقصد(device), y.مقصد(device)
            إذا ddp:
                # we want only the last micro-step to sync grads in a DDP model
                # the official way to do this is with model.no_sync(), but that is a
                # context manager that bloats the code, so we just toggle this variable
                model.require_backward_grad_sync = (micro_گام == grad_accum_گامها - 1)
            # forward pass
            با ctx:
                _, loss = model(x, y, return_logits=خطأ)
                # we have to scale the loss to account for gradient accumulation,
                # because the gradients just add on each successive backward().
                # addition of gradients corresponds to a SUM in the objective, but
                # instead of a SUM we want MEAN, so we scale the loss here
                loss = loss / grad_accum_گامها
                lossf += loss.detach() # keep track of the mean loss
            # backward pass
            إذا ليس آرگومانها.inference_only:
                loss.backward()
        إذا ddp:
            dist.all_reduce(lossf, op=dist.ReduceOp.AVG)
        lossf = lossf.قلم()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), آرگومانها.grad_clip)
        # determine and set the learning rate for this iteration
        lr = get_lr(گام)
        لكل param_گروه في optimizer.param_groups:
            param_گروه['lr'] = lr
        # step the optimizer
        optimizer.گام()
        # --------------- TRAINING SECTION END -------------------
        # everything that follows now is just diagnostics, prints, logging, etc.

        # wait on the CPU for all device work to end so we get accurate per-iteration timings below
        إذا device == "mps":
            torch.mps.synchronize()
        وإلاإذا device == "cuda":
            torch.cuda.synchronize()
        # time and print
        t1 = time.time()
        # the 0th iteration is often an outlier (much slower) => skip logging it
        نشانهها_در_هر_دومی = grad_accum_گامها * ddp_world_اندازه * B * T / (t1-t0)
        print0(f"step {step+1:4d}/{args.num_iterations} | train loss {lossf:.6f} | norm {norm:.4f} | lr {lr:.2e} | ({(t1-t0)*1000:.2f} ms | {tokens_per_second:.0f} tok/s)")
        # log to logile
        إذا master_process و logfile است ليس هیچ:
            با open(logfile, "a") كـ f:
                f.اكتب("s:%d trl:%f\n" % (گام, lossf))

        # keep track of smooth timings, last 20 iterations
        إذا گام > 0 و گام > آرگومانها.رقم_iterations - 20:
            timings.append(t1-t0)

    # print the average of the last 20 timings, to get something smooth-ish
    timings = timings[-20:]
    print0(f"final {len(timings)} iters avg: {np.mean(timings)*1000:.3f}ms")
    print0(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

    # -------------------------------------------------------------------------
    # clean up nice
    إذا ddp:
        destroy_process_گروه()
