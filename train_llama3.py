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

దిగుమతి argparse
దిగుమతి os
దిగుమతి math
దిగుమతి glob
దిగుమతి inspect
తః contextlib దిగుమతి nullcontext
తః dataclasses దిగుమతి dataclass
తః pathlib దిగుమతి మార్గం_2
దిగుమతి time
తః typing దిగుమతి (
    Abstract_సమితి,
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

దిగుమతి numpy ఇవ np
దిగుమతి torch
దిగుమతి torch.nn ఇవ nn
తః torch.nn దిగుమతి functional ఇవ F
దిగుమతి torch._inductor.config ఇవ config
తః torch.nn.parallel దిగుమతి Distributed_దత్తాంశం_Parallel ఇవ DDP
తః torch.distributed దిగుమతి init_process_సమూహం, destroy_process_సమూహం
తః torch.distributed.optim దిగుమతి ZeroRedundancyOptimizer
దిగుమతి torch.distributed ఇవ dist

దిగుమతి tiktoken
తః tiktoken.ఎక్కించు దిగుమతి ఎక్కించు_tiktoken_bpe

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the LLaMA 3.x model

# Used in Grouped Query Attention (GQA), broadcasts the key and value tensors
నిర్వచనం repeat_kv(x: torch.Tensor, n_rep: పూర్ణాంకం) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, తల_dim = x.shape
    ఒకవేళ n_rep == 1:
        తిరిగి x
    తిరిగి (
        x[:, :, :, ఏమీలేదు, :]
        .expand(bs, slen, n_kv_heads, n_rep, తల_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, తల_dim)
    )

# -----------------------------------------------------------------------------
# RoPE related

నిర్వచనం reshape_for_ప్రసారం(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    ప్రతిజ్ఞా 0 <= 1 < ndim
    ప్రతిజ్ఞా freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d ఒకవేళ i == 1 లేదా i == ndim - 1 లేకపోతే 1 కొరకు i, d లో enumerate(x.shape)]
    తిరిగి freqs_cis.view(*shape)

నిర్వచనం apply_scaling(freqs: torch.Tensor):
    # Values obtained from grid search
    scale_గుణకం = 8
    దిగువ_freq_గుణకం = 1
    ఎగువ_freq_గుణకం = 4
    old_context_len = 8192  # original llama3 length

    దిగువ_freq_wavelen = old_context_len / దిగువ_freq_గుణకం
    ఎగువ_freq_wavelen = old_context_len / ఎగువ_freq_గుణకం
    new_freqs = []
    కొరకు freq లో freqs:
        wavelen = 2 * math.pi / freq
        ఒకవేళ wavelen < ఎగువ_freq_wavelen:
            new_freqs.append(freq)
        లేదావేళ wavelen > దిగువ_freq_wavelen:
            new_freqs.append(freq / scale_గుణకం)
        లేకపోతే:
            ప్రతిజ్ఞా దిగువ_freq_wavelen != ఎగువ_freq_wavelen
            smooth = (old_context_len / wavelen - దిగువ_freq_గుణకం) / (
                ఎగువ_freq_గుణకం - దిగువ_freq_గుణకం
            )
            new_freqs.append((1 - smooth) * freq / scale_గుణకం + smooth * freq)
    తిరిగి torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)

నిర్వచనం apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.దశాంశం().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.దశాంశం().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_ప్రసారం(freqs_cis, xq_)
    xq_బహిర్గతం = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_బహిర్గతం = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    తిరిగి xq_బహిర్గతం.type_as(xq), xk_బహిర్గతం.type_as(xk)

నిర్వచనం precompute_freqs_cis(
    dim: పూర్ణాంకం, చివర: పూర్ణాంకం, theta: దశాంశం = 10000.0, use_scaled: bool = అబద్ధం
):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].దశాంశం() / dim))
    t = torch.arange(చివర, device=freqs.device, dtype=torch.float32)
    ఒకవేళ use_scaled:
        freqs = apply_scaling(freqs)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    తిరిగి freqs_cis

# -----------------------------------------------------------------------------
# LLaMA building blocks

# LLaMA reference code explicitly implemented RMSNorm so we copy pasted it
# (https://github.com/meta-llama/llama-models/blob/main/models/llama3_1/api/model.py)
# we could also use nn.RMSNorm, it has slightly different numeric properties, but equivalent
తరగతి RMSNorm(torch.nn.Module):
    నిర్వచనం __init__(self, dim: పూర్ణాంకం, eps: దశాంశం = 1e-6):
        పర().__init__()
        self.eps = eps
        self.బరువు = nn.Parameter(torch.ones(dim))

    నిర్వచనం _norm(self, x):
        తిరిగి x * torch.rsqrt(x.pow(2).సగటు(-1, keepdim=నిజం) + self.eps)

    నిర్వచనం forward(self, x):
        output = self._norm(x.దశాంశం()).type_as(x)
        తిరిగి output * self.బరువు

తరగతి CausalSelfAttention(nn.Module):

    నిర్వచనం __init__(self, config):
        పర().__init__()
        ప్రతిజ్ఞా config.n_embd % config.n_తల == 0

        self.n_తల = config.n_తల
        self.n_kv_తల = config.n_kv_తల
        self.n_rep = self.n_తల // self.n_kv_తల
        self.hd = config.n_embd // config.n_తల
        self.use_kv = config.use_kv
        self.flash = config.flash

        self.c_attn = nn.రేఖీయ(config.n_embd, (config.n_తల + 2 * config.n_kv_తల) * self.hd, bias=అబద్ధం)  # key, query, value projections
        self.c_proj = nn.రేఖీయ(config.n_embd, config.n_embd, bias=అబద్ధం)  # output projection

        # static KV cache - we could alternatively allocate it outside of the model and just pass it in when needed
        ఒకవేళ self.use_kv:
            self.cache_k = torch.zeros((config.max_gen_batch_కొలత, config.ఖండం_కొలత, config.n_kv_తల, self.hd))
            self.cache_v = torch.zeros((config.max_gen_batch_కొలత, config.ఖండం_కొలత, config.n_kv_తల, self.hd))

    నిర్వచనం forward(self, x, freqs_cis=ఏమీలేదు, ప్రారంభం_pos=ఏమీలేదు, ముసుగు=ఏమీలేదు):
        B, T, C = x.కొలత() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.విభజించు([self.n_తల * self.hd, self.n_kv_తల * self.hd, self.n_kv_తల * self.hd], dim=-1)
        q, k, v = map(అజ్ఞాత t: t.view(B, T, -1, self.hd), (q, k, v))  # (B, T, NH, HD)

        q, k = apply_rotary_emb(q, k, freqs_cis=freqs_cis)  # rotate QK (rope)  <-- 1. difference compared to GPT-2

        ఒకవేళ self.use_kv మరియు కాదు self.training మరియు ప్రారంభం_pos >= 0:  # use kv-caching during inference
            self.cache_k[:B, ప్రారంభం_pos : ప్రారంభం_pos + T] = k
            self.cache_v[:B, ప్రారంభం_pos : ప్రారంభం_pos + T] = v
            k = self.cache_k[:B, : ప్రారంభం_pos + T]
            v = self.cache_v[:B, : ప్రారంభం_pos + T]

        k = repeat_kv(k, self.n_rep)  # GQA <-- 2. difference compared to GPT-2
        v = repeat_kv(v, self.n_rep)

        q, k, v = map(అజ్ఞాత t: t.transpose(1, 2), (q, k, v))  # (B, NH, T, HD)

        ఒకవేళ self.flash:
            # flashattention
            # if T == 1 no need to mask, otherwise the function complains
            # scaled_dot_product_attention expects a mask where value of True indicates that the element should take part in attention
            # our mask is the opposite, so we need to invert it
            y = F.scaled_dot_లబ్ధం_attention(q, k, v, ముసుగు == 0 ఒకవేళ T > 1 లేకపోతే ఏమీలేదు)
        లేకపోతే:
            # manual implementation of attention
            # this materializes the large (T,T) matrix for all the queries and keys
            scores = (q @ k.transpose(-2, -1)) * (1.0 / math.వర్గమూలం(self.hd))
            ఒకవేళ ముసుగు అయిన కాదు ఏమీలేదు:
                scores.masked_fill_(ముసుగు, torch.finfo(scores.dtype).min)
            att = F.softmax(scores.దశాంశం(), dim=-1).type_as(q)
            y = att @ v # (B, NH, T, T) x (B, NH, T, HD) -> (B, NH, T, HD)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        తిరిగి y

తరగతి MLP(nn.Module):

    నిర్వచనం __init__(self, config):
        పర().__init__()
        hidden_dim = 4 * config.n_embd
        hidden_dim = పూర్ణాంకం(2 * hidden_dim / 3)
        # custom dim factor multiplier
        ఒకవేళ config.ffn_dim_multiplier అయిన కాదు ఏమీలేదు:
            hidden_dim = పూర్ణాంకం(config.ffn_dim_multiplier * hidden_dim)
        hidden_dim = config.గుణిజం_యొక్క * ((hidden_dim + config.గుణిజం_యొక్క - 1) // config.గుణిజం_యొక్క)
        self.c_fc = nn.రేఖీయ(config.n_embd, hidden_dim, bias=అబద్ధం)
        self.c_fc2 = nn.రేఖీయ(config.n_embd, hidden_dim, bias=అబద్ధం)
        self.c_proj = nn.రేఖీయ(hidden_dim, config.n_embd, bias=అబద్ధం)

    నిర్వచనం forward(self, x):
        # SwiGLU self.c_proj(F.silu(self.c_fc2(x)) * self.c_fc(x))  <-- 3. difference compared to GPT-2
        x1 = self.c_fc(x)
        x2 = self.c_fc2(x)
        x2 = F.silu(x2)
        x = x1 * x2
        x = self.c_proj(x)
        తిరిగి x

తరగతి ఖండం(nn.Module):

    నిర్వచనం __init__(self, config):
        పర().__init__()
        self.ln_1 = RMSNorm(config.n_embd, config.norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = RMSNorm(config.n_embd, config.norm_eps)
        self.mlp = MLP(config)

    నిర్వచనం forward(self, x, freqs_cis=ఏమీలేదు, ప్రారంభం_pos=ఏమీలేదు, ముసుగు=ఏమీలేదు):
        x = x + self.attn(self.ln_1(x), freqs_cis, ప్రారంభం_pos, ముసుగు)
        x = x + self.mlp(self.ln_2(x))
        తిరిగి x

# -----------------------------------------------------------------------------
# The main LLaMA 3.1 model

@dataclass
తరగతి LlamaConfig:
    సంస్కరణ: str = "3.1"
    ఖండం_కొలత: పూర్ణాంకం = 8192
    vocab_కొలత: పూర్ణాంకం = 128256
    n_layer: పూర్ణాంకం = 32
    n_తల: పూర్ణాంకం = 32
    n_kv_తల: పూర్ణాంకం = 8
    n_embd: పూర్ణాంకం = 4096
    ffn_dim_multiplier: దశాంశం = 1.3
    గుణిజం_యొక్క: పూర్ణాంకం = 1024
    norm_eps: దశాంశం = 1e-5
    rope_theta: దశాంశం = 500000.0
    use_scaled_rope: bool = నిజం
    max_gen_batch_కొలత: పూర్ణాంకం = 4
    use_kv: bool = నిజం
    flash: bool = అబద్ధం  # use flashattention?

    నిర్వచనం __init__(self, **kwargs):
        కొరకు k, v లో kwargs.వస్తువులు():
            ఒకవేళ hasattr(self, k):
                setattr(self, k, v)
        ప్రతిజ్ఞా self.n_kv_తల <= self.n_తల
        ప్రతిజ్ఞా self.n_తల % self.n_kv_తల == 0
        ప్రతిజ్ఞా self.n_embd % self.n_తల == 0

తరగతి LLaMA(nn.Module):

    నిర్వచనం __init__(self, config):
        పర().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_కొలత, config.n_embd),
            h = nn.ModuleList([ఖండం(config) కొరకు _ లో శ్రేణి(config.n_layer)]),
            ln_f = RMSNorm(config.n_embd, config.norm_eps),
        ))
        self.lm_తల = nn.రేఖీయ(config.n_embd, config.vocab_కొలత, bias=అబద్ధం)

        # init all weights, use a torch rng object to be very careful
        self.init_rng = torch.Generator()
        self.init_rng.manual_విత్తనం(42)

        self.freqs_cis = precompute_freqs_cis(
            config.n_embd // config.n_తల,
            config.ఖండం_కొలత * 2,
            config.rope_theta,
            config.use_scaled_rope,
        )

    నిర్వచనం forward(self, idx, targets=ఏమీలేదు, return_logits=నిజం, ప్రారంభం_pos=0):
        _, t = idx.కొలత()
        ప్రతిజ్ఞా t <= self.config.ఖండం_కొలత, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        # forward the LLaMA model itself
        x = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        freqs_cis = self.freqs_cis[ప్రారంభం_pos:ప్రారంభం_pos+t]

        ముసుగు = torch.triu(torch.ones((t, t), device=తదుపరి_2(self.parameters()).device, dtype=torch.bool), diagonal=1)

        కొరకు i, ఖండం_2 లో enumerate(self.transformer.h):
            x = ఖండం_2(x, freqs_cis, ప్రారంభం_pos, ముసుగు)
        x = self.transformer.ln_f(x)

        ఒకవేళ targets అయిన కాదు ఏమీలేదు:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_తల(x).దశాంశం()
            loss = F.cross_entropy(logits.view(-1, logits.కొలత(-1)), targets.view(-1), ignore_సూచీ=-1)
        లేకపోతే:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_తల(x[:, [-1], :]).దశాంశం() # note: using list [-1] to preserve the time dim
            loss = ఏమీలేదు

        # there are performance reasons why not returning logits is prudent, if not needed
        ఒకవేళ కాదు return_logits:
            logits = ఏమీలేదు

        తిరిగి logits, loss

    @staticmethod
    నిర్వచనం adapt_llama_state_dict_కీలకాలు(checkpoint, config: LlamaConfig):
        # Modify key names from Meta's LLaMA to our LLaMA
        # our key names are derived from GPT-2's key names
        checkpoint['transformer.wte.weight'] = checkpoint.తీయి('tok_embeddings.weight')

        కొరకు i లో శ్రేణి(config.n_layer):
            కొరకు name లో ['attention_norm', 'ffn_norm']:
                old_కీలకం = f'layers.{i}.{name}.weight'  # e.g. layers.x.attention_norm.weight -> transformer.h.x.ln_1.weight
                new_కీలకం = f'transformer.h.{i}.ln_{1 if name == "attention_norm" else 2}.weight'
                checkpoint[new_కీలకం] = checkpoint.తీయి(old_కీలకం)

        కొరకు i లో శ్రేణి(config.n_layer):
            కొరకు name లో ['attention.wq', 'attention.wk', 'attention.wv']:
                old_కీలకం = f'layers.{i}.{name}.weight'
                new_కీలకం = f'transformer.h.{i}.attn.c_attn.weight'
                ఒకవేళ name == 'attention.wq':
                    checkpoint[new_కీలకం] = checkpoint.తీయి(old_కీలకం)
                లేకపోతే:  # merge 3 weights into transformer.h.x.attn.c_attn.weight
                    checkpoint[new_కీలకం] = torch.cat((checkpoint[new_కీలకం], checkpoint.తీయి(old_కీలకం)), dim=0)
            old_కీలకం = f'layers.{i}.attention.wo.weight'
            new_కీలకం = f'transformer.h.{i}.attn.c_proj.weight'
            checkpoint[new_కీలకం] = checkpoint.తీయి(old_కీలకం)

        ffn_map = {'w1': 'c_fc2', 'w2': 'c_proj', 'w3': 'c_fc'}
        కొరకు i లో శ్రేణి(config.n_layer):
            కొరకు name లో ['feed_forward.w1', 'feed_forward.w2', 'feed_forward.w3']:
                old_కీలకం = f'layers.{i}.{name}.weight'
                new_కీలకం = f'transformer.h.{i}.mlp.{ffn_map[name.split(".")[-1]]}.weight'
                checkpoint[new_కీలకం] = checkpoint.తీయి(old_కీలకం)

        checkpoint['transformer.ln_f.weight'] = checkpoint.తీయి('norm.weight')
        checkpoint['lm_head.weight'] = checkpoint.తీయి('output.weight')

        తిరిగి checkpoint

    @staticmethod
    నిర్వచనం adapt_llama_state_dict_కీలకాలు_hf(checkpoint, config: LlamaConfig):
        # Modify key names from HuggingFace's LLaMA to our LLaMA
        # our key names are derived from GPT-2's key names
        checkpoint['transformer.wte.weight'] = checkpoint.తీయి('model.embed_tokens.weight')

        # We need to unpermute K and V because HF script permuted the original Meta-LLaMA weights
        # see: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/convert_llama_weights_to_hf.py
        నిర్వచనం unpermute(w, n_heads, dim1, dim2):
            తిరిగి w.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)

        కొరకు i లో శ్రేణి(config.n_layer):
            కొరకు name లో ['input_layernorm', 'post_attention_layernorm']:
                old_కీలకం = f'model.layers.{i}.{name}.weight'  # e.g. layers.x.attention_norm.weight -> transformer.h.x.ln_1.weight
                new_కీలకం = f'transformer.h.{i}.ln_{1 if name == "input_layernorm" else 2}.weight'
                checkpoint[new_కీలకం] = checkpoint.తీయి(old_కీలకం)

        కొరకు i లో శ్రేణి(config.n_layer):
            కొరకు name లో ['self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj']:
                old_కీలకం = f'model.layers.{i}.{name}.weight'
                new_కీలకం = f'transformer.h.{i}.attn.c_attn.weight'
                ఒకవేళ name == 'self_attn.q_proj':
                    checkpoint[new_కీలకం] = unpermute(checkpoint.తీయి(old_కీలకం), config.n_తల, config.n_embd, config.n_embd)
                లేకపోతే:  # merge 3 weights into transformer.h.x.attn.c_attn.weight
                    tensor = checkpoint.తీయి(old_కీలకం)
                    ఒకవేళ name == 'self_attn.k_proj':
                        tensor = unpermute(tensor, config.n_kv_తల, config.n_kv_తల * (config.n_embd // config.n_తల), config.n_embd)
                    checkpoint[new_కీలకం] = torch.cat((checkpoint[new_కీలకం], tensor), dim=0)
            old_కీలకం = f'model.layers.{i}.self_attn.o_proj.weight'
            new_కీలకం = f'transformer.h.{i}.attn.c_proj.weight'
            checkpoint[new_కీలకం] = checkpoint.తీయి(old_కీలకం)

        ffn_map = {'gate_proj': 'c_fc2', 'down_proj': 'c_proj', 'up_proj': 'c_fc'}
        కొరకు i లో శ్రేణి(config.n_layer):
            కొరకు name లో ['gate_proj', 'down_proj', 'up_proj']:
                old_కీలకం = f'model.layers.{i}.mlp.{name}.weight'
                new_కీలకం = f'transformer.h.{i}.mlp.{ffn_map[name]}.weight'
                checkpoint[new_కీలకం] = checkpoint.తీయి(old_కీలకం)

        checkpoint['transformer.ln_f.weight'] = checkpoint.తీయి('model.norm.weight')

        తిరిగి checkpoint

    @classmethod
    నిర్వచనం from_pretrained_llama3_hf(cls, model_id):
        """Loads pretrained LLaMA model weights from HuggingFace"""
        తః transformers దిగుమతి AutoModelForCausalLM, AutoTokenizer
        ప్రతిజ్ఞా model_id == "meta-llama/Meta-Llama-3.1-8B", "Only the 8B-base model is supported for now"
        model_వాదనలు = LlamaConfig()

        model = AutoModelForCausalLM.from_pretrained(model_id)
        checkpoint = LLaMA.adapt_llama_state_dict_కీలకాలు_hf(model.state_dict(), model_వాదనలు)

        original_default_type = torch.get_default_dtype()  # save the default type
        torch.సమితి_default_tensor_type(torch.cuda.BFloat16Tensor)  # much faster loading
        model = LLaMA(model_వాదనలు)
        model.ఎక్కించు_state_dict(checkpoint, strict=అబద్ధం)
        torch.సమితి_default_tensor_type(torch.tensor([], dtype=original_default_type, device="cpu").type())  # restore default type

        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_id = 128004  # this is the pad token id for LLaMA 3.1 base, we need to set this explicitly as our generate func expects it
        tokenizer.stop_టోకెన్లు = [tokenizer.eos_token_id]
        model.tokenizer = tokenizer
        తిరిగి model

    @classmethod
    నిర్వచనం from_pretrained_llama3_meta(cls, ckpt_dir, tokenizer_మార్గం):
        """Loads pretrained LLaMA model weights from a checkpoint directory"""
        model_వాదనలు = LlamaConfig()

        ckpt_మార్గం = sorted(మార్గం_2(ckpt_dir).glob("*.pth"))[0]
        checkpoint = torch.ఎక్కించు(ckpt_మార్గం, map_location="cpu", weights_only=నిజం)
        checkpoint = LLaMA.adapt_llama_state_dict_కీలకాలు(checkpoint, model_వాదనలు)

        original_default_type = torch.get_default_dtype()  # save the default type
        torch.సమితి_default_tensor_type(torch.cuda.BFloat16Tensor)  # much faster loading
        model = LLaMA(model_వాదనలు)
        model.ఎక్కించు_state_dict(checkpoint, strict=అబద్ధం)
        torch.సమితి_default_tensor_type(torch.tensor([], dtype=original_default_type, device="cpu").type())  # restore default type

        tokenizer = Tokenizer(model_మార్గం=tokenizer_మార్గం)
        model.tokenizer = tokenizer
        తిరిగి model

    నిర్వచనం configure_optimizers(self, బరువు_decay, learning_రేటు, betas, device_type, zero_stage):
        # start with all of the candidate parameters
        param_dict = {pn: p కొరకు pn, p లో self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p కొరకు pn, p లో param_dict.వస్తువులు() ఒకవేళ p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p కొరకు n, p లో param_dict.వస్తువులు() ఒకవేళ p.dim() >= 2]
        nodecay_params = [p కొరకు n, p లో param_dict.వస్తువులు() ఒకవేళ p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': బరువు_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        అంకె_decay_params = కూడిక(p.numel() కొరకు p లో decay_params)
        అంకె_nodecay_params = కూడిక(p.numel() కొరకు p లో nodecay_params)
        print0(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print0(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_అందుబాటు = 'fused' లో inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_అందుబాటు మరియు device_type == 'cuda'
        print0(f"using fused AdamW: {use_fused}")
        ఒకవేళ zero_stage == 1:
            print0("using ZeroRedundancyOptimizer")
            optimizer = ZeroRedundancyOptimizer(**optim_groups[0], optimizer_class=torch.optim.AdamW,
                                                lr=learning_రేటు, betas=betas, fused=use_fused)
            optimizer.కూడు_param_సమూహం(optim_groups[1])
        లేకపోతే:
            print0("using regular AdamW")
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_రేటు, betas=betas, fused=use_fused)
        తిరిగి optimizer

    @torch.inference_బహుళకం()
    నిర్వచనం generate(
        self,
        prompt_టోకెన్లు: List[List[పూర్ణాంకం]],
        max_gen_len: పూర్ణాంకం,
        temperature: దశాంశం = 0.6,
        శిఖరం_p: దశాంశం = 0.9,
        echo: bool = అబద్ధం,
    ) -> Tuple[List[List[పూర్ణాంకం]], Optional[List[List[దశాంశం]]]]:
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
        bsz = పొడవు(prompt_టోకెన్లు)
        ప్రతిజ్ఞా bsz <= self.config.max_gen_batch_కొలత, f"Batch size {bsz} exceeds the maximum generation batch size {self.config.max_gen_batch_size}"
        device = తదుపరి_2(self.parameters()).device

        min_prompt_len = min(పొడవు(t) కొరకు t లో prompt_టోకెన్లు)
        max_prompt_len = max(పొడవు(t) కొరకు t లో prompt_టోకెన్లు)
        ప్రతిజ్ఞా max_prompt_len <= self.config.ఖండం_కొలత, f"Prompt length {max_prompt_len} exceeds the maximum block size {self.config.block_size}"
        మొత్తం_len = min(self.config.ఖండం_కొలత, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_id
        టోకెన్లు = torch.నిండిన((bsz, మొత్తం_len), pad_id, dtype=torch.దీర్ఘ, device=device)
        కొరకు idx, t లో enumerate(prompt_టోకెన్లు):
            టోకెన్లు[idx, : పొడవు(t)] = torch.tensor(t, dtype=torch.దీర్ఘ, device=device)

        prev_pos = 0
        eos_reached = torch.tensor([అబద్ధం] * bsz, device=device)
        ఆదానం_పాఠ్యం_ముసుగు = టోకెన్లు != pad_id

        ఒకవేళ min_prompt_len == మొత్తం_len:
            logits, _ = self.forward(టోకెన్లు, ప్రారంభం_pos=prev_pos)

        stop_టోకెన్లు = torch.tensor(list(self.tokenizer.stop_టోకెన్లు)).గమ్యం(device)

        కొరకు cur_pos లో శ్రేణి(min_prompt_len, మొత్తం_len):
            logits, _ = self.forward(టోకెన్లు[:, prev_pos:cur_pos], ప్రారంభం_pos=prev_pos)
            ఒకవేళ temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                తదుపరి_token = sample_శిఖరం_p(probs, శిఖరం_p)
            లేకపోతే:
                తదుపరి_token = torch.argmax(logits[:, -1], dim=-1)

            తదుపరి_token = తదుపరి_token.reshape(-1)
            # only replace token if prompt has already been generated
            తదుపరి_token = torch.where(ఆదానం_పాఠ్యం_ముసుగు[:, cur_pos], టోకెన్లు[:, cur_pos], తదుపరి_token)
            టోకెన్లు[:, cur_pos] = తదుపరి_token
            eos_reached |= ~ఆదానం_పాఠ్యం_ముసుగు[:, cur_pos] & torch.isin(తదుపరి_token, stop_టోకెన్లు)
            prev_pos = cur_pos
            ఒకవేళ all(eos_reached):
                ఆపు

        బహిర్గతం_టోకెన్లు = []
        కొరకు i, toks లో enumerate(టోకెన్లు.tolist()):
            # cut to max gen len
            ప్రారంభం = 0 ఒకవేళ echo లేకపోతే పొడవు(prompt_టోకెన్లు[i])
            toks = toks[ప్రారంభం : పొడవు(prompt_టోకెన్లు[i]) + max_gen_len]
            # cut to after eos tok if any
            కొరకు stop_token లో self.tokenizer.stop_టోకెన్లు:
                ప్రయత్నించు:
                    eos_idx = toks.సూచీ(stop_token)
                    toks = toks[:eos_idx]
                మినహాయింపు విలువ_Error:
                    దాటు
            బహిర్గతం_టోకెన్లు.append(toks)
        తిరిగి బహిర్గతం_టోకెన్లు

# -----------------------------------------------------------------------------
# sampling utils

నిర్వచనం sample_శిఖరం_p(probs, p):
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
    probs_క్రమీకరించు, probs_idx = torch.క్రమీకరించు(probs, dim=-1, descending=నిజం)
    probs_కూడిక = torch.cumsum(probs_క్రమీకరించు, dim=-1)
    ముసుగు = probs_కూడిక - probs_క్రమీకరించు > p
    probs_క్రమీకరించు[ముసుగు] = 0.0
    probs_క్రమీకరించు.div_(probs_క్రమీకరించు.కూడిక(dim=-1, keepdim=నిజం))
    తదుపరి_token = torch.multinomial(probs_క్రమీకరించు, అంకె_samples=1)
    తదుపరి_token = torch.gather(probs_idx, -1, తదుపరి_token)
    తిరిగి తదుపరి_token

# -----------------------------------------------------------------------------
# Llama 3.1 Tokenizer

# The tiktoken tokenizer can handle <=400k chars without
# pyo3_runtime.PanicException.
TIKTOKEN_MAX_ENCODE_CHARS = 400_000

# https://github.com/openai/tiktoken/issues/195
# Here we iterate over subsequences and split if we exceed the limit
# of max consecutive non-whitespace or whitespace characters.
MAX_NO_WHITESPACES_CHARS = 25_000


తరగతి Tokenizer:
    """
    Tokenizing and encoding/decoding text using the Tiktoken tokenizer.
    """

    special_టోకెన్లు_2: Dict[str, పూర్ణాంకం]

    అంకె_reserved_special_టోకెన్లు = 256

    pat_str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"  # noqa: E501

    నిర్వచనం __init__(self, model_మార్గం: str):
        """
        Initializes the Tokenizer with a Tiktoken model.

        Args:
            model_path (str): The path to the Tiktoken model file.
        """
        ప్రతిజ్ఞా os.మార్గం.isfile(model_మార్గం), model_మార్గం

        mergeable_ranks = ఎక్కించు_tiktoken_bpe(model_మార్గం)
        అంకె_పునాది_టోకెన్లు = పొడవు(mergeable_ranks)
        special_టోకెన్లు_2 = [
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
        reserved_టోకెన్లు = [
            f"<|reserved_special_token_{2 + i}|>"
            కొరకు i లో శ్రేణి(self.అంకె_reserved_special_టోకెన్లు - పొడవు(special_టోకెన్లు_2))
        ]
        special_టోకెన్లు_2 = special_టోకెన్లు_2 + reserved_టోకెన్లు

        self.special_టోకెన్లు_2 = {
            token: అంకె_పునాది_టోకెన్లు + i కొరకు i, token లో enumerate(special_టోకెన్లు_2)
        }
        self.model = tiktoken.Encoding(
            name=మార్గం_2(model_మార్గం).name,
            pat_str=self.pat_str,
            mergeable_ranks=mergeable_ranks,
            special_టోకెన్లు_2=self.special_టోకెన్లు_2,
        )

        self.n_పదాలు: పూర్ణాంకం = అంకె_పునాది_టోకెన్లు + పొడవు(special_టోకెన్లు_2)
        # BOS / EOS token IDs
        self.bos_id: పూర్ణాంకం = self.special_టోకెన్లు_2["<|begin_of_text|>"]
        self.eos_id: పూర్ణాంకం = self.special_టోకెన్లు_2["<|end_of_text|>"]
        self.eot_id: పూర్ణాంకం = self.special_టోకెన్లు_2["<|eot_id|>"]
        self.eom_id: పూర్ణాంకం = self.special_టోకెన్లు_2["<|eom_id|>"]
        self.python_ట్యాగ్_id = self.special_టోకెన్లు_2["<|python_tag|>"]
        self.pad_id: పూర్ణాంకం = self.special_టోకెన్లు_2["<|finetune_right_pad_id|>"]
        # hardcoded stop tokens for the base model
        self.stop_టోకెన్లు = [
            self.special_టోకెన్లు_2["<|begin_of_text|>"],
            self.special_టోకెన్లు_2["<|end_of_text|>"],
        ]

    నిర్వచనం encode(
        self,
        s: str,
        *,
        bos: bool,
        eos: bool,
        allowed_special: Optional[Union[Literal["all"], Abstract_సమితి[str]]] = ఏమీలేదు,
        disallowed_special: Union[Literal["all"], Collection[str]] = (),
    ) -> List[పూర్ణాంకం]:
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
        ఒకవేళ allowed_special అయిన ఏమీలేదు:
            allowed_special = సమితి()
        ప్రతిజ్ఞా type(s) అయిన str

        substrs = (
            substr
            కొరకు i లో శ్రేణి(0, పొడవు(s), TIKTOKEN_MAX_ENCODE_CHARS)
            కొరకు substr లో self.విభజించు_whitespaces_or_nonwhitespaces(
                s[i : i + TIKTOKEN_MAX_ENCODE_CHARS], MAX_NO_WHITESPACES_CHARS
            )
        )
        t: List[పూర్ణాంకం] = []
        కొరకు substr లో substrs:
            t.extend(
                self.model.encode(
                    substr,
                    allowed_special=allowed_special,
                    disallowed_special=disallowed_special,
                )
            )
        ఒకవేళ bos:
            t.చొప్పించు(0, self.bos_id)
        ఒకవేళ eos:
            t.append(self.eos_id)
        తిరిగి t

    నిర్వచనం decode(self, t: Sequence[పూర్ణాంకం]) -> str:
        # Typecast is safe here. Tiktoken doesn't do anything list-related with the sequence.
        తిరిగి self.model.decode(cast(List[పూర్ణాంకం], t))

    @staticmethod
    నిర్వచనం విభజించు_whitespaces_or_nonwhitespaces(
        s: str, max_consecutive_ముక్క_len: పూర్ణాంకం
    ) -> Iterator[str]:
        """
        Splits the string `s` so that each substring contains no more than `max_consecutive_slice_len`
        consecutive whitespaces or consecutive non-whitespaces.
        """
        current_ముక్క_len = 0
        current_ముక్క_is_space = s[0].isspace() ఒకవేళ పొడవు(s) > 0 లేకపోతే అబద్ధం
        ముక్క_ప్రారంభం = 0

        కొరకు i లో శ్రేణి(పొడవు(s)):
            is_now_space = s[i].isspace()

            ఒకవేళ current_ముక్క_is_space ^ is_now_space:
                current_ముక్క_len = 1
                current_ముక్క_is_space = is_now_space
            లేకపోతే:
                current_ముక్క_len += 1
                ఒకవేళ current_ముక్క_len > max_consecutive_ముక్క_len:
                    ఇవ్వు s[ముక్క_ప్రారంభం:i]
                    ముక్క_ప్రారంభం = i
                    current_ముక్క_len = 1
        ఇవ్వు s[ముక్క_ప్రారంభం:]

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

నిర్వచనం peek_దత్తాంశం_shard(filename):
    # only reads the header, returns header data
    తోడుగా open(filename, "rb") ఇవ f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        శీర్షిక = np.frombuffer(f.పఠ(256*4), dtype=np.int32)
    ఒకవేళ శీర్షిక[0] != 20240801:
        ముద్రించు("ERROR: magic number mismatch in the data .bin file!")
        exit(1)
    ప్రతిజ్ఞా శీర్షిక[1] == 7, "unsupported version"
    ntok = శీర్షిక[2] # number of tokens (claimed)
    తిరిగి ntok # for now just return the number of tokens

నిర్వచనం ఎక్కించు_దత్తాంశం_shard(filename):
    తోడుగా open(filename, "rb") ఇవ f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        శీర్షిక = np.frombuffer(f.పఠ(256*4), dtype=np.int32)
        ప్రతిజ్ఞా శీర్షిక[0] == 20240801, "magic number mismatch in the data .bin file"
        ప్రతిజ్ఞా శీర్షిక[1] == 7, "unsupported version"
        ntok = శీర్షిక[2] # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        టోకెన్లు = np.frombuffer(f.పఠ(), dtype=np.uint32)
    ప్రతిజ్ఞా పొడవు(టోకెన్లు) == ntok, "number of tokens read does not match header?"
    తిరిగి టోకెన్లు

తరగతి Distributed_Sharded_దత్తాంశం_Loader:
    """
    This DataLoader is both:
    - distributed (works correctly in case of multiple processes in DDP)
    - sharded (supports datasets that are broken up into multiple data shards)
    It is not *permuted*, meaning that it itearates over the data in the order
    of the dataset on disk, so the user should make sure to shuffle their examples
    during the creation of their data shards for best performance.
    """
    నిర్వచనం __init__(self, filename_నమూనా, B, T, process_rank, అంకె_processes):
        self.process_rank = process_rank
        self.అంకె_processes = అంకె_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_నమూనా))
        ప్రతిజ్ఞా పొడవు(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_మొత్తం = 0
        కొరకు fname లో self.files:
            shard_ntok = peek_దత్తాంశం_shard(fname)
            ప్రతిజ్ఞా shard_ntok >= అంకె_processes * B * T + 1
            ntok_మొత్తం += shard_ntok
        self.ntok_మొత్తం = ntok_మొత్తం
        print0(f"DataLoader: total number of tokens: {ntok_total:,} across {len(self.files)} files")

        # kick things off
        self.current_shard = ఏమీలేదు
        self.reset()

    నిర్వచనం reset(self):
        # we're being a bit clever here: if we already had shard 0 loaded,
        # then don't do the work to reload it, just reset the pointer
        ఒకవేళ self.current_shard != 0:
            self.current_shard = 0
            self.టోకెన్లు = ఎక్కించు_దత్తాంశం_shard(self.files[self.current_shard])
        self.current_position = self.process_rank * self.B * self.T

    నిర్వచనం advance(self): # advance to next data shard
        self.current_shard = (self.current_shard + 1) % పొడవు(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.టోకెన్లు = ఎక్కించు_దత్తాంశం_shard(self.files[self.current_shard])

    నిర్వచనం తదుపరి_batch(self):
        B = self.B
        T = self.T
        buf = self.టోకెన్లు[self.current_position : self.current_position+B*T+1]
        buf = torch.tensor(buf, dtype=torch.దీర్ఘ)
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the start pointer in current shard
        self.current_position += B * T * self.అంకె_processes
        # if loading the next batch would be out of bounds advance the shard
        ఒకవేళ self.current_position + (B * T * self.అంకె_processes + 1) > పొడవు(self.టోకెన్లు):
            self.advance()
        తిరిగి x, y

# -----------------------------------------------------------------------------
# Python -> C bridge utilities for saving params/grads/activations to .bin files

నిర్వచనం write_fp32(tensor, file):
    t = tensor.detach().cpu().గమ్యం(torch.float32)
    b = t.numpy().tobytes()
    file.లిఖ(b)

నిర్వచనం write_bf16(tensor, file):
    t = tensor.detach().cpu().గమ్యం(torch.bfloat16)
    # numpy doesn't have bf16 datatype so we have to trick it
    t = t.view(torch.int16) # trick: reinterpret as int16
    b = t.numpy().tobytes()
    file.లిఖ(b)

నిర్వచనం write_tensors(model_tensors, L, file, dtype):
    # writes LLaMA 3 model's weights to a binary file
    ప్రతిజ్ఞా dtype లో {"float32", "bfloat16"}
    write_fun = write_fp32 ఒకవేళ dtype == "float32" లేకపోతే write_bf16
    write_fun(model_tensors["transformer.wte.weight"], file) # (V, C)
    కొరకు i లో శ్రేణి(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_1.weight"], file)
    కొరకు i లో శ్రేణి(L): # (L, 3C, C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_attn.weight"], file)
    కొరకు i లో శ్రేణి(L): # (L, C, C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_proj.weight"], file)
    కొరకు i లో శ్రేణి(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_2.weight"], file)
    కొరకు i లో శ్రేణి(L): # (L, 4C, C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_fc.weight"], file)
    కొరకు i లో శ్రేణి(L): # (L, 4C, C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_fc2.weight"], file)
    కొరకు i లో శ్రేణి(L): # (L, C, 4C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_proj.weight"], file)
    write_fun(model_tensors["transformer.ln_f.weight"], file) # (C, )
    write_fun(model_tensors["lm_head.weight"], file) # (V, C)

నిర్వచనం write_model(model, filename, dtype):
    # everything we need to instantiate the model
    # 1) header is: version int, LLaMAConfig ints, padding to 1024 bytes
    ప్రతిజ్ఞా dtype లో {"float32", "bfloat16"}
    సంస్కరణ = {
        "float32": 3, # 3: all tensors are fp32
        "bfloat16": 5, # 5: all tensors are bf16
    }[dtype]
    శీర్షిక = torch.zeros(256, dtype=torch.int32)
    శీర్షిక[0] = 20240803 # magic
    శీర్షిక[1] = సంస్కరణ # checkpoint version
    శీర్షిక[2] = model.config.ఖండం_కొలత
    శీర్షిక[3] = model.config.vocab_కొలత
    శీర్షిక[4] = model.config.n_layer
    శీర్షిక[5] = model.config.n_తల
    శీర్షిక[6] = model.config.n_kv_తల
    శీర్షిక[7] = model.config.n_embd
    శీర్షిక[8] = model.config.ffn_dim_multiplier
    శీర్షిక[9] = model.config.గుణిజం_యొక్క
    శీర్షిక[10] = model.config.norm_eps
    శీర్షిక[11] = model.config.rope_theta
    శీర్షిక[12] = model.config.use_scaled_rope
    శీర్షిక[13] = model.config.max_gen_batch_కొలత
    శీర్షిక[14] = పూర్ణాంకం(model.config.సంస్కరణ.విభజించు('.')[0]) # major version
    శీర్షిక[15] = పూర్ణాంకం(model.config.సంస్కరణ.విభజించు('.')[1]) # minor version
    # 2) the parameters follow the header
    params = {name: param.cpu() కొరకు name, param లో model.named_parameters()}
    # now write to file
    తోడుగా open(filename, "wb") ఇవ file:
        file.లిఖ(శీర్షిక.numpy().tobytes()) #శీర్షిక
        write_tensors(params, model.config.n_layer, file, dtype) # params
    ముద్రించు(f"wrote {filename}")

నిర్వచనం write_state(model, x, y, logits, loss, filename):
    # the state is used for debugging.
    # it contains information about the input, logits, loss, and the parameter gradients
    # this can be used for checking the computation correctness in C
    శీర్షిక = torch.zeros(256, dtype=torch.int32)
    శీర్షిక[0] = 20240803 # magic
    శీర్షిక[1] = x.కొలత(0) # batch size of the batch, B
    శీర్షిక[2] = x.కొలత(1) # temporal extent of the batch, T
    grads = {name: param.grad.cpu() కొరకు name, param లో model.named_parameters()}
    తోడుగా open(filename, "wb") ఇవ file:
        #శీర్షిక
        file.లిఖ(శీర్షిక.numpy().tobytes())
        # input x
        file.లిఖ(x.cpu().numpy().astype("int32").tobytes()) # (B, T)
        # targets y
        file.లిఖ(y.cpu().numpy().astype("int32").tobytes()) # (B, T)
        # logits (result of the model forward pass)
        write_fp32(logits.cpu(), file)
        # loss (single float, result of the cross entropy loss)
        write_fp32(loss.cpu(), file)
        # gradients
        write_tensors(grads, model.config.n_layer, file, "float32")
    ముద్రించు(f"wrote {filename}")

# -----------------------------------------------------------------------------
# int main

నిర్వచనం print0(*వాదనలు, **kwargs):
    # modified print that only prints from the master process
    # if this is not a distributed run, it's just a print
    ఒకవేళ పూర్ణాంకం(os.environ.get("RANK", 0)) == 0:
        ముద్రించు(*వాదనలు, **kwargs)

ఒకవేళ __name__ == "__main__":
    print0(f"Running pytorch {torch.version.__version__}")

    # default settings will overfit a tiny batch of data
    # and save model weights and debug state to disk on the first iteration
    parser = argparse.వాదం_Parser()
    parser.కూడు_వాదం("--use_hf", type=పూర్ణాంకం, అప్రమేయ=1, help="use HuggingFace (default) or use Meta's model")
    parser.కూడు_వాదం("--ckpt_dir", type=str, అప్రమేయ=ఏమీలేదు, help="path to llama3 model checkpoint (needed if use_hf=0)")
    parser.కూడు_వాదం("--tokenizer_path", type=str, అప్రమేయ=ఏమీలేదు, help="path to llama3 tokenizer (needed if use_hf=0)")
    # file system input / output
    parser.కూడు_వాదం("--input_bin", type=str, అప్రమేయ="dev/data/tinyshakespeare/tiny_shakespeare_val.bin", help="input .bin to train on")
    parser.కూడు_వాదం("--input_val_bin", type=str, అప్రమేయ="", help="input .bin to eval validation loss on")
    parser.కూడు_వాదం("--output_dir", type=str, అప్రమేయ="", help="output directory to which to write logs and checkpoints")
    parser.కూడు_వాదం("--model", type=str, అప్రమేయ="meta-llama/Meta-Llama-3.1-8B", help="chose the llama model")
    # token layout for each step of the optimization
    parser.కూడు_వాదం("--batch_size", type=పూర్ణాంకం, అప్రమేయ=4, help="batch size, in units of #batch dimensions")
    parser.కూడు_వాదం("--sequence_length", type=పూర్ణాంకం, అప్రమేయ=64, help="sequence length")
    parser.కూడు_వాదం("--total_batch_size", type=పూర్ణాంకం, అప్రమేయ=256, help="total desired batch size, in units of #tokens")
    # workload (number of steps)
    parser.కూడు_వాదం("--num_iterations", type=పూర్ణాంకం, అప్రమేయ=10, help="number of iterations to run")
    parser.కూడు_వాదం("--inference_only", type=పూర్ణాంకం, అప్రమేయ=0, help="only run inference")
    # optimization
    parser.కూడు_వాదం("--learning_rate", type=దశాంశం, అప్రమేయ=1e-5, help="learning rate warmup iterations")
    parser.కూడు_వాదం("--warmup_iters", type=పూర్ణాంకం, అప్రమేయ=0, help="learning rate warmup iterations")
    parser.కూడు_వాదం("--learning_rate_decay_frac", type=దశాంశం, అప్రమేయ=1.0, help="learning rate warmup iterations")
    parser.కూడు_వాదం("--weight_decay", type=దశాంశం, అప్రమేయ=0.0, help="weight decay")
    parser.కూడు_వాదం("--grad_clip", type=దశాంశం, అప్రమేయ=1.0, help="maximum gradient magnitude")
    # evaluation
    parser.కూడు_వాదం("--val_loss_every", type=పూర్ణాంకం, అప్రమేయ=0, help="every how mant steps to evaluate val loss?")
    parser.కూడు_వాదం("--val_max_steps", type=పూర్ణాంకం, అప్రమేయ=20, help="how many batches of val to average?")
    parser.కూడు_వాదం("--sample_every", type=పూర్ణాంకం, అప్రమేయ=0, help="how often to sample from the model?")
    # debugging
    parser.కూడు_వాదం("--overfit_single_batch", type=పూర్ణాంకం, అప్రమేయ=1, help="overfit just one batch of data")
    # numerics
    parser.కూడు_వాదం("--tensorcores", type=పూర్ణాంకం, అప్రమేయ=0, help="use tensorcores")
    # memory management
    parser.కూడు_వాదం("--device", type=str, అప్రమేయ="", help="by default we autodetect, or set it here")
    parser.కూడు_వాదం("--compile", type=పూర్ణాంకం, అప్రమేయ=0, help="torch.compile the model")
    parser.కూడు_వాదం("--dtype", type=str, అప్రమేయ="bfloat16", help="float32|float16|bfloat16")
    parser.కూడు_వాదం("--zero_stage", type=పూర్ణాంకం, అప్రమేయ=0, help="zero redundancy optimizer stage (0/1/2/3)")
    # python -> C bridge
    parser.కూడు_వాదం("--write_tensors", type=పూర్ణాంకం, అప్రమేయ=0, help="write tensors to disk")
    వాదనలు = parser.parse_వాదనలు()

    # args error checking and convenience variables
    B, T = వాదనలు.batch_కొలత, వాదనలు.sequence_నిడివి
    ప్రతిజ్ఞా 1 <= T <= 8192, "sequence length must be between 1 and 8192"
    ప్రతిజ్ఞా వాదనలు.dtype లో {"float32", "float16", "bfloat16"}
    ప్రతిజ్ఞా వాదనలు.model లో {"meta-llama/Meta-Llama-3.1-8B"}  # only 8B base model supported for now

    # create the logging directory if it does not exist
    logfile = ఏమీలేదు
    ఒకవేళ వాదనలు.output_dir:
        os.makedirs(వాదనలు.output_dir, exist_ok=నిజం)
        logfile = os.మార్గం.join(వాదనలు.output_dir, "main.log")
        # create the log file "main.log" inside it, and wipe it clean
        తోడుగా open(logfile, "w") ఇవ f:
            దాటు

    # set up DDP (distributed data parallel). torchrun sets this env variable
    ddp = పూర్ణాంకం(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    ఒకవేళ ddp:
        # use of DDP atm demands CUDA, we set the device appropriately according to rank
        ప్రతిజ్ఞా torch.cuda.is_అందుబాటు(), "for now i think we need CUDA for DDP"
        init_process_సమూహం(backend='nccl')
        ddp_rank = పూర్ణాంకం(os.environ['RANK'])
        ddp_local_rank = పూర్ణాంకం(os.environ['LOCAL_RANK'])
        ddp_world_కొలత = పూర్ణాంకం(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.సమితి_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        విత్తనం_స్థానభ్రంశం = 0 # each process gets the exact same seed
        zero_stage = వాదనలు.zero_stage
    లేకపోతే:
        ddp_rank = 0
        ddp_local_rank = 0
        zero_stage = 0
        ddp_world_కొలత = 1
        master_process = నిజం
        విత్తనం_స్థానభ్రంశం = 0
        # select the device
        ఒకవేళ వాదనలు.device:
            # provided explicitly by the user
            device = వాదనలు.device
        లేకపోతే:
            # attempt to autodetect the device
            device = "cpu"
            ఒకవేళ torch.cuda.is_అందుబాటు():
                device = "cuda"
            లేదావేళ hasattr(torch.backends, "mps") మరియు torch.backends.mps.is_అందుబాటు():
                device = "mps"
    device_type = 'cuda' ఒకవేళ 'cuda' లో device లేకపోతే 'cpu'
    ప్రతిజ్ఞా device_type లో {'cuda'}, "GPU required to run LLaMA 3"  # we need to load LLaMA as bf16 on CUDA
    ముద్రించు(f"using device: {device}")

    # calculate gradient accumulation from the desired total batch size and the current run configuration
    టోకెన్లు_చొప్పున_fwdbwd = B * T * ddp_world_కొలత
    ప్రతిజ్ఞా వాదనలు.మొత్తం_batch_కొలత % టోకెన్లు_చొప్పున_fwdbwd == 0
    grad_accum_అడుగులు = వాదనలు.మొత్తం_batch_కొలత // టోకెన్లు_చొప్పున_fwdbwd
    print0(f"total desired batch size: {args.total_batch_size}")
    print0(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    # set up a context manager following the desired dtype and device
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[వాదనలు.dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) ఒకవేళ (device_type == "cuda") లేకపోతే nullcontext()

    # rng / reproducibility
    torch.manual_విత్తనం(42)
    ఒకవేళ torch.cuda.is_అందుబాటు():
        torch.cuda.manual_విత్తనం(42)

    # set the torch precision mode to use TensorFloat32 (TF32) for matmuls
    # docs https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    ఒకవేళ వాదనలు.tensorcores:
        torch.సమితి_float32_matmul_precision('high')

    # init the model
    ఒకవేళ వాదనలు.use_hf:
        model = LLaMA.from_pretrained_llama3_hf(వాదనలు.model)
    లేకపోతే:  # use Meta's checkpoint
        ప్రతిజ్ఞా వాదనలు.ckpt_dir అయిన కాదు ఏమీలేదు మరియు os.మార్గం.exists(వాదనలు.ckpt_dir), f"llama3 ckpt dir {args.ckpt_dir} does not exist"
        ప్రతిజ్ఞా వాదనలు.tokenizer_మార్గం అయిన కాదు ఏమీలేదు మరియు os.మార్గం.exists(వాదనలు.tokenizer_మార్గం), f"llama3 tokenizer path {args.tokenizer_path} does not exist"
        model = LLaMA.from_pretrained_llama3_meta(వాదనలు.ckpt_dir, వాదనలు.tokenizer_మార్గం)

    model.train()
    ఒకవేళ వాదనలు.compile:
        ఒకవేళ hasattr(config, "coordinate_descent_tuning"):
            config.coordinate_descent_tuning = నిజం # suggested by @Chillee
        print0("compiling the model...")
        model = torch.compile(model)

    # -------------------------------------------------------------------------
    # Our own version of a simple DistributedDataLoader

    # load tokens
    train_loader = Distributed_Sharded_దత్తాంశం_Loader(వాదనలు.ఆదానం_bin, B, T, ddp_rank, ddp_world_కొలత)
    val_loader = ఏమీలేదు
    ఒకవేళ వాదనలు.ఆదానం_val_bin:
        val_loader = Distributed_Sharded_దత్తాంశం_Loader(వాదనలు.ఆదానం_val_bin, B, T, ddp_rank, ddp_world_కొలత)

    # -------------------------------------------------------------------------
    # PyTorch -> C bridge: save some weights and state for C to load later as reference

    # do one forward pass to generate ground truth for our C tests
    ఒకవేళ master_process మరియు వాదనలు.write_tensors మరియు (కాదు వాదనలు.inference_only):
        x, y = train_loader.తదుపరి_batch()
        x, y = x.గమ్యం(device), y.గమ్యం(device)
        logits, loss = model(x, y)
        loss.backward()
        # save model params, in bfloat16
        model_గమ్యం_కొలత = {"meta-llama/Meta-Llama-3.1-8B": "8B"}
        model_కొలత_str = model_గమ్యం_కొలత[వాదనలు.model] # e.g. "8B"
        write_model(model, os.మార్గం.join(వాదనలు.output_dir, f"llama3.1_{model_size_str}_bf16.bin"), dtype="bfloat16")
        # save x, y, logits, loss, and parameter gradients, for debugging C
        # always store these in fp32 to have an accurate reference (?)
        write_state(model, x, y, logits, loss, os.మార్గం.join(వాదనలు.output_dir, f"llama3_{model_size_str}_debug_state.bin"))
        # reset the train_loader for the optimization below
        train_loader.reset()

    # -------------------------------------------------------------------------
    # main training loop

    # here we wrap model into DDP container
    ఒకవేళ ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module ఒకవేళ ddp లేకపోతే model # always contains the "raw" unwrapped model

    # init the optimizer
    optimizer = raw_model.configure_optimizers(బరువు_decay=వాదనలు.బరువు_decay,
                                               learning_రేటు=వాదనలు.learning_రేటు, betas=(0.9, 0.95),
                                               device_type=device, zero_stage=zero_stage)

    # learning rate decay scheduler (cosine with warmup)
    నిర్వచనం get_lr(it):
        min_lr = వాదనలు.learning_రేటు * వాదనలు.learning_రేటు_decay_frac
        # 1) linear warmup for warmup_iters steps
        ఒకవేళ it < వాదనలు.warmup_iters:
            తిరిగి వాదనలు.learning_రేటు * (it+1) / వాదనలు.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        ఒకవేళ it > వాదనలు.అంకె_iterations:
            తిరిగి min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_నిష్పత్తి = (it - వాదనలు.warmup_iters) / (వాదనలు.అంకె_iterations - వాదనలు.warmup_iters)
        ప్రతిజ్ఞా 0 <= decay_నిష్పత్తి <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_నిష్పత్తి)) # coeff starts at 1 and goes to 0
        తిరిగి min_lr + coeff * (వాదనలు.learning_రేటు - min_lr)

    ఒకవేళ device == "cuda":
        torch.cuda.reset_peak_జ్ఞాపకం_stats()
    timings = []
    norm = -1.0   # dummy value to print in inference-only mode
    కొరకు అడుగు లో శ్రేణి(వాదనలు.అంకె_iterations + 1):
        t0 = time.time()
        చివరి_అడుగు = (అడుగు == వాదనలు.అంకె_iterations)

        # once in a while evaluate the validation dataset
        ఒకవేళ (వాదనలు.val_loss_every > 0 \
            మరియు (అడుగు % వాదనలు.val_loss_every == 0 లేదా చివరి_అడుగు)) \
            మరియు (val_loader అయిన కాదు ఏమీలేదు):
            model.eval()
            val_loader.reset()
            తోడుగా torch.no_grad():
                val_loss = 0.0
                కొరకు _ లో శ్రేణి(వాదనలు.val_max_అడుగులు):
                    x, y = val_loader.తదుపరి_batch()
                    x, y = x.గమ్యం(device), y.గమ్యం(device)
                    _, loss = model(x, y, return_logits=అబద్ధం)
                    val_loss += loss.అంశం()
                val_loss /= వాదనలు.val_max_అడుగులు
            # log to console and to file
            print0(f"val loss {val_loss}")
            ఒకవేళ master_process మరియు logfile అయిన కాదు ఏమీలేదు:
                తోడుగా open(logfile, "a") ఇవ f:
                    f.లిఖ("s:%d tel:%f\n" % (అడుగు, val_loss))

        # once in a while perform model inference on the master process
        ఒకవేళ (వాదనలు.sample_every > 0 \
            మరియు (అడుగు % వాదనలు.sample_every == 0 లేదా చివరి_అడుగు)) \
            మరియు master_process:
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
            ఒకవేళ వాదనలు.use_hf:
                prompt_టోకెన్లు = [model.tokenizer(x).ఆదానం_ids కొరకు x లో prompts]
            లేకపోతే:  # Meta
                prompt_టోకెన్లు = [model.tokenizer.encode(x, bos=నిజం, eos=అబద్ధం) కొరకు x లో prompts]

            generation_టోకెన్లు = model.generate(prompt_టోకెన్లు, max_gen_len=64, temperature=0.6, శిఖరం_p=0.9, echo=అబద్ధం)
            results = [{"generation": model.tokenizer.decode(t)} కొరకు t లో generation_టోకెన్లు]
            కొరకు prompt, ఫలితం లో zip(prompts, results):
                ముద్రించు(prompt, చివర="")
                ముద్రించు(f"{result['generation']}")
                ముద్రించు("\n==================================\n")

        # bit confusing: we want to make sure to eval and sample on 0th iteration
        # but also after the very last iteration. so we loop for step <= num_iterations
        # instead of just < num_iterations (one extra due to <=), only to do
        # the validation/sampling one last time, and then we break right here as we're done.
        ఒకవేళ చివరి_అడుగు:
            ఆపు

        # --------------- TRAINING SECTION BEGIN -----------------
        model.train()
        optimizer.zero_grad(సమితి_గమ్యం_none=నిజం)
        # if we are trying to overfit a single batch, we reset the loader here
        ఒకవేళ వాదనలు.overfit_single_batch:
            train_loader.reset()
        # micro-batch loop where we do gradient accumulation to reach desired total batch size
        lossf = 0.0 # for getting the mean loss (as simple float) over the accumulation steps
        కొరకు micro_అడుగు లో శ్రేణి(grad_accum_అడుగులు):
            # fetch a batch
            x, y = train_loader.తదుపరి_batch()
            x, y = x.గమ్యం(device), y.గమ్యం(device)
            ఒకవేళ ddp:
                # we want only the last micro-step to sync grads in a DDP model
                # the official way to do this is with model.no_sync(), but that is a
                # context manager that bloats the code, so we just toggle this variable
                model.require_backward_grad_sync = (micro_అడుగు == grad_accum_అడుగులు - 1)
            # forward pass
            తోడుగా ctx:
                _, loss = model(x, y, return_logits=అబద్ధం)
                # we have to scale the loss to account for gradient accumulation,
                # because the gradients just add on each successive backward().
                # addition of gradients corresponds to a SUM in the objective, but
                # instead of a SUM we want MEAN, so we scale the loss here
                loss = loss / grad_accum_అడుగులు
                lossf += loss.detach() # keep track of the mean loss
            # backward pass
            ఒకవేళ కాదు వాదనలు.inference_only:
                loss.backward()
        ఒకవేళ ddp:
            dist.all_reduce(lossf, op=dist.ReduceOp.AVG)
        lossf = lossf.అంశం()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), వాదనలు.grad_clip)
        # determine and set the learning rate for this iteration
        lr = get_lr(అడుగు)
        కొరకు param_సమూహం లో optimizer.param_groups:
            param_సమూహం['lr'] = lr
        # step the optimizer
        optimizer.అడుగు()
        # --------------- TRAINING SECTION END -------------------
        # everything that follows now is just diagnostics, prints, logging, etc.

        # wait on the CPU for all device work to end so we get accurate per-iteration timings below
        ఒకవేళ device == "mps":
            torch.mps.synchronize()
        లేదావేళ device == "cuda":
            torch.cuda.synchronize()
        # time and print
        t1 = time.time()
        # the 0th iteration is often an outlier (much slower) => skip logging it
        టోకెన్లు_చొప్పున_రెండవ = grad_accum_అడుగులు * ddp_world_కొలత * B * T / (t1-t0)
        print0(f"step {step+1:4d}/{args.num_iterations} | train loss {lossf:.6f} | norm {norm:.4f} | lr {lr:.2e} | ({(t1-t0)*1000:.2f} ms | {tokens_per_second:.0f} tok/s)")
        # log to logile
        ఒకవేళ master_process మరియు logfile అయిన కాదు ఏమీలేదు:
            తోడుగా open(logfile, "a") ఇవ f:
                f.లిఖ("s:%d trl:%f\n" % (అడుగు, lossf))

        # keep track of smooth timings, last 20 iterations
        ఒకవేళ అడుగు > 0 మరియు అడుగు > వాదనలు.అంకె_iterations - 20:
            timings.append(t1-t0)

    # print the average of the last 20 timings, to get something smooth-ish
    timings = timings[-20:]
    print0(f"final {len(timings)} iters avg: {np.mean(timings)*1000:.3f}ms")
    print0(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

    # -------------------------------------------------------------------------
    # clean up nice
    ఒకవేళ ddp:
        destroy_process_సమూహం()
