"""
Reference code for GPT-2 training and inference.
Will save the model weights into files, to be read from C as initialization.

References:
1) the official GPT-2 TensorFlow implementation released by OpenAI:
https://github.com/openai/gpt-2/blob/master/src/model.py
2) huggingface/transformers PyTorch implementation:
https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py

Example launches to only benchmark the speed of bfloat16 compiled GPU training:
1 GPU:
python train_gpt2.py --write_tensors=0 --num_iterations=50 --sequence_length=1024 --compile=1 --tensorcores=1 --dtype=bfloat16
you can also turn on flash-attention by appending --flash=1
4 GPU:
torchrun --standalone --nproc_per_node=4 train_gpt2.py --write_tensors=0 --num_iterations=50 --sequence_length=1024 --compile=1 --tensorcores=1 --dtype=bfloat16
"""

ਆਯਾਤ os
ਆਯਾਤ math
ਆਯਾਤ glob
ਆਯਾਤ struct
ਆਯਾਤ inspect
ਤਃ contextlib ਆਯਾਤ nullcontext
ਤਃ dataclasses ਆਯਾਤ dataclass

ਆਯਾਤ numpy ਇਵ np
ਆਯਾਤ torch
ਆਯਾਤ torch.nn ਇਵ nn
ਤਃ torch.nn ਆਯਾਤ functional ਇਵ F
ਆਯਾਤ torch._inductor.config ਇਵ config
ਤਃ torch.nn.parallel ਆਯਾਤ Distributed_ਅੰਕੜੇ_Parallel ਇਵ DDP
ਤਃ torch.distributed ਆਯਾਤ init_process_ਸਮੂਹ, destroy_process_ਸਮੂਹ
ਤਃ torch.distributed.optim ਆਯਾਤ ZeroRedundancyOptimizer
ਆਯਾਤ torch.distributed ਇਵ dist

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model

ਵਰਗ NewGELU(nn.Module):
    """Careful there are a few versions of GeLU, this one is the exact one used by OpenAI"""
    ਪਰਿਭਾਸ਼ਾ forward(self, ਨਿਵੇਸ਼):
        ਵਾਪਸ 0.5 * ਨਿਵੇਸ਼ * (1.0 + torch.tanh(math.ਵਰਗਮੂਲ(2.0 / math.pi) * (ਨਿਵੇਸ਼ + 0.044715 * torch.pow(ਨਿਵੇਸ਼, 3.0))))

# using a global to toggle flash-attention
FLASH = 0

ਵਰਗ CausalSelfAttention(nn.Module):

    ਪਰਿਭਾਸ਼ਾ __init__(self, config):
        ਪਰਮ().__init__()
        ਪ੍ਰਤਿਜ੍ਞਾ config.n_embd % config.n_ਸਿਰ == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.ਰੇਖਿਕ(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.ਰੇਖਿਕ(config.n_embd, config.n_embd)
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1
        # regularization
        self.n_ਸਿਰ = config.n_ਸਿਰ
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.ਰਜਿਸਟਰ_ਬਫਰ("bias", torch.tril(torch.ones(config.ਖੰਡ_ਮਾਪ, config.ਖੰਡ_ਮਾਪ))
                                     .view(1, 1, config.ਖੰਡ_ਮਾਪ, config.ਖੰਡ_ਮਾਪ))

    ਪਰਿਭਾਸ਼ਾ forward(self, x):
        B, T, C = x.ਮਾਪ() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.ਵੰਡੋ(self.n_embd, dim=2)
        k = k.view(B, T, self.n_ਸਿਰ, C // self.n_ਸਿਰ).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_ਸਿਰ, C // self.n_ਸਿਰ).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_ਸਿਰ, C // self.n_ਸਿਰ).transpose(1, 2) # (B, nh, T, hs)
        ਜੇ FLASH:
            # flashattention
            y = F.scaled_dot_ਗੁਣਨਫਲ_attention(q, k, v, is_causal=ਸੱਚ)
        ਨਹੀਂਤਾਂ:
            # manual implementation of attention
            # this materializes the large (T,T) matrix for all the queries and keys
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.ਵਰਗਮੂਲ(k.ਮਾਪ(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, ਦਸ਼ਮਲਵ('-inf'))
            att = F.softmax(att, dim=-1)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        ਵਾਪਸ y

ਵਰਗ MLP(nn.Module):

    ਪਰਿਭਾਸ਼ਾ __init__(self, config):
        ਪਰਮ().__init__()
        self.c_fc    = nn.ਰੇਖਿਕ(config.n_embd, 4 * config.n_embd)
        self.gelu    = NewGELU()
        self.c_proj  = nn.ਰੇਖਿਕ(4 * config.n_embd, config.n_embd)
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1

    ਪਰਿਭਾਸ਼ਾ forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        ਵਾਪਸ x

ਵਰਗ ਖੰਡ(nn.Module):

    ਪਰਿਭਾਸ਼ਾ __init__(self, config):
        ਪਰਮ().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    ਪਰਿਭਾਸ਼ਾ forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        ਵਾਪਸ x

# -----------------------------------------------------------------------------
# The main GPT-2 model

@dataclass
ਵਰਗ GPTConfig:
    ਖੰਡ_ਮਾਪ: ਪੂਰਨ = 1024
    vocab_ਮਾਪ: ਪੂਰਨ = 50257
    n_layer: ਪੂਰਨ = 12
    n_ਸਿਰ: ਪੂਰਨ = 12
    n_embd: ਪੂਰਨ = 768

ਵਰਗ GPT(nn.Module):

    ਪਰਿਭਾਸ਼ਾ __init__(self, config):
        ਪਰਮ().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_ਮਾਪ, config.n_embd),
            wpe = nn.Embedding(config.ਖੰਡ_ਮਾਪ, config.n_embd),
            h = nn.ModuleList([ਖੰਡ(config) ਲਈ _ ਵਿੱਚ ਸੀਮਾ(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_ਸਿਰ = nn.ਰੇਖਿਕ(config.n_embd, config.vocab_ਮਾਪ, bias=ਝੂਠ)
        self.lm_ਸਿਰ.LLMC_SKIP_INIT = 1 # don't init this one, we will tie weights
        self.transformer.wte.ਭਾਰ = self.lm_ਸਿਰ.ਭਾਰ # https://paperswithcode.com/method/weight-tying

        # init all weights, use a torch rng object to be very careful
        self.init_rng = torch.Generator()
        self.init_rng.manual_ਬੀਜ(42)
        self.apply(self._init_weights)

    ਪਰਿਭਾਸ਼ਾ _init_weights(self, module):
        ਜੇ isinstance(module, nn.ਰੇਖਿਕ):
            # apply special scaled init to the residual projections, per GPT-2 paper
            std = 0.02 ਜੇ ਨਹੀਂ hasattr(module, 'LLMC_RESIDUAL_SCALE_FLAG') ਨਹੀਂਤਾਂ 0.02/math.ਵਰਗਮੂਲ(2 * self.config.n_layer)
            # we want to skip initializing lm_head, which shares parameters with wte
            # and wte was already initialized down below during the Embedding init
            ਜੇ ਨਹੀਂ hasattr(module, 'LLMC_SKIP_INIT'):
                torch.nn.init.ਸਾਧਾਰਨ(module.ਭਾਰ, ਔਸਤ=0.0, std=std, generator=self.init_rng)
            ਜੇ module.bias ਹੈ ਨਹੀਂ ਕੁਝ_ਨਹੀਂ:
                torch.nn.init.zeros_(module.bias)
        ਜਾਂਜੇ isinstance(module, nn.Embedding):
            torch.nn.init.ਸਾਧਾਰਨ(module.ਭਾਰ, ਔਸਤ=0.0, std=0.02, generator=self.init_rng)

    ਪਰਿਭਾਸ਼ਾ forward(self, idx, targets=ਕੁਝ_ਨਹੀਂ, return_logits=ਸੱਚ):
        device = idx.device
        b, t = idx.ਮਾਪ()
        ਪ੍ਰਤਿਜ੍ਞਾ t <= self.config.ਖੰਡ_ਮਾਪ, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.ਲੰਮਾ, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = tok_emb + pos_emb

        ਲਈ ਖੰਡ_2 ਵਿੱਚ self.transformer.h:
            x = ਖੰਡ_2(x)
        x = self.transformer.ln_f(x)

        ਜੇ targets ਹੈ ਨਹੀਂ ਕੁਝ_ਨਹੀਂ:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_ਸਿਰ(x)
            loss = F.cross_entropy(logits.view(-1, logits.ਮਾਪ(-1)), targets.view(-1), ignore_ਸੂਚਕਾਂਕ=-1)
        ਨਹੀਂਤਾਂ:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_ਸਿਰ(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = ਕੁਝ_ਨਹੀਂ

        # there are performance reasons why not returning logits is prudent, if not needed
        ਜੇ ਨਹੀਂ return_logits:
            logits = ਕੁਝ_ਨਹੀਂ

        ਵਾਪਸ logits, loss

    @classmethod
    ਪਰਿਭਾਸ਼ਾ from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        ਪ੍ਰਤਿਜ੍ਞਾ model_type ਵਿੱਚ {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        ਤਃ transformers ਆਯਾਤ GPT2LMHeadModel
        ਛਾਪੋ("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_ਦਲੀਲਾਂ = {
            'gpt2':         dict(n_layer=12, n_ਸਿਰ=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_ਸਿਰ=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_ਸਿਰ=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_ਸਿਰ=25, n_embd=1600), # 1558M params
        }[model_type]
        config_ਦਲੀਲਾਂ['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_ਦਲੀਲਾਂ['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_ਦਲੀਲਾਂ)
        model = GPT(config)
        sd = model.state_dict()
        sd_ਕੁੰਜੀਆਂ = sd.ਕੁੰਜੀਆਂ()
        sd_ਕੁੰਜੀਆਂ = [k ਲਈ k ਵਿੱਚ sd_ਕੁੰਜੀਆਂ ਜੇ ਨਹੀਂ k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_ਕੁੰਜੀਆਂ_hf = sd_hf.ਕੁੰਜੀਆਂ()
        sd_ਕੁੰਜੀਆਂ_hf = [k ਲਈ k ਵਿੱਚ sd_ਕੁੰਜੀਆਂ_hf ਜੇ ਨਹੀਂ k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_ਕੁੰਜੀਆਂ_hf = [k ਲਈ k ਵਿੱਚ sd_ਕੁੰਜੀਆਂ_hf ਜੇ ਨਹੀਂ k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        ਪ੍ਰਤਿਜ੍ਞਾ ਲੰਬਾਈ(sd_ਕੁੰਜੀਆਂ_hf) == ਲੰਬਾਈ(sd_ਕੁੰਜੀਆਂ), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        ਲਈ k ਵਿੱਚ sd_ਕੁੰਜੀਆਂ_hf:
            ਜੇ any(k.endswith(w) ਲਈ w ਵਿੱਚ transposed):
                # special treatment for the Conv1D weights we need to transpose
                ਪ੍ਰਤਿਜ੍ਞਾ sd_hf[k].shape[::-1] == sd[k].shape
                ਨਾਲ torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            ਨਹੀਂਤਾਂ:
                # vanilla copy over the other parameters
                ਪ੍ਰਤਿਜ੍ਞਾ sd_hf[k].shape == sd[k].shape
                ਨਾਲ torch.no_grad():
                    sd[k].copy_(sd_hf[k])

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

    @torch.no_grad()
    ਪਰਿਭਾਸ਼ਾ generate(self, idx, max_new_ਟੋਕਨ, temperature=1.0, ਸਿਖਰ_k=ਕੁਝ_ਨਹੀਂ):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        ਲਈ _ ਵਿੱਚ ਸੀਮਾ(max_new_ਟੋਕਨ):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx ਜੇ idx.ਮਾਪ(1) <= self.config.ਖੰਡ_ਮਾਪ ਨਹੀਂਤਾਂ idx[:, -self.config.ਖੰਡ_ਮਾਪ:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            ਜੇ ਸਿਖਰ_k ਹੈ ਨਹੀਂ ਕੁਝ_ਨਹੀਂ:
                v, _ = torch.topk(logits, min(ਸਿਖਰ_k, logits.ਮਾਪ(-1)))
                logits[logits < v[:, [-1]]] = -ਦਸ਼ਮਲਵ('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_ਅਗਲਾ = torch.multinomial(probs, ਅੰਕ_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_ਅਗਲਾ), dim=1)

        ਵਾਪਸ idx

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

ਪਰਿਭਾਸ਼ਾ peek_ਅੰਕੜੇ_shard(filename):
    # only reads the header, returns header data
    ਨਾਲ open(filename, "rb") ਇਵ f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        ਸਿਰਲੇਖ = np.frombuffer(f.ਪਠ(256*4), dtype=np.int32)
    ਜੇ ਸਿਰਲੇਖ[0] != 20240520:
        ਛਾਪੋ("ERROR: magic number mismatch in the data .bin file!")
        ਛਾਪੋ("---> HINT: Are you passing in a correct file with --input_bin?")
        ਛਾਪੋ("---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README")
        ਛਾਪੋ("---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try")
        exit(1)
    ਪ੍ਰਤਿਜ੍ਞਾ ਸਿਰਲੇਖ[1] == 1, "unsupported version"
    ntok = ਸਿਰਲੇਖ[2] # number of tokens (claimed)
    ਵਾਪਸ ntok # for now just return the number of tokens

ਪਰਿਭਾਸ਼ਾ ਲੱਦੋ_ਅੰਕੜੇ_shard(filename):
    ਨਾਲ open(filename, "rb") ਇਵ f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        ਸਿਰਲੇਖ = np.frombuffer(f.ਪਠ(256*4), dtype=np.int32)
        ਪ੍ਰਤਿਜ੍ਞਾ ਸਿਰਲੇਖ[0] == 20240520, "magic number mismatch in the data .bin file"
        ਪ੍ਰਤਿਜ੍ਞਾ ਸਿਰਲੇਖ[1] == 1, "unsupported version"
        ntok = ਸਿਰਲੇਖ[2] # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        ਟੋਕਨ = np.frombuffer(f.ਪਠ(), dtype=np.uint16)
    ਪ੍ਰਤਿਜ੍ਞਾ ਲੰਬਾਈ(ਟੋਕਨ) == ntok, "number of tokens read does not match header?"
    ਵਾਪਸ ਟੋਕਨ

ਵਰਗ Distributed_ਅੰਕੜੇ_Loader:
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
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.ਲੰਮਾ)
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
    # writes the GPT-2 model's weights to a binary file
    ਪ੍ਰਤਿਜ੍ਞਾ dtype ਵਿੱਚ {"float32", "bfloat16"}
    write_fun = write_fp32 ਜੇ dtype == "float32" ਨਹੀਂਤਾਂ write_bf16
    write_fun(model_tensors["transformer.wte.weight"], file) # (V, C)
    write_fun(model_tensors["transformer.wpe.weight"], file) # (T, C)
    ਲਈ i ਵਿੱਚ ਸੀਮਾ(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_1.weight"], file)
    ਲਈ i ਵਿੱਚ ਸੀਮਾ(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_1.bias"], file)
    ਲਈ i ਵਿੱਚ ਸੀਮਾ(L): # (L, 3C, C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_attn.weight"], file)
    ਲਈ i ਵਿੱਚ ਸੀਮਾ(L): # (L, 3C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_attn.bias"], file)
    ਲਈ i ਵਿੱਚ ਸੀਮਾ(L): # (L, C, C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_proj.weight"], file)
    ਲਈ i ਵਿੱਚ ਸੀਮਾ(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_proj.bias"], file)
    ਲਈ i ਵਿੱਚ ਸੀਮਾ(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_2.weight"], file)
    ਲਈ i ਵਿੱਚ ਸੀਮਾ(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_2.bias"], file)
    ਲਈ i ਵਿੱਚ ਸੀਮਾ(L): # (L, 4C, C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_fc.weight"], file)
    ਲਈ i ਵਿੱਚ ਸੀਮਾ(L): # (L, 4C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_fc.bias"], file)
    ਲਈ i ਵਿੱਚ ਸੀਮਾ(L): # (L, C, 4C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_proj.weight"], file)
    ਲਈ i ਵਿੱਚ ਸੀਮਾ(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_proj.bias"], file)
    write_fun(model_tensors["transformer.ln_f.weight"], file) # (C, )
    write_fun(model_tensors["transformer.ln_f.bias"], file) # (C, )

@torch.no_grad()
ਪਰਿਭਾਸ਼ਾ pad_vocab(tensor, ਗੁਣਜ=128, ਮੁੱਲ=0):
    """
    The dimension of the vocab size in GPT-2 is 50,257
    which is unfortunately a very unfriendly number for a lot of
    matrix operations on the GPU. So we pad it to the nearest
    friendlier multiple, e.g. 50,304 if multiple=128 when we
    export the weights into C land. This is a NOOP algorithmically
    and is only done to make the tensor operations more efficient.
    """
    ਪ੍ਰਤਿਜ੍ਞਾ tensor.ndim == 2
    V, C = tensor.shape
    ਪ੍ਰਤਿਜ੍ਞਾ V == 50257, "just being defensive here"
    # calculate padded vocab size by rounding up to nearest multiple
    Vp = ((V + ਗੁਣਜ - 1) // ਗੁਣਜ) * ਗੁਣਜ
    # pad the tensor
    pad_ਪੰਕਤੀਆਂ = Vp - V
    padded = tensor ਜੇ pad_ਪੰਕਤੀਆਂ == 0 ਨਹੀਂਤਾਂ F.pad(tensor, (0, 0, 0, pad_ਪੰਕਤੀਆਂ), ਮੁੱਲ=ਮੁੱਲ)
    ਪ੍ਰਤਿਜ੍ਞਾ padded.shape == (Vp, C)
    ਵਾਪਸ padded

ਪਰਿਭਾਸ਼ਾ write_model(model, filename, dtype):
    # everything we need to instantiate the model
    # 1) header is: version int, GPTConfig ints, padding to 1024 bytes
    ਪ੍ਰਤਿਜ੍ਞਾ dtype ਵਿੱਚ {"float32", "bfloat16"} # float16 todo maybe later
    ਸੰਸਕਰਣ = {
        "float32": 3, # 3: all tensors are fp32, padded vocab
        "bfloat16": 5, # 5: all tensors are bf16, padded vocab
    }[dtype]
    ਸਿਰਲੇਖ = torch.zeros(256, dtype=torch.int32)
    ਸਿਰਲੇਖ[0] = 20240326 # magic
    ਸਿਰਲੇਖ[1] = ਸੰਸਕਰਣ # checkpoint version
    ਸਿਰਲੇਖ[2] = model.config.ਖੰਡ_ਮਾਪ
    ਸਿਰਲੇਖ[3] = model.config.vocab_ਮਾਪ
    ਸਿਰਲੇਖ[4] = model.config.n_layer
    ਸਿਰਲੇਖ[5] = model.config.n_ਸਿਰ
    ਸਿਰਲੇਖ[6] = model.config.n_embd
    # 2) the parameters follow the header
    params = {name: param.cpu() ਲਈ name, param ਵਿੱਚ model.named_parameters()}
    # pad the vocab to a multiple of 128 here at export, for efficiency in C
    wte = params["transformer.wte.weight"] # (V, C)
    wte_padded = pad_vocab(wte) # (Vp, C)
    params["transformer.wte.weight"] = wte_padded # (Vp, C)
    ਛਾਪੋ(f"padded vocab size from {wte.size(0)} to {wte_padded.size(0)}")
    ਸਿਰਲੇਖ[7] = wte_padded.ਮਾਪ(0) # padded vocab size store in header
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
    ਸਿਰਲੇਖ[0] = 20240327 # magic
    ਸਿਰਲੇਖ[1] = 2 # run state version = 2 (1 -> 2 for padded vocab changes)
    ਸਿਰਲੇਖ[2] = x.ਮਾਪ(0) # batch size of the batch, B
    ਸਿਰਲੇਖ[3] = x.ਮਾਪ(1) # temporal extent of the batch, T
    grads = {name: param.grad.cpu() ਲਈ name, param ਵਿੱਚ model.named_parameters()}
    # pad the vocab grads here as well, to mirror write_model
    wte_grad = grads["transformer.wte.weight"] # (V, C)
    wte_grad_padded = pad_vocab(wte_grad, ਮੁੱਲ=0) # (Vp, C) # TODO later maybe pad with nan?
    grads["transformer.wte.weight"] = wte_grad_padded # (Vp, C)
    ਛਾਪੋ(f"padded vocab size in reference grads from {wte_grad.size(0)} to {wte_grad_padded.size(0)}")
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

ਪਰਿਭਾਸ਼ਾ write_tokenizer(enc, filename):
    n = enc.max_token_ਮੁੱਲ + 1
    ਸਿਰਲੇਖ = torch.zeros(256, dtype=torch.int32)
    ਸਿਰਲੇਖ[0] = 20240328 # magic
    ਸਿਰਲੇਖ[1] = 2 # tokenizer version = 2 (1 -> 2: includes EOT token)
    ਸਿਰਲੇਖ[2] = n # number of tokens
    ਸਿਰਲੇਖ[3] = enc.eot_token # EOT token
    ਨਾਲ open(filename, "wb") ਇਵ file:
        file.ਲਿਖ(ਸਿਰਲੇਖ.numpy().tobytes())
        ਲਈ i ਵਿੱਚ ਸੀਮਾ(n):
            b = enc.decode_ਬਾਈਟਾਂ([i])
            ਲੰਮਾਈ = ਲੰਬਾਈ(b)
            ਪ੍ਰਤਿਜ੍ਞਾ ਲੰਮਾਈ < 256, f"Token length exceeds 255: {length}"
            file.ਲਿਖ(struct.pack("<B", ਲੰਮਾਈ))  # Write the length as a 1-byte unsigned integer
            file.ਲਿਖ(b)  # Write the actual bytes
    ਛਾਪੋ(f"wrote {filename}")

# -----------------------------------------------------------------------------
# int main

ਪਰਿਭਾਸ਼ਾ print0(*ਦਲੀਲਾਂ, **kwargs):
    # modified print that only prints from the master process
    # if this is not a distributed run, it's just a print
    ਜੇ ਪੂਰਨ(os.environ.get("RANK", 0)) == 0:
        ਛਾਪੋ(*ਦਲੀਲਾਂ, **kwargs)

ਜੇ __name__ == "__main__":
    ਆਯਾਤ time
    ਆਯਾਤ argparse
    ਆਯਾਤ tiktoken
    print0(f"Running pytorch {torch.version.__version__}")

    # default settings will overfit a tiny batch of data
    # and save model weights and debug state to disk on the first iteration
    parser = argparse.ਦਲੀਲ_Parser()
    # file system input / output
    parser.ਜੋੜੋ_ਦਲੀਲ("--input_bin", type=str, ਮੂਲ="dev/data/tinyshakespeare/tiny_shakespeare_val.bin", help="input .bin to train on")
    parser.ਜੋੜੋ_ਦਲੀਲ("--input_val_bin", type=str, ਮੂਲ="", help="input .bin to eval validation loss on")
    parser.ਜੋੜੋ_ਦਲੀਲ("--output_dir", type=str, ਮੂਲ="", help="output directory to which to write logs and checkpoints")
    parser.ਜੋੜੋ_ਦਲੀਲ("--model", type=str, ਮੂਲ="gpt2", help="gpt2|gpt2-medium|gpt2-large|gpt2-xl|d12|d24|d36|d48")
    # token layout for each step of the optimization
    parser.ਜੋੜੋ_ਦਲੀਲ("--batch_size", type=ਪੂਰਨ, ਮੂਲ=4, help="batch size, in units of #batch dimensions")
    parser.ਜੋੜੋ_ਦਲੀਲ("--sequence_length", type=ਪੂਰਨ, ਮੂਲ=64, help="sequence length")
    parser.ਜੋੜੋ_ਦਲੀਲ("--total_batch_size", type=ਪੂਰਨ, ਮੂਲ=256, help="total desired batch size, in units of #tokens")
    # workload (number of steps)
    parser.ਜੋੜੋ_ਦਲੀਲ("--num_iterations", type=ਪੂਰਨ, ਮੂਲ=10, help="number of iterations to run")
    parser.ਜੋੜੋ_ਦਲੀਲ("--inference_only", type=ਪੂਰਨ, ਮੂਲ=0, help="only run inference")
    # optimization
    parser.ਜੋੜੋ_ਦਲੀਲ("--learning_rate", type=ਦਸ਼ਮਲਵ, ਮੂਲ=1e-4, help="learning rate warmup iterations")
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
    parser.ਜੋੜੋ_ਦਲੀਲ("--flash", type=ਪੂਰਨ, ਮੂਲ=0, help="use flash attention")
    parser.ਜੋੜੋ_ਦਲੀਲ("--dtype", type=str, ਮੂਲ="float32", help="float32|float16|bfloat16")
    parser.ਜੋੜੋ_ਦਲੀਲ("--zero_stage", type=ਪੂਰਨ, ਮੂਲ=0, help="zero redundancy optimizer stage (0/1/2/3)")
    # python -> C bridge
    parser.ਜੋੜੋ_ਦਲੀਲ("--write_tensors", type=ਪੂਰਨ, ਮੂਲ=1, help="write tensors to disk")
    ਦਲੀਲਾਂ = parser.parse_ਦਲੀਲਾਂ()

    # args error checking and convenience variables
    B, T = ਦਲੀਲਾਂ.batch_ਮਾਪ, ਦਲੀਲਾਂ.sequence_ਲੰਮਾਈ
    ਪ੍ਰਤਿਜ੍ਞਾ 1 <= T <= 1024
    ਪ੍ਰਤਿਜ੍ਞਾ ਦਲੀਲਾਂ.dtype ਵਿੱਚ {"float32", "float16", "bfloat16"}
    ਪ੍ਰਤਿਜ੍ਞਾ ਦਲੀਲਾਂ.model ਵਿੱਚ {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "d12", "d24", "d36", "d48"}

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
    ਛਾਪੋ(f"using device: {device}")
    device_type = 'cuda' ਜੇ 'cuda' ਵਿੱਚ device ਨਹੀਂਤਾਂ 'cpu'

    # calculate gradient accumulation from the desired total batch size and the current run configuration
    ਟੋਕਨ_ਪ੍ਰਤੀ_fwdbwd = B * T * ddp_world_ਮਾਪ
    ਪ੍ਰਤਿਜ੍ਞਾ ਦਲੀਲਾਂ.ਕੁੱਲ_batch_ਮਾਪ % ਟੋਕਨ_ਪ੍ਰਤੀ_fwdbwd == 0
    grad_accum_ਕਦਮਾਂ = ਦਲੀਲਾਂ.ਕੁੱਲ_batch_ਮਾਪ // ਟੋਕਨ_ਪ੍ਰਤੀ_fwdbwd
    print0(f"total desired batch size: {args.total_batch_size}")
    print0(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    # set up a context manager following the desired dtype and device
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[ਦਲੀਲਾਂ.dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) ਜੇ device_type == "cuda" ਨਹੀਂਤਾਂ nullcontext()

    # rng / reproducibility
    torch.manual_ਬੀਜ(42)
    ਜੇ torch.cuda.is_ਉਪਲਬਧ():
        torch.cuda.manual_ਬੀਜ(42)

    # set the torch precision mode to use TensorFloat32 (TF32) for matmuls
    # docs https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    ਜੇ ਦਲੀਲਾਂ.tensorcores:
        torch.ਸੈੱਟ_float32_matmul_precision('high')

    # turn on/off flash attention
    ਪ੍ਰਤਿਜ੍ਞਾ ਦਲੀਲਾਂ.flash ਵਿੱਚ {0, 1}
    FLASH = ਦਲੀਲਾਂ.flash

    # init (and write) the tokenizer
    enc = tiktoken.get_encoding("gpt2")
    ਜੇ master_process ਅਤੇ ਦਲੀਲਾਂ.write_tensors: # tokenizer is technically not tensors but ok
        write_tokenizer(enc, "gpt2_tokenizer.bin")

    # init the model, either from scratch or from OpenAI pretrained checkpoint
    ਜੇ ਦਲੀਲਾਂ.model[0] == "d":
        # from scratch (random weights)
        model_config = {
            "d12": GPTConfig(ਖੰਡ_ਮਾਪ=1024, vocab_ਮਾਪ=50257, n_layer=12, n_ਸਿਰ=12, n_embd=768),
            "d24": GPTConfig(ਖੰਡ_ਮਾਪ=1024, vocab_ਮਾਪ=50257, n_layer=24, n_ਸਿਰ=16, n_embd=1024),
            "d36": GPTConfig(ਖੰਡ_ਮਾਪ=1024, vocab_ਮਾਪ=50257, n_layer=36, n_ਸਿਰ=20, n_embd=1280),
            "d48": GPTConfig(ਖੰਡ_ਮਾਪ=1024, vocab_ਮਾਪ=50257, n_layer=48, n_ਸਿਰ=25, n_embd=1600),
        }[ਦਲੀਲਾਂ.model]
        model = GPT(model_config)
    ਨਹੀਂਤਾਂ:
        # load the GPT-2 model weights
        model = GPT.from_pretrained(ਦਲੀਲਾਂ.model)
    model.train()
    model.ਮੰਜ਼ਿਲ(device)
    ਜੇ ਦਲੀਲਾਂ.compile:
        ਜੇ hasattr(config, "coordinate_descent_tuning"):
            config.coordinate_descent_tuning = ਸੱਚ # suggested by @Chillee
        print0("compiling the model...")
        model = torch.compile(model)

    # -------------------------------------------------------------------------
    # Our own version of a simple DistributedDataLoader

    # load tokens
    train_loader = Distributed_ਅੰਕੜੇ_Loader(ਦਲੀਲਾਂ.ਨਿਵੇਸ਼_bin, B, T, ddp_rank, ddp_world_ਮਾਪ)
    val_loader = ਕੁਝ_ਨਹੀਂ
    ਜੇ ਦਲੀਲਾਂ.ਨਿਵੇਸ਼_val_bin:
        val_loader = Distributed_ਅੰਕੜੇ_Loader(ਦਲੀਲਾਂ.ਨਿਵੇਸ਼_val_bin, B, T, ddp_rank, ddp_world_ਮਾਪ)

    # -------------------------------------------------------------------------
    # PyTorch -> C bridge: save some weights and state for C to load later as reference

    # do one forward pass to generate ground truth for our C tests
    ਜੇ master_process ਅਤੇ ਦਲੀਲਾਂ.write_tensors ਅਤੇ (ਨਹੀਂ ਦਲੀਲਾਂ.inference_only):
        x, y = train_loader.ਅਗਲਾ_batch()
        x, y = x.ਮੰਜ਼ਿਲ(device), y.ਮੰਜ਼ਿਲ(device)
        logits, loss = model(x, y)
        loss.backward()
        # save model params, in both float32 and bfloat16
        model_ਮੰਜ਼ਿਲ_ਮਾਪ = {"gpt2": "124M", "gpt2-medium": "355M", "gpt2-large": "774M", "gpt2-xl": "1558M"}
        model_ਮੰਜ਼ਿਲ_ਮਾਪ.update({f"d{d}": f"d{d}" ਲਈ d ਵਿੱਚ [12, 24, 36, 48]})
        model_ਮਾਪ_str = model_ਮੰਜ਼ਿਲ_ਮਾਪ[ਦਲੀਲਾਂ.model] # e.g. "124M", or "d12"
        write_model(model, f"gpt2_{model_size_str}.bin", dtype="float32")
        write_model(model, f"gpt2_{model_size_str}_bf16.bin", dtype="bfloat16")
        # save x, y, logits, loss, and parameter gradients, for debugging C
        # always store these in fp32 to have an accurate reference (?)
        write_state(model, x, y, logits, loss, f"gpt2_{model_size_str}_debug_state.bin")
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

    # create the logging directory if it does not exist
    logfile = ਕੁਝ_ਨਹੀਂ
    ਜੇ ਦਲੀਲਾਂ.output_dir:
        os.makedirs(ਦਲੀਲਾਂ.output_dir, exist_ok=ਸੱਚ)
        logfile = os.ਰਸਤਾ.join(ਦਲੀਲਾਂ.output_dir, "main.log")
        # create the log file "main.log" inside it, and wipe it clean
        ਨਾਲ open(logfile, "w") ਇਵ f:
            ਲੰਘੋ

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
            # before we end, let's also do one round of inference
            # we'll kick off the generation with "<|endoftext|>", which designates the start of a new sequence
            ਸ਼ੁਰੂ_ids = [enc.eot_token]
            xg = (torch.tensor(ਸ਼ੁਰੂ_ids, dtype=torch.ਲੰਮਾ, device=device)[ਕੁਝ_ਨਹੀਂ, ...])
            max_new_ਟੋਕਨ = 32
            temperature = 1.0
            ਸਿਖਰ_k = 40
            yg = raw_model.generate(xg, max_new_ਟੋਕਨ, temperature=temperature, ਸਿਖਰ_k=ਸਿਖਰ_k)
            print0('---------------')
            print0(enc.decode(yg[0].tolist()))
            print0('---------------')

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
