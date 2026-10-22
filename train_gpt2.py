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

ಆಮದು os
ಆಮದು math
ಆಮದು glob
ಆಮದು struct
ಆಮದು inspect
ತಃ contextlib ಆಮದು nullcontext
ತಃ dataclasses ಆಮದು dataclass

ಆಮದು numpy ಇವ np
ಆಮದು torch
ಆಮದು torch.nn ಇವ nn
ತಃ torch.nn ಆಮದು functional ಇವ F
ಆಮದು torch._inductor.config ಇವ config
ತಃ torch.nn.parallel ಆಮದು Distributed_ದತ್ತಾಂಶ_Parallel ಇವ DDP
ತಃ torch.distributed ಆಮದು init_process_ಗುಂಪು, destroy_process_ಗುಂಪು
ತಃ torch.distributed.optim ಆಮದು ZeroRedundancyOptimizer
ಆಮದು torch.distributed ಇವ dist

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model

ವರ್ಗ NewGELU(nn.Module):
    """Careful there are a few versions of GeLU, this one is the exact one used by OpenAI"""
    ವ್ಯಾಖ್ಯಾನ forward(self, ಒಳಹರಿವು):
        ಹಿಂತಿರುಗಿ 0.5 * ಒಳಹರಿವು * (1.0 + torch.tanh(math.ವರ್ಗಮೂಲ(2.0 / math.pi) * (ಒಳಹರಿವು + 0.044715 * torch.pow(ಒಳಹರಿವು, 3.0))))

# using a global to toggle flash-attention
FLASH = 0

ವರ್ಗ CausalSelfAttention(nn.Module):

    ವ್ಯಾಖ್ಯಾನ __init__(self, config):
        ಪರ().__init__()
        ಪ್ರತಿಜ್ಞಾ config.n_embd % config.n_ತಲೆ == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.ರೇಖೀಯ(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.ರೇಖೀಯ(config.n_embd, config.n_embd)
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1
        # regularization
        self.n_ತಲೆ = config.n_ತಲೆ
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.ರಿಜಿಸ್ಟರ್_ಬಫರ್("bias", torch.tril(torch.ones(config.ಖಂಡ_ಗಾತ್ರ, config.ಖಂಡ_ಗಾತ್ರ))
                                     .view(1, 1, config.ಖಂಡ_ಗಾತ್ರ, config.ಖಂಡ_ಗಾತ್ರ))

    ವ್ಯಾಖ್ಯಾನ forward(self, x):
        B, T, C = x.ಗಾತ್ರ_2() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.ವಿಭಜಿಸು(self.n_embd, dim=2)
        k = k.view(B, T, self.n_ತಲೆ, C // self.n_ತಲೆ).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_ತಲೆ, C // self.n_ತಲೆ).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_ತಲೆ, C // self.n_ತಲೆ).transpose(1, 2) # (B, nh, T, hs)
        ಆದರೆ FLASH:
            # flashattention
            y = F.scaled_dot_ಗುಣಲಬ್ಧ_attention(q, k, v, is_causal=ನಿಜ)
        ಇಲ್ಲದಿದ್ದರೆ:
            # manual implementation of attention
            # this materializes the large (T,T) matrix for all the queries and keys
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.ವರ್ಗಮೂಲ(k.ಗಾತ್ರ_2(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, ದಶಮಾಂಶ('-inf'))
            att = F.softmax(att, dim=-1)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        ಹಿಂತಿರುಗಿ y

ವರ್ಗ MLP(nn.Module):

    ವ್ಯಾಖ್ಯಾನ __init__(self, config):
        ಪರ().__init__()
        self.c_fc    = nn.ರೇಖೀಯ(config.n_embd, 4 * config.n_embd)
        self.gelu    = NewGELU()
        self.c_proj  = nn.ರೇಖೀಯ(4 * config.n_embd, config.n_embd)
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1

    ವ್ಯಾಖ್ಯಾನ forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        ಹಿಂತಿರುಗಿ x

ವರ್ಗ ಖಂಡ(nn.Module):

    ವ್ಯಾಖ್ಯಾನ __init__(self, config):
        ಪರ().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    ವ್ಯಾಖ್ಯಾನ forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        ಹಿಂತಿರುಗಿ x

# -----------------------------------------------------------------------------
# The main GPT-2 model

@dataclass
ವರ್ಗ GPTConfig:
    ಖಂಡ_ಗಾತ್ರ: ಪೂರ್ಣಾಂಕ = 1024
    vocab_ಗಾತ್ರ: ಪೂರ್ಣಾಂಕ = 50257
    n_layer: ಪೂರ್ಣಾಂಕ = 12
    n_ತಲೆ: ಪೂರ್ಣಾಂಕ = 12
    n_embd: ಪೂರ್ಣಾಂಕ = 768

ವರ್ಗ GPT(nn.Module):

    ವ್ಯಾಖ್ಯಾನ __init__(self, config):
        ಪರ().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_ಗಾತ್ರ, config.n_embd),
            wpe = nn.Embedding(config.ಖಂಡ_ಗಾತ್ರ, config.n_embd),
            h = nn.ModuleList([ಖಂಡ(config) ಗಾಗಿ _ ಒಳಗೆ ಶ್ರೇಣಿ(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_ತಲೆ = nn.ರೇಖೀಯ(config.n_embd, config.vocab_ಗಾತ್ರ, bias=ಸುಳ್ಳು)
        self.lm_ತಲೆ.LLMC_SKIP_INIT = 1 # don't init this one, we will tie weights
        self.transformer.wte.ತೂಕ = self.lm_ತಲೆ.ತೂಕ # https://paperswithcode.com/method/weight-tying

        # init all weights, use a torch rng object to be very careful
        self.init_rng = torch.Generator()
        self.init_rng.manual_ಬೀಜ(42)
        self.apply(self._init_weights)

    ವ್ಯಾಖ್ಯಾನ _init_weights(self, module):
        ಆದರೆ isinstance(module, nn.ರೇಖೀಯ):
            # apply special scaled init to the residual projections, per GPT-2 paper
            std = 0.02 ಆದರೆ ಅಲ್ಲ hasattr(module, 'LLMC_RESIDUAL_SCALE_FLAG') ಇಲ್ಲದಿದ್ದರೆ 0.02/math.ವರ್ಗಮೂಲ(2 * self.config.n_layer)
            # we want to skip initializing lm_head, which shares parameters with wte
            # and wte was already initialized down below during the Embedding init
            ಆದರೆ ಅಲ್ಲ hasattr(module, 'LLMC_SKIP_INIT'):
                torch.nn.init.ಸಾಮಾನ್ಯ(module.ತೂಕ, ಸರಾಸರಿ=0.0, std=std, generator=self.init_rng)
            ಆದರೆ module.bias ಆಗಿರುವ ಅಲ್ಲ ಏನೂಇಲ್ಲ:
                torch.nn.init.zeros_(module.bias)
        ಅಥವಾಆದರೆ isinstance(module, nn.Embedding):
            torch.nn.init.ಸಾಮಾನ್ಯ(module.ತೂಕ, ಸರಾಸರಿ=0.0, std=0.02, generator=self.init_rng)

    ವ್ಯಾಖ್ಯಾನ forward(self, idx, targets=ಏನೂಇಲ್ಲ, return_logits=ನಿಜ):
        device = idx.device
        b, t = idx.ಗಾತ್ರ_2()
        ಪ್ರತಿಜ್ಞಾ t <= self.config.ಖಂಡ_ಗಾತ್ರ, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.ದೀರ್ಘ, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = tok_emb + pos_emb

        ಗಾಗಿ ಖಂಡ_2 ಒಳಗೆ self.transformer.h:
            x = ಖಂಡ_2(x)
        x = self.transformer.ln_f(x)

        ಆದರೆ targets ಆಗಿರುವ ಅಲ್ಲ ಏನೂಇಲ್ಲ:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_ತಲೆ(x)
            loss = F.cross_entropy(logits.view(-1, logits.ಗಾತ್ರ_2(-1)), targets.view(-1), ignore_ಸೂಚ್ಯಂಕ=-1)
        ಇಲ್ಲದಿದ್ದರೆ:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_ತಲೆ(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = ಏನೂಇಲ್ಲ

        # there are performance reasons why not returning logits is prudent, if not needed
        ಆದರೆ ಅಲ್ಲ return_logits:
            logits = ಏನೂಇಲ್ಲ

        ಹಿಂತಿರುಗಿ logits, loss

    @classmethod
    ವ್ಯಾಖ್ಯಾನ from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        ಪ್ರತಿಜ್ಞಾ model_type ಒಳಗೆ {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        ತಃ transformers ಆಮದು GPT2LMHeadModel
        ಮುದ್ರಿಸು("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_ವಾದಗಳು = {
            'gpt2':         dict(n_layer=12, n_ತಲೆ=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_ತಲೆ=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_ತಲೆ=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_ತಲೆ=25, n_embd=1600), # 1558M params
        }[model_type]
        config_ವಾದಗಳು['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_ವಾದಗಳು['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_ವಾದಗಳು)
        model = GPT(config)
        sd = model.state_dict()
        sd_ಕೀಲಿಗಳು = sd.ಕೀಲಿಗಳು()
        sd_ಕೀಲಿಗಳು = [k ಗಾಗಿ k ಒಳಗೆ sd_ಕೀಲಿಗಳು ಆದರೆ ಅಲ್ಲ k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_ಕೀಲಿಗಳು_hf = sd_hf.ಕೀಲಿಗಳು()
        sd_ಕೀಲಿಗಳು_hf = [k ಗಾಗಿ k ಒಳಗೆ sd_ಕೀಲಿಗಳು_hf ಆದರೆ ಅಲ್ಲ k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_ಕೀಲಿಗಳು_hf = [k ಗಾಗಿ k ಒಳಗೆ sd_ಕೀಲಿಗಳು_hf ಆದರೆ ಅಲ್ಲ k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        ಪ್ರತಿಜ್ಞಾ ಉದ್ದ(sd_ಕೀಲಿಗಳು_hf) == ಉದ್ದ(sd_ಕೀಲಿಗಳು), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        ಗಾಗಿ k ಒಳಗೆ sd_ಕೀಲಿಗಳು_hf:
            ಆದರೆ any(k.endswith(w) ಗಾಗಿ w ಒಳಗೆ transposed):
                # special treatment for the Conv1D weights we need to transpose
                ಪ್ರತಿಜ್ಞಾ sd_hf[k].shape[::-1] == sd[k].shape
                ಜೊತೆ torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            ಇಲ್ಲದಿದ್ದರೆ:
                # vanilla copy over the other parameters
                ಪ್ರತಿಜ್ಞಾ sd_hf[k].shape == sd[k].shape
                ಜೊತೆ torch.no_grad():
                    sd[k].copy_(sd_hf[k])

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

    @torch.no_grad()
    ವ್ಯಾಖ್ಯಾನ generate(self, idx, max_new_ಟೋಕನುಗಳು, temperature=1.0, ತುದಿ_k=ಏನೂಇಲ್ಲ):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        ಗಾಗಿ _ ಒಳಗೆ ಶ್ರೇಣಿ(max_new_ಟೋಕನುಗಳು):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx ಆದರೆ idx.ಗಾತ್ರ_2(1) <= self.config.ಖಂಡ_ಗಾತ್ರ ಇಲ್ಲದಿದ್ದರೆ idx[:, -self.config.ಖಂಡ_ಗಾತ್ರ:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            ಆದರೆ ತುದಿ_k ಆಗಿರುವ ಅಲ್ಲ ಏನೂಇಲ್ಲ:
                v, _ = torch.topk(logits, min(ತುದಿ_k, logits.ಗಾತ್ರ_2(-1)))
                logits[logits < v[:, [-1]]] = -ದಶಮಾಂಶ('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_ಮುಂದಿನ = torch.multinomial(probs, ಅಂಕೆ_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_ಮುಂದಿನ), dim=1)

        ಹಿಂತಿರುಗಿ idx

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

ವ್ಯಾಖ್ಯಾನ peek_ದತ್ತಾಂಶ_shard(filename):
    # only reads the header, returns header data
    ಜೊತೆ open(filename, "rb") ಇವ f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        ಶೀರ್ಷಿಕೆ = np.frombuffer(f.ಪಠ(256*4), dtype=np.int32)
    ಆದರೆ ಶೀರ್ಷಿಕೆ[0] != 20240520:
        ಮುದ್ರಿಸು("ERROR: magic number mismatch in the data .bin file!")
        ಮುದ್ರಿಸು("---> HINT: Are you passing in a correct file with --input_bin?")
        ಮುದ್ರಿಸು("---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README")
        ಮುದ್ರಿಸು("---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try")
        exit(1)
    ಪ್ರತಿಜ್ಞಾ ಶೀರ್ಷಿಕೆ[1] == 1, "unsupported version"
    ntok = ಶೀರ್ಷಿಕೆ[2] # number of tokens (claimed)
    ಹಿಂತಿರುಗಿ ntok # for now just return the number of tokens

ವ್ಯಾಖ್ಯಾನ ತುಂಬು_ದತ್ತಾಂಶ_shard(filename):
    ಜೊತೆ open(filename, "rb") ಇವ f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        ಶೀರ್ಷಿಕೆ = np.frombuffer(f.ಪಠ(256*4), dtype=np.int32)
        ಪ್ರತಿಜ್ಞಾ ಶೀರ್ಷಿಕೆ[0] == 20240520, "magic number mismatch in the data .bin file"
        ಪ್ರತಿಜ್ಞಾ ಶೀರ್ಷಿಕೆ[1] == 1, "unsupported version"
        ntok = ಶೀರ್ಷಿಕೆ[2] # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        ಟೋಕನುಗಳು = np.frombuffer(f.ಪಠ(), dtype=np.uint16)
    ಪ್ರತಿಜ್ಞಾ ಉದ್ದ(ಟೋಕನುಗಳು) == ntok, "number of tokens read does not match header?"
    ಹಿಂತಿರುಗಿ ಟೋಕನುಗಳು

ವರ್ಗ Distributed_ದತ್ತಾಂಶ_Loader:
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
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.ದೀರ್ಘ)
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
    # writes the GPT-2 model's weights to a binary file
    ಪ್ರತಿಜ್ಞಾ dtype ಒಳಗೆ {"float32", "bfloat16"}
    write_fun = write_fp32 ಆದರೆ dtype == "float32" ಇಲ್ಲದಿದ್ದರೆ write_bf16
    write_fun(model_tensors["transformer.wte.weight"], file) # (V, C)
    write_fun(model_tensors["transformer.wpe.weight"], file) # (T, C)
    ಗಾಗಿ i ಒಳಗೆ ಶ್ರೇಣಿ(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_1.weight"], file)
    ಗಾಗಿ i ಒಳಗೆ ಶ್ರೇಣಿ(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_1.bias"], file)
    ಗಾಗಿ i ಒಳಗೆ ಶ್ರೇಣಿ(L): # (L, 3C, C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_attn.weight"], file)
    ಗಾಗಿ i ಒಳಗೆ ಶ್ರೇಣಿ(L): # (L, 3C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_attn.bias"], file)
    ಗಾಗಿ i ಒಳಗೆ ಶ್ರೇಣಿ(L): # (L, C, C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_proj.weight"], file)
    ಗಾಗಿ i ಒಳಗೆ ಶ್ರೇಣಿ(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_proj.bias"], file)
    ಗಾಗಿ i ಒಳಗೆ ಶ್ರೇಣಿ(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_2.weight"], file)
    ಗಾಗಿ i ಒಳಗೆ ಶ್ರೇಣಿ(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_2.bias"], file)
    ಗಾಗಿ i ಒಳಗೆ ಶ್ರೇಣಿ(L): # (L, 4C, C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_fc.weight"], file)
    ಗಾಗಿ i ಒಳಗೆ ಶ್ರೇಣಿ(L): # (L, 4C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_fc.bias"], file)
    ಗಾಗಿ i ಒಳಗೆ ಶ್ರೇಣಿ(L): # (L, C, 4C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_proj.weight"], file)
    ಗಾಗಿ i ಒಳಗೆ ಶ್ರೇಣಿ(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_proj.bias"], file)
    write_fun(model_tensors["transformer.ln_f.weight"], file) # (C, )
    write_fun(model_tensors["transformer.ln_f.bias"], file) # (C, )

@torch.no_grad()
ವ್ಯಾಖ್ಯಾನ pad_vocab(tensor, ಅಪವರ್ತ್ಯ=128, ಮೌಲ್ಯ=0):
    """
    The dimension of the vocab size in GPT-2 is 50,257
    which is unfortunately a very unfriendly number for a lot of
    matrix operations on the GPU. So we pad it to the nearest
    friendlier multiple, e.g. 50,304 if multiple=128 when we
    export the weights into C land. This is a NOOP algorithmically
    and is only done to make the tensor operations more efficient.
    """
    ಪ್ರತಿಜ್ಞಾ tensor.ndim == 2
    V, C = tensor.shape
    ಪ್ರತಿಜ್ಞಾ V == 50257, "just being defensive here"
    # calculate padded vocab size by rounding up to nearest multiple
    Vp = ((V + ಅಪವರ್ತ್ಯ - 1) // ಅಪವರ್ತ್ಯ) * ಅಪವರ್ತ್ಯ
    # pad the tensor
    pad_ಅಡ್ಡಸಾಲುಗಳು = Vp - V
    padded = tensor ಆದರೆ pad_ಅಡ್ಡಸಾಲುಗಳು == 0 ಇಲ್ಲದಿದ್ದರೆ F.pad(tensor, (0, 0, 0, pad_ಅಡ್ಡಸಾಲುಗಳು), ಮೌಲ್ಯ=ಮೌಲ್ಯ)
    ಪ್ರತಿಜ್ಞಾ padded.shape == (Vp, C)
    ಹಿಂತಿರುಗಿ padded

ವ್ಯಾಖ್ಯಾನ write_model(model, filename, dtype):
    # everything we need to instantiate the model
    # 1) header is: version int, GPTConfig ints, padding to 1024 bytes
    ಪ್ರತಿಜ್ಞಾ dtype ಒಳಗೆ {"float32", "bfloat16"} # float16 todo maybe later
    ಆವೃತ್ತಿ = {
        "float32": 3, # 3: all tensors are fp32, padded vocab
        "bfloat16": 5, # 5: all tensors are bf16, padded vocab
    }[dtype]
    ಶೀರ್ಷಿಕೆ = torch.zeros(256, dtype=torch.int32)
    ಶೀರ್ಷಿಕೆ[0] = 20240326 # magic
    ಶೀರ್ಷಿಕೆ[1] = ಆವೃತ್ತಿ # checkpoint version
    ಶೀರ್ಷಿಕೆ[2] = model.config.ಖಂಡ_ಗಾತ್ರ
    ಶೀರ್ಷಿಕೆ[3] = model.config.vocab_ಗಾತ್ರ
    ಶೀರ್ಷಿಕೆ[4] = model.config.n_layer
    ಶೀರ್ಷಿಕೆ[5] = model.config.n_ತಲೆ
    ಶೀರ್ಷಿಕೆ[6] = model.config.n_embd
    # 2) the parameters follow the header
    params = {name: param.cpu() ಗಾಗಿ name, param ಒಳಗೆ model.named_parameters()}
    # pad the vocab to a multiple of 128 here at export, for efficiency in C
    wte = params["transformer.wte.weight"] # (V, C)
    wte_padded = pad_vocab(wte) # (Vp, C)
    params["transformer.wte.weight"] = wte_padded # (Vp, C)
    ಮುದ್ರಿಸು(f"padded vocab size from {wte.size(0)} to {wte_padded.size(0)}")
    ಶೀರ್ಷಿಕೆ[7] = wte_padded.ಗಾತ್ರ_2(0) # padded vocab size store in header
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
    ಶೀರ್ಷಿಕೆ[0] = 20240327 # magic
    ಶೀರ್ಷಿಕೆ[1] = 2 # run state version = 2 (1 -> 2 for padded vocab changes)
    ಶೀರ್ಷಿಕೆ[2] = x.ಗಾತ್ರ_2(0) # batch size of the batch, B
    ಶೀರ್ಷಿಕೆ[3] = x.ಗಾತ್ರ_2(1) # temporal extent of the batch, T
    grads = {name: param.grad.cpu() ಗಾಗಿ name, param ಒಳಗೆ model.named_parameters()}
    # pad the vocab grads here as well, to mirror write_model
    wte_grad = grads["transformer.wte.weight"] # (V, C)
    wte_grad_padded = pad_vocab(wte_grad, ಮೌಲ್ಯ=0) # (Vp, C) # TODO later maybe pad with nan?
    grads["transformer.wte.weight"] = wte_grad_padded # (Vp, C)
    ಮುದ್ರಿಸು(f"padded vocab size in reference grads from {wte_grad.size(0)} to {wte_grad_padded.size(0)}")
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

ವ್ಯಾಖ್ಯಾನ write_tokenizer(enc, filename):
    n = enc.max_token_ಮೌಲ್ಯ + 1
    ಶೀರ್ಷಿಕೆ = torch.zeros(256, dtype=torch.int32)
    ಶೀರ್ಷಿಕೆ[0] = 20240328 # magic
    ಶೀರ್ಷಿಕೆ[1] = 2 # tokenizer version = 2 (1 -> 2: includes EOT token)
    ಶೀರ್ಷಿಕೆ[2] = n # number of tokens
    ಶೀರ್ಷಿಕೆ[3] = enc.eot_token # EOT token
    ಜೊತೆ open(filename, "wb") ಇವ file:
        file.ಲಿಖ(ಶೀರ್ಷಿಕೆ.numpy().tobytes())
        ಗಾಗಿ i ಒಳಗೆ ಶ್ರೇಣಿ(n):
            b = enc.decode_ಬೈಟುಗಳು([i])
            ಉದ್ದಳತೆ = ಉದ್ದ(b)
            ಪ್ರತಿಜ್ಞಾ ಉದ್ದಳತೆ < 256, f"Token length exceeds 255: {length}"
            file.ಲಿಖ(struct.pack("<B", ಉದ್ದಳತೆ))  # Write the length as a 1-byte unsigned integer
            file.ಲಿಖ(b)  # Write the actual bytes
    ಮುದ್ರಿಸು(f"wrote {filename}")

# -----------------------------------------------------------------------------
# int main

ವ್ಯಾಖ್ಯಾನ print0(*ವಾದಗಳು, **kwargs):
    # modified print that only prints from the master process
    # if this is not a distributed run, it's just a print
    ಆದರೆ ಪೂರ್ಣಾಂಕ(os.environ.get("RANK", 0)) == 0:
        ಮುದ್ರಿಸು(*ವಾದಗಳು, **kwargs)

ಆದರೆ __name__ == "__main__":
    ಆಮದು time
    ಆಮದು argparse
    ಆಮದು tiktoken
    print0(f"Running pytorch {torch.version.__version__}")

    # default settings will overfit a tiny batch of data
    # and save model weights and debug state to disk on the first iteration
    parser = argparse.ವಾದ_Parser()
    # file system input / output
    parser.ಕೂಡಿಸು_ವಾದ("--input_bin", type=str, ಪೂರ್ವನಿಯೋಜಿತ="dev/data/tinyshakespeare/tiny_shakespeare_val.bin", help="input .bin to train on")
    parser.ಕೂಡಿಸು_ವಾದ("--input_val_bin", type=str, ಪೂರ್ವನಿಯೋಜಿತ="", help="input .bin to eval validation loss on")
    parser.ಕೂಡಿಸು_ವಾದ("--output_dir", type=str, ಪೂರ್ವನಿಯೋಜಿತ="", help="output directory to which to write logs and checkpoints")
    parser.ಕೂಡಿಸು_ವಾದ("--model", type=str, ಪೂರ್ವನಿಯೋಜಿತ="gpt2", help="gpt2|gpt2-medium|gpt2-large|gpt2-xl|d12|d24|d36|d48")
    # token layout for each step of the optimization
    parser.ಕೂಡಿಸು_ವಾದ("--batch_size", type=ಪೂರ್ಣಾಂಕ, ಪೂರ್ವನಿಯೋಜಿತ=4, help="batch size, in units of #batch dimensions")
    parser.ಕೂಡಿಸು_ವಾದ("--sequence_length", type=ಪೂರ್ಣಾಂಕ, ಪೂರ್ವನಿಯೋಜಿತ=64, help="sequence length")
    parser.ಕೂಡಿಸು_ವಾದ("--total_batch_size", type=ಪೂರ್ಣಾಂಕ, ಪೂರ್ವನಿಯೋಜಿತ=256, help="total desired batch size, in units of #tokens")
    # workload (number of steps)
    parser.ಕೂಡಿಸು_ವಾದ("--num_iterations", type=ಪೂರ್ಣಾಂಕ, ಪೂರ್ವನಿಯೋಜಿತ=10, help="number of iterations to run")
    parser.ಕೂಡಿಸು_ವಾದ("--inference_only", type=ಪೂರ್ಣಾಂಕ, ಪೂರ್ವನಿಯೋಜಿತ=0, help="only run inference")
    # optimization
    parser.ಕೂಡಿಸು_ವಾದ("--learning_rate", type=ದಶಮಾಂಶ, ಪೂರ್ವನಿಯೋಜಿತ=1e-4, help="learning rate warmup iterations")
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
    parser.ಕೂಡಿಸು_ವಾದ("--flash", type=ಪೂರ್ಣಾಂಕ, ಪೂರ್ವನಿಯೋಜಿತ=0, help="use flash attention")
    parser.ಕೂಡಿಸು_ವಾದ("--dtype", type=str, ಪೂರ್ವನಿಯೋಜಿತ="float32", help="float32|float16|bfloat16")
    parser.ಕೂಡಿಸು_ವಾದ("--zero_stage", type=ಪೂರ್ಣಾಂಕ, ಪೂರ್ವನಿಯೋಜಿತ=0, help="zero redundancy optimizer stage (0/1/2/3)")
    # python -> C bridge
    parser.ಕೂಡಿಸು_ವಾದ("--write_tensors", type=ಪೂರ್ಣಾಂಕ, ಪೂರ್ವನಿಯೋಜಿತ=1, help="write tensors to disk")
    ವಾದಗಳು = parser.parse_ವಾದಗಳು()

    # args error checking and convenience variables
    B, T = ವಾದಗಳು.batch_ಗಾತ್ರ, ವಾದಗಳು.sequence_ಉದ್ದಳತೆ
    ಪ್ರತಿಜ್ಞಾ 1 <= T <= 1024
    ಪ್ರತಿಜ್ಞಾ ವಾದಗಳು.dtype ಒಳಗೆ {"float32", "float16", "bfloat16"}
    ಪ್ರತಿಜ್ಞಾ ವಾದಗಳು.model ಒಳಗೆ {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "d12", "d24", "d36", "d48"}

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
    ಮುದ್ರಿಸು(f"using device: {device}")
    device_type = 'cuda' ಆದರೆ 'cuda' ಒಳಗೆ device ಇಲ್ಲದಿದ್ದರೆ 'cpu'

    # calculate gradient accumulation from the desired total batch size and the current run configuration
    ಟೋಕನುಗಳು_ಪ್ರತಿ_fwdbwd = B * T * ddp_world_ಗಾತ್ರ
    ಪ್ರತಿಜ್ಞಾ ವಾದಗಳು.ಒಟ್ಟು_batch_ಗಾತ್ರ % ಟೋಕನುಗಳು_ಪ್ರತಿ_fwdbwd == 0
    grad_accum_ಹೆಜ್ಜೆಗಳು = ವಾದಗಳು.ಒಟ್ಟು_batch_ಗಾತ್ರ // ಟೋಕನುಗಳು_ಪ್ರತಿ_fwdbwd
    print0(f"total desired batch size: {args.total_batch_size}")
    print0(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    # set up a context manager following the desired dtype and device
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[ವಾದಗಳು.dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) ಆದರೆ device_type == "cuda" ಇಲ್ಲದಿದ್ದರೆ nullcontext()

    # rng / reproducibility
    torch.manual_ಬೀಜ(42)
    ಆದರೆ torch.cuda.is_ಲಭ್ಯ():
        torch.cuda.manual_ಬೀಜ(42)

    # set the torch precision mode to use TensorFloat32 (TF32) for matmuls
    # docs https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    ಆದರೆ ವಾದಗಳು.tensorcores:
        torch.ಗಣ_float32_matmul_precision('high')

    # turn on/off flash attention
    ಪ್ರತಿಜ್ಞಾ ವಾದಗಳು.flash ಒಳಗೆ {0, 1}
    FLASH = ವಾದಗಳು.flash

    # init (and write) the tokenizer
    enc = tiktoken.get_encoding("gpt2")
    ಆದರೆ master_process ಮತ್ತು ವಾದಗಳು.write_tensors: # tokenizer is technically not tensors but ok
        write_tokenizer(enc, "gpt2_tokenizer.bin")

    # init the model, either from scratch or from OpenAI pretrained checkpoint
    ಆದರೆ ವಾದಗಳು.model[0] == "d":
        # from scratch (random weights)
        model_config = {
            "d12": GPTConfig(ಖಂಡ_ಗಾತ್ರ=1024, vocab_ಗಾತ್ರ=50257, n_layer=12, n_ತಲೆ=12, n_embd=768),
            "d24": GPTConfig(ಖಂಡ_ಗಾತ್ರ=1024, vocab_ಗಾತ್ರ=50257, n_layer=24, n_ತಲೆ=16, n_embd=1024),
            "d36": GPTConfig(ಖಂಡ_ಗಾತ್ರ=1024, vocab_ಗಾತ್ರ=50257, n_layer=36, n_ತಲೆ=20, n_embd=1280),
            "d48": GPTConfig(ಖಂಡ_ಗಾತ್ರ=1024, vocab_ಗಾತ್ರ=50257, n_layer=48, n_ತಲೆ=25, n_embd=1600),
        }[ವಾದಗಳು.model]
        model = GPT(model_config)
    ಇಲ್ಲದಿದ್ದರೆ:
        # load the GPT-2 model weights
        model = GPT.from_pretrained(ವಾದಗಳು.model)
    model.train()
    model.ಗಮ್ಯ(device)
    ಆದರೆ ವಾದಗಳು.compile:
        ಆದರೆ hasattr(config, "coordinate_descent_tuning"):
            config.coordinate_descent_tuning = ನಿಜ # suggested by @Chillee
        print0("compiling the model...")
        model = torch.compile(model)

    # -------------------------------------------------------------------------
    # Our own version of a simple DistributedDataLoader

    # load tokens
    train_loader = Distributed_ದತ್ತಾಂಶ_Loader(ವಾದಗಳು.ಒಳಹರಿವು_bin, B, T, ddp_rank, ddp_world_ಗಾತ್ರ)
    val_loader = ಏನೂಇಲ್ಲ
    ಆದರೆ ವಾದಗಳು.ಒಳಹರಿವು_val_bin:
        val_loader = Distributed_ದತ್ತಾಂಶ_Loader(ವಾದಗಳು.ಒಳಹರಿವು_val_bin, B, T, ddp_rank, ddp_world_ಗಾತ್ರ)

    # -------------------------------------------------------------------------
    # PyTorch -> C bridge: save some weights and state for C to load later as reference

    # do one forward pass to generate ground truth for our C tests
    ಆದರೆ master_process ಮತ್ತು ವಾದಗಳು.write_tensors ಮತ್ತು (ಅಲ್ಲ ವಾದಗಳು.inference_only):
        x, y = train_loader.ಮುಂದಿನ_batch()
        x, y = x.ಗಮ್ಯ(device), y.ಗಮ್ಯ(device)
        logits, loss = model(x, y)
        loss.backward()
        # save model params, in both float32 and bfloat16
        model_ಗಮ್ಯ_ಗಾತ್ರ = {"gpt2": "124M", "gpt2-medium": "355M", "gpt2-large": "774M", "gpt2-xl": "1558M"}
        model_ಗಮ್ಯ_ಗಾತ್ರ.update({f"d{d}": f"d{d}" ಗಾಗಿ d ಒಳಗೆ [12, 24, 36, 48]})
        model_ಗಾತ್ರ_str = model_ಗಮ್ಯ_ಗಾತ್ರ[ವಾದಗಳು.model] # e.g. "124M", or "d12"
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

    # create the logging directory if it does not exist
    logfile = ಏನೂಇಲ್ಲ
    ಆದರೆ ವಾದಗಳು.output_dir:
        os.makedirs(ವಾದಗಳು.output_dir, exist_ok=ನಿಜ)
        logfile = os.ಮಾರ್ಗ.join(ವಾದಗಳು.output_dir, "main.log")
        # create the log file "main.log" inside it, and wipe it clean
        ಜೊತೆ open(logfile, "w") ಇವ f:
            ದಾಟು

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
            # before we end, let's also do one round of inference
            # we'll kick off the generation with "<|endoftext|>", which designates the start of a new sequence
            ಆರಂಭ_ids = [enc.eot_token]
            xg = (torch.tensor(ಆರಂಭ_ids, dtype=torch.ದೀರ್ಘ, device=device)[ಏನೂಇಲ್ಲ, ...])
            max_new_ಟೋಕನುಗಳು = 32
            temperature = 1.0
            ತುದಿ_k = 40
            yg = raw_model.generate(xg, max_new_ಟೋಕನುಗಳು, temperature=temperature, ತುದಿ_k=ತುದಿ_k)
            print0('---------------')
            print0(enc.decode(yg[0].tolist()))
            print0('---------------')

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
