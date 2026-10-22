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

استورد os
استورد math
استورد glob
استورد struct
استورد inspect
من contextlib استورد nullcontext
من dataclasses استورد dataclass

استورد numpy كـ np
استورد torch
استورد torch.nn كـ nn
من torch.nn استورد functional كـ F
استورد torch._inductor.config كـ config
من torch.nn.parallel استورد Distributed_معلومات_Parallel كـ DDP
من torch.distributed استورد init_process_گروهه, destroy_process_گروهه
من torch.distributed.optim استورد ZeroRedundancyOptimizer
استورد torch.distributed كـ dist

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model

صنف NewGELU(nn.Module):
    """Careful there are a few versions of GeLU, this one is the exact one used by OpenAI"""
    وصف forward(self, داخلا):
        أرجع 0.5 * داخلا * (1.0 + torch.tanh(math.جذر(2.0 / math.pi) * (داخلا + 0.044715 * torch.pow(داخلا, 3.0))))

# using a global to toggle flash-attention
FLASH = 0

صنف CausalSelfAttention(nn.Module):

    وصف __init__(self, config):
        أعلى().__init__()
        أكد config.n_embd % config.n_سر == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.لڪيري(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.لڪيري(config.n_embd, config.n_embd)
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1
        # regularization
        self.n_سر = config.n_سر
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.رجسٽر_بفر("bias", torch.tril(torch.ones(config.بلاڪ_سائيز, config.بلاڪ_سائيز))
                                     .view(1, 1, config.بلاڪ_سائيز, config.بلاڪ_سائيز))

    وصف forward(self, x):
        B, T, C = x.سائيز() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.ورهايو(self.n_embd, dim=2)
        k = k.view(B, T, self.n_سر, C // self.n_سر).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_سر, C // self.n_سر).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_سر, C // self.n_سر).transpose(1, 2) # (B, nh, T, hs)
        إذا FLASH:
            # flashattention
            y = F.scaled_dot_حاصل_ضرب_attention(q, k, v, is_causal=سچ)
        وإلا:
            # manual implementation of attention
            # this materializes the large (T,T) matrix for all the queries and keys
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.جذر(k.سائيز(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, عائم('-inf'))
            att = F.softmax(att, dim=-1)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        أرجع y

صنف MLP(nn.Module):

    وصف __init__(self, config):
        أعلى().__init__()
        self.c_fc    = nn.لڪيري(config.n_embd, 4 * config.n_embd)
        self.gelu    = NewGELU()
        self.c_proj  = nn.لڪيري(4 * config.n_embd, config.n_embd)
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1

    وصف forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        أرجع x

صنف بلاڪ(nn.Module):

    وصف __init__(self, config):
        أعلى().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    وصف forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        أرجع x

# -----------------------------------------------------------------------------
# The main GPT-2 model

@dataclass
صنف GPTConfig:
    بلاڪ_سائيز: صحيح = 1024
    vocab_سائيز: صحيح = 50257
    n_layer: صحيح = 12
    n_سر: صحيح = 12
    n_embd: صحيح = 768

صنف GPT(nn.Module):

    وصف __init__(self, config):
        أعلى().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_سائيز, config.n_embd),
            wpe = nn.Embedding(config.بلاڪ_سائيز, config.n_embd),
            h = nn.ModuleList([بلاڪ(config) لكل _ في سلسلو(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_سر = nn.لڪيري(config.n_embd, config.vocab_سائيز, bias=خطأ)
        self.lm_سر.LLMC_SKIP_INIT = 1 # don't init this one, we will tie weights
        self.transformer.wte.وزن = self.lm_سر.وزن # https://paperswithcode.com/method/weight-tying

        # init all weights, use a torch rng object to be very careful
        self.init_rng = torch.Generator()
        self.init_rng.manual_ٻج(42)
        self.apply(self._init_weights)

    وصف _init_weights(self, module):
        إذا isinstance(module, nn.لڪيري):
            # apply special scaled init to the residual projections, per GPT-2 paper
            std = 0.02 إذا ليس hasattr(module, 'LLMC_RESIDUAL_SCALE_FLAG') وإلا 0.02/math.جذر(2 * self.config.n_layer)
            # we want to skip initializing lm_head, which shares parameters with wte
            # and wte was already initialized down below during the Embedding init
            إذا ليس hasattr(module, 'LLMC_SKIP_INIT'):
                torch.nn.init.عام_2(module.وزن, سراسري=0.0, std=std, generator=self.init_rng)
            إذا module.bias آهي ليس ڪجهه_نه:
                torch.nn.init.zeros_(module.bias)
        وإلاإذا isinstance(module, nn.Embedding):
            torch.nn.init.عام_2(module.وزن, سراسري=0.0, std=0.02, generator=self.init_rng)

    وصف forward(self, idx, targets=ڪجهه_نه, return_logits=سچ):
        device = idx.device
        b, t = idx.سائيز()
        أكد t <= self.config.بلاڪ_سائيز, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.ڊگهو, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = tok_emb + pos_emb

        لكل بلاڪ_2 في self.transformer.h:
            x = بلاڪ_2(x)
        x = self.transformer.ln_f(x)

        إذا targets آهي ليس ڪجهه_نه:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_سر(x)
            loss = F.cross_entropy(logits.view(-1, logits.سائيز(-1)), targets.view(-1), ignore_اشاريو=-1)
        وإلا:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_سر(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = ڪجهه_نه

        # there are performance reasons why not returning logits is prudent, if not needed
        إذا ليس return_logits:
            logits = ڪجهه_نه

        أرجع logits, loss

    @classmethod
    وصف from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        أكد model_type في {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        من transformers استورد GPT2LMHeadModel
        ڇاپيو("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_دليلون = {
            'gpt2':         dict(n_layer=12, n_سر=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_سر=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_سر=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_سر=25, n_embd=1600), # 1558M params
        }[model_type]
        config_دليلون['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_دليلون['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_دليلون)
        model = GPT(config)
        sd = model.state_dict()
        sd_ڪنجيون = sd.ڪنجيون()
        sd_ڪنجيون = [k لكل k في sd_ڪنجيون إذا ليس k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_ڪنجيون_hf = sd_hf.ڪنجيون()
        sd_ڪنجيون_hf = [k لكل k في sd_ڪنجيون_hf إذا ليس k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_ڪنجيون_hf = [k لكل k في sd_ڪنجيون_hf إذا ليس k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        أكد ڊيگهه(sd_ڪنجيون_hf) == ڊيگهه(sd_ڪنجيون), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        لكل k في sd_ڪنجيون_hf:
            إذا any(k.endswith(w) لكل w في transposed):
                # special treatment for the Conv1D weights we need to transpose
                أكد sd_hf[k].shape[::-1] == sd[k].shape
                سان torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            وإلا:
                # vanilla copy over the other parameters
                أكد sd_hf[k].shape == sd[k].shape
                سان torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        أرجع model

    وصف configure_optimizers(self, وزن_decay, learning_شرح, betas, device_type, zero_stage):
        # start with all of the candidate parameters
        param_dict = {pn: p لكل pn, p في self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p لكل pn, p في param_dict.شيون() إذا p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p لكل n, p في param_dict.شيون() إذا p.dim() >= 2]
        nodecay_params = [p لكل n, p في param_dict.شيون() إذا p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': وزن_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        انگ_decay_params = جوڙ(p.numel() لكل p في decay_params)
        انگ_nodecay_params = جوڙ(p.numel() لكل p في nodecay_params)
        print0(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print0(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_موجود = 'fused' في inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_موجود و device_type == 'cuda'
        print0(f"using fused AdamW: {use_fused}")
        إذا zero_stage == 1:
            print0("using ZeroRedundancyOptimizer")
            optimizer = ZeroRedundancyOptimizer(**optim_groups[0], optimizer_class=torch.optim.AdamW,
                                                lr=learning_شرح, betas=betas, fused=use_fused)
            optimizer.جوڙيو_param_گروهه(optim_groups[1])
        وإلا:
            print0("using regular AdamW")
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_شرح, betas=betas, fused=use_fused)
        أرجع optimizer

    @torch.no_grad()
    وصف generate(self, idx, max_new_ٽوڪن, temperature=1.0, چوٽي_k=ڪجهه_نه):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        لكل _ في سلسلو(max_new_ٽوڪن):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx إذا idx.سائيز(1) <= self.config.بلاڪ_سائيز وإلا idx[:, -self.config.بلاڪ_سائيز:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            إذا چوٽي_k آهي ليس ڪجهه_نه:
                v, _ = torch.topk(logits, min(چوٽي_k, logits.سائيز(-1)))
                logits[logits < v[:, [-1]]] = -عائم('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_ايندڙ = torch.multinomial(probs, انگ_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_ايندڙ), dim=1)

        أرجع idx

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

وصف peek_معلومات_shard(filename):
    # only reads the header, returns header data
    سان open(filename, "rb") كـ f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        سرو = np.frombuffer(f.اقرأ(256*4), dtype=np.int32)
    إذا سرو[0] != 20240520:
        ڇاپيو("ERROR: magic number mismatch in the data .bin file!")
        ڇاپيو("---> HINT: Are you passing in a correct file with --input_bin?")
        ڇاپيو("---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README")
        ڇاپيو("---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try")
        exit(1)
    أكد سرو[1] == 1, "unsupported version"
    ntok = سرو[2] # number of tokens (claimed)
    أرجع ntok # for now just return the number of tokens

وصف ڀريو_معلومات_shard(filename):
    سان open(filename, "rb") كـ f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        سرو = np.frombuffer(f.اقرأ(256*4), dtype=np.int32)
        أكد سرو[0] == 20240520, "magic number mismatch in the data .bin file"
        أكد سرو[1] == 1, "unsupported version"
        ntok = سرو[2] # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        ٽوڪن = np.frombuffer(f.اقرأ(), dtype=np.uint16)
    أكد ڊيگهه(ٽوڪن) == ntok, "number of tokens read does not match header?"
    أرجع ٽوڪن

صنف Distributed_معلومات_Loader:
    وصف __init__(self, filename_نمونو, B, T, process_rank, انگ_processes):
        self.process_rank = process_rank
        self.انگ_processes = انگ_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_نمونو))
        أكد ڊيگهه(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_ڪل = 0
        لكل fname في self.files:
            shard_ntok = peek_معلومات_shard(fname)
            أكد shard_ntok >= انگ_processes * B * T + 1
            ntok_ڪل += shard_ntok
        self.ntok_ڪل = ntok_ڪل
        print0(f"DataLoader: total number of tokens: {ntok_total:,} across {len(self.files)} files")

        # kick things off
        self.current_shard = ڪجهه_نه
        self.reset()

    وصف reset(self):
        # we're being a bit clever here: if we already had shard 0 loaded,
        # then don't do the work to reload it, just reset the pointer
        إذا self.current_shard != 0:
            self.current_shard = 0
            self.ٽوڪن = ڀريو_معلومات_shard(self.files[self.current_shard])
        self.current_position = self.process_rank * self.B * self.T

    وصف advance(self): # advance to next data shard
        self.current_shard = (self.current_shard + 1) % ڊيگهه(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.ٽوڪن = ڀريو_معلومات_shard(self.files[self.current_shard])

    وصف ايندڙ_batch(self):
        B = self.B
        T = self.T
        buf = self.ٽوڪن[self.current_position : self.current_position+B*T+1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.ڊگهو)
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the start pointer in current shard
        self.current_position += B * T * self.انگ_processes
        # if loading the next batch would be out of bounds advance the shard
        إذا self.current_position + (B * T * self.انگ_processes + 1) > ڊيگهه(self.ٽوڪن):
            self.advance()
        أرجع x, y

# -----------------------------------------------------------------------------
# Python -> C bridge utilities for saving params/grads/activations to .bin files

وصف write_fp32(tensor, file):
    t = tensor.detach().cpu().منزل(torch.float32)
    b = t.numpy().tobytes()
    file.اكتب(b)

وصف write_bf16(tensor, file):
    t = tensor.detach().cpu().منزل(torch.bfloat16)
    # numpy doesn't have bf16 datatype so we have to trick it
    t = t.view(torch.int16) # trick: reinterpret as int16
    b = t.numpy().tobytes()
    file.اكتب(b)

وصف write_tensors(model_tensors, L, file, dtype):
    # writes the GPT-2 model's weights to a binary file
    أكد dtype في {"float32", "bfloat16"}
    write_fun = write_fp32 إذا dtype == "float32" وإلا write_bf16
    write_fun(model_tensors["transformer.wte.weight"], file) # (V, C)
    write_fun(model_tensors["transformer.wpe.weight"], file) # (T, C)
    لكل i في سلسلو(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_1.weight"], file)
    لكل i في سلسلو(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_1.bias"], file)
    لكل i في سلسلو(L): # (L, 3C, C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_attn.weight"], file)
    لكل i في سلسلو(L): # (L, 3C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_attn.bias"], file)
    لكل i في سلسلو(L): # (L, C, C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_proj.weight"], file)
    لكل i في سلسلو(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_proj.bias"], file)
    لكل i في سلسلو(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_2.weight"], file)
    لكل i في سلسلو(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_2.bias"], file)
    لكل i في سلسلو(L): # (L, 4C, C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_fc.weight"], file)
    لكل i في سلسلو(L): # (L, 4C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_fc.bias"], file)
    لكل i في سلسلو(L): # (L, C, 4C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_proj.weight"], file)
    لكل i في سلسلو(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_proj.bias"], file)
    write_fun(model_tensors["transformer.ln_f.weight"], file) # (C, )
    write_fun(model_tensors["transformer.ln_f.bias"], file) # (C, )

@torch.no_grad()
وصف pad_vocab(tensor, گڻو=128, قدر=0):
    """
    The dimension of the vocab size in GPT-2 is 50,257
    which is unfortunately a very unfriendly number for a lot of
    matrix operations on the GPU. So we pad it to the nearest
    friendlier multiple, e.g. 50,304 if multiple=128 when we
    export the weights into C land. This is a NOOP algorithmically
    and is only done to make the tensor operations more efficient.
    """
    أكد tensor.ndim == 2
    V, C = tensor.shape
    أكد V == 50257, "just being defensive here"
    # calculate padded vocab size by rounding up to nearest multiple
    Vp = ((V + گڻو - 1) // گڻو) * گڻو
    # pad the tensor
    pad_صفون = Vp - V
    padded = tensor إذا pad_صفون == 0 وإلا F.pad(tensor, (0, 0, 0, pad_صفون), قدر=قدر)
    أكد padded.shape == (Vp, C)
    أرجع padded

وصف write_model(model, filename, dtype):
    # everything we need to instantiate the model
    # 1) header is: version int, GPTConfig ints, padding to 1024 bytes
    أكد dtype في {"float32", "bfloat16"} # float16 todo maybe later
    نسخو = {
        "float32": 3, # 3: all tensors are fp32, padded vocab
        "bfloat16": 5, # 5: all tensors are bf16, padded vocab
    }[dtype]
    سرو = torch.zeros(256, dtype=torch.int32)
    سرو[0] = 20240326 # magic
    سرو[1] = نسخو # checkpoint version
    سرو[2] = model.config.بلاڪ_سائيز
    سرو[3] = model.config.vocab_سائيز
    سرو[4] = model.config.n_layer
    سرو[5] = model.config.n_سر
    سرو[6] = model.config.n_embd
    # 2) the parameters follow the header
    params = {name: param.cpu() لكل name, param في model.named_parameters()}
    # pad the vocab to a multiple of 128 here at export, for efficiency in C
    wte = params["transformer.wte.weight"] # (V, C)
    wte_padded = pad_vocab(wte) # (Vp, C)
    params["transformer.wte.weight"] = wte_padded # (Vp, C)
    ڇاپيو(f"padded vocab size from {wte.size(0)} to {wte_padded.size(0)}")
    سرو[7] = wte_padded.سائيز(0) # padded vocab size store in header
    # now write to file
    سان open(filename, "wb") كـ file:
        file.اكتب(سرو.numpy().tobytes()) #سرو
        write_tensors(params, model.config.n_layer, file, dtype) # params
    ڇاپيو(f"wrote {filename}")

وصف write_state(model, x, y, logits, loss, filename):
    # the state is used for debugging.
    # it contains information about the input, logits, loss, and the parameter gradients
    # this can be used for checking the computation correctness in C
    سرو = torch.zeros(256, dtype=torch.int32)
    سرو[0] = 20240327 # magic
    سرو[1] = 2 # run state version = 2 (1 -> 2 for padded vocab changes)
    سرو[2] = x.سائيز(0) # batch size of the batch, B
    سرو[3] = x.سائيز(1) # temporal extent of the batch, T
    grads = {name: param.grad.cpu() لكل name, param في model.named_parameters()}
    # pad the vocab grads here as well, to mirror write_model
    wte_grad = grads["transformer.wte.weight"] # (V, C)
    wte_grad_padded = pad_vocab(wte_grad, قدر=0) # (Vp, C) # TODO later maybe pad with nan?
    grads["transformer.wte.weight"] = wte_grad_padded # (Vp, C)
    ڇاپيو(f"padded vocab size in reference grads from {wte_grad.size(0)} to {wte_grad_padded.size(0)}")
    سان open(filename, "wb") كـ file:
        #سرو
        file.اكتب(سرو.numpy().tobytes())
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
    ڇاپيو(f"wrote {filename}")

وصف write_tokenizer(enc, filename):
    n = enc.max_token_قدر + 1
    سرو = torch.zeros(256, dtype=torch.int32)
    سرو[0] = 20240328 # magic
    سرو[1] = 2 # tokenizer version = 2 (1 -> 2: includes EOT token)
    سرو[2] = n # number of tokens
    سرو[3] = enc.eot_token # EOT token
    سان open(filename, "wb") كـ file:
        file.اكتب(سرو.numpy().tobytes())
        لكل i في سلسلو(n):
            b = enc.decode_بائيٽون([i])
            ڊيگهائي = ڊيگهه(b)
            أكد ڊيگهائي < 256, f"Token length exceeds 255: {length}"
            file.اكتب(struct.pack("<B", ڊيگهائي))  # Write the length as a 1-byte unsigned integer
            file.اكتب(b)  # Write the actual bytes
    ڇاپيو(f"wrote {filename}")

# -----------------------------------------------------------------------------
# int main

وصف print0(*دليلون, **kwargs):
    # modified print that only prints from the master process
    # if this is not a distributed run, it's just a print
    إذا صحيح(os.environ.get("RANK", 0)) == 0:
        ڇاپيو(*دليلون, **kwargs)

إذا __name__ == "__main__":
    استورد time
    استورد argparse
    استورد tiktoken
    print0(f"Running pytorch {torch.version.__version__}")

    # default settings will overfit a tiny batch of data
    # and save model weights and debug state to disk on the first iteration
    parser = argparse.دليل_Parser()
    # file system input / output
    parser.جوڙيو_دليل("--input_bin", type=str, افتراضي="dev/data/tinyshakespeare/tiny_shakespeare_val.bin", help="input .bin to train on")
    parser.جوڙيو_دليل("--input_val_bin", type=str, افتراضي="", help="input .bin to eval validation loss on")
    parser.جوڙيو_دليل("--output_dir", type=str, افتراضي="", help="output directory to which to write logs and checkpoints")
    parser.جوڙيو_دليل("--model", type=str, افتراضي="gpt2", help="gpt2|gpt2-medium|gpt2-large|gpt2-xl|d12|d24|d36|d48")
    # token layout for each step of the optimization
    parser.جوڙيو_دليل("--batch_size", type=صحيح, افتراضي=4, help="batch size, in units of #batch dimensions")
    parser.جوڙيو_دليل("--sequence_length", type=صحيح, افتراضي=64, help="sequence length")
    parser.جوڙيو_دليل("--total_batch_size", type=صحيح, افتراضي=256, help="total desired batch size, in units of #tokens")
    # workload (number of steps)
    parser.جوڙيو_دليل("--num_iterations", type=صحيح, افتراضي=10, help="number of iterations to run")
    parser.جوڙيو_دليل("--inference_only", type=صحيح, افتراضي=0, help="only run inference")
    # optimization
    parser.جوڙيو_دليل("--learning_rate", type=عائم, افتراضي=1e-4, help="learning rate warmup iterations")
    parser.جوڙيو_دليل("--warmup_iters", type=صحيح, افتراضي=0, help="learning rate warmup iterations")
    parser.جوڙيو_دليل("--learning_rate_decay_frac", type=عائم, افتراضي=1.0, help="learning rate warmup iterations")
    parser.جوڙيو_دليل("--weight_decay", type=عائم, افتراضي=0.0, help="weight decay")
    parser.جوڙيو_دليل("--grad_clip", type=عائم, افتراضي=1.0, help="maximum gradient magnitude")
    # evaluation
    parser.جوڙيو_دليل("--val_loss_every", type=صحيح, افتراضي=0, help="every how mant steps to evaluate val loss?")
    parser.جوڙيو_دليل("--val_max_steps", type=صحيح, افتراضي=20, help="how many batches of val to average?")
    parser.جوڙيو_دليل("--sample_every", type=صحيح, افتراضي=0, help="how often to sample from the model?")
    # debugging
    parser.جوڙيو_دليل("--overfit_single_batch", type=صحيح, افتراضي=1, help="overfit just one batch of data")
    # numerics
    parser.جوڙيو_دليل("--tensorcores", type=صحيح, افتراضي=0, help="use tensorcores")
    # memory management
    parser.جوڙيو_دليل("--device", type=str, افتراضي="", help="by default we autodetect, or set it here")
    parser.جوڙيو_دليل("--compile", type=صحيح, افتراضي=0, help="torch.compile the model")
    parser.جوڙيو_دليل("--flash", type=صحيح, افتراضي=0, help="use flash attention")
    parser.جوڙيو_دليل("--dtype", type=str, افتراضي="float32", help="float32|float16|bfloat16")
    parser.جوڙيو_دليل("--zero_stage", type=صحيح, افتراضي=0, help="zero redundancy optimizer stage (0/1/2/3)")
    # python -> C bridge
    parser.جوڙيو_دليل("--write_tensors", type=صحيح, افتراضي=1, help="write tensors to disk")
    دليلون = parser.parse_دليلون()

    # args error checking and convenience variables
    B, T = دليلون.batch_سائيز, دليلون.sequence_ڊيگهائي
    أكد 1 <= T <= 1024
    أكد دليلون.dtype في {"float32", "float16", "bfloat16"}
    أكد دليلون.model في {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "d12", "d24", "d36", "d48"}

    # set up DDP (distributed data parallel). torchrun sets this env variable
    ddp = صحيح(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    إذا ddp:
        # use of DDP atm demands CUDA, we set the device appropriately according to rank
        أكد torch.cuda.is_موجود(), "for now i think we need CUDA for DDP"
        init_process_گروهه(backend='nccl')
        ddp_rank = صحيح(os.environ['RANK'])
        ddp_local_rank = صحيح(os.environ['LOCAL_RANK'])
        ddp_world_سائيز = صحيح(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.سيٽ_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        ٻج_وٿي = 0 # each process gets the exact same seed
        zero_stage = دليلون.zero_stage
    وإلا:
        ddp_rank = 0
        ddp_local_rank = 0
        zero_stage = 0
        ddp_world_سائيز = 1
        master_process = سچ
        ٻج_وٿي = 0
        # select the device
        إذا دليلون.device:
            # provided explicitly by the user
            device = دليلون.device
        وإلا:
            # attempt to autodetect the device
            device = "cpu"
            إذا torch.cuda.is_موجود():
                device = "cuda"
            وإلاإذا hasattr(torch.backends, "mps") و torch.backends.mps.is_موجود():
                device = "mps"
    ڇاپيو(f"using device: {device}")
    device_type = 'cuda' إذا 'cuda' في device وإلا 'cpu'

    # calculate gradient accumulation from the desired total batch size and the current run configuration
    ٽوڪن_في_fwdbwd = B * T * ddp_world_سائيز
    أكد دليلون.ڪل_batch_سائيز % ٽوڪن_في_fwdbwd == 0
    grad_accum_قدمن = دليلون.ڪل_batch_سائيز // ٽوڪن_في_fwdbwd
    print0(f"total desired batch size: {args.total_batch_size}")
    print0(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    # set up a context manager following the desired dtype and device
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[دليلون.dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) إذا device_type == "cuda" وإلا nullcontext()

    # rng / reproducibility
    torch.manual_ٻج(42)
    إذا torch.cuda.is_موجود():
        torch.cuda.manual_ٻج(42)

    # set the torch precision mode to use TensorFloat32 (TF32) for matmuls
    # docs https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    إذا دليلون.tensorcores:
        torch.سيٽ_float32_matmul_precision('high')

    # turn on/off flash attention
    أكد دليلون.flash في {0, 1}
    FLASH = دليلون.flash

    # init (and write) the tokenizer
    enc = tiktoken.get_encoding("gpt2")
    إذا master_process و دليلون.write_tensors: # tokenizer is technically not tensors but ok
        write_tokenizer(enc, "gpt2_tokenizer.bin")

    # init the model, either from scratch or from OpenAI pretrained checkpoint
    إذا دليلون.model[0] == "d":
        # from scratch (random weights)
        model_config = {
            "d12": GPTConfig(بلاڪ_سائيز=1024, vocab_سائيز=50257, n_layer=12, n_سر=12, n_embd=768),
            "d24": GPTConfig(بلاڪ_سائيز=1024, vocab_سائيز=50257, n_layer=24, n_سر=16, n_embd=1024),
            "d36": GPTConfig(بلاڪ_سائيز=1024, vocab_سائيز=50257, n_layer=36, n_سر=20, n_embd=1280),
            "d48": GPTConfig(بلاڪ_سائيز=1024, vocab_سائيز=50257, n_layer=48, n_سر=25, n_embd=1600),
        }[دليلون.model]
        model = GPT(model_config)
    وإلا:
        # load the GPT-2 model weights
        model = GPT.from_pretrained(دليلون.model)
    model.train()
    model.منزل(device)
    إذا دليلون.compile:
        إذا hasattr(config, "coordinate_descent_tuning"):
            config.coordinate_descent_tuning = سچ # suggested by @Chillee
        print0("compiling the model...")
        model = torch.compile(model)

    # -------------------------------------------------------------------------
    # Our own version of a simple DistributedDataLoader

    # load tokens
    train_loader = Distributed_معلومات_Loader(دليلون.داخلا_bin, B, T, ddp_rank, ddp_world_سائيز)
    val_loader = ڪجهه_نه
    إذا دليلون.داخلا_val_bin:
        val_loader = Distributed_معلومات_Loader(دليلون.داخلا_val_bin, B, T, ddp_rank, ddp_world_سائيز)

    # -------------------------------------------------------------------------
    # PyTorch -> C bridge: save some weights and state for C to load later as reference

    # do one forward pass to generate ground truth for our C tests
    إذا master_process و دليلون.write_tensors و (ليس دليلون.inference_only):
        x, y = train_loader.ايندڙ_batch()
        x, y = x.منزل(device), y.منزل(device)
        logits, loss = model(x, y)
        loss.backward()
        # save model params, in both float32 and bfloat16
        model_منزل_سائيز = {"gpt2": "124M", "gpt2-medium": "355M", "gpt2-large": "774M", "gpt2-xl": "1558M"}
        model_منزل_سائيز.update({f"d{d}": f"d{d}" لكل d في [12, 24, 36, 48]})
        model_سائيز_str = model_منزل_سائيز[دليلون.model] # e.g. "124M", or "d12"
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
    إذا ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module إذا ddp وإلا model # always contains the "raw" unwrapped model

    # init the optimizer
    optimizer = raw_model.configure_optimizers(وزن_decay=دليلون.وزن_decay,
                                               learning_شرح=دليلون.learning_شرح, betas=(0.9, 0.95),
                                               device_type=device, zero_stage=zero_stage)

    # learning rate decay scheduler (cosine with warmup)
    وصف get_lr(it):
        min_lr = دليلون.learning_شرح * دليلون.learning_شرح_decay_frac
        # 1) linear warmup for warmup_iters steps
        إذا it < دليلون.warmup_iters:
            أرجع دليلون.learning_شرح * (it+1) / دليلون.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        إذا it > دليلون.انگ_iterations:
            أرجع min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_تناسب = (it - دليلون.warmup_iters) / (دليلون.انگ_iterations - دليلون.warmup_iters)
        أكد 0 <= decay_تناسب <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_تناسب)) # coeff starts at 1 and goes to 0
        أرجع min_lr + coeff * (دليلون.learning_شرح - min_lr)

    # create the logging directory if it does not exist
    logfile = ڪجهه_نه
    إذا دليلون.output_dir:
        os.makedirs(دليلون.output_dir, exist_ok=سچ)
        logfile = os.رستو.join(دليلون.output_dir, "main.log")
        # create the log file "main.log" inside it, and wipe it clean
        سان open(logfile, "w") كـ f:
            لنگهه

    إذا device == "cuda":
        torch.cuda.reset_peak_يادگيري_stats()
    timings = []
    norm = -1.0   # dummy value to print in inference-only mode
    لكل قدم في سلسلو(دليلون.انگ_iterations + 1):
        t0 = time.time()
        آخري_قدم = (قدم == دليلون.انگ_iterations)

        # once in a while evaluate the validation dataset
        إذا (دليلون.val_loss_every > 0 \
            و (قدم % دليلون.val_loss_every == 0 أو آخري_قدم)) \
            و (val_loader آهي ليس ڪجهه_نه):
            model.eval()
            val_loader.reset()
            سان torch.no_grad():
                val_loss = 0.0
                لكل _ في سلسلو(دليلون.val_max_قدمن):
                    x, y = val_loader.ايندڙ_batch()
                    x, y = x.منزل(device), y.منزل(device)
                    _, loss = model(x, y, return_logits=خطأ)
                    val_loss += loss.شي()
                val_loss /= دليلون.val_max_قدمن
            # log to console and to file
            print0(f"val loss {val_loss}")
            إذا master_process و logfile آهي ليس ڪجهه_نه:
                سان open(logfile, "a") كـ f:
                    f.اكتب("s:%d tel:%f\n" % (قدم, val_loss))

        # once in a while perform model inference on the master process
        إذا (دليلون.sample_every > 0 \
            و (قدم % دليلون.sample_every == 0 أو آخري_قدم)) \
            و master_process:
            model.eval()
            # before we end, let's also do one round of inference
            # we'll kick off the generation with "<|endoftext|>", which designates the start of a new sequence
            شروعات_ids = [enc.eot_token]
            xg = (torch.tensor(شروعات_ids, dtype=torch.ڊگهو, device=device)[ڪجهه_نه, ...])
            max_new_ٽوڪن = 32
            temperature = 1.0
            چوٽي_k = 40
            yg = raw_model.generate(xg, max_new_ٽوڪن, temperature=temperature, چوٽي_k=چوٽي_k)
            print0('---------------')
            print0(enc.decode(yg[0].tolist()))
            print0('---------------')

        # bit confusing: we want to make sure to eval and sample on 0th iteration
        # but also after the very last iteration. so we loop for step <= num_iterations
        # instead of just < num_iterations (one extra due to <=), only to do
        # the validation/sampling one last time, and then we break right here as we're done.
        إذا آخري_قدم:
            اكسر

        # --------------- TRAINING SECTION BEGIN -----------------
        model.train()
        optimizer.zero_grad(سيٽ_منزل_none=سچ)
        # if we are trying to overfit a single batch, we reset the loader here
        إذا دليلون.overfit_single_batch:
            train_loader.reset()
        # micro-batch loop where we do gradient accumulation to reach desired total batch size
        lossf = 0.0 # for getting the mean loss (as simple float) over the accumulation steps
        لكل micro_قدم في سلسلو(grad_accum_قدمن):
            # fetch a batch
            x, y = train_loader.ايندڙ_batch()
            x, y = x.منزل(device), y.منزل(device)
            إذا ddp:
                # we want only the last micro-step to sync grads in a DDP model
                # the official way to do this is with model.no_sync(), but that is a
                # context manager that bloats the code, so we just toggle this variable
                model.require_backward_grad_sync = (micro_قدم == grad_accum_قدمن - 1)
            # forward pass
            سان ctx:
                _, loss = model(x, y, return_logits=خطأ)
                # we have to scale the loss to account for gradient accumulation,
                # because the gradients just add on each successive backward().
                # addition of gradients corresponds to a SUM in the objective, but
                # instead of a SUM we want MEAN, so we scale the loss here
                loss = loss / grad_accum_قدمن
                lossf += loss.detach() # keep track of the mean loss
            # backward pass
            إذا ليس دليلون.inference_only:
                loss.backward()
        إذا ddp:
            dist.all_reduce(lossf, op=dist.ReduceOp.AVG)
        lossf = lossf.شي()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), دليلون.grad_clip)
        # determine and set the learning rate for this iteration
        lr = get_lr(قدم)
        لكل param_گروهه في optimizer.param_groups:
            param_گروهه['lr'] = lr
        # step the optimizer
        optimizer.قدم()
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
        ٽوڪن_في_ٻيو = grad_accum_قدمن * ddp_world_سائيز * B * T / (t1-t0)
        print0(f"step {step+1:4d}/{args.num_iterations} | train loss {lossf:.6f} | norm {norm:.4f} | lr {lr:.2e} | ({(t1-t0)*1000:.2f} ms | {tokens_per_second:.0f} tok/s)")
        # log to logile
        إذا master_process و logfile آهي ليس ڪجهه_نه:
            سان open(logfile, "a") كـ f:
                f.اكتب("s:%d trl:%f\n" % (قدم, lossf))

        # keep track of smooth timings, last 20 iterations
        إذا قدم > 0 و قدم > دليلون.انگ_iterations - 20:
            timings.append(t1-t0)

    # print the average of the last 20 timings, to get something smooth-ish
    timings = timings[-20:]
    print0(f"final {len(timings)} iters avg: {np.mean(timings)*1000:.3f}ms")
    print0(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

    # -------------------------------------------------------------------------
    # clean up nice
    إذا ddp:
        destroy_process_گروهه()
