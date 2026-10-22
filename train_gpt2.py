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
من torch.nn.parallel استورد Distributed_انکڑے_Parallel كـ DDP
من torch.distributed استورد init_process_گروہ, destroy_process_گروہ
من torch.distributed.optim استورد ZeroRedundancyOptimizer
استورد torch.distributed كـ dist

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model

صنف NewGELU(nn.Module):
    """Careful there are a few versions of GeLU, this one is the exact one used by OpenAI"""
    تعریف forward(self, داخلہ):
        أرجع 0.5 * داخلہ * (1.0 + torch.tanh(math.جذر(2.0 / math.pi) * (داخلہ + 0.044715 * torch.pow(داخلہ, 3.0))))

# using a global to toggle flash-attention
FLASH = 0

صنف CausalSelfAttention(nn.Module):

    تعریف __init__(self, config):
        أعلى().__init__()
        أكد config.n_embd % config.n_سر == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.خطی(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.خطی(config.n_embd, config.n_embd)
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1
        # regularization
        self.n_سر = config.n_سر
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.رجسٹر_بفر("bias", torch.tril(torch.ones(config.بلاک_ناپ, config.بلاک_ناپ))
                                     .view(1, 1, config.بلاک_ناپ, config.بلاک_ناپ))

    تعریف forward(self, x):
        B, T, C = x.ناپ() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.ونڈو(self.n_embd, dim=2)
        k = k.view(B, T, self.n_سر, C // self.n_سر).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_سر, C // self.n_سر).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_سر, C // self.n_سر).transpose(1, 2) # (B, nh, T, hs)
        إذا FLASH:
            # flashattention
            y = F.scaled_dot_ضرب_حاصل_attention(q, k, v, is_causal=سچ)
        وإلا:
            # manual implementation of attention
            # this materializes the large (T,T) matrix for all the queries and keys
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.جذر(k.ناپ(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, عائم('-inf'))
            att = F.softmax(att, dim=-1)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        أرجع y

صنف MLP(nn.Module):

    تعریف __init__(self, config):
        أعلى().__init__()
        self.c_fc    = nn.خطی(config.n_embd, 4 * config.n_embd)
        self.gelu    = NewGELU()
        self.c_proj  = nn.خطی(4 * config.n_embd, config.n_embd)
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1

    تعریف forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        أرجع x

صنف بلاک(nn.Module):

    تعریف __init__(self, config):
        أعلى().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    تعریف forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        أرجع x

# -----------------------------------------------------------------------------
# The main GPT-2 model

@dataclass
صنف GPTConfig:
    بلاک_ناپ: صحيح = 1024
    vocab_ناپ: صحيح = 50257
    n_layer: صحيح = 12
    n_سر: صحيح = 12
    n_embd: صحيح = 768

صنف GPT(nn.Module):

    تعریف __init__(self, config):
        أعلى().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_ناپ, config.n_embd),
            wpe = nn.Embedding(config.بلاک_ناپ, config.n_embd),
            h = nn.ModuleList([بلاک(config) لكل _ في سلسلہ(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_سر = nn.خطی(config.n_embd, config.vocab_ناپ, bias=خطأ)
        self.lm_سر.LLMC_SKIP_INIT = 1 # don't init this one, we will tie weights
        self.transformer.wte.بھار = self.lm_سر.بھار # https://paperswithcode.com/method/weight-tying

        # init all weights, use a torch rng object to be very careful
        self.init_rng = torch.Generator()
        self.init_rng.manual_بیج(42)
        self.apply(self._init_weights)

    تعریف _init_weights(self, module):
        إذا isinstance(module, nn.خطی):
            # apply special scaled init to the residual projections, per GPT-2 paper
            std = 0.02 إذا ليس hasattr(module, 'LLMC_RESIDUAL_SCALE_FLAG') وإلا 0.02/math.جذر(2 * self.config.n_layer)
            # we want to skip initializing lm_head, which shares parameters with wte
            # and wte was already initialized down below during the Embedding init
            إذا ليس hasattr(module, 'LLMC_SKIP_INIT'):
                torch.nn.init.عام_2(module.بھار, اوسط=0.0, std=std, generator=self.init_rng)
            إذا module.bias اے ليس کجھ_نئیں:
                torch.nn.init.zeros_(module.bias)
        وإلاإذا isinstance(module, nn.Embedding):
            torch.nn.init.عام_2(module.بھار, اوسط=0.0, std=0.02, generator=self.init_rng)

    تعریف forward(self, idx, targets=کجھ_نئیں, return_logits=سچ):
        device = idx.device
        b, t = idx.ناپ()
        أكد t <= self.config.بلاک_ناپ, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.لما, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = tok_emb + pos_emb

        لكل بلاک_2 في self.transformer.h:
            x = بلاک_2(x)
        x = self.transformer.ln_f(x)

        إذا targets اے ليس کجھ_نئیں:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_سر(x)
            loss = F.cross_entropy(logits.view(-1, logits.ناپ(-1)), targets.view(-1), ignore_اشاریہ=-1)
        وإلا:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_سر(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = کجھ_نئیں

        # there are performance reasons why not returning logits is prudent, if not needed
        إذا ليس return_logits:
            logits = کجھ_نئیں

        أرجع logits, loss

    @classmethod
    تعریف from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        أكد model_type في {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        من transformers استورد GPT2LMHeadModel
        چھاپو("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_دلیلاں = {
            'gpt2':         dict(n_layer=12, n_سر=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_سر=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_سر=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_سر=25, n_embd=1600), # 1558M params
        }[model_type]
        config_دلیلاں['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_دلیلاں['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_دلیلاں)
        model = GPT(config)
        sd = model.state_dict()
        sd_کلیدیں = sd.کلیدیں()
        sd_کلیدیں = [k لكل k في sd_کلیدیں إذا ليس k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_کلیدیں_hf = sd_hf.کلیدیں()
        sd_کلیدیں_hf = [k لكل k في sd_کلیدیں_hf إذا ليس k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_کلیدیں_hf = [k لكل k في sd_کلیدیں_hf إذا ليس k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        أكد لمبائی(sd_کلیدیں_hf) == لمبائی(sd_کلیدیں), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        لكل k في sd_کلیدیں_hf:
            إذا any(k.endswith(w) لكل w في transposed):
                # special treatment for the Conv1D weights we need to transpose
                أكد sd_hf[k].shape[::-1] == sd[k].shape
                نال torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            وإلا:
                # vanilla copy over the other parameters
                أكد sd_hf[k].shape == sd[k].shape
                نال torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        أرجع model

    تعریف configure_optimizers(self, بھار_decay, learning_شرح, betas, device_type, zero_stage):
        # start with all of the candidate parameters
        param_dict = {pn: p لكل pn, p في self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p لكل pn, p في param_dict.اشیاء() إذا p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p لكل n, p في param_dict.اشیاء() إذا p.dim() >= 2]
        nodecay_params = [p لكل n, p في param_dict.اشیاء() إذا p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': بھار_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        عدد_decay_params = جوڑ(p.numel() لكل p في decay_params)
        عدد_nodecay_params = جوڑ(p.numel() لكل p في nodecay_params)
        print0(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print0(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_دستیاب = 'fused' في inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_دستیاب و device_type == 'cuda'
        print0(f"using fused AdamW: {use_fused}")
        إذا zero_stage == 1:
            print0("using ZeroRedundancyOptimizer")
            optimizer = ZeroRedundancyOptimizer(**optim_groups[0], optimizer_class=torch.optim.AdamW,
                                                lr=learning_شرح, betas=betas, fused=use_fused)
            optimizer.جوڑو_param_گروہ(optim_groups[1])
        وإلا:
            print0("using regular AdamW")
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_شرح, betas=betas, fused=use_fused)
        أرجع optimizer

    @torch.no_grad()
    تعریف generate(self, idx, max_new_ٹوکن, temperature=1.0, چوٹی_k=کجھ_نئیں):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        لكل _ في سلسلہ(max_new_ٹوکن):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx إذا idx.ناپ(1) <= self.config.بلاک_ناپ وإلا idx[:, -self.config.بلاک_ناپ:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            إذا چوٹی_k اے ليس کجھ_نئیں:
                v, _ = torch.topk(logits, min(چوٹی_k, logits.ناپ(-1)))
                logits[logits < v[:, [-1]]] = -عائم('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_اگلا = torch.multinomial(probs, عدد_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_اگلا), dim=1)

        أرجع idx

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

تعریف peek_انکڑے_shard(filename):
    # only reads the header, returns header data
    نال open(filename, "rb") كـ f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        سرخی = np.frombuffer(f.اقرأ(256*4), dtype=np.int32)
    إذا سرخی[0] != 20240520:
        چھاپو("ERROR: magic number mismatch in the data .bin file!")
        چھاپو("---> HINT: Are you passing in a correct file with --input_bin?")
        چھاپو("---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README")
        چھاپو("---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try")
        exit(1)
    أكد سرخی[1] == 1, "unsupported version"
    ntok = سرخی[2] # number of tokens (claimed)
    أرجع ntok # for now just return the number of tokens

تعریف لدو_انکڑے_shard(filename):
    نال open(filename, "rb") كـ f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        سرخی = np.frombuffer(f.اقرأ(256*4), dtype=np.int32)
        أكد سرخی[0] == 20240520, "magic number mismatch in the data .bin file"
        أكد سرخی[1] == 1, "unsupported version"
        ntok = سرخی[2] # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        ٹوکن = np.frombuffer(f.اقرأ(), dtype=np.uint16)
    أكد لمبائی(ٹوکن) == ntok, "number of tokens read does not match header?"
    أرجع ٹوکن

صنف Distributed_انکڑے_Loader:
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
        لكل fname في self.files:
            shard_ntok = peek_انکڑے_shard(fname)
            أكد shard_ntok >= عدد_processes * B * T + 1
            ntok_کل += shard_ntok
        self.ntok_کل = ntok_کل
        print0(f"DataLoader: total number of tokens: {ntok_total:,} across {len(self.files)} files")

        # kick things off
        self.current_shard = کجھ_نئیں
        self.reset()

    تعریف reset(self):
        # we're being a bit clever here: if we already had shard 0 loaded,
        # then don't do the work to reload it, just reset the pointer
        إذا self.current_shard != 0:
            self.current_shard = 0
            self.ٹوکن = لدو_انکڑے_shard(self.files[self.current_shard])
        self.current_position = self.process_rank * self.B * self.T

    تعریف advance(self): # advance to next data shard
        self.current_shard = (self.current_shard + 1) % لمبائی(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.ٹوکن = لدو_انکڑے_shard(self.files[self.current_shard])

    تعریف اگلا_batch(self):
        B = self.B
        T = self.T
        buf = self.ٹوکن[self.current_position : self.current_position+B*T+1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.لما)
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the start pointer in current shard
        self.current_position += B * T * self.عدد_processes
        # if loading the next batch would be out of bounds advance the shard
        إذا self.current_position + (B * T * self.عدد_processes + 1) > لمبائی(self.ٹوکن):
            self.advance()
        أرجع x, y

# -----------------------------------------------------------------------------
# Python -> C bridge utilities for saving params/grads/activations to .bin files

تعریف write_fp32(tensor, file):
    t = tensor.detach().cpu().منزل(torch.float32)
    b = t.numpy().tobytes()
    file.اكتب(b)

تعریف write_bf16(tensor, file):
    t = tensor.detach().cpu().منزل(torch.bfloat16)
    # numpy doesn't have bf16 datatype so we have to trick it
    t = t.view(torch.int16) # trick: reinterpret as int16
    b = t.numpy().tobytes()
    file.اكتب(b)

تعریف write_tensors(model_tensors, L, file, dtype):
    # writes the GPT-2 model's weights to a binary file
    أكد dtype في {"float32", "bfloat16"}
    write_fun = write_fp32 إذا dtype == "float32" وإلا write_bf16
    write_fun(model_tensors["transformer.wte.weight"], file) # (V, C)
    write_fun(model_tensors["transformer.wpe.weight"], file) # (T, C)
    لكل i في سلسلہ(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_1.weight"], file)
    لكل i في سلسلہ(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_1.bias"], file)
    لكل i في سلسلہ(L): # (L, 3C, C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_attn.weight"], file)
    لكل i في سلسلہ(L): # (L, 3C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_attn.bias"], file)
    لكل i في سلسلہ(L): # (L, C, C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_proj.weight"], file)
    لكل i في سلسلہ(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_proj.bias"], file)
    لكل i في سلسلہ(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_2.weight"], file)
    لكل i في سلسلہ(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_2.bias"], file)
    لكل i في سلسلہ(L): # (L, 4C, C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_fc.weight"], file)
    لكل i في سلسلہ(L): # (L, 4C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_fc.bias"], file)
    لكل i في سلسلہ(L): # (L, C, 4C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_proj.weight"], file)
    لكل i في سلسلہ(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_proj.bias"], file)
    write_fun(model_tensors["transformer.ln_f.weight"], file) # (C, )
    write_fun(model_tensors["transformer.ln_f.bias"], file) # (C, )

@torch.no_grad()
تعریف pad_vocab(tensor, مضرب=128, قدر=0):
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
    Vp = ((V + مضرب - 1) // مضرب) * مضرب
    # pad the tensor
    pad_صفاں = Vp - V
    padded = tensor إذا pad_صفاں == 0 وإلا F.pad(tensor, (0, 0, 0, pad_صفاں), قدر=قدر)
    أكد padded.shape == (Vp, C)
    أرجع padded

تعریف write_model(model, filename, dtype):
    # everything we need to instantiate the model
    # 1) header is: version int, GPTConfig ints, padding to 1024 bytes
    أكد dtype في {"float32", "bfloat16"} # float16 todo maybe later
    نسخہ = {
        "float32": 3, # 3: all tensors are fp32, padded vocab
        "bfloat16": 5, # 5: all tensors are bf16, padded vocab
    }[dtype]
    سرخی = torch.zeros(256, dtype=torch.int32)
    سرخی[0] = 20240326 # magic
    سرخی[1] = نسخہ # checkpoint version
    سرخی[2] = model.config.بلاک_ناپ
    سرخی[3] = model.config.vocab_ناپ
    سرخی[4] = model.config.n_layer
    سرخی[5] = model.config.n_سر
    سرخی[6] = model.config.n_embd
    # 2) the parameters follow the header
    params = {name: param.cpu() لكل name, param في model.named_parameters()}
    # pad the vocab to a multiple of 128 here at export, for efficiency in C
    wte = params["transformer.wte.weight"] # (V, C)
    wte_padded = pad_vocab(wte) # (Vp, C)
    params["transformer.wte.weight"] = wte_padded # (Vp, C)
    چھاپو(f"padded vocab size from {wte.size(0)} to {wte_padded.size(0)}")
    سرخی[7] = wte_padded.ناپ(0) # padded vocab size store in header
    # now write to file
    نال open(filename, "wb") كـ file:
        file.اكتب(سرخی.numpy().tobytes()) #سرخی
        write_tensors(params, model.config.n_layer, file, dtype) # params
    چھاپو(f"wrote {filename}")

تعریف write_state(model, x, y, logits, loss, filename):
    # the state is used for debugging.
    # it contains information about the input, logits, loss, and the parameter gradients
    # this can be used for checking the computation correctness in C
    سرخی = torch.zeros(256, dtype=torch.int32)
    سرخی[0] = 20240327 # magic
    سرخی[1] = 2 # run state version = 2 (1 -> 2 for padded vocab changes)
    سرخی[2] = x.ناپ(0) # batch size of the batch, B
    سرخی[3] = x.ناپ(1) # temporal extent of the batch, T
    grads = {name: param.grad.cpu() لكل name, param في model.named_parameters()}
    # pad the vocab grads here as well, to mirror write_model
    wte_grad = grads["transformer.wte.weight"] # (V, C)
    wte_grad_padded = pad_vocab(wte_grad, قدر=0) # (Vp, C) # TODO later maybe pad with nan?
    grads["transformer.wte.weight"] = wte_grad_padded # (Vp, C)
    چھاپو(f"padded vocab size in reference grads from {wte_grad.size(0)} to {wte_grad_padded.size(0)}")
    نال open(filename, "wb") كـ file:
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

تعریف write_tokenizer(enc, filename):
    n = enc.max_token_قدر + 1
    سرخی = torch.zeros(256, dtype=torch.int32)
    سرخی[0] = 20240328 # magic
    سرخی[1] = 2 # tokenizer version = 2 (1 -> 2: includes EOT token)
    سرخی[2] = n # number of tokens
    سرخی[3] = enc.eot_token # EOT token
    نال open(filename, "wb") كـ file:
        file.اكتب(سرخی.numpy().tobytes())
        لكل i في سلسلہ(n):
            b = enc.decode_بائٹاں([i])
            لمائی = لمبائی(b)
            أكد لمائی < 256, f"Token length exceeds 255: {length}"
            file.اكتب(struct.pack("<B", لمائی))  # Write the length as a 1-byte unsigned integer
            file.اكتب(b)  # Write the actual bytes
    چھاپو(f"wrote {filename}")

# -----------------------------------------------------------------------------
# int main

تعریف print0(*دلیلاں, **kwargs):
    # modified print that only prints from the master process
    # if this is not a distributed run, it's just a print
    إذا صحيح(os.environ.get("RANK", 0)) == 0:
        چھاپو(*دلیلاں, **kwargs)

إذا __name__ == "__main__":
    استورد time
    استورد argparse
    استورد tiktoken
    print0(f"Running pytorch {torch.version.__version__}")

    # default settings will overfit a tiny batch of data
    # and save model weights and debug state to disk on the first iteration
    parser = argparse.دلیل_Parser()
    # file system input / output
    parser.جوڑو_دلیل("--input_bin", type=str, افتراضي="dev/data/tinyshakespeare/tiny_shakespeare_val.bin", help="input .bin to train on")
    parser.جوڑو_دلیل("--input_val_bin", type=str, افتراضي="", help="input .bin to eval validation loss on")
    parser.جوڑو_دلیل("--output_dir", type=str, افتراضي="", help="output directory to which to write logs and checkpoints")
    parser.جوڑو_دلیل("--model", type=str, افتراضي="gpt2", help="gpt2|gpt2-medium|gpt2-large|gpt2-xl|d12|d24|d36|d48")
    # token layout for each step of the optimization
    parser.جوڑو_دلیل("--batch_size", type=صحيح, افتراضي=4, help="batch size, in units of #batch dimensions")
    parser.جوڑو_دلیل("--sequence_length", type=صحيح, افتراضي=64, help="sequence length")
    parser.جوڑو_دلیل("--total_batch_size", type=صحيح, افتراضي=256, help="total desired batch size, in units of #tokens")
    # workload (number of steps)
    parser.جوڑو_دلیل("--num_iterations", type=صحيح, افتراضي=10, help="number of iterations to run")
    parser.جوڑو_دلیل("--inference_only", type=صحيح, افتراضي=0, help="only run inference")
    # optimization
    parser.جوڑو_دلیل("--learning_rate", type=عائم, افتراضي=1e-4, help="learning rate warmup iterations")
    parser.جوڑو_دلیل("--warmup_iters", type=صحيح, افتراضي=0, help="learning rate warmup iterations")
    parser.جوڑو_دلیل("--learning_rate_decay_frac", type=عائم, افتراضي=1.0, help="learning rate warmup iterations")
    parser.جوڑو_دلیل("--weight_decay", type=عائم, افتراضي=0.0, help="weight decay")
    parser.جوڑو_دلیل("--grad_clip", type=عائم, افتراضي=1.0, help="maximum gradient magnitude")
    # evaluation
    parser.جوڑو_دلیل("--val_loss_every", type=صحيح, افتراضي=0, help="every how mant steps to evaluate val loss?")
    parser.جوڑو_دلیل("--val_max_steps", type=صحيح, افتراضي=20, help="how many batches of val to average?")
    parser.جوڑو_دلیل("--sample_every", type=صحيح, افتراضي=0, help="how often to sample from the model?")
    # debugging
    parser.جوڑو_دلیل("--overfit_single_batch", type=صحيح, افتراضي=1, help="overfit just one batch of data")
    # numerics
    parser.جوڑو_دلیل("--tensorcores", type=صحيح, افتراضي=0, help="use tensorcores")
    # memory management
    parser.جوڑو_دلیل("--device", type=str, افتراضي="", help="by default we autodetect, or set it here")
    parser.جوڑو_دلیل("--compile", type=صحيح, افتراضي=0, help="torch.compile the model")
    parser.جوڑو_دلیل("--flash", type=صحيح, افتراضي=0, help="use flash attention")
    parser.جوڑو_دلیل("--dtype", type=str, افتراضي="float32", help="float32|float16|bfloat16")
    parser.جوڑو_دلیل("--zero_stage", type=صحيح, افتراضي=0, help="zero redundancy optimizer stage (0/1/2/3)")
    # python -> C bridge
    parser.جوڑو_دلیل("--write_tensors", type=صحيح, افتراضي=1, help="write tensors to disk")
    دلیلاں = parser.parse_دلیلاں()

    # args error checking and convenience variables
    B, T = دلیلاں.batch_ناپ, دلیلاں.sequence_لمائی
    أكد 1 <= T <= 1024
    أكد دلیلاں.dtype في {"float32", "float16", "bfloat16"}
    أكد دلیلاں.model في {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "d12", "d24", "d36", "d48"}

    # set up DDP (distributed data parallel). torchrun sets this env variable
    ddp = صحيح(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    إذا ddp:
        # use of DDP atm demands CUDA, we set the device appropriately according to rank
        أكد torch.cuda.is_دستیاب(), "for now i think we need CUDA for DDP"
        init_process_گروہ(backend='nccl')
        ddp_rank = صحيح(os.environ['RANK'])
        ddp_local_rank = صحيح(os.environ['LOCAL_RANK'])
        ddp_world_ناپ = صحيح(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.سیٹ_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        بیج_وتھ = 0 # each process gets the exact same seed
        zero_stage = دلیلاں.zero_stage
    وإلا:
        ddp_rank = 0
        ddp_local_rank = 0
        zero_stage = 0
        ddp_world_ناپ = 1
        master_process = سچ
        بیج_وتھ = 0
        # select the device
        إذا دلیلاں.device:
            # provided explicitly by the user
            device = دلیلاں.device
        وإلا:
            # attempt to autodetect the device
            device = "cpu"
            إذا torch.cuda.is_دستیاب():
                device = "cuda"
            وإلاإذا hasattr(torch.backends, "mps") و torch.backends.mps.is_دستیاب():
                device = "mps"
    چھاپو(f"using device: {device}")
    device_type = 'cuda' إذا 'cuda' في device وإلا 'cpu'

    # calculate gradient accumulation from the desired total batch size and the current run configuration
    ٹوکن_فی_fwdbwd = B * T * ddp_world_ناپ
    أكد دلیلاں.کل_batch_ناپ % ٹوکن_فی_fwdbwd == 0
    grad_accum_قدماں = دلیلاں.کل_batch_ناپ // ٹوکن_فی_fwdbwd
    print0(f"total desired batch size: {args.total_batch_size}")
    print0(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    # set up a context manager following the desired dtype and device
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[دلیلاں.dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) إذا device_type == "cuda" وإلا nullcontext()

    # rng / reproducibility
    torch.manual_بیج(42)
    إذا torch.cuda.is_دستیاب():
        torch.cuda.manual_بیج(42)

    # set the torch precision mode to use TensorFloat32 (TF32) for matmuls
    # docs https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    إذا دلیلاں.tensorcores:
        torch.سیٹ_float32_matmul_precision('high')

    # turn on/off flash attention
    أكد دلیلاں.flash في {0, 1}
    FLASH = دلیلاں.flash

    # init (and write) the tokenizer
    enc = tiktoken.get_encoding("gpt2")
    إذا master_process و دلیلاں.write_tensors: # tokenizer is technically not tensors but ok
        write_tokenizer(enc, "gpt2_tokenizer.bin")

    # init the model, either from scratch or from OpenAI pretrained checkpoint
    إذا دلیلاں.model[0] == "d":
        # from scratch (random weights)
        model_config = {
            "d12": GPTConfig(بلاک_ناپ=1024, vocab_ناپ=50257, n_layer=12, n_سر=12, n_embd=768),
            "d24": GPTConfig(بلاک_ناپ=1024, vocab_ناپ=50257, n_layer=24, n_سر=16, n_embd=1024),
            "d36": GPTConfig(بلاک_ناپ=1024, vocab_ناپ=50257, n_layer=36, n_سر=20, n_embd=1280),
            "d48": GPTConfig(بلاک_ناپ=1024, vocab_ناپ=50257, n_layer=48, n_سر=25, n_embd=1600),
        }[دلیلاں.model]
        model = GPT(model_config)
    وإلا:
        # load the GPT-2 model weights
        model = GPT.from_pretrained(دلیلاں.model)
    model.train()
    model.منزل(device)
    إذا دلیلاں.compile:
        إذا hasattr(config, "coordinate_descent_tuning"):
            config.coordinate_descent_tuning = سچ # suggested by @Chillee
        print0("compiling the model...")
        model = torch.compile(model)

    # -------------------------------------------------------------------------
    # Our own version of a simple DistributedDataLoader

    # load tokens
    train_loader = Distributed_انکڑے_Loader(دلیلاں.داخلہ_bin, B, T, ddp_rank, ddp_world_ناپ)
    val_loader = کجھ_نئیں
    إذا دلیلاں.داخلہ_val_bin:
        val_loader = Distributed_انکڑے_Loader(دلیلاں.داخلہ_val_bin, B, T, ddp_rank, ddp_world_ناپ)

    # -------------------------------------------------------------------------
    # PyTorch -> C bridge: save some weights and state for C to load later as reference

    # do one forward pass to generate ground truth for our C tests
    إذا master_process و دلیلاں.write_tensors و (ليس دلیلاں.inference_only):
        x, y = train_loader.اگلا_batch()
        x, y = x.منزل(device), y.منزل(device)
        logits, loss = model(x, y)
        loss.backward()
        # save model params, in both float32 and bfloat16
        model_منزل_ناپ = {"gpt2": "124M", "gpt2-medium": "355M", "gpt2-large": "774M", "gpt2-xl": "1558M"}
        model_منزل_ناپ.update({f"d{d}": f"d{d}" لكل d في [12, 24, 36, 48]})
        model_ناپ_str = model_منزل_ناپ[دلیلاں.model] # e.g. "124M", or "d12"
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
    optimizer = raw_model.configure_optimizers(بھار_decay=دلیلاں.بھار_decay,
                                               learning_شرح=دلیلاں.learning_شرح, betas=(0.9, 0.95),
                                               device_type=device, zero_stage=zero_stage)

    # learning rate decay scheduler (cosine with warmup)
    تعریف get_lr(it):
        min_lr = دلیلاں.learning_شرح * دلیلاں.learning_شرح_decay_frac
        # 1) linear warmup for warmup_iters steps
        إذا it < دلیلاں.warmup_iters:
            أرجع دلیلاں.learning_شرح * (it+1) / دلیلاں.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        إذا it > دلیلاں.عدد_iterations:
            أرجع min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_تناسب = (it - دلیلاں.warmup_iters) / (دلیلاں.عدد_iterations - دلیلاں.warmup_iters)
        أكد 0 <= decay_تناسب <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_تناسب)) # coeff starts at 1 and goes to 0
        أرجع min_lr + coeff * (دلیلاں.learning_شرح - min_lr)

    # create the logging directory if it does not exist
    logfile = کجھ_نئیں
    إذا دلیلاں.output_dir:
        os.makedirs(دلیلاں.output_dir, exist_ok=سچ)
        logfile = os.رستہ.join(دلیلاں.output_dir, "main.log")
        # create the log file "main.log" inside it, and wipe it clean
        نال open(logfile, "w") كـ f:
            لنگھو

    إذا device == "cuda":
        torch.cuda.reset_peak_حافظہ_stats()
    timings = []
    norm = -1.0   # dummy value to print in inference-only mode
    لكل قدم في سلسلہ(دلیلاں.عدد_iterations + 1):
        t0 = time.time()
        آخری_قدم = (قدم == دلیلاں.عدد_iterations)

        # once in a while evaluate the validation dataset
        إذا (دلیلاں.val_loss_every > 0 \
            و (قدم % دلیلاں.val_loss_every == 0 أو آخری_قدم)) \
            و (val_loader اے ليس کجھ_نئیں):
            model.eval()
            val_loader.reset()
            نال torch.no_grad():
                val_loss = 0.0
                لكل _ في سلسلہ(دلیلاں.val_max_قدماں):
                    x, y = val_loader.اگلا_batch()
                    x, y = x.منزل(device), y.منزل(device)
                    _, loss = model(x, y, return_logits=خطأ)
                    val_loss += loss.شے()
                val_loss /= دلیلاں.val_max_قدماں
            # log to console and to file
            print0(f"val loss {val_loss}")
            إذا master_process و logfile اے ليس کجھ_نئیں:
                نال open(logfile, "a") كـ f:
                    f.اكتب("s:%d tel:%f\n" % (قدم, val_loss))

        # once in a while perform model inference on the master process
        إذا (دلیلاں.sample_every > 0 \
            و (قدم % دلیلاں.sample_every == 0 أو آخری_قدم)) \
            و master_process:
            model.eval()
            # before we end, let's also do one round of inference
            # we'll kick off the generation with "<|endoftext|>", which designates the start of a new sequence
            شروع_ids = [enc.eot_token]
            xg = (torch.tensor(شروع_ids, dtype=torch.لما, device=device)[کجھ_نئیں, ...])
            max_new_ٹوکن = 32
            temperature = 1.0
            چوٹی_k = 40
            yg = raw_model.generate(xg, max_new_ٹوکن, temperature=temperature, چوٹی_k=چوٹی_k)
            print0('---------------')
            print0(enc.decode(yg[0].tolist()))
            print0('---------------')

        # bit confusing: we want to make sure to eval and sample on 0th iteration
        # but also after the very last iteration. so we loop for step <= num_iterations
        # instead of just < num_iterations (one extra due to <=), only to do
        # the validation/sampling one last time, and then we break right here as we're done.
        إذا آخری_قدم:
            اكسر

        # --------------- TRAINING SECTION BEGIN -----------------
        model.train()
        optimizer.zero_grad(سیٹ_منزل_none=سچ)
        # if we are trying to overfit a single batch, we reset the loader here
        إذا دلیلاں.overfit_single_batch:
            train_loader.reset()
        # micro-batch loop where we do gradient accumulation to reach desired total batch size
        lossf = 0.0 # for getting the mean loss (as simple float) over the accumulation steps
        لكل micro_قدم في سلسلہ(grad_accum_قدماں):
            # fetch a batch
            x, y = train_loader.اگلا_batch()
            x, y = x.منزل(device), y.منزل(device)
            إذا ddp:
                # we want only the last micro-step to sync grads in a DDP model
                # the official way to do this is with model.no_sync(), but that is a
                # context manager that bloats the code, so we just toggle this variable
                model.require_backward_grad_sync = (micro_قدم == grad_accum_قدماں - 1)
            # forward pass
            نال ctx:
                _, loss = model(x, y, return_logits=خطأ)
                # we have to scale the loss to account for gradient accumulation,
                # because the gradients just add on each successive backward().
                # addition of gradients corresponds to a SUM in the objective, but
                # instead of a SUM we want MEAN, so we scale the loss here
                loss = loss / grad_accum_قدماں
                lossf += loss.detach() # keep track of the mean loss
            # backward pass
            إذا ليس دلیلاں.inference_only:
                loss.backward()
        إذا ddp:
            dist.all_reduce(lossf, op=dist.ReduceOp.AVG)
        lossf = lossf.شے()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), دلیلاں.grad_clip)
        # determine and set the learning rate for this iteration
        lr = get_lr(قدم)
        لكل param_گروہ في optimizer.param_groups:
            param_گروہ['lr'] = lr
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
        ٹوکن_فی_دوجا = grad_accum_قدماں * ddp_world_ناپ * B * T / (t1-t0)
        print0(f"step {step+1:4d}/{args.num_iterations} | train loss {lossf:.6f} | norm {norm:.4f} | lr {lr:.2e} | ({(t1-t0)*1000:.2f} ms | {tokens_per_second:.0f} tok/s)")
        # log to logile
        إذا master_process و logfile اے ليس کجھ_نئیں:
            نال open(logfile, "a") كـ f:
                f.اكتب("s:%d trl:%f\n" % (قدم, lossf))

        # keep track of smooth timings, last 20 iterations
        إذا قدم > 0 و قدم > دلیلاں.عدد_iterations - 20:
            timings.append(t1-t0)

    # print the average of the last 20 timings, to get something smooth-ish
    timings = timings[-20:]
    print0(f"final {len(timings)} iters avg: {np.mean(timings)*1000:.3f}ms")
    print0(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

    # -------------------------------------------------------------------------
    # clean up nice
    إذا ddp:
        destroy_process_گروہ()
