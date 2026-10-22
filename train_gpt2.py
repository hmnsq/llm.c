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

درآمد os
درآمد math
درآمد glob
درآمد struct
درآمد inspect
سے contextlib درآمد nullcontext
سے dataclasses درآمد dataclass

درآمد numpy بطور np
درآمد torch
درآمد torch.nn بطور nn
سے torch.nn درآمد functional بطور F
درآمد torch._inductor.config بطور config
سے torch.nn.parallel درآمد Distributed_ڈیٹا_Parallel بطور DDP
سے torch.distributed درآمد init_process_گروہ, destroy_process_گروہ
سے torch.distributed.optim درآمد ZeroRedundancyOptimizer
درآمد torch.distributed بطور dist

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model

جماعت NewGELU(nn.Module):
    """Careful there are a few versions of GeLU, this one is the exact one used by OpenAI"""
    تعریف forward(self, داخلہ):
        واپس 0.5 * داخلہ * (1.0 + torch.tanh(math.جذر(2.0 / math.pi) * (داخلہ + 0.044715 * torch.pow(داخلہ, 3.0))))

# using a global to toggle flash-attention
FLASH = 0

جماعت CausalSelfAttention(nn.Module):

    تعریف __init__(self, config):
        بالا().__init__()
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
        self.رجسٹر_بفر("bias", torch.tril(torch.ones(config.بلاک_سائز, config.بلاک_سائز))
                                     .view(1, 1, config.بلاک_سائز, config.بلاک_سائز))

    تعریف forward(self, x):
        B, T, C = x.سائز() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.بانٹو(self.n_embd, dim=2)
        k = k.view(B, T, self.n_سر, C // self.n_سر).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_سر, C // self.n_سر).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_سر, C // self.n_سر).transpose(1, 2) # (B, nh, T, hs)
        اگر FLASH:
            # flashattention
            y = F.scaled_dot_حاصل_ضرب_attention(q, k, v, is_causal=سچ)
        ورنہ:
            # manual implementation of attention
            # this materializes the large (T,T) matrix for all the queries and keys
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.جذر(k.سائز(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, اعشاری('-inf'))
            att = F.softmax(att, dim=-1)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        واپس y

جماعت MLP(nn.Module):

    تعریف __init__(self, config):
        بالا().__init__()
        self.c_fc    = nn.خطی(config.n_embd, 4 * config.n_embd)
        self.gelu    = NewGELU()
        self.c_proj  = nn.خطی(4 * config.n_embd, config.n_embd)
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1

    تعریف forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        واپس x

جماعت بلاک(nn.Module):

    تعریف __init__(self, config):
        بالا().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    تعریف forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        واپس x

# -----------------------------------------------------------------------------
# The main GPT-2 model

@dataclass
جماعت GPTConfig:
    بلاک_سائز: صحیح = 1024
    vocab_سائز: صحیح = 50257
    n_layer: صحیح = 12
    n_سر: صحیح = 12
    n_embd: صحیح = 768

جماعت GPT(nn.Module):

    تعریف __init__(self, config):
        بالا().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_سائز, config.n_embd),
            wpe = nn.Embedding(config.بلاک_سائز, config.n_embd),
            h = nn.ModuleList([بلاک(config) برائے _ میں سلسلہ(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_سر = nn.خطی(config.n_embd, config.vocab_سائز, bias=جھوٹ)
        self.lm_سر.LLMC_SKIP_INIT = 1 # don't init this one, we will tie weights
        self.transformer.wte.وزن = self.lm_سر.وزن # https://paperswithcode.com/method/weight-tying

        # init all weights, use a torch rng object to be very careful
        self.init_rng = torch.Generator()
        self.init_rng.manual_بیج(42)
        self.apply(self._init_weights)

    تعریف _init_weights(self, module):
        اگر isinstance(module, nn.خطی):
            # apply special scaled init to the residual projections, per GPT-2 paper
            std = 0.02 اگر نہیں hasattr(module, 'LLMC_RESIDUAL_SCALE_FLAG') ورنہ 0.02/math.جذر(2 * self.config.n_layer)
            # we want to skip initializing lm_head, which shares parameters with wte
            # and wte was already initialized down below during the Embedding init
            اگر نہیں hasattr(module, 'LLMC_SKIP_INIT'):
                torch.nn.init.معمول(module.وزن, اوسط=0.0, std=std, generator=self.init_rng)
            اگر module.bias ہے نہیں کچھ_نہیں:
                torch.nn.init.zeros_(module.bias)
        وگرنہ_اگر isinstance(module, nn.Embedding):
            torch.nn.init.معمول(module.وزن, اوسط=0.0, std=0.02, generator=self.init_rng)

    تعریف forward(self, idx, targets=کچھ_نہیں, return_logits=سچ):
        device = idx.device
        b, t = idx.سائز()
        أكد t <= self.config.بلاک_سائز, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.طویل, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = tok_emb + pos_emb

        برائے بلاک_2 میں self.transformer.h:
            x = بلاک_2(x)
        x = self.transformer.ln_f(x)

        اگر targets ہے نہیں کچھ_نہیں:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_سر(x)
            loss = F.cross_entropy(logits.view(-1, logits.سائز(-1)), targets.view(-1), ignore_اشاریہ=-1)
        ورنہ:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_سر(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = کچھ_نہیں

        # there are performance reasons why not returning logits is prudent, if not needed
        اگر نہیں return_logits:
            logits = کچھ_نہیں

        واپس logits, loss

    @classmethod
    تعریف from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        أكد model_type میں {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        سے transformers درآمد GPT2LMHeadModel
        چھاپو("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_دلائل = {
            'gpt2':         dict(n_layer=12, n_سر=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_سر=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_سر=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_سر=25, n_embd=1600), # 1558M params
        }[model_type]
        config_دلائل['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_دلائل['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_دلائل)
        model = GPT(config)
        sd = model.state_dict()
        sd_کلیدیں = sd.کلیدیں()
        sd_کلیدیں = [k برائے k میں sd_کلیدیں اگر نہیں k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_کلیدیں_hf = sd_hf.کلیدیں()
        sd_کلیدیں_hf = [k برائے k میں sd_کلیدیں_hf اگر نہیں k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_کلیدیں_hf = [k برائے k میں sd_کلیدیں_hf اگر نہیں k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        أكد لمبائی(sd_کلیدیں_hf) == لمبائی(sd_کلیدیں), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        برائے k میں sd_کلیدیں_hf:
            اگر any(k.endswith(w) برائے w میں transposed):
                # special treatment for the Conv1D weights we need to transpose
                أكد sd_hf[k].shape[::-1] == sd[k].shape
                ساتھ torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            ورنہ:
                # vanilla copy over the other parameters
                أكد sd_hf[k].shape == sd[k].shape
                ساتھ torch.no_grad():
                    sd[k].copy_(sd_hf[k])

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

    @torch.no_grad()
    تعریف generate(self, idx, max_new_علامات, temperature=1.0, چوٹی_k=کچھ_نہیں):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        برائے _ میں سلسلہ(max_new_علامات):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx اگر idx.سائز(1) <= self.config.بلاک_سائز ورنہ idx[:, -self.config.بلاک_سائز:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            اگر چوٹی_k ہے نہیں کچھ_نہیں:
                v, _ = torch.topk(logits, min(چوٹی_k, logits.سائز(-1)))
                logits[logits < v[:, [-1]]] = -اعشاری('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_اگلا = torch.multinomial(probs, عدد_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_اگلا), dim=1)

        واپس idx

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

تعریف peek_ڈیٹا_shard(filename):
    # only reads the header, returns header data
    ساتھ open(filename, "rb") بطور f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        سرخی = np.frombuffer(f.اقرأ(256*4), dtype=np.int32)
    اگر سرخی[0] != 20240520:
        چھاپو("ERROR: magic number mismatch in the data .bin file!")
        چھاپو("---> HINT: Are you passing in a correct file with --input_bin?")
        چھاپو("---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README")
        چھاپو("---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try")
        exit(1)
    أكد سرخی[1] == 1, "unsupported version"
    ntok = سرخی[2] # number of tokens (claimed)
    واپس ntok # for now just return the number of tokens

تعریف بھرو_ڈیٹا_shard(filename):
    ساتھ open(filename, "rb") بطور f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        سرخی = np.frombuffer(f.اقرأ(256*4), dtype=np.int32)
        أكد سرخی[0] == 20240520, "magic number mismatch in the data .bin file"
        أكد سرخی[1] == 1, "unsupported version"
        ntok = سرخی[2] # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        علامات = np.frombuffer(f.اقرأ(), dtype=np.uint16)
    أكد لمبائی(علامات) == ntok, "number of tokens read does not match header?"
    واپس علامات

جماعت Distributed_ڈیٹا_Loader:
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
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.طویل)
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
    # writes the GPT-2 model's weights to a binary file
    أكد dtype میں {"float32", "bfloat16"}
    write_fun = write_fp32 اگر dtype == "float32" ورنہ write_bf16
    write_fun(model_tensors["transformer.wte.weight"], file) # (V, C)
    write_fun(model_tensors["transformer.wpe.weight"], file) # (T, C)
    برائے i میں سلسلہ(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_1.weight"], file)
    برائے i میں سلسلہ(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_1.bias"], file)
    برائے i میں سلسلہ(L): # (L, 3C, C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_attn.weight"], file)
    برائے i میں سلسلہ(L): # (L, 3C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_attn.bias"], file)
    برائے i میں سلسلہ(L): # (L, C, C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_proj.weight"], file)
    برائے i میں سلسلہ(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_proj.bias"], file)
    برائے i میں سلسلہ(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_2.weight"], file)
    برائے i میں سلسلہ(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_2.bias"], file)
    برائے i میں سلسلہ(L): # (L, 4C, C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_fc.weight"], file)
    برائے i میں سلسلہ(L): # (L, 4C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_fc.bias"], file)
    برائے i میں سلسلہ(L): # (L, C, 4C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_proj.weight"], file)
    برائے i میں سلسلہ(L): # (L, C)
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
    pad_صفیں = Vp - V
    padded = tensor اگر pad_صفیں == 0 ورنہ F.pad(tensor, (0, 0, 0, pad_صفیں), قدر=قدر)
    أكد padded.shape == (Vp, C)
    واپس padded

تعریف write_model(model, filename, dtype):
    # everything we need to instantiate the model
    # 1) header is: version int, GPTConfig ints, padding to 1024 bytes
    أكد dtype میں {"float32", "bfloat16"} # float16 todo maybe later
    نسخہ = {
        "float32": 3, # 3: all tensors are fp32, padded vocab
        "bfloat16": 5, # 5: all tensors are bf16, padded vocab
    }[dtype]
    سرخی = torch.zeros(256, dtype=torch.int32)
    سرخی[0] = 20240326 # magic
    سرخی[1] = نسخہ # checkpoint version
    سرخی[2] = model.config.بلاک_سائز
    سرخی[3] = model.config.vocab_سائز
    سرخی[4] = model.config.n_layer
    سرخی[5] = model.config.n_سر
    سرخی[6] = model.config.n_embd
    # 2) the parameters follow the header
    params = {name: param.cpu() برائے name, param میں model.named_parameters()}
    # pad the vocab to a multiple of 128 here at export, for efficiency in C
    wte = params["transformer.wte.weight"] # (V, C)
    wte_padded = pad_vocab(wte) # (Vp, C)
    params["transformer.wte.weight"] = wte_padded # (Vp, C)
    چھاپو(f"padded vocab size from {wte.size(0)} to {wte_padded.size(0)}")
    سرخی[7] = wte_padded.سائز(0) # padded vocab size store in header
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
    سرخی[0] = 20240327 # magic
    سرخی[1] = 2 # run state version = 2 (1 -> 2 for padded vocab changes)
    سرخی[2] = x.سائز(0) # batch size of the batch, B
    سرخی[3] = x.سائز(1) # temporal extent of the batch, T
    grads = {name: param.grad.cpu() برائے name, param میں model.named_parameters()}
    # pad the vocab grads here as well, to mirror write_model
    wte_grad = grads["transformer.wte.weight"] # (V, C)
    wte_grad_padded = pad_vocab(wte_grad, قدر=0) # (Vp, C) # TODO later maybe pad with nan?
    grads["transformer.wte.weight"] = wte_grad_padded # (Vp, C)
    چھاپو(f"padded vocab size in reference grads from {wte_grad.size(0)} to {wte_grad_padded.size(0)}")
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

تعریف write_tokenizer(enc, filename):
    n = enc.max_token_قدر + 1
    سرخی = torch.zeros(256, dtype=torch.int32)
    سرخی[0] = 20240328 # magic
    سرخی[1] = 2 # tokenizer version = 2 (1 -> 2: includes EOT token)
    سرخی[2] = n # number of tokens
    سرخی[3] = enc.eot_token # EOT token
    ساتھ open(filename, "wb") بطور file:
        file.اكتب(سرخی.numpy().tobytes())
        برائے i میں سلسلہ(n):
            b = enc.decode_بائٹیں([i])
            لمبائی_2 = لمبائی(b)
            أكد لمبائی_2 < 256, f"Token length exceeds 255: {length}"
            file.اكتب(struct.pack("<B", لمبائی_2))  # Write the length as a 1-byte unsigned integer
            file.اكتب(b)  # Write the actual bytes
    چھاپو(f"wrote {filename}")

# -----------------------------------------------------------------------------
# int main

تعریف print0(*دلائل, **kwargs):
    # modified print that only prints from the master process
    # if this is not a distributed run, it's just a print
    اگر صحیح(os.environ.get("RANK", 0)) == 0:
        چھاپو(*دلائل, **kwargs)

اگر __name__ == "__main__":
    درآمد time
    درآمد argparse
    درآمد tiktoken
    print0(f"Running pytorch {torch.version.__version__}")

    # default settings will overfit a tiny batch of data
    # and save model weights and debug state to disk on the first iteration
    parser = argparse.دلیل_Parser()
    # file system input / output
    parser.جوڑو_دلیل("--input_bin", type=str, طےشدہ="dev/data/tinyshakespeare/tiny_shakespeare_val.bin", help="input .bin to train on")
    parser.جوڑو_دلیل("--input_val_bin", type=str, طےشدہ="", help="input .bin to eval validation loss on")
    parser.جوڑو_دلیل("--output_dir", type=str, طےشدہ="", help="output directory to which to write logs and checkpoints")
    parser.جوڑو_دلیل("--model", type=str, طےشدہ="gpt2", help="gpt2|gpt2-medium|gpt2-large|gpt2-xl|d12|d24|d36|d48")
    # token layout for each step of the optimization
    parser.جوڑو_دلیل("--batch_size", type=صحیح, طےشدہ=4, help="batch size, in units of #batch dimensions")
    parser.جوڑو_دلیل("--sequence_length", type=صحیح, طےشدہ=64, help="sequence length")
    parser.جوڑو_دلیل("--total_batch_size", type=صحیح, طےشدہ=256, help="total desired batch size, in units of #tokens")
    # workload (number of steps)
    parser.جوڑو_دلیل("--num_iterations", type=صحیح, طےشدہ=10, help="number of iterations to run")
    parser.جوڑو_دلیل("--inference_only", type=صحیح, طےشدہ=0, help="only run inference")
    # optimization
    parser.جوڑو_دلیل("--learning_rate", type=اعشاری, طےشدہ=1e-4, help="learning rate warmup iterations")
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
    parser.جوڑو_دلیل("--flash", type=صحیح, طےشدہ=0, help="use flash attention")
    parser.جوڑو_دلیل("--dtype", type=str, طےشدہ="float32", help="float32|float16|bfloat16")
    parser.جوڑو_دلیل("--zero_stage", type=صحیح, طےشدہ=0, help="zero redundancy optimizer stage (0/1/2/3)")
    # python -> C bridge
    parser.جوڑو_دلیل("--write_tensors", type=صحیح, طےشدہ=1, help="write tensors to disk")
    دلائل = parser.parse_دلائل()

    # args error checking and convenience variables
    B, T = دلائل.batch_سائز, دلائل.sequence_لمبائی
    أكد 1 <= T <= 1024
    أكد دلائل.dtype میں {"float32", "float16", "bfloat16"}
    أكد دلائل.model میں {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "d12", "d24", "d36", "d48"}

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
    چھاپو(f"using device: {device}")
    device_type = 'cuda' اگر 'cuda' میں device ورنہ 'cpu'

    # calculate gradient accumulation from the desired total batch size and the current run configuration
    علامات_فی_fwdbwd = B * T * ddp_world_سائز
    أكد دلائل.کل_batch_سائز % علامات_فی_fwdbwd == 0
    grad_accum_اقدام = دلائل.کل_batch_سائز // علامات_فی_fwdbwd
    print0(f"total desired batch size: {args.total_batch_size}")
    print0(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    # set up a context manager following the desired dtype and device
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[دلائل.dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) اگر device_type == "cuda" ورنہ nullcontext()

    # rng / reproducibility
    torch.manual_بیج(42)
    اگر torch.cuda.is_دستیاب():
        torch.cuda.manual_بیج(42)

    # set the torch precision mode to use TensorFloat32 (TF32) for matmuls
    # docs https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    اگر دلائل.tensorcores:
        torch.سیٹ_float32_matmul_precision('high')

    # turn on/off flash attention
    أكد دلائل.flash میں {0, 1}
    FLASH = دلائل.flash

    # init (and write) the tokenizer
    enc = tiktoken.get_encoding("gpt2")
    اگر master_process اور دلائل.write_tensors: # tokenizer is technically not tensors but ok
        write_tokenizer(enc, "gpt2_tokenizer.bin")

    # init the model, either from scratch or from OpenAI pretrained checkpoint
    اگر دلائل.model[0] == "d":
        # from scratch (random weights)
        model_config = {
            "d12": GPTConfig(بلاک_سائز=1024, vocab_سائز=50257, n_layer=12, n_سر=12, n_embd=768),
            "d24": GPTConfig(بلاک_سائز=1024, vocab_سائز=50257, n_layer=24, n_سر=16, n_embd=1024),
            "d36": GPTConfig(بلاک_سائز=1024, vocab_سائز=50257, n_layer=36, n_سر=20, n_embd=1280),
            "d48": GPTConfig(بلاک_سائز=1024, vocab_سائز=50257, n_layer=48, n_سر=25, n_embd=1600),
        }[دلائل.model]
        model = GPT(model_config)
    ورنہ:
        # load the GPT-2 model weights
        model = GPT.from_pretrained(دلائل.model)
    model.train()
    model.مقصد(device)
    اگر دلائل.compile:
        اگر hasattr(config, "coordinate_descent_tuning"):
            config.coordinate_descent_tuning = سچ # suggested by @Chillee
        print0("compiling the model...")
        model = torch.compile(model)

    # -------------------------------------------------------------------------
    # Our own version of a simple DistributedDataLoader

    # load tokens
    train_loader = Distributed_ڈیٹا_Loader(دلائل.داخلہ_bin, B, T, ddp_rank, ddp_world_سائز)
    val_loader = کچھ_نہیں
    اگر دلائل.داخلہ_val_bin:
        val_loader = Distributed_ڈیٹا_Loader(دلائل.داخلہ_val_bin, B, T, ddp_rank, ddp_world_سائز)

    # -------------------------------------------------------------------------
    # PyTorch -> C bridge: save some weights and state for C to load later as reference

    # do one forward pass to generate ground truth for our C tests
    اگر master_process اور دلائل.write_tensors اور (نہیں دلائل.inference_only):
        x, y = train_loader.اگلا_batch()
        x, y = x.مقصد(device), y.مقصد(device)
        logits, loss = model(x, y)
        loss.backward()
        # save model params, in both float32 and bfloat16
        model_مقصد_سائز = {"gpt2": "124M", "gpt2-medium": "355M", "gpt2-large": "774M", "gpt2-xl": "1558M"}
        model_مقصد_سائز.update({f"d{d}": f"d{d}" برائے d میں [12, 24, 36, 48]})
        model_سائز_str = model_مقصد_سائز[دلائل.model] # e.g. "124M", or "d12"
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

    # create the logging directory if it does not exist
    logfile = کچھ_نہیں
    اگر دلائل.output_dir:
        os.makedirs(دلائل.output_dir, exist_ok=سچ)
        logfile = os.راستہ.join(دلائل.output_dir, "main.log")
        # create the log file "main.log" inside it, and wipe it clean
        ساتھ open(logfile, "w") بطور f:
            گزرو

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
            # before we end, let's also do one round of inference
            # we'll kick off the generation with "<|endoftext|>", which designates the start of a new sequence
            آغاز_ids = [enc.eot_token]
            xg = (torch.tensor(آغاز_ids, dtype=torch.طویل, device=device)[کچھ_نہیں, ...])
            max_new_علامات = 32
            temperature = 1.0
            چوٹی_k = 40
            yg = raw_model.generate(xg, max_new_علامات, temperature=temperature, چوٹی_k=چوٹی_k)
            print0('---------------')
            print0(enc.decode(yg[0].tolist()))
            print0('---------------')

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
