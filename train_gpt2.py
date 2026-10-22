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

దిగుమతి os
దిగుమతి math
దిగుమతి glob
దిగుమతి struct
దిగుమతి inspect
తః contextlib దిగుమతి nullcontext
తః dataclasses దిగుమతి dataclass

దిగుమతి numpy ఇవ np
దిగుమతి torch
దిగుమతి torch.nn ఇవ nn
తః torch.nn దిగుమతి functional ఇవ F
దిగుమతి torch._inductor.config ఇవ config
తః torch.nn.parallel దిగుమతి Distributed_దత్తాంశం_Parallel ఇవ DDP
తః torch.distributed దిగుమతి init_process_సమూహం, destroy_process_సమూహం
తః torch.distributed.optim దిగుమతి ZeroRedundancyOptimizer
దిగుమతి torch.distributed ఇవ dist

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model

తరగతి NewGELU(nn.Module):
    """Careful there are a few versions of GeLU, this one is the exact one used by OpenAI"""
    నిర్వచనం forward(self, ఆదానం):
        తిరిగి 0.5 * ఆదానం * (1.0 + torch.tanh(math.వర్గమూలం(2.0 / math.pi) * (ఆదానం + 0.044715 * torch.pow(ఆదానం, 3.0))))

# using a global to toggle flash-attention
FLASH = 0

తరగతి CausalSelfAttention(nn.Module):

    నిర్వచనం __init__(self, config):
        పర().__init__()
        ప్రతిజ్ఞా config.n_embd % config.n_తల == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.రేఖీయ(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.రేఖీయ(config.n_embd, config.n_embd)
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1
        # regularization
        self.n_తల = config.n_తల
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.రిజిస్టరు_బఫరు("bias", torch.tril(torch.ones(config.ఖండం_కొలత, config.ఖండం_కొలత))
                                     .view(1, 1, config.ఖండం_కొలత, config.ఖండం_కొలత))

    నిర్వచనం forward(self, x):
        B, T, C = x.కొలత() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.విభజించు(self.n_embd, dim=2)
        k = k.view(B, T, self.n_తల, C // self.n_తల).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_తల, C // self.n_తల).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_తల, C // self.n_తల).transpose(1, 2) # (B, nh, T, hs)
        ఒకవేళ FLASH:
            # flashattention
            y = F.scaled_dot_లబ్ధం_attention(q, k, v, is_causal=నిజం)
        లేకపోతే:
            # manual implementation of attention
            # this materializes the large (T,T) matrix for all the queries and keys
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.వర్గమూలం(k.కొలత(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, దశాంశం('-inf'))
            att = F.softmax(att, dim=-1)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        తిరిగి y

తరగతి MLP(nn.Module):

    నిర్వచనం __init__(self, config):
        పర().__init__()
        self.c_fc    = nn.రేఖీయ(config.n_embd, 4 * config.n_embd)
        self.gelu    = NewGELU()
        self.c_proj  = nn.రేఖీయ(4 * config.n_embd, config.n_embd)
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1

    నిర్వచనం forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        తిరిగి x

తరగతి ఖండం(nn.Module):

    నిర్వచనం __init__(self, config):
        పర().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    నిర్వచనం forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        తిరిగి x

# -----------------------------------------------------------------------------
# The main GPT-2 model

@dataclass
తరగతి GPTConfig:
    ఖండం_కొలత: పూర్ణాంకం = 1024
    vocab_కొలత: పూర్ణాంకం = 50257
    n_layer: పూర్ణాంకం = 12
    n_తల: పూర్ణాంకం = 12
    n_embd: పూర్ణాంకం = 768

తరగతి GPT(nn.Module):

    నిర్వచనం __init__(self, config):
        పర().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_కొలత, config.n_embd),
            wpe = nn.Embedding(config.ఖండం_కొలత, config.n_embd),
            h = nn.ModuleList([ఖండం(config) కొరకు _ లో శ్రేణి(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_తల = nn.రేఖీయ(config.n_embd, config.vocab_కొలత, bias=అబద్ధం)
        self.lm_తల.LLMC_SKIP_INIT = 1 # don't init this one, we will tie weights
        self.transformer.wte.బరువు = self.lm_తల.బరువు # https://paperswithcode.com/method/weight-tying

        # init all weights, use a torch rng object to be very careful
        self.init_rng = torch.Generator()
        self.init_rng.manual_విత్తనం(42)
        self.apply(self._init_weights)

    నిర్వచనం _init_weights(self, module):
        ఒకవేళ isinstance(module, nn.రేఖీయ):
            # apply special scaled init to the residual projections, per GPT-2 paper
            std = 0.02 ఒకవేళ కాదు hasattr(module, 'LLMC_RESIDUAL_SCALE_FLAG') లేకపోతే 0.02/math.వర్గమూలం(2 * self.config.n_layer)
            # we want to skip initializing lm_head, which shares parameters with wte
            # and wte was already initialized down below during the Embedding init
            ఒకవేళ కాదు hasattr(module, 'LLMC_SKIP_INIT'):
                torch.nn.init.సాధారణ(module.బరువు, సగటు=0.0, std=std, generator=self.init_rng)
            ఒకవేళ module.bias అయిన కాదు ఏమీలేదు:
                torch.nn.init.zeros_(module.bias)
        లేదావేళ isinstance(module, nn.Embedding):
            torch.nn.init.సాధారణ(module.బరువు, సగటు=0.0, std=0.02, generator=self.init_rng)

    నిర్వచనం forward(self, idx, targets=ఏమీలేదు, return_logits=నిజం):
        device = idx.device
        b, t = idx.కొలత()
        ప్రతిజ్ఞా t <= self.config.ఖండం_కొలత, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.దీర్ఘ, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = tok_emb + pos_emb

        కొరకు ఖండం_2 లో self.transformer.h:
            x = ఖండం_2(x)
        x = self.transformer.ln_f(x)

        ఒకవేళ targets అయిన కాదు ఏమీలేదు:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_తల(x)
            loss = F.cross_entropy(logits.view(-1, logits.కొలత(-1)), targets.view(-1), ignore_సూచీ=-1)
        లేకపోతే:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_తల(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = ఏమీలేదు

        # there are performance reasons why not returning logits is prudent, if not needed
        ఒకవేళ కాదు return_logits:
            logits = ఏమీలేదు

        తిరిగి logits, loss

    @classmethod
    నిర్వచనం from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        ప్రతిజ్ఞా model_type లో {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        తః transformers దిగుమతి GPT2LMHeadModel
        ముద్రించు("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_వాదనలు = {
            'gpt2':         dict(n_layer=12, n_తల=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_తల=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_తల=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_తల=25, n_embd=1600), # 1558M params
        }[model_type]
        config_వాదనలు['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_వాదనలు['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_వాదనలు)
        model = GPT(config)
        sd = model.state_dict()
        sd_కీలకాలు = sd.కీలకాలు()
        sd_కీలకాలు = [k కొరకు k లో sd_కీలకాలు ఒకవేళ కాదు k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_కీలకాలు_hf = sd_hf.కీలకాలు()
        sd_కీలకాలు_hf = [k కొరకు k లో sd_కీలకాలు_hf ఒకవేళ కాదు k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_కీలకాలు_hf = [k కొరకు k లో sd_కీలకాలు_hf ఒకవేళ కాదు k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        ప్రతిజ్ఞా పొడవు(sd_కీలకాలు_hf) == పొడవు(sd_కీలకాలు), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        కొరకు k లో sd_కీలకాలు_hf:
            ఒకవేళ any(k.endswith(w) కొరకు w లో transposed):
                # special treatment for the Conv1D weights we need to transpose
                ప్రతిజ్ఞా sd_hf[k].shape[::-1] == sd[k].shape
                తోడుగా torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            లేకపోతే:
                # vanilla copy over the other parameters
                ప్రతిజ్ఞా sd_hf[k].shape == sd[k].shape
                తోడుగా torch.no_grad():
                    sd[k].copy_(sd_hf[k])

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

    @torch.no_grad()
    నిర్వచనం generate(self, idx, max_new_టోకెన్లు, temperature=1.0, శిఖరం_k=ఏమీలేదు):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        కొరకు _ లో శ్రేణి(max_new_టోకెన్లు):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx ఒకవేళ idx.కొలత(1) <= self.config.ఖండం_కొలత లేకపోతే idx[:, -self.config.ఖండం_కొలత:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            ఒకవేళ శిఖరం_k అయిన కాదు ఏమీలేదు:
                v, _ = torch.topk(logits, min(శిఖరం_k, logits.కొలత(-1)))
                logits[logits < v[:, [-1]]] = -దశాంశం('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_తదుపరి = torch.multinomial(probs, అంకె_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_తదుపరి), dim=1)

        తిరిగి idx

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

నిర్వచనం peek_దత్తాంశం_shard(filename):
    # only reads the header, returns header data
    తోడుగా open(filename, "rb") ఇవ f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        శీర్షిక = np.frombuffer(f.పఠ(256*4), dtype=np.int32)
    ఒకవేళ శీర్షిక[0] != 20240520:
        ముద్రించు("ERROR: magic number mismatch in the data .bin file!")
        ముద్రించు("---> HINT: Are you passing in a correct file with --input_bin?")
        ముద్రించు("---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README")
        ముద్రించు("---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try")
        exit(1)
    ప్రతిజ్ఞా శీర్షిక[1] == 1, "unsupported version"
    ntok = శీర్షిక[2] # number of tokens (claimed)
    తిరిగి ntok # for now just return the number of tokens

నిర్వచనం ఎక్కించు_దత్తాంశం_shard(filename):
    తోడుగా open(filename, "rb") ఇవ f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        శీర్షిక = np.frombuffer(f.పఠ(256*4), dtype=np.int32)
        ప్రతిజ్ఞా శీర్షిక[0] == 20240520, "magic number mismatch in the data .bin file"
        ప్రతిజ్ఞా శీర్షిక[1] == 1, "unsupported version"
        ntok = శీర్షిక[2] # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        టోకెన్లు = np.frombuffer(f.పఠ(), dtype=np.uint16)
    ప్రతిజ్ఞా పొడవు(టోకెన్లు) == ntok, "number of tokens read does not match header?"
    తిరిగి టోకెన్లు

తరగతి Distributed_దత్తాంశం_Loader:
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
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.దీర్ఘ)
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
    # writes the GPT-2 model's weights to a binary file
    ప్రతిజ్ఞా dtype లో {"float32", "bfloat16"}
    write_fun = write_fp32 ఒకవేళ dtype == "float32" లేకపోతే write_bf16
    write_fun(model_tensors["transformer.wte.weight"], file) # (V, C)
    write_fun(model_tensors["transformer.wpe.weight"], file) # (T, C)
    కొరకు i లో శ్రేణి(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_1.weight"], file)
    కొరకు i లో శ్రేణి(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_1.bias"], file)
    కొరకు i లో శ్రేణి(L): # (L, 3C, C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_attn.weight"], file)
    కొరకు i లో శ్రేణి(L): # (L, 3C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_attn.bias"], file)
    కొరకు i లో శ్రేణి(L): # (L, C, C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_proj.weight"], file)
    కొరకు i లో శ్రేణి(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_proj.bias"], file)
    కొరకు i లో శ్రేణి(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_2.weight"], file)
    కొరకు i లో శ్రేణి(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_2.bias"], file)
    కొరకు i లో శ్రేణి(L): # (L, 4C, C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_fc.weight"], file)
    కొరకు i లో శ్రేణి(L): # (L, 4C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_fc.bias"], file)
    కొరకు i లో శ్రేణి(L): # (L, C, 4C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_proj.weight"], file)
    కొరకు i లో శ్రేణి(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_proj.bias"], file)
    write_fun(model_tensors["transformer.ln_f.weight"], file) # (C, )
    write_fun(model_tensors["transformer.ln_f.bias"], file) # (C, )

@torch.no_grad()
నిర్వచనం pad_vocab(tensor, గుణిజం=128, విలువ=0):
    """
    The dimension of the vocab size in GPT-2 is 50,257
    which is unfortunately a very unfriendly number for a lot of
    matrix operations on the GPU. So we pad it to the nearest
    friendlier multiple, e.g. 50,304 if multiple=128 when we
    export the weights into C land. This is a NOOP algorithmically
    and is only done to make the tensor operations more efficient.
    """
    ప్రతిజ్ఞా tensor.ndim == 2
    V, C = tensor.shape
    ప్రతిజ్ఞా V == 50257, "just being defensive here"
    # calculate padded vocab size by rounding up to nearest multiple
    Vp = ((V + గుణిజం - 1) // గుణిజం) * గుణిజం
    # pad the tensor
    pad_అడ్డువరుసలు = Vp - V
    padded = tensor ఒకవేళ pad_అడ్డువరుసలు == 0 లేకపోతే F.pad(tensor, (0, 0, 0, pad_అడ్డువరుసలు), విలువ=విలువ)
    ప్రతిజ్ఞా padded.shape == (Vp, C)
    తిరిగి padded

నిర్వచనం write_model(model, filename, dtype):
    # everything we need to instantiate the model
    # 1) header is: version int, GPTConfig ints, padding to 1024 bytes
    ప్రతిజ్ఞా dtype లో {"float32", "bfloat16"} # float16 todo maybe later
    సంస్కరణ = {
        "float32": 3, # 3: all tensors are fp32, padded vocab
        "bfloat16": 5, # 5: all tensors are bf16, padded vocab
    }[dtype]
    శీర్షిక = torch.zeros(256, dtype=torch.int32)
    శీర్షిక[0] = 20240326 # magic
    శీర్షిక[1] = సంస్కరణ # checkpoint version
    శీర్షిక[2] = model.config.ఖండం_కొలత
    శీర్షిక[3] = model.config.vocab_కొలత
    శీర్షిక[4] = model.config.n_layer
    శీర్షిక[5] = model.config.n_తల
    శీర్షిక[6] = model.config.n_embd
    # 2) the parameters follow the header
    params = {name: param.cpu() కొరకు name, param లో model.named_parameters()}
    # pad the vocab to a multiple of 128 here at export, for efficiency in C
    wte = params["transformer.wte.weight"] # (V, C)
    wte_padded = pad_vocab(wte) # (Vp, C)
    params["transformer.wte.weight"] = wte_padded # (Vp, C)
    ముద్రించు(f"padded vocab size from {wte.size(0)} to {wte_padded.size(0)}")
    శీర్షిక[7] = wte_padded.కొలత(0) # padded vocab size store in header
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
    శీర్షిక[0] = 20240327 # magic
    శీర్షిక[1] = 2 # run state version = 2 (1 -> 2 for padded vocab changes)
    శీర్షిక[2] = x.కొలత(0) # batch size of the batch, B
    శీర్షిక[3] = x.కొలత(1) # temporal extent of the batch, T
    grads = {name: param.grad.cpu() కొరకు name, param లో model.named_parameters()}
    # pad the vocab grads here as well, to mirror write_model
    wte_grad = grads["transformer.wte.weight"] # (V, C)
    wte_grad_padded = pad_vocab(wte_grad, విలువ=0) # (Vp, C) # TODO later maybe pad with nan?
    grads["transformer.wte.weight"] = wte_grad_padded # (Vp, C)
    ముద్రించు(f"padded vocab size in reference grads from {wte_grad.size(0)} to {wte_grad_padded.size(0)}")
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

నిర్వచనం write_tokenizer(enc, filename):
    n = enc.max_token_విలువ + 1
    శీర్షిక = torch.zeros(256, dtype=torch.int32)
    శీర్షిక[0] = 20240328 # magic
    శీర్షిక[1] = 2 # tokenizer version = 2 (1 -> 2: includes EOT token)
    శీర్షిక[2] = n # number of tokens
    శీర్షిక[3] = enc.eot_token # EOT token
    తోడుగా open(filename, "wb") ఇవ file:
        file.లిఖ(శీర్షిక.numpy().tobytes())
        కొరకు i లో శ్రేణి(n):
            b = enc.decode_బైట్లు([i])
            నిడివి = పొడవు(b)
            ప్రతిజ్ఞా నిడివి < 256, f"Token length exceeds 255: {length}"
            file.లిఖ(struct.pack("<B", నిడివి))  # Write the length as a 1-byte unsigned integer
            file.లిఖ(b)  # Write the actual bytes
    ముద్రించు(f"wrote {filename}")

# -----------------------------------------------------------------------------
# int main

నిర్వచనం print0(*వాదనలు, **kwargs):
    # modified print that only prints from the master process
    # if this is not a distributed run, it's just a print
    ఒకవేళ పూర్ణాంకం(os.environ.get("RANK", 0)) == 0:
        ముద్రించు(*వాదనలు, **kwargs)

ఒకవేళ __name__ == "__main__":
    దిగుమతి time
    దిగుమతి argparse
    దిగుమతి tiktoken
    print0(f"Running pytorch {torch.version.__version__}")

    # default settings will overfit a tiny batch of data
    # and save model weights and debug state to disk on the first iteration
    parser = argparse.వాదం_Parser()
    # file system input / output
    parser.కూడు_వాదం("--input_bin", type=str, అప్రమేయ="dev/data/tinyshakespeare/tiny_shakespeare_val.bin", help="input .bin to train on")
    parser.కూడు_వాదం("--input_val_bin", type=str, అప్రమేయ="", help="input .bin to eval validation loss on")
    parser.కూడు_వాదం("--output_dir", type=str, అప్రమేయ="", help="output directory to which to write logs and checkpoints")
    parser.కూడు_వాదం("--model", type=str, అప్రమేయ="gpt2", help="gpt2|gpt2-medium|gpt2-large|gpt2-xl|d12|d24|d36|d48")
    # token layout for each step of the optimization
    parser.కూడు_వాదం("--batch_size", type=పూర్ణాంకం, అప్రమేయ=4, help="batch size, in units of #batch dimensions")
    parser.కూడు_వాదం("--sequence_length", type=పూర్ణాంకం, అప్రమేయ=64, help="sequence length")
    parser.కూడు_వాదం("--total_batch_size", type=పూర్ణాంకం, అప్రమేయ=256, help="total desired batch size, in units of #tokens")
    # workload (number of steps)
    parser.కూడు_వాదం("--num_iterations", type=పూర్ణాంకం, అప్రమేయ=10, help="number of iterations to run")
    parser.కూడు_వాదం("--inference_only", type=పూర్ణాంకం, అప్రమేయ=0, help="only run inference")
    # optimization
    parser.కూడు_వాదం("--learning_rate", type=దశాంశం, అప్రమేయ=1e-4, help="learning rate warmup iterations")
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
    parser.కూడు_వాదం("--flash", type=పూర్ణాంకం, అప్రమేయ=0, help="use flash attention")
    parser.కూడు_వాదం("--dtype", type=str, అప్రమేయ="float32", help="float32|float16|bfloat16")
    parser.కూడు_వాదం("--zero_stage", type=పూర్ణాంకం, అప్రమేయ=0, help="zero redundancy optimizer stage (0/1/2/3)")
    # python -> C bridge
    parser.కూడు_వాదం("--write_tensors", type=పూర్ణాంకం, అప్రమేయ=1, help="write tensors to disk")
    వాదనలు = parser.parse_వాదనలు()

    # args error checking and convenience variables
    B, T = వాదనలు.batch_కొలత, వాదనలు.sequence_నిడివి
    ప్రతిజ్ఞా 1 <= T <= 1024
    ప్రతిజ్ఞా వాదనలు.dtype లో {"float32", "float16", "bfloat16"}
    ప్రతిజ్ఞా వాదనలు.model లో {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "d12", "d24", "d36", "d48"}

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
    ముద్రించు(f"using device: {device}")
    device_type = 'cuda' ఒకవేళ 'cuda' లో device లేకపోతే 'cpu'

    # calculate gradient accumulation from the desired total batch size and the current run configuration
    టోకెన్లు_చొప్పున_fwdbwd = B * T * ddp_world_కొలత
    ప్రతిజ్ఞా వాదనలు.మొత్తం_batch_కొలత % టోకెన్లు_చొప్పున_fwdbwd == 0
    grad_accum_అడుగులు = వాదనలు.మొత్తం_batch_కొలత // టోకెన్లు_చొప్పున_fwdbwd
    print0(f"total desired batch size: {args.total_batch_size}")
    print0(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    # set up a context manager following the desired dtype and device
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[వాదనలు.dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) ఒకవేళ device_type == "cuda" లేకపోతే nullcontext()

    # rng / reproducibility
    torch.manual_విత్తనం(42)
    ఒకవేళ torch.cuda.is_అందుబాటు():
        torch.cuda.manual_విత్తనం(42)

    # set the torch precision mode to use TensorFloat32 (TF32) for matmuls
    # docs https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    ఒకవేళ వాదనలు.tensorcores:
        torch.సమితి_float32_matmul_precision('high')

    # turn on/off flash attention
    ప్రతిజ్ఞా వాదనలు.flash లో {0, 1}
    FLASH = వాదనలు.flash

    # init (and write) the tokenizer
    enc = tiktoken.get_encoding("gpt2")
    ఒకవేళ master_process మరియు వాదనలు.write_tensors: # tokenizer is technically not tensors but ok
        write_tokenizer(enc, "gpt2_tokenizer.bin")

    # init the model, either from scratch or from OpenAI pretrained checkpoint
    ఒకవేళ వాదనలు.model[0] == "d":
        # from scratch (random weights)
        model_config = {
            "d12": GPTConfig(ఖండం_కొలత=1024, vocab_కొలత=50257, n_layer=12, n_తల=12, n_embd=768),
            "d24": GPTConfig(ఖండం_కొలత=1024, vocab_కొలత=50257, n_layer=24, n_తల=16, n_embd=1024),
            "d36": GPTConfig(ఖండం_కొలత=1024, vocab_కొలత=50257, n_layer=36, n_తల=20, n_embd=1280),
            "d48": GPTConfig(ఖండం_కొలత=1024, vocab_కొలత=50257, n_layer=48, n_తల=25, n_embd=1600),
        }[వాదనలు.model]
        model = GPT(model_config)
    లేకపోతే:
        # load the GPT-2 model weights
        model = GPT.from_pretrained(వాదనలు.model)
    model.train()
    model.గమ్యం(device)
    ఒకవేళ వాదనలు.compile:
        ఒకవేళ hasattr(config, "coordinate_descent_tuning"):
            config.coordinate_descent_tuning = నిజం # suggested by @Chillee
        print0("compiling the model...")
        model = torch.compile(model)

    # -------------------------------------------------------------------------
    # Our own version of a simple DistributedDataLoader

    # load tokens
    train_loader = Distributed_దత్తాంశం_Loader(వాదనలు.ఆదానం_bin, B, T, ddp_rank, ddp_world_కొలత)
    val_loader = ఏమీలేదు
    ఒకవేళ వాదనలు.ఆదానం_val_bin:
        val_loader = Distributed_దత్తాంశం_Loader(వాదనలు.ఆదానం_val_bin, B, T, ddp_rank, ddp_world_కొలత)

    # -------------------------------------------------------------------------
    # PyTorch -> C bridge: save some weights and state for C to load later as reference

    # do one forward pass to generate ground truth for our C tests
    ఒకవేళ master_process మరియు వాదనలు.write_tensors మరియు (కాదు వాదనలు.inference_only):
        x, y = train_loader.తదుపరి_batch()
        x, y = x.గమ్యం(device), y.గమ్యం(device)
        logits, loss = model(x, y)
        loss.backward()
        # save model params, in both float32 and bfloat16
        model_గమ్యం_కొలత = {"gpt2": "124M", "gpt2-medium": "355M", "gpt2-large": "774M", "gpt2-xl": "1558M"}
        model_గమ్యం_కొలత.update({f"d{d}": f"d{d}" కొరకు d లో [12, 24, 36, 48]})
        model_కొలత_str = model_గమ్యం_కొలత[వాదనలు.model] # e.g. "124M", or "d12"
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

    # create the logging directory if it does not exist
    logfile = ఏమీలేదు
    ఒకవేళ వాదనలు.output_dir:
        os.makedirs(వాదనలు.output_dir, exist_ok=నిజం)
        logfile = os.మార్గం.join(వాదనలు.output_dir, "main.log")
        # create the log file "main.log" inside it, and wipe it clean
        తోడుగా open(logfile, "w") ఇవ f:
            దాటు

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
            # before we end, let's also do one round of inference
            # we'll kick off the generation with "<|endoftext|>", which designates the start of a new sequence
            ప్రారంభం_ids = [enc.eot_token]
            xg = (torch.tensor(ప్రారంభం_ids, dtype=torch.దీర్ఘ, device=device)[ఏమీలేదు, ...])
            max_new_టోకెన్లు = 32
            temperature = 1.0
            శిఖరం_k = 40
            yg = raw_model.generate(xg, max_new_టోకెన్లు, temperature=temperature, శిఖరం_k=శిఖరం_k)
            print0('---------------')
            print0(enc.decode(yg[0].tolist()))
            print0('---------------')

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
