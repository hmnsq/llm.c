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

இறக்குமதி os
இறக்குமதி math
இறக்குமதி glob
இறக்குமதி struct
இறக்குமதி inspect
தஃ contextlib இறக்குமதி nullcontext
தஃ dataclasses இறக்குமதி dataclass

இறக்குமதி numpy இவ np
இறக்குமதி torch
இறக்குமதி torch.nn இவ nn
தஃ torch.nn இறக்குமதி functional இவ F
இறக்குமதி torch._inductor.config இவ config
தஃ torch.nn.parallel இறக்குமதி DistributedDataParallel இவ DDP
தஃ torch.distributed இறக்குமதி init_process_group, destroy_process_group
தஃ torch.distributed.optim இறக்குமதி ZeroRedundancyOptimizer
இறக்குமதி torch.distributed இவ dist

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model

வகுப்பு NewGELU(nn.Module):
    """Careful there are a few versions of GeLU, this one is the exact one used by OpenAI"""
    def forward(self, உள்ளீடு):
        திரும்பு 0.5 * உள்ளீடு * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (உள்ளீடு + 0.044715 * torch.pow(உள்ளீடு, 3.0))))

# using a global to toggle flash-attention
FLASH = 0

வகுப்பு CausalSelfAttention(nn.Module):

    def __init__(self, config):
        மேல்().__init__()
        ப்ரதிஜ்ஞா config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        எனில் FLASH:
            # flashattention
            y = F.scaled_dot_product_attention(q, k, v, is_causal=உண்மை)
        இல்லையெனில்:
            # manual implementation of attention
            # this materializes the large (T,T) matrix for all the queries and keys
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, தசம('-inf'))
            att = F.softmax(att, dim=-1)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        திரும்பு y

வகுப்பு MLP(nn.Module):

    def __init__(self, config):
        மேல்().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = NewGELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        திரும்பு x

வகுப்பு Block(nn.Module):

    def __init__(self, config):
        மேல்().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        திரும்பு x

# -----------------------------------------------------------------------------
# The main GPT-2 model

@dataclass
வகுப்பு GPTConfig:
    block_size: முழு = 1024
    vocab_size: முழு = 50257
    n_layer: முழு = 12
    n_head: முழு = 12
    n_embd: முழு = 768

வகுப்பு GPT(nn.Module):

    def __init__(self, config):
        மேல்().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) க்கு _ உள் range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=பொய்)
        self.lm_head.LLMC_SKIP_INIT = 1 # don't init this one, we will tie weights
        self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

        # init all weights, use a torch rng object to be very careful
        self.init_rng = torch.Generator()
        self.init_rng.manual_seed(42)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        எனில் isinstance(module, nn.Linear):
            # apply special scaled init to the residual projections, per GPT-2 paper
            std = 0.02 எனில் இல்லை hasattr(module, 'LLMC_RESIDUAL_SCALE_FLAG') இல்லையெனில் 0.02/math.sqrt(2 * self.config.n_layer)
            # we want to skip initializing lm_head, which shares parameters with wte
            # and wte was already initialized down below during the Embedding init
            எனில் இல்லை hasattr(module, 'LLMC_SKIP_INIT'):
                torch.nn.init.normal_(module.weight, mean=0.0, std=std, generator=self.init_rng)
            எனில் module.bias is இல்லை None:
                torch.nn.init.zeros_(module.bias)
        அல்லதெனில் isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02, generator=self.init_rng)

    def forward(self, idx, targets=None, return_logits=உண்மை):
        device = idx.device
        b, t = idx.size()
        ப்ரதிஜ்ஞா t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = tok_emb + pos_emb

        க்கு block உள் self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        எனில் targets is இல்லை None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        இல்லையெனில்:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        எனில் இல்லை return_logits:
            logits = None

        திரும்பு logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        ப்ரதிஜ்ஞா model_type உள் {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        தஃ transformers இறக்குமதி GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_சாவிகள் = sd.சாவிகள்()
        sd_சாவிகள் = [k க்கு k உள் sd_சாவிகள் எனில் இல்லை k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_சாவிகள்_hf = sd_hf.சாவிகள்()
        sd_சாவிகள்_hf = [k க்கு k உள் sd_சாவிகள்_hf எனில் இல்லை k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_சாவிகள்_hf = [k க்கு k உள் sd_சாவிகள்_hf எனில் இல்லை k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        ப்ரதிஜ்ஞா len(sd_சாவிகள்_hf) == len(sd_சாவிகள்), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        க்கு k உள் sd_சாவிகள்_hf:
            எனில் any(k.endswith(w) க்கு w உள் transposed):
                # special treatment for the Conv1D weights we need to transpose
                ப்ரதிஜ்ஞா sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            இல்லையெனில்:
                # vanilla copy over the other parameters
                ப்ரதிஜ்ஞா sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        திரும்பு model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, zero_stage):
        # start with all of the candidate parameters
        param_dict = {pn: p க்கு pn, p உள் self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p க்கு pn, p உள் param_dict.பொருட்கள்() எனில் p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p க்கு n, p உள் param_dict.பொருட்கள்() எனில் p.dim() >= 2]
        nodecay_params = [p க்கு n, p உள் param_dict.பொருட்கள்() எனில் p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        இலக்கம்_decay_params = sum(p.numel() க்கு p உள் decay_params)
        இலக்கம்_nodecay_params = sum(p.numel() க்கு p உள் nodecay_params)
        print0(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print0(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' உள் inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available மற்றும் device_type == 'cuda'
        print0(f"using fused AdamW: {use_fused}")
        எனில் zero_stage == 1:
            print0("using ZeroRedundancyOptimizer")
            optimizer = ZeroRedundancyOptimizer(**optim_groups[0], optimizer_class=torch.optim.AdamW,
                                                lr=learning_rate, betas=betas, fused=use_fused)
            optimizer.add_param_group(optim_groups[1])
        இல்லையெனில்:
            print0("using regular AdamW")
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=use_fused)
        திரும்பு optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, உச்சி_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        க்கு _ உள் range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx எனில் idx.size(1) <= self.config.block_size இல்லையெனில் idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            எனில் உச்சி_k is இல்லை None:
                v, _ = torch.topk(logits, min(உச்சி_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -தசம('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, இலக்கம்_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        திரும்பு idx

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def _peek_data_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") இவ f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.பட(256*4), dtype=np.int32)
    எனில் header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print("---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README")
        print("---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try")
        exit(1)
    ப்ரதிஜ்ஞா header[1] == 1, "unsupported version"
    ntok = header[2] # number of tokens (claimed)
    திரும்பு ntok # for now just return the number of tokens

def _load_data_shard(filename):
    with open(filename, "rb") இவ f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.பட(256*4), dtype=np.int32)
        ப்ரதிஜ்ஞா header[0] == 20240520, "magic number mismatch in the data .bin file"
        ப்ரதிஜ்ஞா header[1] == 1, "unsupported version"
        ntok = header[2] # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.பட(), dtype=np.uint16)
    ப்ரதிஜ்ஞா len(tokens) == ntok, "number of tokens read does not match header?"
    திரும்பு tokens

வகுப்பு DistributedDataLoader:
    def __init__(self, filename_pattern, B, T, process_rank, இலக்கம்_processes):
        self.process_rank = process_rank
        self.இலக்கம்_processes = இலக்கம்_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        ப்ரதிஜ்ஞா len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_total = 0
        க்கு fname உள் self.files:
            shard_ntok = _peek_data_shard(fname)
            ப்ரதிஜ்ஞா shard_ntok >= இலக்கம்_processes * B * T + 1
            ntok_total += shard_ntok
        self.ntok_total = ntok_total
        print0(f"DataLoader: total number of tokens: {ntok_total:,} across {len(self.files)} files")

        # kick things off
        self.current_shard = None
        self.reset()

    def reset(self):
        # we're being a bit clever here: if we already had shard 0 loaded,
        # then don't do the work to reload it, just reset the pointer
        எனில் self.current_shard != 0:
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
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the start pointer in current shard
        self.current_position += B * T * self.இலக்கம்_processes
        # if loading the next batch would be out of bounds advance the shard
        எனில் self.current_position + (B * T * self.இலக்கம்_processes + 1) > len(self.tokens):
            self.advance()
        திரும்பு x, y

# -----------------------------------------------------------------------------
# Python -> C bridge utilities for saving params/grads/activations to .bin files

def write_fp32(tensor, file):
    t = tensor.detach().cpu().to(torch.float32)
    b = t.numpy().tobytes()
    file.லிக(b)

def write_bf16(tensor, file):
    t = tensor.detach().cpu().to(torch.bfloat16)
    # numpy doesn't have bf16 datatype so we have to trick it
    t = t.view(torch.int16) # trick: reinterpret as int16
    b = t.numpy().tobytes()
    file.லிக(b)

def write_tensors(model_tensors, L, file, dtype):
    # writes the GPT-2 model's weights to a binary file
    ப்ரதிஜ்ஞா dtype உள் {"float32", "bfloat16"}
    write_fun = write_fp32 எனில் dtype == "float32" இல்லையெனில் write_bf16
    write_fun(model_tensors["transformer.wte.weight"], file) # (V, C)
    write_fun(model_tensors["transformer.wpe.weight"], file) # (T, C)
    க்கு i உள் range(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_1.weight"], file)
    க்கு i உள் range(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_1.bias"], file)
    க்கு i உள் range(L): # (L, 3C, C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_attn.weight"], file)
    க்கு i உள் range(L): # (L, 3C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_attn.bias"], file)
    க்கு i உள் range(L): # (L, C, C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_proj.weight"], file)
    க்கு i உள் range(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_proj.bias"], file)
    க்கு i உள் range(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_2.weight"], file)
    க்கு i உள் range(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_2.bias"], file)
    க்கு i உள் range(L): # (L, 4C, C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_fc.weight"], file)
    க்கு i உள் range(L): # (L, 4C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_fc.bias"], file)
    க்கு i உள் range(L): # (L, C, 4C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_proj.weight"], file)
    க்கு i உள் range(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_proj.bias"], file)
    write_fun(model_tensors["transformer.ln_f.weight"], file) # (C, )
    write_fun(model_tensors["transformer.ln_f.bias"], file) # (C, )

@torch.no_grad()
def pad_vocab(tensor, multiple=128, மதிப்பு=0):
    """
    The dimension of the vocab size in GPT-2 is 50,257
    which is unfortunately a very unfriendly number for a lot of
    matrix operations on the GPU. So we pad it to the nearest
    friendlier multiple, e.g. 50,304 if multiple=128 when we
    export the weights into C land. This is a NOOP algorithmically
    and is only done to make the tensor operations more efficient.
    """
    ப்ரதிஜ்ஞா tensor.ndim == 2
    V, C = tensor.shape
    ப்ரதிஜ்ஞா V == 50257, "just being defensive here"
    # calculate padded vocab size by rounding up to nearest multiple
    Vp = ((V + multiple - 1) // multiple) * multiple
    # pad the tensor
    pad_rows = Vp - V
    padded = tensor எனில் pad_rows == 0 இல்லையெனில் F.pad(tensor, (0, 0, 0, pad_rows), மதிப்பு=மதிப்பு)
    ப்ரதிஜ்ஞா padded.shape == (Vp, C)
    திரும்பு padded

def write_model(model, filename, dtype):
    # everything we need to instantiate the model
    # 1) header is: version int, GPTConfig ints, padding to 1024 bytes
    ப்ரதிஜ்ஞா dtype உள் {"float32", "bfloat16"} # float16 todo maybe later
    version = {
        "float32": 3, # 3: all tensors are fp32, padded vocab
        "bfloat16": 5, # 5: all tensors are bf16, padded vocab
    }[dtype]
    header = torch.zeros(256, dtype=torch.int32)
    header[0] = 20240326 # magic
    header[1] = version # checkpoint version
    header[2] = model.config.block_size
    header[3] = model.config.vocab_size
    header[4] = model.config.n_layer
    header[5] = model.config.n_head
    header[6] = model.config.n_embd
    # 2) the parameters follow the header
    params = {name: param.cpu() க்கு name, param உள் model.named_parameters()}
    # pad the vocab to a multiple of 128 here at export, for efficiency in C
    wte = params["transformer.wte.weight"] # (V, C)
    wte_padded = pad_vocab(wte) # (Vp, C)
    params["transformer.wte.weight"] = wte_padded # (Vp, C)
    print(f"padded vocab size from {wte.size(0)} to {wte_padded.size(0)}")
    header[7] = wte_padded.size(0) # padded vocab size store in header
    # now write to file
    with open(filename, "wb") இவ file:
        file.லிக(header.numpy().tobytes()) # header
        write_tensors(params, model.config.n_layer, file, dtype) # params
    print(f"wrote {filename}")

def write_state(model, x, y, logits, loss, filename):
    # the state is used for debugging.
    # it contains information about the input, logits, loss, and the parameter gradients
    # this can be used for checking the computation correctness in C
    header = torch.zeros(256, dtype=torch.int32)
    header[0] = 20240327 # magic
    header[1] = 2 # run state version = 2 (1 -> 2 for padded vocab changes)
    header[2] = x.size(0) # batch size of the batch, B
    header[3] = x.size(1) # temporal extent of the batch, T
    grads = {name: param.grad.cpu() க்கு name, param உள் model.named_parameters()}
    # pad the vocab grads here as well, to mirror write_model
    wte_grad = grads["transformer.wte.weight"] # (V, C)
    wte_grad_padded = pad_vocab(wte_grad, மதிப்பு=0) # (Vp, C) # TODO later maybe pad with nan?
    grads["transformer.wte.weight"] = wte_grad_padded # (Vp, C)
    print(f"padded vocab size in reference grads from {wte_grad.size(0)} to {wte_grad_padded.size(0)}")
    with open(filename, "wb") இவ file:
        # header
        file.லிக(header.numpy().tobytes())
        # input x
        file.லிக(x.cpu().numpy().astype("int32").tobytes()) # (B, T)
        # targets y
        file.லிக(y.cpu().numpy().astype("int32").tobytes()) # (B, T)
        # logits (result of the model forward pass)
        write_fp32(logits.cpu(), file)
        # loss (single float, result of the cross entropy loss)
        write_fp32(loss.cpu(), file)
        # gradients
        write_tensors(grads, model.config.n_layer, file, "float32")
    print(f"wrote {filename}")

def write_tokenizer(enc, filename):
    n = enc.max_token_மதிப்பு + 1
    header = torch.zeros(256, dtype=torch.int32)
    header[0] = 20240328 # magic
    header[1] = 2 # tokenizer version = 2 (1 -> 2: includes EOT token)
    header[2] = n # number of tokens
    header[3] = enc.eot_token # EOT token
    with open(filename, "wb") இவ file:
        file.லிக(header.numpy().tobytes())
        க்கு i உள் range(n):
            b = enc.decode_bytes([i])
            length = len(b)
            ப்ரதிஜ்ஞா length < 256, f"Token length exceeds 255: {length}"
            file.லிக(struct.pack("<B", length))  # Write the length as a 1-byte unsigned integer
            file.லிக(b)  # Write the actual bytes
    print(f"wrote {filename}")

# -----------------------------------------------------------------------------
# int main

def print0(*args, **kwargs):
    # modified print that only prints from the master process
    # if this is not a distributed run, it's just a print
    எனில் முழு(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)

எனில் __name__ == "__main__":
    இறக்குமதி time
    இறக்குமதி argparse
    இறக்குமதி tiktoken
    print0(f"Running pytorch {torch.version.__version__}")

    # default settings will overfit a tiny batch of data
    # and save model weights and debug state to disk on the first iteration
    parser = argparse.ArgumentParser()
    # file system input / output
    parser.add_argument("--input_bin", type=str, இயல்பு="dev/data/tinyshakespeare/tiny_shakespeare_val.bin", help="input .bin to train on")
    parser.add_argument("--input_val_bin", type=str, இயல்பு="", help="input .bin to eval validation loss on")
    parser.add_argument("--output_dir", type=str, இயல்பு="", help="output directory to which to write logs and checkpoints")
    parser.add_argument("--model", type=str, இயல்பு="gpt2", help="gpt2|gpt2-medium|gpt2-large|gpt2-xl|d12|d24|d36|d48")
    # token layout for each step of the optimization
    parser.add_argument("--batch_size", type=முழு, இயல்பு=4, help="batch size, in units of #batch dimensions")
    parser.add_argument("--sequence_length", type=முழு, இயல்பு=64, help="sequence length")
    parser.add_argument("--total_batch_size", type=முழு, இயல்பு=256, help="total desired batch size, in units of #tokens")
    # workload (number of steps)
    parser.add_argument("--num_iterations", type=முழு, இயல்பு=10, help="number of iterations to run")
    parser.add_argument("--inference_only", type=முழு, இயல்பு=0, help="only run inference")
    # optimization
    parser.add_argument("--learning_rate", type=தசம, இயல்பு=1e-4, help="learning rate warmup iterations")
    parser.add_argument("--warmup_iters", type=முழு, இயல்பு=0, help="learning rate warmup iterations")
    parser.add_argument("--learning_rate_decay_frac", type=தசம, இயல்பு=1.0, help="learning rate warmup iterations")
    parser.add_argument("--weight_decay", type=தசம, இயல்பு=0.0, help="weight decay")
    parser.add_argument("--grad_clip", type=தசம, இயல்பு=1.0, help="maximum gradient magnitude")
    # evaluation
    parser.add_argument("--val_loss_every", type=முழு, இயல்பு=0, help="every how mant steps to evaluate val loss?")
    parser.add_argument("--val_max_steps", type=முழு, இயல்பு=20, help="how many batches of val to average?")
    parser.add_argument("--sample_every", type=முழு, இயல்பு=0, help="how often to sample from the model?")
    # debugging
    parser.add_argument("--overfit_single_batch", type=முழு, இயல்பு=1, help="overfit just one batch of data")
    # numerics
    parser.add_argument("--tensorcores", type=முழு, இயல்பு=0, help="use tensorcores")
    # memory management
    parser.add_argument("--device", type=str, இயல்பு="", help="by default we autodetect, or set it here")
    parser.add_argument("--compile", type=முழு, இயல்பு=0, help="torch.compile the model")
    parser.add_argument("--flash", type=முழு, இயல்பு=0, help="use flash attention")
    parser.add_argument("--dtype", type=str, இயல்பு="float32", help="float32|float16|bfloat16")
    parser.add_argument("--zero_stage", type=முழு, இயல்பு=0, help="zero redundancy optimizer stage (0/1/2/3)")
    # python -> C bridge
    parser.add_argument("--write_tensors", type=முழு, இயல்பு=1, help="write tensors to disk")
    args = parser.parse_args()

    # args error checking and convenience variables
    B, T = args.batch_size, args.sequence_length
    ப்ரதிஜ்ஞா 1 <= T <= 1024
    ப்ரதிஜ்ஞா args.dtype உள் {"float32", "float16", "bfloat16"}
    ப்ரதிஜ்ஞா args.model உள் {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "d12", "d24", "d36", "d48"}

    # set up DDP (distributed data parallel). torchrun sets this env variable
    ddp = முழு(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    எனில் ddp:
        # use of DDP atm demands CUDA, we set the device appropriately according to rank
        ப்ரதிஜ்ஞா torch.cuda.is_available(), "for now i think we need CUDA for DDP"
        init_process_group(backend='nccl')
        ddp_rank = முழு(os.environ['RANK'])
        ddp_local_rank = முழு(os.environ['LOCAL_RANK'])
        ddp_world_size = முழு(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        seed_offset = 0 # each process gets the exact same seed
        zero_stage = args.zero_stage
    இல்லையெனில்:
        ddp_rank = 0
        ddp_local_rank = 0
        zero_stage = 0
        ddp_world_size = 1
        master_process = உண்மை
        seed_offset = 0
        # select the device
        எனில் args.device:
            # provided explicitly by the user
            device = args.device
        இல்லையெனில்:
            # attempt to autodetect the device
            device = "cpu"
            எனில் torch.cuda.is_available():
                device = "cuda"
            அல்லதெனில் hasattr(torch.backends, "mps") மற்றும் torch.backends.mps.is_available():
                device = "mps"
    print(f"using device: {device}")
    device_type = 'cuda' எனில் 'cuda' உள் device இல்லையெனில் 'cpu'

    # calculate gradient accumulation from the desired total batch size and the current run configuration
    tokens_per_fwdbwd = B * T * ddp_world_size
    ப்ரதிஜ்ஞா args.total_batch_size % tokens_per_fwdbwd == 0
    grad_accum_steps = args.total_batch_size // tokens_per_fwdbwd
    print0(f"total desired batch size: {args.total_batch_size}")
    print0(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    # set up a context manager following the desired dtype and device
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) எனில் device_type == "cuda" இல்லையெனில் nullcontext()

    # rng / reproducibility
    torch.manual_seed(42)
    எனில் torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # set the torch precision mode to use TensorFloat32 (TF32) for matmuls
    # docs https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    எனில் args.tensorcores:
        torch.set_float32_matmul_precision('high')

    # turn on/off flash attention
    ப்ரதிஜ்ஞா args.flash உள் {0, 1}
    FLASH = args.flash

    # init (and write) the tokenizer
    enc = tiktoken.get_encoding("gpt2")
    எனில் master_process மற்றும் args.write_tensors: # tokenizer is technically not tensors but ok
        write_tokenizer(enc, "gpt2_tokenizer.bin")

    # init the model, either from scratch or from OpenAI pretrained checkpoint
    எனில் args.model[0] == "d":
        # from scratch (random weights)
        model_config = {
            "d12": GPTConfig(block_size=1024, vocab_size=50257, n_layer=12, n_head=12, n_embd=768),
            "d24": GPTConfig(block_size=1024, vocab_size=50257, n_layer=24, n_head=16, n_embd=1024),
            "d36": GPTConfig(block_size=1024, vocab_size=50257, n_layer=36, n_head=20, n_embd=1280),
            "d48": GPTConfig(block_size=1024, vocab_size=50257, n_layer=48, n_head=25, n_embd=1600),
        }[args.model]
        model = GPT(model_config)
    இல்லையெனில்:
        # load the GPT-2 model weights
        model = GPT.from_pretrained(args.model)
    model.train()
    model.to(device)
    எனில் args.compile:
        எனில் hasattr(config, "coordinate_descent_tuning"):
            config.coordinate_descent_tuning = உண்மை # suggested by @Chillee
        print0("compiling the model...")
        model = torch.compile(model)

    # -------------------------------------------------------------------------
    # Our own version of a simple DistributedDataLoader

    # load tokens
    train_loader = DistributedDataLoader(args.உள்ளீடு_bin, B, T, ddp_rank, ddp_world_size)
    val_loader = None
    எனில் args.உள்ளீடு_val_bin:
        val_loader = DistributedDataLoader(args.உள்ளீடு_val_bin, B, T, ddp_rank, ddp_world_size)

    # -------------------------------------------------------------------------
    # PyTorch -> C bridge: save some weights and state for C to load later as reference

    # do one forward pass to generate ground truth for our C tests
    எனில் master_process மற்றும் args.write_tensors மற்றும் (இல்லை args.inference_only):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, y)
        loss.backward()
        # save model params, in both float32 and bfloat16
        model_to_size = {"gpt2": "124M", "gpt2-medium": "355M", "gpt2-large": "774M", "gpt2-xl": "1558M"}
        model_to_size.update({f"d{d}": f"d{d}" க்கு d உள் [12, 24, 36, 48]})
        model_size_str = model_to_size[args.model] # e.g. "124M", or "d12"
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
    எனில் ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module எனில் ddp இல்லையெனில் model # always contains the "raw" unwrapped model

    # init the optimizer
    optimizer = raw_model.configure_optimizers(weight_decay=args.weight_decay,
                                               learning_rate=args.learning_rate, betas=(0.9, 0.95),
                                               device_type=device, zero_stage=zero_stage)

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        min_lr = args.learning_rate * args.learning_rate_decay_frac
        # 1) linear warmup for warmup_iters steps
        எனில் it < args.warmup_iters:
            திரும்பு args.learning_rate * (it+1) / args.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        எனில் it > args.இலக்கம்_iterations:
            திரும்பு min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - args.warmup_iters) / (args.இலக்கம்_iterations - args.warmup_iters)
        ப்ரதிஜ்ஞா 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
        திரும்பு min_lr + coeff * (args.learning_rate - min_lr)

    # create the logging directory if it does not exist
    logfile = None
    எனில் args.output_dir:
        os.makedirs(args.output_dir, exist_ok=உண்மை)
        logfile = os.path.join(args.output_dir, "main.log")
        # create the log file "main.log" inside it, and wipe it clean
        with open(logfile, "w") இவ f:
            pass

    எனில் device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    timings = []
    norm = -1.0   # dummy value to print in inference-only mode
    க்கு step உள் range(args.இலக்கம்_iterations + 1):
        t0 = time.time()
        last_step = (step == args.இலக்கம்_iterations)

        # once in a while evaluate the validation dataset
        எனில் (args.val_loss_every > 0 \
            மற்றும் (step % args.val_loss_every == 0 அல்லது last_step)) \
            மற்றும் (val_loader is இல்லை None):
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss = 0.0
                க்கு _ உள் range(args.val_max_steps):
                    x, y = val_loader.next_batch()
                    x, y = x.to(device), y.to(device)
                    _, loss = model(x, y, return_logits=பொய்)
                    val_loss += loss.item()
                val_loss /= args.val_max_steps
            # log to console and to file
            print0(f"val loss {val_loss}")
            எனில் master_process மற்றும் logfile is இல்லை None:
                with open(logfile, "a") இவ f:
                    f.லிக("s:%d tel:%f\n" % (step, val_loss))

        # once in a while perform model inference on the master process
        எனில் (args.sample_every > 0 \
            மற்றும் (step % args.sample_every == 0 அல்லது last_step)) \
            மற்றும் master_process:
            model.eval()
            # before we end, let's also do one round of inference
            # we'll kick off the generation with "<|endoftext|>", which designates the start of a new sequence
            start_ids = [enc.eot_token]
            xg = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
            max_new_tokens = 32
            temperature = 1.0
            உச்சி_k = 40
            yg = raw_model.generate(xg, max_new_tokens, temperature=temperature, உச்சி_k=உச்சி_k)
            print0('---------------')
            print0(enc.decode(yg[0].tolist()))
            print0('---------------')

        # bit confusing: we want to make sure to eval and sample on 0th iteration
        # but also after the very last iteration. so we loop for step <= num_iterations
        # instead of just < num_iterations (one extra due to <=), only to do
        # the validation/sampling one last time, and then we break right here as we're done.
        எனில் last_step:
            நிறுத்து

        # --------------- TRAINING SECTION BEGIN -----------------
        model.train()
        optimizer.zero_grad(set_to_none=உண்மை)
        # if we are trying to overfit a single batch, we reset the loader here
        எனில் args.overfit_single_batch:
            train_loader.reset()
        # micro-batch loop where we do gradient accumulation to reach desired total batch size
        lossf = 0.0 # for getting the mean loss (as simple float) over the accumulation steps
        க்கு micro_step உள் range(grad_accum_steps):
            # fetch a batch
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)
            எனில் ddp:
                # we want only the last micro-step to sync grads in a DDP model
                # the official way to do this is with model.no_sync(), but that is a
                # context manager that bloats the code, so we just toggle this variable
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            # forward pass
            with ctx:
                _, loss = model(x, y, return_logits=பொய்)
                # we have to scale the loss to account for gradient accumulation,
                # because the gradients just add on each successive backward().
                # addition of gradients corresponds to a SUM in the objective, but
                # instead of a SUM we want MEAN, so we scale the loss here
                loss = loss / grad_accum_steps
                lossf += loss.detach() # keep track of the mean loss
            # backward pass
            எனில் இல்லை args.inference_only:
                loss.backward()
        எனில் ddp:
            dist.all_reduce(lossf, op=dist.ReduceOp.AVG)
        lossf = lossf.item()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        # determine and set the learning rate for this iteration
        lr = get_lr(step)
        க்கு param_group உள் optimizer.param_groups:
            param_group['lr'] = lr
        # step the optimizer
        optimizer.step()
        # --------------- TRAINING SECTION END -------------------
        # everything that follows now is just diagnostics, prints, logging, etc.

        # wait on the CPU for all device work to end so we get accurate per-iteration timings below
        எனில் device == "mps":
            torch.mps.synchronize()
        அல்லதெனில் device == "cuda":
            torch.cuda.synchronize()
        # time and print
        t1 = time.time()
        # the 0th iteration is often an outlier (much slower) => skip logging it
        tokens_per_second = grad_accum_steps * ddp_world_size * B * T / (t1-t0)
        print0(f"step {step+1:4d}/{args.num_iterations} | train loss {lossf:.6f} | norm {norm:.4f} | lr {lr:.2e} | ({(t1-t0)*1000:.2f} ms | {tokens_per_second:.0f} tok/s)")
        # log to logile
        எனில் master_process மற்றும் logfile is இல்லை None:
            with open(logfile, "a") இவ f:
                f.லிக("s:%d trl:%f\n" % (step, lossf))

        # keep track of smooth timings, last 20 iterations
        எனில் step > 0 மற்றும் step > args.இலக்கம்_iterations - 20:
            timings.append(t1-t0)

    # print the average of the last 20 timings, to get something smooth-ish
    timings = timings[-20:]
    print0(f"final {len(timings)} iters avg: {np.mean(timings)*1000:.3f}ms")
    print0(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

    # -------------------------------------------------------------------------
    # clean up nice
    எனில் ddp:
        destroy_process_group()
