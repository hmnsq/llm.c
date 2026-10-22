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

আমদানি os
আমদানি math
আমদানি glob
আমদানি struct
আমদানি inspect
তঃ contextlib আমদানি nullcontext
তঃ dataclasses আমদানি dataclass

আমদানি numpy ইব np
আমদানি torch
আমদানি torch.nn ইব nn
তঃ torch.nn আমদানি functional ইব F
আমদানি torch._inductor.config ইব config
তঃ torch.nn.parallel আমদানি Distributed_তথ্য_Parallel ইব DDP
তঃ torch.distributed আমদানি init_process_group, destroy_process_group
তঃ torch.distributed.optim আমদানি ZeroRedundancyOptimizer
আমদানি torch.distributed ইব dist

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the GPT-2 model

শ্রেণী NewGELU(nn.Module):
    """Careful there are a few versions of GeLU, this one is the exact one used by OpenAI"""
    def forward(self, প্রবেশ):
        ফেরত 0.5 * প্রবেশ * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (প্রবেশ + 0.044715 * torch.pow(প্রবেশ, 3.0))))

# using a global to toggle flash-attention
FLASH = 0

শ্রেণী CausalSelfAttention(nn.Module):

    def __init__(self, config):
        পরম().__init__()
        প্রতিজ্ঞা config.n_embd % config.n_শীর্ষ == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1
        # regularization
        self.n_শীর্ষ = config.n_শীর্ষ
        self.n_embd = config.n_embd
        # not really a 'bias', more of a mask, but following the OpenAI/HF naming though
        self.register_বাফার("bias", torch.tril(torch.ones(config.block_আকার, config.block_আকার))
                                     .view(1, 1, config.block_আকার, config.block_আকার))

    def forward(self, x):
        B, T, C = x.আকার_2() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_শীর্ষ, C // self.n_শীর্ষ).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_শীর্ষ, C // self.n_শীর্ষ).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_শীর্ষ, C // self.n_শীর্ষ).transpose(1, 2) # (B, nh, T, hs)
        যদি FLASH:
            # flashattention
            y = F.scaled_dot_product_attention(q, k, v, is_causal=সত্য)
        অন্যথা:
            # manual implementation of attention
            # this materializes the large (T,T) matrix for all the queries and keys
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.আকার_2(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, দশমিক('-inf'))
            att = F.softmax(att, dim=-1)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        ফেরত y

শ্রেণী MLP(nn.Module):

    def __init__(self, config):
        পরম().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = NewGELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.LLMC_RESIDUAL_SCALE_FLAG = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        ফেরত x

শ্রেণী Block(nn.Module):

    def __init__(self, config):
        পরম().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        ফেরত x

# -----------------------------------------------------------------------------
# The main GPT-2 model

@dataclass
শ্রেণী GPTConfig:
    block_আকার: পূর্ণসংখ্যা = 1024
    vocab_আকার: পূর্ণসংখ্যা = 50257
    n_layer: পূর্ণসংখ্যা = 12
    n_শীর্ষ: পূর্ণসংখ্যা = 12
    n_embd: পূর্ণসংখ্যা = 768

শ্রেণী GPT(nn.Module):

    def __init__(self, config):
        পরম().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_আকার, config.n_embd),
            wpe = nn.Embedding(config.block_আকার, config.n_embd),
            h = nn.ModuleList([Block(config) জন্য _ মধ্যে range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_শীর্ষ = nn.Linear(config.n_embd, config.vocab_আকার, bias=মিথ্যা)
        self.lm_শীর্ষ.LLMC_SKIP_INIT = 1 # don't init this one, we will tie weights
        self.transformer.wte.weight = self.lm_শীর্ষ.weight # https://paperswithcode.com/method/weight-tying

        # init all weights, use a torch rng object to be very careful
        self.init_rng = torch.Generator()
        self.init_rng.manual_seed(42)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        যদি isinstance(module, nn.Linear):
            # apply special scaled init to the residual projections, per GPT-2 paper
            std = 0.02 যদি নয় hasattr(module, 'LLMC_RESIDUAL_SCALE_FLAG') অন্যথা 0.02/math.sqrt(2 * self.config.n_layer)
            # we want to skip initializing lm_head, which shares parameters with wte
            # and wte was already initialized down below during the Embedding init
            যদি নয় hasattr(module, 'LLMC_SKIP_INIT'):
                torch.nn.init.normal_(module.weight, mean=0.0, std=std, generator=self.init_rng)
            যদি module.bias is নয় None:
                torch.nn.init.zeros_(module.bias)
        নতুবা_যদি isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02, generator=self.init_rng)

    def forward(self, idx, targets=None, return_logits=সত্য):
        device = idx.device
        b, t = idx.আকার_2()
        প্রতিজ্ঞা t <= self.config.block_আকার, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
        x = tok_emb + pos_emb

        জন্য block মধ্যে self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        যদি targets is নয় None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_শীর্ষ(x)
            loss = F.cross_entropy(logits.view(-1, logits.আকার_2(-1)), targets.view(-1), ignore_সূচক=-1)
        অন্যথা:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_শীর্ষ(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
            loss = None

        # there are performance reasons why not returning logits is prudent, if not needed
        যদি নয় return_logits:
            logits = None

        ফেরত logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        প্রতিজ্ঞা model_type মধ্যে {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        তঃ transformers আমদানি GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_শীর্ষ=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_শীর্ষ=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_শীর্ষ=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_শীর্ষ=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_চাবিগুলি = sd.চাবিগুলি()
        sd_চাবিগুলি = [k জন্য k মধ্যে sd_চাবিগুলি যদি নয় k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_চাবিগুলি_hf = sd_hf.চাবিগুলি()
        sd_চাবিগুলি_hf = [k জন্য k মধ্যে sd_চাবিগুলি_hf যদি নয় k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_চাবিগুলি_hf = [k জন্য k মধ্যে sd_চাবিগুলি_hf যদি নয় k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        প্রতিজ্ঞা len(sd_চাবিগুলি_hf) == len(sd_চাবিগুলি), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        জন্য k মধ্যে sd_চাবিগুলি_hf:
            যদি any(k.endswith(w) জন্য w মধ্যে transposed):
                # special treatment for the Conv1D weights we need to transpose
                প্রতিজ্ঞা sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            অন্যথা:
                # vanilla copy over the other parameters
                প্রতিজ্ঞা sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        ফেরত model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, zero_stage):
        # start with all of the candidate parameters
        param_dict = {pn: p জন্য pn, p মধ্যে self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p জন্য pn, p মধ্যে param_dict.বস্তুসমূহ() যদি p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p জন্য n, p মধ্যে param_dict.বস্তুসমূহ() যদি p.dim() >= 2]
        nodecay_params = [p জন্য n, p মধ্যে param_dict.বস্তুসমূহ() যদি p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        অঙ্ক_decay_params = যোগ_2(p.numel() জন্য p মধ্যে decay_params)
        অঙ্ক_nodecay_params = যোগ_2(p.numel() জন্য p মধ্যে nodecay_params)
        print0(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print0(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' মধ্যে inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available এবং device_type == 'cuda'
        print0(f"using fused AdamW: {use_fused}")
        যদি zero_stage == 1:
            print0("using ZeroRedundancyOptimizer")
            optimizer = ZeroRedundancyOptimizer(**optim_groups[0], optimizer_class=torch.optim.AdamW,
                                                lr=learning_rate, betas=betas, fused=use_fused)
            optimizer.add_param_group(optim_groups[1])
        অন্যথা:
            print0("using regular AdamW")
            optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=use_fused)
        ফেরত optimizer

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, শীর্ষ_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        জন্য _ মধ্যে range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx যদি idx.আকার_2(1) <= self.config.block_আকার অন্যথা idx[:, -self.config.block_আকার:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            যদি শীর্ষ_k is নয় None:
                v, _ = torch.topk(logits, min(শীর্ষ_k, logits.আকার_2(-1)))
                logits[logits < v[:, [-1]]] = -দশমিক('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_পরবর্তী = torch.multinomial(probs, অঙ্ক_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_পরবর্তী), dim=1)

        ফেরত idx

# -----------------------------------------------------------------------------
# Our own simple Distributed Data Loader

def peek_তথ্য_shard(filename):
    # only reads the header, returns header data
    with open(filename, "rb") ইব f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.পঠ(256*4), dtype=np.int32)
    যদি header[0] != 20240520:
        print("ERROR: magic number mismatch in the data .bin file!")
        print("---> HINT: Are you passing in a correct file with --input_bin?")
        print("---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README")
        print("---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try")
        exit(1)
    প্রতিজ্ঞা header[1] == 1, "unsupported version"
    ntok = header[2] # number of tokens (claimed)
    ফেরত ntok # for now just return the number of tokens

def load_তথ্য_shard(filename):
    with open(filename, "rb") ইব f:
        # first read the header, which is 256 int32 integers (4 bytes each)
        header = np.frombuffer(f.পঠ(256*4), dtype=np.int32)
        প্রতিজ্ঞা header[0] == 20240520, "magic number mismatch in the data .bin file"
        প্রতিজ্ঞা header[1] == 1, "unsupported version"
        ntok = header[2] # number of tokens (claimed)
        # the rest of it are tokens, stored as uint16
        tokens = np.frombuffer(f.পঠ(), dtype=np.uint16)
    প্রতিজ্ঞা len(tokens) == ntok, "number of tokens read does not match header?"
    ফেরত tokens

শ্রেণী Distributed_তথ্য_Loader:
    def __init__(self, filename_pattern, B, T, process_rank, অঙ্ক_processes):
        self.process_rank = process_rank
        self.অঙ্ক_processes = অঙ্ক_processes
        self.B = B
        self.T = T

        # glob files that match the pattern
        self.files = sorted(glob.glob(filename_pattern))
        প্রতিজ্ঞা len(self.files) > 0, f"did not find any files that match the pattern {filename_pattern}"

        # load and validate all data shards, count number of tokens in total
        ntok_যোগ = 0
        জন্য fname মধ্যে self.files:
            shard_ntok = peek_তথ্য_shard(fname)
            প্রতিজ্ঞা shard_ntok >= অঙ্ক_processes * B * T + 1
            ntok_যোগ += shard_ntok
        self.ntok_যোগ = ntok_যোগ
        print0(f"DataLoader: total number of tokens: {ntok_total:,} across {len(self.files)} files")

        # kick things off
        self.current_shard = None
        self.reset()

    def reset(self):
        # we're being a bit clever here: if we already had shard 0 loaded,
        # then don't do the work to reload it, just reset the pointer
        যদি self.current_shard != 0:
            self.current_shard = 0
            self.tokens = load_তথ্য_shard(self.files[self.current_shard])
        self.current_position = self.process_rank * self.B * self.T

    def advance(self): # advance to next data shard
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.B * self.T
        self.tokens = load_তথ্য_shard(self.files[self.current_shard])

    def পরবর্তী_batch(self):
        B = self.B
        T = self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the start pointer in current shard
        self.current_position += B * T * self.অঙ্ক_processes
        # if loading the next batch would be out of bounds advance the shard
        যদি self.current_position + (B * T * self.অঙ্ক_processes + 1) > len(self.tokens):
            self.advance()
        ফেরত x, y

# -----------------------------------------------------------------------------
# Python -> C bridge utilities for saving params/grads/activations to .bin files

def write_fp32(tensor, file):
    t = tensor.detach().cpu().to(torch.float32)
    b = t.numpy().tobytes()
    file.লিখ(b)

def write_bf16(tensor, file):
    t = tensor.detach().cpu().to(torch.bfloat16)
    # numpy doesn't have bf16 datatype so we have to trick it
    t = t.view(torch.int16) # trick: reinterpret as int16
    b = t.numpy().tobytes()
    file.লিখ(b)

def write_tensors(model_tensors, L, file, dtype):
    # writes the GPT-2 model's weights to a binary file
    প্রতিজ্ঞা dtype মধ্যে {"float32", "bfloat16"}
    write_fun = write_fp32 যদি dtype == "float32" অন্যথা write_bf16
    write_fun(model_tensors["transformer.wte.weight"], file) # (V, C)
    write_fun(model_tensors["transformer.wpe.weight"], file) # (T, C)
    জন্য i মধ্যে range(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_1.weight"], file)
    জন্য i মধ্যে range(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_1.bias"], file)
    জন্য i মধ্যে range(L): # (L, 3C, C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_attn.weight"], file)
    জন্য i মধ্যে range(L): # (L, 3C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_attn.bias"], file)
    জন্য i মধ্যে range(L): # (L, C, C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_proj.weight"], file)
    জন্য i মধ্যে range(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.attn.c_proj.bias"], file)
    জন্য i মধ্যে range(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_2.weight"], file)
    জন্য i মধ্যে range(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.ln_2.bias"], file)
    জন্য i মধ্যে range(L): # (L, 4C, C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_fc.weight"], file)
    জন্য i মধ্যে range(L): # (L, 4C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_fc.bias"], file)
    জন্য i মধ্যে range(L): # (L, C, 4C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_proj.weight"], file)
    জন্য i মধ্যে range(L): # (L, C)
        write_fun(model_tensors[f"transformer.h.{i}.mlp.c_proj.bias"], file)
    write_fun(model_tensors["transformer.ln_f.weight"], file) # (C, )
    write_fun(model_tensors["transformer.ln_f.bias"], file) # (C, )

@torch.no_grad()
def pad_vocab(tensor, multiple=128, মান=0):
    """
    The dimension of the vocab size in GPT-2 is 50,257
    which is unfortunately a very unfriendly number for a lot of
    matrix operations on the GPU. So we pad it to the nearest
    friendlier multiple, e.g. 50,304 if multiple=128 when we
    export the weights into C land. This is a NOOP algorithmically
    and is only done to make the tensor operations more efficient.
    """
    প্রতিজ্ঞা tensor.ndim == 2
    V, C = tensor.shape
    প্রতিজ্ঞা V == 50257, "just being defensive here"
    # calculate padded vocab size by rounding up to nearest multiple
    Vp = ((V + multiple - 1) // multiple) * multiple
    # pad the tensor
    pad_rows = Vp - V
    padded = tensor যদি pad_rows == 0 অন্যথা F.pad(tensor, (0, 0, 0, pad_rows), মান=মান)
    প্রতিজ্ঞা padded.shape == (Vp, C)
    ফেরত padded

def write_model(model, filename, dtype):
    # everything we need to instantiate the model
    # 1) header is: version int, GPTConfig ints, padding to 1024 bytes
    প্রতিজ্ঞা dtype মধ্যে {"float32", "bfloat16"} # float16 todo maybe later
    version = {
        "float32": 3, # 3: all tensors are fp32, padded vocab
        "bfloat16": 5, # 5: all tensors are bf16, padded vocab
    }[dtype]
    header = torch.zeros(256, dtype=torch.int32)
    header[0] = 20240326 # magic
    header[1] = version # checkpoint version
    header[2] = model.config.block_আকার
    header[3] = model.config.vocab_আকার
    header[4] = model.config.n_layer
    header[5] = model.config.n_শীর্ষ
    header[6] = model.config.n_embd
    # 2) the parameters follow the header
    params = {name: param.cpu() জন্য name, param মধ্যে model.named_parameters()}
    # pad the vocab to a multiple of 128 here at export, for efficiency in C
    wte = params["transformer.wte.weight"] # (V, C)
    wte_padded = pad_vocab(wte) # (Vp, C)
    params["transformer.wte.weight"] = wte_padded # (Vp, C)
    print(f"padded vocab size from {wte.size(0)} to {wte_padded.size(0)}")
    header[7] = wte_padded.আকার_2(0) # padded vocab size store in header
    # now write to file
    with open(filename, "wb") ইব file:
        file.লিখ(header.numpy().tobytes()) # header
        write_tensors(params, model.config.n_layer, file, dtype) # params
    print(f"wrote {filename}")

def write_state(model, x, y, logits, loss, filename):
    # the state is used for debugging.
    # it contains information about the input, logits, loss, and the parameter gradients
    # this can be used for checking the computation correctness in C
    header = torch.zeros(256, dtype=torch.int32)
    header[0] = 20240327 # magic
    header[1] = 2 # run state version = 2 (1 -> 2 for padded vocab changes)
    header[2] = x.আকার_2(0) # batch size of the batch, B
    header[3] = x.আকার_2(1) # temporal extent of the batch, T
    grads = {name: param.grad.cpu() জন্য name, param মধ্যে model.named_parameters()}
    # pad the vocab grads here as well, to mirror write_model
    wte_grad = grads["transformer.wte.weight"] # (V, C)
    wte_grad_padded = pad_vocab(wte_grad, মান=0) # (Vp, C) # TODO later maybe pad with nan?
    grads["transformer.wte.weight"] = wte_grad_padded # (Vp, C)
    print(f"padded vocab size in reference grads from {wte_grad.size(0)} to {wte_grad_padded.size(0)}")
    with open(filename, "wb") ইব file:
        # header
        file.লিখ(header.numpy().tobytes())
        # input x
        file.লিখ(x.cpu().numpy().astype("int32").tobytes()) # (B, T)
        # targets y
        file.লিখ(y.cpu().numpy().astype("int32").tobytes()) # (B, T)
        # logits (result of the model forward pass)
        write_fp32(logits.cpu(), file)
        # loss (single float, result of the cross entropy loss)
        write_fp32(loss.cpu(), file)
        # gradients
        write_tensors(grads, model.config.n_layer, file, "float32")
    print(f"wrote {filename}")

def write_tokenizer(enc, filename):
    n = enc.max_token_মান + 1
    header = torch.zeros(256, dtype=torch.int32)
    header[0] = 20240328 # magic
    header[1] = 2 # tokenizer version = 2 (1 -> 2: includes EOT token)
    header[2] = n # number of tokens
    header[3] = enc.eot_token # EOT token
    with open(filename, "wb") ইব file:
        file.লিখ(header.numpy().tobytes())
        জন্য i মধ্যে range(n):
            b = enc.decode_bytes([i])
            দৈর্ঘ্য = len(b)
            প্রতিজ্ঞা দৈর্ঘ্য < 256, f"Token length exceeds 255: {length}"
            file.লিখ(struct.pack("<B", দৈর্ঘ্য))  # Write the length as a 1-byte unsigned integer
            file.লিখ(b)  # Write the actual bytes
    print(f"wrote {filename}")

# -----------------------------------------------------------------------------
# int main

def print0(*args, **kwargs):
    # modified print that only prints from the master process
    # if this is not a distributed run, it's just a print
    যদি পূর্ণসংখ্যা(os.environ.get("RANK", 0)) == 0:
        print(*args, **kwargs)

যদি __name__ == "__main__":
    আমদানি time
    আমদানি argparse
    আমদানি tiktoken
    print0(f"Running pytorch {torch.version.__version__}")

    # default settings will overfit a tiny batch of data
    # and save model weights and debug state to disk on the first iteration
    parser = argparse.ArgumentParser()
    # file system input / output
    parser.add_argument("--input_bin", type=str, পূর্বনির্ধারিত="dev/data/tinyshakespeare/tiny_shakespeare_val.bin", help="input .bin to train on")
    parser.add_argument("--input_val_bin", type=str, পূর্বনির্ধারিত="", help="input .bin to eval validation loss on")
    parser.add_argument("--output_dir", type=str, পূর্বনির্ধারিত="", help="output directory to which to write logs and checkpoints")
    parser.add_argument("--model", type=str, পূর্বনির্ধারিত="gpt2", help="gpt2|gpt2-medium|gpt2-large|gpt2-xl|d12|d24|d36|d48")
    # token layout for each step of the optimization
    parser.add_argument("--batch_size", type=পূর্ণসংখ্যা, পূর্বনির্ধারিত=4, help="batch size, in units of #batch dimensions")
    parser.add_argument("--sequence_length", type=পূর্ণসংখ্যা, পূর্বনির্ধারিত=64, help="sequence length")
    parser.add_argument("--total_batch_size", type=পূর্ণসংখ্যা, পূর্বনির্ধারিত=256, help="total desired batch size, in units of #tokens")
    # workload (number of steps)
    parser.add_argument("--num_iterations", type=পূর্ণসংখ্যা, পূর্বনির্ধারিত=10, help="number of iterations to run")
    parser.add_argument("--inference_only", type=পূর্ণসংখ্যা, পূর্বনির্ধারিত=0, help="only run inference")
    # optimization
    parser.add_argument("--learning_rate", type=দশমিক, পূর্বনির্ধারিত=1e-4, help="learning rate warmup iterations")
    parser.add_argument("--warmup_iters", type=পূর্ণসংখ্যা, পূর্বনির্ধারিত=0, help="learning rate warmup iterations")
    parser.add_argument("--learning_rate_decay_frac", type=দশমিক, পূর্বনির্ধারিত=1.0, help="learning rate warmup iterations")
    parser.add_argument("--weight_decay", type=দশমিক, পূর্বনির্ধারিত=0.0, help="weight decay")
    parser.add_argument("--grad_clip", type=দশমিক, পূর্বনির্ধারিত=1.0, help="maximum gradient magnitude")
    # evaluation
    parser.add_argument("--val_loss_every", type=পূর্ণসংখ্যা, পূর্বনির্ধারিত=0, help="every how mant steps to evaluate val loss?")
    parser.add_argument("--val_max_steps", type=পূর্ণসংখ্যা, পূর্বনির্ধারিত=20, help="how many batches of val to average?")
    parser.add_argument("--sample_every", type=পূর্ণসংখ্যা, পূর্বনির্ধারিত=0, help="how often to sample from the model?")
    # debugging
    parser.add_argument("--overfit_single_batch", type=পূর্ণসংখ্যা, পূর্বনির্ধারিত=1, help="overfit just one batch of data")
    # numerics
    parser.add_argument("--tensorcores", type=পূর্ণসংখ্যা, পূর্বনির্ধারিত=0, help="use tensorcores")
    # memory management
    parser.add_argument("--device", type=str, পূর্বনির্ধারিত="", help="by default we autodetect, or set it here")
    parser.add_argument("--compile", type=পূর্ণসংখ্যা, পূর্বনির্ধারিত=0, help="torch.compile the model")
    parser.add_argument("--flash", type=পূর্ণসংখ্যা, পূর্বনির্ধারিত=0, help="use flash attention")
    parser.add_argument("--dtype", type=str, পূর্বনির্ধারিত="float32", help="float32|float16|bfloat16")
    parser.add_argument("--zero_stage", type=পূর্ণসংখ্যা, পূর্বনির্ধারিত=0, help="zero redundancy optimizer stage (0/1/2/3)")
    # python -> C bridge
    parser.add_argument("--write_tensors", type=পূর্ণসংখ্যা, পূর্বনির্ধারিত=1, help="write tensors to disk")
    args = parser.parse_args()

    # args error checking and convenience variables
    B, T = args.batch_আকার, args.sequence_দৈর্ঘ্য
    প্রতিজ্ঞা 1 <= T <= 1024
    প্রতিজ্ঞা args.dtype মধ্যে {"float32", "float16", "bfloat16"}
    প্রতিজ্ঞা args.model মধ্যে {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl", "d12", "d24", "d36", "d48"}

    # set up DDP (distributed data parallel). torchrun sets this env variable
    ddp = পূর্ণসংখ্যা(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    যদি ddp:
        # use of DDP atm demands CUDA, we set the device appropriately according to rank
        প্রতিজ্ঞা torch.cuda.is_available(), "for now i think we need CUDA for DDP"
        init_process_group(backend='nccl')
        ddp_rank = পূর্ণসংখ্যা(os.environ['RANK'])
        ddp_local_rank = পূর্ণসংখ্যা(os.environ['LOCAL_RANK'])
        ddp_world_আকার = পূর্ণসংখ্যা(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        seed_offset = 0 # each process gets the exact same seed
        zero_stage = args.zero_stage
    অন্যথা:
        ddp_rank = 0
        ddp_local_rank = 0
        zero_stage = 0
        ddp_world_আকার = 1
        master_process = সত্য
        seed_offset = 0
        # select the device
        যদি args.device:
            # provided explicitly by the user
            device = args.device
        অন্যথা:
            # attempt to autodetect the device
            device = "cpu"
            যদি torch.cuda.is_available():
                device = "cuda"
            নতুবা_যদি hasattr(torch.backends, "mps") এবং torch.backends.mps.is_available():
                device = "mps"
    print(f"using device: {device}")
    device_type = 'cuda' যদি 'cuda' মধ্যে device অন্যথা 'cpu'

    # calculate gradient accumulation from the desired total batch size and the current run configuration
    tokens_per_fwdbwd = B * T * ddp_world_আকার
    প্রতিজ্ঞা args.যোগ_batch_আকার % tokens_per_fwdbwd == 0
    grad_accum_steps = args.যোগ_batch_আকার // tokens_per_fwdbwd
    print0(f"total desired batch size: {args.total_batch_size}")
    print0(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

    # set up a context manager following the desired dtype and device
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
    ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype) যদি device_type == "cuda" অন্যথা nullcontext()

    # rng / reproducibility
    torch.manual_seed(42)
    যদি torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # set the torch precision mode to use TensorFloat32 (TF32) for matmuls
    # docs https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
    যদি args.tensorcores:
        torch.set_float32_matmul_precision('high')

    # turn on/off flash attention
    প্রতিজ্ঞা args.flash মধ্যে {0, 1}
    FLASH = args.flash

    # init (and write) the tokenizer
    enc = tiktoken.get_encoding("gpt2")
    যদি master_process এবং args.write_tensors: # tokenizer is technically not tensors but ok
        write_tokenizer(enc, "gpt2_tokenizer.bin")

    # init the model, either from scratch or from OpenAI pretrained checkpoint
    যদি args.model[0] == "d":
        # from scratch (random weights)
        model_config = {
            "d12": GPTConfig(block_আকার=1024, vocab_আকার=50257, n_layer=12, n_শীর্ষ=12, n_embd=768),
            "d24": GPTConfig(block_আকার=1024, vocab_আকার=50257, n_layer=24, n_শীর্ষ=16, n_embd=1024),
            "d36": GPTConfig(block_আকার=1024, vocab_আকার=50257, n_layer=36, n_শীর্ষ=20, n_embd=1280),
            "d48": GPTConfig(block_আকার=1024, vocab_আকার=50257, n_layer=48, n_শীর্ষ=25, n_embd=1600),
        }[args.model]
        model = GPT(model_config)
    অন্যথা:
        # load the GPT-2 model weights
        model = GPT.from_pretrained(args.model)
    model.train()
    model.to(device)
    যদি args.compile:
        যদি hasattr(config, "coordinate_descent_tuning"):
            config.coordinate_descent_tuning = সত্য # suggested by @Chillee
        print0("compiling the model...")
        model = torch.compile(model)

    # -------------------------------------------------------------------------
    # Our own version of a simple DistributedDataLoader

    # load tokens
    train_loader = Distributed_তথ্য_Loader(args.প্রবেশ_bin, B, T, ddp_rank, ddp_world_আকার)
    val_loader = None
    যদি args.প্রবেশ_val_bin:
        val_loader = Distributed_তথ্য_Loader(args.প্রবেশ_val_bin, B, T, ddp_rank, ddp_world_আকার)

    # -------------------------------------------------------------------------
    # PyTorch -> C bridge: save some weights and state for C to load later as reference

    # do one forward pass to generate ground truth for our C tests
    যদি master_process এবং args.write_tensors এবং (নয় args.inference_only):
        x, y = train_loader.পরবর্তী_batch()
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, y)
        loss.backward()
        # save model params, in both float32 and bfloat16
        model_to_আকার = {"gpt2": "124M", "gpt2-medium": "355M", "gpt2-large": "774M", "gpt2-xl": "1558M"}
        model_to_আকার.update({f"d{d}": f"d{d}" জন্য d মধ্যে [12, 24, 36, 48]})
        model_আকার_str = model_to_আকার[args.model] # e.g. "124M", or "d12"
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
    যদি ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module যদি ddp অন্যথা model # always contains the "raw" unwrapped model

    # init the optimizer
    optimizer = raw_model.configure_optimizers(weight_decay=args.weight_decay,
                                               learning_rate=args.learning_rate, betas=(0.9, 0.95),
                                               device_type=device, zero_stage=zero_stage)

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        min_lr = args.learning_rate * args.learning_rate_decay_frac
        # 1) linear warmup for warmup_iters steps
        যদি it < args.warmup_iters:
            ফেরত args.learning_rate * (it+1) / args.warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        যদি it > args.অঙ্ক_iterations:
            ফেরত min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - args.warmup_iters) / (args.অঙ্ক_iterations - args.warmup_iters)
        প্রতিজ্ঞা 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff starts at 1 and goes to 0
        ফেরত min_lr + coeff * (args.learning_rate - min_lr)

    # create the logging directory if it does not exist
    logfile = None
    যদি args.output_dir:
        os.makedirs(args.output_dir, exist_ok=সত্য)
        logfile = os.path.join(args.output_dir, "main.log")
        # create the log file "main.log" inside it, and wipe it clean
        with open(logfile, "w") ইব f:
            pass

    যদি device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    timings = []
    norm = -1.0   # dummy value to print in inference-only mode
    জন্য step মধ্যে range(args.অঙ্ক_iterations + 1):
        t0 = time.time()
        last_step = (step == args.অঙ্ক_iterations)

        # once in a while evaluate the validation dataset
        যদি (args.val_loss_every > 0 \
            এবং (step % args.val_loss_every == 0 অথবা last_step)) \
            এবং (val_loader is নয় None):
            model.eval()
            val_loader.reset()
            with torch.no_grad():
                val_loss = 0.0
                জন্য _ মধ্যে range(args.val_max_steps):
                    x, y = val_loader.পরবর্তী_batch()
                    x, y = x.to(device), y.to(device)
                    _, loss = model(x, y, return_logits=মিথ্যা)
                    val_loss += loss.item()
                val_loss /= args.val_max_steps
            # log to console and to file
            print0(f"val loss {val_loss}")
            যদি master_process এবং logfile is নয় None:
                with open(logfile, "a") ইব f:
                    f.লিখ("s:%d tel:%f\n" % (step, val_loss))

        # once in a while perform model inference on the master process
        যদি (args.sample_every > 0 \
            এবং (step % args.sample_every == 0 অথবা last_step)) \
            এবং master_process:
            model.eval()
            # before we end, let's also do one round of inference
            # we'll kick off the generation with "<|endoftext|>", which designates the start of a new sequence
            start_ids = [enc.eot_token]
            xg = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
            max_new_tokens = 32
            temperature = 1.0
            শীর্ষ_k = 40
            yg = raw_model.generate(xg, max_new_tokens, temperature=temperature, শীর্ষ_k=শীর্ষ_k)
            print0('---------------')
            print0(enc.decode(yg[0].tolist()))
            print0('---------------')

        # bit confusing: we want to make sure to eval and sample on 0th iteration
        # but also after the very last iteration. so we loop for step <= num_iterations
        # instead of just < num_iterations (one extra due to <=), only to do
        # the validation/sampling one last time, and then we break right here as we're done.
        যদি last_step:
            বিরতি

        # --------------- TRAINING SECTION BEGIN -----------------
        model.train()
        optimizer.zero_grad(set_to_none=সত্য)
        # if we are trying to overfit a single batch, we reset the loader here
        যদি args.overfit_single_batch:
            train_loader.reset()
        # micro-batch loop where we do gradient accumulation to reach desired total batch size
        lossf = 0.0 # for getting the mean loss (as simple float) over the accumulation steps
        জন্য micro_step মধ্যে range(grad_accum_steps):
            # fetch a batch
            x, y = train_loader.পরবর্তী_batch()
            x, y = x.to(device), y.to(device)
            যদি ddp:
                # we want only the last micro-step to sync grads in a DDP model
                # the official way to do this is with model.no_sync(), but that is a
                # context manager that bloats the code, so we just toggle this variable
                model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
            # forward pass
            with ctx:
                _, loss = model(x, y, return_logits=মিথ্যা)
                # we have to scale the loss to account for gradient accumulation,
                # because the gradients just add on each successive backward().
                # addition of gradients corresponds to a SUM in the objective, but
                # instead of a SUM we want MEAN, so we scale the loss here
                loss = loss / grad_accum_steps
                lossf += loss.detach() # keep track of the mean loss
            # backward pass
            যদি নয় args.inference_only:
                loss.backward()
        যদি ddp:
            dist.all_reduce(lossf, op=dist.ReduceOp.AVG)
        lossf = lossf.item()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        # determine and set the learning rate for this iteration
        lr = get_lr(step)
        জন্য param_group মধ্যে optimizer.param_groups:
            param_group['lr'] = lr
        # step the optimizer
        optimizer.step()
        # --------------- TRAINING SECTION END -------------------
        # everything that follows now is just diagnostics, prints, logging, etc.

        # wait on the CPU for all device work to end so we get accurate per-iteration timings below
        যদি device == "mps":
            torch.mps.synchronize()
        নতুবা_যদি device == "cuda":
            torch.cuda.synchronize()
        # time and print
        t1 = time.time()
        # the 0th iteration is often an outlier (much slower) => skip logging it
        tokens_per_second = grad_accum_steps * ddp_world_আকার * B * T / (t1-t0)
        print0(f"step {step+1:4d}/{args.num_iterations} | train loss {lossf:.6f} | norm {norm:.4f} | lr {lr:.2e} | ({(t1-t0)*1000:.2f} ms | {tokens_per_second:.0f} tok/s)")
        # log to logile
        যদি master_process এবং logfile is নয় None:
            with open(logfile, "a") ইব f:
                f.লিখ("s:%d trl:%f\n" % (step, lossf))

        # keep track of smooth timings, last 20 iterations
        যদি step > 0 এবং step > args.অঙ্ক_iterations - 20:
            timings.append(t1-t0)

    # print the average of the last 20 timings, to get something smooth-ish
    timings = timings[-20:]
    print0(f"final {len(timings)} iters avg: {np.mean(timings)*1000:.3f}ms")
    print0(f"peak memory consumption: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB")

    # -------------------------------------------------------------------------
    # clean up nice
    যদি ddp:
        destroy_process_group()
