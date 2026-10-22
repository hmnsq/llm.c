"""
Script to convert GPT2 models from llm.c binary format to Hugging Face

It can optinally upload to your account on Hugging Face if you have the CLI:
  pip install -U "huggingface_hub[cli]"
  huggingface-cli login

Export to a local HF model:
  python export_hf.py --input input_file.bin --output output_dir

Export to a local HF model and also push to your account on Hugging Face:
  python export_hf.py --input input_file.bin --output output_dir --push true
"""

ਆਯਾਤ numpy ਇਵ np
ਆਯਾਤ torch
ਆਯਾਤ argparse, sys
ਤਃ transformers ਆਯਾਤ GPT2Config, GPT2Tokenizer, GPT2LMHeadModel

# -----------------------------------------------------------------------------
# Tensor functions for both bfloat16 (from int16) and normal float32
# Both return float32 tensors

ਪਰਿਭਾਸ਼ਾ tensor_bf16(ਅੰਕੜੇ_int16, transpose=ਝੂਠ):
    ਜੇ transpose:
        ਅੰਕੜੇ_int16 = ਅੰਕੜੇ_int16.transpose(1,0)
    ਵਾਪਸ torch.tensor(ਅੰਕੜੇ_int16).view(torch.bfloat16).ਮੰਜ਼ਿਲ(torch.float32)

ਪਰਿਭਾਸ਼ਾ tensor_fp32(ਅੰਕੜੇ_float32, transpose=ਝੂਠ):
    ਜੇ transpose:
        ਅੰਕੜੇ_float32 = ਅੰਕੜੇ_float32.transpose(1,0)
    ਵਾਪਸ torch.tensor(ਅੰਕੜੇ_float32).view(torch.float32)

# -----------------------------------------------------------------------------
# Main conversion function

ਪਰਿਭਾਸ਼ਾ convert(filepath, output, ਧੱਕੋ_ਮੰਜ਼ਿਲ_hub=ਝੂਠ, ਬਾਹਰਲਾ_dtype="bfloat16"):
    ਛਾਪੋ(f"Converting model {filepath} to {output} in {out_dtype} format and pushing to Hugging Face: {push_to_hub}")

    f = open(filepath, 'rb')
    # Read in our header, checking the magic number and version
    # version 3 = fp32, padded vocab
    # version 5 = bf16, padded vocab
    model_ਸਿਰਲੇਖ = np.frombuffer(f.ਪਠ(256*4), dtype=np.int32)
    ਜੇ model_ਸਿਰਲੇਖ[0] != 20240326:
        ਛਾਪੋ("ERROR: magic number mismatch in the data .bin file!")
        exit(1)
    ਸੰਸਕਰਣ = model_ਸਿਰਲੇਖ[1]
    ਜੇ ਨਹੀਂ ਸੰਸਕਰਣ ਵਿੱਚ [3, 5]:
        ਛਾਪੋ("Bad version in model file")
        exit(1)

    # Load in our model parameters
    maxT = model_ਸਿਰਲੇਖ[2].ਵਸਤੂ_2() # max sequence length
    V = model_ਸਿਰਲੇਖ[3].ਵਸਤੂ_2() # vocab size
    L =  model_ਸਿਰਲੇਖ[4].ਵਸਤੂ_2() # num layers
    H = model_ਸਿਰਲੇਖ[5].ਵਸਤੂ_2() # num heads
    C = model_ਸਿਰਲੇਖ[6].ਵਸਤੂ_2() # channels
    Vp = model_ਸਿਰਲੇਖ[7].ਵਸਤੂ_2() # padded vocab size

    ਛਾਪੋ(f"{version=}, {maxT=}, {V=}, {Vp=}, {L=}, {H=}, {C=}")

    # Define the shapes of our parameters
    shapes = {
        'wte': (Vp, C),
        'wpe': (maxT, C),
        'ln1w': (L, C),
        'ln1b': (L, C),
        'qkvw': (L, 3 * C, C),
        'qkvb': (L, 3 * C),
        'attprojw': (L, C, C),
        'attprojb': (L, C),
        'ln2w': (L, C),
        'ln2b': (L, C),
        'fcw': (L, 4 * C, C),
        'fcb': (L, 4 * C),
        'fcprojw': (L, C, 4 * C),
        'fcprojb': (L, C),
        'lnfw': (C,),
        'lnfb': (C,),
    }

    # Load in our weights given our parameter shapes
    dtype = np.float32 ਜੇ ਸੰਸਕਰਣ == 3 ਨਹੀਂਤਾਂ np.int16
    w = {}
    ਲਈ ਕੁੰਜੀ, shape ਵਿੱਚ shapes.ਚੀਜ਼ਾਂ():
        ਅੰਕ_elements = np.prod(shape)
        ਅੰਕੜੇ = np.frombuffer(f.ਪਠ(ਅੰਕ_elements * np.dtype(dtype).itemsize), dtype=dtype)
        w[ਕੁੰਜੀ] = ਅੰਕੜੇ.reshape(shape)
        # The binary file saves the padded vocab - drop the padding back to GPT2 size
        ਜੇ shape[0] == Vp:
            w[ਕੁੰਜੀ] = w[ਕੁੰਜੀ].reshape(shape)[:(V-Vp), :]
    # Ensure the file is fully read and then close
    ਪ੍ਰਤਿਜ੍ਞਾ f.ਪਠ() == b''
    f.close()

    # Map to our model dict, the tensors at this stage are always fp32
    mk_tensor = {
        3 : tensor_fp32,
        5 : tensor_bf16,
    }[ਸੰਸਕਰਣ]
    model_dict = {}
    model_dict['transformer.wte.weight'] = mk_tensor(w['wte'])
    model_dict['transformer.wpe.weight'] = mk_tensor(w['wpe'])
    model_dict['lm_head.weight'] = model_dict['transformer.wte.weight'] # Tie weights
    ਲਈ i ਵਿੱਚ ਸੀਮਾ(L):
        model_dict[f'transformer.h.{i}.ln_1.weight'] = mk_tensor(w['ln1w'][i])
        model_dict[f'transformer.h.{i}.ln_1.bias'] = mk_tensor(w['ln1b'][i])
        model_dict[f'transformer.h.{i}.attn.c_attn.weight'] = mk_tensor(w['qkvw'][i], ਸੱਚ)
        model_dict[f'transformer.h.{i}.attn.c_attn.bias'] = mk_tensor(w['qkvb'][i])
        model_dict[f'transformer.h.{i}.attn.c_proj.weight'] = mk_tensor(w['attprojw'][i], ਸੱਚ)
        model_dict[f'transformer.h.{i}.attn.c_proj.bias'] = mk_tensor(w['attprojb'][i])
        model_dict[f'transformer.h.{i}.ln_2.weight'] = mk_tensor(w['ln2w'][i])
        model_dict[f'transformer.h.{i}.ln_2.bias'] = mk_tensor(w['ln2b'][i])
        model_dict[f'transformer.h.{i}.mlp.c_fc.weight'] = mk_tensor(w['fcw'][i], ਸੱਚ)
        model_dict[f'transformer.h.{i}.mlp.c_fc.bias'] = mk_tensor(w['fcb'][i])
        model_dict[f'transformer.h.{i}.mlp.c_proj.weight'] = mk_tensor(w['fcprojw'][i], ਸੱਚ)
        model_dict[f'transformer.h.{i}.mlp.c_proj.bias'] = mk_tensor(w['fcprojb'][i])
    model_dict['transformer.ln_f.weight'] = mk_tensor(w['lnfw'])
    model_dict['transformer.ln_f.bias'] = mk_tensor(w['lnfb'])

    # Create a GPT-2 model instance, in the requested dtype
    config = GPT2Config(vocab_ਮਾਪ = V,
                        n_positions = maxT,
                        n_ctx = maxT,
                        n_embd = C,
                        n_layer = L,
                        n_ਸਿਰ = H)
    model = GPT2LMHeadModel(config)
    ਜੇ ਬਾਹਰਲਾ_dtype == "bfloat16":
        model = model.ਮੰਜ਼ਿਲ(torch.bfloat16)

    # Set the model dict and save
    model.ਲੱਦੋ_state_dict(model_dict)
    model.save_pretrained(output, max_shard_ਮਾਪ="5GB", ਸੁਰੱਖਿਅਤ_serialization=ਸੱਚ)

    # Copy over a standard gpt2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.save_pretrained(output)

    ਜੇ ਧੱਕੋ_ਮੰਜ਼ਿਲ_hub:
        ਛਾਪੋ(f"Uploading {output} to Hugging Face")
        model.ਧੱਕੋ_ਮੰਜ਼ਿਲ_hub(output)
        tokenizer.ਧੱਕੋ_ਮੰਜ਼ਿਲ_hub(output)

ਪਰਿਭਾਸ਼ਾ spin(output):
    ਛਾਪੋ("Taking the exported model for a spin...")
    ਛਾਪੋ('-'*80)
    ਤਃ transformers ਆਯਾਤ AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(output)
    model = AutoModelForCausalLM.from_pretrained(output, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16, device_map='cuda')
    model.eval()
    ਟੋਕਨ = tokenizer.encode("During photosynthesis in green plants", return_tensors="pt")
    ਟੋਕਨ = ਟੋਕਨ.ਮੰਜ਼ਿਲ('cuda')
    output = model.generate(ਟੋਕਨ, max_new_ਟੋਕਨ=64, repetition_penalty=1.3)
    samples = tokenizer.batch_decode(output)
    ਲਈ sample ਵਿੱਚ samples:
        ਛਾਪੋ('-'*30)
        ਛਾਪੋ(sample)

# -----------------------------------------------------------------------------

ਜੇ __name__== '__main__':
    parser=argparse.ਦਲੀਲ_Parser()
    parser.ਜੋੜੋ_ਦਲੀਲ("--input", "-i", help="The name of the llm.c model.bin file", type=str, required=ਸੱਚ)
    parser.ਜੋੜੋ_ਦਲੀਲ("--output","-o",  help="The Hugging Face output model directory", type=str, required=ਸੱਚ)
    parser.ਜੋੜੋ_ਦਲੀਲ("--dtype", "-d", help="Output as either float32 or bfloat16 (default)", type=str, ਮੂਲ="bfloat16")
    parser.ਜੋੜੋ_ਦਲੀਲ("--push", "-p", help="Push the model to your Hugging Face account", type=bool, ਮੂਲ=ਝੂਠ)
    parser.ਜੋੜੋ_ਦਲੀਲ("--spin", "-s", help="Take the model for a spin at the end?", type=bool, ਮੂਲ=ਸੱਚ)
    ਦਲੀਲਾਂ = parser.parse_ਦਲੀਲਾਂ()
    convert(ਦਲੀਲਾਂ.ਨਿਵੇਸ਼, ਦਲੀਲਾਂ.output, ਦਲੀਲਾਂ.ਧੱਕੋ, ਦਲੀਲਾਂ.dtype)
    ਜੇ ਦਲੀਲਾਂ.spin:
        spin(ਦਲੀਲਾਂ.output)
