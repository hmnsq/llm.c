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

استورد numpy كـ np
استورد torch
استورد argparse, sys
من transformers استورد GPT2Config, GPT2Tokenizer, GPT2LMHeadModel

# -----------------------------------------------------------------------------
# Tensor functions for both bfloat16 (from int16) and normal float32
# Both return float32 tensors

وصف tensor_bf16(معلومات_int16, transpose=خطأ):
    إذا transpose:
        معلومات_int16 = معلومات_int16.transpose(1,0)
    أرجع torch.tensor(معلومات_int16).view(torch.bfloat16).منزل(torch.float32)

وصف tensor_fp32(معلومات_float32, transpose=خطأ):
    إذا transpose:
        معلومات_float32 = معلومات_float32.transpose(1,0)
    أرجع torch.tensor(معلومات_float32).view(torch.float32)

# -----------------------------------------------------------------------------
# Main conversion function

وصف convert(filepath, output, ڌڪو_منزل_hub=خطأ, ٻاهريون_dtype="bfloat16"):
    ڇاپيو(f"Converting model {filepath} to {output} in {out_dtype} format and pushing to Hugging Face: {push_to_hub}")

    f = open(filepath, 'rb')
    # Read in our header, checking the magic number and version
    # version 3 = fp32, padded vocab
    # version 5 = bf16, padded vocab
    model_سرو = np.frombuffer(f.اقرأ(256*4), dtype=np.int32)
    إذا model_سرو[0] != 20240326:
        ڇاپيو("ERROR: magic number mismatch in the data .bin file!")
        exit(1)
    نسخو = model_سرو[1]
    إذا ليس نسخو في [3, 5]:
        ڇاپيو("Bad version in model file")
        exit(1)

    # Load in our model parameters
    maxT = model_سرو[2].شي() # max sequence length
    V = model_سرو[3].شي() # vocab size
    L =  model_سرو[4].شي() # num layers
    H = model_سرو[5].شي() # num heads
    C = model_سرو[6].شي() # channels
    Vp = model_سرو[7].شي() # padded vocab size

    ڇاپيو(f"{version=}, {maxT=}, {V=}, {Vp=}, {L=}, {H=}, {C=}")

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
    dtype = np.float32 إذا نسخو == 3 وإلا np.int16
    w = {}
    لكل ڪنجي, shape في shapes.شيون():
        انگ_elements = np.prod(shape)
        معلومات = np.frombuffer(f.اقرأ(انگ_elements * np.dtype(dtype).itemsize), dtype=dtype)
        w[ڪنجي] = معلومات.reshape(shape)
        # The binary file saves the padded vocab - drop the padding back to GPT2 size
        إذا shape[0] == Vp:
            w[ڪنجي] = w[ڪنجي].reshape(shape)[:(V-Vp), :]
    # Ensure the file is fully read and then close
    أكد f.اقرأ() == b''
    f.close()

    # Map to our model dict, the tensors at this stage are always fp32
    mk_tensor = {
        3 : tensor_fp32,
        5 : tensor_bf16,
    }[نسخو]
    model_dict = {}
    model_dict['transformer.wte.weight'] = mk_tensor(w['wte'])
    model_dict['transformer.wpe.weight'] = mk_tensor(w['wpe'])
    model_dict['lm_head.weight'] = model_dict['transformer.wte.weight'] # Tie weights
    لكل i في سلسلو(L):
        model_dict[f'transformer.h.{i}.ln_1.weight'] = mk_tensor(w['ln1w'][i])
        model_dict[f'transformer.h.{i}.ln_1.bias'] = mk_tensor(w['ln1b'][i])
        model_dict[f'transformer.h.{i}.attn.c_attn.weight'] = mk_tensor(w['qkvw'][i], سچ)
        model_dict[f'transformer.h.{i}.attn.c_attn.bias'] = mk_tensor(w['qkvb'][i])
        model_dict[f'transformer.h.{i}.attn.c_proj.weight'] = mk_tensor(w['attprojw'][i], سچ)
        model_dict[f'transformer.h.{i}.attn.c_proj.bias'] = mk_tensor(w['attprojb'][i])
        model_dict[f'transformer.h.{i}.ln_2.weight'] = mk_tensor(w['ln2w'][i])
        model_dict[f'transformer.h.{i}.ln_2.bias'] = mk_tensor(w['ln2b'][i])
        model_dict[f'transformer.h.{i}.mlp.c_fc.weight'] = mk_tensor(w['fcw'][i], سچ)
        model_dict[f'transformer.h.{i}.mlp.c_fc.bias'] = mk_tensor(w['fcb'][i])
        model_dict[f'transformer.h.{i}.mlp.c_proj.weight'] = mk_tensor(w['fcprojw'][i], سچ)
        model_dict[f'transformer.h.{i}.mlp.c_proj.bias'] = mk_tensor(w['fcprojb'][i])
    model_dict['transformer.ln_f.weight'] = mk_tensor(w['lnfw'])
    model_dict['transformer.ln_f.bias'] = mk_tensor(w['lnfb'])

    # Create a GPT-2 model instance, in the requested dtype
    config = GPT2Config(vocab_سائيز = V,
                        n_positions = maxT,
                        n_ctx = maxT,
                        n_embd = C,
                        n_layer = L,
                        n_سر = H)
    model = GPT2LMHeadModel(config)
    إذا ٻاهريون_dtype == "bfloat16":
        model = model.منزل(torch.bfloat16)

    # Set the model dict and save
    model.ڀريو_state_dict(model_dict)
    model.save_pretrained(output, max_shard_سائيز="5GB", محفوظ_serialization=سچ)

    # Copy over a standard gpt2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.save_pretrained(output)

    إذا ڌڪو_منزل_hub:
        ڇاپيو(f"Uploading {output} to Hugging Face")
        model.ڌڪو_منزل_hub(output)
        tokenizer.ڌڪو_منزل_hub(output)

وصف spin(output):
    ڇاپيو("Taking the exported model for a spin...")
    ڇاپيو('-'*80)
    من transformers استورد AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(output)
    model = AutoModelForCausalLM.from_pretrained(output, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16, device_map='cuda')
    model.eval()
    ٽوڪن = tokenizer.encode("During photosynthesis in green plants", return_tensors="pt")
    ٽوڪن = ٽوڪن.منزل('cuda')
    output = model.generate(ٽوڪن, max_new_ٽوڪن=64, repetition_penalty=1.3)
    samples = tokenizer.batch_decode(output)
    لكل sample في samples:
        ڇاپيو('-'*30)
        ڇاپيو(sample)

# -----------------------------------------------------------------------------

إذا __name__== '__main__':
    parser=argparse.دليل_Parser()
    parser.جوڙيو_دليل("--input", "-i", help="The name of the llm.c model.bin file", type=str, required=سچ)
    parser.جوڙيو_دليل("--output","-o",  help="The Hugging Face output model directory", type=str, required=سچ)
    parser.جوڙيو_دليل("--dtype", "-d", help="Output as either float32 or bfloat16 (default)", type=str, افتراضي="bfloat16")
    parser.جوڙيو_دليل("--push", "-p", help="Push the model to your Hugging Face account", type=bool, افتراضي=خطأ)
    parser.جوڙيو_دليل("--spin", "-s", help="Take the model for a spin at the end?", type=bool, افتراضي=سچ)
    دليلون = parser.parse_دليلون()
    convert(دليلون.داخلا, دليلون.output, دليلون.ڌڪو, دليلون.dtype)
    إذا دليلون.spin:
        spin(دليلون.output)
