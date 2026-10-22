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

درآمد numpy بطور np
درآمد torch
درآمد argparse, sys
سے transformers درآمد GPT2Config, GPT2Tokenizer, GPT2LMHeadModel

# -----------------------------------------------------------------------------
# Tensor functions for both bfloat16 (from int16) and normal float32
# Both return float32 tensors

تعریف tensor_bf16(ڈیٹا_int16, transpose=جھوٹ):
    اگر transpose:
        ڈیٹا_int16 = ڈیٹا_int16.transpose(1,0)
    واپس torch.tensor(ڈیٹا_int16).view(torch.bfloat16).مقصد(torch.float32)

تعریف tensor_fp32(ڈیٹا_float32, transpose=جھوٹ):
    اگر transpose:
        ڈیٹا_float32 = ڈیٹا_float32.transpose(1,0)
    واپس torch.tensor(ڈیٹا_float32).view(torch.float32)

# -----------------------------------------------------------------------------
# Main conversion function

تعریف convert(filepath, output, ڈالو_مقصد_hub=جھوٹ, برآمد_dtype="bfloat16"):
    چھاپو(f"Converting model {filepath} to {output} in {out_dtype} format and pushing to Hugging Face: {push_to_hub}")

    f = open(filepath, 'rb')
    # Read in our header, checking the magic number and version
    # version 3 = fp32, padded vocab
    # version 5 = bf16, padded vocab
    model_سرخی = np.frombuffer(f.اقرأ(256*4), dtype=np.int32)
    اگر model_سرخی[0] != 20240326:
        چھاپو("ERROR: magic number mismatch in the data .bin file!")
        exit(1)
    نسخہ = model_سرخی[1]
    اگر نہیں نسخہ میں [3, 5]:
        چھاپو("Bad version in model file")
        exit(1)

    # Load in our model parameters
    maxT = model_سرخی[2].شے_2() # max sequence length
    V = model_سرخی[3].شے_2() # vocab size
    L =  model_سرخی[4].شے_2() # num layers
    H = model_سرخی[5].شے_2() # num heads
    C = model_سرخی[6].شے_2() # channels
    Vp = model_سرخی[7].شے_2() # padded vocab size

    چھاپو(f"{version=}, {maxT=}, {V=}, {Vp=}, {L=}, {H=}, {C=}")

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
    dtype = np.float32 اگر نسخہ == 3 ورنہ np.int16
    w = {}
    برائے کلید, shape میں shapes.اشیاء():
        عدد_elements = np.prod(shape)
        ڈیٹا = np.frombuffer(f.اقرأ(عدد_elements * np.dtype(dtype).itemsize), dtype=dtype)
        w[کلید] = ڈیٹا.reshape(shape)
        # The binary file saves the padded vocab - drop the padding back to GPT2 size
        اگر shape[0] == Vp:
            w[کلید] = w[کلید].reshape(shape)[:(V-Vp), :]
    # Ensure the file is fully read and then close
    أكد f.اقرأ() == b''
    f.close()

    # Map to our model dict, the tensors at this stage are always fp32
    mk_tensor = {
        3 : tensor_fp32,
        5 : tensor_bf16,
    }[نسخہ]
    model_dict = {}
    model_dict['transformer.wte.weight'] = mk_tensor(w['wte'])
    model_dict['transformer.wpe.weight'] = mk_tensor(w['wpe'])
    model_dict['lm_head.weight'] = model_dict['transformer.wte.weight'] # Tie weights
    برائے i میں سلسلہ(L):
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
    config = GPT2Config(vocab_سائز = V,
                        n_positions = maxT,
                        n_ctx = maxT,
                        n_embd = C,
                        n_layer = L,
                        n_سر = H)
    model = GPT2LMHeadModel(config)
    اگر برآمد_dtype == "bfloat16":
        model = model.مقصد(torch.bfloat16)

    # Set the model dict and save
    model.بھرو_state_dict(model_dict)
    model.save_pretrained(output, max_shard_سائز="5GB", محفوظ_serialization=سچ)

    # Copy over a standard gpt2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.save_pretrained(output)

    اگر ڈالو_مقصد_hub:
        چھاپو(f"Uploading {output} to Hugging Face")
        model.ڈالو_مقصد_hub(output)
        tokenizer.ڈالو_مقصد_hub(output)

تعریف spin(output):
    چھاپو("Taking the exported model for a spin...")
    چھاپو('-'*80)
    سے transformers درآمد AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(output)
    model = AutoModelForCausalLM.from_pretrained(output, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16, device_map='cuda')
    model.eval()
    علامات = tokenizer.encode("During photosynthesis in green plants", return_tensors="pt")
    علامات = علامات.مقصد('cuda')
    output = model.generate(علامات, max_new_علامات=64, repetition_penalty=1.3)
    samples = tokenizer.batch_decode(output)
    برائے sample میں samples:
        چھاپو('-'*30)
        چھاپو(sample)

# -----------------------------------------------------------------------------

اگر __name__== '__main__':
    parser=argparse.دلیل_Parser()
    parser.جوڑو_دلیل("--input", "-i", help="The name of the llm.c model.bin file", type=str, required=سچ)
    parser.جوڑو_دلیل("--output","-o",  help="The Hugging Face output model directory", type=str, required=سچ)
    parser.جوڑو_دلیل("--dtype", "-d", help="Output as either float32 or bfloat16 (default)", type=str, طےشدہ="bfloat16")
    parser.جوڑو_دلیل("--push", "-p", help="Push the model to your Hugging Face account", type=bool, طےشدہ=جھوٹ)
    parser.جوڑو_دلیل("--spin", "-s", help="Take the model for a spin at the end?", type=bool, طےشدہ=سچ)
    دلائل = parser.parse_دلائل()
    convert(دلائل.داخلہ, دلائل.output, دلائل.ڈالو, دلائل.dtype)
    اگر دلائل.spin:
        spin(دلائل.output)
