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

ಆಮದು numpy ಇವ np
ಆಮದು torch
ಆಮದು argparse, sys
ತಃ transformers ಆಮದು GPT2Config, GPT2Tokenizer, GPT2LMHeadModel

# -----------------------------------------------------------------------------
# Tensor functions for both bfloat16 (from int16) and normal float32
# Both return float32 tensors

ವ್ಯಾಖ್ಯಾನ tensor_bf16(ದತ್ತಾಂಶ_int16, transpose=ಸುಳ್ಳು):
    ಆದರೆ transpose:
        ದತ್ತಾಂಶ_int16 = ದತ್ತಾಂಶ_int16.transpose(1,0)
    ಹಿಂತಿರುಗಿ torch.tensor(ದತ್ತಾಂಶ_int16).view(torch.bfloat16).ಗಮ್ಯ(torch.float32)

ವ್ಯಾಖ್ಯಾನ tensor_fp32(ದತ್ತಾಂಶ_float32, transpose=ಸುಳ್ಳು):
    ಆದರೆ transpose:
        ದತ್ತಾಂಶ_float32 = ದತ್ತಾಂಶ_float32.transpose(1,0)
    ಹಿಂತಿರುಗಿ torch.tensor(ದತ್ತಾಂಶ_float32).view(torch.float32)

# -----------------------------------------------------------------------------
# Main conversion function

ವ್ಯಾಖ್ಯಾನ convert(filepath, output, ತಳ್ಳು_ಗಮ್ಯ_hub=ಸುಳ್ಳು, ಹೊರಹೋಗು_dtype="bfloat16"):
    ಮುದ್ರಿಸು(f"Converting model {filepath} to {output} in {out_dtype} format and pushing to Hugging Face: {push_to_hub}")

    f = open(filepath, 'rb')
    # Read in our header, checking the magic number and version
    # version 3 = fp32, padded vocab
    # version 5 = bf16, padded vocab
    model_ಶೀರ್ಷಿಕೆ = np.frombuffer(f.ಪಠ(256*4), dtype=np.int32)
    ಆದರೆ model_ಶೀರ್ಷಿಕೆ[0] != 20240326:
        ಮುದ್ರಿಸು("ERROR: magic number mismatch in the data .bin file!")
        exit(1)
    ಆವೃತ್ತಿ = model_ಶೀರ್ಷಿಕೆ[1]
    ಆದರೆ ಅಲ್ಲ ಆವೃತ್ತಿ ಒಳಗೆ [3, 5]:
        ಮುದ್ರಿಸು("Bad version in model file")
        exit(1)

    # Load in our model parameters
    maxT = model_ಶೀರ್ಷಿಕೆ[2].ಅಂಶ() # max sequence length
    V = model_ಶೀರ್ಷಿಕೆ[3].ಅಂಶ() # vocab size
    L =  model_ಶೀರ್ಷಿಕೆ[4].ಅಂಶ() # num layers
    H = model_ಶೀರ್ಷಿಕೆ[5].ಅಂಶ() # num heads
    C = model_ಶೀರ್ಷಿಕೆ[6].ಅಂಶ() # channels
    Vp = model_ಶೀರ್ಷಿಕೆ[7].ಅಂಶ() # padded vocab size

    ಮುದ್ರಿಸು(f"{version=}, {maxT=}, {V=}, {Vp=}, {L=}, {H=}, {C=}")

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
    dtype = np.float32 ಆದರೆ ಆವೃತ್ತಿ == 3 ಇಲ್ಲದಿದ್ದರೆ np.int16
    w = {}
    ಗಾಗಿ ಕೀಲಿ, shape ಒಳಗೆ shapes.ವಸ್ತುಗಳು():
        ಅಂಕೆ_elements = np.prod(shape)
        ದತ್ತಾಂಶ = np.frombuffer(f.ಪಠ(ಅಂಕೆ_elements * np.dtype(dtype).itemsize), dtype=dtype)
        w[ಕೀಲಿ] = ದತ್ತಾಂಶ.reshape(shape)
        # The binary file saves the padded vocab - drop the padding back to GPT2 size
        ಆದರೆ shape[0] == Vp:
            w[ಕೀಲಿ] = w[ಕೀಲಿ].reshape(shape)[:(V-Vp), :]
    # Ensure the file is fully read and then close
    ಪ್ರತಿಜ್ಞಾ f.ಪಠ() == b''
    f.close()

    # Map to our model dict, the tensors at this stage are always fp32
    mk_tensor = {
        3 : tensor_fp32,
        5 : tensor_bf16,
    }[ಆವೃತ್ತಿ]
    model_dict = {}
    model_dict['transformer.wte.weight'] = mk_tensor(w['wte'])
    model_dict['transformer.wpe.weight'] = mk_tensor(w['wpe'])
    model_dict['lm_head.weight'] = model_dict['transformer.wte.weight'] # Tie weights
    ಗಾಗಿ i ಒಳಗೆ ಶ್ರೇಣಿ(L):
        model_dict[f'transformer.h.{i}.ln_1.weight'] = mk_tensor(w['ln1w'][i])
        model_dict[f'transformer.h.{i}.ln_1.bias'] = mk_tensor(w['ln1b'][i])
        model_dict[f'transformer.h.{i}.attn.c_attn.weight'] = mk_tensor(w['qkvw'][i], ನಿಜ)
        model_dict[f'transformer.h.{i}.attn.c_attn.bias'] = mk_tensor(w['qkvb'][i])
        model_dict[f'transformer.h.{i}.attn.c_proj.weight'] = mk_tensor(w['attprojw'][i], ನಿಜ)
        model_dict[f'transformer.h.{i}.attn.c_proj.bias'] = mk_tensor(w['attprojb'][i])
        model_dict[f'transformer.h.{i}.ln_2.weight'] = mk_tensor(w['ln2w'][i])
        model_dict[f'transformer.h.{i}.ln_2.bias'] = mk_tensor(w['ln2b'][i])
        model_dict[f'transformer.h.{i}.mlp.c_fc.weight'] = mk_tensor(w['fcw'][i], ನಿಜ)
        model_dict[f'transformer.h.{i}.mlp.c_fc.bias'] = mk_tensor(w['fcb'][i])
        model_dict[f'transformer.h.{i}.mlp.c_proj.weight'] = mk_tensor(w['fcprojw'][i], ನಿಜ)
        model_dict[f'transformer.h.{i}.mlp.c_proj.bias'] = mk_tensor(w['fcprojb'][i])
    model_dict['transformer.ln_f.weight'] = mk_tensor(w['lnfw'])
    model_dict['transformer.ln_f.bias'] = mk_tensor(w['lnfb'])

    # Create a GPT-2 model instance, in the requested dtype
    config = GPT2Config(vocab_ಗಾತ್ರ = V,
                        n_positions = maxT,
                        n_ctx = maxT,
                        n_embd = C,
                        n_layer = L,
                        n_ತಲೆ = H)
    model = GPT2LMHeadModel(config)
    ಆದರೆ ಹೊರಹೋಗು_dtype == "bfloat16":
        model = model.ಗಮ್ಯ(torch.bfloat16)

    # Set the model dict and save
    model.ತುಂಬು_state_dict(model_dict)
    model.save_pretrained(output, max_shard_ಗಾತ್ರ="5GB", ಸುರಕ್ಷಿತ_serialization=ನಿಜ)

    # Copy over a standard gpt2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.save_pretrained(output)

    ಆದರೆ ತಳ್ಳು_ಗಮ್ಯ_hub:
        ಮುದ್ರಿಸು(f"Uploading {output} to Hugging Face")
        model.ತಳ್ಳು_ಗಮ್ಯ_hub(output)
        tokenizer.ತಳ್ಳು_ಗಮ್ಯ_hub(output)

ವ್ಯಾಖ್ಯಾನ spin(output):
    ಮುದ್ರಿಸು("Taking the exported model for a spin...")
    ಮುದ್ರಿಸು('-'*80)
    ತಃ transformers ಆಮದು AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(output)
    model = AutoModelForCausalLM.from_pretrained(output, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16, device_map='cuda')
    model.eval()
    ಟೋಕನುಗಳು = tokenizer.encode("During photosynthesis in green plants", return_tensors="pt")
    ಟೋಕನುಗಳು = ಟೋಕನುಗಳು.ಗಮ್ಯ('cuda')
    output = model.generate(ಟೋಕನುಗಳು, max_new_ಟೋಕನುಗಳು=64, repetition_penalty=1.3)
    samples = tokenizer.batch_decode(output)
    ಗಾಗಿ sample ಒಳಗೆ samples:
        ಮುದ್ರಿಸು('-'*30)
        ಮುದ್ರಿಸು(sample)

# -----------------------------------------------------------------------------

ಆದರೆ __name__== '__main__':
    parser=argparse.ವಾದ_Parser()
    parser.ಕೂಡಿಸು_ವಾದ("--input", "-i", help="The name of the llm.c model.bin file", type=str, required=ನಿಜ)
    parser.ಕೂಡಿಸು_ವಾದ("--output","-o",  help="The Hugging Face output model directory", type=str, required=ನಿಜ)
    parser.ಕೂಡಿಸು_ವಾದ("--dtype", "-d", help="Output as either float32 or bfloat16 (default)", type=str, ಪೂರ್ವನಿಯೋಜಿತ="bfloat16")
    parser.ಕೂಡಿಸು_ವಾದ("--push", "-p", help="Push the model to your Hugging Face account", type=bool, ಪೂರ್ವನಿಯೋಜಿತ=ಸುಳ್ಳು)
    parser.ಕೂಡಿಸು_ವಾದ("--spin", "-s", help="Take the model for a spin at the end?", type=bool, ಪೂರ್ವನಿಯೋಜಿತ=ನಿಜ)
    ವಾದಗಳು = parser.parse_ವಾದಗಳು()
    convert(ವಾದಗಳು.ಒಳಹರಿವು, ವಾದಗಳು.output, ವಾದಗಳು.ತಳ್ಳು, ವಾದಗಳು.dtype)
    ಆದರೆ ವಾದಗಳು.spin:
        spin(ವಾದಗಳು.output)
