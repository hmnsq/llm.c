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

దిగుమతి numpy ఇవ np
దిగుమతి torch
దిగుమతి argparse, sys
తః transformers దిగుమతి GPT2Config, GPT2Tokenizer, GPT2LMHeadModel

# -----------------------------------------------------------------------------
# Tensor functions for both bfloat16 (from int16) and normal float32
# Both return float32 tensors

నిర్వచనం tensor_bf16(దత్తాంశం_int16, transpose=అబద్ధం):
    ఒకవేళ transpose:
        దత్తాంశం_int16 = దత్తాంశం_int16.transpose(1,0)
    తిరిగి torch.tensor(దత్తాంశం_int16).view(torch.bfloat16).గమ్యం(torch.float32)

నిర్వచనం tensor_fp32(దత్తాంశం_float32, transpose=అబద్ధం):
    ఒకవేళ transpose:
        దత్తాంశం_float32 = దత్తాంశం_float32.transpose(1,0)
    తిరిగి torch.tensor(దత్తాంశం_float32).view(torch.float32)

# -----------------------------------------------------------------------------
# Main conversion function

నిర్వచనం convert(filepath, output, నెట్టు_గమ్యం_hub=అబద్ధం, బహిర్గతం_dtype="bfloat16"):
    ముద్రించు(f"Converting model {filepath} to {output} in {out_dtype} format and pushing to Hugging Face: {push_to_hub}")

    f = open(filepath, 'rb')
    # Read in our header, checking the magic number and version
    # version 3 = fp32, padded vocab
    # version 5 = bf16, padded vocab
    model_శీర్షిక = np.frombuffer(f.పఠ(256*4), dtype=np.int32)
    ఒకవేళ model_శీర్షిక[0] != 20240326:
        ముద్రించు("ERROR: magic number mismatch in the data .bin file!")
        exit(1)
    సంస్కరణ = model_శీర్షిక[1]
    ఒకవేళ కాదు సంస్కరణ లో [3, 5]:
        ముద్రించు("Bad version in model file")
        exit(1)

    # Load in our model parameters
    maxT = model_శీర్షిక[2].అంశం() # max sequence length
    V = model_శీర్షిక[3].అంశం() # vocab size
    L =  model_శీర్షిక[4].అంశం() # num layers
    H = model_శీర్షిక[5].అంశం() # num heads
    C = model_శీర్షిక[6].అంశం() # channels
    Vp = model_శీర్షిక[7].అంశం() # padded vocab size

    ముద్రించు(f"{version=}, {maxT=}, {V=}, {Vp=}, {L=}, {H=}, {C=}")

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
    dtype = np.float32 ఒకవేళ సంస్కరణ == 3 లేకపోతే np.int16
    w = {}
    కొరకు కీలకం, shape లో shapes.వస్తువులు():
        అంకె_elements = np.prod(shape)
        దత్తాంశం = np.frombuffer(f.పఠ(అంకె_elements * np.dtype(dtype).itemsize), dtype=dtype)
        w[కీలకం] = దత్తాంశం.reshape(shape)
        # The binary file saves the padded vocab - drop the padding back to GPT2 size
        ఒకవేళ shape[0] == Vp:
            w[కీలకం] = w[కీలకం].reshape(shape)[:(V-Vp), :]
    # Ensure the file is fully read and then close
    ప్రతిజ్ఞా f.పఠ() == b''
    f.close()

    # Map to our model dict, the tensors at this stage are always fp32
    mk_tensor = {
        3 : tensor_fp32,
        5 : tensor_bf16,
    }[సంస్కరణ]
    model_dict = {}
    model_dict['transformer.wte.weight'] = mk_tensor(w['wte'])
    model_dict['transformer.wpe.weight'] = mk_tensor(w['wpe'])
    model_dict['lm_head.weight'] = model_dict['transformer.wte.weight'] # Tie weights
    కొరకు i లో శ్రేణి(L):
        model_dict[f'transformer.h.{i}.ln_1.weight'] = mk_tensor(w['ln1w'][i])
        model_dict[f'transformer.h.{i}.ln_1.bias'] = mk_tensor(w['ln1b'][i])
        model_dict[f'transformer.h.{i}.attn.c_attn.weight'] = mk_tensor(w['qkvw'][i], నిజం)
        model_dict[f'transformer.h.{i}.attn.c_attn.bias'] = mk_tensor(w['qkvb'][i])
        model_dict[f'transformer.h.{i}.attn.c_proj.weight'] = mk_tensor(w['attprojw'][i], నిజం)
        model_dict[f'transformer.h.{i}.attn.c_proj.bias'] = mk_tensor(w['attprojb'][i])
        model_dict[f'transformer.h.{i}.ln_2.weight'] = mk_tensor(w['ln2w'][i])
        model_dict[f'transformer.h.{i}.ln_2.bias'] = mk_tensor(w['ln2b'][i])
        model_dict[f'transformer.h.{i}.mlp.c_fc.weight'] = mk_tensor(w['fcw'][i], నిజం)
        model_dict[f'transformer.h.{i}.mlp.c_fc.bias'] = mk_tensor(w['fcb'][i])
        model_dict[f'transformer.h.{i}.mlp.c_proj.weight'] = mk_tensor(w['fcprojw'][i], నిజం)
        model_dict[f'transformer.h.{i}.mlp.c_proj.bias'] = mk_tensor(w['fcprojb'][i])
    model_dict['transformer.ln_f.weight'] = mk_tensor(w['lnfw'])
    model_dict['transformer.ln_f.bias'] = mk_tensor(w['lnfb'])

    # Create a GPT-2 model instance, in the requested dtype
    config = GPT2Config(vocab_కొలత = V,
                        n_positions = maxT,
                        n_ctx = maxT,
                        n_embd = C,
                        n_layer = L,
                        n_తల = H)
    model = GPT2LMHeadModel(config)
    ఒకవేళ బహిర్గతం_dtype == "bfloat16":
        model = model.గమ్యం(torch.bfloat16)

    # Set the model dict and save
    model.ఎక్కించు_state_dict(model_dict)
    model.save_pretrained(output, max_shard_కొలత="5GB", సురక్షిత_serialization=నిజం)

    # Copy over a standard gpt2 tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.save_pretrained(output)

    ఒకవేళ నెట్టు_గమ్యం_hub:
        ముద్రించు(f"Uploading {output} to Hugging Face")
        model.నెట్టు_గమ్యం_hub(output)
        tokenizer.నెట్టు_గమ్యం_hub(output)

నిర్వచనం spin(output):
    ముద్రించు("Taking the exported model for a spin...")
    ముద్రించు('-'*80)
    తః transformers దిగుమతి AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(output)
    model = AutoModelForCausalLM.from_pretrained(output, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16, device_map='cuda')
    model.eval()
    టోకెన్లు = tokenizer.encode("During photosynthesis in green plants", return_tensors="pt")
    టోకెన్లు = టోకెన్లు.గమ్యం('cuda')
    output = model.generate(టోకెన్లు, max_new_టోకెన్లు=64, repetition_penalty=1.3)
    samples = tokenizer.batch_decode(output)
    కొరకు sample లో samples:
        ముద్రించు('-'*30)
        ముద్రించు(sample)

# -----------------------------------------------------------------------------

ఒకవేళ __name__== '__main__':
    parser=argparse.వాదం_Parser()
    parser.కూడు_వాదం("--input", "-i", help="The name of the llm.c model.bin file", type=str, required=నిజం)
    parser.కూడు_వాదం("--output","-o",  help="The Hugging Face output model directory", type=str, required=నిజం)
    parser.కూడు_వాదం("--dtype", "-d", help="Output as either float32 or bfloat16 (default)", type=str, అప్రమేయ="bfloat16")
    parser.కూడు_వాదం("--push", "-p", help="Push the model to your Hugging Face account", type=bool, అప్రమేయ=అబద్ధం)
    parser.కూడు_వాదం("--spin", "-s", help="Take the model for a spin at the end?", type=bool, అప్రమేయ=నిజం)
    వాదనలు = parser.parse_వాదనలు()
    convert(వాదనలు.ఆదానం, వాదనలు.output, వాదనలు.నెట్టు, వాదనలు.dtype)
    ఒకవేళ వాదనలు.spin:
        spin(వాదనలు.output)
