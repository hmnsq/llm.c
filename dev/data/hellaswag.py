"""
Downloads and evaluates HellaSwag in Python.
This then acts as the reference file for llm.c
Also writes the data (tokens, labels) to .bin files for parallel evaluation in C.
https://github.com/rowanz/hellaswag

Example HellaSwag json item:

{"ind": 24, "activity_label": "Roof shingle removal", "ctx_a": "A man is sitting on a roof.", "ctx_b": "he", "ctx": "A man is sitting on a roof. he", "split": "val", "split_type": "indomain", "label": 3, "endings": ["is using wrap to wrap a pair of skis.", "is ripping level tiles off.", "is holding a rubik's cube.", "starts pulling up roofing on a roof."], "source_id": "activitynet~v_-JhWjGDPHMY"}

ind: dataset ID
activity_label: The ActivityNet or WikiHow label for this example
context: There are two formats. The full context is in ctx. When the context ends in an (incomplete) noun phrase, like for ActivityNet, this incomplete noun phrase is in ctx_b, and the context up until then is in ctx_a. This can be useful for models such as BERT that need the last sentence to be complete. However, it's never required. If ctx_b is nonempty, then ctx is the same thing as ctx_a, followed by a space, then ctx_b.
endings: a list of 4 endings. The correct index is given by label (0,1,2, or 3)
split: train, val, or test.
split_type: indomain if the activity label is seen during training, else zeroshot
source_id: Which video or WikiHow article this example came from

gpt2 (124M)
- eleuther harness reports acc 28.92%, acc_norm 31.14% (multiple choice style)
- this script: 10042 acc: 0.2859 acc_norm: 0.2955 (completion style)

gpt2-xl (1558M)
- eleuther harness reports acc 40.04%, acc_norm 50.89% (multiple choice style)
- this script: 10042 acc: 0.3842 acc_norm: 0.4893 (completion style)

The validation set of HellaSwag has a total of 10,042 examples.
"""

استورد os
استورد json
استورد غوښتنې
استورد tiktoken
من tqdm استورد tqdm
استورد torch
استورد torch.nn كـ nn
من torch.nn استورد functional كـ F
من transformers استورد GPT2LMHeadModel
من معلومات_common استورد download_file, write_evalfile

# -----------------------------------------------------------------------------
معلومات_CACHE_DIR = os.لار.join(os.لار.dirname(__file__), "hellaswag")

hellaswags = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

enc = tiktoken.get_encoding("gpt2")

تعریف download(وېشل):
    """Downloads HellaSwag DATA_CACHE_DIR"""
    os.makedirs(معلومات_CACHE_DIR, exist_ok=ریښتیا)
    معلومات_url = hellaswags[وېشل]
    معلومات_filename = os.لار.join(معلومات_CACHE_DIR, f"hellaswag_{split}.jsonl")
    إذا ليس os.لار.exists(معلومات_filename):
        چاپ(f"Downloading {data_url} to {data_filename}...")
        download_file(معلومات_url, معلومات_filename)
    وإلا:
        چاپ(f"{data_filename} already exists, skipping download...")

تعریف render_example(example):
    """
    Given the example as a dictionary, render it as three torch tensors:
    - tokens (the tokens of context + completion, of size 4xN, as there are always 4 candidates)
    - mask (is 1 in the region of the candidate completion, where we evaluate likelihoods)
    - label (the index of the correct completion, which we hope has the highest likelihood)
    """
    ctx = example["ctx"]
    label = example["label"]
    endings = example["endings"]

    # data needed to reproduce this eval on the C size
    معلومات = {
        "label": label,
        "ctx_tokens": هیڅ,
        "ending_tokens": [],
    }

    # gather up all the tokens
    ctx_نښې = enc.encode(ctx)
    معلومات["ctx_tokens"] = ctx_نښې
    tok_لیکې = []
    ماسک_لیکې = []
    لكل پای_ټکی في endings:
        پای_ټکی_نښې = enc.encode(" " + پای_ټکی) # note: prepending " " because GPT-2 tokenizer
        tok_لیکې.append(ctx_نښې + پای_ټکی_نښې)
        ماسک_لیکې.append([0]*اوږدوالی(ctx_نښې) + [1]*اوږدوالی(پای_ټکی_نښې))
        معلومات["ending_tokens"].append(پای_ټکی_نښې)

    # have to be careful during the collation because the number of tokens in each row can differ
    max_len = max(اوږدوالی(لیکه) لكل لیکه في tok_لیکې)
    نښې = torch.zeros((4, max_len), dtype=torch.اوږد)
    ماسک = torch.zeros((4, max_len), dtype=torch.اوږد)
    لكل i, (tok_لیکه, ماسک_لیکه) في enumerate(zip(tok_لیکې, ماسک_لیکې)):
        نښې[i, :اوږدوالی(tok_لیکه)] = torch.tensor(tok_لیکه)
        ماسک[i, :اوږدوالی(ماسک_لیکه)] = torch.tensor(ماسک_لیکه)

    أرجع معلومات, نښې, ماسک, label

تعریف iterate_examples(وېشل):
    # there are 10,042 examples in total in val
    download(وېشل)
    سره open(os.لار.join(معلومات_CACHE_DIR, f"hellaswag_{split}.jsonl"), "r") كـ f:
        لكل کرښه في f:
            example = json.loads(کرښه)
            أنتج example

@torch.no_grad()
تعریف evaluate(model_type, device):

    torch.سېټ_float32_matmul_precision('high') # use tf32

    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.منزل(device)
    # model = torch.compile(model)

    datas = []
    عدد_correct_norm = 0
    عدد_correct = 0
    عدد_ټول = 0
    لكل example في iterate_examples("val"):
        معلومات, نښې, ماسک, label = render_example(example)
        datas.append(معلومات)
        نښې = نښې.منزل(device)
        ماسک = ماسک.منزل(device)

        # get the logits
        logits = model(نښې).logits
        # evaluate the autoregressive loss at all positions
        ښويدنه_logits = (logits[..., :-1, :]).contiguous()
        ښويدنه_نښې = (نښې[..., 1:]).contiguous()
        flat_ښويدنه_logits = ښويدنه_logits.view(-1, ښويدنه_logits.اندازه(-1))
        flat_ښويدنه_نښې = ښويدنه_نښې.view(-1)
        ښويدنه_losses = F.cross_entropy(flat_ښويدنه_logits, flat_ښويدنه_نښې, reduction='none')
        ښويدنه_losses = ښويدنه_losses.view(نښې.اندازه(0), -1)
        # now get the average loss just for the completion region (where mask == 1), in each row
        ښويدنه_ماسک = (ماسک[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
        masked_ښويدنه_losses = ښويدنه_losses * ښويدنه_ماسک
        # sum and divide by the number of 1s in the mask
        مجموعه_loss = masked_ښويدنه_losses.مجموعه(dim=1)
        avg_loss = مجموعه_loss / ښويدنه_ماسک.مجموعه(dim=1)
        # now we have a loss for each of the 4 completions
        # the one with the lowest loss should be the most likely
        pred = مجموعه_loss.argmin().توکی()
        pred_norm = avg_loss.argmin().توکی()

        # accumulate stats
        عدد_ټول += 1
        عدد_correct += صحيح(pred == label)
        عدد_correct_norm += صحيح(pred_norm == label)
        چاپ(f"{num_total} acc: {num_correct/num_total:.4f} acc_norm: {num_correct_norm}/{num_total}={num_correct_norm/num_total:.4f}")

        # debug: pretty print a few examples, and the losses in each case
        إذا عدد_ټول < 10:
            چاپ("---")
            چاپ(f"Context:\n {example['ctx']}")
            چاپ(f"Endings:")
            لكل i, پای_ټکی في enumerate(example["endings"]):
                چاپ(f"{i} (loss: {avg_loss[i].item():.4f}) {end}")
            چاپ(f"predicted: {pred_norm}, actual: {label}")

    # now write the data to a .bin file
    filename = os.لار.join(معلومات_CACHE_DIR, f"hellaswag_val.bin")
    write_evalfile(filename, datas)

إذا __name__ == "__main__":
    استورد argparse
    parser = argparse.دلیل_Parser()
    parser.جمع_کړه_دلیل("-m", "--model_type", type=str, افتراضي="gpt2", help="the model type to use")
    parser.جمع_کړه_دلیل("-d", "--device", type=str, افتراضي="cuda", help="the device to use")
    دلیلونه = parser.parse_دلیلونه()
    evaluate(دلیلونه.model_type, دلیلونه.device)
