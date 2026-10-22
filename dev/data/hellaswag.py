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
استورد بینتیاں
استورد tiktoken
من tqdm استورد tqdm
استورد torch
استورد torch.nn كـ nn
من torch.nn استورد functional كـ F
من transformers استورد GPT2LMHeadModel
من انکڑے_common استورد download_file, write_evalfile

# -----------------------------------------------------------------------------
انکڑے_CACHE_DIR = os.رستہ.join(os.رستہ.dirname(__file__), "hellaswag")

hellaswags = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

enc = tiktoken.get_encoding("gpt2")

تعریف download(ونڈو):
    """Downloads HellaSwag DATA_CACHE_DIR"""
    os.makedirs(انکڑے_CACHE_DIR, exist_ok=سچ)
    انکڑے_url = hellaswags[ونڈو]
    انکڑے_filename = os.رستہ.join(انکڑے_CACHE_DIR, f"hellaswag_{split}.jsonl")
    إذا ليس os.رستہ.exists(انکڑے_filename):
        چھاپو(f"Downloading {data_url} to {data_filename}...")
        download_file(انکڑے_url, انکڑے_filename)
    وإلا:
        چھاپو(f"{data_filename} already exists, skipping download...")

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
    انکڑے = {
        "label": label,
        "ctx_tokens": کجھ_نئیں,
        "ending_tokens": [],
    }

    # gather up all the tokens
    ctx_ٹوکن = enc.encode(ctx)
    انکڑے["ctx_tokens"] = ctx_ٹوکن
    tok_صفاں = []
    نقاب_صفاں = []
    لكل سرا في endings:
        سرا_ٹوکن = enc.encode(" " + سرا) # note: prepending " " because GPT-2 tokenizer
        tok_صفاں.append(ctx_ٹوکن + سرا_ٹوکن)
        نقاب_صفاں.append([0]*لمبائی(ctx_ٹوکن) + [1]*لمبائی(سرا_ٹوکن))
        انکڑے["ending_tokens"].append(سرا_ٹوکن)

    # have to be careful during the collation because the number of tokens in each row can differ
    max_len = max(لمبائی(صف) لكل صف في tok_صفاں)
    ٹوکن = torch.zeros((4, max_len), dtype=torch.لما)
    نقاب = torch.zeros((4, max_len), dtype=torch.لما)
    لكل i, (tok_صف, نقاب_صف) في enumerate(zip(tok_صفاں, نقاب_صفاں)):
        ٹوکن[i, :لمبائی(tok_صف)] = torch.tensor(tok_صف)
        نقاب[i, :لمبائی(نقاب_صف)] = torch.tensor(نقاب_صف)

    أرجع انکڑے, ٹوکن, نقاب, label

تعریف iterate_examples(ونڈو):
    # there are 10,042 examples in total in val
    download(ونڈو)
    نال open(os.رستہ.join(انکڑے_CACHE_DIR, f"hellaswag_{split}.jsonl"), "r") كـ f:
        لكل سطر في f:
            example = json.loads(سطر)
            أنتج example

@torch.no_grad()
تعریف evaluate(model_type, device):

    torch.سیٹ_float32_matmul_precision('high') # use tf32

    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.منزل(device)
    # model = torch.compile(model)

    datas = []
    عدد_correct_norm = 0
    عدد_correct = 0
    عدد_کل = 0
    لكل example في iterate_examples("val"):
        انکڑے, ٹوکن, نقاب, label = render_example(example)
        datas.append(انکڑے)
        ٹوکن = ٹوکن.منزل(device)
        نقاب = نقاب.منزل(device)

        # get the logits
        logits = model(ٹوکن).logits
        # evaluate the autoregressive loss at all positions
        کھسکاؤ_logits = (logits[..., :-1, :]).contiguous()
        کھسکاؤ_ٹوکن = (ٹوکن[..., 1:]).contiguous()
        flat_کھسکاؤ_logits = کھسکاؤ_logits.view(-1, کھسکاؤ_logits.ناپ(-1))
        flat_کھسکاؤ_ٹوکن = کھسکاؤ_ٹوکن.view(-1)
        کھسکاؤ_losses = F.cross_entropy(flat_کھسکاؤ_logits, flat_کھسکاؤ_ٹوکن, reduction='none')
        کھسکاؤ_losses = کھسکاؤ_losses.view(ٹوکن.ناپ(0), -1)
        # now get the average loss just for the completion region (where mask == 1), in each row
        کھسکاؤ_نقاب = (نقاب[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
        masked_کھسکاؤ_losses = کھسکاؤ_losses * کھسکاؤ_نقاب
        # sum and divide by the number of 1s in the mask
        جوڑ_loss = masked_کھسکاؤ_losses.جوڑ(dim=1)
        avg_loss = جوڑ_loss / کھسکاؤ_نقاب.جوڑ(dim=1)
        # now we have a loss for each of the 4 completions
        # the one with the lowest loss should be the most likely
        pred = جوڑ_loss.argmin().شے()
        pred_norm = avg_loss.argmin().شے()

        # accumulate stats
        عدد_کل += 1
        عدد_correct += صحيح(pred == label)
        عدد_correct_norm += صحيح(pred_norm == label)
        چھاپو(f"{num_total} acc: {num_correct/num_total:.4f} acc_norm: {num_correct_norm}/{num_total}={num_correct_norm/num_total:.4f}")

        # debug: pretty print a few examples, and the losses in each case
        إذا عدد_کل < 10:
            چھاپو("---")
            چھاپو(f"Context:\n {example['ctx']}")
            چھاپو(f"Endings:")
            لكل i, سرا في enumerate(example["endings"]):
                چھاپو(f"{i} (loss: {avg_loss[i].item():.4f}) {end}")
            چھاپو(f"predicted: {pred_norm}, actual: {label}")

    # now write the data to a .bin file
    filename = os.رستہ.join(انکڑے_CACHE_DIR, f"hellaswag_val.bin")
    write_evalfile(filename, datas)

إذا __name__ == "__main__":
    استورد argparse
    parser = argparse.دلیل_Parser()
    parser.جوڑو_دلیل("-m", "--model_type", type=str, افتراضي="gpt2", help="the model type to use")
    parser.جوڑو_دلیل("-d", "--device", type=str, افتراضي="cuda", help="the device to use")
    دلیلاں = parser.parse_دلیلاں()
    evaluate(دلیلاں.model_type, دلیلاں.device)
