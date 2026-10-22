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

درآمد os
درآمد json
درآمد درخواستیں
درآمد tiktoken
سے tqdm درآمد tqdm
درآمد torch
درآمد torch.nn بطور nn
سے torch.nn درآمد functional بطور F
سے transformers درآمد GPT2LMHeadModel
سے ڈیٹا_common درآمد download_file, write_evalfile

# -----------------------------------------------------------------------------
ڈیٹا_CACHE_DIR = os.راستہ.join(os.راستہ.dirname(__file__), "hellaswag")

hellaswags = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

enc = tiktoken.get_encoding("gpt2")

تعریف download(بانٹو):
    """Downloads HellaSwag DATA_CACHE_DIR"""
    os.makedirs(ڈیٹا_CACHE_DIR, exist_ok=سچ)
    ڈیٹا_url = hellaswags[بانٹو]
    ڈیٹا_filename = os.راستہ.join(ڈیٹا_CACHE_DIR, f"hellaswag_{split}.jsonl")
    اگر نہیں os.راستہ.exists(ڈیٹا_filename):
        چھاپو(f"Downloading {data_url} to {data_filename}...")
        download_file(ڈیٹا_url, ڈیٹا_filename)
    ورنہ:
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
    ڈیٹا = {
        "label": label,
        "ctx_tokens": کچھ_نہیں,
        "ending_tokens": [],
    }

    # gather up all the tokens
    ctx_علامات = enc.encode(ctx)
    ڈیٹا["ctx_tokens"] = ctx_علامات
    tok_صفیں = []
    نقاب_صفیں = []
    برائے انتہا میں endings:
        انتہا_علامات = enc.encode(" " + انتہا) # note: prepending " " because GPT-2 tokenizer
        tok_صفیں.append(ctx_علامات + انتہا_علامات)
        نقاب_صفیں.append([0]*لمبائی(ctx_علامات) + [1]*لمبائی(انتہا_علامات))
        ڈیٹا["ending_tokens"].append(انتہا_علامات)

    # have to be careful during the collation because the number of tokens in each row can differ
    max_len = max(لمبائی(صف) برائے صف میں tok_صفیں)
    علامات = torch.zeros((4, max_len), dtype=torch.طویل)
    نقاب = torch.zeros((4, max_len), dtype=torch.طویل)
    برائے i, (tok_صف, نقاب_صف) میں enumerate(zip(tok_صفیں, نقاب_صفیں)):
        علامات[i, :لمبائی(tok_صف)] = torch.tensor(tok_صف)
        نقاب[i, :لمبائی(نقاب_صف)] = torch.tensor(نقاب_صف)

    واپس ڈیٹا, علامات, نقاب, label

تعریف iterate_examples(بانٹو):
    # there are 10,042 examples in total in val
    download(بانٹو)
    ساتھ open(os.راستہ.join(ڈیٹا_CACHE_DIR, f"hellaswag_{split}.jsonl"), "r") بطور f:
        برائے سطر میں f:
            example = json.loads(سطر)
            دو example

@torch.no_grad()
تعریف evaluate(model_type, device):

    torch.سیٹ_float32_matmul_precision('high') # use tf32

    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.مقصد(device)
    # model = torch.compile(model)

    datas = []
    عدد_correct_norm = 0
    عدد_correct = 0
    عدد_کل = 0
    برائے example میں iterate_examples("val"):
        ڈیٹا, علامات, نقاب, label = render_example(example)
        datas.append(ڈیٹا)
        علامات = علامات.مقصد(device)
        نقاب = نقاب.مقصد(device)

        # get the logits
        logits = model(علامات).logits
        # evaluate the autoregressive loss at all positions
        سرکاؤ_logits = (logits[..., :-1, :]).contiguous()
        سرکاؤ_علامات = (علامات[..., 1:]).contiguous()
        flat_سرکاؤ_logits = سرکاؤ_logits.view(-1, سرکاؤ_logits.سائز(-1))
        flat_سرکاؤ_علامات = سرکاؤ_علامات.view(-1)
        سرکاؤ_losses = F.cross_entropy(flat_سرکاؤ_logits, flat_سرکاؤ_علامات, reduction='none')
        سرکاؤ_losses = سرکاؤ_losses.view(علامات.سائز(0), -1)
        # now get the average loss just for the completion region (where mask == 1), in each row
        سرکاؤ_نقاب = (نقاب[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
        masked_سرکاؤ_losses = سرکاؤ_losses * سرکاؤ_نقاب
        # sum and divide by the number of 1s in the mask
        مجموع_loss = masked_سرکاؤ_losses.مجموع(dim=1)
        avg_loss = مجموع_loss / سرکاؤ_نقاب.مجموع(dim=1)
        # now we have a loss for each of the 4 completions
        # the one with the lowest loss should be the most likely
        pred = مجموع_loss.argmin().شے_2()
        pred_norm = avg_loss.argmin().شے_2()

        # accumulate stats
        عدد_کل += 1
        عدد_correct += صحیح(pred == label)
        عدد_correct_norm += صحیح(pred_norm == label)
        چھاپو(f"{num_total} acc: {num_correct/num_total:.4f} acc_norm: {num_correct_norm}/{num_total}={num_correct_norm/num_total:.4f}")

        # debug: pretty print a few examples, and the losses in each case
        اگر عدد_کل < 10:
            چھاپو("---")
            چھاپو(f"Context:\n {example['ctx']}")
            چھاپو(f"Endings:")
            برائے i, انتہا میں enumerate(example["endings"]):
                چھاپو(f"{i} (loss: {avg_loss[i].item():.4f}) {end}")
            چھاپو(f"predicted: {pred_norm}, actual: {label}")

    # now write the data to a .bin file
    filename = os.راستہ.join(ڈیٹا_CACHE_DIR, f"hellaswag_val.bin")
    write_evalfile(filename, datas)

اگر __name__ == "__main__":
    درآمد argparse
    parser = argparse.دلیل_Parser()
    parser.جوڑو_دلیل("-m", "--model_type", type=str, طےشدہ="gpt2", help="the model type to use")
    parser.جوڑو_دلیل("-d", "--device", type=str, طےشدہ="cuda", help="the device to use")
    دلائل = parser.parse_دلائل()
    evaluate(دلائل.model_type, دلائل.device)
