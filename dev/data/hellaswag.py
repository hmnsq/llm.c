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
استورد درخواستون
استورد tiktoken
من tqdm استورد tqdm
استورد torch
استورد torch.nn كـ nn
من torch.nn استورد functional كـ F
من transformers استورد GPT2LMHeadModel
من معلومات_common استورد download_file, write_evalfile

# -----------------------------------------------------------------------------
معلومات_CACHE_DIR = os.رستو.join(os.رستو.dirname(__file__), "hellaswag")

hellaswags = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

enc = tiktoken.get_encoding("gpt2")

وصف download(ورهايو):
    """Downloads HellaSwag DATA_CACHE_DIR"""
    os.makedirs(معلومات_CACHE_DIR, exist_ok=سچ)
    معلومات_url = hellaswags[ورهايو]
    معلومات_filename = os.رستو.join(معلومات_CACHE_DIR, f"hellaswag_{split}.jsonl")
    إذا ليس os.رستو.exists(معلومات_filename):
        ڇاپيو(f"Downloading {data_url} to {data_filename}...")
        download_file(معلومات_url, معلومات_filename)
    وإلا:
        ڇاپيو(f"{data_filename} already exists, skipping download...")

وصف render_example(example):
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
        "ctx_tokens": ڪجهه_نه,
        "ending_tokens": [],
    }

    # gather up all the tokens
    ctx_ٽوڪن = enc.encode(ctx)
    معلومات["ctx_tokens"] = ctx_ٽوڪن
    tok_صفون = []
    نقاب_صفون = []
    لكل پڇاڙي في endings:
        پڇاڙي_ٽوڪن = enc.encode(" " + پڇاڙي) # note: prepending " " because GPT-2 tokenizer
        tok_صفون.append(ctx_ٽوڪن + پڇاڙي_ٽوڪن)
        نقاب_صفون.append([0]*ڊيگهه(ctx_ٽوڪن) + [1]*ڊيگهه(پڇاڙي_ٽوڪن))
        معلومات["ending_tokens"].append(پڇاڙي_ٽوڪن)

    # have to be careful during the collation because the number of tokens in each row can differ
    max_len = max(ڊيگهه(صف) لكل صف في tok_صفون)
    ٽوڪن = torch.zeros((4, max_len), dtype=torch.ڊگهو)
    نقاب = torch.zeros((4, max_len), dtype=torch.ڊگهو)
    لكل i, (tok_صف, نقاب_صف) في enumerate(zip(tok_صفون, نقاب_صفون)):
        ٽوڪن[i, :ڊيگهه(tok_صف)] = torch.tensor(tok_صف)
        نقاب[i, :ڊيگهه(نقاب_صف)] = torch.tensor(نقاب_صف)

    أرجع معلومات, ٽوڪن, نقاب, label

وصف iterate_examples(ورهايو):
    # there are 10,042 examples in total in val
    download(ورهايو)
    سان open(os.رستو.join(معلومات_CACHE_DIR, f"hellaswag_{split}.jsonl"), "r") كـ f:
        لكل سٽ في f:
            example = json.loads(سٽ)
            أنتج example

@torch.no_grad()
وصف evaluate(model_type, device):

    torch.سيٽ_float32_matmul_precision('high') # use tf32

    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.منزل(device)
    # model = torch.compile(model)

    datas = []
    انگ_correct_norm = 0
    انگ_correct = 0
    انگ_ڪل = 0
    لكل example في iterate_examples("val"):
        معلومات, ٽوڪن, نقاب, label = render_example(example)
        datas.append(معلومات)
        ٽوڪن = ٽوڪن.منزل(device)
        نقاب = نقاب.منزل(device)

        # get the logits
        logits = model(ٽوڪن).logits
        # evaluate the autoregressive loss at all positions
        سرڪايو_logits = (logits[..., :-1, :]).contiguous()
        سرڪايو_ٽوڪن = (ٽوڪن[..., 1:]).contiguous()
        flat_سرڪايو_logits = سرڪايو_logits.view(-1, سرڪايو_logits.سائيز(-1))
        flat_سرڪايو_ٽوڪن = سرڪايو_ٽوڪن.view(-1)
        سرڪايو_losses = F.cross_entropy(flat_سرڪايو_logits, flat_سرڪايو_ٽوڪن, reduction='none')
        سرڪايو_losses = سرڪايو_losses.view(ٽوڪن.سائيز(0), -1)
        # now get the average loss just for the completion region (where mask == 1), in each row
        سرڪايو_نقاب = (نقاب[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
        masked_سرڪايو_losses = سرڪايو_losses * سرڪايو_نقاب
        # sum and divide by the number of 1s in the mask
        جوڙ_loss = masked_سرڪايو_losses.جوڙ(dim=1)
        avg_loss = جوڙ_loss / سرڪايو_نقاب.جوڙ(dim=1)
        # now we have a loss for each of the 4 completions
        # the one with the lowest loss should be the most likely
        pred = جوڙ_loss.argmin().شي()
        pred_norm = avg_loss.argmin().شي()

        # accumulate stats
        انگ_ڪل += 1
        انگ_correct += صحيح(pred == label)
        انگ_correct_norm += صحيح(pred_norm == label)
        ڇاپيو(f"{num_total} acc: {num_correct/num_total:.4f} acc_norm: {num_correct_norm}/{num_total}={num_correct_norm/num_total:.4f}")

        # debug: pretty print a few examples, and the losses in each case
        إذا انگ_ڪل < 10:
            ڇاپيو("---")
            ڇاپيو(f"Context:\n {example['ctx']}")
            ڇاپيو(f"Endings:")
            لكل i, پڇاڙي في enumerate(example["endings"]):
                ڇاپيو(f"{i} (loss: {avg_loss[i].item():.4f}) {end}")
            ڇاپيو(f"predicted: {pred_norm}, actual: {label}")

    # now write the data to a .bin file
    filename = os.رستو.join(معلومات_CACHE_DIR, f"hellaswag_val.bin")
    write_evalfile(filename, datas)

إذا __name__ == "__main__":
    استورد argparse
    parser = argparse.دليل_Parser()
    parser.جوڙيو_دليل("-m", "--model_type", type=str, افتراضي="gpt2", help="the model type to use")
    parser.جوڙيو_دليل("-d", "--device", type=str, افتراضي="cuda", help="the device to use")
    دليلون = parser.parse_دليلون()
    evaluate(دليلون.model_type, دليلون.device)
