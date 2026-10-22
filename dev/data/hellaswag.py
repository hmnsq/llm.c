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
استورد درخواستها
استورد tiktoken
من tqdm استورد tqdm
استورد torch
استورد torch.nn كـ nn
من torch.nn استورد functional كـ F
من transformers استورد GPT2LMHeadModel
من داده_common استورد download_file, write_evalfile

# -----------------------------------------------------------------------------
داده_CACHE_DIR = os.مسیر.join(os.مسیر.dirname(__file__), "hellaswag")

hellaswags = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

enc = tiktoken.get_encoding("gpt2")

تعریف download(تقسیم):
    """Downloads HellaSwag DATA_CACHE_DIR"""
    os.makedirs(داده_CACHE_DIR, exist_ok=درست)
    داده_url = hellaswags[تقسیم]
    داده_filename = os.مسیر.join(داده_CACHE_DIR, f"hellaswag_{split}.jsonl")
    إذا ليس os.مسیر.exists(داده_filename):
        چاپ(f"Downloading {data_url} to {data_filename}...")
        download_file(داده_url, داده_filename)
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
    داده = {
        "label": label,
        "ctx_tokens": هیچ,
        "ending_tokens": [],
    }

    # gather up all the tokens
    ctx_نشانهها = enc.encode(ctx)
    داده["ctx_tokens"] = ctx_نشانهها
    tok_ردیفها = []
    ماسک_ردیفها = []
    لكل انتها في endings:
        انتها_نشانهها = enc.encode(" " + انتها) # note: prepending " " because GPT-2 tokenizer
        tok_ردیفها.append(ctx_نشانهها + انتها_نشانهها)
        ماسک_ردیفها.append([0]*طول(ctx_نشانهها) + [1]*طول(انتها_نشانهها))
        داده["ending_tokens"].append(انتها_نشانهها)

    # have to be careful during the collation because the number of tokens in each row can differ
    max_len = max(طول(ردیف) لكل ردیف في tok_ردیفها)
    نشانهها = torch.zeros((4, max_len), dtype=torch.دراز)
    ماسک = torch.zeros((4, max_len), dtype=torch.دراز)
    لكل i, (tok_ردیف, ماسک_ردیف) في enumerate(zip(tok_ردیفها, ماسک_ردیفها)):
        نشانهها[i, :طول(tok_ردیف)] = torch.tensor(tok_ردیف)
        ماسک[i, :طول(ماسک_ردیف)] = torch.tensor(ماسک_ردیف)

    أرجع داده, نشانهها, ماسک, label

تعریف iterate_examples(تقسیم):
    # there are 10,042 examples in total in val
    download(تقسیم)
    با open(os.مسیر.join(داده_CACHE_DIR, f"hellaswag_{split}.jsonl"), "r") كـ f:
        لكل سطر في f:
            example = json.loads(سطر)
            أنتج example

@torch.no_grad()
تعریف evaluate(model_type, device):

    torch.مجموعه_float32_matmul_precision('high') # use tf32

    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.مقصد(device)
    # model = torch.compile(model)

    datas = []
    رقم_correct_norm = 0
    رقم_correct = 0
    رقم_مجموع = 0
    لكل example في iterate_examples("val"):
        داده, نشانهها, ماسک, label = render_example(example)
        datas.append(داده)
        نشانهها = نشانهها.مقصد(device)
        ماسک = ماسک.مقصد(device)

        # get the logits
        logits = model(نشانهها).logits
        # evaluate the autoregressive loss at all positions
        لغزش_logits = (logits[..., :-1, :]).contiguous()
        لغزش_نشانهها = (نشانهها[..., 1:]).contiguous()
        flat_لغزش_logits = لغزش_logits.view(-1, لغزش_logits.اندازه(-1))
        flat_لغزش_نشانهها = لغزش_نشانهها.view(-1)
        لغزش_losses = F.cross_entropy(flat_لغزش_logits, flat_لغزش_نشانهها, reduction='none')
        لغزش_losses = لغزش_losses.view(نشانهها.اندازه(0), -1)
        # now get the average loss just for the completion region (where mask == 1), in each row
        لغزش_ماسک = (ماسک[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
        masked_لغزش_losses = لغزش_losses * لغزش_ماسک
        # sum and divide by the number of 1s in the mask
        جمع_loss = masked_لغزش_losses.جمع(dim=1)
        avg_loss = جمع_loss / لغزش_ماسک.جمع(dim=1)
        # now we have a loss for each of the 4 completions
        # the one with the lowest loss should be the most likely
        pred = جمع_loss.argmin().قلم()
        pred_norm = avg_loss.argmin().قلم()

        # accumulate stats
        رقم_مجموع += 1
        رقم_correct += صحيح(pred == label)
        رقم_correct_norm += صحيح(pred_norm == label)
        چاپ(f"{num_total} acc: {num_correct/num_total:.4f} acc_norm: {num_correct_norm}/{num_total}={num_correct_norm/num_total:.4f}")

        # debug: pretty print a few examples, and the losses in each case
        إذا رقم_مجموع < 10:
            چاپ("---")
            چاپ(f"Context:\n {example['ctx']}")
            چاپ(f"Endings:")
            لكل i, انتها في enumerate(example["endings"]):
                چاپ(f"{i} (loss: {avg_loss[i].item():.4f}) {end}")
            چاپ(f"predicted: {pred_norm}, actual: {label}")

    # now write the data to a .bin file
    filename = os.مسیر.join(داده_CACHE_DIR, f"hellaswag_val.bin")
    write_evalfile(filename, datas)

إذا __name__ == "__main__":
    استورد argparse
    parser = argparse.آرگومان_Parser()
    parser.جمع_کن_آرگومان("-m", "--model_type", type=str, افتراضي="gpt2", help="the model type to use")
    parser.جمع_کن_آرگومان("-d", "--device", type=str, افتراضي="cuda", help="the device to use")
    آرگومانها = parser.parse_آرگومانها()
    evaluate(آرگومانها.model_type, آرگومانها.device)
