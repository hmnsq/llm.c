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

ਆਯਾਤ os
ਆਯਾਤ json
ਆਯਾਤ ਬੇਨਤੀਆਂ
ਆਯਾਤ tiktoken
ਤਃ tqdm ਆਯਾਤ tqdm
ਆਯਾਤ torch
ਆਯਾਤ torch.nn ਇਵ nn
ਤਃ torch.nn ਆਯਾਤ functional ਇਵ F
ਤਃ transformers ਆਯਾਤ GPT2LMHeadModel
ਤਃ ਅੰਕੜੇ_common ਆਯਾਤ download_file, write_evalfile

# -----------------------------------------------------------------------------
ਅੰਕੜੇ_CACHE_DIR = os.ਰਸਤਾ.join(os.ਰਸਤਾ.dirname(__file__), "hellaswag")

hellaswags = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

enc = tiktoken.get_encoding("gpt2")

ਪਰਿਭਾਸ਼ਾ download(ਵੰਡੋ):
    """Downloads HellaSwag DATA_CACHE_DIR"""
    os.makedirs(ਅੰਕੜੇ_CACHE_DIR, exist_ok=ਸੱਚ)
    ਅੰਕੜੇ_url = hellaswags[ਵੰਡੋ]
    ਅੰਕੜੇ_filename = os.ਰਸਤਾ.join(ਅੰਕੜੇ_CACHE_DIR, f"hellaswag_{split}.jsonl")
    ਜੇ ਨਹੀਂ os.ਰਸਤਾ.exists(ਅੰਕੜੇ_filename):
        ਛਾਪੋ(f"Downloading {data_url} to {data_filename}...")
        download_file(ਅੰਕੜੇ_url, ਅੰਕੜੇ_filename)
    ਨਹੀਂਤਾਂ:
        ਛਾਪੋ(f"{data_filename} already exists, skipping download...")

ਪਰਿਭਾਸ਼ਾ render_example(example):
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
    ਅੰਕੜੇ = {
        "label": label,
        "ctx_tokens": ਕੁਝ_ਨਹੀਂ,
        "ending_tokens": [],
    }

    # gather up all the tokens
    ctx_ਟੋਕਨ = enc.encode(ctx)
    ਅੰਕੜੇ["ctx_tokens"] = ctx_ਟੋਕਨ
    tok_ਪੰਕਤੀਆਂ = []
    ਨਕਾਬ_ਪੰਕਤੀਆਂ = []
    ਲਈ ਸਿਰਾ ਵਿੱਚ endings:
        ਸਿਰਾ_ਟੋਕਨ = enc.encode(" " + ਸਿਰਾ) # note: prepending " " because GPT-2 tokenizer
        tok_ਪੰਕਤੀਆਂ.append(ctx_ਟੋਕਨ + ਸਿਰਾ_ਟੋਕਨ)
        ਨਕਾਬ_ਪੰਕਤੀਆਂ.append([0]*ਲੰਬਾਈ(ctx_ਟੋਕਨ) + [1]*ਲੰਬਾਈ(ਸਿਰਾ_ਟੋਕਨ))
        ਅੰਕੜੇ["ending_tokens"].append(ਸਿਰਾ_ਟੋਕਨ)

    # have to be careful during the collation because the number of tokens in each row can differ
    max_len = max(ਲੰਬਾਈ(ਪੰਕਤੀ) ਲਈ ਪੰਕਤੀ ਵਿੱਚ tok_ਪੰਕਤੀਆਂ)
    ਟੋਕਨ = torch.zeros((4, max_len), dtype=torch.ਲੰਮਾ)
    ਨਕਾਬ = torch.zeros((4, max_len), dtype=torch.ਲੰਮਾ)
    ਲਈ i, (tok_ਪੰਕਤੀ, ਨਕਾਬ_ਪੰਕਤੀ) ਵਿੱਚ enumerate(zip(tok_ਪੰਕਤੀਆਂ, ਨਕਾਬ_ਪੰਕਤੀਆਂ)):
        ਟੋਕਨ[i, :ਲੰਬਾਈ(tok_ਪੰਕਤੀ)] = torch.tensor(tok_ਪੰਕਤੀ)
        ਨਕਾਬ[i, :ਲੰਬਾਈ(ਨਕਾਬ_ਪੰਕਤੀ)] = torch.tensor(ਨਕਾਬ_ਪੰਕਤੀ)

    ਵਾਪਸ ਅੰਕੜੇ, ਟੋਕਨ, ਨਕਾਬ, label

ਪਰਿਭਾਸ਼ਾ iterate_examples(ਵੰਡੋ):
    # there are 10,042 examples in total in val
    download(ਵੰਡੋ)
    ਨਾਲ open(os.ਰਸਤਾ.join(ਅੰਕੜੇ_CACHE_DIR, f"hellaswag_{split}.jsonl"), "r") ਇਵ f:
        ਲਈ ਸਤਰ ਵਿੱਚ f:
            example = json.loads(ਸਤਰ)
            ਦਿਓ example

@torch.no_grad()
ਪਰਿਭਾਸ਼ਾ evaluate(model_type, device):

    torch.ਸੈੱਟ_float32_matmul_precision('high') # use tf32

    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.ਮੰਜ਼ਿਲ(device)
    # model = torch.compile(model)

    datas = []
    ਅੰਕ_correct_norm = 0
    ਅੰਕ_correct = 0
    ਅੰਕ_ਕੁੱਲ = 0
    ਲਈ example ਵਿੱਚ iterate_examples("val"):
        ਅੰਕੜੇ, ਟੋਕਨ, ਨਕਾਬ, label = render_example(example)
        datas.append(ਅੰਕੜੇ)
        ਟੋਕਨ = ਟੋਕਨ.ਮੰਜ਼ਿਲ(device)
        ਨਕਾਬ = ਨਕਾਬ.ਮੰਜ਼ਿਲ(device)

        # get the logits
        logits = model(ਟੋਕਨ).logits
        # evaluate the autoregressive loss at all positions
        ਖਿਸਕਾਓ_logits = (logits[..., :-1, :]).contiguous()
        ਖਿਸਕਾਓ_ਟੋਕਨ = (ਟੋਕਨ[..., 1:]).contiguous()
        flat_ਖਿਸਕਾਓ_logits = ਖਿਸਕਾਓ_logits.view(-1, ਖਿਸਕਾਓ_logits.ਮਾਪ(-1))
        flat_ਖਿਸਕਾਓ_ਟੋਕਨ = ਖਿਸਕਾਓ_ਟੋਕਨ.view(-1)
        ਖਿਸਕਾਓ_losses = F.cross_entropy(flat_ਖਿਸਕਾਓ_logits, flat_ਖਿਸਕਾਓ_ਟੋਕਨ, reduction='none')
        ਖਿਸਕਾਓ_losses = ਖਿਸਕਾਓ_losses.view(ਟੋਕਨ.ਮਾਪ(0), -1)
        # now get the average loss just for the completion region (where mask == 1), in each row
        ਖਿਸਕਾਓ_ਨਕਾਬ = (ਨਕਾਬ[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
        masked_ਖਿਸਕਾਓ_losses = ਖਿਸਕਾਓ_losses * ਖਿਸਕਾਓ_ਨਕਾਬ
        # sum and divide by the number of 1s in the mask
        ਜੋੜ_loss = masked_ਖਿਸਕਾਓ_losses.ਜੋੜ(dim=1)
        avg_loss = ਜੋੜ_loss / ਖਿਸਕਾਓ_ਨਕਾਬ.ਜੋੜ(dim=1)
        # now we have a loss for each of the 4 completions
        # the one with the lowest loss should be the most likely
        pred = ਜੋੜ_loss.argmin().ਵਸਤੂ_2()
        pred_norm = avg_loss.argmin().ਵਸਤੂ_2()

        # accumulate stats
        ਅੰਕ_ਕੁੱਲ += 1
        ਅੰਕ_correct += ਪੂਰਨ(pred == label)
        ਅੰਕ_correct_norm += ਪੂਰਨ(pred_norm == label)
        ਛਾਪੋ(f"{num_total} acc: {num_correct/num_total:.4f} acc_norm: {num_correct_norm}/{num_total}={num_correct_norm/num_total:.4f}")

        # debug: pretty print a few examples, and the losses in each case
        ਜੇ ਅੰਕ_ਕੁੱਲ < 10:
            ਛਾਪੋ("---")
            ਛਾਪੋ(f"Context:\n {example['ctx']}")
            ਛਾਪੋ(f"Endings:")
            ਲਈ i, ਸਿਰਾ ਵਿੱਚ enumerate(example["endings"]):
                ਛਾਪੋ(f"{i} (loss: {avg_loss[i].item():.4f}) {end}")
            ਛਾਪੋ(f"predicted: {pred_norm}, actual: {label}")

    # now write the data to a .bin file
    filename = os.ਰਸਤਾ.join(ਅੰਕੜੇ_CACHE_DIR, f"hellaswag_val.bin")
    write_evalfile(filename, datas)

ਜੇ __name__ == "__main__":
    ਆਯਾਤ argparse
    parser = argparse.ਦਲੀਲ_Parser()
    parser.ਜੋੜੋ_ਦਲੀਲ("-m", "--model_type", type=str, ਮੂਲ="gpt2", help="the model type to use")
    parser.ਜੋੜੋ_ਦਲੀਲ("-d", "--device", type=str, ਮੂਲ="cuda", help="the device to use")
    ਦਲੀਲਾਂ = parser.parse_ਦਲੀਲਾਂ()
    evaluate(ਦਲੀਲਾਂ.model_type, ਦਲੀਲਾਂ.device)
