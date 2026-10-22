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

ಆಮದು os
ಆಮದು json
ಆಮದು ವಿನಂತಿಗಳು
ಆಮದು tiktoken
ತಃ tqdm ಆಮದು tqdm
ಆಮದು torch
ಆಮದು torch.nn ಇವ nn
ತಃ torch.nn ಆಮದು functional ಇವ F
ತಃ transformers ಆಮದು GPT2LMHeadModel
ತಃ ದತ್ತಾಂಶ_common ಆಮದು download_file, write_evalfile

# -----------------------------------------------------------------------------
ದತ್ತಾಂಶ_CACHE_DIR = os.ಮಾರ್ಗ.join(os.ಮಾರ್ಗ.dirname(__file__), "hellaswag")

hellaswags = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

enc = tiktoken.get_encoding("gpt2")

ವ್ಯಾಖ್ಯಾನ download(ವಿಭಜಿಸು):
    """Downloads HellaSwag DATA_CACHE_DIR"""
    os.makedirs(ದತ್ತಾಂಶ_CACHE_DIR, exist_ok=ನಿಜ)
    ದತ್ತಾಂಶ_url = hellaswags[ವಿಭಜಿಸು]
    ದತ್ತಾಂಶ_filename = os.ಮಾರ್ಗ.join(ದತ್ತಾಂಶ_CACHE_DIR, f"hellaswag_{split}.jsonl")
    ಆದರೆ ಅಲ್ಲ os.ಮಾರ್ಗ.exists(ದತ್ತಾಂಶ_filename):
        ಮುದ್ರಿಸು(f"Downloading {data_url} to {data_filename}...")
        download_file(ದತ್ತಾಂಶ_url, ದತ್ತಾಂಶ_filename)
    ಇಲ್ಲದಿದ್ದರೆ:
        ಮುದ್ರಿಸು(f"{data_filename} already exists, skipping download...")

ವ್ಯಾಖ್ಯಾನ render_example(example):
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
    ದತ್ತಾಂಶ = {
        "label": label,
        "ctx_tokens": ಏನೂಇಲ್ಲ,
        "ending_tokens": [],
    }

    # gather up all the tokens
    ctx_ಟೋಕನುಗಳು = enc.encode(ctx)
    ದತ್ತಾಂಶ["ctx_tokens"] = ctx_ಟೋಕನುಗಳು
    tok_ಅಡ್ಡಸಾಲುಗಳು = []
    ಮುಖವಾಡ_ಅಡ್ಡಸಾಲುಗಳು = []
    ಗಾಗಿ ಕೊನೆ ಒಳಗೆ endings:
        ಕೊನೆ_ಟೋಕನುಗಳು = enc.encode(" " + ಕೊನೆ) # note: prepending " " because GPT-2 tokenizer
        tok_ಅಡ್ಡಸಾಲುಗಳು.append(ctx_ಟೋಕನುಗಳು + ಕೊನೆ_ಟೋಕನುಗಳು)
        ಮುಖವಾಡ_ಅಡ್ಡಸಾಲುಗಳು.append([0]*ಉದ್ದ(ctx_ಟೋಕನುಗಳು) + [1]*ಉದ್ದ(ಕೊನೆ_ಟೋಕನುಗಳು))
        ದತ್ತಾಂಶ["ending_tokens"].append(ಕೊನೆ_ಟೋಕನುಗಳು)

    # have to be careful during the collation because the number of tokens in each row can differ
    max_len = max(ಉದ್ದ(ಅಡ್ಡಸಾಲು) ಗಾಗಿ ಅಡ್ಡಸಾಲು ಒಳಗೆ tok_ಅಡ್ಡಸಾಲುಗಳು)
    ಟೋಕನುಗಳು = torch.zeros((4, max_len), dtype=torch.ದೀರ್ಘ)
    ಮುಖವಾಡ = torch.zeros((4, max_len), dtype=torch.ದೀರ್ಘ)
    ಗಾಗಿ i, (tok_ಅಡ್ಡಸಾಲು, ಮುಖವಾಡ_ಅಡ್ಡಸಾಲು) ಒಳಗೆ enumerate(zip(tok_ಅಡ್ಡಸಾಲುಗಳು, ಮುಖವಾಡ_ಅಡ್ಡಸಾಲುಗಳು)):
        ಟೋಕನುಗಳು[i, :ಉದ್ದ(tok_ಅಡ್ಡಸಾಲು)] = torch.tensor(tok_ಅಡ್ಡಸಾಲು)
        ಮುಖವಾಡ[i, :ಉದ್ದ(ಮುಖವಾಡ_ಅಡ್ಡಸಾಲು)] = torch.tensor(ಮುಖವಾಡ_ಅಡ್ಡಸಾಲು)

    ಹಿಂತಿರುಗಿ ದತ್ತಾಂಶ, ಟೋಕನುಗಳು, ಮುಖವಾಡ, label

ವ್ಯಾಖ್ಯಾನ iterate_examples(ವಿಭಜಿಸು):
    # there are 10,042 examples in total in val
    download(ವಿಭಜಿಸು)
    ಜೊತೆ open(os.ಮಾರ್ಗ.join(ದತ್ತಾಂಶ_CACHE_DIR, f"hellaswag_{split}.jsonl"), "r") ಇವ f:
        ಗಾಗಿ ಸಾಲು ಒಳಗೆ f:
            example = json.loads(ಸಾಲು)
            ನೀಡು example

@torch.no_grad()
ವ್ಯಾಖ್ಯಾನ evaluate(model_type, device):

    torch.ಗಣ_float32_matmul_precision('high') # use tf32

    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.ಗಮ್ಯ(device)
    # model = torch.compile(model)

    datas = []
    ಅಂಕೆ_correct_norm = 0
    ಅಂಕೆ_correct = 0
    ಅಂಕೆ_ಒಟ್ಟು = 0
    ಗಾಗಿ example ಒಳಗೆ iterate_examples("val"):
        ದತ್ತಾಂಶ, ಟೋಕನುಗಳು, ಮುಖವಾಡ, label = render_example(example)
        datas.append(ದತ್ತಾಂಶ)
        ಟೋಕನುಗಳು = ಟೋಕನುಗಳು.ಗಮ್ಯ(device)
        ಮುಖವಾಡ = ಮುಖವಾಡ.ಗಮ್ಯ(device)

        # get the logits
        logits = model(ಟೋಕನುಗಳು).logits
        # evaluate the autoregressive loss at all positions
        ಸರಿಸು_logits = (logits[..., :-1, :]).contiguous()
        ಸರಿಸು_ಟೋಕನುಗಳು = (ಟೋಕನುಗಳು[..., 1:]).contiguous()
        flat_ಸರಿಸು_logits = ಸರಿಸು_logits.view(-1, ಸರಿಸು_logits.ಗಾತ್ರ_2(-1))
        flat_ಸರಿಸು_ಟೋಕನುಗಳು = ಸರಿಸು_ಟೋಕನುಗಳು.view(-1)
        ಸರಿಸು_losses = F.cross_entropy(flat_ಸರಿಸು_logits, flat_ಸರಿಸು_ಟೋಕನುಗಳು, reduction='none')
        ಸರಿಸು_losses = ಸರಿಸು_losses.view(ಟೋಕನುಗಳು.ಗಾತ್ರ_2(0), -1)
        # now get the average loss just for the completion region (where mask == 1), in each row
        ಸರಿಸು_ಮುಖವಾಡ = (ಮುಖವಾಡ[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
        masked_ಸರಿಸು_losses = ಸರಿಸು_losses * ಸರಿಸು_ಮುಖವಾಡ
        # sum and divide by the number of 1s in the mask
        ಮೊತ್ತ_loss = masked_ಸರಿಸು_losses.ಮೊತ್ತ(dim=1)
        avg_loss = ಮೊತ್ತ_loss / ಸರಿಸು_ಮುಖವಾಡ.ಮೊತ್ತ(dim=1)
        # now we have a loss for each of the 4 completions
        # the one with the lowest loss should be the most likely
        pred = ಮೊತ್ತ_loss.argmin().ಅಂಶ()
        pred_norm = avg_loss.argmin().ಅಂಶ()

        # accumulate stats
        ಅಂಕೆ_ಒಟ್ಟು += 1
        ಅಂಕೆ_correct += ಪೂರ್ಣಾಂಕ(pred == label)
        ಅಂಕೆ_correct_norm += ಪೂರ್ಣಾಂಕ(pred_norm == label)
        ಮುದ್ರಿಸು(f"{num_total} acc: {num_correct/num_total:.4f} acc_norm: {num_correct_norm}/{num_total}={num_correct_norm/num_total:.4f}")

        # debug: pretty print a few examples, and the losses in each case
        ಆದರೆ ಅಂಕೆ_ಒಟ್ಟು < 10:
            ಮುದ್ರಿಸು("---")
            ಮುದ್ರಿಸು(f"Context:\n {example['ctx']}")
            ಮುದ್ರಿಸು(f"Endings:")
            ಗಾಗಿ i, ಕೊನೆ ಒಳಗೆ enumerate(example["endings"]):
                ಮುದ್ರಿಸು(f"{i} (loss: {avg_loss[i].item():.4f}) {end}")
            ಮುದ್ರಿಸು(f"predicted: {pred_norm}, actual: {label}")

    # now write the data to a .bin file
    filename = os.ಮಾರ್ಗ.join(ದತ್ತಾಂಶ_CACHE_DIR, f"hellaswag_val.bin")
    write_evalfile(filename, datas)

ಆದರೆ __name__ == "__main__":
    ಆಮದು argparse
    parser = argparse.ವಾದ_Parser()
    parser.ಕೂಡಿಸು_ವಾದ("-m", "--model_type", type=str, ಪೂರ್ವನಿಯೋಜಿತ="gpt2", help="the model type to use")
    parser.ಕೂಡಿಸು_ವಾದ("-d", "--device", type=str, ಪೂರ್ವನಿಯೋಜಿತ="cuda", help="the device to use")
    ವಾದಗಳು = parser.parse_ವಾದಗಳು()
    evaluate(ವಾದಗಳು.model_type, ವಾದಗಳು.device)
