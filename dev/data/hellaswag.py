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

దిగుమతి os
దిగుమతి json
దిగుమతి అభ్యర్థనలు
దిగుమతి tiktoken
తః tqdm దిగుమతి tqdm
దిగుమతి torch
దిగుమతి torch.nn ఇవ nn
తః torch.nn దిగుమతి functional ఇవ F
తః transformers దిగుమతి GPT2LMHeadModel
తః దత్తాంశం_common దిగుమతి download_file, write_evalfile

# -----------------------------------------------------------------------------
దత్తాంశం_CACHE_DIR = os.మార్గం.join(os.మార్గం.dirname(__file__), "hellaswag")

hellaswags = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

enc = tiktoken.get_encoding("gpt2")

నిర్వచనం download(విభజించు):
    """Downloads HellaSwag DATA_CACHE_DIR"""
    os.makedirs(దత్తాంశం_CACHE_DIR, exist_ok=నిజం)
    దత్తాంశం_url = hellaswags[విభజించు]
    దత్తాంశం_filename = os.మార్గం.join(దత్తాంశం_CACHE_DIR, f"hellaswag_{split}.jsonl")
    ఒకవేళ కాదు os.మార్గం.exists(దత్తాంశం_filename):
        ముద్రించు(f"Downloading {data_url} to {data_filename}...")
        download_file(దత్తాంశం_url, దత్తాంశం_filename)
    లేకపోతే:
        ముద్రించు(f"{data_filename} already exists, skipping download...")

నిర్వచనం render_example(example):
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
    దత్తాంశం = {
        "label": label,
        "ctx_tokens": ఏమీలేదు,
        "ending_tokens": [],
    }

    # gather up all the tokens
    ctx_టోకెన్లు = enc.encode(ctx)
    దత్తాంశం["ctx_tokens"] = ctx_టోకెన్లు
    tok_అడ్డువరుసలు = []
    ముసుగు_అడ్డువరుసలు = []
    కొరకు చివర లో endings:
        చివర_టోకెన్లు = enc.encode(" " + చివర) # note: prepending " " because GPT-2 tokenizer
        tok_అడ్డువరుసలు.append(ctx_టోకెన్లు + చివర_టోకెన్లు)
        ముసుగు_అడ్డువరుసలు.append([0]*పొడవు(ctx_టోకెన్లు) + [1]*పొడవు(చివర_టోకెన్లు))
        దత్తాంశం["ending_tokens"].append(చివర_టోకెన్లు)

    # have to be careful during the collation because the number of tokens in each row can differ
    max_len = max(పొడవు(అడ్డువరుస) కొరకు అడ్డువరుస లో tok_అడ్డువరుసలు)
    టోకెన్లు = torch.zeros((4, max_len), dtype=torch.దీర్ఘ)
    ముసుగు = torch.zeros((4, max_len), dtype=torch.దీర్ఘ)
    కొరకు i, (tok_అడ్డువరుస, ముసుగు_అడ్డువరుస) లో enumerate(zip(tok_అడ్డువరుసలు, ముసుగు_అడ్డువరుసలు)):
        టోకెన్లు[i, :పొడవు(tok_అడ్డువరుస)] = torch.tensor(tok_అడ్డువరుస)
        ముసుగు[i, :పొడవు(ముసుగు_అడ్డువరుస)] = torch.tensor(ముసుగు_అడ్డువరుస)

    తిరిగి దత్తాంశం, టోకెన్లు, ముసుగు, label

నిర్వచనం iterate_examples(విభజించు):
    # there are 10,042 examples in total in val
    download(విభజించు)
    తోడుగా open(os.మార్గం.join(దత్తాంశం_CACHE_DIR, f"hellaswag_{split}.jsonl"), "r") ఇవ f:
        కొరకు పంక్తి లో f:
            example = json.loads(పంక్తి)
            ఇవ్వు example

@torch.no_grad()
నిర్వచనం evaluate(model_type, device):

    torch.సమితి_float32_matmul_precision('high') # use tf32

    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.గమ్యం(device)
    # model = torch.compile(model)

    datas = []
    అంకె_correct_norm = 0
    అంకె_correct = 0
    అంకె_మొత్తం = 0
    కొరకు example లో iterate_examples("val"):
        దత్తాంశం, టోకెన్లు, ముసుగు, label = render_example(example)
        datas.append(దత్తాంశం)
        టోకెన్లు = టోకెన్లు.గమ్యం(device)
        ముసుగు = ముసుగు.గమ్యం(device)

        # get the logits
        logits = model(టోకెన్లు).logits
        # evaluate the autoregressive loss at all positions
        జరుపు_logits = (logits[..., :-1, :]).contiguous()
        జరుపు_టోకెన్లు = (టోకెన్లు[..., 1:]).contiguous()
        flat_జరుపు_logits = జరుపు_logits.view(-1, జరుపు_logits.కొలత(-1))
        flat_జరుపు_టోకెన్లు = జరుపు_టోకెన్లు.view(-1)
        జరుపు_losses = F.cross_entropy(flat_జరుపు_logits, flat_జరుపు_టోకెన్లు, reduction='none')
        జరుపు_losses = జరుపు_losses.view(టోకెన్లు.కొలత(0), -1)
        # now get the average loss just for the completion region (where mask == 1), in each row
        జరుపు_ముసుగు = (ముసుగు[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
        masked_జరుపు_losses = జరుపు_losses * జరుపు_ముసుగు
        # sum and divide by the number of 1s in the mask
        కూడిక_loss = masked_జరుపు_losses.కూడిక(dim=1)
        avg_loss = కూడిక_loss / జరుపు_ముసుగు.కూడిక(dim=1)
        # now we have a loss for each of the 4 completions
        # the one with the lowest loss should be the most likely
        pred = కూడిక_loss.argmin().అంశం()
        pred_norm = avg_loss.argmin().అంశం()

        # accumulate stats
        అంకె_మొత్తం += 1
        అంకె_correct += పూర్ణాంకం(pred == label)
        అంకె_correct_norm += పూర్ణాంకం(pred_norm == label)
        ముద్రించు(f"{num_total} acc: {num_correct/num_total:.4f} acc_norm: {num_correct_norm}/{num_total}={num_correct_norm/num_total:.4f}")

        # debug: pretty print a few examples, and the losses in each case
        ఒకవేళ అంకె_మొత్తం < 10:
            ముద్రించు("---")
            ముద్రించు(f"Context:\n {example['ctx']}")
            ముద్రించు(f"Endings:")
            కొరకు i, చివర లో enumerate(example["endings"]):
                ముద్రించు(f"{i} (loss: {avg_loss[i].item():.4f}) {end}")
            ముద్రించు(f"predicted: {pred_norm}, actual: {label}")

    # now write the data to a .bin file
    filename = os.మార్గం.join(దత్తాంశం_CACHE_DIR, f"hellaswag_val.bin")
    write_evalfile(filename, datas)

ఒకవేళ __name__ == "__main__":
    దిగుమతి argparse
    parser = argparse.వాదం_Parser()
    parser.కూడు_వాదం("-m", "--model_type", type=str, అప్రమేయ="gpt2", help="the model type to use")
    parser.కూడు_వాదం("-d", "--device", type=str, అప్రమేయ="cuda", help="the device to use")
    వాదనలు = parser.parse_వాదనలు()
    evaluate(వాదనలు.model_type, వాదనలు.device)
