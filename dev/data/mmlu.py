"""
Downloads and evaluates MMLU in Python.
This then acts as the reference file for llm.c
https://github.com/hendrycks/test

gpt2 (124M)
- this script: 14042 acc: 0.2557 acc_norm: 0.2721

gpt2-xl (1558M)
- this script: 14042 acc: 0.2927 acc_norm: 0.3035
"""

దిగుమతి os
దిగుమతి అభ్యర్థనలు
దిగుమతి tiktoken
దిగుమతి pandas ఇవ pd
తః tqdm దిగుమతి tqdm
దిగుమతి torch
దిగుమతి torch.nn ఇవ nn
తః torch.nn దిగుమతి functional ఇవ F
తః transformers దిగుమతి GPT2LMHeadModel
తః దత్తాంశం_common దిగుమతి download_file

# -----------------------------------------------------------------------------
దత్తాంశం_CACHE_DIR = os.మార్గం.join(os.మార్గం.dirname(__file__), "mmlu")

enc = tiktoken.get_encoding("gpt2")
దత్తాంశం_url = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"

నిర్వచనం download():
    """Downloads MMLU to DATA_CACHE_DIR"""
    os.makedirs(దత్తాంశం_CACHE_DIR, exist_ok=నిజం)
    దత్తాంశం_filename = os.మార్గం.join(దత్తాంశం_CACHE_DIR, f"data.tar")
    ఒకవేళ కాదు os.మార్గం.exists(దత్తాంశం_filename):
        ముద్రించు(f"Downloading {data_url} to {data_filename}...")
        download_file(దత్తాంశం_url, దత్తాంశం_filename)
        os.system(f"tar -xf {data_filename} -C {DATA_CACHE_DIR}") # untar
        # creates a directory "data" inside it, with e.g. data/test/*csv
    లేకపోతే:
        ముద్రించు(f"{data_filename} already exists, skipping download...")

నిర్వచనం iterate_examples():
    # there are 14,042 examples in total in the test set

    download()
    test_dir = os.మార్గం.join(దత్తాంశం_CACHE_DIR, "data", "test")
    csv_files = [f కొరకు f లో os.listdir(test_dir) ఒకవేళ f.endswith(".csv")]
    కొరకు csv_file లో csv_files:
        csv_మార్గం = os.మార్గం.join(test_dir, csv_file)
        ముద్రించు(csv_మార్గం)
        df = pd.read_csv(csv_మార్గం, శీర్షిక=ఏమీలేదు)
        n = df.shape[0]
        కొరకు idx లో శ్రేణి(n):
            example = {
                "question": df.iloc[idx, 0],
                "endings": [df.iloc[idx, 1], df.iloc[idx, 2], df.iloc[idx, 3], df.iloc[idx, 4]],
                "label": df.iloc[idx, 5],
            }
            ఇవ్వు example

నిర్వచనం render_example(example):
    """
    Given the example as a dictionary, render it as three torch tensors:
    - tokens (the tokens of context + completion, of size 4xN, as there are always 4 candidates)
    - mask (is 1 in the region of the candidate completion, where we evaluate likelihoods)
    - label (the index of the correct completion, which we hope has the highest likelihood)
    """
    ctx = f"Question: {example['question']}\n\nAnswer:"
    ctx_టోకెన్లు = enc.encode(ctx)

    tok_అడ్డువరుసలు = []
    ముసుగు_అడ్డువరుసలు = []
    కొరకు చివర లో example["endings"]:
        చివర_టోకెన్లు = enc.encode(" " + str(చివర)) # note: prepending " " because GPT-2 tokenizer
        tok_అడ్డువరుసలు.append(ctx_టోకెన్లు + చివర_టోకెన్లు)
        ముసుగు_అడ్డువరుసలు.append([0]*పొడవు(ctx_టోకెన్లు) + [1]*పొడవు(చివర_టోకెన్లు))

    # have to be careful during the collation because the number of tokens in each row can differ
    max_len = max(పొడవు(అడ్డువరుస) కొరకు అడ్డువరుస లో tok_అడ్డువరుసలు)
    టోకెన్లు = torch.zeros((4, max_len), dtype=torch.దీర్ఘ)
    ముసుగు = torch.zeros((4, max_len), dtype=torch.దీర్ఘ)
    కొరకు i, (tok_అడ్డువరుస, ముసుగు_అడ్డువరుస) లో enumerate(zip(tok_అడ్డువరుసలు, ముసుగు_అడ్డువరుసలు)):
        టోకెన్లు[i, :పొడవు(tok_అడ్డువరుస)] = torch.tensor(tok_అడ్డువరుస)
        ముసుగు[i, :పొడవు(ముసుగు_అడ్డువరుస)] = torch.tensor(ముసుగు_అడ్డువరుస)

    label = "ABCD".సూచీ(example["label"])
    తిరిగి టోకెన్లు, ముసుగు, label

@torch.no_grad()
నిర్వచనం evaluate(model_type, device):

    torch.సమితి_float32_matmul_precision('high') # use tf32

    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.గమ్యం(device)
    # model = torch.compile(model)

    అంకె_correct_norm = 0
    అంకె_correct = 0
    అంకె_మొత్తం = 0
    కొరకు example లో iterate_examples():
        టోకెన్లు, ముసుగు, label = render_example(example)
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
        ముద్రించు(f"{num_total} acc: {num_correct/num_total:.4f} acc_norm: {num_correct_norm/num_total:.4f}")

        # debug prints
        ఒకవేళ అంకె_మొత్తం < 10:
            ముద్రించు("---")
            ముద్రించు(f"Context:\n {example['question']}")
            ముద్రించు(f"Endings:")
            కొరకు i, చివర లో enumerate(example["endings"]):
                ముద్రించు(f"{i} (loss: {avg_loss[i].item():.4f}) {end}")
            ముద్రించు(f"predicted: {pred}, actual: {label}")

ఒకవేళ __name__ == "__main__":
    దిగుమతి argparse
    parser = argparse.వాదం_Parser()
    parser.కూడు_వాదం("-m", "--model_type", type=str, అప్రమేయ="gpt2", help="the model type to use")
    parser.కూడు_వాదం("-d", "--device", type=str, అప్రమేయ="cuda", help="the device to use")
    వాదనలు = parser.parse_వాదనలు()
    evaluate(వాదనలు.model_type, వాదనలు.device)
