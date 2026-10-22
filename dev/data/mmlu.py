"""
Downloads and evaluates MMLU in Python.
This then acts as the reference file for llm.c
https://github.com/hendrycks/test

gpt2 (124M)
- this script: 14042 acc: 0.2557 acc_norm: 0.2721

gpt2-xl (1558M)
- this script: 14042 acc: 0.2927 acc_norm: 0.3035
"""

استورد os
استورد غوښتنې
استورد tiktoken
استورد pandas كـ pd
من tqdm استورد tqdm
استورد torch
استورد torch.nn كـ nn
من torch.nn استورد functional كـ F
من transformers استورد GPT2LMHeadModel
من معلومات_common استورد download_file

# -----------------------------------------------------------------------------
معلومات_CACHE_DIR = os.لار.join(os.لار.dirname(__file__), "mmlu")

enc = tiktoken.get_encoding("gpt2")
معلومات_url = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"

تعریف download():
    """Downloads MMLU to DATA_CACHE_DIR"""
    os.makedirs(معلومات_CACHE_DIR, exist_ok=ریښتیا)
    معلومات_filename = os.لار.join(معلومات_CACHE_DIR, f"data.tar")
    إذا ليس os.لار.exists(معلومات_filename):
        چاپ(f"Downloading {data_url} to {data_filename}...")
        download_file(معلومات_url, معلومات_filename)
        os.system(f"tar -xf {data_filename} -C {DATA_CACHE_DIR}") # untar
        # creates a directory "data" inside it, with e.g. data/test/*csv
    وإلا:
        چاپ(f"{data_filename} already exists, skipping download...")

تعریف iterate_examples():
    # there are 14,042 examples in total in the test set

    download()
    test_dir = os.لار.join(معلومات_CACHE_DIR, "data", "test")
    csv_files = [f لكل f في os.listdir(test_dir) إذا f.endswith(".csv")]
    لكل csv_file في csv_files:
        csv_لار = os.لار.join(test_dir, csv_file)
        چاپ(csv_لار)
        df = pd.read_csv(csv_لار, سرلیک=هیڅ)
        n = df.shape[0]
        لكل idx في لړ(n):
            example = {
                "question": df.iloc[idx, 0],
                "endings": [df.iloc[idx, 1], df.iloc[idx, 2], df.iloc[idx, 3], df.iloc[idx, 4]],
                "label": df.iloc[idx, 5],
            }
            أنتج example

تعریف render_example(example):
    """
    Given the example as a dictionary, render it as three torch tensors:
    - tokens (the tokens of context + completion, of size 4xN, as there are always 4 candidates)
    - mask (is 1 in the region of the candidate completion, where we evaluate likelihoods)
    - label (the index of the correct completion, which we hope has the highest likelihood)
    """
    ctx = f"Question: {example['question']}\n\nAnswer:"
    ctx_نښې = enc.encode(ctx)

    tok_لیکې = []
    ماسک_لیکې = []
    لكل پای_ټکی في example["endings"]:
        پای_ټکی_نښې = enc.encode(" " + str(پای_ټکی)) # note: prepending " " because GPT-2 tokenizer
        tok_لیکې.append(ctx_نښې + پای_ټکی_نښې)
        ماسک_لیکې.append([0]*اوږدوالی(ctx_نښې) + [1]*اوږدوالی(پای_ټکی_نښې))

    # have to be careful during the collation because the number of tokens in each row can differ
    max_len = max(اوږدوالی(لیکه) لكل لیکه في tok_لیکې)
    نښې = torch.zeros((4, max_len), dtype=torch.اوږد)
    ماسک = torch.zeros((4, max_len), dtype=torch.اوږد)
    لكل i, (tok_لیکه, ماسک_لیکه) في enumerate(zip(tok_لیکې, ماسک_لیکې)):
        نښې[i, :اوږدوالی(tok_لیکه)] = torch.tensor(tok_لیکه)
        ماسک[i, :اوږدوالی(ماسک_لیکه)] = torch.tensor(ماسک_لیکه)

    label = "ABCD".شاخص(example["label"])
    أرجع نښې, ماسک, label

@torch.no_grad()
تعریف evaluate(model_type, device):

    torch.سېټ_float32_matmul_precision('high') # use tf32

    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.منزل(device)
    # model = torch.compile(model)

    عدد_correct_norm = 0
    عدد_correct = 0
    عدد_ټول = 0
    لكل example في iterate_examples():
        نښې, ماسک, label = render_example(example)
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
        چاپ(f"{num_total} acc: {num_correct/num_total:.4f} acc_norm: {num_correct_norm/num_total:.4f}")

        # debug prints
        إذا عدد_ټول < 10:
            چاپ("---")
            چاپ(f"Context:\n {example['question']}")
            چاپ(f"Endings:")
            لكل i, پای_ټکی في enumerate(example["endings"]):
                چاپ(f"{i} (loss: {avg_loss[i].item():.4f}) {end}")
            چاپ(f"predicted: {pred}, actual: {label}")

إذا __name__ == "__main__":
    استورد argparse
    parser = argparse.دلیل_Parser()
    parser.جمع_کړه_دلیل("-m", "--model_type", type=str, افتراضي="gpt2", help="the model type to use")
    parser.جمع_کړه_دلیل("-d", "--device", type=str, افتراضي="cuda", help="the device to use")
    دلیلونه = parser.parse_دلیلونه()
    evaluate(دلیلونه.model_type, دلیلونه.device)
