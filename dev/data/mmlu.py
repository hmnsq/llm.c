"""
Downloads and evaluates MMLU in Python.
This then acts as the reference file for llm.c
https://github.com/hendrycks/test

gpt2 (124M)
- this script: 14042 acc: 0.2557 acc_norm: 0.2721

gpt2-xl (1558M)
- this script: 14042 acc: 0.2927 acc_norm: 0.3035
"""

درآمد os
درآمد درخواستیں
درآمد tiktoken
درآمد pandas بطور pd
سے tqdm درآمد tqdm
درآمد torch
درآمد torch.nn بطور nn
سے torch.nn درآمد functional بطور F
سے transformers درآمد GPT2LMHeadModel
سے ڈیٹا_common درآمد download_file

# -----------------------------------------------------------------------------
ڈیٹا_CACHE_DIR = os.راستہ.join(os.راستہ.dirname(__file__), "mmlu")

enc = tiktoken.get_encoding("gpt2")
ڈیٹا_url = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"

تعریف download():
    """Downloads MMLU to DATA_CACHE_DIR"""
    os.makedirs(ڈیٹا_CACHE_DIR, exist_ok=سچ)
    ڈیٹا_filename = os.راستہ.join(ڈیٹا_CACHE_DIR, f"data.tar")
    اگر نہیں os.راستہ.exists(ڈیٹا_filename):
        چھاپو(f"Downloading {data_url} to {data_filename}...")
        download_file(ڈیٹا_url, ڈیٹا_filename)
        os.system(f"tar -xf {data_filename} -C {DATA_CACHE_DIR}") # untar
        # creates a directory "data" inside it, with e.g. data/test/*csv
    ورنہ:
        چھاپو(f"{data_filename} already exists, skipping download...")

تعریف iterate_examples():
    # there are 14,042 examples in total in the test set

    download()
    test_dir = os.راستہ.join(ڈیٹا_CACHE_DIR, "data", "test")
    csv_files = [f برائے f میں os.listdir(test_dir) اگر f.endswith(".csv")]
    برائے csv_file میں csv_files:
        csv_راستہ = os.راستہ.join(test_dir, csv_file)
        چھاپو(csv_راستہ)
        df = pd.read_csv(csv_راستہ, سرخی=کچھ_نہیں)
        n = df.shape[0]
        برائے idx میں سلسلہ(n):
            example = {
                "question": df.iloc[idx, 0],
                "endings": [df.iloc[idx, 1], df.iloc[idx, 2], df.iloc[idx, 3], df.iloc[idx, 4]],
                "label": df.iloc[idx, 5],
            }
            دو example

تعریف render_example(example):
    """
    Given the example as a dictionary, render it as three torch tensors:
    - tokens (the tokens of context + completion, of size 4xN, as there are always 4 candidates)
    - mask (is 1 in the region of the candidate completion, where we evaluate likelihoods)
    - label (the index of the correct completion, which we hope has the highest likelihood)
    """
    ctx = f"Question: {example['question']}\n\nAnswer:"
    ctx_علامات = enc.encode(ctx)

    tok_صفیں = []
    نقاب_صفیں = []
    برائے انتہا میں example["endings"]:
        انتہا_علامات = enc.encode(" " + str(انتہا)) # note: prepending " " because GPT-2 tokenizer
        tok_صفیں.append(ctx_علامات + انتہا_علامات)
        نقاب_صفیں.append([0]*لمبائی(ctx_علامات) + [1]*لمبائی(انتہا_علامات))

    # have to be careful during the collation because the number of tokens in each row can differ
    max_len = max(لمبائی(صف) برائے صف میں tok_صفیں)
    علامات = torch.zeros((4, max_len), dtype=torch.طویل)
    نقاب = torch.zeros((4, max_len), dtype=torch.طویل)
    برائے i, (tok_صف, نقاب_صف) میں enumerate(zip(tok_صفیں, نقاب_صفیں)):
        علامات[i, :لمبائی(tok_صف)] = torch.tensor(tok_صف)
        نقاب[i, :لمبائی(نقاب_صف)] = torch.tensor(نقاب_صف)

    label = "ABCD".اشاریہ(example["label"])
    واپس علامات, نقاب, label

@torch.no_grad()
تعریف evaluate(model_type, device):

    torch.سیٹ_float32_matmul_precision('high') # use tf32

    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.مقصد(device)
    # model = torch.compile(model)

    عدد_correct_norm = 0
    عدد_correct = 0
    عدد_کل = 0
    برائے example میں iterate_examples():
        علامات, نقاب, label = render_example(example)
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
        چھاپو(f"{num_total} acc: {num_correct/num_total:.4f} acc_norm: {num_correct_norm/num_total:.4f}")

        # debug prints
        اگر عدد_کل < 10:
            چھاپو("---")
            چھاپو(f"Context:\n {example['question']}")
            چھاپو(f"Endings:")
            برائے i, انتہا میں enumerate(example["endings"]):
                چھاپو(f"{i} (loss: {avg_loss[i].item():.4f}) {end}")
            چھاپو(f"predicted: {pred}, actual: {label}")

اگر __name__ == "__main__":
    درآمد argparse
    parser = argparse.دلیل_Parser()
    parser.جوڑو_دلیل("-m", "--model_type", type=str, طےشدہ="gpt2", help="the model type to use")
    parser.جوڑو_دلیل("-d", "--device", type=str, طےشدہ="cuda", help="the device to use")
    دلائل = parser.parse_دلائل()
    evaluate(دلائل.model_type, دلائل.device)
