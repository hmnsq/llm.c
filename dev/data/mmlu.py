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
استورد بینتیاں
استورد tiktoken
استورد pandas كـ pd
من tqdm استورد tqdm
استورد torch
استورد torch.nn كـ nn
من torch.nn استورد functional كـ F
من transformers استورد GPT2LMHeadModel
من انکڑے_common استورد download_file

# -----------------------------------------------------------------------------
انکڑے_CACHE_DIR = os.رستہ.join(os.رستہ.dirname(__file__), "mmlu")

enc = tiktoken.get_encoding("gpt2")
انکڑے_url = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"

تعریف download():
    """Downloads MMLU to DATA_CACHE_DIR"""
    os.makedirs(انکڑے_CACHE_DIR, exist_ok=سچ)
    انکڑے_filename = os.رستہ.join(انکڑے_CACHE_DIR, f"data.tar")
    إذا ليس os.رستہ.exists(انکڑے_filename):
        چھاپو(f"Downloading {data_url} to {data_filename}...")
        download_file(انکڑے_url, انکڑے_filename)
        os.system(f"tar -xf {data_filename} -C {DATA_CACHE_DIR}") # untar
        # creates a directory "data" inside it, with e.g. data/test/*csv
    وإلا:
        چھاپو(f"{data_filename} already exists, skipping download...")

تعریف iterate_examples():
    # there are 14,042 examples in total in the test set

    download()
    test_dir = os.رستہ.join(انکڑے_CACHE_DIR, "data", "test")
    csv_files = [f لكل f في os.listdir(test_dir) إذا f.endswith(".csv")]
    لكل csv_file في csv_files:
        csv_رستہ = os.رستہ.join(test_dir, csv_file)
        چھاپو(csv_رستہ)
        df = pd.read_csv(csv_رستہ, سرخی=کجھ_نئیں)
        n = df.shape[0]
        لكل idx في سلسلہ(n):
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
    ctx_ٹوکن = enc.encode(ctx)

    tok_صفاں = []
    نقاب_صفاں = []
    لكل سرا في example["endings"]:
        سرا_ٹوکن = enc.encode(" " + str(سرا)) # note: prepending " " because GPT-2 tokenizer
        tok_صفاں.append(ctx_ٹوکن + سرا_ٹوکن)
        نقاب_صفاں.append([0]*لمبائی(ctx_ٹوکن) + [1]*لمبائی(سرا_ٹوکن))

    # have to be careful during the collation because the number of tokens in each row can differ
    max_len = max(لمبائی(صف) لكل صف في tok_صفاں)
    ٹوکن = torch.zeros((4, max_len), dtype=torch.لما)
    نقاب = torch.zeros((4, max_len), dtype=torch.لما)
    لكل i, (tok_صف, نقاب_صف) في enumerate(zip(tok_صفاں, نقاب_صفاں)):
        ٹوکن[i, :لمبائی(tok_صف)] = torch.tensor(tok_صف)
        نقاب[i, :لمبائی(نقاب_صف)] = torch.tensor(نقاب_صف)

    label = "ABCD".اشاریہ(example["label"])
    أرجع ٹوکن, نقاب, label

@torch.no_grad()
تعریف evaluate(model_type, device):

    torch.سیٹ_float32_matmul_precision('high') # use tf32

    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.منزل(device)
    # model = torch.compile(model)

    عدد_correct_norm = 0
    عدد_correct = 0
    عدد_کل = 0
    لكل example في iterate_examples():
        ٹوکن, نقاب, label = render_example(example)
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
        چھاپو(f"{num_total} acc: {num_correct/num_total:.4f} acc_norm: {num_correct_norm/num_total:.4f}")

        # debug prints
        إذا عدد_کل < 10:
            چھاپو("---")
            چھاپو(f"Context:\n {example['question']}")
            چھاپو(f"Endings:")
            لكل i, سرا في enumerate(example["endings"]):
                چھاپو(f"{i} (loss: {avg_loss[i].item():.4f}) {end}")
            چھاپو(f"predicted: {pred}, actual: {label}")

إذا __name__ == "__main__":
    استورد argparse
    parser = argparse.دلیل_Parser()
    parser.جوڑو_دلیل("-m", "--model_type", type=str, افتراضي="gpt2", help="the model type to use")
    parser.جوڑو_دلیل("-d", "--device", type=str, افتراضي="cuda", help="the device to use")
    دلیلاں = parser.parse_دلیلاں()
    evaluate(دلیلاں.model_type, دلیلاں.device)
