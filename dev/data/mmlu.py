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
استورد درخواستون
استورد tiktoken
استورد pandas كـ pd
من tqdm استورد tqdm
استورد torch
استورد torch.nn كـ nn
من torch.nn استورد functional كـ F
من transformers استورد GPT2LMHeadModel
من معلومات_common استورد download_file

# -----------------------------------------------------------------------------
معلومات_CACHE_DIR = os.رستو.join(os.رستو.dirname(__file__), "mmlu")

enc = tiktoken.get_encoding("gpt2")
معلومات_url = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"

وصف download():
    """Downloads MMLU to DATA_CACHE_DIR"""
    os.makedirs(معلومات_CACHE_DIR, exist_ok=سچ)
    معلومات_filename = os.رستو.join(معلومات_CACHE_DIR, f"data.tar")
    إذا ليس os.رستو.exists(معلومات_filename):
        ڇاپيو(f"Downloading {data_url} to {data_filename}...")
        download_file(معلومات_url, معلومات_filename)
        os.system(f"tar -xf {data_filename} -C {DATA_CACHE_DIR}") # untar
        # creates a directory "data" inside it, with e.g. data/test/*csv
    وإلا:
        ڇاپيو(f"{data_filename} already exists, skipping download...")

وصف iterate_examples():
    # there are 14,042 examples in total in the test set

    download()
    test_dir = os.رستو.join(معلومات_CACHE_DIR, "data", "test")
    csv_files = [f لكل f في os.listdir(test_dir) إذا f.endswith(".csv")]
    لكل csv_file في csv_files:
        csv_رستو = os.رستو.join(test_dir, csv_file)
        ڇاپيو(csv_رستو)
        df = pd.read_csv(csv_رستو, سرو=ڪجهه_نه)
        n = df.shape[0]
        لكل idx في سلسلو(n):
            example = {
                "question": df.iloc[idx, 0],
                "endings": [df.iloc[idx, 1], df.iloc[idx, 2], df.iloc[idx, 3], df.iloc[idx, 4]],
                "label": df.iloc[idx, 5],
            }
            أنتج example

وصف render_example(example):
    """
    Given the example as a dictionary, render it as three torch tensors:
    - tokens (the tokens of context + completion, of size 4xN, as there are always 4 candidates)
    - mask (is 1 in the region of the candidate completion, where we evaluate likelihoods)
    - label (the index of the correct completion, which we hope has the highest likelihood)
    """
    ctx = f"Question: {example['question']}\n\nAnswer:"
    ctx_ٽوڪن = enc.encode(ctx)

    tok_صفون = []
    نقاب_صفون = []
    لكل پڇاڙي في example["endings"]:
        پڇاڙي_ٽوڪن = enc.encode(" " + str(پڇاڙي)) # note: prepending " " because GPT-2 tokenizer
        tok_صفون.append(ctx_ٽوڪن + پڇاڙي_ٽوڪن)
        نقاب_صفون.append([0]*ڊيگهه(ctx_ٽوڪن) + [1]*ڊيگهه(پڇاڙي_ٽوڪن))

    # have to be careful during the collation because the number of tokens in each row can differ
    max_len = max(ڊيگهه(صف) لكل صف في tok_صفون)
    ٽوڪن = torch.zeros((4, max_len), dtype=torch.ڊگهو)
    نقاب = torch.zeros((4, max_len), dtype=torch.ڊگهو)
    لكل i, (tok_صف, نقاب_صف) في enumerate(zip(tok_صفون, نقاب_صفون)):
        ٽوڪن[i, :ڊيگهه(tok_صف)] = torch.tensor(tok_صف)
        نقاب[i, :ڊيگهه(نقاب_صف)] = torch.tensor(نقاب_صف)

    label = "ABCD".اشاريو(example["label"])
    أرجع ٽوڪن, نقاب, label

@torch.no_grad()
وصف evaluate(model_type, device):

    torch.سيٽ_float32_matmul_precision('high') # use tf32

    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.منزل(device)
    # model = torch.compile(model)

    انگ_correct_norm = 0
    انگ_correct = 0
    انگ_ڪل = 0
    لكل example في iterate_examples():
        ٽوڪن, نقاب, label = render_example(example)
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
        ڇاپيو(f"{num_total} acc: {num_correct/num_total:.4f} acc_norm: {num_correct_norm/num_total:.4f}")

        # debug prints
        إذا انگ_ڪل < 10:
            ڇاپيو("---")
            ڇاپيو(f"Context:\n {example['question']}")
            ڇاپيو(f"Endings:")
            لكل i, پڇاڙي في enumerate(example["endings"]):
                ڇاپيو(f"{i} (loss: {avg_loss[i].item():.4f}) {end}")
            ڇاپيو(f"predicted: {pred}, actual: {label}")

إذا __name__ == "__main__":
    استورد argparse
    parser = argparse.دليل_Parser()
    parser.جوڙيو_دليل("-m", "--model_type", type=str, افتراضي="gpt2", help="the model type to use")
    parser.جوڙيو_دليل("-d", "--device", type=str, افتراضي="cuda", help="the device to use")
    دليلون = parser.parse_دليلون()
    evaluate(دليلون.model_type, دليلون.device)
