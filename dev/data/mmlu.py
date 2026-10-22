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
استورد درخواستها
استورد tiktoken
استورد pandas كـ pd
من tqdm استورد tqdm
استورد torch
استورد torch.nn كـ nn
من torch.nn استورد functional كـ F
من transformers استورد GPT2LMHeadModel
من داده_common استورد download_file

# -----------------------------------------------------------------------------
داده_CACHE_DIR = os.مسیر.join(os.مسیر.dirname(__file__), "mmlu")

enc = tiktoken.get_encoding("gpt2")
داده_url = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"

تعریف download():
    """Downloads MMLU to DATA_CACHE_DIR"""
    os.makedirs(داده_CACHE_DIR, exist_ok=درست)
    داده_filename = os.مسیر.join(داده_CACHE_DIR, f"data.tar")
    إذا ليس os.مسیر.exists(داده_filename):
        چاپ(f"Downloading {data_url} to {data_filename}...")
        download_file(داده_url, داده_filename)
        os.system(f"tar -xf {data_filename} -C {DATA_CACHE_DIR}") # untar
        # creates a directory "data" inside it, with e.g. data/test/*csv
    وإلا:
        چاپ(f"{data_filename} already exists, skipping download...")

تعریف iterate_examples():
    # there are 14,042 examples in total in the test set

    download()
    test_dir = os.مسیر.join(داده_CACHE_DIR, "data", "test")
    csv_files = [f لكل f في os.listdir(test_dir) إذا f.endswith(".csv")]
    لكل csv_file في csv_files:
        csv_مسیر = os.مسیر.join(test_dir, csv_file)
        چاپ(csv_مسیر)
        df = pd.read_csv(csv_مسیر, سرآیند=هیچ)
        n = df.shape[0]
        لكل idx في دامنه(n):
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
    ctx_نشانهها = enc.encode(ctx)

    tok_ردیفها = []
    ماسک_ردیفها = []
    لكل انتها في example["endings"]:
        انتها_نشانهها = enc.encode(" " + str(انتها)) # note: prepending " " because GPT-2 tokenizer
        tok_ردیفها.append(ctx_نشانهها + انتها_نشانهها)
        ماسک_ردیفها.append([0]*طول(ctx_نشانهها) + [1]*طول(انتها_نشانهها))

    # have to be careful during the collation because the number of tokens in each row can differ
    max_len = max(طول(ردیف) لكل ردیف في tok_ردیفها)
    نشانهها = torch.zeros((4, max_len), dtype=torch.دراز)
    ماسک = torch.zeros((4, max_len), dtype=torch.دراز)
    لكل i, (tok_ردیف, ماسک_ردیف) في enumerate(zip(tok_ردیفها, ماسک_ردیفها)):
        نشانهها[i, :طول(tok_ردیف)] = torch.tensor(tok_ردیف)
        ماسک[i, :طول(ماسک_ردیف)] = torch.tensor(ماسک_ردیف)

    label = "ABCD".شاخص(example["label"])
    أرجع نشانهها, ماسک, label

@torch.no_grad()
تعریف evaluate(model_type, device):

    torch.مجموعه_float32_matmul_precision('high') # use tf32

    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.مقصد(device)
    # model = torch.compile(model)

    رقم_correct_norm = 0
    رقم_correct = 0
    رقم_مجموع = 0
    لكل example في iterate_examples():
        نشانهها, ماسک, label = render_example(example)
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
        چاپ(f"{num_total} acc: {num_correct/num_total:.4f} acc_norm: {num_correct_norm/num_total:.4f}")

        # debug prints
        إذا رقم_مجموع < 10:
            چاپ("---")
            چاپ(f"Context:\n {example['question']}")
            چاپ(f"Endings:")
            لكل i, انتها في enumerate(example["endings"]):
                چاپ(f"{i} (loss: {avg_loss[i].item():.4f}) {end}")
            چاپ(f"predicted: {pred}, actual: {label}")

إذا __name__ == "__main__":
    استورد argparse
    parser = argparse.آرگومان_Parser()
    parser.جمع_کن_آرگومان("-m", "--model_type", type=str, افتراضي="gpt2", help="the model type to use")
    parser.جمع_کن_آرگومان("-d", "--device", type=str, افتراضي="cuda", help="the device to use")
    آرگومانها = parser.parse_آرگومانها()
    evaluate(آرگومانها.model_type, آرگومانها.device)
