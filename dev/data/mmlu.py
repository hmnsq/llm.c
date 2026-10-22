"""
Downloads and evaluates MMLU in Python.
This then acts as the reference file for llm.c
https://github.com/hendrycks/test

gpt2 (124M)
- this script: 14042 acc: 0.2557 acc_norm: 0.2721

gpt2-xl (1558M)
- this script: 14042 acc: 0.2927 acc_norm: 0.3035
"""

ਆਯਾਤ os
ਆਯਾਤ ਬੇਨਤੀਆਂ
ਆਯਾਤ tiktoken
ਆਯਾਤ pandas ਇਵ pd
ਤਃ tqdm ਆਯਾਤ tqdm
ਆਯਾਤ torch
ਆਯਾਤ torch.nn ਇਵ nn
ਤਃ torch.nn ਆਯਾਤ functional ਇਵ F
ਤਃ transformers ਆਯਾਤ GPT2LMHeadModel
ਤਃ ਅੰਕੜੇ_common ਆਯਾਤ download_file

# -----------------------------------------------------------------------------
ਅੰਕੜੇ_CACHE_DIR = os.ਰਸਤਾ.join(os.ਰਸਤਾ.dirname(__file__), "mmlu")

enc = tiktoken.get_encoding("gpt2")
ਅੰਕੜੇ_url = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"

ਪਰਿਭਾਸ਼ਾ download():
    """Downloads MMLU to DATA_CACHE_DIR"""
    os.makedirs(ਅੰਕੜੇ_CACHE_DIR, exist_ok=ਸੱਚ)
    ਅੰਕੜੇ_filename = os.ਰਸਤਾ.join(ਅੰਕੜੇ_CACHE_DIR, f"data.tar")
    ਜੇ ਨਹੀਂ os.ਰਸਤਾ.exists(ਅੰਕੜੇ_filename):
        ਛਾਪੋ(f"Downloading {data_url} to {data_filename}...")
        download_file(ਅੰਕੜੇ_url, ਅੰਕੜੇ_filename)
        os.system(f"tar -xf {data_filename} -C {DATA_CACHE_DIR}") # untar
        # creates a directory "data" inside it, with e.g. data/test/*csv
    ਨਹੀਂਤਾਂ:
        ਛਾਪੋ(f"{data_filename} already exists, skipping download...")

ਪਰਿਭਾਸ਼ਾ iterate_examples():
    # there are 14,042 examples in total in the test set

    download()
    test_dir = os.ਰਸਤਾ.join(ਅੰਕੜੇ_CACHE_DIR, "data", "test")
    csv_files = [f ਲਈ f ਵਿੱਚ os.listdir(test_dir) ਜੇ f.endswith(".csv")]
    ਲਈ csv_file ਵਿੱਚ csv_files:
        csv_ਰਸਤਾ = os.ਰਸਤਾ.join(test_dir, csv_file)
        ਛਾਪੋ(csv_ਰਸਤਾ)
        df = pd.read_csv(csv_ਰਸਤਾ, ਸਿਰਲੇਖ=ਕੁਝ_ਨਹੀਂ)
        n = df.shape[0]
        ਲਈ idx ਵਿੱਚ ਸੀਮਾ(n):
            example = {
                "question": df.iloc[idx, 0],
                "endings": [df.iloc[idx, 1], df.iloc[idx, 2], df.iloc[idx, 3], df.iloc[idx, 4]],
                "label": df.iloc[idx, 5],
            }
            ਦਿਓ example

ਪਰਿਭਾਸ਼ਾ render_example(example):
    """
    Given the example as a dictionary, render it as three torch tensors:
    - tokens (the tokens of context + completion, of size 4xN, as there are always 4 candidates)
    - mask (is 1 in the region of the candidate completion, where we evaluate likelihoods)
    - label (the index of the correct completion, which we hope has the highest likelihood)
    """
    ctx = f"Question: {example['question']}\n\nAnswer:"
    ctx_ਟੋਕਨ = enc.encode(ctx)

    tok_ਪੰਕਤੀਆਂ = []
    ਨਕਾਬ_ਪੰਕਤੀਆਂ = []
    ਲਈ ਸਿਰਾ ਵਿੱਚ example["endings"]:
        ਸਿਰਾ_ਟੋਕਨ = enc.encode(" " + str(ਸਿਰਾ)) # note: prepending " " because GPT-2 tokenizer
        tok_ਪੰਕਤੀਆਂ.append(ctx_ਟੋਕਨ + ਸਿਰਾ_ਟੋਕਨ)
        ਨਕਾਬ_ਪੰਕਤੀਆਂ.append([0]*ਲੰਬਾਈ(ctx_ਟੋਕਨ) + [1]*ਲੰਬਾਈ(ਸਿਰਾ_ਟੋਕਨ))

    # have to be careful during the collation because the number of tokens in each row can differ
    max_len = max(ਲੰਬਾਈ(ਪੰਕਤੀ) ਲਈ ਪੰਕਤੀ ਵਿੱਚ tok_ਪੰਕਤੀਆਂ)
    ਟੋਕਨ = torch.zeros((4, max_len), dtype=torch.ਲੰਮਾ)
    ਨਕਾਬ = torch.zeros((4, max_len), dtype=torch.ਲੰਮਾ)
    ਲਈ i, (tok_ਪੰਕਤੀ, ਨਕਾਬ_ਪੰਕਤੀ) ਵਿੱਚ enumerate(zip(tok_ਪੰਕਤੀਆਂ, ਨਕਾਬ_ਪੰਕਤੀਆਂ)):
        ਟੋਕਨ[i, :ਲੰਬਾਈ(tok_ਪੰਕਤੀ)] = torch.tensor(tok_ਪੰਕਤੀ)
        ਨਕਾਬ[i, :ਲੰਬਾਈ(ਨਕਾਬ_ਪੰਕਤੀ)] = torch.tensor(ਨਕਾਬ_ਪੰਕਤੀ)

    label = "ABCD".ਸੂਚਕਾਂਕ(example["label"])
    ਵਾਪਸ ਟੋਕਨ, ਨਕਾਬ, label

@torch.no_grad()
ਪਰਿਭਾਸ਼ਾ evaluate(model_type, device):

    torch.ਸੈੱਟ_float32_matmul_precision('high') # use tf32

    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.ਮੰਜ਼ਿਲ(device)
    # model = torch.compile(model)

    ਅੰਕ_correct_norm = 0
    ਅੰਕ_correct = 0
    ਅੰਕ_ਕੁੱਲ = 0
    ਲਈ example ਵਿੱਚ iterate_examples():
        ਟੋਕਨ, ਨਕਾਬ, label = render_example(example)
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
        ਛਾਪੋ(f"{num_total} acc: {num_correct/num_total:.4f} acc_norm: {num_correct_norm/num_total:.4f}")

        # debug prints
        ਜੇ ਅੰਕ_ਕੁੱਲ < 10:
            ਛਾਪੋ("---")
            ਛਾਪੋ(f"Context:\n {example['question']}")
            ਛਾਪੋ(f"Endings:")
            ਲਈ i, ਸਿਰਾ ਵਿੱਚ enumerate(example["endings"]):
                ਛਾਪੋ(f"{i} (loss: {avg_loss[i].item():.4f}) {end}")
            ਛਾਪੋ(f"predicted: {pred}, actual: {label}")

ਜੇ __name__ == "__main__":
    ਆਯਾਤ argparse
    parser = argparse.ਦਲੀਲ_Parser()
    parser.ਜੋੜੋ_ਦਲੀਲ("-m", "--model_type", type=str, ਮੂਲ="gpt2", help="the model type to use")
    parser.ਜੋੜੋ_ਦਲੀਲ("-d", "--device", type=str, ਮੂਲ="cuda", help="the device to use")
    ਦਲੀਲਾਂ = parser.parse_ਦਲੀਲਾਂ()
    evaluate(ਦਲੀਲਾਂ.model_type, ਦਲੀਲਾਂ.device)
