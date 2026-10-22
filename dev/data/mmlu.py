"""
Downloads and evaluates MMLU in Python.
This then acts as the reference file for llm.c
https://github.com/hendrycks/test

gpt2 (124M)
- this script: 14042 acc: 0.2557 acc_norm: 0.2721

gpt2-xl (1558M)
- this script: 14042 acc: 0.2927 acc_norm: 0.3035
"""

ಆಮದು os
ಆಮದು ವಿನಂತಿಗಳು
ಆಮದು tiktoken
ಆಮದು pandas ಇವ pd
ತಃ tqdm ಆಮದು tqdm
ಆಮದು torch
ಆಮದು torch.nn ಇವ nn
ತಃ torch.nn ಆಮದು functional ಇವ F
ತಃ transformers ಆಮದು GPT2LMHeadModel
ತಃ ದತ್ತಾಂಶ_common ಆಮದು download_file

# -----------------------------------------------------------------------------
ದತ್ತಾಂಶ_CACHE_DIR = os.ಮಾರ್ಗ.join(os.ಮಾರ್ಗ.dirname(__file__), "mmlu")

enc = tiktoken.get_encoding("gpt2")
ದತ್ತಾಂಶ_url = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"

ವ್ಯಾಖ್ಯಾನ download():
    """Downloads MMLU to DATA_CACHE_DIR"""
    os.makedirs(ದತ್ತಾಂಶ_CACHE_DIR, exist_ok=ನಿಜ)
    ದತ್ತಾಂಶ_filename = os.ಮಾರ್ಗ.join(ದತ್ತಾಂಶ_CACHE_DIR, f"data.tar")
    ಆದರೆ ಅಲ್ಲ os.ಮಾರ್ಗ.exists(ದತ್ತಾಂಶ_filename):
        ಮುದ್ರಿಸು(f"Downloading {data_url} to {data_filename}...")
        download_file(ದತ್ತಾಂಶ_url, ದತ್ತಾಂಶ_filename)
        os.system(f"tar -xf {data_filename} -C {DATA_CACHE_DIR}") # untar
        # creates a directory "data" inside it, with e.g. data/test/*csv
    ಇಲ್ಲದಿದ್ದರೆ:
        ಮುದ್ರಿಸು(f"{data_filename} already exists, skipping download...")

ವ್ಯಾಖ್ಯಾನ iterate_examples():
    # there are 14,042 examples in total in the test set

    download()
    test_dir = os.ಮಾರ್ಗ.join(ದತ್ತಾಂಶ_CACHE_DIR, "data", "test")
    csv_files = [f ಗಾಗಿ f ಒಳಗೆ os.listdir(test_dir) ಆದರೆ f.endswith(".csv")]
    ಗಾಗಿ csv_file ಒಳಗೆ csv_files:
        csv_ಮಾರ್ಗ = os.ಮಾರ್ಗ.join(test_dir, csv_file)
        ಮುದ್ರಿಸು(csv_ಮಾರ್ಗ)
        df = pd.read_csv(csv_ಮಾರ್ಗ, ಶೀರ್ಷಿಕೆ=ಏನೂಇಲ್ಲ)
        n = df.shape[0]
        ಗಾಗಿ idx ಒಳಗೆ ಶ್ರೇಣಿ(n):
            example = {
                "question": df.iloc[idx, 0],
                "endings": [df.iloc[idx, 1], df.iloc[idx, 2], df.iloc[idx, 3], df.iloc[idx, 4]],
                "label": df.iloc[idx, 5],
            }
            ನೀಡು example

ವ್ಯಾಖ್ಯಾನ render_example(example):
    """
    Given the example as a dictionary, render it as three torch tensors:
    - tokens (the tokens of context + completion, of size 4xN, as there are always 4 candidates)
    - mask (is 1 in the region of the candidate completion, where we evaluate likelihoods)
    - label (the index of the correct completion, which we hope has the highest likelihood)
    """
    ctx = f"Question: {example['question']}\n\nAnswer:"
    ctx_ಟೋಕನುಗಳು = enc.encode(ctx)

    tok_ಅಡ್ಡಸಾಲುಗಳು = []
    ಮುಖವಾಡ_ಅಡ್ಡಸಾಲುಗಳು = []
    ಗಾಗಿ ಕೊನೆ ಒಳಗೆ example["endings"]:
        ಕೊನೆ_ಟೋಕನುಗಳು = enc.encode(" " + str(ಕೊನೆ)) # note: prepending " " because GPT-2 tokenizer
        tok_ಅಡ್ಡಸಾಲುಗಳು.append(ctx_ಟೋಕನುಗಳು + ಕೊನೆ_ಟೋಕನುಗಳು)
        ಮುಖವಾಡ_ಅಡ್ಡಸಾಲುಗಳು.append([0]*ಉದ್ದ(ctx_ಟೋಕನುಗಳು) + [1]*ಉದ್ದ(ಕೊನೆ_ಟೋಕನುಗಳು))

    # have to be careful during the collation because the number of tokens in each row can differ
    max_len = max(ಉದ್ದ(ಅಡ್ಡಸಾಲು) ಗಾಗಿ ಅಡ್ಡಸಾಲು ಒಳಗೆ tok_ಅಡ್ಡಸಾಲುಗಳು)
    ಟೋಕನುಗಳು = torch.zeros((4, max_len), dtype=torch.ದೀರ್ಘ)
    ಮುಖವಾಡ = torch.zeros((4, max_len), dtype=torch.ದೀರ್ಘ)
    ಗಾಗಿ i, (tok_ಅಡ್ಡಸಾಲು, ಮುಖವಾಡ_ಅಡ್ಡಸಾಲು) ಒಳಗೆ enumerate(zip(tok_ಅಡ್ಡಸಾಲುಗಳು, ಮುಖವಾಡ_ಅಡ್ಡಸಾಲುಗಳು)):
        ಟೋಕನುಗಳು[i, :ಉದ್ದ(tok_ಅಡ್ಡಸಾಲು)] = torch.tensor(tok_ಅಡ್ಡಸಾಲು)
        ಮುಖವಾಡ[i, :ಉದ್ದ(ಮುಖವಾಡ_ಅಡ್ಡಸಾಲು)] = torch.tensor(ಮುಖವಾಡ_ಅಡ್ಡಸಾಲು)

    label = "ABCD".ಸೂಚ್ಯಂಕ(example["label"])
    ಹಿಂತಿರುಗಿ ಟೋಕನುಗಳು, ಮುಖವಾಡ, label

@torch.no_grad()
ವ್ಯಾಖ್ಯಾನ evaluate(model_type, device):

    torch.ಗಣ_float32_matmul_precision('high') # use tf32

    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.ಗಮ್ಯ(device)
    # model = torch.compile(model)

    ಅಂಕೆ_correct_norm = 0
    ಅಂಕೆ_correct = 0
    ಅಂಕೆ_ಒಟ್ಟು = 0
    ಗಾಗಿ example ಒಳಗೆ iterate_examples():
        ಟೋಕನುಗಳು, ಮುಖವಾಡ, label = render_example(example)
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
        ಮುದ್ರಿಸು(f"{num_total} acc: {num_correct/num_total:.4f} acc_norm: {num_correct_norm/num_total:.4f}")

        # debug prints
        ಆದರೆ ಅಂಕೆ_ಒಟ್ಟು < 10:
            ಮುದ್ರಿಸು("---")
            ಮುದ್ರಿಸು(f"Context:\n {example['question']}")
            ಮುದ್ರಿಸು(f"Endings:")
            ಗಾಗಿ i, ಕೊನೆ ಒಳಗೆ enumerate(example["endings"]):
                ಮುದ್ರಿಸು(f"{i} (loss: {avg_loss[i].item():.4f}) {end}")
            ಮುದ್ರಿಸು(f"predicted: {pred}, actual: {label}")

ಆದರೆ __name__ == "__main__":
    ಆಮದು argparse
    parser = argparse.ವಾದ_Parser()
    parser.ಕೂಡಿಸು_ವಾದ("-m", "--model_type", type=str, ಪೂರ್ವನಿಯೋಜಿತ="gpt2", help="the model type to use")
    parser.ಕೂಡಿಸು_ವಾದ("-d", "--device", type=str, ಪೂರ್ವನಿಯೋಜಿತ="cuda", help="the device to use")
    ವಾದಗಳು = parser.parse_ವಾದಗಳು()
    evaluate(ವಾದಗಳು.model_type, ವಾದಗಳು.device)
