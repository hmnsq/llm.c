"""
Downloads and evaluates MMLU in Python.
This then acts as the reference file for llm.c
https://github.com/hendrycks/test

gpt2 (124M)
- this script: 14042 acc: 0.2557 acc_norm: 0.2721

gpt2-xl (1558M)
- this script: 14042 acc: 0.2927 acc_norm: 0.3035
"""

আমদানি os
আমদানি requests
আমদানি tiktoken
আমদানি pandas ইব pd
তঃ tqdm আমদানি tqdm
আমদানি torch
আমদানি torch.nn ইব nn
তঃ torch.nn আমদানি functional ইব F
তঃ transformers আমদানি GPT2LMHeadModel
তঃ তথ্য_common আমদানি download_file

# -----------------------------------------------------------------------------
তথ্য_CACHE_DIR = os.path.join(os.path.dirname(__file__), "mmlu")

enc = tiktoken.get_encoding("gpt2")
তথ্য_url = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"

def download():
    """Downloads MMLU to DATA_CACHE_DIR"""
    os.makedirs(তথ্য_CACHE_DIR, exist_ok=সত্য)
    তথ্য_filename = os.path.join(তথ্য_CACHE_DIR, f"data.tar")
    যদি নয় os.path.exists(তথ্য_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(তথ্য_url, তথ্য_filename)
        os.system(f"tar -xf {data_filename} -C {DATA_CACHE_DIR}") # untar
        # creates a directory "data" inside it, with e.g. data/test/*csv
    অন্যথা:
        print(f"{data_filename} already exists, skipping download...")

def iterate_examples():
    # there are 14,042 examples in total in the test set

    download()
    test_dir = os.path.join(তথ্য_CACHE_DIR, "data", "test")
    csv_files = [f জন্য f মধ্যে os.listdir(test_dir) যদি f.endswith(".csv")]
    জন্য csv_file মধ্যে csv_files:
        csv_path = os.path.join(test_dir, csv_file)
        print(csv_path)
        df = pd.read_csv(csv_path, header=None)
        n = df.shape[0]
        জন্য idx মধ্যে range(n):
            example = {
                "question": df.iloc[idx, 0],
                "endings": [df.iloc[idx, 1], df.iloc[idx, 2], df.iloc[idx, 3], df.iloc[idx, 4]],
                "label": df.iloc[idx, 5],
            }
            দাও example

def render_example(example):
    """
    Given the example as a dictionary, render it as three torch tensors:
    - tokens (the tokens of context + completion, of size 4xN, as there are always 4 candidates)
    - mask (is 1 in the region of the candidate completion, where we evaluate likelihoods)
    - label (the index of the correct completion, which we hope has the highest likelihood)
    """
    ctx = f"Question: {example['question']}\n\nAnswer:"
    ctx_tokens = enc.encode(ctx)

    tok_rows = []
    mask_rows = []
    জন্য end মধ্যে example["endings"]:
        end_tokens = enc.encode(" " + str(end)) # note: prepending " " because GPT-2 tokenizer
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0]*len(ctx_tokens) + [1]*len(end_tokens))

    # have to be careful during the collation because the number of tokens in each row can differ
    max_len = max(len(row) জন্য row মধ্যে tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    জন্য i, (tok_row, mask_row) মধ্যে enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)

    label = "ABCD".সূচক(example["label"])
    ফেরত tokens, mask, label

@torch.no_grad()
def evaluate(model_type, device):

    torch.set_float32_matmul_precision('high') # use tf32

    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.to(device)
    # model = torch.compile(model)

    অঙ্ক_correct_norm = 0
    অঙ্ক_correct = 0
    অঙ্ক_যোগ = 0
    জন্য example মধ্যে iterate_examples():
        tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)

        # get the logits
        logits = model(tokens).logits
        # evaluate the autoregressive loss at all positions
        shift_logits = (logits[..., :-1, :]).contiguous()
        shift_tokens = (tokens[..., 1:]).contiguous()
        flat_shift_logits = shift_logits.view(-1, shift_logits.আকার_2(-1))
        flat_shift_tokens = shift_tokens.view(-1)
        shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
        shift_losses = shift_losses.view(tokens.আকার_2(0), -1)
        # now get the average loss just for the completion region (where mask == 1), in each row
        shift_mask = (mask[..., 1:]).contiguous() # we must shift mask, so we start at the last prompt token
        masked_shift_losses = shift_losses * shift_mask
        # sum and divide by the number of 1s in the mask
        যোগ_loss = masked_shift_losses.যোগ_2(dim=1)
        avg_loss = যোগ_loss / shift_mask.যোগ_2(dim=1)
        # now we have a loss for each of the 4 completions
        # the one with the lowest loss should be the most likely
        pred = যোগ_loss.argmin().item()
        pred_norm = avg_loss.argmin().item()

        # accumulate stats
        অঙ্ক_যোগ += 1
        অঙ্ক_correct += পূর্ণসংখ্যা(pred == label)
        অঙ্ক_correct_norm += পূর্ণসংখ্যা(pred_norm == label)
        print(f"{num_total} acc: {num_correct/num_total:.4f} acc_norm: {num_correct_norm/num_total:.4f}")

        # debug prints
        যদি অঙ্ক_যোগ < 10:
            print("---")
            print(f"Context:\n {example['question']}")
            print(f"Endings:")
            জন্য i, end মধ্যে enumerate(example["endings"]):
                print(f"{i} (loss: {avg_loss[i].item():.4f}) {end}")
            print(f"predicted: {pred}, actual: {label}")

যদি __name__ == "__main__":
    আমদানি argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_type", type=str, পূর্বনির্ধারিত="gpt2", help="the model type to use")
    parser.add_argument("-d", "--device", type=str, পূর্বনির্ধারিত="cuda", help="the device to use")
    args = parser.parse_args()
    evaluate(args.model_type, args.device)
