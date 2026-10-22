"""
Downloads and tokenizes the TinyShakespeare dataset.
- The download is from Github.
- The tokenization is GPT-2 tokenizer with tiktoken

The output is written to a newly created tinyshakespeare/ folder.
The script prints:

For GPT-2:
$ python dev/data/tinyshakespeare.py --model=gpt-2
writing 32,768 tokens to /home/ubuntu/llm.c/dev/data/tinyshakespeare/tiny_shakespeare_val.bin (66,560 bytes) in the gpt-2 format
writing 305,260 tokens to /home/ubuntu/llm.c/dev/data/tinyshakespeare/tiny_shakespeare_train.bin (611,544 bytes) in the gpt-2 format

For LLaMA 3:
$ python dev/data/tinyshakespeare.py --model=llama-3
writing 32,768 tokens to /home/ubuntu/llm.c/dev/data/tinyshakespeare/tiny_shakespeare_val.bin (132,096 bytes) in the llama-3 format
writing 276,224 tokens to /home/ubuntu/llm.c/dev/data/tinyshakespeare/tiny_shakespeare_train.bin (1,105,920 bytes) in the llama-3 format

And runs in a few seconds depending on your internet
connection and computer. The .bin files are raw byte
streams of uint16 (gpt-2) or uint32 (llama) numbers indicating the token ids.
"""

আমদানি argparse
আমদানি os

আমদানি tiktoken
তঃ transformers আমদানি AutoTokenizer

তঃ তথ্য_common আমদানি download_file, write_datafile

# -----------------------------------------------------------------------------
তথ্য_CACHE_DIR = os.path.join(os.path.dirname(__file__), "tinyshakespeare")

def download():
    """Downloads the TinyShakespeare dataset to DATA_CACHE_DIR"""
    os.makedirs(তথ্য_CACHE_DIR, exist_ok=সত্য)
    # download the TinyShakespeare dataset, unless it's already downloaded
    তথ্য_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    তথ্য_filename = os.path.join(তথ্য_CACHE_DIR, "tiny_shakespeare.txt")
    যদি নয় os.path.exists(তথ্য_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(তথ্য_url, তথ্য_filename)
    অন্যথা:
        print(f"{data_filename} already exists, skipping download...")

def tokenize(model_desc):
    যদি model_desc == "gpt-2":
        enc = tiktoken.get_encoding("gpt2")
        encode = অনামা s: enc.encode_ordinary(s)
        eot = enc._special_tokens['<|endoftext|>'] # end of text token
    নতুবা_যদি model_desc == "llama-3":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
        encode = অনামা s: tokenizer.encode(s, add_special_tokens=মিথ্যা, verbose=মিথ্যা, split_special_tokens=সত্য)
        eot = tokenizer.encode('')[0] # by default the tokenizer adds the EOT token (128000)
    অন্যথা:
        তোলো মান_Error(f"unknown model descriptor {model_desc}")
    তথ্য_filename = os.path.join(তথ্য_CACHE_DIR, "tiny_shakespeare.txt")
    text = open(তথ্য_filename, 'r').পঠ()
    # let's treat every individual chunk of text as a separate "document"
    sections = text.split("\n\n")
    tokens = []
    জন্য i, s মধ্যে enumerate(sections):
        tokens.append(eot)
        # there was a mild bug where I originally intended to remove \n\n, but instead just added
        # the EOT right after each \n\n, so I'm keeping that behavior for backwards compatibility
        # therefore we have to here add an extra \n\n at the end of each section, except the last
        spad = s + "\n\n" যদি i != len(sections) - 1 অন্যথা s
        tokens.extend(encode(spad))
    # let's take the first 32,768 tokens as the validation split (~10%)
    val_tokens = tokens[:32768]
    train_tokens = tokens[32768:]
    # save to file
    val_filename = os.path.join(তথ্য_CACHE_DIR, "tiny_shakespeare_val.bin")
    train_filename = os.path.join(তথ্য_CACHE_DIR, "tiny_shakespeare_train.bin")
    write_datafile(val_filename, val_tokens, model_desc)
    write_datafile(train_filename, train_tokens, model_desc)

যদি __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tiny Shakespeare dataset preprocessing")
    parser.add_argument("-m", "--model_desc", type=str, পূর্বনির্ধারিত="gpt-2", choices=["gpt-2", "llama-3"], help="Model type, gpt-2|llama-3")
    args = parser.parse_args()
    download()
    tokenize(args.model_desc)
