"""
Downloads and tokenizes the TinyStories dataset.
- The download is from HuggingFace datasets.
- The tokenization is using either GPT-2 or LLaMA 3 tokenizer.

The output is written to a newly created tinystories/ folder.
The script prints:

For GPT-2:
Number of shards: 50
Tokenizing val split...
writing 19,043,638 tokens to tinystories/TinyStories_val.bin
Tokenizing train split...
writing 925,653,391 tokens to tinystories/TinyStories_train.bin

For LLaMA 3:
Number of shards: 50
Tokenizing val split...
writing 18,660,516 tokens to tinystories/TinyStories_val.bin
Tokenizing train split...
writing 907,021,844 tokens to tinystories/TinyStories_train.bin

And runs in few minutes two depending on your internet
connection and computer. The .bin files are raw byte
streams of uint16 (gpt-2) or uint32 (llama) numbers indicating the token ids.
"""

আমদানি argparse
আমদানি os
আমদানি glob
আমদানি json
আমদানি random
তঃ concurrent.futures আমদানি ProcessPoolExecutor, as_completed

আমদানি tiktoken
তঃ transformers আমদানি AutoTokenizer

তঃ তথ্য_common আমদানি download_file, write_datafile

# -----------------------------------------------------------------------------
তথ্য_CACHE_DIR = os.path.join(os.path.dirname(__file__), "tinystories")

def download():
    """Downloads the TinyStories dataset to DATA_CACHE_DIR"""
    os.makedirs(তথ্য_CACHE_DIR, exist_ok=সত্য)

    # download the TinyStories dataset, unless it's already downloaded
    তথ্য_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
    তথ্য_filename = os.path.join(তথ্য_CACHE_DIR, "TinyStories_all_data.tar.gz")
    যদি নয় os.path.exists(তথ্য_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(তথ্য_url, তথ্য_filename)
    অন্যথা:
        print(f"{data_filename} already exists, skipping download...")

    # unpack the tar.gz file into all the data shards (json files)
    তথ্য_dir = os.path.join(তথ্য_CACHE_DIR, "TinyStories_all_data")
    যদি নয় os.path.exists(তথ্য_dir):
        os.makedirs(তথ্য_dir, exist_ok=সত্য)
        print(f"Unpacking {data_filename}...")
        os.system(f"tar -xzf {data_filename} -C {data_dir}")
    অন্যথা:
        print(f"{data_dir} already exists, skipping unpacking...")

    # print a single example just for debugging and such
    shard_filenames = sorted(glob.glob(os.path.join(তথ্য_dir, "*.json")))
    print("Download done.")
    print(f"Number of shards: {len(shard_filenames)}")
    # with open(shard_filenames[0], "r") as f:
    #     data = json.load(f)
    # print(f"Example story:\n{data[0]}")

def process_shard(shard_সূচক, shard_filename, model_desc):
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

    with open(shard_filename, "r") ইব f:
        তথ্য = json.load(f)
    rng = random.Random(1337 + shard_সূচক)
    rng.shuffle(তথ্য)
    all_tokens = []
    জন্য example মধ্যে তথ্য:
        text = example["story"]
        text = text.strip()  # get rid of leading/trailing whitespace
        tokens = encode(text)
        all_tokens.append(eot)
        all_tokens.extend(tokens)
    ফেরত all_tokens

def tokenize(model_desc):
    # shard 0 will be the val split, rest is train
    তথ্য_dir = os.path.join(তথ্য_CACHE_DIR, "TinyStories_all_data")
    shard_filenames = sorted(glob.glob(os.path.join(তথ্য_dir, "*.json")))
    val_shards = [shard_filenames[0]]
    train_shards = shard_filenames[1:]
    জন্য split_name, split_shards মধ্যে [("val", val_shards), ("train", train_shards)]:

        print(f"Tokenizing {split_name} split...")
        all_tokens = []
        with ProcessPoolExecutor() ইব executor:
            futures = [executor.submit(process_shard, shard_সূচক, shard_filename, model_desc)
                       জন্য shard_সূচক, shard_filename মধ্যে enumerate(split_shards)]
            জন্য future মধ্যে as_completed(futures):
                all_tokens.extend(future.ফলাফল())

        split_filename = os.path.join(তথ্য_CACHE_DIR, f"TinyStories_{split_name}.bin")
        write_datafile(split_filename, all_tokens, model_desc)

যদি __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tiny Stories dataset preprocessing")
    parser.add_argument("-m", "--model_desc", type=str, পূর্বনির্ধারিত="gpt-2", choices=["gpt-2", "llama-3"], help="Model type, gpt-2|llama-3")
    args = parser.parse_args()
    download()
    tokenize(args.model_desc)