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

இறக்குமதி argparse
இறக்குமதி os
இறக்குமதி glob
இறக்குமதி json
இறக்குமதி random
தஃ concurrent.futures இறக்குமதி ProcessPoolExecutor, as_completed

இறக்குமதி tiktoken
தஃ transformers இறக்குமதி AutoTokenizer

தஃ data_common இறக்குமதி download_file, write_datafile

# -----------------------------------------------------------------------------
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "tinystories")

def download():
    """Downloads the TinyStories dataset to DATA_CACHE_DIR"""
    os.makedirs(DATA_CACHE_DIR, exist_ok=உண்மை)

    # download the TinyStories dataset, unless it's already downloaded
    data_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
    data_filename = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data.tar.gz")
    எனில் இல்லை os.path.exists(data_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)
    இல்லையெனில்:
        print(f"{data_filename} already exists, skipping download...")

    # unpack the tar.gz file into all the data shards (json files)
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    எனில் இல்லை os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=உண்மை)
        print(f"Unpacking {data_filename}...")
        os.system(f"tar -xzf {data_filename} -C {data_dir}")
    இல்லையெனில்:
        print(f"{data_dir} already exists, skipping unpacking...")

    # print a single example just for debugging and such
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    print("Download done.")
    print(f"Number of shards: {len(shard_filenames)}")
    # with open(shard_filenames[0], "r") as f:
    #     data = json.load(f)
    # print(f"Example story:\n{data[0]}")

def process_shard(shard_index, shard_filename, model_desc):
    எனில் model_desc == "gpt-2":
        enc = tiktoken.get_encoding("gpt2")
        encode = பெயரிலி s: enc.encode_ordinary(s)
        eot = enc._special_tokens['<|endoftext|>'] # end of text token
    அல்லதெனில் model_desc == "llama-3":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
        encode = பெயரிலி s: tokenizer.encode(s, add_special_tokens=பொய், verbose=பொய், split_special_tokens=உண்மை)
        eot = tokenizer.encode('')[0] # by default the tokenizer adds the EOT token (128000)
    இல்லையெனில்:
        எழுப்பு மதிப்பு_Error(f"unknown model descriptor {model_desc}")

    with open(shard_filename, "r") இவ f:
        data = json.load(f)
    rng = random.Random(1337 + shard_index)
    rng.shuffle(data)
    all_tokens = []
    க்கு example உள் data:
        text = example["story"]
        text = text.strip()  # get rid of leading/trailing whitespace
        tokens = encode(text)
        all_tokens.append(eot)
        all_tokens.extend(tokens)
    திரும்பு all_tokens

def tokenize(model_desc):
    # shard 0 will be the val split, rest is train
    data_dir = os.path.join(DATA_CACHE_DIR, "TinyStories_all_data")
    shard_filenames = sorted(glob.glob(os.path.join(data_dir, "*.json")))
    val_shards = [shard_filenames[0]]
    train_shards = shard_filenames[1:]
    க்கு split_name, split_shards உள் [("val", val_shards), ("train", train_shards)]:

        print(f"Tokenizing {split_name} split...")
        all_tokens = []
        with ProcessPoolExecutor() இவ executor:
            futures = [executor.submit(process_shard, shard_index, shard_filename, model_desc)
                       க்கு shard_index, shard_filename உள் enumerate(split_shards)]
            க்கு future உள் as_completed(futures):
                all_tokens.extend(future.result())

        split_filename = os.path.join(DATA_CACHE_DIR, f"TinyStories_{split_name}.bin")
        write_datafile(split_filename, all_tokens, model_desc)

எனில் __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tiny Stories dataset preprocessing")
    parser.add_argument("-m", "--model_desc", type=str, இயல்பு="gpt-2", choices=["gpt-2", "llama-3"], help="Model type, gpt-2|llama-3")
    args = parser.parse_args()
    download()
    tokenize(args.model_desc)