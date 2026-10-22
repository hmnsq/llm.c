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

ਆਯਾਤ argparse
ਆਯਾਤ os
ਆਯਾਤ glob
ਆਯਾਤ json
ਆਯਾਤ random
ਤਃ concurrent.futures ਆਯਾਤ ProcessPoolExecutor, as_completed

ਆਯਾਤ tiktoken
ਤਃ transformers ਆਯਾਤ AutoTokenizer

ਤਃ ਅੰਕੜੇ_common ਆਯਾਤ download_file, write_datafile

# -----------------------------------------------------------------------------
ਅੰਕੜੇ_CACHE_DIR = os.ਰਸਤਾ.join(os.ਰਸਤਾ.dirname(__file__), "tinystories")

ਪਰਿਭਾਸ਼ਾ download():
    """Downloads the TinyStories dataset to DATA_CACHE_DIR"""
    os.makedirs(ਅੰਕੜੇ_CACHE_DIR, exist_ok=ਸੱਚ)

    # download the TinyStories dataset, unless it's already downloaded
    ਅੰਕੜੇ_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
    ਅੰਕੜੇ_filename = os.ਰਸਤਾ.join(ਅੰਕੜੇ_CACHE_DIR, "TinyStories_all_data.tar.gz")
    ਜੇ ਨਹੀਂ os.ਰਸਤਾ.exists(ਅੰਕੜੇ_filename):
        ਛਾਪੋ(f"Downloading {data_url} to {data_filename}...")
        download_file(ਅੰਕੜੇ_url, ਅੰਕੜੇ_filename)
    ਨਹੀਂਤਾਂ:
        ਛਾਪੋ(f"{data_filename} already exists, skipping download...")

    # unpack the tar.gz file into all the data shards (json files)
    ਅੰਕੜੇ_dir = os.ਰਸਤਾ.join(ਅੰਕੜੇ_CACHE_DIR, "TinyStories_all_data")
    ਜੇ ਨਹੀਂ os.ਰਸਤਾ.exists(ਅੰਕੜੇ_dir):
        os.makedirs(ਅੰਕੜੇ_dir, exist_ok=ਸੱਚ)
        ਛਾਪੋ(f"Unpacking {data_filename}...")
        os.system(f"tar -xzf {data_filename} -C {data_dir}")
    ਨਹੀਂਤਾਂ:
        ਛਾਪੋ(f"{data_dir} already exists, skipping unpacking...")

    # print a single example just for debugging and such
    shard_filenames = sorted(glob.glob(os.ਰਸਤਾ.join(ਅੰਕੜੇ_dir, "*.json")))
    ਛਾਪੋ("Download done.")
    ਛਾਪੋ(f"Number of shards: {len(shard_filenames)}")
    # with open(shard_filenames[0], "r") as f:
    #     data = json.load(f)
    # print(f"Example story:\n{data[0]}")

ਪਰਿਭਾਸ਼ਾ process_shard(shard_ਸੂਚਕਾਂਕ, shard_filename, model_desc):
    ਜੇ model_desc == "gpt-2":
        enc = tiktoken.get_encoding("gpt2")
        encode = ਅਨਾਮ s: enc.encode_ordinary(s)
        eot = enc.special_ਟੋਕਨ['<|endoftext|>'] # end of text token
    ਜਾਂਜੇ model_desc == "llama-3":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
        encode = ਅਨਾਮ s: tokenizer.encode(s, ਜੋੜੋ_special_ਟੋਕਨ=ਝੂਠ, verbose=ਝੂਠ, ਵੰਡੋ_special_ਟੋਕਨ=ਸੱਚ)
        eot = tokenizer.encode('')[0] # by default the tokenizer adds the EOT token (128000)
    ਨਹੀਂਤਾਂ:
        ਉਠਾਓ ਮੁੱਲ_Error(f"unknown model descriptor {model_desc}")

    ਨਾਲ open(shard_filename, "r") ਇਵ f:
        ਅੰਕੜੇ = json.ਲੱਦੋ(f)
    rng = random.Random(1337 + shard_ਸੂਚਕਾਂਕ)
    rng.shuffle(ਅੰਕੜੇ)
    all_ਟੋਕਨ = []
    ਲਈ example ਵਿੱਚ ਅੰਕੜੇ:
        ਪਾਠ = example["story"]
        ਪਾਠ = ਪਾਠ.strip()  # get rid of leading/trailing whitespace
        ਟੋਕਨ = encode(ਪਾਠ)
        all_ਟੋਕਨ.append(eot)
        all_ਟੋਕਨ.extend(ਟੋਕਨ)
    ਵਾਪਸ all_ਟੋਕਨ

ਪਰਿਭਾਸ਼ਾ tokenize(model_desc):
    # shard 0 will be the val split, rest is train
    ਅੰਕੜੇ_dir = os.ਰਸਤਾ.join(ਅੰਕੜੇ_CACHE_DIR, "TinyStories_all_data")
    shard_filenames = sorted(glob.glob(os.ਰਸਤਾ.join(ਅੰਕੜੇ_dir, "*.json")))
    val_shards = [shard_filenames[0]]
    train_shards = shard_filenames[1:]
    ਲਈ ਵੰਡੋ_name, ਵੰਡੋ_shards ਵਿੱਚ [("val", val_shards), ("train", train_shards)]:

        ਛਾਪੋ(f"Tokenizing {split_name} split...")
        all_ਟੋਕਨ = []
        ਨਾਲ ProcessPoolExecutor() ਇਵ executor:
            futures = [executor.submit(process_shard, shard_ਸੂਚਕਾਂਕ, shard_filename, model_desc)
                       ਲਈ shard_ਸੂਚਕਾਂਕ, shard_filename ਵਿੱਚ enumerate(ਵੰਡੋ_shards)]
            ਲਈ future ਵਿੱਚ as_completed(futures):
                all_ਟੋਕਨ.extend(future.ਨਤੀਜਾ())

        ਵੰਡੋ_filename = os.ਰਸਤਾ.join(ਅੰਕੜੇ_CACHE_DIR, f"TinyStories_{split_name}.bin")
        write_datafile(ਵੰਡੋ_filename, all_ਟੋਕਨ, model_desc)

ਜੇ __name__ == "__main__":
    parser = argparse.ਦਲੀਲ_Parser(description="Tiny Stories dataset preprocessing")
    parser.ਜੋੜੋ_ਦਲੀਲ("-m", "--model_desc", type=str, ਮੂਲ="gpt-2", choices=["gpt-2", "llama-3"], help="Model type, gpt-2|llama-3")
    ਦਲੀਲਾਂ = parser.parse_ਦਲੀਲਾਂ()
    download()
    tokenize(ਦਲੀਲਾਂ.model_desc)