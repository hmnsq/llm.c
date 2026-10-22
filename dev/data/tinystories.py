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

درآمد argparse
درآمد os
درآمد glob
درآمد json
درآمد random
سے concurrent.futures درآمد ProcessPoolExecutor, as_completed

درآمد tiktoken
سے transformers درآمد AutoTokenizer

سے ڈیٹا_common درآمد download_file, write_datafile

# -----------------------------------------------------------------------------
ڈیٹا_CACHE_DIR = os.راستہ.join(os.راستہ.dirname(__file__), "tinystories")

تعریف download():
    """Downloads the TinyStories dataset to DATA_CACHE_DIR"""
    os.makedirs(ڈیٹا_CACHE_DIR, exist_ok=سچ)

    # download the TinyStories dataset, unless it's already downloaded
    ڈیٹا_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
    ڈیٹا_filename = os.راستہ.join(ڈیٹا_CACHE_DIR, "TinyStories_all_data.tar.gz")
    اگر نہیں os.راستہ.exists(ڈیٹا_filename):
        چھاپو(f"Downloading {data_url} to {data_filename}...")
        download_file(ڈیٹا_url, ڈیٹا_filename)
    ورنہ:
        چھاپو(f"{data_filename} already exists, skipping download...")

    # unpack the tar.gz file into all the data shards (json files)
    ڈیٹا_dir = os.راستہ.join(ڈیٹا_CACHE_DIR, "TinyStories_all_data")
    اگر نہیں os.راستہ.exists(ڈیٹا_dir):
        os.makedirs(ڈیٹا_dir, exist_ok=سچ)
        چھاپو(f"Unpacking {data_filename}...")
        os.system(f"tar -xzf {data_filename} -C {data_dir}")
    ورنہ:
        چھاپو(f"{data_dir} already exists, skipping unpacking...")

    # print a single example just for debugging and such
    shard_filenames = sorted(glob.glob(os.راستہ.join(ڈیٹا_dir, "*.json")))
    چھاپو("Download done.")
    چھاپو(f"Number of shards: {len(shard_filenames)}")
    # with open(shard_filenames[0], "r") as f:
    #     data = json.load(f)
    # print(f"Example story:\n{data[0]}")

تعریف process_shard(shard_اشاریہ, shard_filename, model_desc):
    اگر model_desc == "gpt-2":
        enc = tiktoken.get_encoding("gpt2")
        encode = بےنام s: enc.encode_ordinary(s)
        eot = enc.special_علامات['<|endoftext|>'] # end of text token
    وگرنہ_اگر model_desc == "llama-3":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
        encode = بےنام s: tokenizer.encode(s, جوڑو_special_علامات=جھوٹ, verbose=جھوٹ, بانٹو_special_علامات=سچ)
        eot = tokenizer.encode('')[0] # by default the tokenizer adds the EOT token (128000)
    ورنہ:
        اٹھاؤ قدر_Error(f"unknown model descriptor {model_desc}")

    ساتھ open(shard_filename, "r") بطور f:
        ڈیٹا = json.بھرو(f)
    rng = random.Random(1337 + shard_اشاریہ)
    rng.shuffle(ڈیٹا)
    all_علامات = []
    برائے example میں ڈیٹا:
        متن = example["story"]
        متن = متن.strip()  # get rid of leading/trailing whitespace
        علامات = encode(متن)
        all_علامات.append(eot)
        all_علامات.extend(علامات)
    واپس all_علامات

تعریف tokenize(model_desc):
    # shard 0 will be the val split, rest is train
    ڈیٹا_dir = os.راستہ.join(ڈیٹا_CACHE_DIR, "TinyStories_all_data")
    shard_filenames = sorted(glob.glob(os.راستہ.join(ڈیٹا_dir, "*.json")))
    val_shards = [shard_filenames[0]]
    train_shards = shard_filenames[1:]
    برائے بانٹو_name, بانٹو_shards میں [("val", val_shards), ("train", train_shards)]:

        چھاپو(f"Tokenizing {split_name} split...")
        all_علامات = []
        ساتھ ProcessPoolExecutor() بطور executor:
            futures = [executor.submit(process_shard, shard_اشاریہ, shard_filename, model_desc)
                       برائے shard_اشاریہ, shard_filename میں enumerate(بانٹو_shards)]
            برائے future میں as_completed(futures):
                all_علامات.extend(future.نتیجہ())

        بانٹو_filename = os.راستہ.join(ڈیٹا_CACHE_DIR, f"TinyStories_{split_name}.bin")
        write_datafile(بانٹو_filename, all_علامات, model_desc)

اگر __name__ == "__main__":
    parser = argparse.دلیل_Parser(description="Tiny Stories dataset preprocessing")
    parser.جوڑو_دلیل("-m", "--model_desc", type=str, طےشدہ="gpt-2", choices=["gpt-2", "llama-3"], help="Model type, gpt-2|llama-3")
    دلائل = parser.parse_دلائل()
    download()
    tokenize(دلائل.model_desc)