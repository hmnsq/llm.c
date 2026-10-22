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

استورد argparse
استورد os
استورد glob
استورد json
استورد random
من concurrent.futures استورد ProcessPoolExecutor, as_completed

استورد tiktoken
من transformers استورد AutoTokenizer

من انکڑے_common استورد download_file, write_datafile

# -----------------------------------------------------------------------------
انکڑے_CACHE_DIR = os.رستہ.join(os.رستہ.dirname(__file__), "tinystories")

تعریف download():
    """Downloads the TinyStories dataset to DATA_CACHE_DIR"""
    os.makedirs(انکڑے_CACHE_DIR, exist_ok=سچ)

    # download the TinyStories dataset, unless it's already downloaded
    انکڑے_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
    انکڑے_filename = os.رستہ.join(انکڑے_CACHE_DIR, "TinyStories_all_data.tar.gz")
    إذا ليس os.رستہ.exists(انکڑے_filename):
        چھاپو(f"Downloading {data_url} to {data_filename}...")
        download_file(انکڑے_url, انکڑے_filename)
    وإلا:
        چھاپو(f"{data_filename} already exists, skipping download...")

    # unpack the tar.gz file into all the data shards (json files)
    انکڑے_dir = os.رستہ.join(انکڑے_CACHE_DIR, "TinyStories_all_data")
    إذا ليس os.رستہ.exists(انکڑے_dir):
        os.makedirs(انکڑے_dir, exist_ok=سچ)
        چھاپو(f"Unpacking {data_filename}...")
        os.system(f"tar -xzf {data_filename} -C {data_dir}")
    وإلا:
        چھاپو(f"{data_dir} already exists, skipping unpacking...")

    # print a single example just for debugging and such
    shard_filenames = sorted(glob.glob(os.رستہ.join(انکڑے_dir, "*.json")))
    چھاپو("Download done.")
    چھاپو(f"Number of shards: {len(shard_filenames)}")
    # with open(shard_filenames[0], "r") as f:
    #     data = json.load(f)
    # print(f"Example story:\n{data[0]}")

تعریف process_shard(shard_اشاریہ, shard_filename, model_desc):
    إذا model_desc == "gpt-2":
        enc = tiktoken.get_encoding("gpt2")
        encode = مجهول s: enc.encode_ordinary(s)
        eot = enc.special_ٹوکن['<|endoftext|>'] # end of text token
    وإلاإذا model_desc == "llama-3":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
        encode = مجهول s: tokenizer.encode(s, جوڑو_special_ٹوکن=خطأ, verbose=خطأ, ونڈو_special_ٹوکن=سچ)
        eot = tokenizer.encode('')[0] # by default the tokenizer adds the EOT token (128000)
    وإلا:
        ارفع قدر_Error(f"unknown model descriptor {model_desc}")

    نال open(shard_filename, "r") كـ f:
        انکڑے = json.لدو(f)
    rng = random.Random(1337 + shard_اشاریہ)
    rng.shuffle(انکڑے)
    all_ٹوکن = []
    لكل example في انکڑے:
        متن = example["story"]
        متن = متن.strip()  # get rid of leading/trailing whitespace
        ٹوکن = encode(متن)
        all_ٹوکن.append(eot)
        all_ٹوکن.extend(ٹوکن)
    أرجع all_ٹوکن

تعریف tokenize(model_desc):
    # shard 0 will be the val split, rest is train
    انکڑے_dir = os.رستہ.join(انکڑے_CACHE_DIR, "TinyStories_all_data")
    shard_filenames = sorted(glob.glob(os.رستہ.join(انکڑے_dir, "*.json")))
    val_shards = [shard_filenames[0]]
    train_shards = shard_filenames[1:]
    لكل ونڈو_name, ونڈو_shards في [("val", val_shards), ("train", train_shards)]:

        چھاپو(f"Tokenizing {split_name} split...")
        all_ٹوکن = []
        نال ProcessPoolExecutor() كـ executor:
            futures = [executor.submit(process_shard, shard_اشاریہ, shard_filename, model_desc)
                       لكل shard_اشاریہ, shard_filename في enumerate(ونڈو_shards)]
            لكل future في as_completed(futures):
                all_ٹوکن.extend(future.نتیجہ())

        ونڈو_filename = os.رستہ.join(انکڑے_CACHE_DIR, f"TinyStories_{split_name}.bin")
        write_datafile(ونڈو_filename, all_ٹوکن, model_desc)

إذا __name__ == "__main__":
    parser = argparse.دلیل_Parser(description="Tiny Stories dataset preprocessing")
    parser.جوڑو_دلیل("-m", "--model_desc", type=str, افتراضي="gpt-2", choices=["gpt-2", "llama-3"], help="Model type, gpt-2|llama-3")
    دلیلاں = parser.parse_دلیلاں()
    download()
    tokenize(دلیلاں.model_desc)