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

من معلومات_common استورد download_file, write_datafile

# -----------------------------------------------------------------------------
معلومات_CACHE_DIR = os.لار.join(os.لار.dirname(__file__), "tinystories")

تعریف download():
    """Downloads the TinyStories dataset to DATA_CACHE_DIR"""
    os.makedirs(معلومات_CACHE_DIR, exist_ok=ریښتیا)

    # download the TinyStories dataset, unless it's already downloaded
    معلومات_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
    معلومات_filename = os.لار.join(معلومات_CACHE_DIR, "TinyStories_all_data.tar.gz")
    إذا ليس os.لار.exists(معلومات_filename):
        چاپ(f"Downloading {data_url} to {data_filename}...")
        download_file(معلومات_url, معلومات_filename)
    وإلا:
        چاپ(f"{data_filename} already exists, skipping download...")

    # unpack the tar.gz file into all the data shards (json files)
    معلومات_dir = os.لار.join(معلومات_CACHE_DIR, "TinyStories_all_data")
    إذا ليس os.لار.exists(معلومات_dir):
        os.makedirs(معلومات_dir, exist_ok=ریښتیا)
        چاپ(f"Unpacking {data_filename}...")
        os.system(f"tar -xzf {data_filename} -C {data_dir}")
    وإلا:
        چاپ(f"{data_dir} already exists, skipping unpacking...")

    # print a single example just for debugging and such
    shard_filenames = sorted(glob.glob(os.لار.join(معلومات_dir, "*.json")))
    چاپ("Download done.")
    چاپ(f"Number of shards: {len(shard_filenames)}")
    # with open(shard_filenames[0], "r") as f:
    #     data = json.load(f)
    # print(f"Example story:\n{data[0]}")

تعریف process_shard(shard_شاخص, shard_filename, model_desc):
    إذا model_desc == "gpt-2":
        enc = tiktoken.get_encoding("gpt2")
        encode = مجهول s: enc.encode_ordinary(s)
        eot = enc.special_نښې['<|endoftext|>'] # end of text token
    وإلاإذا model_desc == "llama-3":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
        encode = مجهول s: tokenizer.encode(s, جمع_کړه_special_نښې=خطأ, verbose=خطأ, وېشل_special_نښې=ریښتیا)
        eot = tokenizer.encode('')[0] # by default the tokenizer adds the EOT token (128000)
    وإلا:
        ارفع ارزښت_Error(f"unknown model descriptor {model_desc}")

    سره open(shard_filename, "r") كـ f:
        معلومات = json.بارول(f)
    rng = random.Random(1337 + shard_شاخص)
    rng.shuffle(معلومات)
    all_نښې = []
    لكل example في معلومات:
        متن = example["story"]
        متن = متن.strip()  # get rid of leading/trailing whitespace
        نښې = encode(متن)
        all_نښې.append(eot)
        all_نښې.extend(نښې)
    أرجع all_نښې

تعریف tokenize(model_desc):
    # shard 0 will be the val split, rest is train
    معلومات_dir = os.لار.join(معلومات_CACHE_DIR, "TinyStories_all_data")
    shard_filenames = sorted(glob.glob(os.لار.join(معلومات_dir, "*.json")))
    val_shards = [shard_filenames[0]]
    train_shards = shard_filenames[1:]
    لكل وېشل_name, وېشل_shards في [("val", val_shards), ("train", train_shards)]:

        چاپ(f"Tokenizing {split_name} split...")
        all_نښې = []
        سره ProcessPoolExecutor() كـ executor:
            futures = [executor.submit(process_shard, shard_شاخص, shard_filename, model_desc)
                       لكل shard_شاخص, shard_filename في enumerate(وېشل_shards)]
            لكل future في as_completed(futures):
                all_نښې.extend(future.پایله())

        وېشل_filename = os.لار.join(معلومات_CACHE_DIR, f"TinyStories_{split_name}.bin")
        write_datafile(وېشل_filename, all_نښې, model_desc)

إذا __name__ == "__main__":
    parser = argparse.دلیل_Parser(description="Tiny Stories dataset preprocessing")
    parser.جمع_کړه_دلیل("-m", "--model_desc", type=str, افتراضي="gpt-2", choices=["gpt-2", "llama-3"], help="Model type, gpt-2|llama-3")
    دلیلونه = parser.parse_دلیلونه()
    download()
    tokenize(دلیلونه.model_desc)