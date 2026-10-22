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

దిగుమతి argparse
దిగుమతి os
దిగుమతి glob
దిగుమతి json
దిగుమతి random
తః concurrent.futures దిగుమతి ProcessPoolExecutor, as_completed

దిగుమతి tiktoken
తః transformers దిగుమతి AutoTokenizer

తః దత్తాంశం_common దిగుమతి download_file, write_datafile

# -----------------------------------------------------------------------------
దత్తాంశం_CACHE_DIR = os.మార్గం.join(os.మార్గం.dirname(__file__), "tinystories")

నిర్వచనం download():
    """Downloads the TinyStories dataset to DATA_CACHE_DIR"""
    os.makedirs(దత్తాంశం_CACHE_DIR, exist_ok=నిజం)

    # download the TinyStories dataset, unless it's already downloaded
    దత్తాంశం_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
    దత్తాంశం_filename = os.మార్గం.join(దత్తాంశం_CACHE_DIR, "TinyStories_all_data.tar.gz")
    ఒకవేళ కాదు os.మార్గం.exists(దత్తాంశం_filename):
        ముద్రించు(f"Downloading {data_url} to {data_filename}...")
        download_file(దత్తాంశం_url, దత్తాంశం_filename)
    లేకపోతే:
        ముద్రించు(f"{data_filename} already exists, skipping download...")

    # unpack the tar.gz file into all the data shards (json files)
    దత్తాంశం_dir = os.మార్గం.join(దత్తాంశం_CACHE_DIR, "TinyStories_all_data")
    ఒకవేళ కాదు os.మార్గం.exists(దత్తాంశం_dir):
        os.makedirs(దత్తాంశం_dir, exist_ok=నిజం)
        ముద్రించు(f"Unpacking {data_filename}...")
        os.system(f"tar -xzf {data_filename} -C {data_dir}")
    లేకపోతే:
        ముద్రించు(f"{data_dir} already exists, skipping unpacking...")

    # print a single example just for debugging and such
    shard_filenames = sorted(glob.glob(os.మార్గం.join(దత్తాంశం_dir, "*.json")))
    ముద్రించు("Download done.")
    ముద్రించు(f"Number of shards: {len(shard_filenames)}")
    # with open(shard_filenames[0], "r") as f:
    #     data = json.load(f)
    # print(f"Example story:\n{data[0]}")

నిర్వచనం process_shard(shard_సూచీ, shard_filename, model_desc):
    ఒకవేళ model_desc == "gpt-2":
        enc = tiktoken.get_encoding("gpt2")
        encode = అజ్ఞాత s: enc.encode_ordinary(s)
        eot = enc.special_టోకెన్లు['<|endoftext|>'] # end of text token
    లేదావేళ model_desc == "llama-3":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
        encode = అజ్ఞాత s: tokenizer.encode(s, కూడు_special_టోకెన్లు=అబద్ధం, verbose=అబద్ధం, విభజించు_special_టోకెన్లు=నిజం)
        eot = tokenizer.encode('')[0] # by default the tokenizer adds the EOT token (128000)
    లేకపోతే:
        లేపు విలువ_Error(f"unknown model descriptor {model_desc}")

    తోడుగా open(shard_filename, "r") ఇవ f:
        దత్తాంశం = json.ఎక్కించు(f)
    rng = random.Random(1337 + shard_సూచీ)
    rng.shuffle(దత్తాంశం)
    all_టోకెన్లు = []
    కొరకు example లో దత్తాంశం:
        పాఠ్యం = example["story"]
        పాఠ్యం = పాఠ్యం.strip()  # get rid of leading/trailing whitespace
        టోకెన్లు = encode(పాఠ్యం)
        all_టోకెన్లు.append(eot)
        all_టోకెన్లు.extend(టోకెన్లు)
    తిరిగి all_టోకెన్లు

నిర్వచనం tokenize(model_desc):
    # shard 0 will be the val split, rest is train
    దత్తాంశం_dir = os.మార్గం.join(దత్తాంశం_CACHE_DIR, "TinyStories_all_data")
    shard_filenames = sorted(glob.glob(os.మార్గం.join(దత్తాంశం_dir, "*.json")))
    val_shards = [shard_filenames[0]]
    train_shards = shard_filenames[1:]
    కొరకు విభజించు_name, విభజించు_shards లో [("val", val_shards), ("train", train_shards)]:

        ముద్రించు(f"Tokenizing {split_name} split...")
        all_టోకెన్లు = []
        తోడుగా ProcessPoolExecutor() ఇవ executor:
            futures = [executor.submit(process_shard, shard_సూచీ, shard_filename, model_desc)
                       కొరకు shard_సూచీ, shard_filename లో enumerate(విభజించు_shards)]
            కొరకు future లో as_completed(futures):
                all_టోకెన్లు.extend(future.ఫలితం())

        విభజించు_filename = os.మార్గం.join(దత్తాంశం_CACHE_DIR, f"TinyStories_{split_name}.bin")
        write_datafile(విభజించు_filename, all_టోకెన్లు, model_desc)

ఒకవేళ __name__ == "__main__":
    parser = argparse.వాదం_Parser(description="Tiny Stories dataset preprocessing")
    parser.కూడు_వాదం("-m", "--model_desc", type=str, అప్రమేయ="gpt-2", choices=["gpt-2", "llama-3"], help="Model type, gpt-2|llama-3")
    వాదనలు = parser.parse_వాదనలు()
    download()
    tokenize(వాదనలు.model_desc)