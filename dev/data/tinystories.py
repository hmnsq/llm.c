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

ಆಮದು argparse
ಆಮದು os
ಆಮದು glob
ಆಮದು json
ಆಮದು random
ತಃ concurrent.futures ಆಮದು ProcessPoolExecutor, as_completed

ಆಮದು tiktoken
ತಃ transformers ಆಮದು AutoTokenizer

ತಃ ದತ್ತಾಂಶ_common ಆಮದು download_file, write_datafile

# -----------------------------------------------------------------------------
ದತ್ತಾಂಶ_CACHE_DIR = os.ಮಾರ್ಗ.join(os.ಮಾರ್ಗ.dirname(__file__), "tinystories")

ವ್ಯಾಖ್ಯಾನ download():
    """Downloads the TinyStories dataset to DATA_CACHE_DIR"""
    os.makedirs(ದತ್ತಾಂಶ_CACHE_DIR, exist_ok=ನಿಜ)

    # download the TinyStories dataset, unless it's already downloaded
    ದತ್ತಾಂಶ_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
    ದತ್ತಾಂಶ_filename = os.ಮಾರ್ಗ.join(ದತ್ತಾಂಶ_CACHE_DIR, "TinyStories_all_data.tar.gz")
    ಆದರೆ ಅಲ್ಲ os.ಮಾರ್ಗ.exists(ದತ್ತಾಂಶ_filename):
        ಮುದ್ರಿಸು(f"Downloading {data_url} to {data_filename}...")
        download_file(ದತ್ತಾಂಶ_url, ದತ್ತಾಂಶ_filename)
    ಇಲ್ಲದಿದ್ದರೆ:
        ಮುದ್ರಿಸು(f"{data_filename} already exists, skipping download...")

    # unpack the tar.gz file into all the data shards (json files)
    ದತ್ತಾಂಶ_dir = os.ಮಾರ್ಗ.join(ದತ್ತಾಂಶ_CACHE_DIR, "TinyStories_all_data")
    ಆದರೆ ಅಲ್ಲ os.ಮಾರ್ಗ.exists(ದತ್ತಾಂಶ_dir):
        os.makedirs(ದತ್ತಾಂಶ_dir, exist_ok=ನಿಜ)
        ಮುದ್ರಿಸು(f"Unpacking {data_filename}...")
        os.system(f"tar -xzf {data_filename} -C {data_dir}")
    ಇಲ್ಲದಿದ್ದರೆ:
        ಮುದ್ರಿಸು(f"{data_dir} already exists, skipping unpacking...")

    # print a single example just for debugging and such
    shard_filenames = sorted(glob.glob(os.ಮಾರ್ಗ.join(ದತ್ತಾಂಶ_dir, "*.json")))
    ಮುದ್ರಿಸು("Download done.")
    ಮುದ್ರಿಸು(f"Number of shards: {len(shard_filenames)}")
    # with open(shard_filenames[0], "r") as f:
    #     data = json.load(f)
    # print(f"Example story:\n{data[0]}")

ವ್ಯಾಖ್ಯಾನ process_shard(shard_ಸೂಚ್ಯಂಕ, shard_filename, model_desc):
    ಆದರೆ model_desc == "gpt-2":
        enc = tiktoken.get_encoding("gpt2")
        encode = ಅನಾಮಿಕ s: enc.encode_ordinary(s)
        eot = enc.special_ಟೋಕನುಗಳು['<|endoftext|>'] # end of text token
    ಅಥವಾಆದರೆ model_desc == "llama-3":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
        encode = ಅನಾಮಿಕ s: tokenizer.encode(s, ಕೂಡಿಸು_special_ಟೋಕನುಗಳು=ಸುಳ್ಳು, verbose=ಸುಳ್ಳು, ವಿಭಜಿಸು_special_ಟೋಕನುಗಳು=ನಿಜ)
        eot = tokenizer.encode('')[0] # by default the tokenizer adds the EOT token (128000)
    ಇಲ್ಲದಿದ್ದರೆ:
        ಎತ್ತು ಮೌಲ್ಯ_Error(f"unknown model descriptor {model_desc}")

    ಜೊತೆ open(shard_filename, "r") ಇವ f:
        ದತ್ತಾಂಶ = json.ತುಂಬು(f)
    rng = random.Random(1337 + shard_ಸೂಚ್ಯಂಕ)
    rng.shuffle(ದತ್ತಾಂಶ)
    all_ಟೋಕನುಗಳು = []
    ಗಾಗಿ example ಒಳಗೆ ದತ್ತಾಂಶ:
        ಪಠ್ಯ = example["story"]
        ಪಠ್ಯ = ಪಠ್ಯ.strip()  # get rid of leading/trailing whitespace
        ಟೋಕನುಗಳು = encode(ಪಠ್ಯ)
        all_ಟೋಕನುಗಳು.append(eot)
        all_ಟೋಕನುಗಳು.extend(ಟೋಕನುಗಳು)
    ಹಿಂತಿರುಗಿ all_ಟೋಕನುಗಳು

ವ್ಯಾಖ್ಯಾನ tokenize(model_desc):
    # shard 0 will be the val split, rest is train
    ದತ್ತಾಂಶ_dir = os.ಮಾರ್ಗ.join(ದತ್ತಾಂಶ_CACHE_DIR, "TinyStories_all_data")
    shard_filenames = sorted(glob.glob(os.ಮಾರ್ಗ.join(ದತ್ತಾಂಶ_dir, "*.json")))
    val_shards = [shard_filenames[0]]
    train_shards = shard_filenames[1:]
    ಗಾಗಿ ವಿಭಜಿಸು_name, ವಿಭಜಿಸು_shards ಒಳಗೆ [("val", val_shards), ("train", train_shards)]:

        ಮುದ್ರಿಸು(f"Tokenizing {split_name} split...")
        all_ಟೋಕನುಗಳು = []
        ಜೊತೆ ProcessPoolExecutor() ಇವ executor:
            futures = [executor.submit(process_shard, shard_ಸೂಚ್ಯಂಕ, shard_filename, model_desc)
                       ಗಾಗಿ shard_ಸೂಚ್ಯಂಕ, shard_filename ಒಳಗೆ enumerate(ವಿಭಜಿಸು_shards)]
            ಗಾಗಿ future ಒಳಗೆ as_completed(futures):
                all_ಟೋಕನುಗಳು.extend(future.ಫಲಿತಾಂಶ())

        ವಿಭಜಿಸು_filename = os.ಮಾರ್ಗ.join(ದತ್ತಾಂಶ_CACHE_DIR, f"TinyStories_{split_name}.bin")
        write_datafile(ವಿಭಜಿಸು_filename, all_ಟೋಕನುಗಳು, model_desc)

ಆದರೆ __name__ == "__main__":
    parser = argparse.ವಾದ_Parser(description="Tiny Stories dataset preprocessing")
    parser.ಕೂಡಿಸು_ವಾದ("-m", "--model_desc", type=str, ಪೂರ್ವನಿಯೋಜಿತ="gpt-2", choices=["gpt-2", "llama-3"], help="Model type, gpt-2|llama-3")
    ವಾದಗಳು = parser.parse_ವಾದಗಳು()
    download()
    tokenize(ವಾದಗಳು.model_desc)