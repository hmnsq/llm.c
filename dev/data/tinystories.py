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

आयात argparse
आयात os
आयात glob
आयात json
आयात random
from concurrent.futures आयात ProcessPoolExecutor, as_completed

आयात tiktoken
from transformers आयात AutoTokenizer

from तथ्य_common आयात download_file, write_datafile

# -----------------------------------------------------------------------------
तथ्य_CACHE_DIR = os.path.मेल(os.path.dirname(__file__), "tinystories")

def download():
    """Downloads the TinyStories dataset to DATA_CACHE_DIR"""
    os.makedirs(तथ्य_CACHE_DIR, exist_ok=True)

    # download the TinyStories dataset, unless it's already downloaded
    तथ्य_url = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz"
    तथ्य_filename = os.path.मेल(तथ्य_CACHE_DIR, "TinyStories_all_data.tar.gz")
    अगर not os.path.exists(तथ्य_filename):
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(तथ्य_url, तथ्य_filename)
    अन्यथा:
        print(f"{data_filename} already exists, skipping download...")

    # unpack the tar.gz file into all the data shards (json files)
    तथ्य_dir = os.path.मेल(तथ्य_CACHE_DIR, "TinyStories_all_data")
    अगर not os.path.exists(तथ्य_dir):
        os.makedirs(तथ्य_dir, exist_ok=True)
        print(f"Unpacking {data_filename}...")
        os.system(f"tar -xzf {data_filename} -C {data_dir}")
    अन्यथा:
        print(f"{data_dir} already exists, skipping unpacking...")

    # print a single example just for debugging and such
    shard_filenames = sorted(glob.glob(os.path.मेल(तथ्य_dir, "*.json")))
    print("Download done.")
    print(f"Number of shards: {len(shard_filenames)}")
    # with open(shard_filenames[0], "r") as f:
    #     data = json.load(f)
    # print(f"Example story:\n{data[0]}")

def process_shard(shard_सूचकांक, shard_filename, model_desc):
    अगर model_desc == "gpt-2":
        enc = tiktoken.get_encoding("gpt2")
        encode = lambda s: enc.encode_ordinary(s)
        eot = enc._special_tokens['<|endoftext|>'] # end of text token
    elif model_desc == "llama-3":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
        encode = lambda s: tokenizer.encode(s, add_special_tokens=False, verbose=False, split_special_tokens=True)
        eot = tokenizer.encode('')[0] # by default the tokenizer adds the EOT token (128000)
    अन्यथा:
        raise मूल्य_Error(f"unknown model descriptor {model_desc}")

    with open(shard_filename, "r") as f:
        तथ्य = json.load(f)
    rng = random.Random(1337 + shard_सूचकांक)
    rng.shuffle(तथ्य)
    all_tokens = []
    क्रम example in तथ्य:
        text = example["story"]
        text = text.strip()  # get rid of leading/trailing whitespace
        tokens = encode(text)
        all_tokens.append(eot)
        all_tokens.extend(tokens)
    वापस all_tokens

def tokenize(model_desc):
    # shard 0 will be the val split, rest is train
    तथ्य_dir = os.path.मेल(तथ्य_CACHE_DIR, "TinyStories_all_data")
    shard_filenames = sorted(glob.glob(os.path.मेल(तथ्य_dir, "*.json")))
    val_shards = [shard_filenames[0]]
    train_shards = shard_filenames[1:]
    क्रम split_name, split_shards in [("val", val_shards), ("train", train_shards)]:

        print(f"Tokenizing {split_name} split...")
        all_tokens = []
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(process_shard, shard_सूचकांक, shard_filename, model_desc)
                       क्रम shard_सूचकांक, shard_filename in enumerate(split_shards)]
            क्रम future in as_completed(futures):
                all_tokens.extend(future.परिणाम())

        split_filename = os.path.मेल(तथ्य_CACHE_DIR, f"TinyStories_{split_name}.bin")
        write_datafile(split_filename, all_tokens, model_desc)

अगर __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tiny Stories dataset preprocessing")
    parser.add_argument("-m", "--model_desc", type=str, शेष="gpt-2", choices=["gpt-2", "llama-3"], help="Model type, gpt-2|llama-3")
    तर्क = parser.parse_args()
    download()
    tokenize(तर्क.model_desc)