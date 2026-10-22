"""
FineWeb dataset (for srs pretraining)
https://huggingface.co/datasets/HuggingFaceFW/fineweb

example doc to highlight the structure of the dataset:
{
  "text": "Posted by mattsmith on 20th April 2012\nStraight from...",
  "id": "<urn:uuid:d853d453-196e-4488-a411-efc2b26c40d2>",
  "dump": "CC-MAIN-2013-20",
  "url": "http://nleastchatter.com/philliesphandom/tag/freddy-galvis/",
  "date": "2013-05-18T07:24:47Z",
  "file_path": "s3://commoncrawl/long.../path.../file.gz",
  "language": "en",
  "language_score": 0.9185474514961243,
  "token_count": 594
}

Example of downloading the 100B dataset of FineWebEDU, from root directory:
python dev/data/fineweb.py -t edu -v 100B
100B runs for small few hours, depending on your internet and computer.
"""
आयात os
आयात argparse
आयात multiprocessing as mp

आयात numpy as np
आयात tiktoken
from datasets आयात load_dataset
from tqdm आयात tqdm

from transformers आयात AutoTokenizer


from तथ्य_common आयात write_datafile
# ------------------------------------------

parser = argparse.ArgumentParser(description="FineWeb and Edu-FineWeb dataset preprocessing")
parser.add_argument("-t", "--type", type=str, शेष="classic", help="Fineweb type, edu|classic")
parser.add_argument("-v", "--version", type=str, शेष="10B", help="Fineweb data sample size, 10B|100B")
parser.add_argument("-m", "--model_desc", type=str, शेष="gpt-2", help="Model descriptor, gpt-2|llama-3")
parser.add_argument("-s", "--shard_size", type=पूर्णांक, शेष=10**8, help="Size of each data shard in the output .bin files, in tokens")
तर्क = parser.parse_args()

# FineWeb has a few possible subsamples available
assert तर्क.version in {"10B", "100B"}, "version must be one of: 10B, 100B"
assert तर्क.type in {"edu", "classic"}, "type must be one of: edu, classic"
directories = {
    ("classic", "10B"): ("fineweb10B", "sample-10BT"),
    ("classic", "100B"): ("fineweb100B", "sample-100BT"),
    ("edu", "10B"): ("edu_fineweb10B", "sample-10BT"),
    ("edu", "100B"): ("edu_fineweb100B", "sample-100BT")
}
local_dir, remote_name = directories[(तर्क.type, तर्क.version)]

# create the cache the local directory if it doesn't exist yet
तथ्य_CACHE_DIR = os.path.मेल(os.path.dirname(__file__), local_dir)
os.makedirs(तथ्य_CACHE_DIR, exist_ok=True)

# download the dataset
अगर तर्क.type == "classic":
    fw = load_dataset("HuggingFaceFW/fineweb", name=remote_name, split="train")
    name = "fineweb"
elif तर्क.type =="edu":
    fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")
    name = "edu_fineweb"

def tokenize_llama(doc):
    # tokenizes a single document and returns a numpy array of uint32 tokens
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
    encode = lambda s: tokenizer.encode(s, add_special_tokens=False, verbose=False, split_special_tokens=True)
    eot = tokenizer.encode('')[0] # by default the tokenizer adds the EOT token (128000)
    tokens = [eot] # the special <|endoftext|> token delimits all documents
    tokens.extend(encode(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**32).all(), "token dictionary too large for uint32"
    tokens_np_uint = tokens_np.astype(np.uint32)
    वापस tokens_np_uint

def tokenize_gpt2(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode_ordinary(s)
    eot = enc._special_tokens['<|endoftext|>'] # end of text token
    tokens = [eot] # the special <|endoftext|> token delimits all documents
    tokens.extend(encode(doc["text"]))
    tokens_np = np.array(tokens)
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint = tokens_np.astype(np.uint16)
    वापस tokens_np_uint

token_dtype = {
    "gpt-2": np.uint16,
    "llama-3": np.uint32
}[तर्क.model_desc]

# tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
nprocs = max(1, os.cpu_गिनती() - 2) # don't hog the entire system
with mp.Pool(nprocs) as pool:
    shard_सूचकांक = 0
    # preallocate buffer to hold current shard
    all_tokens_np = np.empty((तर्क.shard_आकार,), dtype=token_dtype)
    token_गिनती = 0
    progress_bar = None

    tokenize = lambda x: None
    अगर तर्क.model_desc == "gpt-2":
        tokenize = tokenize_gpt2
    elif तर्क.model_desc == "llama-3":
        tokenize = tokenize_llama
    अन्यथा:
        raise मूल्य_Error(f"unknown model {args.model_desc}")

    क्रम tokens in pool.imap(tokenize, fw, chunksize=16):

        # is there enough space in the current shard for the new tokens?
        अगर token_गिनती + len(tokens) < तर्क.shard_आकार:
            # simply append tokens to current shard
            all_tokens_np[token_गिनती:token_गिनती+len(tokens)] = tokens
            token_गिनती += len(tokens)
            # update progress bar
            अगर progress_bar is None:
                progress_bar = tqdm(योग=तर्क.shard_आकार, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.नया_हाल(len(tokens))
        अन्यथा:
            # write the current shard and start a new one
            split = "val" अगर shard_सूचकांक == 0 अन्यथा "train"
            filename = os.path.मेल(तथ्य_CACHE_DIR, f"{name}_{split}_{shard_index:06d}.bin")
            # split the document into whatever fits in this shard; the remainder goes to next one
            remainder = तर्क.shard_आकार - token_गिनती
            progress_bar.नया_हाल(remainder)
            all_tokens_np[token_गिनती:token_गिनती+remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np.tolist(), तर्क.model_desc)
            shard_सूचकांक += 1
            progress_bar = None
            # populate the next shard with the leftovers of the current doc
            all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
            token_गिनती = len(tokens)-remainder

    # write any remaining tokens as the last shard
    अगर token_गिनती != 0:
        split = "val" अगर shard_सूचकांक == 0 अन्यथा "train"
        filename = os.path.मेल(तथ्य_CACHE_DIR, f"{name}_{split}_{shard_index:06d}.bin")
        write_datafile(filename, (all_tokens_np[:token_गिनती]).tolist(), तर्क.model_desc)
