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
ਆਯਾਤ os
ਆਯਾਤ argparse
ਆਯਾਤ multiprocessing ਇਵ mp

ਆਯਾਤ numpy ਇਵ np
ਆਯਾਤ tiktoken
ਤਃ datasets ਆਯਾਤ ਲੱਦੋ_dataset
ਤਃ tqdm ਆਯਾਤ tqdm

ਤਃ transformers ਆਯਾਤ AutoTokenizer


ਤਃ ਅੰਕੜੇ_common ਆਯਾਤ write_datafile
# ------------------------------------------

parser = argparse.ਦਲੀਲ_Parser(description="FineWeb and Edu-FineWeb dataset preprocessing")
parser.ਜੋੜੋ_ਦਲੀਲ("-t", "--type", type=str, ਮੂਲ="classic", help="Fineweb type, edu|classic")
parser.ਜੋੜੋ_ਦਲੀਲ("-v", "--version", type=str, ਮੂਲ="10B", help="Fineweb data sample size, 10B|100B")
parser.ਜੋੜੋ_ਦਲੀਲ("-m", "--model_desc", type=str, ਮੂਲ="gpt-2", help="Model descriptor, gpt-2|llama-3")
parser.ਜੋੜੋ_ਦਲੀਲ("-s", "--shard_size", type=ਪੂਰਨ, ਮੂਲ=10**8, help="Size of each data shard in the output .bin files, in tokens")
ਦਲੀਲਾਂ = parser.parse_ਦਲੀਲਾਂ()

# FineWeb has a few possible subsamples available
ਪ੍ਰਤਿਜ੍ਞਾ ਦਲੀਲਾਂ.ਸੰਸਕਰਣ ਵਿੱਚ {"10B", "100B"}, "version must be one of: 10B, 100B"
ਪ੍ਰਤਿਜ੍ਞਾ ਦਲੀਲਾਂ.type ਵਿੱਚ {"edu", "classic"}, "type must be one of: edu, classic"
directories = {
    ("classic", "10B"): ("fineweb10B", "sample-10BT"),
    ("classic", "100B"): ("fineweb100B", "sample-100BT"),
    ("edu", "10B"): ("edu_fineweb10B", "sample-10BT"),
    ("edu", "100B"): ("edu_fineweb100B", "sample-100BT")
}
local_dir, remote_name = directories[(ਦਲੀਲਾਂ.type, ਦਲੀਲਾਂ.ਸੰਸਕਰਣ)]

# create the cache the local directory if it doesn't exist yet
ਅੰਕੜੇ_CACHE_DIR = os.ਰਸਤਾ.join(os.ਰਸਤਾ.dirname(__file__), local_dir)
os.makedirs(ਅੰਕੜੇ_CACHE_DIR, exist_ok=ਸੱਚ)

# download the dataset
ਜੇ ਦਲੀਲਾਂ.type == "classic":
    fw = ਲੱਦੋ_dataset("HuggingFaceFW/fineweb", name=remote_name, ਵੰਡੋ="train")
    name = "fineweb"
ਜਾਂਜੇ ਦਲੀਲਾਂ.type =="edu":
    fw = ਲੱਦੋ_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, ਵੰਡੋ="train")
    name = "edu_fineweb"

ਪਰਿਭਾਸ਼ਾ tokenize_llama(doc):
    # tokenizes a single document and returns a numpy array of uint32 tokens
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
    encode = ਅਨਾਮ s: tokenizer.encode(s, ਜੋੜੋ_special_ਟੋਕਨ=ਝੂਠ, verbose=ਝੂਠ, ਵੰਡੋ_special_ਟੋਕਨ=ਸੱਚ)
    eot = tokenizer.encode('')[0] # by default the tokenizer adds the EOT token (128000)
    ਟੋਕਨ = [eot] # the special <|endoftext|> token delimits all documents
    ਟੋਕਨ.extend(encode(doc["text"]))
    ਟੋਕਨ_np = np.array(ਟੋਕਨ)
    ਪ੍ਰਤਿਜ੍ਞਾ (0 <= ਟੋਕਨ_np).all() ਅਤੇ (ਟੋਕਨ_np < 2**32).all(), "token dictionary too large for uint32"
    ਟੋਕਨ_np_uint = ਟੋਕਨ_np.astype(np.uint32)
    ਵਾਪਸ ਟੋਕਨ_np_uint

ਪਰਿਭਾਸ਼ਾ tokenize_gpt2(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    enc = tiktoken.get_encoding("gpt2")
    encode = ਅਨਾਮ s: enc.encode_ordinary(s)
    eot = enc.special_ਟੋਕਨ['<|endoftext|>'] # end of text token
    ਟੋਕਨ = [eot] # the special <|endoftext|> token delimits all documents
    ਟੋਕਨ.extend(encode(doc["text"]))
    ਟੋਕਨ_np = np.array(ਟੋਕਨ)
    ਪ੍ਰਤਿਜ੍ਞਾ (0 <= ਟੋਕਨ_np).all() ਅਤੇ (ਟੋਕਨ_np < 2**16).all(), "token dictionary too large for uint16"
    ਟੋਕਨ_np_uint = ਟੋਕਨ_np.astype(np.uint16)
    ਵਾਪਸ ਟੋਕਨ_np_uint

token_dtype = {
    "gpt-2": np.uint16,
    "llama-3": np.uint32
}[ਦਲੀਲਾਂ.model_desc]

# tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
nprocs = max(1, os.cpu_ਗਿਣਤੀ() - 2) # don't hog the entire system
ਨਾਲ mp.Pool(nprocs) ਇਵ pool:
    shard_ਸੂਚਕਾਂਕ = 0
    # preallocate buffer to hold current shard
    all_ਟੋਕਨ_np = np.ਖਾਲੀ_ਥਾਵਾਂ((ਦਲੀਲਾਂ.shard_ਮਾਪ,), dtype=token_dtype)
    token_ਗਿਣਤੀ = 0
    ਤਰੱਕੀ_bar = ਕੁਝ_ਨਹੀਂ

    tokenize = ਅਨਾਮ x: ਕੁਝ_ਨਹੀਂ
    ਜੇ ਦਲੀਲਾਂ.model_desc == "gpt-2":
        tokenize = tokenize_gpt2
    ਜਾਂਜੇ ਦਲੀਲਾਂ.model_desc == "llama-3":
        tokenize = tokenize_llama
    ਨਹੀਂਤਾਂ:
        ਉਠਾਓ ਮੁੱਲ_Error(f"unknown model {args.model_desc}")

    ਲਈ ਟੋਕਨ ਵਿੱਚ pool.imap(tokenize, fw, chunksize=16):

        # is there enough space in the current shard for the new tokens?
        ਜੇ token_ਗਿਣਤੀ + ਲੰਬਾਈ(ਟੋਕਨ) < ਦਲੀਲਾਂ.shard_ਮਾਪ:
            # simply append tokens to current shard
            all_ਟੋਕਨ_np[token_ਗਿਣਤੀ:token_ਗਿਣਤੀ+ਲੰਬਾਈ(ਟੋਕਨ)] = ਟੋਕਨ
            token_ਗਿਣਤੀ += ਲੰਬਾਈ(ਟੋਕਨ)
            # update progress bar
            ਜੇ ਤਰੱਕੀ_bar ਹੈ ਕੁਝ_ਨਹੀਂ:
                ਤਰੱਕੀ_bar = tqdm(ਕੁੱਲ=ਦਲੀਲਾਂ.shard_ਮਾਪ, unit="tokens", desc=f"Shard {shard_index}")
            ਤਰੱਕੀ_bar.update(ਲੰਬਾਈ(ਟੋਕਨ))
        ਨਹੀਂਤਾਂ:
            # write the current shard and start a new one
            ਵੰਡੋ = "val" ਜੇ shard_ਸੂਚਕਾਂਕ == 0 ਨਹੀਂਤਾਂ "train"
            filename = os.ਰਸਤਾ.join(ਅੰਕੜੇ_CACHE_DIR, f"{name}_{split}_{shard_index:06d}.bin")
            # split the document into whatever fits in this shard; the remainder goes to next one
            remainder = ਦਲੀਲਾਂ.shard_ਮਾਪ - token_ਗਿਣਤੀ
            ਤਰੱਕੀ_bar.update(remainder)
            all_ਟੋਕਨ_np[token_ਗਿਣਤੀ:token_ਗਿਣਤੀ+remainder] = ਟੋਕਨ[:remainder]
            write_datafile(filename, all_ਟੋਕਨ_np.tolist(), ਦਲੀਲਾਂ.model_desc)
            shard_ਸੂਚਕਾਂਕ += 1
            ਤਰੱਕੀ_bar = ਕੁਝ_ਨਹੀਂ
            # populate the next shard with the leftovers of the current doc
            all_ਟੋਕਨ_np[0:ਲੰਬਾਈ(ਟੋਕਨ)-remainder] = ਟੋਕਨ[remainder:]
            token_ਗਿਣਤੀ = ਲੰਬਾਈ(ਟੋਕਨ)-remainder

    # write any remaining tokens as the last shard
    ਜੇ token_ਗਿਣਤੀ != 0:
        ਵੰਡੋ = "val" ਜੇ shard_ਸੂਚਕਾਂਕ == 0 ਨਹੀਂਤਾਂ "train"
        filename = os.ਰਸਤਾ.join(ਅੰਕੜੇ_CACHE_DIR, f"{name}_{split}_{shard_index:06d}.bin")
        write_datafile(filename, (all_ਟੋਕਨ_np[:token_ਗਿਣਤੀ]).tolist(), ਦਲੀਲਾਂ.model_desc)
