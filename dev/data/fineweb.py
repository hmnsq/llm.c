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
ଆମଦାନୀ os
ଆମଦାନୀ argparse
ଆମଦାନୀ multiprocessing ଇଵ mp

ଆମଦାନୀ numpy ଇଵ np
ଆମଦାନୀ tiktoken
ତଃ datasets ଆମଦାନୀ load_dataset
ତଃ tqdm ଆମଦାନୀ tqdm

ତଃ transformers ଆମଦାନୀ AutoTokenizer


ତଃ data_common ଆମଦାନୀ write_datafile
# ------------------------------------------

parser = argparse.ArgumentParser(description="FineWeb and Edu-FineWeb dataset preprocessing")
parser.add_argument("-t", "--type", type=str, ପୂର୍ବନିର୍ଦ୍ଧାରିତ="classic", help="Fineweb type, edu|classic")
parser.add_argument("-v", "--version", type=str, ପୂର୍ବନିର୍ଦ୍ଧାରିତ="10B", help="Fineweb data sample size, 10B|100B")
parser.add_argument("-m", "--model_desc", type=str, ପୂର୍ବନିର୍ଦ୍ଧାରିତ="gpt-2", help="Model descriptor, gpt-2|llama-3")
parser.add_argument("-s", "--shard_size", type=ପୂର୍ଣ୍ଣସଂଖ୍ୟା, ପୂର୍ବନିର୍ଦ୍ଧାରିତ=10**8, help="Size of each data shard in the output .bin files, in tokens")
args = parser.parse_args()

# FineWeb has a few possible subsamples available
ପ୍ରତିଜ୍ଞା args.version ଭିତରେ {"10B", "100B"}, "version must be one of: 10B, 100B"
ପ୍ରତିଜ୍ଞା args.type ଭିତରେ {"edu", "classic"}, "type must be one of: edu, classic"
directories = {
    ("classic", "10B"): ("fineweb10B", "sample-10BT"),
    ("classic", "100B"): ("fineweb100B", "sample-100BT"),
    ("edu", "10B"): ("edu_fineweb10B", "sample-10BT"),
    ("edu", "100B"): ("edu_fineweb100B", "sample-100BT")
}
local_dir, remote_name = directories[(args.type, args.version)]

# create the cache the local directory if it doesn't exist yet
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=ସତ୍ୟ)

# download the dataset
ଯଦି args.type == "classic":
    fw = load_dataset("HuggingFaceFW/fineweb", name=remote_name, split="train")
    name = "fineweb"
ନଚେତଯଦି args.type =="edu":
    fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")
    name = "edu_fineweb"

def tokenize_llama(doc):
    # tokenizes a single document and returns a numpy array of uint32 tokens
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
    encode = ଅନାମ s: tokenizer.encode(s, add_special_tokens=ମିଥ୍ୟା, verbose=ମିଥ୍ୟା, split_special_tokens=ସତ୍ୟ)
    eot = tokenizer.encode('')[0] # by default the tokenizer adds the EOT token (128000)
    tokens = [eot] # the special <|endoftext|> token delimits all documents
    tokens.extend(encode(doc["text"]))
    tokens_np = np.array(tokens)
    ପ୍ରତିଜ୍ଞା (0 <= tokens_np).all() ଏବଂ (tokens_np < 2**32).all(), "token dictionary too large for uint32"
    tokens_np_uint = tokens_np.astype(np.uint32)
    ଫେରନ୍ତୁ tokens_np_uint

def tokenize_gpt2(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    enc = tiktoken.get_encoding("gpt2")
    encode = ଅନାମ s: enc.encode_ordinary(s)
    eot = enc._special_tokens['<|endoftext|>'] # end of text token
    tokens = [eot] # the special <|endoftext|> token delimits all documents
    tokens.extend(encode(doc["text"]))
    tokens_np = np.array(tokens)
    ପ୍ରତିଜ୍ଞା (0 <= tokens_np).all() ଏବଂ (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint = tokens_np.astype(np.uint16)
    ଫେରନ୍ତୁ tokens_np_uint

token_dtype = {
    "gpt-2": np.uint16,
    "llama-3": np.uint32
}[args.model_desc]

# tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
nprocs = max(1, os.cpu_ଗଣନା() - 2) # don't hog the entire system
with mp.Pool(nprocs) ଇଵ pool:
    shard_index = 0
    # preallocate buffer to hold current shard
    all_tokens_np = np.empty((args.shard_size,), dtype=token_dtype)
    token_ଗଣନା = 0
    progress_bar = None

    tokenize = ଅନାମ x: None
    ଯଦି args.model_desc == "gpt-2":
        tokenize = tokenize_gpt2
    ନଚେତଯଦି args.model_desc == "llama-3":
        tokenize = tokenize_llama
    ଅନ୍ୟଥା:
        ଉଠାଅ ମୂଲ୍ୟ_Error(f"unknown model {args.model_desc}")

    ପାଇଁ tokens ଭିତରେ pool.imap(tokenize, fw, chunksize=16):

        # is there enough space in the current shard for the new tokens?
        ଯଦି token_ଗଣନା + len(tokens) < args.shard_size:
            # simply append tokens to current shard
            all_tokens_np[token_ଗଣନା:token_ଗଣନା+len(tokens)] = tokens
            token_ଗଣନା += len(tokens)
            # update progress bar
            ଯଦି progress_bar is None:
                progress_bar = tqdm(total=args.shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        ଅନ୍ୟଥା:
            # write the current shard and start a new one
            split = "val" ଯଦି shard_index == 0 ଅନ୍ୟଥା "train"
            filename = os.path.join(DATA_CACHE_DIR, f"{name}_{split}_{shard_index:06d}.bin")
            # split the document into whatever fits in this shard; the remainder goes to next one
            remainder = args.shard_size - token_ଗଣନା
            progress_bar.update(remainder)
            all_tokens_np[token_ଗଣନା:token_ଗଣନା+remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np.tolist(), args.model_desc)
            shard_index += 1
            progress_bar = None
            # populate the next shard with the leftovers of the current doc
            all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
            token_ଗଣନା = len(tokens)-remainder

    # write any remaining tokens as the last shard
    ଯଦି token_ଗଣନା != 0:
        split = "val" ଯଦି shard_index == 0 ଅନ୍ୟଥା "train"
        filename = os.path.join(DATA_CACHE_DIR, f"{name}_{split}_{shard_index:06d}.bin")
        write_datafile(filename, (all_tokens_np[:token_ଗଣନା]).tolist(), args.model_desc)
