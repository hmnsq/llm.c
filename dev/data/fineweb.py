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
استورد os
استورد argparse
استورد multiprocessing كـ mp

استورد numpy كـ np
استورد tiktoken
من datasets استورد ڀريو_dataset
من tqdm استورد tqdm

من transformers استورد AutoTokenizer


من معلومات_common استورد write_datafile
# ------------------------------------------

parser = argparse.دليل_Parser(description="FineWeb and Edu-FineWeb dataset preprocessing")
parser.جوڙيو_دليل("-t", "--type", type=str, افتراضي="classic", help="Fineweb type, edu|classic")
parser.جوڙيو_دليل("-v", "--version", type=str, افتراضي="10B", help="Fineweb data sample size, 10B|100B")
parser.جوڙيو_دليل("-m", "--model_desc", type=str, افتراضي="gpt-2", help="Model descriptor, gpt-2|llama-3")
parser.جوڙيو_دليل("-s", "--shard_size", type=صحيح, افتراضي=10**8, help="Size of each data shard in the output .bin files, in tokens")
دليلون = parser.parse_دليلون()

# FineWeb has a few possible subsamples available
أكد دليلون.نسخو في {"10B", "100B"}, "version must be one of: 10B, 100B"
أكد دليلون.type في {"edu", "classic"}, "type must be one of: edu, classic"
directories = {
    ("classic", "10B"): ("fineweb10B", "sample-10BT"),
    ("classic", "100B"): ("fineweb100B", "sample-100BT"),
    ("edu", "10B"): ("edu_fineweb10B", "sample-10BT"),
    ("edu", "100B"): ("edu_fineweb100B", "sample-100BT")
}
local_dir, remote_name = directories[(دليلون.type, دليلون.نسخو)]

# create the cache the local directory if it doesn't exist yet
معلومات_CACHE_DIR = os.رستو.join(os.رستو.dirname(__file__), local_dir)
os.makedirs(معلومات_CACHE_DIR, exist_ok=سچ)

# download the dataset
إذا دليلون.type == "classic":
    fw = ڀريو_dataset("HuggingFaceFW/fineweb", name=remote_name, ورهايو="train")
    name = "fineweb"
وإلاإذا دليلون.type =="edu":
    fw = ڀريو_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, ورهايو="train")
    name = "edu_fineweb"

وصف tokenize_llama(doc):
    # tokenizes a single document and returns a numpy array of uint32 tokens
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
    encode = مجهول s: tokenizer.encode(s, جوڙيو_special_ٽوڪن=خطأ, verbose=خطأ, ورهايو_special_ٽوڪن=سچ)
    eot = tokenizer.encode('')[0] # by default the tokenizer adds the EOT token (128000)
    ٽوڪن = [eot] # the special <|endoftext|> token delimits all documents
    ٽوڪن.extend(encode(doc["text"]))
    ٽوڪن_np = np.array(ٽوڪن)
    أكد (0 <= ٽوڪن_np).all() و (ٽوڪن_np < 2**32).all(), "token dictionary too large for uint32"
    ٽوڪن_np_uint = ٽوڪن_np.astype(np.uint32)
    أرجع ٽوڪن_np_uint

وصف tokenize_gpt2(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    enc = tiktoken.get_encoding("gpt2")
    encode = مجهول s: enc.encode_ordinary(s)
    eot = enc.special_ٽوڪن['<|endoftext|>'] # end of text token
    ٽوڪن = [eot] # the special <|endoftext|> token delimits all documents
    ٽوڪن.extend(encode(doc["text"]))
    ٽوڪن_np = np.array(ٽوڪن)
    أكد (0 <= ٽوڪن_np).all() و (ٽوڪن_np < 2**16).all(), "token dictionary too large for uint16"
    ٽوڪن_np_uint = ٽوڪن_np.astype(np.uint16)
    أرجع ٽوڪن_np_uint

token_dtype = {
    "gpt-2": np.uint16,
    "llama-3": np.uint32
}[دليلون.model_desc]

# tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
nprocs = max(1, os.cpu_ڳڻپ() - 2) # don't hog the entire system
سان mp.Pool(nprocs) كـ pool:
    shard_اشاريو = 0
    # preallocate buffer to hold current shard
    all_ٽوڪن_np = np.خالي_جايون((دليلون.shard_سائيز,), dtype=token_dtype)
    token_ڳڻپ = 0
    ترقي_bar = ڪجهه_نه

    tokenize = مجهول x: ڪجهه_نه
    إذا دليلون.model_desc == "gpt-2":
        tokenize = tokenize_gpt2
    وإلاإذا دليلون.model_desc == "llama-3":
        tokenize = tokenize_llama
    وإلا:
        ارفع قدر_Error(f"unknown model {args.model_desc}")

    لكل ٽوڪن في pool.imap(tokenize, fw, chunksize=16):

        # is there enough space in the current shard for the new tokens?
        إذا token_ڳڻپ + ڊيگهه(ٽوڪن) < دليلون.shard_سائيز:
            # simply append tokens to current shard
            all_ٽوڪن_np[token_ڳڻپ:token_ڳڻپ+ڊيگهه(ٽوڪن)] = ٽوڪن
            token_ڳڻپ += ڊيگهه(ٽوڪن)
            # update progress bar
            إذا ترقي_bar آهي ڪجهه_نه:
                ترقي_bar = tqdm(ڪل=دليلون.shard_سائيز, unit="tokens", desc=f"Shard {shard_index}")
            ترقي_bar.update(ڊيگهه(ٽوڪن))
        وإلا:
            # write the current shard and start a new one
            ورهايو = "val" إذا shard_اشاريو == 0 وإلا "train"
            filename = os.رستو.join(معلومات_CACHE_DIR, f"{name}_{split}_{shard_index:06d}.bin")
            # split the document into whatever fits in this shard; the remainder goes to next one
            remainder = دليلون.shard_سائيز - token_ڳڻپ
            ترقي_bar.update(remainder)
            all_ٽوڪن_np[token_ڳڻپ:token_ڳڻپ+remainder] = ٽوڪن[:remainder]
            write_datafile(filename, all_ٽوڪن_np.tolist(), دليلون.model_desc)
            shard_اشاريو += 1
            ترقي_bar = ڪجهه_نه
            # populate the next shard with the leftovers of the current doc
            all_ٽوڪن_np[0:ڊيگهه(ٽوڪن)-remainder] = ٽوڪن[remainder:]
            token_ڳڻپ = ڊيگهه(ٽوڪن)-remainder

    # write any remaining tokens as the last shard
    إذا token_ڳڻپ != 0:
        ورهايو = "val" إذا shard_اشاريو == 0 وإلا "train"
        filename = os.رستو.join(معلومات_CACHE_DIR, f"{name}_{split}_{shard_index:06d}.bin")
        write_datafile(filename, (all_ٽوڪن_np[:token_ڳڻپ]).tolist(), دليلون.model_desc)
