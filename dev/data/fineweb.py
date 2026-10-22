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
من datasets استورد لدو_dataset
من tqdm استورد tqdm

من transformers استورد AutoTokenizer


من انکڑے_common استورد write_datafile
# ------------------------------------------

parser = argparse.دلیل_Parser(description="FineWeb and Edu-FineWeb dataset preprocessing")
parser.جوڑو_دلیل("-t", "--type", type=str, افتراضي="classic", help="Fineweb type, edu|classic")
parser.جوڑو_دلیل("-v", "--version", type=str, افتراضي="10B", help="Fineweb data sample size, 10B|100B")
parser.جوڑو_دلیل("-m", "--model_desc", type=str, افتراضي="gpt-2", help="Model descriptor, gpt-2|llama-3")
parser.جوڑو_دلیل("-s", "--shard_size", type=صحيح, افتراضي=10**8, help="Size of each data shard in the output .bin files, in tokens")
دلیلاں = parser.parse_دلیلاں()

# FineWeb has a few possible subsamples available
أكد دلیلاں.نسخہ في {"10B", "100B"}, "version must be one of: 10B, 100B"
أكد دلیلاں.type في {"edu", "classic"}, "type must be one of: edu, classic"
directories = {
    ("classic", "10B"): ("fineweb10B", "sample-10BT"),
    ("classic", "100B"): ("fineweb100B", "sample-100BT"),
    ("edu", "10B"): ("edu_fineweb10B", "sample-10BT"),
    ("edu", "100B"): ("edu_fineweb100B", "sample-100BT")
}
local_dir, remote_name = directories[(دلیلاں.type, دلیلاں.نسخہ)]

# create the cache the local directory if it doesn't exist yet
انکڑے_CACHE_DIR = os.رستہ.join(os.رستہ.dirname(__file__), local_dir)
os.makedirs(انکڑے_CACHE_DIR, exist_ok=سچ)

# download the dataset
إذا دلیلاں.type == "classic":
    fw = لدو_dataset("HuggingFaceFW/fineweb", name=remote_name, ونڈو="train")
    name = "fineweb"
وإلاإذا دلیلاں.type =="edu":
    fw = لدو_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, ونڈو="train")
    name = "edu_fineweb"

تعریف tokenize_llama(doc):
    # tokenizes a single document and returns a numpy array of uint32 tokens
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
    encode = مجهول s: tokenizer.encode(s, جوڑو_special_ٹوکن=خطأ, verbose=خطأ, ونڈو_special_ٹوکن=سچ)
    eot = tokenizer.encode('')[0] # by default the tokenizer adds the EOT token (128000)
    ٹوکن = [eot] # the special <|endoftext|> token delimits all documents
    ٹوکن.extend(encode(doc["text"]))
    ٹوکن_np = np.array(ٹوکن)
    أكد (0 <= ٹوکن_np).all() و (ٹوکن_np < 2**32).all(), "token dictionary too large for uint32"
    ٹوکن_np_uint = ٹوکن_np.astype(np.uint32)
    أرجع ٹوکن_np_uint

تعریف tokenize_gpt2(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    enc = tiktoken.get_encoding("gpt2")
    encode = مجهول s: enc.encode_ordinary(s)
    eot = enc.special_ٹوکن['<|endoftext|>'] # end of text token
    ٹوکن = [eot] # the special <|endoftext|> token delimits all documents
    ٹوکن.extend(encode(doc["text"]))
    ٹوکن_np = np.array(ٹوکن)
    أكد (0 <= ٹوکن_np).all() و (ٹوکن_np < 2**16).all(), "token dictionary too large for uint16"
    ٹوکن_np_uint = ٹوکن_np.astype(np.uint16)
    أرجع ٹوکن_np_uint

token_dtype = {
    "gpt-2": np.uint16,
    "llama-3": np.uint32
}[دلیلاں.model_desc]

# tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
nprocs = max(1, os.cpu_گنتی() - 2) # don't hog the entire system
نال mp.Pool(nprocs) كـ pool:
    shard_اشاریہ = 0
    # preallocate buffer to hold current shard
    all_ٹوکن_np = np.خالی_تھاواں((دلیلاں.shard_ناپ,), dtype=token_dtype)
    token_گنتی = 0
    ترقی_bar = کجھ_نئیں

    tokenize = مجهول x: کجھ_نئیں
    إذا دلیلاں.model_desc == "gpt-2":
        tokenize = tokenize_gpt2
    وإلاإذا دلیلاں.model_desc == "llama-3":
        tokenize = tokenize_llama
    وإلا:
        ارفع قدر_Error(f"unknown model {args.model_desc}")

    لكل ٹوکن في pool.imap(tokenize, fw, chunksize=16):

        # is there enough space in the current shard for the new tokens?
        إذا token_گنتی + لمبائی(ٹوکن) < دلیلاں.shard_ناپ:
            # simply append tokens to current shard
            all_ٹوکن_np[token_گنتی:token_گنتی+لمبائی(ٹوکن)] = ٹوکن
            token_گنتی += لمبائی(ٹوکن)
            # update progress bar
            إذا ترقی_bar اے کجھ_نئیں:
                ترقی_bar = tqdm(کل=دلیلاں.shard_ناپ, unit="tokens", desc=f"Shard {shard_index}")
            ترقی_bar.update(لمبائی(ٹوکن))
        وإلا:
            # write the current shard and start a new one
            ونڈو = "val" إذا shard_اشاریہ == 0 وإلا "train"
            filename = os.رستہ.join(انکڑے_CACHE_DIR, f"{name}_{split}_{shard_index:06d}.bin")
            # split the document into whatever fits in this shard; the remainder goes to next one
            remainder = دلیلاں.shard_ناپ - token_گنتی
            ترقی_bar.update(remainder)
            all_ٹوکن_np[token_گنتی:token_گنتی+remainder] = ٹوکن[:remainder]
            write_datafile(filename, all_ٹوکن_np.tolist(), دلیلاں.model_desc)
            shard_اشاریہ += 1
            ترقی_bar = کجھ_نئیں
            # populate the next shard with the leftovers of the current doc
            all_ٹوکن_np[0:لمبائی(ٹوکن)-remainder] = ٹوکن[remainder:]
            token_گنتی = لمبائی(ٹوکن)-remainder

    # write any remaining tokens as the last shard
    إذا token_گنتی != 0:
        ونڈو = "val" إذا shard_اشاریہ == 0 وإلا "train"
        filename = os.رستہ.join(انکڑے_CACHE_DIR, f"{name}_{split}_{shard_index:06d}.bin")
        write_datafile(filename, (all_ٹوکن_np[:token_گنتی]).tolist(), دلیلاں.model_desc)
