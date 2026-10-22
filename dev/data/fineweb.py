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
درآمد os
درآمد argparse
درآمد multiprocessing بطور mp

درآمد numpy بطور np
درآمد tiktoken
سے datasets درآمد بھرو_dataset
سے tqdm درآمد tqdm

سے transformers درآمد AutoTokenizer


سے ڈیٹا_common درآمد write_datafile
# ------------------------------------------

parser = argparse.دلیل_Parser(description="FineWeb and Edu-FineWeb dataset preprocessing")
parser.جوڑو_دلیل("-t", "--type", type=str, طےشدہ="classic", help="Fineweb type, edu|classic")
parser.جوڑو_دلیل("-v", "--version", type=str, طےشدہ="10B", help="Fineweb data sample size, 10B|100B")
parser.جوڑو_دلیل("-m", "--model_desc", type=str, طےشدہ="gpt-2", help="Model descriptor, gpt-2|llama-3")
parser.جوڑو_دلیل("-s", "--shard_size", type=صحیح, طےشدہ=10**8, help="Size of each data shard in the output .bin files, in tokens")
دلائل = parser.parse_دلائل()

# FineWeb has a few possible subsamples available
أكد دلائل.نسخہ میں {"10B", "100B"}, "version must be one of: 10B, 100B"
أكد دلائل.type میں {"edu", "classic"}, "type must be one of: edu, classic"
directories = {
    ("classic", "10B"): ("fineweb10B", "sample-10BT"),
    ("classic", "100B"): ("fineweb100B", "sample-100BT"),
    ("edu", "10B"): ("edu_fineweb10B", "sample-10BT"),
    ("edu", "100B"): ("edu_fineweb100B", "sample-100BT")
}
local_dir, remote_name = directories[(دلائل.type, دلائل.نسخہ)]

# create the cache the local directory if it doesn't exist yet
ڈیٹا_CACHE_DIR = os.راستہ.join(os.راستہ.dirname(__file__), local_dir)
os.makedirs(ڈیٹا_CACHE_DIR, exist_ok=سچ)

# download the dataset
اگر دلائل.type == "classic":
    fw = بھرو_dataset("HuggingFaceFW/fineweb", name=remote_name, بانٹو="train")
    name = "fineweb"
وگرنہ_اگر دلائل.type =="edu":
    fw = بھرو_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, بانٹو="train")
    name = "edu_fineweb"

تعریف tokenize_llama(doc):
    # tokenizes a single document and returns a numpy array of uint32 tokens
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
    encode = بےنام s: tokenizer.encode(s, جوڑو_special_علامات=جھوٹ, verbose=جھوٹ, بانٹو_special_علامات=سچ)
    eot = tokenizer.encode('')[0] # by default the tokenizer adds the EOT token (128000)
    علامات = [eot] # the special <|endoftext|> token delimits all documents
    علامات.extend(encode(doc["text"]))
    علامات_np = np.array(علامات)
    أكد (0 <= علامات_np).all() اور (علامات_np < 2**32).all(), "token dictionary too large for uint32"
    علامات_np_uint = علامات_np.astype(np.uint32)
    واپس علامات_np_uint

تعریف tokenize_gpt2(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    enc = tiktoken.get_encoding("gpt2")
    encode = بےنام s: enc.encode_ordinary(s)
    eot = enc.special_علامات['<|endoftext|>'] # end of text token
    علامات = [eot] # the special <|endoftext|> token delimits all documents
    علامات.extend(encode(doc["text"]))
    علامات_np = np.array(علامات)
    أكد (0 <= علامات_np).all() اور (علامات_np < 2**16).all(), "token dictionary too large for uint16"
    علامات_np_uint = علامات_np.astype(np.uint16)
    واپس علامات_np_uint

token_dtype = {
    "gpt-2": np.uint16,
    "llama-3": np.uint32
}[دلائل.model_desc]

# tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
nprocs = max(1, os.cpu_گنتی() - 2) # don't hog the entire system
ساتھ mp.Pool(nprocs) بطور pool:
    shard_اشاریہ = 0
    # preallocate buffer to hold current shard
    all_علامات_np = np.خالی_خانے((دلائل.shard_سائز,), dtype=token_dtype)
    token_گنتی = 0
    پیش_رفت_bar = کچھ_نہیں

    tokenize = بےنام x: کچھ_نہیں
    اگر دلائل.model_desc == "gpt-2":
        tokenize = tokenize_gpt2
    وگرنہ_اگر دلائل.model_desc == "llama-3":
        tokenize = tokenize_llama
    ورنہ:
        اٹھاؤ قدر_Error(f"unknown model {args.model_desc}")

    برائے علامات میں pool.imap(tokenize, fw, chunksize=16):

        # is there enough space in the current shard for the new tokens?
        اگر token_گنتی + لمبائی(علامات) < دلائل.shard_سائز:
            # simply append tokens to current shard
            all_علامات_np[token_گنتی:token_گنتی+لمبائی(علامات)] = علامات
            token_گنتی += لمبائی(علامات)
            # update progress bar
            اگر پیش_رفت_bar ہے کچھ_نہیں:
                پیش_رفت_bar = tqdm(کل=دلائل.shard_سائز, unit="tokens", desc=f"Shard {shard_index}")
            پیش_رفت_bar.update(لمبائی(علامات))
        ورنہ:
            # write the current shard and start a new one
            بانٹو = "val" اگر shard_اشاریہ == 0 ورنہ "train"
            filename = os.راستہ.join(ڈیٹا_CACHE_DIR, f"{name}_{split}_{shard_index:06d}.bin")
            # split the document into whatever fits in this shard; the remainder goes to next one
            remainder = دلائل.shard_سائز - token_گنتی
            پیش_رفت_bar.update(remainder)
            all_علامات_np[token_گنتی:token_گنتی+remainder] = علامات[:remainder]
            write_datafile(filename, all_علامات_np.tolist(), دلائل.model_desc)
            shard_اشاریہ += 1
            پیش_رفت_bar = کچھ_نہیں
            # populate the next shard with the leftovers of the current doc
            all_علامات_np[0:لمبائی(علامات)-remainder] = علامات[remainder:]
            token_گنتی = لمبائی(علامات)-remainder

    # write any remaining tokens as the last shard
    اگر token_گنتی != 0:
        بانٹو = "val" اگر shard_اشاریہ == 0 ورنہ "train"
        filename = os.راستہ.join(ڈیٹا_CACHE_DIR, f"{name}_{split}_{shard_index:06d}.bin")
        write_datafile(filename, (all_علامات_np[:token_گنتی]).tolist(), دلائل.model_desc)
