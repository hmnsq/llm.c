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
من datasets استورد بارگیری_dataset
من tqdm استورد tqdm

من transformers استورد AutoTokenizer


من داده_common استورد write_datafile
# ------------------------------------------

parser = argparse.آرگومان_Parser(description="FineWeb and Edu-FineWeb dataset preprocessing")
parser.جمع_کن_آرگومان("-t", "--type", type=str, افتراضي="classic", help="Fineweb type, edu|classic")
parser.جمع_کن_آرگومان("-v", "--version", type=str, افتراضي="10B", help="Fineweb data sample size, 10B|100B")
parser.جمع_کن_آرگومان("-m", "--model_desc", type=str, افتراضي="gpt-2", help="Model descriptor, gpt-2|llama-3")
parser.جمع_کن_آرگومان("-s", "--shard_size", type=صحيح, افتراضي=10**8, help="Size of each data shard in the output .bin files, in tokens")
آرگومانها = parser.parse_آرگومانها()

# FineWeb has a few possible subsamples available
أكد آرگومانها.نسخه في {"10B", "100B"}, "version must be one of: 10B, 100B"
أكد آرگومانها.type في {"edu", "classic"}, "type must be one of: edu, classic"
directories = {
    ("classic", "10B"): ("fineweb10B", "sample-10BT"),
    ("classic", "100B"): ("fineweb100B", "sample-100BT"),
    ("edu", "10B"): ("edu_fineweb10B", "sample-10BT"),
    ("edu", "100B"): ("edu_fineweb100B", "sample-100BT")
}
local_dir, remote_name = directories[(آرگومانها.type, آرگومانها.نسخه)]

# create the cache the local directory if it doesn't exist yet
داده_CACHE_DIR = os.مسیر.join(os.مسیر.dirname(__file__), local_dir)
os.makedirs(داده_CACHE_DIR, exist_ok=درست)

# download the dataset
إذا آرگومانها.type == "classic":
    fw = بارگیری_dataset("HuggingFaceFW/fineweb", name=remote_name, تقسیم="train")
    name = "fineweb"
وإلاإذا آرگومانها.type =="edu":
    fw = بارگیری_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, تقسیم="train")
    name = "edu_fineweb"

تعریف tokenize_llama(doc):
    # tokenizes a single document and returns a numpy array of uint32 tokens
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
    encode = مجهول s: tokenizer.encode(s, جمع_کن_special_نشانهها=خطأ, verbose=خطأ, تقسیم_special_نشانهها=درست)
    eot = tokenizer.encode('')[0] # by default the tokenizer adds the EOT token (128000)
    نشانهها = [eot] # the special <|endoftext|> token delimits all documents
    نشانهها.extend(encode(doc["text"]))
    نشانهها_np = np.array(نشانهها)
    أكد (0 <= نشانهها_np).all() و (نشانهها_np < 2**32).all(), "token dictionary too large for uint32"
    نشانهها_np_uint = نشانهها_np.astype(np.uint32)
    أرجع نشانهها_np_uint

تعریف tokenize_gpt2(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    enc = tiktoken.get_encoding("gpt2")
    encode = مجهول s: enc.encode_ordinary(s)
    eot = enc.special_نشانهها['<|endoftext|>'] # end of text token
    نشانهها = [eot] # the special <|endoftext|> token delimits all documents
    نشانهها.extend(encode(doc["text"]))
    نشانهها_np = np.array(نشانهها)
    أكد (0 <= نشانهها_np).all() و (نشانهها_np < 2**16).all(), "token dictionary too large for uint16"
    نشانهها_np_uint = نشانهها_np.astype(np.uint16)
    أرجع نشانهها_np_uint

token_dtype = {
    "gpt-2": np.uint16,
    "llama-3": np.uint32
}[آرگومانها.model_desc]

# tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
nprocs = max(1, os.cpu_شمار() - 2) # don't hog the entire system
با mp.Pool(nprocs) كـ pool:
    shard_شاخص = 0
    # preallocate buffer to hold current shard
    all_نشانهها_np = np.خالی((آرگومانها.shard_اندازه,), dtype=token_dtype)
    token_شمار = 0
    پیشرفت_bar = هیچ

    tokenize = مجهول x: هیچ
    إذا آرگومانها.model_desc == "gpt-2":
        tokenize = tokenize_gpt2
    وإلاإذا آرگومانها.model_desc == "llama-3":
        tokenize = tokenize_llama
    وإلا:
        ارفع مقدار_Error(f"unknown model {args.model_desc}")

    لكل نشانهها في pool.imap(tokenize, fw, chunksize=16):

        # is there enough space in the current shard for the new tokens?
        إذا token_شمار + طول(نشانهها) < آرگومانها.shard_اندازه:
            # simply append tokens to current shard
            all_نشانهها_np[token_شمار:token_شمار+طول(نشانهها)] = نشانهها
            token_شمار += طول(نشانهها)
            # update progress bar
            إذا پیشرفت_bar است هیچ:
                پیشرفت_bar = tqdm(مجموع=آرگومانها.shard_اندازه, unit="tokens", desc=f"Shard {shard_index}")
            پیشرفت_bar.update(طول(نشانهها))
        وإلا:
            # write the current shard and start a new one
            تقسیم = "val" إذا shard_شاخص == 0 وإلا "train"
            filename = os.مسیر.join(داده_CACHE_DIR, f"{name}_{split}_{shard_index:06d}.bin")
            # split the document into whatever fits in this shard; the remainder goes to next one
            remainder = آرگومانها.shard_اندازه - token_شمار
            پیشرفت_bar.update(remainder)
            all_نشانهها_np[token_شمار:token_شمار+remainder] = نشانهها[:remainder]
            write_datafile(filename, all_نشانهها_np.tolist(), آرگومانها.model_desc)
            shard_شاخص += 1
            پیشرفت_bar = هیچ
            # populate the next shard with the leftovers of the current doc
            all_نشانهها_np[0:طول(نشانهها)-remainder] = نشانهها[remainder:]
            token_شمار = طول(نشانهها)-remainder

    # write any remaining tokens as the last shard
    إذا token_شمار != 0:
        تقسیم = "val" إذا shard_شاخص == 0 وإلا "train"
        filename = os.مسیر.join(داده_CACHE_DIR, f"{name}_{split}_{shard_index:06d}.bin")
        write_datafile(filename, (all_نشانهها_np[:token_شمار]).tolist(), آرگومانها.model_desc)
