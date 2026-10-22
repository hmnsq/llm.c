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
من datasets استورد بارول_dataset
من tqdm استورد tqdm

من transformers استورد AutoTokenizer


من معلومات_common استورد write_datafile
# ------------------------------------------

parser = argparse.دلیل_Parser(description="FineWeb and Edu-FineWeb dataset preprocessing")
parser.جمع_کړه_دلیل("-t", "--type", type=str, افتراضي="classic", help="Fineweb type, edu|classic")
parser.جمع_کړه_دلیل("-v", "--version", type=str, افتراضي="10B", help="Fineweb data sample size, 10B|100B")
parser.جمع_کړه_دلیل("-m", "--model_desc", type=str, افتراضي="gpt-2", help="Model descriptor, gpt-2|llama-3")
parser.جمع_کړه_دلیل("-s", "--shard_size", type=صحيح, افتراضي=10**8, help="Size of each data shard in the output .bin files, in tokens")
دلیلونه = parser.parse_دلیلونه()

# FineWeb has a few possible subsamples available
أكد دلیلونه.نسخه في {"10B", "100B"}, "version must be one of: 10B, 100B"
أكد دلیلونه.type في {"edu", "classic"}, "type must be one of: edu, classic"
directories = {
    ("classic", "10B"): ("fineweb10B", "sample-10BT"),
    ("classic", "100B"): ("fineweb100B", "sample-100BT"),
    ("edu", "10B"): ("edu_fineweb10B", "sample-10BT"),
    ("edu", "100B"): ("edu_fineweb100B", "sample-100BT")
}
local_dir, remote_name = directories[(دلیلونه.type, دلیلونه.نسخه)]

# create the cache the local directory if it doesn't exist yet
معلومات_CACHE_DIR = os.لار.join(os.لار.dirname(__file__), local_dir)
os.makedirs(معلومات_CACHE_DIR, exist_ok=ریښتیا)

# download the dataset
إذا دلیلونه.type == "classic":
    fw = بارول_dataset("HuggingFaceFW/fineweb", name=remote_name, وېشل="train")
    name = "fineweb"
وإلاإذا دلیلونه.type =="edu":
    fw = بارول_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, وېشل="train")
    name = "edu_fineweb"

تعریف tokenize_llama(doc):
    # tokenizes a single document and returns a numpy array of uint32 tokens
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
    encode = مجهول s: tokenizer.encode(s, جمع_کړه_special_نښې=خطأ, verbose=خطأ, وېشل_special_نښې=ریښتیا)
    eot = tokenizer.encode('')[0] # by default the tokenizer adds the EOT token (128000)
    نښې = [eot] # the special <|endoftext|> token delimits all documents
    نښې.extend(encode(doc["text"]))
    نښې_np = np.array(نښې)
    أكد (0 <= نښې_np).all() و (نښې_np < 2**32).all(), "token dictionary too large for uint32"
    نښې_np_uint = نښې_np.astype(np.uint32)
    أرجع نښې_np_uint

تعریف tokenize_gpt2(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    enc = tiktoken.get_encoding("gpt2")
    encode = مجهول s: enc.encode_ordinary(s)
    eot = enc.special_نښې['<|endoftext|>'] # end of text token
    نښې = [eot] # the special <|endoftext|> token delimits all documents
    نښې.extend(encode(doc["text"]))
    نښې_np = np.array(نښې)
    أكد (0 <= نښې_np).all() و (نښې_np < 2**16).all(), "token dictionary too large for uint16"
    نښې_np_uint = نښې_np.astype(np.uint16)
    أرجع نښې_np_uint

token_dtype = {
    "gpt-2": np.uint16,
    "llama-3": np.uint32
}[دلیلونه.model_desc]

# tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
nprocs = max(1, os.cpu_شمېر() - 2) # don't hog the entire system
سره mp.Pool(nprocs) كـ pool:
    shard_شاخص = 0
    # preallocate buffer to hold current shard
    all_نښې_np = np.تشه((دلیلونه.shard_اندازه,), dtype=token_dtype)
    token_شمېر = 0
    پرمختګ_bar = هیڅ

    tokenize = مجهول x: هیڅ
    إذا دلیلونه.model_desc == "gpt-2":
        tokenize = tokenize_gpt2
    وإلاإذا دلیلونه.model_desc == "llama-3":
        tokenize = tokenize_llama
    وإلا:
        ارفع ارزښت_Error(f"unknown model {args.model_desc}")

    لكل نښې في pool.imap(tokenize, fw, chunksize=16):

        # is there enough space in the current shard for the new tokens?
        إذا token_شمېر + اوږدوالی(نښې) < دلیلونه.shard_اندازه:
            # simply append tokens to current shard
            all_نښې_np[token_شمېر:token_شمېر+اوږدوالی(نښې)] = نښې
            token_شمېر += اوږدوالی(نښې)
            # update progress bar
            إذا پرمختګ_bar دی هیڅ:
                پرمختګ_bar = tqdm(ټول=دلیلونه.shard_اندازه, unit="tokens", desc=f"Shard {shard_index}")
            پرمختګ_bar.update(اوږدوالی(نښې))
        وإلا:
            # write the current shard and start a new one
            وېشل = "val" إذا shard_شاخص == 0 وإلا "train"
            filename = os.لار.join(معلومات_CACHE_DIR, f"{name}_{split}_{shard_index:06d}.bin")
            # split the document into whatever fits in this shard; the remainder goes to next one
            remainder = دلیلونه.shard_اندازه - token_شمېر
            پرمختګ_bar.update(remainder)
            all_نښې_np[token_شمېر:token_شمېر+remainder] = نښې[:remainder]
            write_datafile(filename, all_نښې_np.tolist(), دلیلونه.model_desc)
            shard_شاخص += 1
            پرمختګ_bar = هیڅ
            # populate the next shard with the leftovers of the current doc
            all_نښې_np[0:اوږدوالی(نښې)-remainder] = نښې[remainder:]
            token_شمېر = اوږدوالی(نښې)-remainder

    # write any remaining tokens as the last shard
    إذا token_شمېر != 0:
        وېشل = "val" إذا shard_شاخص == 0 وإلا "train"
        filename = os.لار.join(معلومات_CACHE_DIR, f"{name}_{split}_{shard_index:06d}.bin")
        write_datafile(filename, (all_نښې_np[:token_شمېر]).tolist(), دلیلونه.model_desc)
