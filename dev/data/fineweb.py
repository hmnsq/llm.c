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
దిగుమతి os
దిగుమతి argparse
దిగుమతి multiprocessing ఇవ mp

దిగుమతి numpy ఇవ np
దిగుమతి tiktoken
తః datasets దిగుమతి ఎక్కించు_dataset
తః tqdm దిగుమతి tqdm

తః transformers దిగుమతి AutoTokenizer


తః దత్తాంశం_common దిగుమతి write_datafile
# ------------------------------------------

parser = argparse.వాదం_Parser(description="FineWeb and Edu-FineWeb dataset preprocessing")
parser.కూడు_వాదం("-t", "--type", type=str, అప్రమేయ="classic", help="Fineweb type, edu|classic")
parser.కూడు_వాదం("-v", "--version", type=str, అప్రమేయ="10B", help="Fineweb data sample size, 10B|100B")
parser.కూడు_వాదం("-m", "--model_desc", type=str, అప్రమేయ="gpt-2", help="Model descriptor, gpt-2|llama-3")
parser.కూడు_వాదం("-s", "--shard_size", type=పూర్ణాంకం, అప్రమేయ=10**8, help="Size of each data shard in the output .bin files, in tokens")
వాదనలు = parser.parse_వాదనలు()

# FineWeb has a few possible subsamples available
ప్రతిజ్ఞా వాదనలు.సంస్కరణ లో {"10B", "100B"}, "version must be one of: 10B, 100B"
ప్రతిజ్ఞా వాదనలు.type లో {"edu", "classic"}, "type must be one of: edu, classic"
directories = {
    ("classic", "10B"): ("fineweb10B", "sample-10BT"),
    ("classic", "100B"): ("fineweb100B", "sample-100BT"),
    ("edu", "10B"): ("edu_fineweb10B", "sample-10BT"),
    ("edu", "100B"): ("edu_fineweb100B", "sample-100BT")
}
local_dir, remote_name = directories[(వాదనలు.type, వాదనలు.సంస్కరణ)]

# create the cache the local directory if it doesn't exist yet
దత్తాంశం_CACHE_DIR = os.మార్గం.join(os.మార్గం.dirname(__file__), local_dir)
os.makedirs(దత్తాంశం_CACHE_DIR, exist_ok=నిజం)

# download the dataset
ఒకవేళ వాదనలు.type == "classic":
    fw = ఎక్కించు_dataset("HuggingFaceFW/fineweb", name=remote_name, విభజించు="train")
    name = "fineweb"
లేదావేళ వాదనలు.type =="edu":
    fw = ఎక్కించు_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, విభజించు="train")
    name = "edu_fineweb"

నిర్వచనం tokenize_llama(doc):
    # tokenizes a single document and returns a numpy array of uint32 tokens
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
    encode = అజ్ఞాత s: tokenizer.encode(s, కూడు_special_టోకెన్లు=అబద్ధం, verbose=అబద్ధం, విభజించు_special_టోకెన్లు=నిజం)
    eot = tokenizer.encode('')[0] # by default the tokenizer adds the EOT token (128000)
    టోకెన్లు = [eot] # the special <|endoftext|> token delimits all documents
    టోకెన్లు.extend(encode(doc["text"]))
    టోకెన్లు_np = np.array(టోకెన్లు)
    ప్రతిజ్ఞా (0 <= టోకెన్లు_np).all() మరియు (టోకెన్లు_np < 2**32).all(), "token dictionary too large for uint32"
    టోకెన్లు_np_uint = టోకెన్లు_np.astype(np.uint32)
    తిరిగి టోకెన్లు_np_uint

నిర్వచనం tokenize_gpt2(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    enc = tiktoken.get_encoding("gpt2")
    encode = అజ్ఞాత s: enc.encode_ordinary(s)
    eot = enc.special_టోకెన్లు['<|endoftext|>'] # end of text token
    టోకెన్లు = [eot] # the special <|endoftext|> token delimits all documents
    టోకెన్లు.extend(encode(doc["text"]))
    టోకెన్లు_np = np.array(టోకెన్లు)
    ప్రతిజ్ఞా (0 <= టోకెన్లు_np).all() మరియు (టోకెన్లు_np < 2**16).all(), "token dictionary too large for uint16"
    టోకెన్లు_np_uint = టోకెన్లు_np.astype(np.uint16)
    తిరిగి టోకెన్లు_np_uint

token_dtype = {
    "gpt-2": np.uint16,
    "llama-3": np.uint32
}[వాదనలు.model_desc]

# tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
nprocs = max(1, os.cpu_లెక్క() - 2) # don't hog the entire system
తోడుగా mp.Pool(nprocs) ఇవ pool:
    shard_సూచీ = 0
    # preallocate buffer to hold current shard
    all_టోకెన్లు_np = np.ఖాళీలు((వాదనలు.shard_కొలత,), dtype=token_dtype)
    token_లెక్క = 0
    పురోగతి_bar = ఏమీలేదు

    tokenize = అజ్ఞాత x: ఏమీలేదు
    ఒకవేళ వాదనలు.model_desc == "gpt-2":
        tokenize = tokenize_gpt2
    లేదావేళ వాదనలు.model_desc == "llama-3":
        tokenize = tokenize_llama
    లేకపోతే:
        లేపు విలువ_Error(f"unknown model {args.model_desc}")

    కొరకు టోకెన్లు లో pool.imap(tokenize, fw, chunksize=16):

        # is there enough space in the current shard for the new tokens?
        ఒకవేళ token_లెక్క + పొడవు(టోకెన్లు) < వాదనలు.shard_కొలత:
            # simply append tokens to current shard
            all_టోకెన్లు_np[token_లెక్క:token_లెక్క+పొడవు(టోకెన్లు)] = టోకెన్లు
            token_లెక్క += పొడవు(టోకెన్లు)
            # update progress bar
            ఒకవేళ పురోగతి_bar అయిన ఏమీలేదు:
                పురోగతి_bar = tqdm(మొత్తం=వాదనలు.shard_కొలత, unit="tokens", desc=f"Shard {shard_index}")
            పురోగతి_bar.update(పొడవు(టోకెన్లు))
        లేకపోతే:
            # write the current shard and start a new one
            విభజించు = "val" ఒకవేళ shard_సూచీ == 0 లేకపోతే "train"
            filename = os.మార్గం.join(దత్తాంశం_CACHE_DIR, f"{name}_{split}_{shard_index:06d}.bin")
            # split the document into whatever fits in this shard; the remainder goes to next one
            remainder = వాదనలు.shard_కొలత - token_లెక్క
            పురోగతి_bar.update(remainder)
            all_టోకెన్లు_np[token_లెక్క:token_లెక్క+remainder] = టోకెన్లు[:remainder]
            write_datafile(filename, all_టోకెన్లు_np.tolist(), వాదనలు.model_desc)
            shard_సూచీ += 1
            పురోగతి_bar = ఏమీలేదు
            # populate the next shard with the leftovers of the current doc
            all_టోకెన్లు_np[0:పొడవు(టోకెన్లు)-remainder] = టోకెన్లు[remainder:]
            token_లెక్క = పొడవు(టోకెన్లు)-remainder

    # write any remaining tokens as the last shard
    ఒకవేళ token_లెక్క != 0:
        విభజించు = "val" ఒకవేళ shard_సూచీ == 0 లేకపోతే "train"
        filename = os.మార్గం.join(దత్తాంశం_CACHE_DIR, f"{name}_{split}_{shard_index:06d}.bin")
        write_datafile(filename, (all_టోకెన్లు_np[:token_లెక్క]).tolist(), వాదనలు.model_desc)
