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
ಆಮದು os
ಆಮದು argparse
ಆಮದು multiprocessing ಇವ mp

ಆಮದು numpy ಇವ np
ಆಮದು tiktoken
ತಃ datasets ಆಮದು ತುಂಬು_dataset
ತಃ tqdm ಆಮದು tqdm

ತಃ transformers ಆಮದು AutoTokenizer


ತಃ ದತ್ತಾಂಶ_common ಆಮದು write_datafile
# ------------------------------------------

parser = argparse.ವಾದ_Parser(description="FineWeb and Edu-FineWeb dataset preprocessing")
parser.ಕೂಡಿಸು_ವಾದ("-t", "--type", type=str, ಪೂರ್ವನಿಯೋಜಿತ="classic", help="Fineweb type, edu|classic")
parser.ಕೂಡಿಸು_ವಾದ("-v", "--version", type=str, ಪೂರ್ವನಿಯೋಜಿತ="10B", help="Fineweb data sample size, 10B|100B")
parser.ಕೂಡಿಸು_ವಾದ("-m", "--model_desc", type=str, ಪೂರ್ವನಿಯೋಜಿತ="gpt-2", help="Model descriptor, gpt-2|llama-3")
parser.ಕೂಡಿಸು_ವಾದ("-s", "--shard_size", type=ಪೂರ್ಣಾಂಕ, ಪೂರ್ವನಿಯೋಜಿತ=10**8, help="Size of each data shard in the output .bin files, in tokens")
ವಾದಗಳು = parser.parse_ವಾದಗಳು()

# FineWeb has a few possible subsamples available
ಪ್ರತಿಜ್ಞಾ ವಾದಗಳು.ಆವೃತ್ತಿ ಒಳಗೆ {"10B", "100B"}, "version must be one of: 10B, 100B"
ಪ್ರತಿಜ್ಞಾ ವಾದಗಳು.type ಒಳಗೆ {"edu", "classic"}, "type must be one of: edu, classic"
directories = {
    ("classic", "10B"): ("fineweb10B", "sample-10BT"),
    ("classic", "100B"): ("fineweb100B", "sample-100BT"),
    ("edu", "10B"): ("edu_fineweb10B", "sample-10BT"),
    ("edu", "100B"): ("edu_fineweb100B", "sample-100BT")
}
local_dir, remote_name = directories[(ವಾದಗಳು.type, ವಾದಗಳು.ಆವೃತ್ತಿ)]

# create the cache the local directory if it doesn't exist yet
ದತ್ತಾಂಶ_CACHE_DIR = os.ಮಾರ್ಗ.join(os.ಮಾರ್ಗ.dirname(__file__), local_dir)
os.makedirs(ದತ್ತಾಂಶ_CACHE_DIR, exist_ok=ನಿಜ)

# download the dataset
ಆದರೆ ವಾದಗಳು.type == "classic":
    fw = ತುಂಬು_dataset("HuggingFaceFW/fineweb", name=remote_name, ವಿಭಜಿಸು="train")
    name = "fineweb"
ಅಥವಾಆದರೆ ವಾದಗಳು.type =="edu":
    fw = ತುಂಬು_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, ವಿಭಜಿಸು="train")
    name = "edu_fineweb"

ವ್ಯಾಖ್ಯಾನ tokenize_llama(doc):
    # tokenizes a single document and returns a numpy array of uint32 tokens
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
    encode = ಅನಾಮಿಕ s: tokenizer.encode(s, ಕೂಡಿಸು_special_ಟೋಕನುಗಳು=ಸುಳ್ಳು, verbose=ಸುಳ್ಳು, ವಿಭಜಿಸು_special_ಟೋಕನುಗಳು=ನಿಜ)
    eot = tokenizer.encode('')[0] # by default the tokenizer adds the EOT token (128000)
    ಟೋಕನುಗಳು = [eot] # the special <|endoftext|> token delimits all documents
    ಟೋಕನುಗಳು.extend(encode(doc["text"]))
    ಟೋಕನುಗಳು_np = np.array(ಟೋಕನುಗಳು)
    ಪ್ರತಿಜ್ಞಾ (0 <= ಟೋಕನುಗಳು_np).all() ಮತ್ತು (ಟೋಕನುಗಳು_np < 2**32).all(), "token dictionary too large for uint32"
    ಟೋಕನುಗಳು_np_uint = ಟೋಕನುಗಳು_np.astype(np.uint32)
    ಹಿಂತಿರುಗಿ ಟೋಕನುಗಳು_np_uint

ವ್ಯಾಖ್ಯಾನ tokenize_gpt2(doc):
    # tokenizes a single document and returns a numpy array of uint16 tokens
    enc = tiktoken.get_encoding("gpt2")
    encode = ಅನಾಮಿಕ s: enc.encode_ordinary(s)
    eot = enc.special_ಟೋಕನುಗಳು['<|endoftext|>'] # end of text token
    ಟೋಕನುಗಳು = [eot] # the special <|endoftext|> token delimits all documents
    ಟೋಕನುಗಳು.extend(encode(doc["text"]))
    ಟೋಕನುಗಳು_np = np.array(ಟೋಕನುಗಳು)
    ಪ್ರತಿಜ್ಞಾ (0 <= ಟೋಕನುಗಳು_np).all() ಮತ್ತು (ಟೋಕನುಗಳು_np < 2**16).all(), "token dictionary too large for uint16"
    ಟೋಕನುಗಳು_np_uint = ಟೋಕನುಗಳು_np.astype(np.uint16)
    ಹಿಂತಿರುಗಿ ಟೋಕನುಗಳು_np_uint

token_dtype = {
    "gpt-2": np.uint16,
    "llama-3": np.uint32
}[ವಾದಗಳು.model_desc]

# tokenize all documents and write output shards, each of shard_size tokens (last shard has remainder)
nprocs = max(1, os.cpu_ಎಣಿಕೆ() - 2) # don't hog the entire system
ಜೊತೆ mp.Pool(nprocs) ಇವ pool:
    shard_ಸೂಚ್ಯಂಕ = 0
    # preallocate buffer to hold current shard
    all_ಟೋಕನುಗಳು_np = np.ಖಾಲಿಗಳು((ವಾದಗಳು.shard_ಗಾತ್ರ,), dtype=token_dtype)
    token_ಎಣಿಕೆ = 0
    ಪ್ರಗತಿ_bar = ಏನೂಇಲ್ಲ

    tokenize = ಅನಾಮಿಕ x: ಏನೂಇಲ್ಲ
    ಆದರೆ ವಾದಗಳು.model_desc == "gpt-2":
        tokenize = tokenize_gpt2
    ಅಥವಾಆದರೆ ವಾದಗಳು.model_desc == "llama-3":
        tokenize = tokenize_llama
    ಇಲ್ಲದಿದ್ದರೆ:
        ಎತ್ತು ಮೌಲ್ಯ_Error(f"unknown model {args.model_desc}")

    ಗಾಗಿ ಟೋಕನುಗಳು ಒಳಗೆ pool.imap(tokenize, fw, chunksize=16):

        # is there enough space in the current shard for the new tokens?
        ಆದರೆ token_ಎಣಿಕೆ + ಉದ್ದ(ಟೋಕನುಗಳು) < ವಾದಗಳು.shard_ಗಾತ್ರ:
            # simply append tokens to current shard
            all_ಟೋಕನುಗಳು_np[token_ಎಣಿಕೆ:token_ಎಣಿಕೆ+ಉದ್ದ(ಟೋಕನುಗಳು)] = ಟೋಕನುಗಳು
            token_ಎಣಿಕೆ += ಉದ್ದ(ಟೋಕನುಗಳು)
            # update progress bar
            ಆದರೆ ಪ್ರಗತಿ_bar ಆಗಿರುವ ಏನೂಇಲ್ಲ:
                ಪ್ರಗತಿ_bar = tqdm(ಒಟ್ಟು=ವಾದಗಳು.shard_ಗಾತ್ರ, unit="tokens", desc=f"Shard {shard_index}")
            ಪ್ರಗತಿ_bar.update(ಉದ್ದ(ಟೋಕನುಗಳು))
        ಇಲ್ಲದಿದ್ದರೆ:
            # write the current shard and start a new one
            ವಿಭಜಿಸು = "val" ಆದರೆ shard_ಸೂಚ್ಯಂಕ == 0 ಇಲ್ಲದಿದ್ದರೆ "train"
            filename = os.ಮಾರ್ಗ.join(ದತ್ತಾಂಶ_CACHE_DIR, f"{name}_{split}_{shard_index:06d}.bin")
            # split the document into whatever fits in this shard; the remainder goes to next one
            remainder = ವಾದಗಳು.shard_ಗಾತ್ರ - token_ಎಣಿಕೆ
            ಪ್ರಗತಿ_bar.update(remainder)
            all_ಟೋಕನುಗಳು_np[token_ಎಣಿಕೆ:token_ಎಣಿಕೆ+remainder] = ಟೋಕನುಗಳು[:remainder]
            write_datafile(filename, all_ಟೋಕನುಗಳು_np.tolist(), ವಾದಗಳು.model_desc)
            shard_ಸೂಚ್ಯಂಕ += 1
            ಪ್ರಗತಿ_bar = ಏನೂಇಲ್ಲ
            # populate the next shard with the leftovers of the current doc
            all_ಟೋಕನುಗಳು_np[0:ಉದ್ದ(ಟೋಕನುಗಳು)-remainder] = ಟೋಕನುಗಳು[remainder:]
            token_ಎಣಿಕೆ = ಉದ್ದ(ಟೋಕನುಗಳು)-remainder

    # write any remaining tokens as the last shard
    ಆದರೆ token_ಎಣಿಕೆ != 0:
        ವಿಭಜಿಸು = "val" ಆದರೆ shard_ಸೂಚ್ಯಂಕ == 0 ಇಲ್ಲದಿದ್ದರೆ "train"
        filename = os.ಮಾರ್ಗ.join(ದತ್ತಾಂಶ_CACHE_DIR, f"{name}_{split}_{shard_index:06d}.bin")
        write_datafile(filename, (all_ಟೋಕನುಗಳು_np[:token_ಎಣಿಕೆ]).tolist(), ವಾದಗಳು.model_desc)
