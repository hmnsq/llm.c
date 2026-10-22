"""
Downloads and tokenizes the TinyShakespeare dataset.
- The download is from Github.
- The tokenization is GPT-2 tokenizer with tiktoken

The output is written to a newly created tinyshakespeare/ folder.
The script prints:

For GPT-2:
$ python dev/data/tinyshakespeare.py --model=gpt-2
writing 32,768 tokens to /home/ubuntu/llm.c/dev/data/tinyshakespeare/tiny_shakespeare_val.bin (66,560 bytes) in the gpt-2 format
writing 305,260 tokens to /home/ubuntu/llm.c/dev/data/tinyshakespeare/tiny_shakespeare_train.bin (611,544 bytes) in the gpt-2 format

For LLaMA 3:
$ python dev/data/tinyshakespeare.py --model=llama-3
writing 32,768 tokens to /home/ubuntu/llm.c/dev/data/tinyshakespeare/tiny_shakespeare_val.bin (132,096 bytes) in the llama-3 format
writing 276,224 tokens to /home/ubuntu/llm.c/dev/data/tinyshakespeare/tiny_shakespeare_train.bin (1,105,920 bytes) in the llama-3 format

And runs in a few seconds depending on your internet
connection and computer. The .bin files are raw byte
streams of uint16 (gpt-2) or uint32 (llama) numbers indicating the token ids.
"""

ਆਯਾਤ argparse
ਆਯਾਤ os

ਆਯਾਤ tiktoken
ਤਃ transformers ਆਯਾਤ AutoTokenizer

ਤਃ ਅੰਕੜੇ_common ਆਯਾਤ download_file, write_datafile

# -----------------------------------------------------------------------------
ਅੰਕੜੇ_CACHE_DIR = os.ਰਸਤਾ.join(os.ਰਸਤਾ.dirname(__file__), "tinyshakespeare")

ਪਰਿਭਾਸ਼ਾ download():
    """Downloads the TinyShakespeare dataset to DATA_CACHE_DIR"""
    os.makedirs(ਅੰਕੜੇ_CACHE_DIR, exist_ok=ਸੱਚ)
    # download the TinyShakespeare dataset, unless it's already downloaded
    ਅੰਕੜੇ_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    ਅੰਕੜੇ_filename = os.ਰਸਤਾ.join(ਅੰਕੜੇ_CACHE_DIR, "tiny_shakespeare.txt")
    ਜੇ ਨਹੀਂ os.ਰਸਤਾ.exists(ਅੰਕੜੇ_filename):
        ਛਾਪੋ(f"Downloading {data_url} to {data_filename}...")
        download_file(ਅੰਕੜੇ_url, ਅੰਕੜੇ_filename)
    ਨਹੀਂਤਾਂ:
        ਛਾਪੋ(f"{data_filename} already exists, skipping download...")

ਪਰਿਭਾਸ਼ਾ tokenize(model_desc):
    ਜੇ model_desc == "gpt-2":
        enc = tiktoken.get_encoding("gpt2")
        encode = ਅਨਾਮ s: enc.encode_ordinary(s)
        eot = enc.special_ਟੋਕਨ['<|endoftext|>'] # end of text token
    ਜਾਂਜੇ model_desc == "llama-3":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
        encode = ਅਨਾਮ s: tokenizer.encode(s, ਜੋੜੋ_special_ਟੋਕਨ=ਝੂਠ, verbose=ਝੂਠ, ਵੰਡੋ_special_ਟੋਕਨ=ਸੱਚ)
        eot = tokenizer.encode('')[0] # by default the tokenizer adds the EOT token (128000)
    ਨਹੀਂਤਾਂ:
        ਉਠਾਓ ਮੁੱਲ_Error(f"unknown model descriptor {model_desc}")
    ਅੰਕੜੇ_filename = os.ਰਸਤਾ.join(ਅੰਕੜੇ_CACHE_DIR, "tiny_shakespeare.txt")
    ਪਾਠ = open(ਅੰਕੜੇ_filename, 'r').ਪਠ()
    # let's treat every individual chunk of text as a separate "document"
    sections = ਪਾਠ.ਵੰਡੋ("\n\n")
    ਟੋਕਨ = []
    ਲਈ i, s ਵਿੱਚ enumerate(sections):
        ਟੋਕਨ.append(eot)
        # there was a mild bug where I originally intended to remove \n\n, but instead just added
        # the EOT right after each \n\n, so I'm keeping that behavior for backwards compatibility
        # therefore we have to here add an extra \n\n at the end of each section, except the last
        spad = s + "\n\n" ਜੇ i != ਲੰਬਾਈ(sections) - 1 ਨਹੀਂਤਾਂ s
        ਟੋਕਨ.extend(encode(spad))
    # let's take the first 32,768 tokens as the validation split (~10%)
    val_ਟੋਕਨ = ਟੋਕਨ[:32768]
    train_ਟੋਕਨ = ਟੋਕਨ[32768:]
    # save to file
    val_filename = os.ਰਸਤਾ.join(ਅੰਕੜੇ_CACHE_DIR, "tiny_shakespeare_val.bin")
    train_filename = os.ਰਸਤਾ.join(ਅੰਕੜੇ_CACHE_DIR, "tiny_shakespeare_train.bin")
    write_datafile(val_filename, val_ਟੋਕਨ, model_desc)
    write_datafile(train_filename, train_ਟੋਕਨ, model_desc)

ਜੇ __name__ == "__main__":
    parser = argparse.ਦਲੀਲ_Parser(description="Tiny Shakespeare dataset preprocessing")
    parser.ਜੋੜੋ_ਦਲੀਲ("-m", "--model_desc", type=str, ਮੂਲ="gpt-2", choices=["gpt-2", "llama-3"], help="Model type, gpt-2|llama-3")
    ਦਲੀਲਾਂ = parser.parse_ਦਲੀਲਾਂ()
    download()
    tokenize(ਦਲੀਲਾਂ.model_desc)
