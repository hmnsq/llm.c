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

درآمد argparse
درآمد os

درآمد tiktoken
سے transformers درآمد AutoTokenizer

سے ڈیٹا_common درآمد download_file, write_datafile

# -----------------------------------------------------------------------------
ڈیٹا_CACHE_DIR = os.راستہ.join(os.راستہ.dirname(__file__), "tinyshakespeare")

تعریف download():
    """Downloads the TinyShakespeare dataset to DATA_CACHE_DIR"""
    os.makedirs(ڈیٹا_CACHE_DIR, exist_ok=سچ)
    # download the TinyShakespeare dataset, unless it's already downloaded
    ڈیٹا_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    ڈیٹا_filename = os.راستہ.join(ڈیٹا_CACHE_DIR, "tiny_shakespeare.txt")
    اگر نہیں os.راستہ.exists(ڈیٹا_filename):
        چھاپو(f"Downloading {data_url} to {data_filename}...")
        download_file(ڈیٹا_url, ڈیٹا_filename)
    ورنہ:
        چھاپو(f"{data_filename} already exists, skipping download...")

تعریف tokenize(model_desc):
    اگر model_desc == "gpt-2":
        enc = tiktoken.get_encoding("gpt2")
        encode = بےنام s: enc.encode_ordinary(s)
        eot = enc.special_علامات['<|endoftext|>'] # end of text token
    وگرنہ_اگر model_desc == "llama-3":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
        encode = بےنام s: tokenizer.encode(s, جوڑو_special_علامات=جھوٹ, verbose=جھوٹ, بانٹو_special_علامات=سچ)
        eot = tokenizer.encode('')[0] # by default the tokenizer adds the EOT token (128000)
    ورنہ:
        اٹھاؤ قدر_Error(f"unknown model descriptor {model_desc}")
    ڈیٹا_filename = os.راستہ.join(ڈیٹا_CACHE_DIR, "tiny_shakespeare.txt")
    متن = open(ڈیٹا_filename, 'r').اقرأ()
    # let's treat every individual chunk of text as a separate "document"
    sections = متن.بانٹو("\n\n")
    علامات = []
    برائے i, s میں enumerate(sections):
        علامات.append(eot)
        # there was a mild bug where I originally intended to remove \n\n, but instead just added
        # the EOT right after each \n\n, so I'm keeping that behavior for backwards compatibility
        # therefore we have to here add an extra \n\n at the end of each section, except the last
        spad = s + "\n\n" اگر i != لمبائی(sections) - 1 ورنہ s
        علامات.extend(encode(spad))
    # let's take the first 32,768 tokens as the validation split (~10%)
    val_علامات = علامات[:32768]
    train_علامات = علامات[32768:]
    # save to file
    val_filename = os.راستہ.join(ڈیٹا_CACHE_DIR, "tiny_shakespeare_val.bin")
    train_filename = os.راستہ.join(ڈیٹا_CACHE_DIR, "tiny_shakespeare_train.bin")
    write_datafile(val_filename, val_علامات, model_desc)
    write_datafile(train_filename, train_علامات, model_desc)

اگر __name__ == "__main__":
    parser = argparse.دلیل_Parser(description="Tiny Shakespeare dataset preprocessing")
    parser.جوڑو_دلیل("-m", "--model_desc", type=str, طےشدہ="gpt-2", choices=["gpt-2", "llama-3"], help="Model type, gpt-2|llama-3")
    دلائل = parser.parse_دلائل()
    download()
    tokenize(دلائل.model_desc)
