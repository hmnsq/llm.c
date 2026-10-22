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

దిగుమతి argparse
దిగుమతి os

దిగుమతి tiktoken
తః transformers దిగుమతి AutoTokenizer

తః దత్తాంశం_common దిగుమతి download_file, write_datafile

# -----------------------------------------------------------------------------
దత్తాంశం_CACHE_DIR = os.మార్గం.join(os.మార్గం.dirname(__file__), "tinyshakespeare")

నిర్వచనం download():
    """Downloads the TinyShakespeare dataset to DATA_CACHE_DIR"""
    os.makedirs(దత్తాంశం_CACHE_DIR, exist_ok=నిజం)
    # download the TinyShakespeare dataset, unless it's already downloaded
    దత్తాంశం_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    దత్తాంశం_filename = os.మార్గం.join(దత్తాంశం_CACHE_DIR, "tiny_shakespeare.txt")
    ఒకవేళ కాదు os.మార్గం.exists(దత్తాంశం_filename):
        ముద్రించు(f"Downloading {data_url} to {data_filename}...")
        download_file(దత్తాంశం_url, దత్తాంశం_filename)
    లేకపోతే:
        ముద్రించు(f"{data_filename} already exists, skipping download...")

నిర్వచనం tokenize(model_desc):
    ఒకవేళ model_desc == "gpt-2":
        enc = tiktoken.get_encoding("gpt2")
        encode = అజ్ఞాత s: enc.encode_ordinary(s)
        eot = enc.special_టోకెన్లు['<|endoftext|>'] # end of text token
    లేదావేళ model_desc == "llama-3":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
        encode = అజ్ఞాత s: tokenizer.encode(s, కూడు_special_టోకెన్లు=అబద్ధం, verbose=అబద్ధం, విభజించు_special_టోకెన్లు=నిజం)
        eot = tokenizer.encode('')[0] # by default the tokenizer adds the EOT token (128000)
    లేకపోతే:
        లేపు విలువ_Error(f"unknown model descriptor {model_desc}")
    దత్తాంశం_filename = os.మార్గం.join(దత్తాంశం_CACHE_DIR, "tiny_shakespeare.txt")
    పాఠ్యం = open(దత్తాంశం_filename, 'r').పఠ()
    # let's treat every individual chunk of text as a separate "document"
    sections = పాఠ్యం.విభజించు("\n\n")
    టోకెన్లు = []
    కొరకు i, s లో enumerate(sections):
        టోకెన్లు.append(eot)
        # there was a mild bug where I originally intended to remove \n\n, but instead just added
        # the EOT right after each \n\n, so I'm keeping that behavior for backwards compatibility
        # therefore we have to here add an extra \n\n at the end of each section, except the last
        spad = s + "\n\n" ఒకవేళ i != పొడవు(sections) - 1 లేకపోతే s
        టోకెన్లు.extend(encode(spad))
    # let's take the first 32,768 tokens as the validation split (~10%)
    val_టోకెన్లు = టోకెన్లు[:32768]
    train_టోకెన్లు = టోకెన్లు[32768:]
    # save to file
    val_filename = os.మార్గం.join(దత్తాంశం_CACHE_DIR, "tiny_shakespeare_val.bin")
    train_filename = os.మార్గం.join(దత్తాంశం_CACHE_DIR, "tiny_shakespeare_train.bin")
    write_datafile(val_filename, val_టోకెన్లు, model_desc)
    write_datafile(train_filename, train_టోకెన్లు, model_desc)

ఒకవేళ __name__ == "__main__":
    parser = argparse.వాదం_Parser(description="Tiny Shakespeare dataset preprocessing")
    parser.కూడు_వాదం("-m", "--model_desc", type=str, అప్రమేయ="gpt-2", choices=["gpt-2", "llama-3"], help="Model type, gpt-2|llama-3")
    వాదనలు = parser.parse_వాదనలు()
    download()
    tokenize(వాదనలు.model_desc)
