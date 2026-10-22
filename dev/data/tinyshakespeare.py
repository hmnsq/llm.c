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

ಆಮದು argparse
ಆಮದು os

ಆಮದು tiktoken
ತಃ transformers ಆಮದು AutoTokenizer

ತಃ ದತ್ತಾಂಶ_common ಆಮದು download_file, write_datafile

# -----------------------------------------------------------------------------
ದತ್ತಾಂಶ_CACHE_DIR = os.ಮಾರ್ಗ.join(os.ಮಾರ್ಗ.dirname(__file__), "tinyshakespeare")

ವ್ಯಾಖ್ಯಾನ download():
    """Downloads the TinyShakespeare dataset to DATA_CACHE_DIR"""
    os.makedirs(ದತ್ತಾಂಶ_CACHE_DIR, exist_ok=ನಿಜ)
    # download the TinyShakespeare dataset, unless it's already downloaded
    ದತ್ತಾಂಶ_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    ದತ್ತಾಂಶ_filename = os.ಮಾರ್ಗ.join(ದತ್ತಾಂಶ_CACHE_DIR, "tiny_shakespeare.txt")
    ಆದರೆ ಅಲ್ಲ os.ಮಾರ್ಗ.exists(ದತ್ತಾಂಶ_filename):
        ಮುದ್ರಿಸು(f"Downloading {data_url} to {data_filename}...")
        download_file(ದತ್ತಾಂಶ_url, ದತ್ತಾಂಶ_filename)
    ಇಲ್ಲದಿದ್ದರೆ:
        ಮುದ್ರಿಸು(f"{data_filename} already exists, skipping download...")

ವ್ಯಾಖ್ಯಾನ tokenize(model_desc):
    ಆದರೆ model_desc == "gpt-2":
        enc = tiktoken.get_encoding("gpt2")
        encode = ಅನಾಮಿಕ s: enc.encode_ordinary(s)
        eot = enc.special_ಟೋಕನುಗಳು['<|endoftext|>'] # end of text token
    ಅಥವಾಆದರೆ model_desc == "llama-3":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
        encode = ಅನಾಮಿಕ s: tokenizer.encode(s, ಕೂಡಿಸು_special_ಟೋಕನುಗಳು=ಸುಳ್ಳು, verbose=ಸುಳ್ಳು, ವಿಭಜಿಸು_special_ಟೋಕನುಗಳು=ನಿಜ)
        eot = tokenizer.encode('')[0] # by default the tokenizer adds the EOT token (128000)
    ಇಲ್ಲದಿದ್ದರೆ:
        ಎತ್ತು ಮೌಲ್ಯ_Error(f"unknown model descriptor {model_desc}")
    ದತ್ತಾಂಶ_filename = os.ಮಾರ್ಗ.join(ದತ್ತಾಂಶ_CACHE_DIR, "tiny_shakespeare.txt")
    ಪಠ್ಯ = open(ದತ್ತಾಂಶ_filename, 'r').ಪಠ()
    # let's treat every individual chunk of text as a separate "document"
    sections = ಪಠ್ಯ.ವಿಭಜಿಸು("\n\n")
    ಟೋಕನುಗಳು = []
    ಗಾಗಿ i, s ಒಳಗೆ enumerate(sections):
        ಟೋಕನುಗಳು.append(eot)
        # there was a mild bug where I originally intended to remove \n\n, but instead just added
        # the EOT right after each \n\n, so I'm keeping that behavior for backwards compatibility
        # therefore we have to here add an extra \n\n at the end of each section, except the last
        spad = s + "\n\n" ಆದರೆ i != ಉದ್ದ(sections) - 1 ಇಲ್ಲದಿದ್ದರೆ s
        ಟೋಕನುಗಳು.extend(encode(spad))
    # let's take the first 32,768 tokens as the validation split (~10%)
    val_ಟೋಕನುಗಳು = ಟೋಕನುಗಳು[:32768]
    train_ಟೋಕನುಗಳು = ಟೋಕನುಗಳು[32768:]
    # save to file
    val_filename = os.ಮಾರ್ಗ.join(ದತ್ತಾಂಶ_CACHE_DIR, "tiny_shakespeare_val.bin")
    train_filename = os.ಮಾರ್ಗ.join(ದತ್ತಾಂಶ_CACHE_DIR, "tiny_shakespeare_train.bin")
    write_datafile(val_filename, val_ಟೋಕನುಗಳು, model_desc)
    write_datafile(train_filename, train_ಟೋಕನುಗಳು, model_desc)

ಆದರೆ __name__ == "__main__":
    parser = argparse.ವಾದ_Parser(description="Tiny Shakespeare dataset preprocessing")
    parser.ಕೂಡಿಸು_ವಾದ("-m", "--model_desc", type=str, ಪೂರ್ವನಿಯೋಜಿತ="gpt-2", choices=["gpt-2", "llama-3"], help="Model type, gpt-2|llama-3")
    ವಾದಗಳು = parser.parse_ವಾದಗಳು()
    download()
    tokenize(ವಾದಗಳು.model_desc)
