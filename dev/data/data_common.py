"""
Common utilities for the datasets
"""

ಆನಯನ requests
ತಃ tqdm ಆನಯನ tqdm
ಆನಯನ numpy ಇವ np


def download_file(url: str, fname: str, chunk_size=1024):
    """Helper function to download a file from a given url"""
    resp = requests.get(url, stream=ಸತ್ಯ)
    total = ಪೂರ್ಣಾಙ್ಕ(resp.headers.get("content-length", 0))
    with open(fname, "wb") ಇವ file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=ಸತ್ಯ,
        unit_ಭಾಜಕ=1024,
    ) ಇವ bar:
        ಪ್ರತಿ data ಅನ್ತಃ resp.iter_content(chunk_size=chunk_size):
            size = file.ಲಿಖ(data)
            bar.update(size)


HEADERS_INFO = {
    "gpt-2": {
        "magic": 20240520,
        "version": 1,
        "token_dtype": np.uint16,
    },
    "llama-3": {
        "magic": 20240801,
        "version": 7,
        "token_dtype": np.uint32,
    },
}

def write_datafile(filename, toks, model_desc="gpt-2"):
    """
    Saves token data as a .bin file, for reading in C.
    - First comes a header with 256 int32s
    - The tokens follow, each as uint16 (gpt-2) or uint32 (llama)
    """
    ಪ್ರತಿಜ್ಞಾ len(toks) < 2**31, "token count too large" # ~2.1B tokens
    ಪ್ರತಿಜ್ಞಾ model_desc ಅನ್ತಃ ["gpt-2", "llama-3"], f"unknown model descriptor {model_desc}"
    info = HEADERS_INFO[model_desc]
    # construct the header
    header = np.zeros(256, dtype=np.int32) # header is always 256 int32 values
    header[0] = info["magic"]
    header[1] = info["version"]
    header[2] = len(toks) # number of tokens after the 256*4 bytes of header
    # construct the data (numpy array of tokens)
    toks_np = np.array(toks, dtype=info["token_dtype"])
    # write to file
    ಅಂಕೆ_bytes = (256 * 4) + (len(toks) * toks_np.itemsize)
    print(f"writing {len(toks):,} tokens to {filename} ({num_bytes:,} bytes) in the {model_desc} format")
    with open(filename, "wb") ಇವ f:
        f.ಲಿಖ(header.tobytes())
        f.ಲಿಖ(toks_np.tobytes())

def write_evalfile(filename, datas):
    """
    Saves eval data as a .bin file, for reading in C.
    Used for multiple-choice style evals, e.g. HellaSwag and MMLU
    - First comes a header with 256 int32s
    - The examples follow, each example is a stream of uint16_t:
        - <START_EXAMPLE> delimiter of 2**16-1, i.e. 65,535
        - <EXAMPLE_BYTES>, bytes encoding this example, allowing efficient skip to next
        - <EXAMPLE_INDEX>, the index of the example in the dataset
        - <LABEL>, the index of the correct completion
        - <NUM_COMPLETIONS>, indicating the number of completions (usually 4)
        - <NUM><CONTEXT_TOKENS>, where <NUM> is the number of tokens in the context
        - <NUM><COMPLETION_TOKENS>, repeated NUM_COMPLETIONS times
    """
    # construct the header
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240522 # magic
    header[1] = 1 # version
    header[2] = len(datas) # number of examples
    header[3] = 0 # reserved for longest_example_bytes, fill in later
    # now write the individual examples
    longest_example_bytes = 0 # in units of uint16s
    full_stream = [] # the stream of uint16s, we'll write a single time at the end
    ಪ್ರತಿಜ್ಞಾ len(datas) < 2**16, "too many examples?"
    ಪ್ರತಿ idx, data ಅನ್ತಃ enumerate(datas):
        stream = []
        # header of the example
        stream.append(2**16-1) # <START_EXAMPLE>
        stream.append(0) # <EXAMPLE_BYTES> (fill in later)
        stream.append(idx) # <EXAMPLE_INDEX>
        stream.append(data["label"]) # <LABEL>
        ending_tokens = data["ending_tokens"]
        ಪ್ರತಿಜ್ಞಾ len(ending_tokens) == 4, "expected 4 completions for now? can relax later"
        stream.append(len(ending_tokens)) # <NUM_COMPLETIONS>
        # the (shared) context tokens
        ctx_tokens = data["ctx_tokens"]
        ಪ್ರತಿಜ್ಞಾ all(0 <= t < 2**16-1 ಪ್ರತಿ t ಅನ್ತಃ ctx_tokens), "bad context token"
        stream.append(len(ctx_tokens))
        stream.extend(ctx_tokens)
        # the completion tokens
        ಪ್ರತಿ end_tokens ಅನ್ತಃ ending_tokens:
            ಪ್ರತಿಜ್ಞಾ all(0 <= t < 2**16-1 ಪ್ರತಿ t ಅನ್ತಃ end_tokens), "bad completion token"
            stream.append(len(end_tokens))
            stream.extend(end_tokens)
        # write to full stream
        nbytes = len(stream)*2 # 2 bytes per uint16
        ಪ್ರತಿಜ್ಞಾ nbytes < 2**16, "example too large?"
        stream[1] = nbytes # fill in the <EXAMPLE_BYTES> field
        longest_example_bytes = max(longest_example_bytes, nbytes)
        full_stream.extend(stream)
    # construct the numpy array
    stream_np = np.array(full_stream, dtype=np.uint16)
    # fill in the longest_example field
    ಪ್ರತಿಜ್ಞಾ 0 < longest_example_bytes < 2**16, f"bad longest_example"
    header[3] = longest_example_bytes
    # write to file (for HellaSwag val this is 10,042 examples, 3.6MB file)
    print(f"writing {len(datas):,} examples to {filename}")
    with open(filename, "wb") ಇವ f:
        f.ಲಿಖ(header.tobytes())
        f.ಲಿಖ(stream_np.tobytes())
