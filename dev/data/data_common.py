"""
Common utilities for the datasets
"""

استورد بینتیاں
من tqdm استورد tqdm
استورد numpy كـ np


تعریف download_file(url: str, fname: str, chunk_ناپ=1024):
    """Helper function to download a file from a given url"""
    resp = بینتیاں.get(url, دھارا=سچ)
    کل = صحيح(resp.سرخیاں.get("content-length", 0))
    نال open(fname, "wb") كـ file, tqdm(
        desc=fname,
        کل=کل,
        unit="iB",
        unit_scale=سچ,
        unit_مقسوم_علیہ=1024,
    ) كـ bar:
        لكل انکڑے في resp.iter_content(chunk_ناپ=chunk_ناپ):
            ناپ = file.اكتب(انکڑے)
            bar.update(ناپ)


سرخیاں_INFO = {
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

تعریف write_datafile(filename, toks, model_desc="gpt-2"):
    """
    Saves token data as a .bin file, for reading in C.
    - First comes a header with 256 int32s
    - The tokens follow, each as uint16 (gpt-2) or uint32 (llama)
    """
    أكد لمبائی(toks) < 2**31, "token count too large" # ~2.1B tokens
    أكد model_desc في ["gpt-2", "llama-3"], f"unknown model descriptor {model_desc}"
    info = سرخیاں_INFO[model_desc]
    # construct the header
    سرخی = np.zeros(256, dtype=np.int32) # header is always 256 int32 values
    سرخی[0] = info["magic"]
    سرخی[1] = info["version"]
    سرخی[2] = لمبائی(toks) # number of tokens after the 256*4 bytes of header
    # construct the data (numpy array of tokens)
    toks_np = np.array(toks, dtype=info["token_dtype"])
    # write to file
    عدد_بائٹاں = (256 * 4) + (لمبائی(toks) * toks_np.itemsize)
    چھاپو(f"writing {len(toks):,} tokens to {filename} ({num_bytes:,} bytes) in the {model_desc} format")
    نال open(filename, "wb") كـ f:
        f.اكتب(سرخی.tobytes())
        f.اكتب(toks_np.tobytes())

تعریف write_evalfile(filename, datas):
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
    سرخی = np.zeros(256, dtype=np.int32)
    سرخی[0] = 20240522 # magic
    سرخی[1] = 1 #نسخہ
    سرخی[2] = لمبائی(datas) # number of examples
    سرخی[3] = 0 # reserved for longest_example_bytes, fill in later
    # now write the individual examples
    longest_example_بائٹاں = 0 # in units of uint16s
    بھریا_دھارا = [] # the stream of uint16s, we'll write a single time at the end
    أكد لمبائی(datas) < 2**16, "too many examples?"
    لكل idx, انکڑے في enumerate(datas):
        دھارا = []
        # header of the example
        دھارا.append(2**16-1) # <START_EXAMPLE>
        دھارا.append(0) # <EXAMPLE_BYTES> (fill in later)
        دھارا.append(idx) # <EXAMPLE_INDEX>
        دھارا.append(انکڑے["label"]) # <LABEL>
        ending_ٹوکن = انکڑے["ending_tokens"]
        أكد لمبائی(ending_ٹوکن) == 4, "expected 4 completions for now? can relax later"
        دھارا.append(لمبائی(ending_ٹوکن)) # <NUM_COMPLETIONS>
        # the (shared) context tokens
        ctx_ٹوکن = انکڑے["ctx_tokens"]
        أكد all(0 <= t < 2**16-1 لكل t في ctx_ٹوکن), "bad context token"
        دھارا.append(لمبائی(ctx_ٹوکن))
        دھارا.extend(ctx_ٹوکن)
        # the completion tokens
        لكل سرا_ٹوکن في ending_ٹوکن:
            أكد all(0 <= t < 2**16-1 لكل t في سرا_ٹوکن), "bad completion token"
            دھارا.append(لمبائی(سرا_ٹوکن))
            دھارا.extend(سرا_ٹوکن)
        # write to full stream
        nbytes = لمبائی(دھارا)*2 # 2 bytes per uint16
        أكد nbytes < 2**16, "example too large?"
        دھارا[1] = nbytes # fill in the <EXAMPLE_BYTES> field
        longest_example_بائٹاں = max(longest_example_بائٹاں, nbytes)
        بھریا_دھارا.extend(دھارا)
    # construct the numpy array
    دھارا_np = np.array(بھریا_دھارا, dtype=np.uint16)
    # fill in the longest_example field
    أكد 0 < longest_example_بائٹاں < 2**16, f"bad longest_example"
    سرخی[3] = longest_example_بائٹاں
    # write to file (for HellaSwag val this is 10,042 examples, 3.6MB file)
    چھاپو(f"writing {len(datas):,} examples to {filename}")
    نال open(filename, "wb") كـ f:
        f.اكتب(سرخی.tobytes())
        f.اكتب(دھارا_np.tobytes())
