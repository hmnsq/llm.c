"""
Common utilities for the datasets
"""

استورد غوښتنې
من tqdm استورد tqdm
استورد numpy كـ np


تعریف download_file(url: str, fname: str, chunk_اندازه=1024):
    """Helper function to download a file from a given url"""
    resp = غوښتنې.get(url, بهیر=ریښتیا)
    ټول = صحيح(resp.سرلیکونه.get("content-length", 0))
    سره open(fname, "wb") كـ file, tqdm(
        desc=fname,
        ټول=ټول,
        unit="iB",
        unit_scale=ریښتیا,
        unit_وېشونکی=1024,
    ) كـ bar:
        لكل معلومات في resp.iter_content(chunk_اندازه=chunk_اندازه):
            اندازه = file.اكتب(معلومات)
            bar.update(اندازه)


سرلیکونه_INFO = {
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
    أكد اوږدوالی(toks) < 2**31, "token count too large" # ~2.1B tokens
    أكد model_desc في ["gpt-2", "llama-3"], f"unknown model descriptor {model_desc}"
    info = سرلیکونه_INFO[model_desc]
    # construct the header
    سرلیک = np.zeros(256, dtype=np.int32) # header is always 256 int32 values
    سرلیک[0] = info["magic"]
    سرلیک[1] = info["version"]
    سرلیک[2] = اوږدوالی(toks) # number of tokens after the 256*4 bytes of header
    # construct the data (numpy array of tokens)
    toks_np = np.array(toks, dtype=info["token_dtype"])
    # write to file
    عدد_بایټونه = (256 * 4) + (اوږدوالی(toks) * toks_np.itemsize)
    چاپ(f"writing {len(toks):,} tokens to {filename} ({num_bytes:,} bytes) in the {model_desc} format")
    سره open(filename, "wb") كـ f:
        f.اكتب(سرلیک.tobytes())
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
    سرلیک = np.zeros(256, dtype=np.int32)
    سرلیک[0] = 20240522 # magic
    سرلیک[1] = 1 #نسخه
    سرلیک[2] = اوږدوالی(datas) # number of examples
    سرلیک[3] = 0 # reserved for longest_example_bytes, fill in later
    # now write the individual examples
    longest_example_بایټونه = 0 # in units of uint16s
    ډک_بهیر = [] # the stream of uint16s, we'll write a single time at the end
    أكد اوږدوالی(datas) < 2**16, "too many examples?"
    لكل idx, معلومات في enumerate(datas):
        بهیر = []
        # header of the example
        بهیر.append(2**16-1) # <START_EXAMPLE>
        بهیر.append(0) # <EXAMPLE_BYTES> (fill in later)
        بهیر.append(idx) # <EXAMPLE_INDEX>
        بهیر.append(معلومات["label"]) # <LABEL>
        ending_نښې = معلومات["ending_tokens"]
        أكد اوږدوالی(ending_نښې) == 4, "expected 4 completions for now? can relax later"
        بهیر.append(اوږدوالی(ending_نښې)) # <NUM_COMPLETIONS>
        # the (shared) context tokens
        ctx_نښې = معلومات["ctx_tokens"]
        أكد all(0 <= t < 2**16-1 لكل t في ctx_نښې), "bad context token"
        بهیر.append(اوږدوالی(ctx_نښې))
        بهیر.extend(ctx_نښې)
        # the completion tokens
        لكل پای_ټکی_نښې في ending_نښې:
            أكد all(0 <= t < 2**16-1 لكل t في پای_ټکی_نښې), "bad completion token"
            بهیر.append(اوږدوالی(پای_ټکی_نښې))
            بهیر.extend(پای_ټکی_نښې)
        # write to full stream
        nbytes = اوږدوالی(بهیر)*2 # 2 bytes per uint16
        أكد nbytes < 2**16, "example too large?"
        بهیر[1] = nbytes # fill in the <EXAMPLE_BYTES> field
        longest_example_بایټونه = max(longest_example_بایټونه, nbytes)
        ډک_بهیر.extend(بهیر)
    # construct the numpy array
    بهیر_np = np.array(ډک_بهیر, dtype=np.uint16)
    # fill in the longest_example field
    أكد 0 < longest_example_بایټونه < 2**16, f"bad longest_example"
    سرلیک[3] = longest_example_بایټونه
    # write to file (for HellaSwag val this is 10,042 examples, 3.6MB file)
    چاپ(f"writing {len(datas):,} examples to {filename}")
    سره open(filename, "wb") كـ f:
        f.اكتب(سرلیک.tobytes())
        f.اكتب(بهیر_np.tobytes())
