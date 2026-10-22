"""
Common utilities for the datasets
"""

استورد درخواستون
من tqdm استورد tqdm
استورد numpy كـ np


وصف download_file(url: str, fname: str, chunk_سائيز=1024):
    """Helper function to download a file from a given url"""
    resp = درخواستون.get(url, وهڪرو=سچ)
    ڪل = صحيح(resp.سرا.get("content-length", 0))
    سان open(fname, "wb") كـ file, tqdm(
        desc=fname,
        ڪل=ڪل,
        unit="iB",
        unit_scale=سچ,
        unit_ورهائيندڙ=1024,
    ) كـ bar:
        لكل معلومات في resp.iter_content(chunk_سائيز=chunk_سائيز):
            سائيز = file.اكتب(معلومات)
            bar.update(سائيز)


سرا_INFO = {
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

وصف write_datafile(filename, toks, model_desc="gpt-2"):
    """
    Saves token data as a .bin file, for reading in C.
    - First comes a header with 256 int32s
    - The tokens follow, each as uint16 (gpt-2) or uint32 (llama)
    """
    أكد ڊيگهه(toks) < 2**31, "token count too large" # ~2.1B tokens
    أكد model_desc في ["gpt-2", "llama-3"], f"unknown model descriptor {model_desc}"
    info = سرا_INFO[model_desc]
    # construct the header
    سرو = np.zeros(256, dtype=np.int32) # header is always 256 int32 values
    سرو[0] = info["magic"]
    سرو[1] = info["version"]
    سرو[2] = ڊيگهه(toks) # number of tokens after the 256*4 bytes of header
    # construct the data (numpy array of tokens)
    toks_np = np.array(toks, dtype=info["token_dtype"])
    # write to file
    انگ_بائيٽون = (256 * 4) + (ڊيگهه(toks) * toks_np.itemsize)
    ڇاپيو(f"writing {len(toks):,} tokens to {filename} ({num_bytes:,} bytes) in the {model_desc} format")
    سان open(filename, "wb") كـ f:
        f.اكتب(سرو.tobytes())
        f.اكتب(toks_np.tobytes())

وصف write_evalfile(filename, datas):
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
    سرو = np.zeros(256, dtype=np.int32)
    سرو[0] = 20240522 # magic
    سرو[1] = 1 #نسخو
    سرو[2] = ڊيگهه(datas) # number of examples
    سرو[3] = 0 # reserved for longest_example_bytes, fill in later
    # now write the individual examples
    longest_example_بائيٽون = 0 # in units of uint16s
    ڀريل_وهڪرو = [] # the stream of uint16s, we'll write a single time at the end
    أكد ڊيگهه(datas) < 2**16, "too many examples?"
    لكل idx, معلومات في enumerate(datas):
        وهڪرو = []
        # header of the example
        وهڪرو.append(2**16-1) # <START_EXAMPLE>
        وهڪرو.append(0) # <EXAMPLE_BYTES> (fill in later)
        وهڪرو.append(idx) # <EXAMPLE_INDEX>
        وهڪرو.append(معلومات["label"]) # <LABEL>
        ending_ٽوڪن = معلومات["ending_tokens"]
        أكد ڊيگهه(ending_ٽوڪن) == 4, "expected 4 completions for now? can relax later"
        وهڪرو.append(ڊيگهه(ending_ٽوڪن)) # <NUM_COMPLETIONS>
        # the (shared) context tokens
        ctx_ٽوڪن = معلومات["ctx_tokens"]
        أكد all(0 <= t < 2**16-1 لكل t في ctx_ٽوڪن), "bad context token"
        وهڪرو.append(ڊيگهه(ctx_ٽوڪن))
        وهڪرو.extend(ctx_ٽوڪن)
        # the completion tokens
        لكل پڇاڙي_ٽوڪن في ending_ٽوڪن:
            أكد all(0 <= t < 2**16-1 لكل t في پڇاڙي_ٽوڪن), "bad completion token"
            وهڪرو.append(ڊيگهه(پڇاڙي_ٽوڪن))
            وهڪرو.extend(پڇاڙي_ٽوڪن)
        # write to full stream
        nbytes = ڊيگهه(وهڪرو)*2 # 2 bytes per uint16
        أكد nbytes < 2**16, "example too large?"
        وهڪرو[1] = nbytes # fill in the <EXAMPLE_BYTES> field
        longest_example_بائيٽون = max(longest_example_بائيٽون, nbytes)
        ڀريل_وهڪرو.extend(وهڪرو)
    # construct the numpy array
    وهڪرو_np = np.array(ڀريل_وهڪرو, dtype=np.uint16)
    # fill in the longest_example field
    أكد 0 < longest_example_بائيٽون < 2**16, f"bad longest_example"
    سرو[3] = longest_example_بائيٽون
    # write to file (for HellaSwag val this is 10,042 examples, 3.6MB file)
    ڇاپيو(f"writing {len(datas):,} examples to {filename}")
    سان open(filename, "wb") كـ f:
        f.اكتب(سرو.tobytes())
        f.اكتب(وهڪرو_np.tobytes())
