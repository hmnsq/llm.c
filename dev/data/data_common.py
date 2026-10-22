"""
Common utilities for the datasets
"""

استورد درخواستها
من tqdm استورد tqdm
استورد numpy كـ np


تعریف download_file(url: str, fname: str, chunk_اندازه=1024):
    """Helper function to download a file from a given url"""
    resp = درخواستها.get(url, جریان=درست)
    مجموع = صحيح(resp.سرآیندها.get("content-length", 0))
    با open(fname, "wb") كـ file, tqdm(
        desc=fname,
        مجموع=مجموع,
        unit="iB",
        unit_scale=درست,
        unit_مقسوم_علیه=1024,
    ) كـ bar:
        لكل داده في resp.iter_content(chunk_اندازه=chunk_اندازه):
            اندازه = file.اكتب(داده)
            bar.update(اندازه)


سرآیندها_INFO = {
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
    أكد طول(toks) < 2**31, "token count too large" # ~2.1B tokens
    أكد model_desc في ["gpt-2", "llama-3"], f"unknown model descriptor {model_desc}"
    info = سرآیندها_INFO[model_desc]
    # construct the header
    سرآیند = np.zeros(256, dtype=np.int32) # header is always 256 int32 values
    سرآیند[0] = info["magic"]
    سرآیند[1] = info["version"]
    سرآیند[2] = طول(toks) # number of tokens after the 256*4 bytes of header
    # construct the data (numpy array of tokens)
    toks_np = np.array(toks, dtype=info["token_dtype"])
    # write to file
    رقم_بایتها = (256 * 4) + (طول(toks) * toks_np.itemsize)
    چاپ(f"writing {len(toks):,} tokens to {filename} ({num_bytes:,} bytes) in the {model_desc} format")
    با open(filename, "wb") كـ f:
        f.اكتب(سرآیند.tobytes())
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
    سرآیند = np.zeros(256, dtype=np.int32)
    سرآیند[0] = 20240522 # magic
    سرآیند[1] = 1 #نسخه
    سرآیند[2] = طول(datas) # number of examples
    سرآیند[3] = 0 # reserved for longest_example_bytes, fill in later
    # now write the individual examples
    longest_example_بایتها = 0 # in units of uint16s
    پر_جریان = [] # the stream of uint16s, we'll write a single time at the end
    أكد طول(datas) < 2**16, "too many examples?"
    لكل idx, داده في enumerate(datas):
        جریان = []
        # header of the example
        جریان.append(2**16-1) # <START_EXAMPLE>
        جریان.append(0) # <EXAMPLE_BYTES> (fill in later)
        جریان.append(idx) # <EXAMPLE_INDEX>
        جریان.append(داده["label"]) # <LABEL>
        ending_نشانهها = داده["ending_tokens"]
        أكد طول(ending_نشانهها) == 4, "expected 4 completions for now? can relax later"
        جریان.append(طول(ending_نشانهها)) # <NUM_COMPLETIONS>
        # the (shared) context tokens
        ctx_نشانهها = داده["ctx_tokens"]
        أكد all(0 <= t < 2**16-1 لكل t في ctx_نشانهها), "bad context token"
        جریان.append(طول(ctx_نشانهها))
        جریان.extend(ctx_نشانهها)
        # the completion tokens
        لكل انتها_نشانهها في ending_نشانهها:
            أكد all(0 <= t < 2**16-1 لكل t في انتها_نشانهها), "bad completion token"
            جریان.append(طول(انتها_نشانهها))
            جریان.extend(انتها_نشانهها)
        # write to full stream
        nbytes = طول(جریان)*2 # 2 bytes per uint16
        أكد nbytes < 2**16, "example too large?"
        جریان[1] = nbytes # fill in the <EXAMPLE_BYTES> field
        longest_example_بایتها = max(longest_example_بایتها, nbytes)
        پر_جریان.extend(جریان)
    # construct the numpy array
    جریان_np = np.array(پر_جریان, dtype=np.uint16)
    # fill in the longest_example field
    أكد 0 < longest_example_بایتها < 2**16, f"bad longest_example"
    سرآیند[3] = longest_example_بایتها
    # write to file (for HellaSwag val this is 10,042 examples, 3.6MB file)
    چاپ(f"writing {len(datas):,} examples to {filename}")
    با open(filename, "wb") كـ f:
        f.اكتب(سرآیند.tobytes())
        f.اكتب(جریان_np.tobytes())
