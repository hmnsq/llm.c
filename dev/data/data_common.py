"""
Common utilities for the datasets
"""

ਆਯਾਤ ਬੇਨਤੀਆਂ
ਤਃ tqdm ਆਯਾਤ tqdm
ਆਯਾਤ numpy ਇਵ np


ਪਰਿਭਾਸ਼ਾ download_file(url: str, fname: str, chunk_ਮਾਪ=1024):
    """Helper function to download a file from a given url"""
    resp = ਬੇਨਤੀਆਂ.get(url, ਧਾਰਾ=ਸੱਚ)
    ਕੁੱਲ = ਪੂਰਨ(resp.ਸਿਰਲੇਖਾਂ.get("content-length", 0))
    ਨਾਲ open(fname, "wb") ਇਵ file, tqdm(
        desc=fname,
        ਕੁੱਲ=ਕੁੱਲ,
        unit="iB",
        unit_scale=ਸੱਚ,
        unit_ਭਾਜਕ=1024,
    ) ਇਵ bar:
        ਲਈ ਅੰਕੜੇ ਵਿੱਚ resp.iter_content(chunk_ਮਾਪ=chunk_ਮਾਪ):
            ਮਾਪ = file.ਲਿਖ(ਅੰਕੜੇ)
            bar.update(ਮਾਪ)


ਸਿਰਲੇਖਾਂ_INFO = {
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

ਪਰਿਭਾਸ਼ਾ write_datafile(filename, toks, model_desc="gpt-2"):
    """
    Saves token data as a .bin file, for reading in C.
    - First comes a header with 256 int32s
    - The tokens follow, each as uint16 (gpt-2) or uint32 (llama)
    """
    ਪ੍ਰਤਿਜ੍ਞਾ ਲੰਬਾਈ(toks) < 2**31, "token count too large" # ~2.1B tokens
    ਪ੍ਰਤਿਜ੍ਞਾ model_desc ਵਿੱਚ ["gpt-2", "llama-3"], f"unknown model descriptor {model_desc}"
    info = ਸਿਰਲੇਖਾਂ_INFO[model_desc]
    # construct the header
    ਸਿਰਲੇਖ = np.zeros(256, dtype=np.int32) # header is always 256 int32 values
    ਸਿਰਲੇਖ[0] = info["magic"]
    ਸਿਰਲੇਖ[1] = info["version"]
    ਸਿਰਲੇਖ[2] = ਲੰਬਾਈ(toks) # number of tokens after the 256*4 bytes of header
    # construct the data (numpy array of tokens)
    toks_np = np.array(toks, dtype=info["token_dtype"])
    # write to file
    ਅੰਕ_ਬਾਈਟਾਂ = (256 * 4) + (ਲੰਬਾਈ(toks) * toks_np.itemsize)
    ਛਾਪੋ(f"writing {len(toks):,} tokens to {filename} ({num_bytes:,} bytes) in the {model_desc} format")
    ਨਾਲ open(filename, "wb") ਇਵ f:
        f.ਲਿਖ(ਸਿਰਲੇਖ.tobytes())
        f.ਲਿਖ(toks_np.tobytes())

ਪਰਿਭਾਸ਼ਾ write_evalfile(filename, datas):
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
    ਸਿਰਲੇਖ = np.zeros(256, dtype=np.int32)
    ਸਿਰਲੇਖ[0] = 20240522 # magic
    ਸਿਰਲੇਖ[1] = 1 #ਸੰਸਕਰਣ
    ਸਿਰਲੇਖ[2] = ਲੰਬਾਈ(datas) # number of examples
    ਸਿਰਲੇਖ[3] = 0 # reserved for longest_example_bytes, fill in later
    # now write the individual examples
    longest_example_ਬਾਈਟਾਂ = 0 # in units of uint16s
    ਭਰਿਆ_ਧਾਰਾ = [] # the stream of uint16s, we'll write a single time at the end
    ਪ੍ਰਤਿਜ੍ਞਾ ਲੰਬਾਈ(datas) < 2**16, "too many examples?"
    ਲਈ idx, ਅੰਕੜੇ ਵਿੱਚ enumerate(datas):
        ਧਾਰਾ = []
        # header of the example
        ਧਾਰਾ.append(2**16-1) # <START_EXAMPLE>
        ਧਾਰਾ.append(0) # <EXAMPLE_BYTES> (fill in later)
        ਧਾਰਾ.append(idx) # <EXAMPLE_INDEX>
        ਧਾਰਾ.append(ਅੰਕੜੇ["label"]) # <LABEL>
        ending_ਟੋਕਨ = ਅੰਕੜੇ["ending_tokens"]
        ਪ੍ਰਤਿਜ੍ਞਾ ਲੰਬਾਈ(ending_ਟੋਕਨ) == 4, "expected 4 completions for now? can relax later"
        ਧਾਰਾ.append(ਲੰਬਾਈ(ending_ਟੋਕਨ)) # <NUM_COMPLETIONS>
        # the (shared) context tokens
        ctx_ਟੋਕਨ = ਅੰਕੜੇ["ctx_tokens"]
        ਪ੍ਰਤਿਜ੍ਞਾ all(0 <= t < 2**16-1 ਲਈ t ਵਿੱਚ ctx_ਟੋਕਨ), "bad context token"
        ਧਾਰਾ.append(ਲੰਬਾਈ(ctx_ਟੋਕਨ))
        ਧਾਰਾ.extend(ctx_ਟੋਕਨ)
        # the completion tokens
        ਲਈ ਸਿਰਾ_ਟੋਕਨ ਵਿੱਚ ending_ਟੋਕਨ:
            ਪ੍ਰਤਿਜ੍ਞਾ all(0 <= t < 2**16-1 ਲਈ t ਵਿੱਚ ਸਿਰਾ_ਟੋਕਨ), "bad completion token"
            ਧਾਰਾ.append(ਲੰਬਾਈ(ਸਿਰਾ_ਟੋਕਨ))
            ਧਾਰਾ.extend(ਸਿਰਾ_ਟੋਕਨ)
        # write to full stream
        nbytes = ਲੰਬਾਈ(ਧਾਰਾ)*2 # 2 bytes per uint16
        ਪ੍ਰਤਿਜ੍ਞਾ nbytes < 2**16, "example too large?"
        ਧਾਰਾ[1] = nbytes # fill in the <EXAMPLE_BYTES> field
        longest_example_ਬਾਈਟਾਂ = max(longest_example_ਬਾਈਟਾਂ, nbytes)
        ਭਰਿਆ_ਧਾਰਾ.extend(ਧਾਰਾ)
    # construct the numpy array
    ਧਾਰਾ_np = np.array(ਭਰਿਆ_ਧਾਰਾ, dtype=np.uint16)
    # fill in the longest_example field
    ਪ੍ਰਤਿਜ੍ਞਾ 0 < longest_example_ਬਾਈਟਾਂ < 2**16, f"bad longest_example"
    ਸਿਰਲੇਖ[3] = longest_example_ਬਾਈਟਾਂ
    # write to file (for HellaSwag val this is 10,042 examples, 3.6MB file)
    ਛਾਪੋ(f"writing {len(datas):,} examples to {filename}")
    ਨਾਲ open(filename, "wb") ਇਵ f:
        f.ਲਿਖ(ਸਿਰਲੇਖ.tobytes())
        f.ਲਿਖ(ਧਾਰਾ_np.tobytes())
