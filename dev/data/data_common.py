"""
Common utilities for the datasets
"""

దిగుమతి అభ్యర్థనలు
తః tqdm దిగుమతి tqdm
దిగుమతి numpy ఇవ np


నిర్వచనం download_file(url: str, fname: str, chunk_కొలత=1024):
    """Helper function to download a file from a given url"""
    resp = అభ్యర్థనలు.get(url, ప్రవాహం=నిజం)
    మొత్తం = పూర్ణాంకం(resp.శీర్షికలు.get("content-length", 0))
    తోడుగా open(fname, "wb") ఇవ file, tqdm(
        desc=fname,
        మొత్తం=మొత్తం,
        unit="iB",
        unit_scale=నిజం,
        unit_భాజకం=1024,
    ) ఇవ bar:
        కొరకు దత్తాంశం లో resp.iter_content(chunk_కొలత=chunk_కొలత):
            కొలత = file.లిఖ(దత్తాంశం)
            bar.update(కొలత)


శీర్షికలు_INFO = {
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

నిర్వచనం write_datafile(filename, toks, model_desc="gpt-2"):
    """
    Saves token data as a .bin file, for reading in C.
    - First comes a header with 256 int32s
    - The tokens follow, each as uint16 (gpt-2) or uint32 (llama)
    """
    ప్రతిజ్ఞా పొడవు(toks) < 2**31, "token count too large" # ~2.1B tokens
    ప్రతిజ్ఞా model_desc లో ["gpt-2", "llama-3"], f"unknown model descriptor {model_desc}"
    info = శీర్షికలు_INFO[model_desc]
    # construct the header
    శీర్షిక = np.zeros(256, dtype=np.int32) # header is always 256 int32 values
    శీర్షిక[0] = info["magic"]
    శీర్షిక[1] = info["version"]
    శీర్షిక[2] = పొడవు(toks) # number of tokens after the 256*4 bytes of header
    # construct the data (numpy array of tokens)
    toks_np = np.array(toks, dtype=info["token_dtype"])
    # write to file
    అంకె_బైట్లు = (256 * 4) + (పొడవు(toks) * toks_np.itemsize)
    ముద్రించు(f"writing {len(toks):,} tokens to {filename} ({num_bytes:,} bytes) in the {model_desc} format")
    తోడుగా open(filename, "wb") ఇవ f:
        f.లిఖ(శీర్షిక.tobytes())
        f.లిఖ(toks_np.tobytes())

నిర్వచనం write_evalfile(filename, datas):
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
    శీర్షిక = np.zeros(256, dtype=np.int32)
    శీర్షిక[0] = 20240522 # magic
    శీర్షిక[1] = 1 #సంస్కరణ
    శీర్షిక[2] = పొడవు(datas) # number of examples
    శీర్షిక[3] = 0 # reserved for longest_example_bytes, fill in later
    # now write the individual examples
    longest_example_బైట్లు = 0 # in units of uint16s
    నిండిన_ప్రవాహం = [] # the stream of uint16s, we'll write a single time at the end
    ప్రతిజ్ఞా పొడవు(datas) < 2**16, "too many examples?"
    కొరకు idx, దత్తాంశం లో enumerate(datas):
        ప్రవాహం = []
        # header of the example
        ప్రవాహం.append(2**16-1) # <START_EXAMPLE>
        ప్రవాహం.append(0) # <EXAMPLE_BYTES> (fill in later)
        ప్రవాహం.append(idx) # <EXAMPLE_INDEX>
        ప్రవాహం.append(దత్తాంశం["label"]) # <LABEL>
        ending_టోకెన్లు = దత్తాంశం["ending_tokens"]
        ప్రతిజ్ఞా పొడవు(ending_టోకెన్లు) == 4, "expected 4 completions for now? can relax later"
        ప్రవాహం.append(పొడవు(ending_టోకెన్లు)) # <NUM_COMPLETIONS>
        # the (shared) context tokens
        ctx_టోకెన్లు = దత్తాంశం["ctx_tokens"]
        ప్రతిజ్ఞా all(0 <= t < 2**16-1 కొరకు t లో ctx_టోకెన్లు), "bad context token"
        ప్రవాహం.append(పొడవు(ctx_టోకెన్లు))
        ప్రవాహం.extend(ctx_టోకెన్లు)
        # the completion tokens
        కొరకు చివర_టోకెన్లు లో ending_టోకెన్లు:
            ప్రతిజ్ఞా all(0 <= t < 2**16-1 కొరకు t లో చివర_టోకెన్లు), "bad completion token"
            ప్రవాహం.append(పొడవు(చివర_టోకెన్లు))
            ప్రవాహం.extend(చివర_టోకెన్లు)
        # write to full stream
        nbytes = పొడవు(ప్రవాహం)*2 # 2 bytes per uint16
        ప్రతిజ్ఞా nbytes < 2**16, "example too large?"
        ప్రవాహం[1] = nbytes # fill in the <EXAMPLE_BYTES> field
        longest_example_బైట్లు = max(longest_example_బైట్లు, nbytes)
        నిండిన_ప్రవాహం.extend(ప్రవాహం)
    # construct the numpy array
    ప్రవాహం_np = np.array(నిండిన_ప్రవాహం, dtype=np.uint16)
    # fill in the longest_example field
    ప్రతిజ్ఞా 0 < longest_example_బైట్లు < 2**16, f"bad longest_example"
    శీర్షిక[3] = longest_example_బైట్లు
    # write to file (for HellaSwag val this is 10,042 examples, 3.6MB file)
    ముద్రించు(f"writing {len(datas):,} examples to {filename}")
    తోడుగా open(filename, "wb") ఇవ f:
        f.లిఖ(శీర్షిక.tobytes())
        f.లిఖ(ప్రవాహం_np.tobytes())
