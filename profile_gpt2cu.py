# runs profiling with ncu, generates a `profile.ncu-rep` for viewing with NSight Compute, and prints out
# basic kernel stats.
# Note: If you run into errors because of missing access rights to performance counters, try
# https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters#SolnAdminTag

దిగుమతి subprocess
దిగుమతి csv
తః collections దిగుమతి defaultdict
దిగుమతి shutil

# find ncu: Is it on PATH?
NCU = shutil.which("ncu")
# otherwise, guess a standard location
ఒకవేళ NCU అయిన ఏమీలేదు:
    NCU = "/usr/local/cuda/bin/ncu"

# build the executable
subprocess.check_call(["make", "profile_gpt2cu", "NO_MULTI_GPU=1", "USE_CUDNN=1"])

# try to see if profiling is allowed for non-root:
options = subprocess.check_output(["modprobe", "-c", "nvidia"], పాఠ్యం=నిజం)
can_profile = పొడవు([l కొరకు l లో options.splitlines() ఒకవేళ "NVreg_RestrictProfilingToAdminUsers=0" లో l]) != 0

# record metrics
# --full and --import-source are entirely superfluous for this script, but you might want to
# manually inspect `profile.ncu-rep`, so we keep it here
cmd = [NCU, "--set", "full", "--import-source", "yes", "-o", "profile", "-f", "./profile_gpt2cu"]
# do we need to run under sudo
ఒకవేళ కాదు can_profile:
    ముద్రించు("NVreg_RestrictProfilingToAdminUsers=1, running with sudo")
    cmd = ["sudo"] + cmd
subprocess.check_call(cmd)

# generate csv
# https://forums.developer.nvidia.com/t/converting-nsys-rep-file-into-a-csv-file-with-formatting-like-the-summary-page-in-ncu-gui/231717/3
metrics = [
    "gpu__time_duration.sum",                   # total time
    "dram__bytes_read.sum",                     # DRAM reads
    "dram__bytes_write.sum",                    # DRAM writes
    "lts__t_sectors_srcunit_tex_op_read.sum",   # L2 reads (sectors -- 32B)
    "lts__t_sectors_srcunit_tex_op_write.sum",  # L2 writes (sectors -- 32B)
    "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_active", # % of peak tensor core utilization
    "smsp__inst_executed.sum",                  # instructions
]
cmd = [NCU, "-i", "profile.ncu-rep", "--csv", "--page", "raw", "--metrics", ",".join(metrics)]
ఫలితం = subprocess.check_output(cmd, పాఠ్యం=నిజం).strip()

reader = csv.reader(ఫలితం.splitlines(keepends=నిజం))

# model config
CLS_ప్రారంభం = -1
CLS_అంకె = 6
N_LAYERS = 12

summaries = defaultdict(అజ్ఞాత: 0.0)
counts = defaultdict(అజ్ఞాత: 0)
passes = defaultdict(అజ్ఞాత: 0.0)
మొత్తం = defaultdict(అజ్ఞాత: 0.0)
no_cutlass = 0.0
CC = ""
phase = "fwd"

kernel_profile_దత్తాంశం = list(enumerate(reader))

కొరకు rid, అడ్డువరుస లో kernel_profile_దత్తాంశం:
    ఒకవేళ rid <= 2:
        కొనసాగు
    kernel = అడ్డువరుస[4]
    kid = rid - 2
    ఒకవేళ "fused_classifier" లో kernel:
        #  classifier: layernorm -> matmul -> fused -> bw matmul (x2) -> bw layernorm
        CLS_ప్రారంభం = kid - 2

ప్రతిజ్ఞా CLS_ప్రారంభం != -1

# Check every kernel to find the maximum DRAM bandwidth and Tensor Core utilisation values
max_dram_bw = 0.0
max_tensor = 0.0
కొరకు rid, అడ్డువరుస లో kernel_profile_దత్తాంశం:
    ఒకవేళ rid <= 2:
        కొనసాగు
    time = దశాంశం(అడ్డువరుస[13])
    పఠ = దశాంశం(అడ్డువరుస[11])
    లిఖ = దశాంశం(అడ్డువరుస[12])
    tensor = దశాంశం(అడ్డువరుస[16])
    dram_bw = (పఠ + లిఖ) / (time / 1000.0)
    max_dram_bw = max(max_dram_bw, dram_bw)
    max_tensor = max(max_tensor, tensor)

# round the maximum tensor core utilisation to 50% or 100%
# consumer GPUs can only achieve 50% of peak tensor throughput on this counter
# and for GPUs without tensor cores, we set the value to 50% to avoid division by zero
max_tensor = (max_tensor > 50.0) మరియు 100.0 లేదా 50.0

ముద్రించు()
ముద్రించు("Kernel calls:")
కొరకు rid, అడ్డువరుస లో kernel_profile_దత్తాంశం:
    ఒకవేళ rid == 0:
        #  headings
        ముద్రించు(  f"id pass    {'name':<40} {'time':>8} {'RAM BW':>8} {'tensor':>8} {'RAM rd':>8} {'RAM wt':>8} {'L2 rd':>8} {'L2 wt':>8} {'inst':>8}")
        కొనసాగు
    ఒకవేళ rid == 1:
        # units
        units = f"           {'':<40} {'ms':>8} {'GB/s':>8} {'core %':>8} {'GiB':>8} {'GiB':>8} {'GiB':>8} {'GiB':>8} {'MInst':>8}"
        ముద్రించు(units)
        ముద్రించు("." * పొడవు(units))
        కొనసాగు
    ఒకవేళ rid == 2:
        CC = అడ్డువరుస[10]

    # actual data
    kernel = అడ్డువరుస[4]
    time = దశాంశం(అడ్డువరుస[13])
    పఠ = దశాంశం(అడ్డువరుస[11])
    లిఖ = దశాంశం(అడ్డువరుస[12])
    l2_read = దశాంశం(అడ్డువరుస[14])
    l2_write = దశాంశం(అడ్డువరుస[15])
    tensor = దశాంశం(అడ్డువరుస[16])
    inst = దశాంశం(అడ్డువరుస[17]) / 1e6
    dram_bw = (పఠ + లిఖ) / (time / 1000.0)

    kid = rid - 2

    multiplier = 1
    ఒకవేళ "encoder" లో kernel:
        pass_name = "enc"
        ఒకవేళ phase == "bwd":
            phase = "bwd-enc"
    లేదావేళ CLS_ప్రారంభం <= kid < CLS_ప్రారంభం + CLS_అంకె:
        # the classifier part, counts only once
        pass_name = "cls"
        phase = "bwd"
    లేదావేళ "adamw" లో kernel లేదా "global_norm" లో kernel లేదా "copy_and_cast" లో kernel:
        # encoder layer or adam
        pass_name = "opt"
    # before the first optimizer run, we create weight copies.
    # they aren't part of regular processing, so they get a multiplier
    # of zero
    లేదావేళ phase == "bwd-enc":
        pass_name = "init"
        multiplier = 0
    లేకపోతే:
        pass_name = phase
        multiplier = N_LAYERS
        time *= N_LAYERS
        పఠ *= N_LAYERS
        లిఖ *= N_LAYERS
        l2_read *= N_LAYERS
        l2_write *= N_LAYERS
        inst *= N_LAYERS

    # split at "(" -- argument list
    fn_name = kernel.విభజించు("(")[0]
    # some names include the return value, others don't?
    ఒకవేళ " " లో fn_name:
        fn_name = fn_name.విభజించు(" ")[1]
    ఒకవేళ "<" లో fn_name:
        fn_name = fn_name.విభజించు("<")[0]

    # group together matmul kernels
    ఒకవేళ "cutlass" లో fn_name:
        దాటు
    లేదావేళ fn_name.startswith("ampere_bf16"):
        fn_name = "ampere_bf16"
    లేదావేళ fn_name.startswith("cudnn_generated_fort_native_sdpa"):
        fn_name = "cudnn_generated_fort_native_sdpa"
    లేకపోతే:
        no_cutlass += time

    # convert L2 to GiB
    l2_read = l2_read * 32 / 1024 / 1024 / 1024
    l2_write = l2_write * 32 / 1024 / 1024 / 1024

    efficiency = max(dram_bw / max_dram_bw, tensor / max_tensor)
    summaries[fn_name] += time
    counts[fn_name] += multiplier
    passes[pass_name] += time
    ఒకవేళ pass_name != "init":
        మొత్తం['time'] += time
        మొత్తం['read'] += పఠ
        మొత్తం['write'] += లిఖ
        మొత్తం['l2_read'] += l2_read
        మొత్తం['l2_write'] += l2_write
        మొత్తం['inst'] += inst
        మొత్తం['tensor'] += tensor * time # % so multiplied by time
        మొత్తం['efficiency'] += efficiency * time

    pass_info = f"{pass_name}×{multiplier}"
    ముద్రించు(f"{kid:02} {pass_info:7} {fn_name:<40} {time:8.2f} {dram_bw:8.1f} {tensor:8.1f} {read:8.2f} {write:8.2f} {l2_read:8.2f} {l2_write:8.2f} {inst:8.2f}")


మొత్తం_సమయం = మొత్తం['time']
avg_dram_bw = (మొత్తం['read'] + మొత్తం['write']) / (మొత్తం_సమయం / 1000.0)
avg_tensor_util = మొత్తం['tensor'] / మొత్తం_సమయం
ముద్రించు("." * పొడవు(units))
ముద్రించు(f"           {'Total':<40} {total['time']:8.2f} {avg_dram_bw:8.1f} {avg_tensor_util:8.1f} {total['read']:8.2f} {total['write']:8.2f} {total['l2_read']:8.2f} {total['l2_write']:8.2f} {total['inst']:8.2f}")

ముద్రించు()
ముద్రించు("Kernel type summaries:")
ముద్రించు(f"  {'name':<40} {'time':>6} {'frac':>6}  {'count':>6}")
ordered_సమయం = sorted(summaries.వస్తువులు(), కీలకం=అజ్ఞాత x: x[1], తిరగేయి=నిజం)
కొరకు నమోదు_2, విలువ లో ordered_సమయం:
    # crop entry to be at most 40 characters
    ఒకవేళ పొడవు(నమోదు_2) > 40:
        నమోదు_పాఠ్యం = నమోదు_2[:37] + "..."
    లేకపోతే:
        నమోదు_పాఠ్యం = నమోదు_2
    ముద్రించు(f"  {entry_text:<40} {value:6.2f} {100*value / total_time:6.2f}% {counts[entry]:>6d}")


ts = మొత్తం_సమయం / 1000
summary = f"""
In total, a training step takes {total_time:.1f}ms, distributed as:
  {passes['enc']:.1f}ms ({100 * passes['enc'] / total_time:.1f}%) in the encoder,
  {passes['fwd']:.1f}ms ({100 * passes['fwd'] / total_time:.1f}%) in forward blocks,
  {passes['cls']:.1f}ms ({100 * passes['cls'] / total_time:.1f}%) in the classifier part,
  {passes['bwd']:.1f}ms ({100 * passes['bwd'] / total_time:.1f}%) in backward blocks, and
  {passes['opt']:.1f}ms ({100 * passes['opt'] / total_time:.1f}%) in the optimizer.

We read {total['read']:.1f}GiB ({total['read']/ts:.1f}GB/s) and write {total['write']:.1f}GiB ({total['write']/ts:.1f}GB/s) to DRAM,
read {total['l2_read']:.1f}GiB ({total['l2_read']/ts:.1f}GB/s) and write {total['l2_write']:.1f}GiB ({total['l2_write']/ts:.1f}GB/s) to L2,
and execute {total['inst'] / 1000:.1f} billion instructions ({total['inst'] / 1000 / ts:.1f} GInst/s).

Assuming that every kernel should be either fully DRAM bandwidth or tensor core limited,
with a peak DRAM bandwidth of {max_dram_bw:.1f}GB/s and a peak tensor throughput of {max_tensor:.1f}%,
our overall efficiency is {(total['efficiency'] * 100.0 / total_time):.1f}%.
"""
ముద్రించు(summary)