# runs profiling with ncu, generates a `profile.ncu-rep` for viewing with NSight Compute, and prints out
# basic kernel stats.
# Note: If you run into errors because of missing access rights to performance counters, try
# https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters#SolnAdminTag

ಆಮದು subprocess
ಆಮದು csv
ತಃ collections ಆಮದು defaultdict
ಆಮದು shutil

# find ncu: Is it on PATH?
NCU = shutil.which("ncu")
# otherwise, guess a standard location
ಆದರೆ NCU ಆಗಿರುವ ಏನೂಇಲ್ಲ:
    NCU = "/usr/local/cuda/bin/ncu"

# build the executable
subprocess.check_call(["make", "profile_gpt2cu", "NO_MULTI_GPU=1", "USE_CUDNN=1"])

# try to see if profiling is allowed for non-root:
options = subprocess.check_output(["modprobe", "-c", "nvidia"], ಪಠ್ಯ=ನಿಜ)
can_profile = ಉದ್ದ([l ಗಾಗಿ l ಒಳಗೆ options.splitlines() ಆದರೆ "NVreg_RestrictProfilingToAdminUsers=0" ಒಳಗೆ l]) != 0

# record metrics
# --full and --import-source are entirely superfluous for this script, but you might want to
# manually inspect `profile.ncu-rep`, so we keep it here
cmd = [NCU, "--set", "full", "--import-source", "yes", "-o", "profile", "-f", "./profile_gpt2cu"]
# do we need to run under sudo
ಆದರೆ ಅಲ್ಲ can_profile:
    ಮುದ್ರಿಸು("NVreg_RestrictProfilingToAdminUsers=1, running with sudo")
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
ಫಲಿತಾಂಶ = subprocess.check_output(cmd, ಪಠ್ಯ=ನಿಜ).strip()

reader = csv.reader(ಫಲಿತಾಂಶ.splitlines(keepends=ನಿಜ))

# model config
CLS_ಆರಂಭ = -1
CLS_ಅಂಕೆ = 6
N_LAYERS = 12

summaries = defaultdict(ಅನಾಮಿಕ: 0.0)
counts = defaultdict(ಅನಾಮಿಕ: 0)
passes = defaultdict(ಅನಾಮಿಕ: 0.0)
ಒಟ್ಟು = defaultdict(ಅನಾಮಿಕ: 0.0)
no_cutlass = 0.0
CC = ""
phase = "fwd"

kernel_profile_ದತ್ತಾಂಶ = list(enumerate(reader))

ಗಾಗಿ rid, ಅಡ್ಡಸಾಲು ಒಳಗೆ kernel_profile_ದತ್ತಾಂಶ:
    ಆದರೆ rid <= 2:
        ಮುಂದುವರಿ
    kernel = ಅಡ್ಡಸಾಲು[4]
    kid = rid - 2
    ಆದರೆ "fused_classifier" ಒಳಗೆ kernel:
        #  classifier: layernorm -> matmul -> fused -> bw matmul (x2) -> bw layernorm
        CLS_ಆರಂಭ = kid - 2

ಪ್ರತಿಜ್ಞಾ CLS_ಆರಂಭ != -1

# Check every kernel to find the maximum DRAM bandwidth and Tensor Core utilisation values
max_dram_bw = 0.0
max_tensor = 0.0
ಗಾಗಿ rid, ಅಡ್ಡಸಾಲು ಒಳಗೆ kernel_profile_ದತ್ತಾಂಶ:
    ಆದರೆ rid <= 2:
        ಮುಂದುವರಿ
    time = ದಶಮಾಂಶ(ಅಡ್ಡಸಾಲು[13])
    ಪಠ = ದಶಮಾಂಶ(ಅಡ್ಡಸಾಲು[11])
    ಲಿಖ = ದಶಮಾಂಶ(ಅಡ್ಡಸಾಲು[12])
    tensor = ದಶಮಾಂಶ(ಅಡ್ಡಸಾಲು[16])
    dram_bw = (ಪಠ + ಲಿಖ) / (time / 1000.0)
    max_dram_bw = max(max_dram_bw, dram_bw)
    max_tensor = max(max_tensor, tensor)

# round the maximum tensor core utilisation to 50% or 100%
# consumer GPUs can only achieve 50% of peak tensor throughput on this counter
# and for GPUs without tensor cores, we set the value to 50% to avoid division by zero
max_tensor = (max_tensor > 50.0) ಮತ್ತು 100.0 ಅಥವಾ 50.0

ಮುದ್ರಿಸು()
ಮುದ್ರಿಸು("Kernel calls:")
ಗಾಗಿ rid, ಅಡ್ಡಸಾಲು ಒಳಗೆ kernel_profile_ದತ್ತಾಂಶ:
    ಆದರೆ rid == 0:
        #  headings
        ಮುದ್ರಿಸು(  f"id pass    {'name':<40} {'time':>8} {'RAM BW':>8} {'tensor':>8} {'RAM rd':>8} {'RAM wt':>8} {'L2 rd':>8} {'L2 wt':>8} {'inst':>8}")
        ಮುಂದುವರಿ
    ಆದರೆ rid == 1:
        # units
        units = f"           {'':<40} {'ms':>8} {'GB/s':>8} {'core %':>8} {'GiB':>8} {'GiB':>8} {'GiB':>8} {'GiB':>8} {'MInst':>8}"
        ಮುದ್ರಿಸು(units)
        ಮುದ್ರಿಸು("." * ಉದ್ದ(units))
        ಮುಂದುವರಿ
    ಆದರೆ rid == 2:
        CC = ಅಡ್ಡಸಾಲು[10]

    # actual data
    kernel = ಅಡ್ಡಸಾಲು[4]
    time = ದಶಮಾಂಶ(ಅಡ್ಡಸಾಲು[13])
    ಪಠ = ದಶಮಾಂಶ(ಅಡ್ಡಸಾಲು[11])
    ಲಿಖ = ದಶಮಾಂಶ(ಅಡ್ಡಸಾಲು[12])
    l2_read = ದಶಮಾಂಶ(ಅಡ್ಡಸಾಲು[14])
    l2_write = ದಶಮಾಂಶ(ಅಡ್ಡಸಾಲು[15])
    tensor = ದಶಮಾಂಶ(ಅಡ್ಡಸಾಲು[16])
    inst = ದಶಮಾಂಶ(ಅಡ್ಡಸಾಲು[17]) / 1e6
    dram_bw = (ಪಠ + ಲಿಖ) / (time / 1000.0)

    kid = rid - 2

    multiplier = 1
    ಆದರೆ "encoder" ಒಳಗೆ kernel:
        pass_name = "enc"
        ಆದರೆ phase == "bwd":
            phase = "bwd-enc"
    ಅಥವಾಆದರೆ CLS_ಆರಂಭ <= kid < CLS_ಆರಂಭ + CLS_ಅಂಕೆ:
        # the classifier part, counts only once
        pass_name = "cls"
        phase = "bwd"
    ಅಥವಾಆದರೆ "adamw" ಒಳಗೆ kernel ಅಥವಾ "global_norm" ಒಳಗೆ kernel ಅಥವಾ "copy_and_cast" ಒಳಗೆ kernel:
        # encoder layer or adam
        pass_name = "opt"
    # before the first optimizer run, we create weight copies.
    # they aren't part of regular processing, so they get a multiplier
    # of zero
    ಅಥವಾಆದರೆ phase == "bwd-enc":
        pass_name = "init"
        multiplier = 0
    ಇಲ್ಲದಿದ್ದರೆ:
        pass_name = phase
        multiplier = N_LAYERS
        time *= N_LAYERS
        ಪಠ *= N_LAYERS
        ಲಿಖ *= N_LAYERS
        l2_read *= N_LAYERS
        l2_write *= N_LAYERS
        inst *= N_LAYERS

    # split at "(" -- argument list
    fn_name = kernel.ವಿಭಜಿಸು("(")[0]
    # some names include the return value, others don't?
    ಆದರೆ " " ಒಳಗೆ fn_name:
        fn_name = fn_name.ವಿಭಜಿಸು(" ")[1]
    ಆದರೆ "<" ಒಳಗೆ fn_name:
        fn_name = fn_name.ವಿಭಜಿಸು("<")[0]

    # group together matmul kernels
    ಆದರೆ "cutlass" ಒಳಗೆ fn_name:
        ದಾಟು
    ಅಥವಾಆದರೆ fn_name.startswith("ampere_bf16"):
        fn_name = "ampere_bf16"
    ಅಥವಾಆದರೆ fn_name.startswith("cudnn_generated_fort_native_sdpa"):
        fn_name = "cudnn_generated_fort_native_sdpa"
    ಇಲ್ಲದಿದ್ದರೆ:
        no_cutlass += time

    # convert L2 to GiB
    l2_read = l2_read * 32 / 1024 / 1024 / 1024
    l2_write = l2_write * 32 / 1024 / 1024 / 1024

    efficiency = max(dram_bw / max_dram_bw, tensor / max_tensor)
    summaries[fn_name] += time
    counts[fn_name] += multiplier
    passes[pass_name] += time
    ಆದರೆ pass_name != "init":
        ಒಟ್ಟು['time'] += time
        ಒಟ್ಟು['read'] += ಪಠ
        ಒಟ್ಟು['write'] += ಲಿಖ
        ಒಟ್ಟು['l2_read'] += l2_read
        ಒಟ್ಟು['l2_write'] += l2_write
        ಒಟ್ಟು['inst'] += inst
        ಒಟ್ಟು['tensor'] += tensor * time # % so multiplied by time
        ಒಟ್ಟು['efficiency'] += efficiency * time

    pass_info = f"{pass_name}×{multiplier}"
    ಮುದ್ರಿಸು(f"{kid:02} {pass_info:7} {fn_name:<40} {time:8.2f} {dram_bw:8.1f} {tensor:8.1f} {read:8.2f} {write:8.2f} {l2_read:8.2f} {l2_write:8.2f} {inst:8.2f}")


ಒಟ್ಟು_ಸಮಯ = ಒಟ್ಟು['time']
avg_dram_bw = (ಒಟ್ಟು['read'] + ಒಟ್ಟು['write']) / (ಒಟ್ಟು_ಸಮಯ / 1000.0)
avg_tensor_util = ಒಟ್ಟು['tensor'] / ಒಟ್ಟು_ಸಮಯ
ಮುದ್ರಿಸು("." * ಉದ್ದ(units))
ಮುದ್ರಿಸು(f"           {'Total':<40} {total['time']:8.2f} {avg_dram_bw:8.1f} {avg_tensor_util:8.1f} {total['read']:8.2f} {total['write']:8.2f} {total['l2_read']:8.2f} {total['l2_write']:8.2f} {total['inst']:8.2f}")

ಮುದ್ರಿಸು()
ಮುದ್ರಿಸು("Kernel type summaries:")
ಮುದ್ರಿಸು(f"  {'name':<40} {'time':>6} {'frac':>6}  {'count':>6}")
ordered_ಸಮಯ = sorted(summaries.ವಸ್ತುಗಳು(), ಕೀಲಿ=ಅನಾಮಿಕ x: x[1], ಹಿಮ್ಮುಖ=ನಿಜ)
ಗಾಗಿ ನಮೂದು, ಮೌಲ್ಯ ಒಳಗೆ ordered_ಸಮಯ:
    # crop entry to be at most 40 characters
    ಆದರೆ ಉದ್ದ(ನಮೂದು) > 40:
        ನಮೂದು_ಪಠ್ಯ = ನಮೂದು[:37] + "..."
    ಇಲ್ಲದಿದ್ದರೆ:
        ನಮೂದು_ಪಠ್ಯ = ನಮೂದು
    ಮುದ್ರಿಸು(f"  {entry_text:<40} {value:6.2f} {100*value / total_time:6.2f}% {counts[entry]:>6d}")


ts = ಒಟ್ಟು_ಸಮಯ / 1000
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
ಮುದ್ರಿಸು(summary)