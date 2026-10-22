# runs profiling with ncu, generates a `profile.ncu-rep` for viewing with NSight Compute, and prints out
# basic kernel stats.
# Note: If you run into errors because of missing access rights to performance counters, try
# https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters#SolnAdminTag

درآمد subprocess
درآمد csv
سے collections درآمد defaultdict
درآمد shutil

# find ncu: Is it on PATH?
NCU = shutil.which("ncu")
# otherwise, guess a standard location
اگر NCU ہے کچھ_نہیں:
    NCU = "/usr/local/cuda/bin/ncu"

# build the executable
subprocess.check_call(["make", "profile_gpt2cu", "NO_MULTI_GPU=1", "USE_CUDNN=1"])

# try to see if profiling is allowed for non-root:
options = subprocess.check_output(["modprobe", "-c", "nvidia"], متن=سچ)
can_profile = لمبائی([l برائے l میں options.splitlines() اگر "NVreg_RestrictProfilingToAdminUsers=0" میں l]) != 0

# record metrics
# --full and --import-source are entirely superfluous for this script, but you might want to
# manually inspect `profile.ncu-rep`, so we keep it here
cmd = [NCU, "--set", "full", "--import-source", "yes", "-o", "profile", "-f", "./profile_gpt2cu"]
# do we need to run under sudo
اگر نہیں can_profile:
    چھاپو("NVreg_RestrictProfilingToAdminUsers=1, running with sudo")
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
نتیجہ = subprocess.check_output(cmd, متن=سچ).strip()

reader = csv.reader(نتیجہ.splitlines(keepends=سچ))

# model config
CLS_آغاز = -1
CLS_عدد = 6
N_LAYERS = 12

summaries = defaultdict(بےنام: 0.0)
counts = defaultdict(بےنام: 0)
passes = defaultdict(بےنام: 0.0)
کل = defaultdict(بےنام: 0.0)
no_cutlass = 0.0
CC = ""
phase = "fwd"

kernel_profile_ڈیٹا = list(enumerate(reader))

برائے rid, صف میں kernel_profile_ڈیٹا:
    اگر rid <= 2:
        جاری
    kernel = صف[4]
    kid = rid - 2
    اگر "fused_classifier" میں kernel:
        #  classifier: layernorm -> matmul -> fused -> bw matmul (x2) -> bw layernorm
        CLS_آغاز = kid - 2

أكد CLS_آغاز != -1

# Check every kernel to find the maximum DRAM bandwidth and Tensor Core utilisation values
max_dram_bw = 0.0
max_tensor = 0.0
برائے rid, صف میں kernel_profile_ڈیٹا:
    اگر rid <= 2:
        جاری
    time = اعشاری(صف[13])
    اقرأ = اعشاری(صف[11])
    اكتب = اعشاری(صف[12])
    tensor = اعشاری(صف[16])
    dram_bw = (اقرأ + اكتب) / (time / 1000.0)
    max_dram_bw = max(max_dram_bw, dram_bw)
    max_tensor = max(max_tensor, tensor)

# round the maximum tensor core utilisation to 50% or 100%
# consumer GPUs can only achieve 50% of peak tensor throughput on this counter
# and for GPUs without tensor cores, we set the value to 50% to avoid division by zero
max_tensor = (max_tensor > 50.0) اور 100.0 یا 50.0

چھاپو()
چھاپو("Kernel calls:")
برائے rid, صف میں kernel_profile_ڈیٹا:
    اگر rid == 0:
        #  headings
        چھاپو(  f"id pass    {'name':<40} {'time':>8} {'RAM BW':>8} {'tensor':>8} {'RAM rd':>8} {'RAM wt':>8} {'L2 rd':>8} {'L2 wt':>8} {'inst':>8}")
        جاری
    اگر rid == 1:
        # units
        units = f"           {'':<40} {'ms':>8} {'GB/s':>8} {'core %':>8} {'GiB':>8} {'GiB':>8} {'GiB':>8} {'GiB':>8} {'MInst':>8}"
        چھاپو(units)
        چھاپو("." * لمبائی(units))
        جاری
    اگر rid == 2:
        CC = صف[10]

    # actual data
    kernel = صف[4]
    time = اعشاری(صف[13])
    اقرأ = اعشاری(صف[11])
    اكتب = اعشاری(صف[12])
    l2_read = اعشاری(صف[14])
    l2_write = اعشاری(صف[15])
    tensor = اعشاری(صف[16])
    inst = اعشاری(صف[17]) / 1e6
    dram_bw = (اقرأ + اكتب) / (time / 1000.0)

    kid = rid - 2

    multiplier = 1
    اگر "encoder" میں kernel:
        pass_name = "enc"
        اگر phase == "bwd":
            phase = "bwd-enc"
    وگرنہ_اگر CLS_آغاز <= kid < CLS_آغاز + CLS_عدد:
        # the classifier part, counts only once
        pass_name = "cls"
        phase = "bwd"
    وگرنہ_اگر "adamw" میں kernel یا "global_norm" میں kernel یا "copy_and_cast" میں kernel:
        # encoder layer or adam
        pass_name = "opt"
    # before the first optimizer run, we create weight copies.
    # they aren't part of regular processing, so they get a multiplier
    # of zero
    وگرنہ_اگر phase == "bwd-enc":
        pass_name = "init"
        multiplier = 0
    ورنہ:
        pass_name = phase
        multiplier = N_LAYERS
        time *= N_LAYERS
        اقرأ *= N_LAYERS
        اكتب *= N_LAYERS
        l2_read *= N_LAYERS
        l2_write *= N_LAYERS
        inst *= N_LAYERS

    # split at "(" -- argument list
    fn_name = kernel.بانٹو("(")[0]
    # some names include the return value, others don't?
    اگر " " میں fn_name:
        fn_name = fn_name.بانٹو(" ")[1]
    اگر "<" میں fn_name:
        fn_name = fn_name.بانٹو("<")[0]

    # group together matmul kernels
    اگر "cutlass" میں fn_name:
        گزرو
    وگرنہ_اگر fn_name.startswith("ampere_bf16"):
        fn_name = "ampere_bf16"
    وگرنہ_اگر fn_name.startswith("cudnn_generated_fort_native_sdpa"):
        fn_name = "cudnn_generated_fort_native_sdpa"
    ورنہ:
        no_cutlass += time

    # convert L2 to GiB
    l2_read = l2_read * 32 / 1024 / 1024 / 1024
    l2_write = l2_write * 32 / 1024 / 1024 / 1024

    efficiency = max(dram_bw / max_dram_bw, tensor / max_tensor)
    summaries[fn_name] += time
    counts[fn_name] += multiplier
    passes[pass_name] += time
    اگر pass_name != "init":
        کل['time'] += time
        کل['read'] += اقرأ
        کل['write'] += اكتب
        کل['l2_read'] += l2_read
        کل['l2_write'] += l2_write
        کل['inst'] += inst
        کل['tensor'] += tensor * time # % so multiplied by time
        کل['efficiency'] += efficiency * time

    pass_info = f"{pass_name}×{multiplier}"
    چھاپو(f"{kid:02} {pass_info:7} {fn_name:<40} {time:8.2f} {dram_bw:8.1f} {tensor:8.1f} {read:8.2f} {write:8.2f} {l2_read:8.2f} {l2_write:8.2f} {inst:8.2f}")


کل_وقت = کل['time']
avg_dram_bw = (کل['read'] + کل['write']) / (کل_وقت / 1000.0)
avg_tensor_util = کل['tensor'] / کل_وقت
چھاپو("." * لمبائی(units))
چھاپو(f"           {'Total':<40} {total['time']:8.2f} {avg_dram_bw:8.1f} {avg_tensor_util:8.1f} {total['read']:8.2f} {total['write']:8.2f} {total['l2_read']:8.2f} {total['l2_write']:8.2f} {total['inst']:8.2f}")

چھاپو()
چھاپو("Kernel type summaries:")
چھاپو(f"  {'name':<40} {'time':>6} {'frac':>6}  {'count':>6}")
ordered_وقت = sorted(summaries.اشیاء(), کلید=بےنام x: x[1], الٹ=سچ)
برائے اندراج_2, قدر میں ordered_وقت:
    # crop entry to be at most 40 characters
    اگر لمبائی(اندراج_2) > 40:
        اندراج_متن = اندراج_2[:37] + "..."
    ورنہ:
        اندراج_متن = اندراج_2
    چھاپو(f"  {entry_text:<40} {value:6.2f} {100*value / total_time:6.2f}% {counts[entry]:>6d}")


ts = کل_وقت / 1000
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
چھاپو(summary)