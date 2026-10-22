# runs profiling with ncu, generates a `profile.ncu-rep` for viewing with NSight Compute, and prints out
# basic kernel stats.
# Note: If you run into errors because of missing access rights to performance counters, try
# https://developer.nvidia.com/nvidia-development-tools-solutions-err_nvgpuctrperm-permission-issue-performance-counters#SolnAdminTag

আনযন subprocess
আনযন csv
তঃ collections আনযন defaultdict
আনযন shutil

# find ncu: Is it on PATH?
NCU = shutil.which("ncu")
# otherwise, guess a standard location
যদি NCU is None:
    NCU = "/usr/local/cuda/bin/ncu"

# build the executable
subprocess.check_call(["make", "profile_gpt2cu", "NO_MULTI_GPU=1", "USE_CUDNN=1"])

# try to see if profiling is allowed for non-root:
options = subprocess.check_output(["modprobe", "-c", "nvidia"], text=সঁচা)
can_profile = len([l বাবে l অন্তঃ options.splitlines() যদি "NVreg_RestrictProfilingToAdminUsers=0" অন্তঃ l]) != 0

# record metrics
# --full and --import-source are entirely superfluous for this script, but you might want to
# manually inspect `profile.ncu-rep`, so we keep it here
cmd = [NCU, "--set", "full", "--import-source", "yes", "-o", "profile", "-f", "./profile_gpt2cu"]
# do we need to run under sudo
যদি নহয় can_profile:
    print("NVreg_RestrictProfilingToAdminUsers=1, running with sudo")
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
result = subprocess.check_output(cmd, text=সঁচা).strip()

reader = csv.reader(result.splitlines(keepends=সঁচা))

# model config
CLS_START = -1
CLS_অঙ্ক = 6
N_LAYERS = 12

summaries = defaultdict(অনামন্: 0.0)
counts = defaultdict(অনামন্: 0)
passes = defaultdict(অনামন্: 0.0)
total = defaultdict(অনামন্: 0.0)
no_cutlass = 0.0
CC = ""
phase = "fwd"

kernel_profile_data = list(enumerate(reader))

বাবে rid, row অন্তঃ kernel_profile_data:
    যদি rid <= 2:
        অনুৱর্ত
    kernel = row[4]
    kid = rid - 2
    যদি "fused_classifier" অন্তঃ kernel:
        #  classifier: layernorm -> matmul -> fused -> bw matmul (x2) -> bw layernorm
        CLS_START = kid - 2

প্রতিজ্ঞা CLS_START != -1

# Check every kernel to find the maximum DRAM bandwidth and Tensor Core utilisation values
max_dram_bw = 0.0
max_tensor = 0.0
বাবে rid, row অন্তঃ kernel_profile_data:
    যদি rid <= 2:
        অনুৱর্ত
    time = ভিন্ন(row[13])
    পঠ = ভিন্ন(row[11])
    লিখ = ভিন্ন(row[12])
    tensor = ভিন্ন(row[16])
    dram_bw = (পঠ + লিখ) / (time / 1000.0)
    max_dram_bw = max(max_dram_bw, dram_bw)
    max_tensor = max(max_tensor, tensor)

# round the maximum tensor core utilisation to 50% or 100%
# consumer GPUs can only achieve 50% of peak tensor throughput on this counter
# and for GPUs without tensor cores, we set the value to 50% to avoid division by zero
max_tensor = (max_tensor > 50.0) আৰু 100.0 বা 50.0

print()
print("Kernel calls:")
বাবে rid, row অন্তঃ kernel_profile_data:
    যদি rid == 0:
        #  headings
        print(  f"id pass    {'name':<40} {'time':>8} {'RAM BW':>8} {'tensor':>8} {'RAM rd':>8} {'RAM wt':>8} {'L2 rd':>8} {'L2 wt':>8} {'inst':>8}")
        অনুৱর্ত
    যদি rid == 1:
        # units
        units = f"           {'':<40} {'ms':>8} {'GB/s':>8} {'core %':>8} {'GiB':>8} {'GiB':>8} {'GiB':>8} {'GiB':>8} {'MInst':>8}"
        print(units)
        print("." * len(units))
        অনুৱর্ত
    যদি rid == 2:
        CC = row[10]

    # actual data
    kernel = row[4]
    time = ভিন্ন(row[13])
    পঠ = ভিন্ন(row[11])
    লিখ = ভিন্ন(row[12])
    l2_read = ভিন্ন(row[14])
    l2_write = ভিন্ন(row[15])
    tensor = ভিন্ন(row[16])
    inst = ভিন্ন(row[17]) / 1e6
    dram_bw = (পঠ + লিখ) / (time / 1000.0)

    kid = rid - 2

    multiplier = 1
    যদি "encoder" অন্তঃ kernel:
        pass_name = "enc"
        যদি phase == "bwd":
            phase = "bwd-enc"
    নোচেদ্যদি CLS_START <= kid < CLS_START + CLS_অঙ্ক:
        # the classifier part, counts only once
        pass_name = "cls"
        phase = "bwd"
    নোচেদ্যদি "adamw" অন্তঃ kernel বা "global_norm" অন্তঃ kernel বা "copy_and_cast" অন্তঃ kernel:
        # encoder layer or adam
        pass_name = "opt"
    # before the first optimizer run, we create weight copies.
    # they aren't part of regular processing, so they get a multiplier
    # of zero
    নোচেদ্যদি phase == "bwd-enc":
        pass_name = "init"
        multiplier = 0
    অন্যথা:
        pass_name = phase
        multiplier = N_LAYERS
        time *= N_LAYERS
        পঠ *= N_LAYERS
        লিখ *= N_LAYERS
        l2_read *= N_LAYERS
        l2_write *= N_LAYERS
        inst *= N_LAYERS

    # split at "(" -- argument list
    fn_name = kernel.split("(")[0]
    # some names include the return value, others don't?
    যদি " " অন্তঃ fn_name:
        fn_name = fn_name.split(" ")[1]
    যদি "<" অন্তঃ fn_name:
        fn_name = fn_name.split("<")[0]

    # group together matmul kernels
    যদি "cutlass" অন্তঃ fn_name:
        pass
    নোচেদ্যদি fn_name.startswith("ampere_bf16"):
        fn_name = "ampere_bf16"
    নোচেদ্যদি fn_name.startswith("cudnn_generated_fort_native_sdpa"):
        fn_name = "cudnn_generated_fort_native_sdpa"
    অন্যথা:
        no_cutlass += time

    # convert L2 to GiB
    l2_read = l2_read * 32 / 1024 / 1024 / 1024
    l2_write = l2_write * 32 / 1024 / 1024 / 1024

    efficiency = max(dram_bw / max_dram_bw, tensor / max_tensor)
    summaries[fn_name] += time
    counts[fn_name] += multiplier
    passes[pass_name] += time
    যদি pass_name != "init":
        total['time'] += time
        total['read'] += পঠ
        total['write'] += লিখ
        total['l2_read'] += l2_read
        total['l2_write'] += l2_write
        total['inst'] += inst
        total['tensor'] += tensor * time # % so multiplied by time
        total['efficiency'] += efficiency * time

    pass_info = f"{pass_name}×{multiplier}"
    print(f"{kid:02} {pass_info:7} {fn_name:<40} {time:8.2f} {dram_bw:8.1f} {tensor:8.1f} {read:8.2f} {write:8.2f} {l2_read:8.2f} {l2_write:8.2f} {inst:8.2f}")


total_time = total['time']
avg_dram_bw = (total['read'] + total['write']) / (total_time / 1000.0)
avg_tensor_util = total['tensor'] / total_time
print("." * len(units))
print(f"           {'Total':<40} {total['time']:8.2f} {avg_dram_bw:8.1f} {avg_tensor_util:8.1f} {total['read']:8.2f} {total['write']:8.2f} {total['l2_read']:8.2f} {total['l2_write']:8.2f} {total['inst']:8.2f}")

print()
print("Kernel type summaries:")
print(f"  {'name':<40} {'time':>6} {'frac':>6}  {'count':>6}")
ordered_time = sorted(summaries.বস্তুসমূহ(), চাবি=অনামন্ x: x[1], reverse=সঁচা)
বাবে entry, মান অন্তঃ ordered_time:
    # crop entry to be at most 40 characters
    যদি len(entry) > 40:
        entry_text = entry[:37] + "..."
    অন্যথা:
        entry_text = entry
    print(f"  {entry_text:<40} {value:6.2f} {100*value / total_time:6.2f}% {counts[entry]:>6d}")


ts = total_time / 1000
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
print(summary)