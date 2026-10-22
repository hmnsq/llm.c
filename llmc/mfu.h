#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ MFU_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ MFU_H

#ਅਨ੍ਤਰ੍ਭਾਵ <stdio.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdlib.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <string.h>
#ਜੇ __has_include(<nvml.h>)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ USE_NVML 1
#ਅਨ੍ਤਰ੍ਭਾਵ <nvml.h>
#ਨਹੀਂਤਾਂ
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ USE_NVML 0
#ਜੇ_ਅੰਤ

// tied to enum PrecisionMode, in a future refactor make them the same
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ MFUH_PRECISION_FP32 0
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ MFUH_PRECISION_FP16 1
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ MFUH_PRECISION_BF16 2

#ਜੇ USE_NVML
ਸਤਰਅੰਦਰ ਖਾਲੀ nvml_check(nvmlReturn_t ਸਥਿਤੀ, ਅਚਲ ਅੱਖਰ *file, ਪੂਰਨ ਸਤਰ) {
    ਜੇ (ਸਥਿਤੀ != NVML_SUCCESS) {
        ਲਿਖੋ("[NVML ERROR] at file %s:%d:\n%s\n", file, ਸਤਰ, nvmlErrorString(ਸਥਿਤੀ));
        ਬਾਹਰ_ਨਿਕਲੋ(EXIT_FAILURE);
    }
};
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ nvmlCheck(err) (nvml_check(err, __FILE__, __LINE__))
#ਜੇ_ਅੰਤ


ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    ਦਸ਼ਮਲਵ TF_32;       // tensor-core performance 32 bit
    ਦਸ਼ਮਲਵ BF_16_32;    // bf16 with 32 bit accumulate
    ਦਸ਼ਮਲਵ FP_16_32;    // fp16 with 32 bit accumulate
    ਦਸ਼ਮਲਵ FP_16_16;    // fp16 with 16 bit accumulate
    ਦਸ਼ਮਲਵ FP_8_32;     // and so on
    ਦਸ਼ਮਲਵ FP_8_16;
    ਦਸ਼ਮਲਵ CLOCK;        // clock frequency from the spec sheet
    ਦਸ਼ਮਲਵ CORES;        // #TCs from the spec sheet
} Perf_ਅੰਕੜੇ;

// basic default data from the nvidia whitepapers
ਸਥਿਰ ਅਚਲ Perf_ਅੰਕੜੇ VOLTA = {125.0f, -1.f, 125.f, -1.f, -1.f, -1.f, 1530.f, 640.f};
ਸਥਿਰ ਅਚਲ Perf_ਅੰਕੜੇ AMPERE_DATACENTER = {156.f, 312.f, 312.f, 312.f, -1.f, -1.f, 1410.f, 432.f};
ਸਥਿਰ ਅਚਲ Perf_ਅੰਕੜੇ AMPERE_ਖਪਤਕਾਰ = {40.f, 80.f, 80.f, 160.f, -1.f, -1.f, 1860.f, 336.f};
ਸਥਿਰ ਅਚਲ Perf_ਅੰਕੜੇ HOPPER = {378.f, 756.f, 756.f, 756.f, 1513.f, 1513.f, 1620.f, 456.f};
ਸਥਿਰ ਅਚਲ Perf_ਅੰਕੜੇ ADA = {82.6f, 165.2f, 165.2f, 330.3f, 330.3f, 660.6f, 2520.f, 512.f};

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    ਅਚਲ ਅੱਖਰ* name;
    ਅਚਲ Perf_ਅੰਕੜੇ* perf_ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ new_cores;
    ਦਸ਼ਮਲਵ new_mhz;
} GPUEntry;

// the overrides for each specific GPU
ਸਥਿਰ GPUEntry gpu_db[] = {
    {"Tesla V100-SXM2-16GB", &VOLTA, 640, 1530},
    {"Tesla V100-PCIE-32GB", &VOLTA, 640, 1530},
    {"NVIDIA A100-PCIE-40GB", &AMPERE_DATACENTER, 432, 1410},
    {"NVIDIA A100-PCIE-80GB", &AMPERE_DATACENTER, 432, 1410},
    {"NVIDIA A100-SXM4-40GB", &AMPERE_DATACENTER, 432, 1410},
    {"NVIDIA A100-SXM4-80GB", &AMPERE_DATACENTER, 432, 1410},
    {"NVIDIA RTX A2000", &AMPERE_ਖਪਤਕਾਰ, 104, 1200},
    {"NVIDIA RTX A4000", &AMPERE_ਖਪਤਕਾਰ, 192, 1560},
    {"NVIDIA RTX A4500", &AMPERE_ਖਪਤਕਾਰ, 224, 1650},
    {"NVIDIA RTX A5000", &AMPERE_ਖਪਤਕਾਰ, 256, 1695},
    {"NVIDIA RTX A5500", &AMPERE_ਖਪਤਕਾਰ, 320, 1770},
    {"NVIDIA RTX A6000", &AMPERE_ਖਪਤਕਾਰ, 336, 1800},
    {"NVIDIA GeForce RTX 3090 Ti", &AMPERE_ਖਪਤਕਾਰ, 336, 1860},
    {"NVIDIA GeForce RTX 3090", &AMPERE_ਖਪਤਕਾਰ, 328, 1695},
    {"NVIDIA GeForce RTX 3080 Ti", &AMPERE_ਖਪਤਕਾਰ, 320, 1665},
    {"NVIDIA GeForce RTX 3080", &AMPERE_ਖਪਤਕਾਰ, 272, 1710},
    {"NVIDIA GeForce RTX 3070 Ti", &AMPERE_ਖਪਤਕਾਰ, 192, 1770},
    {"NVIDIA GeForce RTX 3070", &AMPERE_ਖਪਤਕਾਰ, 184, 1725},
    {"NVIDIA GeForce RTX 3060 Ti", &AMPERE_ਖਪਤਕਾਰ, 152, 1665},
    {"NVIDIA GeForce RTX 3060", &AMPERE_ਖਪਤਕਾਰ, 112, 1777},
    {"NVIDIA RTX A2000 ADA", &ADA, 88, 2130},
    {"NVIDIA RTX A4000 ADA", &ADA, 192, 2175},
    {"NVIDIA RTX A4500 ADA", &ADA, 224, 2580},
    {"NVIDIA RTX A5000 ADA", &ADA, 400, 2550},
    {"NVIDIA RTX A5880 ADA", &ADA, 440, 2460},
    {"NVIDIA RTX A6000 ADA", &ADA, 568, 2505},
    {"NVIDIA GeForce RTX 4090", &ADA, 512, 2520},
    {"NVIDIA GeForce RTX 4080 SUPER", &ADA, 320, 2550},
    {"NVIDIA GeForce RTX 4080", &ADA, 304, 2505},
    {"NVIDIA GeForce RTX 4070 Ti SUPER", &ADA, 264, 2610},
    {"NVIDIA GeForce RTX 4070 Ti", &ADA, 240, 2610},
    {"NVIDIA GeForce RTX 4070 SUPER", &ADA, 224, 2475},
    {"NVIDIA GeForce RTX 4070", &ADA, 184, 2475},
    {"NVIDIA GeForce RTX 4070", &ADA, 184, 2475},
    {"NVIDIA GeForce RTX 4060 Ti", &ADA, 136, 2535},
    {"NVIDIA GeForce RTX 4060", &ADA, 96, 2460},
    {"NVIDIA H100 PCIe", &HOPPER, 456, 1620},
    {"NVIDIA H100 80GB HBM3", &HOPPER, 528, 1830}, // HBM3 = SXM5
};

ਦਸ਼ਮਲਵ get_flops_promised(ਅਚਲ ਅੱਖਰ* device, ਪੂਰਨ precision_ਬਹੁਲਕ) {
    /*
    This function is used to estimate the Model Flops Utilization (MFU)
    basically we have to figure out how many flops the GPU can do per second.
    Note that this is not a simple endeavor and may well go wrong! The details are tricky.
    The returned value is in units of 1e12.

    For the non-top models, actual performance numbers aren't that easy to find, e.g.,
    here https://www.techpowerup.com/gpu-specs/rtx-a4000.c3756, does "Theoretical Performance"
    seems to be without tensor cores.

    So, instead we use that all these cards just use the same types of tensor cores in different
    numbers and at different frequencies. Then we just need to look up these two easily accesible
    numbers for all the other GPUs.
    linear scaling seems to work: comparing spec sheet and calculation:
    4080: 304TCs, 2505 GHz; 97.5TFlops = 165.2/512*304 /2520 * 2505

    Original numbers for the top GPUS are from.
    https://resources.nvidia.com/en-us-tensor-core
    https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf
    */

   // validate the precision mode as one of the three possible values
    ਜੇ (!(precision_ਬਹੁਲਕ == MFUH_PRECISION_FP32 || precision_ਬਹੁਲਕ == MFUH_PRECISION_FP16 || precision_ਬਹੁਲਕ == MFUH_PRECISION_BF16)) {
        fprintf(stderr, "Invalid precision mode: %d\n", precision_ਬਹੁਲਕ);
        ਵਾਪਸ -1.0f;
    }

    // do a linear search until you find our GPU, then calculate the flops promised
    ਪੂਰਨ ਅੰਕ_gpu_entries = ਆਕਾਰ(gpu_db) / ਆਕਾਰ(gpu_db[0]);
    ਲਈ (ਪੂਰਨ i = 0; i < ਅੰਕ_gpu_entries; i++) {
        ਜੇ (strcmp(gpu_db[i].name, device) == 0) {
            ਅਚਲ Perf_ਅੰਕੜੇ* perf_ਅੰਕੜੇ = gpu_db[i].perf_ਅੰਕੜੇ;

            // look up the default flops value for the given precision mode
            ਦਸ਼ਮਲਵ ਮੁੱਲ = -1.0f;
            ਜੇ (precision_ਬਹੁਲਕ == MFUH_PRECISION_BF16) { ਮੁੱਲ = perf_ਅੰਕੜੇ->BF_16_32; }
            ਜੇ (precision_ਬਹੁਲਕ == MFUH_PRECISION_FP32) { ਮੁੱਲ = perf_ਅੰਕੜੇ->TF_32; }
            ਜੇ (precision_ਬਹੁਲਕ == MFUH_PRECISION_FP16) { ਮੁੱਲ = perf_ਅੰਕੜੇ->FP_16_32; }

            // we'd get here if we're e.g. trying to use BF16 on Volta GPU or something...
            ਜੇ (ਮੁੱਲ < 0.0f) {
                fprintf(stderr, "No data for GPU %s and precision mode %d\n", device, precision_ਬਹੁਲਕ);
                ਵਾਪਸ -1.0f;
            }

            // adjust flops based on the specific core count and clock frequency of this GPU
            ਦਸ਼ਮਲਵ new_cores = gpu_db[i].new_cores;
            ਦਸ਼ਮਲਵ new_mhz = gpu_db[i].new_mhz;
            ਦਸ਼ਮਲਵ adjusted = ਮੁੱਲ * (new_cores / perf_ਅੰਕੜੇ->CORES) * (new_mhz / perf_ਅੰਕੜੇ->CLOCK);
            ਵਾਪਸ adjusted;
        }
    }

    ਵਾਪਸ -1.0f; // ¯\_(ツ)_/¯
}

ਬਣਤਰ GPUUtilInfo {
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ clock;
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ max_clock;
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ power;
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ power_ਹੱਦਬੰਦੀ;
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ fan;
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ temperature;
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ਆਰਜ਼ੀ_slowdown;

    ਦਸ਼ਮਲਵ gpu_utilization;
    ਦਸ਼ਮਲਵ mem_utilization;
    ਅਚਲ ਅੱਖਰ* throttle_reason;
};

// lazily initialize nvml and generate a handle to the GPU
#ਜੇ USE_NVML
nvmlDevice_t nvml_get_device() {
    ਸਥਿਰ bool needs_init = true;
    ਸਥਿਰ nvmlDevice_t device;
    ਜੇ(needs_init) {
        needs_init = false;
        nvmlCheck(nvmlInit());
        nvmlCheck(nvml_Device_Get_Handle_By_ਸੂਚਕਾਂਕ_v2(0, &device));
    }
    ਵਾਪਸ device;
}

// convert throttle reason bitfield into a text reason.
// this is a lossy conversion; we just want to give some idea of what is happening
ਅਚਲ ਅੱਖਰ* get_throttle_reason(ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਲੰਮਾ ਬਿੱਟ) {
    ਜੇ(ਬਿੱਟ & (nvmlClocksThrottleReasonSwPowerCap | nvmlClocksThrottleReasonHwPowerBrakeSlowdown)) {
        ਵਾਪਸ "power cap";
    } ਨਹੀਂਤਾਂ ਜੇ (ਬਿੱਟ & (nvmlClocksThrottleReasonSwThermalSlowdown | nvmlClocksThrottleReasonHwThermalSlowdown)) {
        ਵਾਪਸ "thermal cap";
    } ਨਹੀਂਤਾਂ ਜੇ (ਬਿੱਟ & (nvmlClocksThrottleReasonAll)) {
        ਵਾਪਸ "other cap";
    } ਨਹੀਂਤਾਂ {
        ਵਾਪਸ "no cap";
    }
}

// gather data for a GPUUtilInfo object
GPUUtilInfo get_gpu_utilization_info() {
    GPUUtilInfo info;
    nvmlDevice_t device = nvml_get_device();
    // query different infos directly
    nvmlCheck(nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &info.clock));
    nvmlCheck(nvmlDeviceGetMaxClockInfo(device, NVML_CLOCK_SM, &info.max_clock));
    nvmlCheck(nvml_Device_Get_Power_Management_ਹੱਦਬੰਦੀ(device, &info.power_ਹੱਦਬੰਦੀ));
    nvmlCheck(nvmlDeviceGetPowerUsage(device, &info.power));
    nvmlCheck(nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &info.temperature));
    nvmlCheck(nvml_Device_Get_Temperature_ਹੱਦ(device, NVML_TEMPERATURE_ਹੱਦ_SLOWDOWN, &info.ਆਰਜ਼ੀ_slowdown));
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਲੰਮਾ throttle;
    nvmlCheck(nvmlDeviceGetCurrentClocksThrottleReasons(device, &throttle));
    info.throttle_reason = get_throttle_reason(throttle);
    nvmlCheck(nvml_Device_Get_Fan_ਰਫਤਾਰ(device, &info.fan));

    // for "utilization", we look at recorded samples. In principle, we could query the driver for how many samples
    // to request, but then we'd need to dynamically allocate sufficient space. Let's just hard-code a limit of 128,
    // and have no memory management required
    constexpr ਅਚਲ ਪੂਰਨ ਬਫਰ_ਹੱਦਬੰਦੀ = 128;
    nvmlSample_t ਬਫਰ[ਬਫਰ_ਹੱਦਬੰਦੀ];
    nvml_ਮੁੱਲ_Type_t v_type;
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ sample_ਗਿਣਤੀ = ਬਫਰ_ਹੱਦਬੰਦੀ;
    nvmlCheck(nvmlDeviceGetSamples(device, NVML_GPU_UTILIZATION_SAMPLES, 0, &v_type, &sample_ਗਿਣਤੀ, ਬਫਰ));
    ਦਸ਼ਮਲਵ gpu_utilization = 0.f;
    ਲਈ(ਬਿਨਾਨਿਸ਼ਾਨੀ i = 0; i < sample_ਗਿਣਤੀ; ++i) {
        gpu_utilization += (ਦਸ਼ਮਲਵ)ਬਫਰ[i].sample_ਮੁੱਲ.uiVal;
    }
    gpu_utilization /= (ਦਸ਼ਮਲਵ)sample_ਗਿਣਤੀ;

    // sample count may have been modified by the query above; reset back to buffer size
    sample_ਗਿਣਤੀ = ਬਫਰ_ਹੱਦਬੰਦੀ;
    nvmlCheck(nvmlDeviceGetSamples(device, NVML_ਸਮ੍ਰਿਤੀ_UTILIZATION_SAMPLES, 0, &v_type, &sample_ਗਿਣਤੀ, ਬਫਰ));
    ਦਸ਼ਮਲਵ mem_utilization = 0.f;
    ਲਈ(ਬਿਨਾਨਿਸ਼ਾਨੀ i = 0; i < sample_ਗਿਣਤੀ; ++i) {
        mem_utilization += (ਦਸ਼ਮਲਵ)ਬਫਰ[i].sample_ਮੁੱਲ.uiVal;
    }
    mem_utilization /= (ਦਸ਼ਮਲਵ)sample_ਗਿਣਤੀ;

    info.gpu_utilization = gpu_utilization;
    info.mem_utilization = mem_utilization;
    ਵਾਪਸ info;
}
#ਨਹੀਂਤਾਂ
GPUUtilInfo get_gpu_utilization_info() {
    fprintf(stderr, "Error: Compiled without nvml support. Cannot perform additional GPU state tracking.");
    ਬਾਹਰ_ਨਿਕਲੋ(EXIT_FAILURE);
}
#ਜੇ_ਅੰਤ
#ਜੇ_ਅੰਤ // MFU_H
