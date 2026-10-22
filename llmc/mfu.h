#ifndef MFU_H
#define MFU_H

#அந்தர்பாவ <stdio.h>
#அந்தர்பாவ <stdlib.h>
#அந்தர்பாவ <string.h>
#எனில் __has_include(<nvml.h>)
#define USE_NVML 1
#அந்தர்பாவ <nvml.h>
#இல்லையெனில்
#define USE_NVML 0
#endif

// tied to enum PrecisionMode, in a future refactor make them the same
#define MFUH_PRECISION_FP32 0
#define MFUH_PRECISION_FP16 1
#define MFUH_PRECISION_BF16 2

#எனில் USE_NVML
உள்வரி வெறுமை nvml_check(nvmlReturn_t status, மாறிலி எழுத்து *file, முழு வரி) {
    எனில் (status != NVML_SUCCESS) {
        printf("[NVML ERROR] at file %s:%d:\n%s\n", file, வரி, nvmlErrorString(status));
        exit(EXIT_FAILURE);
    }
};
#define nvmlCheck(err) (nvml_check(err, __FILE__, __LINE__))
#endif


வகைவரை கட்டமைப்பு {
    தசம TF_32;       // tensor-core performance 32 bit
    தசம BF_16_32;    // bf16 with 32 bit accumulate
    தசம FP_16_32;    // fp16 with 32 bit accumulate
    தசம FP_16_16;    // fp16 with 16 bit accumulate
    தசம FP_8_32;     // and so on
    தசம FP_8_16;
    தசம CLOCK;        // clock frequency from the spec sheet
    தசம CORES;        // #TCs from the spec sheet
} PerfData;

// basic default data from the nvidia whitepapers
நிலையான மாறிலி PerfData VOLTA = {125.0f, -1.f, 125.f, -1.f, -1.f, -1.f, 1530.f, 640.f};
நிலையான மாறிலி PerfData AMPERE_DATACENTER = {156.f, 312.f, 312.f, 312.f, -1.f, -1.f, 1410.f, 432.f};
நிலையான மாறிலி PerfData AMPERE_CONSUMER = {40.f, 80.f, 80.f, 160.f, -1.f, -1.f, 1860.f, 336.f};
நிலையான மாறிலி PerfData HOPPER = {378.f, 756.f, 756.f, 756.f, 1513.f, 1513.f, 1620.f, 456.f};
நிலையான மாறிலி PerfData ADA = {82.6f, 165.2f, 165.2f, 330.3f, 330.3f, 660.6f, 2520.f, 512.f};

வகைவரை கட்டமைப்பு {
    மாறிலி எழுத்து* name;
    மாறிலி PerfData* perf_data;
    தசம new_cores;
    தசம new_mhz;
} GPUEntry;

// the overrides for each specific GPU
நிலையான GPUEntry gpu_db[] = {
    {"Tesla V100-SXM2-16GB", &VOLTA, 640, 1530},
    {"Tesla V100-PCIE-32GB", &VOLTA, 640, 1530},
    {"NVIDIA A100-PCIE-40GB", &AMPERE_DATACENTER, 432, 1410},
    {"NVIDIA A100-PCIE-80GB", &AMPERE_DATACENTER, 432, 1410},
    {"NVIDIA A100-SXM4-40GB", &AMPERE_DATACENTER, 432, 1410},
    {"NVIDIA A100-SXM4-80GB", &AMPERE_DATACENTER, 432, 1410},
    {"NVIDIA RTX A2000", &AMPERE_CONSUMER, 104, 1200},
    {"NVIDIA RTX A4000", &AMPERE_CONSUMER, 192, 1560},
    {"NVIDIA RTX A4500", &AMPERE_CONSUMER, 224, 1650},
    {"NVIDIA RTX A5000", &AMPERE_CONSUMER, 256, 1695},
    {"NVIDIA RTX A5500", &AMPERE_CONSUMER, 320, 1770},
    {"NVIDIA RTX A6000", &AMPERE_CONSUMER, 336, 1800},
    {"NVIDIA GeForce RTX 3090 Ti", &AMPERE_CONSUMER, 336, 1860},
    {"NVIDIA GeForce RTX 3090", &AMPERE_CONSUMER, 328, 1695},
    {"NVIDIA GeForce RTX 3080 Ti", &AMPERE_CONSUMER, 320, 1665},
    {"NVIDIA GeForce RTX 3080", &AMPERE_CONSUMER, 272, 1710},
    {"NVIDIA GeForce RTX 3070 Ti", &AMPERE_CONSUMER, 192, 1770},
    {"NVIDIA GeForce RTX 3070", &AMPERE_CONSUMER, 184, 1725},
    {"NVIDIA GeForce RTX 3060 Ti", &AMPERE_CONSUMER, 152, 1665},
    {"NVIDIA GeForce RTX 3060", &AMPERE_CONSUMER, 112, 1777},
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

தசம get_flops_promised(மாறிலி எழுத்து* device, முழு precision_mode) {
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
    எனில் (!(precision_mode == MFUH_PRECISION_FP32 || precision_mode == MFUH_PRECISION_FP16 || precision_mode == MFUH_PRECISION_BF16)) {
        fprintf(stderr, "Invalid precision mode: %d\n", precision_mode);
        திரும்பு -1.0f;
    }

    // do a linear search until you find our GPU, then calculate the flops promised
    முழு இலக்கம்_gpu_entries = அளவு(gpu_db) / அளவு(gpu_db[0]);
    க்கு (முழு i = 0; i < இலக்கம்_gpu_entries; i++) {
        எனில் (strcmp(gpu_db[i].name, device) == 0) {
            மாறிலி PerfData* perf_data = gpu_db[i].perf_data;

            // look up the default flops value for the given precision mode
            தசம மதிப்பு = -1.0f;
            எனில் (precision_mode == MFUH_PRECISION_BF16) { மதிப்பு = perf_data->BF_16_32; }
            எனில் (precision_mode == MFUH_PRECISION_FP32) { மதிப்பு = perf_data->TF_32; }
            எனில் (precision_mode == MFUH_PRECISION_FP16) { மதிப்பு = perf_data->FP_16_32; }

            // we'd get here if we're e.g. trying to use BF16 on Volta GPU or something...
            எனில் (மதிப்பு < 0.0f) {
                fprintf(stderr, "No data for GPU %s and precision mode %d\n", device, precision_mode);
                திரும்பு -1.0f;
            }

            // adjust flops based on the specific core count and clock frequency of this GPU
            தசம new_cores = gpu_db[i].new_cores;
            தசம new_mhz = gpu_db[i].new_mhz;
            தசம adjusted = மதிப்பு * (new_cores / perf_data->CORES) * (new_mhz / perf_data->CLOCK);
            திரும்பு adjusted;
        }
    }

    திரும்பு -1.0f; // ¯\_(ツ)_/¯
}

கட்டமைப்பு GPUUtilInfo {
    unsigned முழு clock;
    unsigned முழு max_clock;
    unsigned முழு power;
    unsigned முழு power_limit;
    unsigned முழு fan;
    unsigned முழு temperature;
    unsigned முழு temp_slowdown;

    தசம gpu_utilization;
    தசம mem_utilization;
    மாறிலி எழுத்து* throttle_reason;
};

// lazily initialize nvml and generate a handle to the GPU
#எனில் USE_NVML
nvmlDevice_t nvml_get_device() {
    நிலையான bool needs_init = true;
    நிலையான nvmlDevice_t device;
    எனில்(needs_init) {
        needs_init = false;
        nvmlCheck(nvmlInit());
        nvmlCheck(nvmlDeviceGetHandleByIndex_v2(0, &device));
    }
    திரும்பு device;
}

// convert throttle reason bitfield into a text reason.
// this is a lossy conversion; we just want to give some idea of what is happening
மாறிலி எழுத்து* get_throttle_reason(unsigned long long bits) {
    எனில்(bits & (nvmlClocksThrottleReasonSwPowerCap | nvmlClocksThrottleReasonHwPowerBrakeSlowdown)) {
        திரும்பு "power cap";
    } இல்லையெனில் எனில் (bits & (nvmlClocksThrottleReasonSwThermalSlowdown | nvmlClocksThrottleReasonHwThermalSlowdown)) {
        திரும்பு "thermal cap";
    } இல்லையெனில் எனில் (bits & (nvmlClocksThrottleReasonAll)) {
        திரும்பு "other cap";
    } இல்லையெனில் {
        திரும்பு "no cap";
    }
}

// gather data for a GPUUtilInfo object
GPUUtilInfo get_gpu_utilization_info() {
    GPUUtilInfo info;
    nvmlDevice_t device = nvml_get_device();
    // query different infos directly
    nvmlCheck(nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &info.clock));
    nvmlCheck(nvmlDeviceGetMaxClockInfo(device, NVML_CLOCK_SM, &info.max_clock));
    nvmlCheck(nvmlDeviceGetPowerManagementLimit(device, &info.power_limit));
    nvmlCheck(nvmlDeviceGetPowerUsage(device, &info.power));
    nvmlCheck(nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &info.temperature));
    nvmlCheck(nvmlDeviceGetTemperatureThreshold(device, NVML_TEMPERATURE_THRESHOLD_SLOWDOWN, &info.temp_slowdown));
    unsigned long long throttle;
    nvmlCheck(nvmlDeviceGetCurrentClocksThrottleReasons(device, &throttle));
    info.throttle_reason = get_throttle_reason(throttle);
    nvmlCheck(nvmlDeviceGetFanSpeed(device, &info.fan));

    // for "utilization", we look at recorded samples. In principle, we could query the driver for how many samples
    // to request, but then we'd need to dynamically allocate sufficient space. Let's just hard-code a limit of 128,
    // and have no memory management required
    constexpr மாறிலி முழு BUFFER_LIMIT = 128;
    nvmlSample_t buffer[BUFFER_LIMIT];
    nvml_மதிப்பு_Type_t v_type;
    unsigned முழு sample_மொத்தம் = BUFFER_LIMIT;
    nvmlCheck(nvmlDeviceGetSamples(device, NVML_GPU_UTILIZATION_SAMPLES, 0, &v_type, &sample_மொத்தம், buffer));
    தசம gpu_utilization = 0.f;
    க்கு(unsigned i = 0; i < sample_மொத்தம்; ++i) {
        gpu_utilization += (தசம)buffer[i].sample_மதிப்பு.uiVal;
    }
    gpu_utilization /= (தசம)sample_மொத்தம்;

    // sample count may have been modified by the query above; reset back to buffer size
    sample_மொத்தம் = BUFFER_LIMIT;
    nvmlCheck(nvmlDeviceGetSamples(device, NVML_MEMORY_UTILIZATION_SAMPLES, 0, &v_type, &sample_மொத்தம், buffer));
    தசம mem_utilization = 0.f;
    க்கு(unsigned i = 0; i < sample_மொத்தம்; ++i) {
        mem_utilization += (தசம)buffer[i].sample_மதிப்பு.uiVal;
    }
    mem_utilization /= (தசம)sample_மொத்தம்;

    info.gpu_utilization = gpu_utilization;
    info.mem_utilization = mem_utilization;
    திரும்பு info;
}
#இல்லையெனில்
GPUUtilInfo get_gpu_utilization_info() {
    fprintf(stderr, "Error: Compiled without nvml support. Cannot perform additional GPU state tracking.");
    exit(EXIT_FAILURE);
}
#endif
#endif // MFU_H
