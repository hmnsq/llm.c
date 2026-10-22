#ifndef MFU_H
#define MFU_H

#અન્તર્ભાવ <stdio.h>
#અન્તર્ભાવ <stdlib.h>
#અન્તર્ભાવ <string.h>
#જો __has_include(<nvml.h>)
#define USE_NVML 1
#અન્તર્ભાવ <nvml.h>
#નહીંતર
#define USE_NVML 0
#endif

// tied to enum PrecisionMode, in a future refactor make them the same
#define MFUH_PRECISION_FP32 0
#define MFUH_PRECISION_FP16 1
#define MFUH_PRECISION_BF16 2

#જો USE_NVML
અંતઃરેખ શૂન્ય nvml_check(nvmlReturn_t status, અચલ અક્ષર *file, પૂર્ણાંક લીટી) {
    જો (status != NVML_SUCCESS) {
        printf("[NVML ERROR] at file %s:%d:\n%s\n", file, લીટી, nvmlErrorString(status));
        exit(EXIT_FAILURE);
    }
};
#define nvmlCheck(err) (nvml_check(err, __FILE__, __LINE__))
#endif


પ્રકારવ્યાખ્યા સંરચના {
    દશાંશ TF_32;       // tensor-core performance 32 bit
    દશાંશ BF_16_32;    // bf16 with 32 bit accumulate
    દશાંશ FP_16_32;    // fp16 with 32 bit accumulate
    દશાંશ FP_16_16;    // fp16 with 16 bit accumulate
    દશાંશ FP_8_32;     // and so on
    દશાંશ FP_8_16;
    દશાંશ CLOCK;        // clock frequency from the spec sheet
    દશાંશ CORES;        // #TCs from the spec sheet
} PerfData;

// basic default data from the nvidia whitepapers
સ્થિર અચલ PerfData VOLTA = {125.0f, -1.f, 125.f, -1.f, -1.f, -1.f, 1530.f, 640.f};
સ્થિર અચલ PerfData AMPERE_DATACENTER = {156.f, 312.f, 312.f, 312.f, -1.f, -1.f, 1410.f, 432.f};
સ્થિર અચલ PerfData AMPERE_CONSUMER = {40.f, 80.f, 80.f, 160.f, -1.f, -1.f, 1860.f, 336.f};
સ્થિર અચલ PerfData HOPPER = {378.f, 756.f, 756.f, 756.f, 1513.f, 1513.f, 1620.f, 456.f};
સ્થિર અચલ PerfData ADA = {82.6f, 165.2f, 165.2f, 330.3f, 330.3f, 660.6f, 2520.f, 512.f};

પ્રકારવ્યાખ્યા સંરચના {
    અચલ અક્ષર* name;
    અચલ PerfData* perf_data;
    દશાંશ new_cores;
    દશાંશ new_mhz;
} GPUEntry;

// the overrides for each specific GPU
સ્થિર GPUEntry gpu_db[] = {
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

દશાંશ get_flops_promised(અચલ અક્ષર* device, પૂર્ણાંક precision_mode) {
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
    જો (!(precision_mode == MFUH_PRECISION_FP32 || precision_mode == MFUH_PRECISION_FP16 || precision_mode == MFUH_PRECISION_BF16)) {
        fprintf(stderr, "Invalid precision mode: %d\n", precision_mode);
        પાછા -1.0f;
    }

    // do a linear search until you find our GPU, then calculate the flops promised
    પૂર્ણાંક અંક_gpu_entries = કદ(gpu_db) / કદ(gpu_db[0]);
    માટે (પૂર્ણાંક i = 0; i < અંક_gpu_entries; i++) {
        જો (strcmp(gpu_db[i].name, device) == 0) {
            અચલ PerfData* perf_data = gpu_db[i].perf_data;

            // look up the default flops value for the given precision mode
            દશાંશ કિંમત = -1.0f;
            જો (precision_mode == MFUH_PRECISION_BF16) { કિંમત = perf_data->BF_16_32; }
            જો (precision_mode == MFUH_PRECISION_FP32) { કિંમત = perf_data->TF_32; }
            જો (precision_mode == MFUH_PRECISION_FP16) { કિંમત = perf_data->FP_16_32; }

            // we'd get here if we're e.g. trying to use BF16 on Volta GPU or something...
            જો (કિંમત < 0.0f) {
                fprintf(stderr, "No data for GPU %s and precision mode %d\n", device, precision_mode);
                પાછા -1.0f;
            }

            // adjust flops based on the specific core count and clock frequency of this GPU
            દશાંશ new_cores = gpu_db[i].new_cores;
            દશાંશ new_mhz = gpu_db[i].new_mhz;
            દશાંશ adjusted = કિંમત * (new_cores / perf_data->CORES) * (new_mhz / perf_data->CLOCK);
            પાછા adjusted;
        }
    }

    પાછા -1.0f; // ¯\_(ツ)_/¯
}

સંરચના GPUUtilInfo {
    unsigned પૂર્ણાંક clock;
    unsigned પૂર્ણાંક max_clock;
    unsigned પૂર્ણાંક power;
    unsigned પૂર્ણાંક power_limit;
    unsigned પૂર્ણાંક fan;
    unsigned પૂર્ણાંક temperature;
    unsigned પૂર્ણાંક temp_slowdown;

    દશાંશ gpu_utilization;
    દશાંશ mem_utilization;
    અચલ અક્ષર* throttle_reason;
};

// lazily initialize nvml and generate a handle to the GPU
#જો USE_NVML
nvmlDevice_t nvml_get_device() {
    સ્થિર bool needs_init = true;
    સ્થિર nvmlDevice_t device;
    જો(needs_init) {
        needs_init = false;
        nvmlCheck(nvmlInit());
        nvmlCheck(nvmlDeviceGetHandleByIndex_v2(0, &device));
    }
    પાછા device;
}

// convert throttle reason bitfield into a text reason.
// this is a lossy conversion; we just want to give some idea of what is happening
અચલ અક્ષર* get_throttle_reason(unsigned long long bits) {
    જો(bits & (nvmlClocksThrottleReasonSwPowerCap | nvmlClocksThrottleReasonHwPowerBrakeSlowdown)) {
        પાછા "power cap";
    } નહીંતર જો (bits & (nvmlClocksThrottleReasonSwThermalSlowdown | nvmlClocksThrottleReasonHwThermalSlowdown)) {
        પાછા "thermal cap";
    } નહીંતર જો (bits & (nvmlClocksThrottleReasonAll)) {
        પાછા "other cap";
    } નહીંતર {
        પાછા "no cap";
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
    constexpr અચલ પૂર્ણાંક BUFFER_LIMIT = 128;
    nvmlSample_t buffer[BUFFER_LIMIT];
    nvml_કિંમત_Type_t v_type;
    unsigned પૂર્ણાંક sample_ગણતરી = BUFFER_LIMIT;
    nvmlCheck(nvmlDeviceGetSamples(device, NVML_GPU_UTILIZATION_SAMPLES, 0, &v_type, &sample_ગણતરી, buffer));
    દશાંશ gpu_utilization = 0.f;
    માટે(unsigned i = 0; i < sample_ગણતરી; ++i) {
        gpu_utilization += (દશાંશ)buffer[i].sample_કિંમત.uiVal;
    }
    gpu_utilization /= (દશાંશ)sample_ગણતરી;

    // sample count may have been modified by the query above; reset back to buffer size
    sample_ગણતરી = BUFFER_LIMIT;
    nvmlCheck(nvmlDeviceGetSamples(device, NVML_MEMORY_UTILIZATION_SAMPLES, 0, &v_type, &sample_ગણતરી, buffer));
    દશાંશ mem_utilization = 0.f;
    માટે(unsigned i = 0; i < sample_ગણતરી; ++i) {
        mem_utilization += (દશાંશ)buffer[i].sample_કિંમત.uiVal;
    }
    mem_utilization /= (દશાંશ)sample_ગણતરી;

    info.gpu_utilization = gpu_utilization;
    info.mem_utilization = mem_utilization;
    પાછા info;
}
#નહીંતર
GPUUtilInfo get_gpu_utilization_info() {
    fprintf(stderr, "Error: Compiled without nvml support. Cannot perform additional GPU state tracking.");
    exit(EXIT_FAILURE);
}
#endif
#endif // MFU_H
