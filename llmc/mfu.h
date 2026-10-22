#ifndef MFU_H
#define MFU_H

#അന്തര്ഭാവ <stdio.h>
#അന്തര്ഭാവ <stdlib.h>
#അന്തര്ഭാവ <string.h>
#എങ്കിൽ __has_include(<nvml.h>)
#define USE_NVML 1
#അന്തര്ഭാവ <nvml.h>
#അല്ലെങ്കിൽ
#define USE_NVML 0
#endif

// tied to enum PrecisionMode, in a future refactor make them the same
#define MFUH_PRECISION_FP32 0
#define MFUH_PRECISION_FP16 1
#define MFUH_PRECISION_BF16 2

#എങ്കിൽ USE_NVML
വരിയിൽ ശൂന്യം nvml_check(nvmlReturn_t status, സ്ഥിരം അക്ഷരം *file, പൂർണ്ണസംഖ്യ വരി) {
    എങ്കിൽ (status != NVML_SUCCESS) {
        printf("[NVML ERROR] at file %s:%d:\n%s\n", file, വരി, nvmlErrorString(status));
        exit(EXIT_FAILURE);
    }
};
#define nvmlCheck(err) (nvml_check(err, __FILE__, __LINE__))
#endif


തരനിർവചനം ഘടന {
    ദശാംശം TF_32;       // tensor-core performance 32 bit
    ദശാംശം BF_16_32;    // bf16 with 32 bit accumulate
    ദശാംശം FP_16_32;    // fp16 with 32 bit accumulate
    ദശാംശം FP_16_16;    // fp16 with 16 bit accumulate
    ദശാംശം FP_8_32;     // and so on
    ദശാംശം FP_8_16;
    ദശാംശം CLOCK;        // clock frequency from the spec sheet
    ദശാംശം CORES;        // #TCs from the spec sheet
} PerfData;

// basic default data from the nvidia whitepapers
സ്ഥിതം സ്ഥിരം PerfData VOLTA = {125.0f, -1.f, 125.f, -1.f, -1.f, -1.f, 1530.f, 640.f};
സ്ഥിതം സ്ഥിരം PerfData AMPERE_DATACENTER = {156.f, 312.f, 312.f, 312.f, -1.f, -1.f, 1410.f, 432.f};
സ്ഥിതം സ്ഥിരം PerfData AMPERE_CONSUMER = {40.f, 80.f, 80.f, 160.f, -1.f, -1.f, 1860.f, 336.f};
സ്ഥിതം സ്ഥിരം PerfData HOPPER = {378.f, 756.f, 756.f, 756.f, 1513.f, 1513.f, 1620.f, 456.f};
സ്ഥിതം സ്ഥിരം PerfData ADA = {82.6f, 165.2f, 165.2f, 330.3f, 330.3f, 660.6f, 2520.f, 512.f};

തരനിർവചനം ഘടന {
    സ്ഥിരം അക്ഷരം* name;
    സ്ഥിരം PerfData* perf_data;
    ദശാംശം new_cores;
    ദശാംശം new_mhz;
} GPUEntry;

// the overrides for each specific GPU
സ്ഥിതം GPUEntry gpu_db[] = {
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

ദശാംശം get_flops_promised(സ്ഥിരം അക്ഷരം* device, പൂർണ്ണസംഖ്യ precision_mode) {
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
    എങ്കിൽ (!(precision_mode == MFUH_PRECISION_FP32 || precision_mode == MFUH_PRECISION_FP16 || precision_mode == MFUH_PRECISION_BF16)) {
        fprintf(stderr, "Invalid precision mode: %d\n", precision_mode);
        തിരികെ -1.0f;
    }

    // do a linear search until you find our GPU, then calculate the flops promised
    പൂർണ്ണസംഖ്യ അക്കം_gpu_entries = വലുപ്പം(gpu_db) / വലുപ്പം(gpu_db[0]);
    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < അക്കം_gpu_entries; i++) {
        എങ്കിൽ (strcmp(gpu_db[i].name, device) == 0) {
            സ്ഥിരം PerfData* perf_data = gpu_db[i].perf_data;

            // look up the default flops value for the given precision mode
            ദശാംശം മൂല്യം = -1.0f;
            എങ്കിൽ (precision_mode == MFUH_PRECISION_BF16) { മൂല്യം = perf_data->BF_16_32; }
            എങ്കിൽ (precision_mode == MFUH_PRECISION_FP32) { മൂല്യം = perf_data->TF_32; }
            എങ്കിൽ (precision_mode == MFUH_PRECISION_FP16) { മൂല്യം = perf_data->FP_16_32; }

            // we'd get here if we're e.g. trying to use BF16 on Volta GPU or something...
            എങ്കിൽ (മൂല്യം < 0.0f) {
                fprintf(stderr, "No data for GPU %s and precision mode %d\n", device, precision_mode);
                തിരികെ -1.0f;
            }

            // adjust flops based on the specific core count and clock frequency of this GPU
            ദശാംശം new_cores = gpu_db[i].new_cores;
            ദശാംശം new_mhz = gpu_db[i].new_mhz;
            ദശാംശം adjusted = മൂല്യം * (new_cores / perf_data->CORES) * (new_mhz / perf_data->CLOCK);
            തിരികെ adjusted;
        }
    }

    തിരികെ -1.0f; // ¯\_(ツ)_/¯
}

ഘടന GPUUtilInfo {
    unsigned പൂർണ്ണസംഖ്യ clock;
    unsigned പൂർണ്ണസംഖ്യ max_clock;
    unsigned പൂർണ്ണസംഖ്യ power;
    unsigned പൂർണ്ണസംഖ്യ power_limit;
    unsigned പൂർണ്ണസംഖ്യ fan;
    unsigned പൂർണ്ണസംഖ്യ temperature;
    unsigned പൂർണ്ണസംഖ്യ temp_slowdown;

    ദശാംശം gpu_utilization;
    ദശാംശം mem_utilization;
    സ്ഥിരം അക്ഷരം* throttle_reason;
};

// lazily initialize nvml and generate a handle to the GPU
#എങ്കിൽ USE_NVML
nvmlDevice_t nvml_get_device() {
    സ്ഥിതം bool needs_init = true;
    സ്ഥിതം nvmlDevice_t device;
    എങ്കിൽ(needs_init) {
        needs_init = false;
        nvmlCheck(nvmlInit());
        nvmlCheck(nvmlDeviceGetHandleByIndex_v2(0, &device));
    }
    തിരികെ device;
}

// convert throttle reason bitfield into a text reason.
// this is a lossy conversion; we just want to give some idea of what is happening
സ്ഥിരം അക്ഷരം* get_throttle_reason(unsigned long long bits) {
    എങ്കിൽ(bits & (nvmlClocksThrottleReasonSwPowerCap | nvmlClocksThrottleReasonHwPowerBrakeSlowdown)) {
        തിരികെ "power cap";
    } അല്ലെങ്കിൽ എങ്കിൽ (bits & (nvmlClocksThrottleReasonSwThermalSlowdown | nvmlClocksThrottleReasonHwThermalSlowdown)) {
        തിരികെ "thermal cap";
    } അല്ലെങ്കിൽ എങ്കിൽ (bits & (nvmlClocksThrottleReasonAll)) {
        തിരികെ "other cap";
    } അല്ലെങ്കിൽ {
        തിരികെ "no cap";
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
    constexpr സ്ഥിരം പൂർണ്ണസംഖ്യ BUFFER_LIMIT = 128;
    nvmlSample_t buffer[BUFFER_LIMIT];
    nvml_മൂല്യം_Type_t v_type;
    unsigned പൂർണ്ണസംഖ്യ sample_എണ്ണം = BUFFER_LIMIT;
    nvmlCheck(nvmlDeviceGetSamples(device, NVML_GPU_UTILIZATION_SAMPLES, 0, &v_type, &sample_എണ്ണം, buffer));
    ദശാംശം gpu_utilization = 0.f;
    വേണ്ടി(unsigned i = 0; i < sample_എണ്ണം; ++i) {
        gpu_utilization += (ദശാംശം)buffer[i].sample_മൂല്യം.uiVal;
    }
    gpu_utilization /= (ദശാംശം)sample_എണ്ണം;

    // sample count may have been modified by the query above; reset back to buffer size
    sample_എണ്ണം = BUFFER_LIMIT;
    nvmlCheck(nvmlDeviceGetSamples(device, NVML_MEMORY_UTILIZATION_SAMPLES, 0, &v_type, &sample_എണ്ണം, buffer));
    ദശാംശം mem_utilization = 0.f;
    വേണ്ടി(unsigned i = 0; i < sample_എണ്ണം; ++i) {
        mem_utilization += (ദശാംശം)buffer[i].sample_മൂല്യം.uiVal;
    }
    mem_utilization /= (ദശാംശം)sample_എണ്ണം;

    info.gpu_utilization = gpu_utilization;
    info.mem_utilization = mem_utilization;
    തിരികെ info;
}
#അല്ലെങ്കിൽ
GPUUtilInfo get_gpu_utilization_info() {
    fprintf(stderr, "Error: Compiled without nvml support. Cannot perform additional GPU state tracking.");
    exit(EXIT_FAILURE);
}
#endif
#endif // MFU_H
