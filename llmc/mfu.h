#अगर_अघोषित MFU_H
#घोषणा MFU_H

#समावेश <stdio.h>
#समावेश <stdlib.h>
#समावेश <string.h>
#अगर __has_include(<nvml.h>)
#घोषणा USE_NVML 1
#समावेश <nvml.h>
#अन्यथा
#घोषणा USE_NVML 0
#पूर्ण_अगर

// tied to enum PrecisionMode, in a future refactor make them the same
#घोषणा MFUH_PRECISION_FP32 0
#घोषणा MFUH_PRECISION_FP16 1
#घोषणा MFUH_PRECISION_BF16 2

#अगर USE_NVML
अंतरभूत व्योम nvml_check(nvmlReturn_t status, स्थिर अक्षर *file, पूर्णांक पंक्ति) {
    अगर (status != NVML_SUCCESS) {
        म_लिखो("[NVML ERROR] at file %s:%d:\n%s\n", file, पंक्ति, nvmlErrorString(status));
        निकास(निकास_त्रुटि);
    }
};
#घोषणा nvmlCheck(err) (nvml_check(err, __खाता__, __LINE__))
#पूर्ण_अगर


प्रकार काष्ठा {
    भग्न TF_32;       // tensor-core performance 32 bit
    भग्न BF_16_32;    // bf16 with 32 bit accumulate
    भग्न FP_16_32;    // fp16 with 32 bit accumulate
    भग्न FP_16_16;    // fp16 with 16 bit accumulate
    भग्न FP_8_32;     // and so on
    भग्न FP_8_16;
    भग्न CLOCK;        // clock frequency from the spec sheet
    भग्न CORES;        // #TCs from the spec sheet
} Perf_तथ्य;

// basic default data from the nvidia whitepapers
अटल स्थिर Perf_तथ्य VOLTA = {125.0f, -1.f, 125.f, -1.f, -1.f, -1.f, 1530.f, 640.f};
अटल स्थिर Perf_तथ्य AMPERE_DATACENTER = {156.f, 312.f, 312.f, 312.f, -1.f, -1.f, 1410.f, 432.f};
अटल स्थिर Perf_तथ्य AMPERE_CONSUMER = {40.f, 80.f, 80.f, 160.f, -1.f, -1.f, 1860.f, 336.f};
अटल स्थिर Perf_तथ्य HOPPER = {378.f, 756.f, 756.f, 756.f, 1513.f, 1513.f, 1620.f, 456.f};
अटल स्थिर Perf_तथ्य ADA = {82.6f, 165.2f, 165.2f, 330.3f, 330.3f, 660.6f, 2520.f, 512.f};

प्रकार काष्ठा {
    स्थिर अक्षर* name;
    स्थिर Perf_तथ्य* perf_तथ्य;
    भग्न new_cores;
    भग्न new_mhz;
} GPUEntry;

// the overrides for each specific GPU
अटल GPUEntry gpu_db[] = {
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

भग्न get_flops_promised(स्थिर अक्षर* device, पूर्णांक precision_mode) {
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
    अगर (!(precision_mode == MFUH_PRECISION_FP32 || precision_mode == MFUH_PRECISION_FP16 || precision_mode == MFUH_PRECISION_BF16)) {
        ख_लिखो(मानक_त्रुटि, "Invalid precision mode: %d\n", precision_mode);
        वापस -1.0f;
    }

    // do a linear search until you find our GPU, then calculate the flops promised
    पूर्णांक अंक_gpu_entries = माप(gpu_db) / माप(gpu_db[0]);
    क्रम (पूर्णांक i = 0; i < अंक_gpu_entries; i++) {
        अगर (म_भेद(gpu_db[i].name, device) == 0) {
            स्थिर Perf_तथ्य* perf_तथ्य = gpu_db[i].perf_तथ्य;

            // look up the default flops value for the given precision mode
            भग्न मूल्य = -1.0f;
            अगर (precision_mode == MFUH_PRECISION_BF16) { मूल्य = perf_तथ्य->BF_16_32; }
            अगर (precision_mode == MFUH_PRECISION_FP32) { मूल्य = perf_तथ्य->TF_32; }
            अगर (precision_mode == MFUH_PRECISION_FP16) { मूल्य = perf_तथ्य->FP_16_32; }

            // we'd get here if we're e.g. trying to use BF16 on Volta GPU or something...
            अगर (मूल्य < 0.0f) {
                ख_लिखो(मानक_त्रुटि, "No data for GPU %s and precision mode %d\n", device, precision_mode);
                वापस -1.0f;
            }

            // adjust flops based on the specific core count and clock frequency of this GPU
            भग्न new_cores = gpu_db[i].new_cores;
            भग्न new_mhz = gpu_db[i].new_mhz;
            भग्न adjusted = मूल्य * (new_cores / perf_तथ्य->CORES) * (new_mhz / perf_तथ्य->CLOCK);
            वापस adjusted;
        }
    }

    वापस -1.0f; // ¯\_(ツ)_/¯
}

काष्ठा GPUUtilInfo {
    अचिन्हित पूर्णांक घड़ी;
    अचिन्हित पूर्णांक max_clock;
    अचिन्हित पूर्णांक power;
    अचिन्हित पूर्णांक power_limit;
    अचिन्हित पूर्णांक fan;
    अचिन्हित पूर्णांक temperature;
    अचिन्हित पूर्णांक अस्थायी_slowdown;

    भग्न gpu_utilization;
    भग्न mem_utilization;
    स्थिर अक्षर* throttle_reason;
};

// lazily initialize nvml and generate a handle to the GPU
#अगर USE_NVML
nvmlDevice_t nvml_get_device() {
    अटल bool needs_init = true;
    अटल nvmlDevice_t device;
    अगर(needs_init) {
        needs_init = false;
        nvmlCheck(nvmlInit());
        nvmlCheck(nvml_Device_Get_Handle_By_सूचकांक_v2(0, &device));
    }
    वापस device;
}

// convert throttle reason bitfield into a text reason.
// this is a lossy conversion; we just want to give some idea of what is happening
स्थिर अक्षर* get_throttle_reason(अचिन्हित दीर्घ दीर्घ bits) {
    अगर(bits & (nvmlClocksThrottleReasonSwPowerCap | nvmlClocksThrottleReasonHwPowerBrakeSlowdown)) {
        वापस "power cap";
    } अन्यथा अगर (bits & (nvmlClocksThrottleReasonSwThermalSlowdown | nvmlClocksThrottleReasonHwThermalSlowdown)) {
        वापस "thermal cap";
    } अन्यथा अगर (bits & (nvmlClocksThrottleReasonAll)) {
        वापस "other cap";
    } अन्यथा {
        वापस "no cap";
    }
}

// gather data for a GPUUtilInfo object
GPUUtilInfo get_gpu_utilization_info() {
    GPUUtilInfo info;
    nvmlDevice_t device = nvml_get_device();
    // query different infos directly
    nvmlCheck(nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &info.घड़ी));
    nvmlCheck(nvmlDeviceGetMaxClockInfo(device, NVML_CLOCK_SM, &info.max_clock));
    nvmlCheck(nvmlDeviceGetPowerManagementLimit(device, &info.power_limit));
    nvmlCheck(nvmlDeviceGetPowerUsage(device, &info.power));
    nvmlCheck(nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &info.temperature));
    nvmlCheck(nvmlDeviceGetTemperatureThreshold(device, NVML_TEMPERATURE_THRESHOLD_SLOWDOWN, &info.अस्थायी_slowdown));
    अचिन्हित दीर्घ दीर्घ throttle;
    nvmlCheck(nvmlDeviceGetCurrentClocksThrottleReasons(device, &throttle));
    info.throttle_reason = get_throttle_reason(throttle);
    nvmlCheck(nvmlDeviceGetFanSpeed(device, &info.fan));

    // for "utilization", we look at recorded samples. In principle, we could query the driver for how many samples
    // to request, but then we'd need to dynamically allocate sufficient space. Let's just hard-code a limit of 128,
    // and have no memory management required
    constexpr स्थिर पूर्णांक बफर_LIMIT = 128;
    nvmlSample_t बफर[बफर_LIMIT];
    nvml_मूल्य_Type_t v_type;
    अचिन्हित पूर्णांक sample_गिनती = बफर_LIMIT;
    nvmlCheck(nvmlDeviceGetSamples(device, NVML_GPU_UTILIZATION_SAMPLES, 0, &v_type, &sample_गिनती, बफर));
    भग्न gpu_utilization = 0.f;
    क्रम(अचिन्हित i = 0; i < sample_गिनती; ++i) {
        gpu_utilization += (भग्न)बफर[i].sample_मूल्य.uiVal;
    }
    gpu_utilization /= (भग्न)sample_गिनती;

    // sample count may have been modified by the query above; reset back to buffer size
    sample_गिनती = बफर_LIMIT;
    nvmlCheck(nvmlDeviceGetSamples(device, NVML_MEMORY_UTILIZATION_SAMPLES, 0, &v_type, &sample_गिनती, बफर));
    भग्न mem_utilization = 0.f;
    क्रम(अचिन्हित i = 0; i < sample_गिनती; ++i) {
        mem_utilization += (भग्न)बफर[i].sample_मूल्य.uiVal;
    }
    mem_utilization /= (भग्न)sample_गिनती;

    info.gpu_utilization = gpu_utilization;
    info.mem_utilization = mem_utilization;
    वापस info;
}
#अन्यथा
GPUUtilInfo get_gpu_utilization_info() {
    ख_लिखो(मानक_त्रुटि, "Error: Compiled without nvml support. Cannot perform additional GPU state tracking.");
    निकास(निकास_त्रुटि);
}
#पूर्ण_अगर
#पूर्ण_अगर // MFU_H
