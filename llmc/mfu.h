#اگر_تعریف_نشده MFU_H
#تعریف_کن MFU_H

#ضمن <stdio.h>
#ضمن <stdlib.h>
#ضمن <string.h>
#إذا __has_include(<nvml.h>)
#تعریف_کن USE_NVML 1
#ضمن <nvml.h>
#وإلا
#تعریف_کن USE_NVML 0
#پایان_اگر

// tied to enum PrecisionMode, in a future refactor make them the same
#تعریف_کن MFUH_PRECISION_FP32 0
#تعریف_کن MFUH_PRECISION_FP16 1
#تعریف_کن MFUH_PRECISION_BF16 2

#إذا USE_NVML
مضمن فراغ nvml_check(nvmlReturn_t وضعیت, ثابت حرف *file, صحيح سطر) {
    إذا (وضعیت != NVML_SUCCESS) {
        چاپ_کن("[NVML ERROR] at file %s:%d:\n%s\n", file, سطر, nvmlErrorString(وضعیت));
        خروج(EXIT_FAILURE);
    }
};
#تعریف_کن nvmlCheck(err) (nvml_check(err, __FILE__, __LINE__))
#پایان_اگر


تعريفنوع بنية {
    عائم TF_32;       // tensor-core performance 32 bit
    عائم BF_16_32;    // bf16 with 32 bit accumulate
    عائم FP_16_32;    // fp16 with 32 bit accumulate
    عائم FP_16_16;    // fp16 with 16 bit accumulate
    عائم FP_8_32;     // and so on
    عائم FP_8_16;
    عائم CLOCK;        // clock frequency from the spec sheet
    عائم CORES;        // #TCs from the spec sheet
} Perf_داده;

// basic default data from the nvidia whitepapers
ساكن ثابت Perf_داده VOLTA = {125.0f, -1.f, 125.f, -1.f, -1.f, -1.f, 1530.f, 640.f};
ساكن ثابت Perf_داده AMPERE_DATACENTER = {156.f, 312.f, 312.f, 312.f, -1.f, -1.f, 1410.f, 432.f};
ساكن ثابت Perf_داده AMPERE_مصرفکننده = {40.f, 80.f, 80.f, 160.f, -1.f, -1.f, 1860.f, 336.f};
ساكن ثابت Perf_داده HOPPER = {378.f, 756.f, 756.f, 756.f, 1513.f, 1513.f, 1620.f, 456.f};
ساكن ثابت Perf_داده ADA = {82.6f, 165.2f, 165.2f, 330.3f, 330.3f, 660.6f, 2520.f, 512.f};

تعريفنوع بنية {
    ثابت حرف* name;
    ثابت Perf_داده* perf_داده;
    عائم new_cores;
    عائم new_mhz;
} GPUEntry;

// the overrides for each specific GPU
ساكن GPUEntry gpu_db[] = {
    {"Tesla V100-SXM2-16GB", &VOLTA, 640, 1530},
    {"Tesla V100-PCIE-32GB", &VOLTA, 640, 1530},
    {"NVIDIA A100-PCIE-40GB", &AMPERE_DATACENTER, 432, 1410},
    {"NVIDIA A100-PCIE-80GB", &AMPERE_DATACENTER, 432, 1410},
    {"NVIDIA A100-SXM4-40GB", &AMPERE_DATACENTER, 432, 1410},
    {"NVIDIA A100-SXM4-80GB", &AMPERE_DATACENTER, 432, 1410},
    {"NVIDIA RTX A2000", &AMPERE_مصرفکننده, 104, 1200},
    {"NVIDIA RTX A4000", &AMPERE_مصرفکننده, 192, 1560},
    {"NVIDIA RTX A4500", &AMPERE_مصرفکننده, 224, 1650},
    {"NVIDIA RTX A5000", &AMPERE_مصرفکننده, 256, 1695},
    {"NVIDIA RTX A5500", &AMPERE_مصرفکننده, 320, 1770},
    {"NVIDIA RTX A6000", &AMPERE_مصرفکننده, 336, 1800},
    {"NVIDIA GeForce RTX 3090 Ti", &AMPERE_مصرفکننده, 336, 1860},
    {"NVIDIA GeForce RTX 3090", &AMPERE_مصرفکننده, 328, 1695},
    {"NVIDIA GeForce RTX 3080 Ti", &AMPERE_مصرفکننده, 320, 1665},
    {"NVIDIA GeForce RTX 3080", &AMPERE_مصرفکننده, 272, 1710},
    {"NVIDIA GeForce RTX 3070 Ti", &AMPERE_مصرفکننده, 192, 1770},
    {"NVIDIA GeForce RTX 3070", &AMPERE_مصرفکننده, 184, 1725},
    {"NVIDIA GeForce RTX 3060 Ti", &AMPERE_مصرفکننده, 152, 1665},
    {"NVIDIA GeForce RTX 3060", &AMPERE_مصرفکننده, 112, 1777},
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

عائم get_flops_promised(ثابت حرف* device, صحيح precision_نما) {
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
    إذا (!(precision_نما == MFUH_PRECISION_FP32 || precision_نما == MFUH_PRECISION_FP16 || precision_نما == MFUH_PRECISION_BF16)) {
        fprintf(stderr, "Invalid precision mode: %d\n", precision_نما);
        أرجع -1.0f;
    }

    // do a linear search until you find our GPU, then calculate the flops promised
    صحيح رقم_gpu_entries = حجم(gpu_db) / حجم(gpu_db[0]);
    لكل (صحيح i = 0; i < رقم_gpu_entries; i++) {
        إذا (strcmp(gpu_db[i].name, device) == 0) {
            ثابت Perf_داده* perf_داده = gpu_db[i].perf_داده;

            // look up the default flops value for the given precision mode
            عائم مقدار = -1.0f;
            إذا (precision_نما == MFUH_PRECISION_BF16) { مقدار = perf_داده->BF_16_32; }
            إذا (precision_نما == MFUH_PRECISION_FP32) { مقدار = perf_داده->TF_32; }
            إذا (precision_نما == MFUH_PRECISION_FP16) { مقدار = perf_داده->FP_16_32; }

            // we'd get here if we're e.g. trying to use BF16 on Volta GPU or something...
            إذا (مقدار < 0.0f) {
                fprintf(stderr, "No data for GPU %s and precision mode %d\n", device, precision_نما);
                أرجع -1.0f;
            }

            // adjust flops based on the specific core count and clock frequency of this GPU
            عائم new_cores = gpu_db[i].new_cores;
            عائم new_mhz = gpu_db[i].new_mhz;
            عائم adjusted = مقدار * (new_cores / perf_داده->CORES) * (new_mhz / perf_داده->CLOCK);
            أرجع adjusted;
        }
    }

    أرجع -1.0f; // ¯\_(ツ)_/¯
}

بنية GPUUtilInfo {
    بیعلامت صحيح clock;
    بیعلامت صحيح max_clock;
    بیعلامت صحيح power;
    بیعلامت صحيح power_حد;
    بیعلامت صحيح fan;
    بیعلامت صحيح temperature;
    بیعلامت صحيح موقت_slowdown;

    عائم gpu_utilization;
    عائم mem_utilization;
    ثابت حرف* throttle_reason;
};

// lazily initialize nvml and generate a handle to the GPU
#إذا USE_NVML
nvmlDevice_t nvml_get_device() {
    ساكن bool needs_init = true;
    ساكن nvmlDevice_t device;
    إذا(needs_init) {
        needs_init = false;
        nvmlCheck(nvmlInit());
        nvmlCheck(nvml_Device_Get_Handle_By_شاخص_v2(0, &device));
    }
    أرجع device;
}

// convert throttle reason bitfield into a text reason.
// this is a lossy conversion; we just want to give some idea of what is happening
ثابت حرف* get_throttle_reason(بیعلامت دراز دراز بیتها) {
    إذا(بیتها & (nvmlClocksThrottleReasonSwPowerCap | nvmlClocksThrottleReasonHwPowerBrakeSlowdown)) {
        أرجع "power cap";
    } وإلا إذا (بیتها & (nvmlClocksThrottleReasonSwThermalSlowdown | nvmlClocksThrottleReasonHwThermalSlowdown)) {
        أرجع "thermal cap";
    } وإلا إذا (بیتها & (nvmlClocksThrottleReasonAll)) {
        أرجع "other cap";
    } وإلا {
        أرجع "no cap";
    }
}

// gather data for a GPUUtilInfo object
GPUUtilInfo get_gpu_utilization_info() {
    GPUUtilInfo info;
    nvmlDevice_t device = nvml_get_device();
    // query different infos directly
    nvmlCheck(nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &info.clock));
    nvmlCheck(nvmlDeviceGetMaxClockInfo(device, NVML_CLOCK_SM, &info.max_clock));
    nvmlCheck(nvml_Device_Get_Power_Management_حد(device, &info.power_حد));
    nvmlCheck(nvmlDeviceGetPowerUsage(device, &info.power));
    nvmlCheck(nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &info.temperature));
    nvmlCheck(nvml_Device_Get_Temperature_آستانه(device, NVML_TEMPERATURE_آستانه_SLOWDOWN, &info.موقت_slowdown));
    بیعلامت دراز دراز throttle;
    nvmlCheck(nvmlDeviceGetCurrentClocksThrottleReasons(device, &throttle));
    info.throttle_reason = get_throttle_reason(throttle);
    nvmlCheck(nvml_Device_Get_Fan_سرعت(device, &info.fan));

    // for "utilization", we look at recorded samples. In principle, we could query the driver for how many samples
    // to request, but then we'd need to dynamically allocate sufficient space. Let's just hard-code a limit of 128,
    // and have no memory management required
    constexpr ثابت صحيح میانگیر_حد = 128;
    nvmlSample_t میانگیر[میانگیر_حد];
    nvml_مقدار_Type_t v_type;
    بیعلامت صحيح sample_شمار = میانگیر_حد;
    nvmlCheck(nvmlDeviceGetSamples(device, NVML_GPU_UTILIZATION_SAMPLES, 0, &v_type, &sample_شمار, میانگیر));
    عائم gpu_utilization = 0.f;
    لكل(بیعلامت i = 0; i < sample_شمار; ++i) {
        gpu_utilization += (عائم)میانگیر[i].sample_مقدار.uiVal;
    }
    gpu_utilization /= (عائم)sample_شمار;

    // sample count may have been modified by the query above; reset back to buffer size
    sample_شمار = میانگیر_حد;
    nvmlCheck(nvmlDeviceGetSamples(device, NVML_حافظه_UTILIZATION_SAMPLES, 0, &v_type, &sample_شمار, میانگیر));
    عائم mem_utilization = 0.f;
    لكل(بیعلامت i = 0; i < sample_شمار; ++i) {
        mem_utilization += (عائم)میانگیر[i].sample_مقدار.uiVal;
    }
    mem_utilization /= (عائم)sample_شمار;

    info.gpu_utilization = gpu_utilization;
    info.mem_utilization = mem_utilization;
    أرجع info;
}
#وإلا
GPUUtilInfo get_gpu_utilization_info() {
    fprintf(stderr, "Error: Compiled without nvml support. Cannot perform additional GPU state tracking.");
    خروج(EXIT_FAILURE);
}
#پایان_اگر
#پایان_اگر // MFU_H
