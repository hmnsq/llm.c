#اگر_غیر_تعریف_شدہ MFU_H
#تعریف_کرو MFU_H

#شامل <stdio.h>
#شامل <stdlib.h>
#شامل <string.h>
#اگر __has_include(<nvml.h>)
#تعریف_کرو USE_NVML 1
#شامل <nvml.h>
#ورنہ
#تعریف_کرو USE_NVML 0
#اگر_ختم

// tied to enum PrecisionMode, in a future refactor make them the same
#تعریف_کرو MFUH_PRECISION_FP32 0
#تعریف_کرو MFUH_PRECISION_FP16 1
#تعریف_کرو MFUH_PRECISION_BF16 2

#اگر USE_NVML
اندرونی خالی nvml_check(nvmlReturn_t حالت_2, ثابت حرف *file, صحیح سطر) {
    اگر (حالت_2 != NVML_SUCCESS) {
        لکھو("[NVML ERROR] at file %s:%d:\n%s\n", file, سطر, nvmlErrorString(حالت_2));
        باہر_نکلو(EXIT_FAILURE);
    }
};
#تعریف_کرو nvmlCheck(err) (nvml_check(err, __FILE__, __LINE__))
#اگر_ختم


قسمتعریف ساخت {
    اعشاری TF_32;       // tensor-core performance 32 bit
    اعشاری BF_16_32;    // bf16 with 32 bit accumulate
    اعشاری FP_16_32;    // fp16 with 32 bit accumulate
    اعشاری FP_16_16;    // fp16 with 16 bit accumulate
    اعشاری FP_8_32;     // and so on
    اعشاری FP_8_16;
    اعشاری CLOCK;        // clock frequency from the spec sheet
    اعشاری CORES;        // #TCs from the spec sheet
} Perf_ڈیٹا;

// basic default data from the nvidia whitepapers
ساکن ثابت Perf_ڈیٹا VOLTA = {125.0f, -1.f, 125.f, -1.f, -1.f, -1.f, 1530.f, 640.f};
ساکن ثابت Perf_ڈیٹا AMPERE_DATACENTER = {156.f, 312.f, 312.f, 312.f, -1.f, -1.f, 1410.f, 432.f};
ساکن ثابت Perf_ڈیٹا AMPERE_صارف = {40.f, 80.f, 80.f, 160.f, -1.f, -1.f, 1860.f, 336.f};
ساکن ثابت Perf_ڈیٹا HOPPER = {378.f, 756.f, 756.f, 756.f, 1513.f, 1513.f, 1620.f, 456.f};
ساکن ثابت Perf_ڈیٹا ADA = {82.6f, 165.2f, 165.2f, 330.3f, 330.3f, 660.6f, 2520.f, 512.f};

قسمتعریف ساخت {
    ثابت حرف* name;
    ثابت Perf_ڈیٹا* perf_ڈیٹا;
    اعشاری new_cores;
    اعشاری new_mhz;
} GPUEntry;

// the overrides for each specific GPU
ساکن GPUEntry gpu_db[] = {
    {"Tesla V100-SXM2-16GB", &VOLTA, 640, 1530},
    {"Tesla V100-PCIE-32GB", &VOLTA, 640, 1530},
    {"NVIDIA A100-PCIE-40GB", &AMPERE_DATACENTER, 432, 1410},
    {"NVIDIA A100-PCIE-80GB", &AMPERE_DATACENTER, 432, 1410},
    {"NVIDIA A100-SXM4-40GB", &AMPERE_DATACENTER, 432, 1410},
    {"NVIDIA A100-SXM4-80GB", &AMPERE_DATACENTER, 432, 1410},
    {"NVIDIA RTX A2000", &AMPERE_صارف, 104, 1200},
    {"NVIDIA RTX A4000", &AMPERE_صارف, 192, 1560},
    {"NVIDIA RTX A4500", &AMPERE_صارف, 224, 1650},
    {"NVIDIA RTX A5000", &AMPERE_صارف, 256, 1695},
    {"NVIDIA RTX A5500", &AMPERE_صارف, 320, 1770},
    {"NVIDIA RTX A6000", &AMPERE_صارف, 336, 1800},
    {"NVIDIA GeForce RTX 3090 Ti", &AMPERE_صارف, 336, 1860},
    {"NVIDIA GeForce RTX 3090", &AMPERE_صارف, 328, 1695},
    {"NVIDIA GeForce RTX 3080 Ti", &AMPERE_صارف, 320, 1665},
    {"NVIDIA GeForce RTX 3080", &AMPERE_صارف, 272, 1710},
    {"NVIDIA GeForce RTX 3070 Ti", &AMPERE_صارف, 192, 1770},
    {"NVIDIA GeForce RTX 3070", &AMPERE_صارف, 184, 1725},
    {"NVIDIA GeForce RTX 3060 Ti", &AMPERE_صارف, 152, 1665},
    {"NVIDIA GeForce RTX 3060", &AMPERE_صارف, 112, 1777},
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

اعشاری get_flops_promised(ثابت حرف* device, صحیح precision_منوال) {
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
    اگر (!(precision_منوال == MFUH_PRECISION_FP32 || precision_منوال == MFUH_PRECISION_FP16 || precision_منوال == MFUH_PRECISION_BF16)) {
        fprintf(stderr, "Invalid precision mode: %d\n", precision_منوال);
        واپس -1.0f;
    }

    // do a linear search until you find our GPU, then calculate the flops promised
    صحیح عدد_gpu_entries = حجم(gpu_db) / حجم(gpu_db[0]);
    برائے (صحیح i = 0; i < عدد_gpu_entries; i++) {
        اگر (strcmp(gpu_db[i].name, device) == 0) {
            ثابت Perf_ڈیٹا* perf_ڈیٹا = gpu_db[i].perf_ڈیٹا;

            // look up the default flops value for the given precision mode
            اعشاری قدر = -1.0f;
            اگر (precision_منوال == MFUH_PRECISION_BF16) { قدر = perf_ڈیٹا->BF_16_32; }
            اگر (precision_منوال == MFUH_PRECISION_FP32) { قدر = perf_ڈیٹا->TF_32; }
            اگر (precision_منوال == MFUH_PRECISION_FP16) { قدر = perf_ڈیٹا->FP_16_32; }

            // we'd get here if we're e.g. trying to use BF16 on Volta GPU or something...
            اگر (قدر < 0.0f) {
                fprintf(stderr, "No data for GPU %s and precision mode %d\n", device, precision_منوال);
                واپس -1.0f;
            }

            // adjust flops based on the specific core count and clock frequency of this GPU
            اعشاری new_cores = gpu_db[i].new_cores;
            اعشاری new_mhz = gpu_db[i].new_mhz;
            اعشاری adjusted = قدر * (new_cores / perf_ڈیٹا->CORES) * (new_mhz / perf_ڈیٹا->CLOCK);
            واپس adjusted;
        }
    }

    واپس -1.0f; // ¯\_(ツ)_/¯
}

ساخت GPUUtilInfo {
    غیرعلامتی صحیح clock;
    غیرعلامتی صحیح max_clock;
    غیرعلامتی صحیح power;
    غیرعلامتی صحیح power_حد_بندی;
    غیرعلامتی صحیح fan;
    غیرعلامتی صحیح temperature;
    غیرعلامتی صحیح عارضی_slowdown;

    اعشاری gpu_utilization;
    اعشاری mem_utilization;
    ثابت حرف* throttle_reason;
};

// lazily initialize nvml and generate a handle to the GPU
#اگر USE_NVML
nvmlDevice_t nvml_get_device() {
    ساکن bool needs_init = true;
    ساکن nvmlDevice_t device;
    اگر(needs_init) {
        needs_init = false;
        nvmlCheck(nvmlInit());
        nvmlCheck(nvml_Device_Get_Handle_By_اشاریہ_v2(0, &device));
    }
    واپس device;
}

// convert throttle reason bitfield into a text reason.
// this is a lossy conversion; we just want to give some idea of what is happening
ثابت حرف* get_throttle_reason(غیرعلامتی طویل طویل بٹ) {
    اگر(بٹ & (nvmlClocksThrottleReasonSwPowerCap | nvmlClocksThrottleReasonHwPowerBrakeSlowdown)) {
        واپس "power cap";
    } ورنہ اگر (بٹ & (nvmlClocksThrottleReasonSwThermalSlowdown | nvmlClocksThrottleReasonHwThermalSlowdown)) {
        واپس "thermal cap";
    } ورنہ اگر (بٹ & (nvmlClocksThrottleReasonAll)) {
        واپس "other cap";
    } ورنہ {
        واپس "no cap";
    }
}

// gather data for a GPUUtilInfo object
GPUUtilInfo get_gpu_utilization_info() {
    GPUUtilInfo info;
    nvmlDevice_t device = nvml_get_device();
    // query different infos directly
    nvmlCheck(nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &info.clock));
    nvmlCheck(nvmlDeviceGetMaxClockInfo(device, NVML_CLOCK_SM, &info.max_clock));
    nvmlCheck(nvml_Device_Get_Power_Management_حد_بندی(device, &info.power_حد_بندی));
    nvmlCheck(nvmlDeviceGetPowerUsage(device, &info.power));
    nvmlCheck(nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &info.temperature));
    nvmlCheck(nvml_Device_Get_Temperature_حد(device, NVML_TEMPERATURE_حد_SLOWDOWN, &info.عارضی_slowdown));
    غیرعلامتی طویل طویل throttle;
    nvmlCheck(nvmlDeviceGetCurrentClocksThrottleReasons(device, &throttle));
    info.throttle_reason = get_throttle_reason(throttle);
    nvmlCheck(nvml_Device_Get_Fan_رفتار(device, &info.fan));

    // for "utilization", we look at recorded samples. In principle, we could query the driver for how many samples
    // to request, but then we'd need to dynamically allocate sufficient space. Let's just hard-code a limit of 128,
    // and have no memory management required
    constexpr ثابت صحیح بفر_حد_بندی = 128;
    nvmlSample_t بفر[بفر_حد_بندی];
    nvml_قدر_Type_t v_type;
    غیرعلامتی صحیح sample_گنتی = بفر_حد_بندی;
    nvmlCheck(nvmlDeviceGetSamples(device, NVML_GPU_UTILIZATION_SAMPLES, 0, &v_type, &sample_گنتی, بفر));
    اعشاری gpu_utilization = 0.f;
    برائے(غیرعلامتی i = 0; i < sample_گنتی; ++i) {
        gpu_utilization += (اعشاری)بفر[i].sample_قدر.uiVal;
    }
    gpu_utilization /= (اعشاری)sample_گنتی;

    // sample count may have been modified by the query above; reset back to buffer size
    sample_گنتی = بفر_حد_بندی;
    nvmlCheck(nvmlDeviceGetSamples(device, NVML_حافظہ_UTILIZATION_SAMPLES, 0, &v_type, &sample_گنتی, بفر));
    اعشاری mem_utilization = 0.f;
    برائے(غیرعلامتی i = 0; i < sample_گنتی; ++i) {
        mem_utilization += (اعشاری)بفر[i].sample_قدر.uiVal;
    }
    mem_utilization /= (اعشاری)sample_گنتی;

    info.gpu_utilization = gpu_utilization;
    info.mem_utilization = mem_utilization;
    واپس info;
}
#ورنہ
GPUUtilInfo get_gpu_utilization_info() {
    fprintf(stderr, "Error: Compiled without nvml support. Cannot perform additional GPU state tracking.");
    باہر_نکلو(EXIT_FAILURE);
}
#اگر_ختم
#اگر_ختم // MFU_H
