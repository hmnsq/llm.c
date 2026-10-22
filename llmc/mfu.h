#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ MFU_H
#ವ್ಯಾಖ್ಯಾನಿಸು MFU_H

#ಅನ್ತರ್ಭಾವ <stdio.h>
#ಅನ್ತರ್ಭಾವ <stdlib.h>
#ಅನ್ತರ್ಭಾವ <string.h>
#ಆದರೆ __has_include(<nvml.h>)
#ವ್ಯಾಖ್ಯಾನಿಸು USE_NVML 1
#ಅನ್ತರ್ಭಾವ <nvml.h>
#ಇಲ್ಲದಿದ್ದರೆ
#ವ್ಯಾಖ್ಯಾನಿಸು USE_NVML 0
#ಆದರೆ_ಅಂತ್ಯ

// tied to enum PrecisionMode, in a future refactor make them the same
#ವ್ಯಾಖ್ಯಾನಿಸು MFUH_PRECISION_FP32 0
#ವ್ಯಾಖ್ಯಾನಿಸು MFUH_PRECISION_FP16 1
#ವ್ಯಾಖ್ಯಾನಿಸು MFUH_PRECISION_BF16 2

#ಆದರೆ USE_NVML
ಸಾಲೊಳಗೆ ಶೂನ್ಯ nvml_check(nvmlReturn_t ಸ್ಥಿತಿ_2, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *file, ಪೂರ್ಣಾಂಕ ಸಾಲು) {
    ಆದರೆ (ಸ್ಥಿತಿ_2 != NVML_SUCCESS) {
        ಬರೆ("[NVML ERROR] at file %s:%d:\n%s\n", file, ಸಾಲು, nvmlErrorString(ಸ್ಥಿತಿ_2));
        ನಿರ್ಗಮಿಸು(EXIT_FAILURE);
    }
};
#ವ್ಯಾಖ್ಯಾನಿಸು nvmlCheck(err) (nvml_check(err, __FILE__, __LINE__))
#ಆದರೆ_ಅಂತ್ಯ


ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    ದಶಮಾಂಶ TF_32;       // tensor-core performance 32 bit
    ದಶಮಾಂಶ BF_16_32;    // bf16 with 32 bit accumulate
    ದಶಮಾಂಶ FP_16_32;    // fp16 with 32 bit accumulate
    ದಶಮಾಂಶ FP_16_16;    // fp16 with 16 bit accumulate
    ದಶಮಾಂಶ FP_8_32;     // and so on
    ದಶಮಾಂಶ FP_8_16;
    ದಶಮಾಂಶ CLOCK;        // clock frequency from the spec sheet
    ದಶಮಾಂಶ CORES;        // #TCs from the spec sheet
} Perf_ದತ್ತಾಂಶ;

// basic default data from the nvidia whitepapers
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ Perf_ದತ್ತಾಂಶ VOLTA = {125.0f, -1.f, 125.f, -1.f, -1.f, -1.f, 1530.f, 640.f};
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ Perf_ದತ್ತಾಂಶ AMPERE_DATACENTER = {156.f, 312.f, 312.f, 312.f, -1.f, -1.f, 1410.f, 432.f};
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ Perf_ದತ್ತಾಂಶ AMPERE_ಗ್ರಾಹಕ = {40.f, 80.f, 80.f, 160.f, -1.f, -1.f, 1860.f, 336.f};
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ Perf_ದತ್ತಾಂಶ HOPPER = {378.f, 756.f, 756.f, 756.f, 1513.f, 1513.f, 1620.f, 456.f};
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ Perf_ದತ್ತಾಂಶ ADA = {82.6f, 165.2f, 165.2f, 330.3f, 330.3f, 660.6f, 2520.f, 512.f};

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* name;
    ಸ್ಥಿರಾಂಕ Perf_ದತ್ತಾಂಶ* perf_ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ new_cores;
    ದಶಮಾಂಶ new_mhz;
} GPUEntry;

// the overrides for each specific GPU
ಸ್ಥಿರ GPUEntry gpu_db[] = {
    {"Tesla V100-SXM2-16GB", &VOLTA, 640, 1530},
    {"Tesla V100-PCIE-32GB", &VOLTA, 640, 1530},
    {"NVIDIA A100-PCIE-40GB", &AMPERE_DATACENTER, 432, 1410},
    {"NVIDIA A100-PCIE-80GB", &AMPERE_DATACENTER, 432, 1410},
    {"NVIDIA A100-SXM4-40GB", &AMPERE_DATACENTER, 432, 1410},
    {"NVIDIA A100-SXM4-80GB", &AMPERE_DATACENTER, 432, 1410},
    {"NVIDIA RTX A2000", &AMPERE_ಗ್ರಾಹಕ, 104, 1200},
    {"NVIDIA RTX A4000", &AMPERE_ಗ್ರಾಹಕ, 192, 1560},
    {"NVIDIA RTX A4500", &AMPERE_ಗ್ರಾಹಕ, 224, 1650},
    {"NVIDIA RTX A5000", &AMPERE_ಗ್ರಾಹಕ, 256, 1695},
    {"NVIDIA RTX A5500", &AMPERE_ಗ್ರಾಹಕ, 320, 1770},
    {"NVIDIA RTX A6000", &AMPERE_ಗ್ರಾಹಕ, 336, 1800},
    {"NVIDIA GeForce RTX 3090 Ti", &AMPERE_ಗ್ರಾಹಕ, 336, 1860},
    {"NVIDIA GeForce RTX 3090", &AMPERE_ಗ್ರಾಹಕ, 328, 1695},
    {"NVIDIA GeForce RTX 3080 Ti", &AMPERE_ಗ್ರಾಹಕ, 320, 1665},
    {"NVIDIA GeForce RTX 3080", &AMPERE_ಗ್ರಾಹಕ, 272, 1710},
    {"NVIDIA GeForce RTX 3070 Ti", &AMPERE_ಗ್ರಾಹಕ, 192, 1770},
    {"NVIDIA GeForce RTX 3070", &AMPERE_ಗ್ರಾಹಕ, 184, 1725},
    {"NVIDIA GeForce RTX 3060 Ti", &AMPERE_ಗ್ರಾಹಕ, 152, 1665},
    {"NVIDIA GeForce RTX 3060", &AMPERE_ಗ್ರಾಹಕ, 112, 1777},
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

ದಶಮಾಂಶ get_flops_promised(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* device, ಪೂರ್ಣಾಂಕ precision_ಬಹುಲಕ) {
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
    ಆದರೆ (!(precision_ಬಹುಲಕ == MFUH_PRECISION_FP32 || precision_ಬಹುಲಕ == MFUH_PRECISION_FP16 || precision_ಬಹುಲಕ == MFUH_PRECISION_BF16)) {
        fprintf(stderr, "Invalid precision mode: %d\n", precision_ಬಹುಲಕ);
        ಹಿಂತಿರುಗಿ -1.0f;
    }

    // do a linear search until you find our GPU, then calculate the flops promised
    ಪೂರ್ಣಾಂಕ ಅಂಕೆ_gpu_entries = ಗಾತ್ರ(gpu_db) / ಗಾತ್ರ(gpu_db[0]);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ಅಂಕೆ_gpu_entries; i++) {
        ಆದರೆ (strcmp(gpu_db[i].name, device) == 0) {
            ಸ್ಥಿರಾಂಕ Perf_ದತ್ತಾಂಶ* perf_ದತ್ತಾಂಶ = gpu_db[i].perf_ದತ್ತಾಂಶ;

            // look up the default flops value for the given precision mode
            ದಶಮಾಂಶ ಮೌಲ್ಯ = -1.0f;
            ಆದರೆ (precision_ಬಹುಲಕ == MFUH_PRECISION_BF16) { ಮೌಲ್ಯ = perf_ದತ್ತಾಂಶ->BF_16_32; }
            ಆದರೆ (precision_ಬಹುಲಕ == MFUH_PRECISION_FP32) { ಮೌಲ್ಯ = perf_ದತ್ತಾಂಶ->TF_32; }
            ಆದರೆ (precision_ಬಹುಲಕ == MFUH_PRECISION_FP16) { ಮೌಲ್ಯ = perf_ದತ್ತಾಂಶ->FP_16_32; }

            // we'd get here if we're e.g. trying to use BF16 on Volta GPU or something...
            ಆದರೆ (ಮೌಲ್ಯ < 0.0f) {
                fprintf(stderr, "No data for GPU %s and precision mode %d\n", device, precision_ಬಹುಲಕ);
                ಹಿಂತಿರುಗಿ -1.0f;
            }

            // adjust flops based on the specific core count and clock frequency of this GPU
            ದಶಮಾಂಶ new_cores = gpu_db[i].new_cores;
            ದಶಮಾಂಶ new_mhz = gpu_db[i].new_mhz;
            ದಶಮಾಂಶ adjusted = ಮೌಲ್ಯ * (new_cores / perf_ದತ್ತಾಂಶ->CORES) * (new_mhz / perf_ದತ್ತಾಂಶ->CLOCK);
            ಹಿಂತಿರುಗಿ adjusted;
        }
    }

    ಹಿಂತಿರುಗಿ -1.0f; // ¯\_(ツ)_/¯
}

ರಚನೆ GPUUtilInfo {
    ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ clock;
    ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ max_clock;
    ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ power;
    ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ power_ಎಲ್ಲೆ;
    ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ fan;
    ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ temperature;
    ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ತಾತ್ಕಾಲಿಕ_slowdown;

    ದಶಮಾಂಶ gpu_utilization;
    ದಶಮಾಂಶ mem_utilization;
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* throttle_reason;
};

// lazily initialize nvml and generate a handle to the GPU
#ಆದರೆ USE_NVML
nvmlDevice_t nvml_get_device() {
    ಸ್ಥಿರ bool needs_init = true;
    ಸ್ಥಿರ nvmlDevice_t device;
    ಆದರೆ(needs_init) {
        needs_init = false;
        nvmlCheck(nvmlInit());
        nvmlCheck(nvml_Device_Get_Handle_By_ಸೂಚ್ಯಂಕ_v2(0, &device));
    }
    ಹಿಂತಿರುಗಿ device;
}

// convert throttle reason bitfield into a text reason.
// this is a lossy conversion; we just want to give some idea of what is happening
ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ* get_throttle_reason(ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ದೀರ್ಘ ಬಿಟ್ಟುಗಳು) {
    ಆದರೆ(ಬಿಟ್ಟುಗಳು & (nvmlClocksThrottleReasonSwPowerCap | nvmlClocksThrottleReasonHwPowerBrakeSlowdown)) {
        ಹಿಂತಿರುಗಿ "power cap";
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಬಿಟ್ಟುಗಳು & (nvmlClocksThrottleReasonSwThermalSlowdown | nvmlClocksThrottleReasonHwThermalSlowdown)) {
        ಹಿಂತಿರುಗಿ "thermal cap";
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ಬಿಟ್ಟುಗಳು & (nvmlClocksThrottleReasonAll)) {
        ಹಿಂತಿರುಗಿ "other cap";
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಹಿಂತಿರುಗಿ "no cap";
    }
}

// gather data for a GPUUtilInfo object
GPUUtilInfo get_gpu_utilization_info() {
    GPUUtilInfo info;
    nvmlDevice_t device = nvml_get_device();
    // query different infos directly
    nvmlCheck(nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &info.clock));
    nvmlCheck(nvmlDeviceGetMaxClockInfo(device, NVML_CLOCK_SM, &info.max_clock));
    nvmlCheck(nvml_Device_Get_Power_Management_ಎಲ್ಲೆ(device, &info.power_ಎಲ್ಲೆ));
    nvmlCheck(nvmlDeviceGetPowerUsage(device, &info.power));
    nvmlCheck(nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &info.temperature));
    nvmlCheck(nvml_Device_Get_Temperature_ಮಿತಿ(device, NVML_TEMPERATURE_ಮಿತಿ_SLOWDOWN, &info.ತಾತ್ಕಾಲಿಕ_slowdown));
    ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ದೀರ್ಘ throttle;
    nvmlCheck(nvmlDeviceGetCurrentClocksThrottleReasons(device, &throttle));
    info.throttle_reason = get_throttle_reason(throttle);
    nvmlCheck(nvml_Device_Get_Fan_ವೇಗ(device, &info.fan));

    // for "utilization", we look at recorded samples. In principle, we could query the driver for how many samples
    // to request, but then we'd need to dynamically allocate sufficient space. Let's just hard-code a limit of 128,
    // and have no memory management required
    constexpr ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಬಫರ್_ಎಲ್ಲೆ = 128;
    nvmlSample_t ಬಫರ್[ಬಫರ್_ಎಲ್ಲೆ];
    nvml_ಮೌಲ್ಯ_Type_t v_type;
    ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ sample_ಎಣಿಕೆ = ಬಫರ್_ಎಲ್ಲೆ;
    nvmlCheck(nvmlDeviceGetSamples(device, NVML_GPU_UTILIZATION_SAMPLES, 0, &v_type, &sample_ಎಣಿಕೆ, ಬಫರ್));
    ದಶಮಾಂಶ gpu_utilization = 0.f;
    ಗಾಗಿ(ಚಿಹ್ನೆರಹಿತ i = 0; i < sample_ಎಣಿಕೆ; ++i) {
        gpu_utilization += (ದಶಮಾಂಶ)ಬಫರ್[i].sample_ಮೌಲ್ಯ.uiVal;
    }
    gpu_utilization /= (ದಶಮಾಂಶ)sample_ಎಣಿಕೆ;

    // sample count may have been modified by the query above; reset back to buffer size
    sample_ಎಣಿಕೆ = ಬಫರ್_ಎಲ್ಲೆ;
    nvmlCheck(nvmlDeviceGetSamples(device, NVML_ಸ್ಮರಣೆ_UTILIZATION_SAMPLES, 0, &v_type, &sample_ಎಣಿಕೆ, ಬಫರ್));
    ದಶಮಾಂಶ mem_utilization = 0.f;
    ಗಾಗಿ(ಚಿಹ್ನೆರಹಿತ i = 0; i < sample_ಎಣಿಕೆ; ++i) {
        mem_utilization += (ದಶಮಾಂಶ)ಬಫರ್[i].sample_ಮೌಲ್ಯ.uiVal;
    }
    mem_utilization /= (ದಶಮಾಂಶ)sample_ಎಣಿಕೆ;

    info.gpu_utilization = gpu_utilization;
    info.mem_utilization = mem_utilization;
    ಹಿಂತಿರುಗಿ info;
}
#ಇಲ್ಲದಿದ್ದರೆ
GPUUtilInfo get_gpu_utilization_info() {
    fprintf(stderr, "Error: Compiled without nvml support. Cannot perform additional GPU state tracking.");
    ನಿರ್ಗಮಿಸು(EXIT_FAILURE);
}
#ಆದರೆ_ಅಂತ್ಯ
#ಆದರೆ_ಅಂತ್ಯ // MFU_H
