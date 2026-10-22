#నిర్వచితం_కాకపోతే MFU_H
#నిర్వచించు MFU_H

#అన్తర్భావ <stdio.h>
#అన్తర్భావ <stdlib.h>
#అన్తర్భావ <string.h>
#ఒకవేళ __has_include(<nvml.h>)
#నిర్వచించు USE_NVML 1
#అన్తర్భావ <nvml.h>
#లేకపోతే
#నిర్వచించు USE_NVML 0
#అయితే_ముగింపు

// tied to enum PrecisionMode, in a future refactor make them the same
#నిర్వచించు MFUH_PRECISION_FP32 0
#నిర్వచించు MFUH_PRECISION_FP16 1
#నిర్వచించు MFUH_PRECISION_BF16 2

#ఒకవేళ USE_NVML
అంతఃవరుస శూన్య nvml_check(nvmlReturn_t స్థితి_2, స్థిరరాశి అక్షరం *file, పూర్ణాంకం పంక్తి) {
    ఒకవేళ (స్థితి_2 != NVML_SUCCESS) {
        రాయి("[NVML ERROR] at file %s:%d:\n%s\n", file, పంక్తి, nvmlErrorString(స్థితి_2));
        నిష్క్రమించు(EXIT_FAILURE);
    }
};
#నిర్వచించు nvmlCheck(err) (nvml_check(err, __FILE__, __LINE__))
#అయితే_ముగింపు


రకనిర్వచనం నిర్మాణం {
    దశాంశం TF_32;       // tensor-core performance 32 bit
    దశాంశం BF_16_32;    // bf16 with 32 bit accumulate
    దశాంశం FP_16_32;    // fp16 with 32 bit accumulate
    దశాంశం FP_16_16;    // fp16 with 16 bit accumulate
    దశాంశం FP_8_32;     // and so on
    దశాంశం FP_8_16;
    దశాంశం CLOCK;        // clock frequency from the spec sheet
    దశాంశం CORES;        // #TCs from the spec sheet
} Perf_దత్తాంశం;

// basic default data from the nvidia whitepapers
స్థిర స్థిరరాశి Perf_దత్తాంశం VOLTA = {125.0f, -1.f, 125.f, -1.f, -1.f, -1.f, 1530.f, 640.f};
స్థిర స్థిరరాశి Perf_దత్తాంశం AMPERE_DATACENTER = {156.f, 312.f, 312.f, 312.f, -1.f, -1.f, 1410.f, 432.f};
స్థిర స్థిరరాశి Perf_దత్తాంశం AMPERE_వినియోగదారు = {40.f, 80.f, 80.f, 160.f, -1.f, -1.f, 1860.f, 336.f};
స్థిర స్థిరరాశి Perf_దత్తాంశం HOPPER = {378.f, 756.f, 756.f, 756.f, 1513.f, 1513.f, 1620.f, 456.f};
స్థిర స్థిరరాశి Perf_దత్తాంశం ADA = {82.6f, 165.2f, 165.2f, 330.3f, 330.3f, 660.6f, 2520.f, 512.f};

రకనిర్వచనం నిర్మాణం {
    స్థిరరాశి అక్షరం* name;
    స్థిరరాశి Perf_దత్తాంశం* perf_దత్తాంశం;
    దశాంశం new_cores;
    దశాంశం new_mhz;
} GPUEntry;

// the overrides for each specific GPU
స్థిర GPUEntry gpu_db[] = {
    {"Tesla V100-SXM2-16GB", &VOLTA, 640, 1530},
    {"Tesla V100-PCIE-32GB", &VOLTA, 640, 1530},
    {"NVIDIA A100-PCIE-40GB", &AMPERE_DATACENTER, 432, 1410},
    {"NVIDIA A100-PCIE-80GB", &AMPERE_DATACENTER, 432, 1410},
    {"NVIDIA A100-SXM4-40GB", &AMPERE_DATACENTER, 432, 1410},
    {"NVIDIA A100-SXM4-80GB", &AMPERE_DATACENTER, 432, 1410},
    {"NVIDIA RTX A2000", &AMPERE_వినియోగదారు, 104, 1200},
    {"NVIDIA RTX A4000", &AMPERE_వినియోగదారు, 192, 1560},
    {"NVIDIA RTX A4500", &AMPERE_వినియోగదారు, 224, 1650},
    {"NVIDIA RTX A5000", &AMPERE_వినియోగదారు, 256, 1695},
    {"NVIDIA RTX A5500", &AMPERE_వినియోగదారు, 320, 1770},
    {"NVIDIA RTX A6000", &AMPERE_వినియోగదారు, 336, 1800},
    {"NVIDIA GeForce RTX 3090 Ti", &AMPERE_వినియోగదారు, 336, 1860},
    {"NVIDIA GeForce RTX 3090", &AMPERE_వినియోగదారు, 328, 1695},
    {"NVIDIA GeForce RTX 3080 Ti", &AMPERE_వినియోగదారు, 320, 1665},
    {"NVIDIA GeForce RTX 3080", &AMPERE_వినియోగదారు, 272, 1710},
    {"NVIDIA GeForce RTX 3070 Ti", &AMPERE_వినియోగదారు, 192, 1770},
    {"NVIDIA GeForce RTX 3070", &AMPERE_వినియోగదారు, 184, 1725},
    {"NVIDIA GeForce RTX 3060 Ti", &AMPERE_వినియోగదారు, 152, 1665},
    {"NVIDIA GeForce RTX 3060", &AMPERE_వినియోగదారు, 112, 1777},
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

దశాంశం get_flops_promised(స్థిరరాశి అక్షరం* device, పూర్ణాంకం precision_బహుళకం) {
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
    ఒకవేళ (!(precision_బహుళకం == MFUH_PRECISION_FP32 || precision_బహుళకం == MFUH_PRECISION_FP16 || precision_బహుళకం == MFUH_PRECISION_BF16)) {
        fprintf(stderr, "Invalid precision mode: %d\n", precision_బహుళకం);
        తిరిగి -1.0f;
    }

    // do a linear search until you find our GPU, then calculate the flops promised
    పూర్ణాంకం అంకె_gpu_entries = పరిమాణం(gpu_db) / పరిమాణం(gpu_db[0]);
    కొరకు (పూర్ణాంకం i = 0; i < అంకె_gpu_entries; i++) {
        ఒకవేళ (strcmp(gpu_db[i].name, device) == 0) {
            స్థిరరాశి Perf_దత్తాంశం* perf_దత్తాంశం = gpu_db[i].perf_దత్తాంశం;

            // look up the default flops value for the given precision mode
            దశాంశం విలువ = -1.0f;
            ఒకవేళ (precision_బహుళకం == MFUH_PRECISION_BF16) { విలువ = perf_దత్తాంశం->BF_16_32; }
            ఒకవేళ (precision_బహుళకం == MFUH_PRECISION_FP32) { విలువ = perf_దత్తాంశం->TF_32; }
            ఒకవేళ (precision_బహుళకం == MFUH_PRECISION_FP16) { విలువ = perf_దత్తాంశం->FP_16_32; }

            // we'd get here if we're e.g. trying to use BF16 on Volta GPU or something...
            ఒకవేళ (విలువ < 0.0f) {
                fprintf(stderr, "No data for GPU %s and precision mode %d\n", device, precision_బహుళకం);
                తిరిగి -1.0f;
            }

            // adjust flops based on the specific core count and clock frequency of this GPU
            దశాంశం new_cores = gpu_db[i].new_cores;
            దశాంశం new_mhz = gpu_db[i].new_mhz;
            దశాంశం adjusted = విలువ * (new_cores / perf_దత్తాంశం->CORES) * (new_mhz / perf_దత్తాంశం->CLOCK);
            తిరిగి adjusted;
        }
    }

    తిరిగి -1.0f; // ¯\_(ツ)_/¯
}

నిర్మాణం GPUUtilInfo {
    గుర్తులేని పూర్ణాంకం clock;
    గుర్తులేని పూర్ణాంకం max_clock;
    గుర్తులేని పూర్ణాంకం power;
    గుర్తులేని పూర్ణాంకం power_హద్దు;
    గుర్తులేని పూర్ణాంకం fan;
    గుర్తులేని పూర్ణాంకం temperature;
    గుర్తులేని పూర్ణాంకం తాత్కాలిక_slowdown;

    దశాంశం gpu_utilization;
    దశాంశం mem_utilization;
    స్థిరరాశి అక్షరం* throttle_reason;
};

// lazily initialize nvml and generate a handle to the GPU
#ఒకవేళ USE_NVML
nvmlDevice_t nvml_get_device() {
    స్థిర bool needs_init = true;
    స్థిర nvmlDevice_t device;
    ఒకవేళ(needs_init) {
        needs_init = false;
        nvmlCheck(nvmlInit());
        nvmlCheck(nvml_Device_Get_Handle_By_సూచీ_v2(0, &device));
    }
    తిరిగి device;
}

// convert throttle reason bitfield into a text reason.
// this is a lossy conversion; we just want to give some idea of what is happening
స్థిరరాశి అక్షరం* get_throttle_reason(గుర్తులేని దీర్ఘ దీర్ఘ బిట్లు) {
    ఒకవేళ(బిట్లు & (nvmlClocksThrottleReasonSwPowerCap | nvmlClocksThrottleReasonHwPowerBrakeSlowdown)) {
        తిరిగి "power cap";
    } లేకపోతే ఒకవేళ (బిట్లు & (nvmlClocksThrottleReasonSwThermalSlowdown | nvmlClocksThrottleReasonHwThermalSlowdown)) {
        తిరిగి "thermal cap";
    } లేకపోతే ఒకవేళ (బిట్లు & (nvmlClocksThrottleReasonAll)) {
        తిరిగి "other cap";
    } లేకపోతే {
        తిరిగి "no cap";
    }
}

// gather data for a GPUUtilInfo object
GPUUtilInfo get_gpu_utilization_info() {
    GPUUtilInfo info;
    nvmlDevice_t device = nvml_get_device();
    // query different infos directly
    nvmlCheck(nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &info.clock));
    nvmlCheck(nvmlDeviceGetMaxClockInfo(device, NVML_CLOCK_SM, &info.max_clock));
    nvmlCheck(nvml_Device_Get_Power_Management_హద్దు(device, &info.power_హద్దు));
    nvmlCheck(nvmlDeviceGetPowerUsage(device, &info.power));
    nvmlCheck(nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &info.temperature));
    nvmlCheck(nvml_Device_Get_Temperature_పరిమితి(device, NVML_TEMPERATURE_పరిమితి_SLOWDOWN, &info.తాత్కాలిక_slowdown));
    గుర్తులేని దీర్ఘ దీర్ఘ throttle;
    nvmlCheck(nvmlDeviceGetCurrentClocksThrottleReasons(device, &throttle));
    info.throttle_reason = get_throttle_reason(throttle);
    nvmlCheck(nvml_Device_Get_Fan_వేగం(device, &info.fan));

    // for "utilization", we look at recorded samples. In principle, we could query the driver for how many samples
    // to request, but then we'd need to dynamically allocate sufficient space. Let's just hard-code a limit of 128,
    // and have no memory management required
    constexpr స్థిరరాశి పూర్ణాంకం బఫరు_హద్దు = 128;
    nvmlSample_t బఫరు[బఫరు_హద్దు];
    nvml_విలువ_Type_t v_type;
    గుర్తులేని పూర్ణాంకం sample_లెక్క = బఫరు_హద్దు;
    nvmlCheck(nvmlDeviceGetSamples(device, NVML_GPU_UTILIZATION_SAMPLES, 0, &v_type, &sample_లెక్క, బఫరు));
    దశాంశం gpu_utilization = 0.f;
    కొరకు(గుర్తులేని i = 0; i < sample_లెక్క; ++i) {
        gpu_utilization += (దశాంశం)బఫరు[i].sample_విలువ.uiVal;
    }
    gpu_utilization /= (దశాంశం)sample_లెక్క;

    // sample count may have been modified by the query above; reset back to buffer size
    sample_లెక్క = బఫరు_హద్దు;
    nvmlCheck(nvmlDeviceGetSamples(device, NVML_జ్ఞాపకం_UTILIZATION_SAMPLES, 0, &v_type, &sample_లెక్క, బఫరు));
    దశాంశం mem_utilization = 0.f;
    కొరకు(గుర్తులేని i = 0; i < sample_లెక్క; ++i) {
        mem_utilization += (దశాంశం)బఫరు[i].sample_విలువ.uiVal;
    }
    mem_utilization /= (దశాంశం)sample_లెక్క;

    info.gpu_utilization = gpu_utilization;
    info.mem_utilization = mem_utilization;
    తిరిగి info;
}
#లేకపోతే
GPUUtilInfo get_gpu_utilization_info() {
    fprintf(stderr, "Error: Compiled without nvml support. Cannot perform additional GPU state tracking.");
    నిష్క్రమించు(EXIT_FAILURE);
}
#అయితే_ముగింపు
#అయితే_ముగింపు // MFU_H
