# ffmpeg-VAAPI-OpenVINO
Demo on iGPU for FFmpeg decode and scale, OpenVINO inference. All workload run on iGPU, fast Uspeed, low CPU utilization and make full use of iGPU.this is zero-copy solution, which means No frame data copy from CPU to iGPU.

Dependences：
Intel(R) OpenVINO(TM) Toolkit
FFmpeg
VA-API


Device request:
Intel(R) CPU with Integrated Graphics

Validated software version
OpenVINO 2020.3 LTS
FFmpeg 4.3.1
LibVA 2.1.0, VA-VAPI 1.6.0 Intel iHD driver 19.3.1
