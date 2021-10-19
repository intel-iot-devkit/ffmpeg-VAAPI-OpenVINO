# ffmpeg-VAAPI-OpenVINO
Sample for FFmpeg with VAAPI hardware decode and scale, combining OpenVINO iGPU inference with no frame data copy between CPu and iCPU. 
The whole workload runs on iGPU, fast Uspeed, low CPU utilization and make full use of iGPU decode unit.
For FFmpeg user/customer, we provide a solution about FFMPEG iGPU hardwaredecode connectiong OpenVINO iGPU inference engine avoiding frame data copy. The frame data after decoding/scaling can be read by iGPU’s inference engine, so that we can have low CPU utilization and maxinum iGPU usage.  


## Dependences：

* Intel(R) OpenVINO(TM) Toolkit
* FFmpeg
* VA-API (libVA)


## Device request:

Intel(R) CPU with Integrated Graphics

## Validated software version:

* OpenVINO 2020.3 LTS
* FFmpeg 4.3.1
* LibVA 2.1.0, VA-VAPI 1.6.0 Intel iHD driver 19.3.1
