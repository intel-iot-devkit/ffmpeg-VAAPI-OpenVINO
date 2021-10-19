# ffmpeg-VAAPI-OpenVINO
Sample for FFmpeg with VAAPI hardware decode and scale, combining OpenVINO iGPU inference  

For FFmpeg user/customer, we provide a solution about FFMPEG iGPU hardware decode connectiong OpenVINO iGPU inference engine avoiding frame data copy. The frame data after decoding/scaling in the buffer can be directly read by iGPU’s inference engine, so that we can have low CPU utilization and maxinum iGPU usage.    
The whole workload runs on iGPU, compared with before ,we have fast pre-process speed and more bandwidth between CPU/iGPU communication.

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
