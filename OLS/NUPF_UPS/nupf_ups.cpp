#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>
#include <clFFT.h>  

#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>

constexpr unsigned N_SAMPLES = 8192;
constexpr unsigned B = 128;  

constexpr unsigned NUM_SEGMENTS = 3;
constexpr unsigned L_SIZES[NUM_SEGMENTS] = {128, 256, 512};    
constexpr unsigned BLOCK_SIZES[NUM_SEGMENTS] = {128, 256, 512}; 
constexpr unsigned P_VALUES[NUM_SEGMENTS] = {2, 2, 2};         

constexpr unsigned FIR_LEN = 933;

constexpr unsigned TOTAL_PARTS = P_VALUES[0] + P_VALUES[1] + P_VALUES[2];
constexpr unsigned FFT_SIZES[NUM_SEGMENTS] = {256, 512, 1024}; 
constexpr unsigned SPEC_SIZES[NUM_SEGMENTS] = {
    FFT_SIZES[0]/2 + 1,
    FFT_SIZES[1]/2 + 1,
    FFT_SIZES[2]/2 + 1
};  

constexpr unsigned BLKS_IN = N_SAMPLES / B;
constexpr unsigned OUT_LEN = N_SAMPLES + FIR_LEN - 1;

static inline double now_ns() {
  return std::chrono::duration<double, std::nano>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

static void dump(const char *fn, const std::vector<float> &v) {
  std::ofstream(fn, std::ios::binary)
      .write(reinterpret_cast<const char *>(v.data()),
             v.size() * sizeof(float));
}

static void hannLPF(std::vector<float> &h, float fc_norm) {
  const float M = float(h.size() - 1);
  for (unsigned n = 0; n < h.size(); n++) {
    const float w = 0.5f * (1.0f - std::cos(2.0f * M_PI * n / M));
    const float t = float(n) - M / 2.0f;
    const float si = (t == 0.0f) ? 1.0f
                                 : sin(2.0f * M_PI * fc_norm * t) /
                                       (M_PI * t * fc_norm * 2.0f);
    h[n] = w * si;
  }
}

static const char *kSrc = R"CLC(
__kernel void freqMult(__global const float2* input_spectrum,
                       __global const float2* filter_spectrum,
                       __global float2* output_spectrum,
                       const uint filter_offset)
{
    const uint gid = get_global_id(0);
    float2 a = input_spectrum[gid];
    float2 b = filter_spectrum[filter_offset + gid];
    
    output_spectrum[gid] = (float2)(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

__kernel void freqAccum(__global float2* accum_spectrum,
                        __global const float2* part_spectrum)
{
    const uint gid = get_global_id(0);
    accum_spectrum[gid] += part_spectrum[gid];
}
)CLC";

int main() {
  std::vector<float> x(N_SAMPLES);
  for (unsigned n = 0; n < N_SAMPLES; n++)
    x[n] = sinf(2.0f * M_PI * 100.0f * n / N_SAMPLES);
  dump("input.bin", x);

  std::vector<float> h(FIR_LEN);
  hannLPF(h, 500.0f / 8192.0f);
  dump("fir.bin", h);

  cl_int err;
  cl_platform_id plat;
  cl_device_id dev;
  clGetPlatformIDs(1, &plat, nullptr);
  clGetDeviceIDs(plat, CL_DEVICE_TYPE_DEFAULT, 1, &dev, nullptr);

  cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
  cl_command_queue q =
      clCreateCommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE, &err);
  cl_program pr = clCreateProgramWithSource(ctx, 1, &kSrc, nullptr, &err);
  clBuildProgram(pr, 1, &dev, "", nullptr, nullptr);
  
  cl_kernel kMult = clCreateKernel(pr, "freqMult", &err);
  cl_kernel kAccum = clCreateKernel(pr, "freqAccum", &err);

  unsigned segmentOffsets[NUM_SEGMENTS];
  segmentOffsets[0] = 0;
  for (unsigned i = 1; i < NUM_SEGMENTS; i++) {
    segmentOffsets[i] = segmentOffsets[i-1] + L_SIZES[i-1] * P_VALUES[i-1];
  }

  unsigned maxFFTSize = FFT_SIZES[NUM_SEGMENTS-1];
  
  std::vector<cl_mem> dInputs(NUM_SEGMENTS);
  std::vector<cl_mem> dInputSpecs(NUM_SEGMENTS);
  std::vector<cl_mem> dOutputSpecs(NUM_SEGMENTS);
  std::vector<cl_mem> dPartSpecs(NUM_SEGMENTS);
  std::vector<cl_mem> dOutputs(NUM_SEGMENTS);
  std::vector<cl_mem> dFilterSpecs(NUM_SEGMENTS);
  
  std::vector<std::vector<cl_mem>> dFDLs(NUM_SEGMENTS);
  
  std::vector<clfftPlanHandle> planForwards(NUM_SEGMENTS);
  std::vector<clfftPlanHandle> planInverses(NUM_SEGMENTS);
  
  clfftSetupData fftSetup;
  clfftSetup(&fftSetup);
  
  for (unsigned segIdx = 0; segIdx < NUM_SEGMENTS; segIdx++) {
    dInputs[segIdx] = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 
                                    FFT_SIZES[segIdx] * sizeof(float), nullptr, &err);
    dInputSpecs[segIdx] = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 
                                        SPEC_SIZES[segIdx] * sizeof(cl_float2), nullptr, &err);
    dOutputSpecs[segIdx] = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 
                                         SPEC_SIZES[segIdx] * sizeof(cl_float2), nullptr, &err);
    dPartSpecs[segIdx] = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 
                                       SPEC_SIZES[segIdx] * sizeof(cl_float2), nullptr, &err);
    dOutputs[segIdx] = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, 
                                     FFT_SIZES[segIdx] * sizeof(float), nullptr, &err);
    dFilterSpecs[segIdx] = clCreateBuffer(ctx, CL_MEM_READ_ONLY, 
                                         P_VALUES[segIdx] * SPEC_SIZES[segIdx] * sizeof(cl_float2), nullptr, &err);
    
    dFDLs[segIdx].resize(P_VALUES[segIdx]);
    for (unsigned i = 0; i < P_VALUES[segIdx]; i++) {
      dFDLs[segIdx][i] = clCreateBuffer(ctx, CL_MEM_READ_WRITE, 
                                       SPEC_SIZES[segIdx] * sizeof(cl_float2), nullptr, &err);
    }
    
    size_t fftDim[] = {FFT_SIZES[segIdx]};
    clfftCreateDefaultPlan(&planForwards[segIdx], ctx, CLFFT_1D, fftDim);
    clfftCreateDefaultPlan(&planInverses[segIdx], ctx, CLFFT_1D, fftDim);
    clfftSetPlanScale(planInverses[segIdx], CLFFT_BACKWARD, 1.0f / static_cast<float>(FFT_SIZES[segIdx]));
    clfftSetPlanPrecision(planForwards[segIdx], CLFFT_SINGLE);
    clfftSetPlanPrecision(planInverses[segIdx], CLFFT_SINGLE);
    clfftSetLayout(planForwards[segIdx], CLFFT_REAL, CLFFT_HERMITIAN_INTERLEAVED);
    clfftSetLayout(planInverses[segIdx], CLFFT_HERMITIAN_INTERLEAVED, CLFFT_REAL);
    clfftSetResultLocation(planForwards[segIdx], CLFFT_OUTOFPLACE);
    clfftSetResultLocation(planInverses[segIdx], CLFFT_OUTOFPLACE);
    clfftBakePlan(planForwards[segIdx], 1, &q, nullptr, nullptr);
    clfftBakePlan(planInverses[segIdx], 1, &q, nullptr, nullptr);
  }

  for (unsigned segIdx = 0; segIdx < NUM_SEGMENTS; segIdx++) {
    std::vector<float> filterSegment(FFT_SIZES[segIdx], 0.0f);
    std::vector<cl_float2> filterSpectrum(P_VALUES[segIdx] * SPEC_SIZES[segIdx]);
    
    for (unsigned p = 0; p < P_VALUES[segIdx]; p++) {
      std::memset(filterSegment.data(), 0, FFT_SIZES[segIdx] * sizeof(float));
      
      unsigned segmentStart = segmentOffsets[segIdx] + p * L_SIZES[segIdx];
      
      if (segmentStart >= FIR_LEN) continue;
      
      const unsigned segmentLength = std::min(L_SIZES[segIdx], FIR_LEN - segmentStart);
      for (unsigned k = 0; k < segmentLength; k++) {
        if (segmentStart + k < FIR_LEN) {
          filterSegment[k] = h[segmentStart + k];
        }
      }
      
      clEnqueueWriteBuffer(q, dInputs[segIdx], CL_TRUE, 0, 
                          FFT_SIZES[segIdx] * sizeof(float), 
                          filterSegment.data(), 0, nullptr, nullptr);
      clfftEnqueueTransform(planForwards[segIdx], CLFFT_FORWARD, 1, &q, 0, 
                           nullptr, nullptr, &dInputs[segIdx], &dInputSpecs[segIdx], nullptr);
      clFinish(q);
      
      clEnqueueReadBuffer(q, dInputSpecs[segIdx], CL_TRUE, 0, 
                         SPEC_SIZES[segIdx] * sizeof(cl_float2),
                         &filterSpectrum[p * SPEC_SIZES[segIdx]], 0, nullptr, nullptr);
    }
    
    clEnqueueWriteBuffer(q, dFilterSpecs[segIdx], CL_TRUE, 0, 
                        P_VALUES[segIdx] * SPEC_SIZES[segIdx] * sizeof(cl_float2),
                        filterSpectrum.data(), 0, nullptr, nullptr);
  }

  std::vector<float> yGPU(OUT_LEN, 0.0f);
  std::vector<std::vector<float>> tempBuffers(NUM_SEGMENTS);
  
  for (unsigned segIdx = 0; segIdx < NUM_SEGMENTS; segIdx++) {
    tempBuffers[segIdx].resize(FFT_SIZES[segIdx], 0.0f);
  }

  unsigned maxBlocks = (N_SAMPLES + FIR_LEN - 1) / B + 1;
  
  double kernel_ns = 0.0;
  
  std::vector<float> inputBuffer(maxFFTSize, 0.0f);
  
  std::vector<unsigned> segmentCycles(NUM_SEGMENTS);
  for (unsigned segIdx = 0; segIdx < NUM_SEGMENTS; segIdx++) {
    segmentCycles[segIdx] = BLOCK_SIZES[segIdx] / B;
  }
  
  for (unsigned blk = 0; blk < maxBlocks; blk++) {
    for (unsigned segIdx = 0; segIdx < NUM_SEGMENTS; segIdx++) {
      if (blk % segmentCycles[segIdx] != 0 && segIdx > 0) continue;
      
      unsigned segBlk = blk / segmentCycles[segIdx];
      unsigned segBlksIn = (N_SAMPLES + BLOCK_SIZES[segIdx] - 1) / BLOCK_SIZES[segIdx];
      unsigned segBlksAll = segBlksIn + P_VALUES[segIdx] - 1;
      
      if (segBlk >= segBlksAll) continue;
      
      std::memset(inputBuffer.data(), 0, FFT_SIZES[segIdx] * sizeof(float));
      
      if (segBlk < segBlksIn) {
        unsigned startSample = segBlk * BLOCK_SIZES[segIdx];
        unsigned copyLen = std::min(BLOCK_SIZES[segIdx], N_SAMPLES - startSample);
        if (copyLen > 0) {
          std::memcpy(inputBuffer.data(), x.data() + startSample, copyLen * sizeof(float));
        }
      }
      
      clEnqueueWriteBuffer(q, dInputs[segIdx], CL_TRUE, 0, 
                          FFT_SIZES[segIdx] * sizeof(float), 
                          inputBuffer.data(), 0, nullptr, nullptr);
      clfftEnqueueTransform(planForwards[segIdx], CLFFT_FORWARD, 1, &q, 0, 
                           nullptr, nullptr, &dInputs[segIdx], &dInputSpecs[segIdx], nullptr);
      
      for (int i = P_VALUES[segIdx]-1; i > 0; i--) {
        clEnqueueCopyBuffer(q, dFDLs[segIdx][i-1], dFDLs[segIdx][i], 0, 0, 
                           SPEC_SIZES[segIdx] * sizeof(cl_float2),
                           0, nullptr, nullptr);
      }
      
      clEnqueueCopyBuffer(q, dInputSpecs[segIdx], dFDLs[segIdx][0], 0, 0, 
                         SPEC_SIZES[segIdx] * sizeof(cl_float2),
                         0, nullptr, nullptr);
      
      cl_float2 zeros = {0.0f, 0.0f};
      clEnqueueFillBuffer(q, dOutputSpecs[segIdx], &zeros, sizeof(cl_float2), 0, 
                         SPEC_SIZES[segIdx] * sizeof(cl_float2), 0, nullptr, nullptr);
      
      const unsigned maxPart = (segBlk < P_VALUES[segIdx]) ? segBlk : P_VALUES[segIdx] - 1;
      
      for (unsigned part = 0; part <= maxPart; part++) {
        cl_uint filterOffset = part * SPEC_SIZES[segIdx];
        
        clSetKernelArg(kMult, 0, sizeof(cl_mem), &dFDLs[segIdx][part]);
        clSetKernelArg(kMult, 1, sizeof(cl_mem), &dFilterSpecs[segIdx]);
        clSetKernelArg(kMult, 2, sizeof(cl_mem), &dPartSpecs[segIdx]);
        clSetKernelArg(kMult, 3, sizeof(cl_uint), &filterOffset);
        
        size_t specSize = SPEC_SIZES[segIdx];
        cl_event ev;
        clEnqueueNDRangeKernel(q, kMult, 1, nullptr, &specSize, nullptr, 
                              0, nullptr, &ev);
        clWaitForEvents(1, &ev);
        
        cl_ulong ts, te;
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(ts), &ts, nullptr);
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(te), &te, nullptr);
        kernel_ns += double(te - ts);
        clReleaseEvent(ev);
        
        clSetKernelArg(kAccum, 0, sizeof(cl_mem), &dOutputSpecs[segIdx]);
        clSetKernelArg(kAccum, 1, sizeof(cl_mem), &dPartSpecs[segIdx]);
        
        cl_event ev2;
        clEnqueueNDRangeKernel(q, kAccum, 1, nullptr, &specSize, nullptr, 
                              0, nullptr, &ev2);
        clWaitForEvents(1, &ev2);
        clReleaseEvent(ev2);
      }
      
      clfftEnqueueTransform(planInverses[segIdx], CLFFT_BACKWARD, 1, &q, 0, 
                           nullptr, nullptr, &dOutputSpecs[segIdx], &dOutputs[segIdx], nullptr);
      
      clEnqueueReadBuffer(q, dOutputs[segIdx], CL_TRUE, 0, 
                         FFT_SIZES[segIdx] * sizeof(float),
                         tempBuffers[segIdx].data(), 0, nullptr, nullptr);

      unsigned segmentOffset = segmentOffsets[segIdx];
      unsigned outputStart = segBlk * BLOCK_SIZES[segIdx] + segmentOffset;
      
      if (segBlk > 0) {
        for (unsigned i = 0; i < BLOCK_SIZES[segIdx]; i++) {
          if (outputStart - BLOCK_SIZES[segIdx] + i < OUT_LEN) {
            yGPU[outputStart - BLOCK_SIZES[segIdx] + i] += tempBuffers[segIdx][i];
          }
        }
      }
      
      for (unsigned i = 0; i < BLOCK_SIZES[segIdx]; i++) {
        if (outputStart + i < OUT_LEN) {
          yGPU[outputStart + i] += tempBuffers[segIdx][FFT_SIZES[segIdx] - BLOCK_SIZES[segIdx] + i];
        }
      }
    }
  }

  std::cout << "GPU kernel time: " << kernel_ns * 1e-6 << " ms  ("
            << kernel_ns * 1e-3 / BLKS_IN << " µs/block)\n";

  dump("gpu.bin", yGPU);

  for (unsigned segIdx = 0; segIdx < NUM_SEGMENTS; segIdx++) {
    clfftDestroyPlan(&planForwards[segIdx]);
    clfftDestroyPlan(&planInverses[segIdx]);
    
    for (unsigned i = 0; i < P_VALUES[segIdx]; i++) {
      clReleaseMemObject(dFDLs[segIdx][i]);
    }
    
    clReleaseMemObject(dInputs[segIdx]);
    clReleaseMemObject(dInputSpecs[segIdx]);
    clReleaseMemObject(dFilterSpecs[segIdx]);
    clReleaseMemObject(dOutputSpecs[segIdx]);
    clReleaseMemObject(dPartSpecs[segIdx]);
    clReleaseMemObject(dOutputs[segIdx]);
  }
  
  clfftTeardown();
  
  clReleaseKernel(kMult);
  clReleaseKernel(kAccum);
  clReleaseProgram(pr);
  clReleaseCommandQueue(q);
  clReleaseContext(ctx);

  std::cout << "done – binaries written. Run python3 check_conv.py\n";
  return 0;
}