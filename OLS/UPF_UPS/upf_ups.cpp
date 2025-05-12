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
constexpr unsigned L = 256;
constexpr unsigned FIR_LEN = 933;
constexpr unsigned P = FIR_LEN / L + (FIR_LEN % L != 0 ? 1 : 0);  
constexpr unsigned B = L;

constexpr unsigned FFT_SIZE = 512;  
constexpr unsigned SPEC_SIZE = FFT_SIZE/2 + 1;  

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

static void refConv(const std::vector<double> &x, const std::vector<double> &h,
                    std::vector<double> &y) {
  const std::size_t Nx = x.size(), Nh = h.size(), Ny = Nx + Nh - 1;
  y.assign(Ny, 0.0);
  for (std::size_t n = 0; n < Ny; n++) {
    const std::size_t kmin = (n >= Nh - 1) ? n - (Nh - 1) : 0;
    const std::size_t kmax = (n < Nx - 1) ? n : Nx - 1;
    double acc = 0.0;
    for (std::size_t k = kmin; k <= kmax; k++)
      acc += x[k] * h[n - k];
    y[n] = acc;
  }
}

static const char *kSrc = R"CLC(
// Complex multiplication kernel
__kernel void freqMult(__global const float2* input_spectrum,
                       __global const float2* filter_spectrum,
                       __global float2* output_spectrum,
                       const uint filter_offset)
{
    const uint gid = get_global_id(0);
    float2 a = input_spectrum[gid];
    float2 b = filter_spectrum[filter_offset + gid];
    
    // Complex multiplication: (a+bi)*(c+di) = (ac-bd) + (ad+bc)i
    output_spectrum[gid] = (float2)(a.x*b.x - a.y*b.y, a.x*b.y + a.y*b.x);
}

// Accumulate in frequency domain
__kernel void freqAccum(__global float2* accum_spectrum,
                        __global const float2* part_spectrum)
{
    const uint gid = get_global_id(0);
    accum_spectrum[gid] += part_spectrum[gid];
}
)CLC";

int main() {
  std::vector<float> x(N_SAMPLES);
  static std::vector<float> overlap(B, 0.0f);
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

  cl_mem dInput = clCreateBuffer(ctx, CL_MEM_READ_WRITE, FFT_SIZE * sizeof(float), nullptr, &err);
  cl_mem dInputSpec = clCreateBuffer(ctx, CL_MEM_READ_WRITE, SPEC_SIZE * sizeof(cl_float2), nullptr, &err);
  cl_mem dFilterSpec = clCreateBuffer(ctx, CL_MEM_READ_ONLY, P * SPEC_SIZE * sizeof(cl_float2), nullptr, &err);
  cl_mem dOutputSpec = clCreateBuffer(ctx, CL_MEM_READ_WRITE, SPEC_SIZE * sizeof(cl_float2), nullptr, &err);
  cl_mem dPartSpec = clCreateBuffer(ctx, CL_MEM_READ_WRITE, SPEC_SIZE * sizeof(cl_float2), nullptr, &err);
  cl_mem dOutput = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, FFT_SIZE * sizeof(float), nullptr, &err);

  clfftPlanHandle planForward, planInverse;
  clfftSetupData fftSetup;
  clfftSetup(&fftSetup);
  size_t fftDim[] = {FFT_SIZE};
  clfftCreateDefaultPlan(&planForward, ctx, CLFFT_1D, fftDim);
  clfftCreateDefaultPlan(&planInverse, ctx, CLFFT_1D, fftDim);
  clfftSetPlanPrecision(planForward, CLFFT_SINGLE);
  clfftSetPlanPrecision(planInverse, CLFFT_SINGLE);
  clfftSetLayout(planForward, CLFFT_REAL, CLFFT_HERMITIAN_INTERLEAVED);
  clfftSetLayout(planInverse, CLFFT_HERMITIAN_INTERLEAVED, CLFFT_REAL);
  clfftSetResultLocation(planForward, CLFFT_OUTOFPLACE);
  clfftSetResultLocation(planInverse, CLFFT_OUTOFPLACE);
  clfftBakePlan(planForward, 1, &q, nullptr, nullptr);
  clfftBakePlan(planInverse, 1, &q, nullptr, nullptr);

  std::vector<cl_mem> dFDL(P);
  for (unsigned i = 0; i < P; i++)
    dFDL[i] = clCreateBuffer(ctx, CL_MEM_READ_WRITE, SPEC_SIZE * sizeof(cl_float2), nullptr, &err);

  std::vector<float> filterSegment(FFT_SIZE, 0.0f);
  std::vector<cl_float2> filterSpectrum(P * SPEC_SIZE);

  for (unsigned p = 0; p < P; p++) {
    std::memset(filterSegment.data(), 0, FFT_SIZE * sizeof(float));
    const unsigned segmentLength = std::min(L, FIR_LEN - p*L);
    for (unsigned k = 0; k < segmentLength; k++)
      filterSegment[k] = h[p*L + k];
    
    clEnqueueWriteBuffer(q, dInput, CL_TRUE, 0, FFT_SIZE * sizeof(float), 
                        filterSegment.data(), 0, nullptr, nullptr);
    clfftEnqueueTransform(planForward, CLFFT_FORWARD, 1, &q, 0, nullptr, nullptr,
                         &dInput, &dInputSpec, nullptr);
    clFinish(q);
    
    clEnqueueReadBuffer(q, dInputSpec, CL_TRUE, 0, SPEC_SIZE * sizeof(cl_float2),
                       &filterSpectrum[p * SPEC_SIZE], 0, nullptr, nullptr);
  }

  clEnqueueWriteBuffer(q, dFilterSpec, CL_TRUE, 0, P * SPEC_SIZE * sizeof(cl_float2),
                      filterSpectrum.data(), 0, nullptr, nullptr);

  std::vector<float> yGPU(OUT_LEN, 0.0f);
  std::vector<float> tempBuffer(FFT_SIZE);

  double kernel_ns = 0.0;
  const unsigned BLKS_ALL = BLKS_IN + P - 1;  

  for (unsigned blk = 0; blk < BLKS_ALL; blk++) {
    std::memset(tempBuffer.data(), 0, FFT_SIZE * sizeof(float));
    if (blk < BLKS_IN)
      std::memcpy(tempBuffer.data(), x.data() + blk * B, B * sizeof(float));
    
    clEnqueueWriteBuffer(q, dInput, CL_TRUE, 0, FFT_SIZE * sizeof(float), 
                        tempBuffer.data(), 0, nullptr, nullptr);
    clfftEnqueueTransform(planForward, CLFFT_FORWARD, 1, &q, 0, nullptr, nullptr,
                         &dInput, &dInputSpec, nullptr);
    
    // Shift frequencyb domain delay line
    for (int i = P-1; i > 0; i--)
      clEnqueueCopyBuffer(q, dFDL[i-1], dFDL[i], 0, 0, SPEC_SIZE * sizeof(cl_float2),
                         0, nullptr, nullptr);
    
    // Store new input spectrum in FDL[0]
    clEnqueueCopyBuffer(q, dInputSpec, dFDL[0], 0, 0, SPEC_SIZE * sizeof(cl_float2),
                       0, nullptr, nullptr);
    
    cl_float2 zeros = {0.0f, 0.0f};
    clEnqueueFillBuffer(q, dOutputSpec, &zeros, sizeof(cl_float2), 0, 
                       SPEC_SIZE * sizeof(cl_float2), 0, nullptr, nullptr);
    
    const unsigned maxPart = (blk < P) ? blk : P - 1;
    for (unsigned part = 0; part <= maxPart; part++) {
      cl_uint filterOffset = part * SPEC_SIZE;
      
      clSetKernelArg(kMult, 0, sizeof(cl_mem), &dFDL[part]);
      clSetKernelArg(kMult, 1, sizeof(cl_mem), &dFilterSpec);
      clSetKernelArg(kMult, 2, sizeof(cl_mem), &dPartSpec);
      clSetKernelArg(kMult, 3, sizeof(cl_uint), &filterOffset);
      
      size_t specSize = SPEC_SIZE;
      cl_event ev;
      clEnqueueNDRangeKernel(q, kMult, 1, nullptr, &specSize, nullptr, 0, nullptr, &ev);
      clWaitForEvents(1, &ev);
      cl_ulong ts, te;
      clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(ts), &ts, nullptr);
      clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(te), &te, nullptr);
      kernel_ns += double(te - ts);
      clReleaseEvent(ev);
      
      clSetKernelArg(kAccum, 0, sizeof(cl_mem), &dOutputSpec);
      clSetKernelArg(kAccum, 1, sizeof(cl_mem), &dPartSpec);
      
      cl_event ev2;
      clEnqueueNDRangeKernel(q, kAccum, 1, nullptr, &specSize, nullptr, 0, nullptr, &ev2);
      clWaitForEvents(1, &ev2);
      clReleaseEvent(ev2);
    }
    
    clfftEnqueueTransform(planInverse, CLFFT_BACKWARD, 1, &q, 0, nullptr, nullptr,
                         &dOutputSpec, &dOutput, nullptr);
    
    clEnqueueReadBuffer(q, dOutput, CL_TRUE, 0, FFT_SIZE * sizeof(float),
                       tempBuffer.data(), 0, nullptr, nullptr);

if (blk > 0) {                         
        for (unsigned i = 0; i < B; ++i)
            yGPU[(blk - 1) * B + i] += tempBuffer[i];
    }


    const unsigned remaining = OUT_LEN - blk * B;
    const unsigned copyLen = (remaining < B) ? remaining : B;
    std::memcpy(yGPU.data() + blk * B, tempBuffer.data() + (FFT_SIZE - B), 
               copyLen * sizeof(float));

  }

  std::cout << "GPU kernel time: " << kernel_ns * 1e-6 << " ms  ("
            << kernel_ns * 1e-3 / BLKS_IN << " µs/block)\n";

  dump("gpu.bin", yGPU);

  clfftDestroyPlan(&planForward);
  clfftDestroyPlan(&planInverse);
  clfftTeardown();
  
  for (unsigned i = 0; i < P; i++)
    clReleaseMemObject(dFDL[i]);
  
  clReleaseMemObject(dInput);
  clReleaseMemObject(dInputSpec);
  clReleaseMemObject(dFilterSpec);
  clReleaseMemObject(dOutputSpec);
  clReleaseMemObject(dPartSpec);
  clReleaseMemObject(dOutput);
  
  clReleaseKernel(kMult);
  clReleaseKernel(kAccum);
  clReleaseProgram(pr);
  clReleaseCommandQueue(q);
  clReleaseContext(ctx);

  std::cout << "done – binaries written.  Run python3 check_conv.py\n";
  return 0;
}
