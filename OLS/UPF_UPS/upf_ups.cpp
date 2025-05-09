#define CL_TARGET_OPENCL_VERSION 120
#include <CL/cl.h>

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
constexpr unsigned P = FIR_LEN / L;
constexpr unsigned B = L;

// static_assert(FIR_LEN % L == 0);
// static_assert(N_SAMPLES  % B == 0);

constexpr unsigned BLKS_IN = N_SAMPLES / B;
constexpr unsigned OUT_LEN = N_SAMPLES + FIR_LEN - 1;
constexpr unsigned RING_SAMPLES = (P + 1) * L - 1;

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
__kernel void upfUps(__global const float* ring,
                     __global const float* Hrev,
                     __global float* out,
                     const uint Ltap,
                     const uint partIdx,
                     const uint base)
{
    const uint gid = get_global_id(0);
    float acc = 0.0f;
    #pragma unroll
    for (uint k = 0; k < Ltap; k++)
        acc += ring[base + gid + k] * Hrev[partIdx*Ltap + k];
    out[gid] = acc;
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

  std::vector<float> hrev(FIR_LEN);
  for (unsigned p = 0; p < P; p++)
    for (unsigned k = 0; k < L; k++)
      hrev[p * L + k] = h[p * L + (L - 1 - k)];

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
  cl_kernel kn = clCreateKernel(pr, "upfUps", &err);

  const size_t ringBytes = RING_SAMPLES * sizeof(float);
  cl_mem dRing =
      clCreateBuffer(ctx, CL_MEM_READ_ONLY, ringBytes, nullptr, &err);
  cl_mem dH = clCreateBuffer(ctx, CL_MEM_READ_ONLY, FIR_LEN * sizeof(float),
                             nullptr, &err);
  cl_mem dY =
      clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, B * sizeof(float), nullptr, &err);

  clEnqueueWriteBuffer(q, dH, CL_TRUE, 0, FIR_LEN * sizeof(float), hrev.data(),
                       0, nullptr, nullptr);

  const cl_uint Ltap = L;
  clSetKernelArg(kn, 3, sizeof(Ltap), &Ltap);

  std::vector<float> ring(RING_SAMPLES, 0.0f);
  std::vector<float> yGPU(OUT_LEN, 0.0f);
  std::vector<float> blockOut(B);
  std::vector<float> partBuf(B);

  const size_t gsz = B;

  double kernel_ns = 0.0;
  const unsigned BLKS_ALL = BLKS_IN + P;

  for (unsigned blk = 0; blk < BLKS_ALL; blk++) {

    std::memmove(ring.data(), ring.data() + L,
                 (RING_SAMPLES - L) * sizeof(float));

    if (blk < BLKS_IN)
      std::memcpy(ring.data() + (RING_SAMPLES - L), x.data() + blk * B,
                  B * sizeof(float));
    else
      std::memset(ring.data() + (RING_SAMPLES - L), 0, B * sizeof(float));

    clEnqueueWriteBuffer(q, dRing, CL_TRUE, 0, ringBytes, ring.data(), 0,
                         nullptr, nullptr);

    std::fill(blockOut.begin(), blockOut.end(), 0.0f);

    const unsigned maxPart = (blk < P) ? blk : P - 1;
    for (unsigned part = 0; part <= maxPart; part++) {
      const cl_uint partIdx = part;
      const cl_uint base = (P - 1 - part) * L;

      clSetKernelArg(kn, 0, sizeof(cl_mem), &dRing);
      clSetKernelArg(kn, 1, sizeof(cl_mem), &dH);
      clSetKernelArg(kn, 2, sizeof(cl_mem), &dY);
      clSetKernelArg(kn, 4, sizeof(partIdx), &partIdx);
      clSetKernelArg(kn, 5, sizeof(base), &base);

      cl_event ev;
      clEnqueueNDRangeKernel(q, kn, 1, nullptr, &gsz, nullptr, 0, nullptr, &ev);
      clWaitForEvents(1, &ev);
      cl_ulong ts, te;
      clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(ts), &ts,
                              nullptr);
      clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(te), &te,
                              nullptr);
      kernel_ns += double(te - ts);
      clReleaseEvent(ev);

      clEnqueueReadBuffer(q, dY, CL_TRUE, 0, B * sizeof(float), partBuf.data(),
                          0, nullptr, nullptr);
      for (unsigned i = 0; i < B; i++)
        blockOut[i] += partBuf[i];
    }

    const unsigned remaining = OUT_LEN - blk * B;
    const unsigned copyLen = (remaining < B) ? remaining : B;
    std::memcpy(yGPU.data() + blk * B, blockOut.data(),
                copyLen * sizeof(float));
  }

  std::cout << "GPU kernel time: " << kernel_ns * 1e-6 << " ms  ("
            << kernel_ns * 1e-3 / BLKS_IN << " µs/block)\n";

  dump("gpu.bin", yGPU);

  clReleaseMemObject(dRing);
  clReleaseMemObject(dH);
  clReleaseMemObject(dY);
  clReleaseKernel(kn);
  clReleaseProgram(pr);
  clReleaseCommandQueue(q);
  clReleaseContext(ctx);

  std::cout << "done – binaries written.  Run python3 check_conv.py\n";
  return 0;
}
