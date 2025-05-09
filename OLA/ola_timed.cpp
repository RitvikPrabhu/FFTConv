#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <CL/cl.h>
#include <clFFT.h>
#include <math.h>
#include <chrono>

int main(int argc, char** argv) {
    if (argc != 2) {
        printf("Usage: %s <batch_size>\n", argv[0]);
        return 1;
    }

    size_t batchSize = atoi(argv[1]);
    if (batchSize == 0) {
        printf("Error: batch_size must be positive.\n");
        return 1;
    }

    cl_int err;
    cl_platform_id platform = 0;
    cl_device_id device = 0;
    cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, 0, 0 };
    cl_context ctx = 0;
    cl_command_queue queue = 0;

    cl_mem bufX, bufH, bufY;
    float *X, *H, *Y;

    size_t N_fft = 8192;
    size_t L_filter = 933;
    size_t L_block = N_fft - L_filter + 1;

    size_t fft_bytes = N_fft * 2 * sizeof(float);
    size_t batch_bytes = batchSize * fft_bytes;

    err = clGetPlatformIDs(1, &platform, NULL);
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    char device_name[256];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    printf("Running clFFT convolution on device: %s\n", device_name);
    printf("Batch size: %zu | FFT size: %zu | Block length: %zu | Filter length: %zu\n",
           batchSize, N_fft, L_block, L_filter);

    props[1] = (cl_context_properties)platform;
    ctx = clCreateContext(props, 1, &device, NULL, NULL, &err);

    const cl_queue_properties queue_props[] = {
        CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE,
        0
    };
    queue = clCreateCommandQueueWithProperties(ctx, device, queue_props, &err);

    clfftSetupData fftSetup;
    clfftInitSetupData(&fftSetup);
    clfftSetup(&fftSetup);

    X = (float *)calloc(batchSize * N_fft * 2, sizeof(float));
    H = (float *)calloc(N_fft * 2, sizeof(float));
    
    /*
    for (size_t b = 0; b < batchSize; b++) {
        for (size_t i = 0; i < L_block; i++) {
            size_t idx = b * N_fft * 2 + 2 * i;
            X[idx] = 1.0f;
            X[idx + 1] = 0.0f;
        }
    }

  
    for (size_t i = 0; i < L_filter; i++) {
        H[2*i] = 1.0f / L_filter;
        H[2*i + 1] = 0.0f;
    }
    */
    
    for (size_t b = 0; b < batchSize; b++) {
      for (size_t i = 0; i < L_block; i++) {
          float t = static_cast<float>(i) / 8000.0f;
          float val = sinf(2.0f * M_PI * 100.0f * t);
          size_t idx = b * N_fft * 2 + 2 * i;
          X[idx] = val;
          X[idx + 1] = 0.0f;
      }
    }

    // Custom filter: 466 +1s followed by 467 -1s
  
    for (size_t i = 0; i < L_filter; i++) {
        float val = (i < 466) ? 1.0f : -1.0f;
        H[2*i] = val;
        H[2*i + 1] = 0.0f;
    }
    
    Y = (float *)calloc(batchSize * N_fft * 2, sizeof(float));

    bufX = clCreateBuffer(ctx, CL_MEM_READ_WRITE, batch_bytes, NULL, &err);
    bufH = clCreateBuffer(ctx, CL_MEM_READ_WRITE, fft_bytes, NULL, &err);
    bufY = clCreateBuffer(ctx, CL_MEM_READ_WRITE, batch_bytes, NULL, &err);

    clEnqueueWriteBuffer(queue, bufX, CL_TRUE, 0, batch_bytes, X, 0, NULL, NULL);
    clEnqueueWriteBuffer(queue, bufH, CL_TRUE, 0, fft_bytes, H, 0, NULL, NULL);

    clfftPlanHandle plan;
    clfftCreateDefaultPlan(&plan, ctx, CLFFT_1D, &N_fft);
    clfftSetPlanPrecision(plan, CLFFT_SINGLE);
    clfftSetLayout(plan, CLFFT_COMPLEX_INTERLEAVED, CLFFT_COMPLEX_INTERLEAVED);
    clfftSetResultLocation(plan, CLFFT_INPLACE);

    cl_event event;
    cl_ulong start, end;

    clfftSetPlanBatchSize(plan, batchSize);
    clfftBakePlan(plan, 1, &queue, NULL, NULL);
    clfftEnqueueTransform(plan, CLFFT_FORWARD, 1, &queue, 0, NULL, &event, &bufX, NULL, NULL);
    clWaitForEvents(1, &event);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
    double t_fwd = (end - start) * 1e-6;

    clfftSetPlanBatchSize(plan, 1);
    clfftBakePlan(plan, 1, &queue, NULL, NULL);
    clfftEnqueueTransform(plan, CLFFT_FORWARD, 1, &queue, 0, NULL, &event, &bufH, NULL, NULL);
    clWaitForEvents(1, &event);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
    double t_filt = (end - start) * 1e-6;

    const char* src = R"CLC(
        __kernel void complex_multiply(
            __global float2* x,
            __global const float2* h,
            __global float2* y,
            int N)
        {
            int gid = get_global_id(0);
            int batch = get_global_id(1);

            int idx = batch * N + gid;
            float2 a = x[idx];
            float2 b = h[gid];

            y[idx].x = a.x * b.x - a.y * b.y;
            y[idx].y = a.x * b.y + a.y * b.x;
        }
    )CLC";

    cl_program program = clCreateProgramWithSource(ctx, 1, &src, NULL, &err);
    clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
    cl_kernel kernel = clCreateKernel(program, "complex_multiply", &err);

    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufX);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufH);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufY);
    clSetKernelArg(kernel, 3, sizeof(int), &N_fft);

    size_t global[2] = { N_fft, batchSize };
    clEnqueueNDRangeKernel(queue, kernel, 2, NULL, global, NULL, 0, NULL, &event);
    clWaitForEvents(1, &event);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
    double t_mult = (end - start) * 1e-6;

    clfftSetPlanBatchSize(plan, batchSize);
    clfftBakePlan(plan, 1, &queue, NULL, NULL);
    clfftEnqueueTransform(plan, CLFFT_BACKWARD, 1, &queue, 0, NULL, &event, &bufY, NULL, NULL);
    clWaitForEvents(1, &event);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
    double t_inv = (end - start) * 1e-6;

    clEnqueueReadBuffer(queue, bufY, CL_TRUE, 0, batch_bytes, Y, 0, NULL, NULL);

    size_t final_len = batchSize * L_block + L_filter - 1;
    float* final_output = (float*)calloc(final_len, sizeof(float));

    auto t_add_start = std::chrono::high_resolution_clock::now();
    for (size_t b = 0; b < batchSize; b++) {
        size_t base = b * N_fft * 2;
        size_t offset = b * L_block;
        for (size_t i = 0; i < N_fft; i++) {
            final_output[offset + i] += Y[base + 2 * i];
        }
    }
    auto t_add_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> t_add = t_add_end - t_add_start;

    printf("\n--- Timing Report ---\n");
    printf("Forward FFT (batched): %.3f ms\n", t_fwd);
    printf("Filter FFT:            %.3f ms\n", t_filt);
    printf("Multiply Kernel:       %.3f ms\n", t_mult);
    printf("Inverse FFT (batched): %.3f ms\n", t_inv);
    printf("Overlap-Add (host):    %.3f ms\n", t_add.count());

    FILE* fout = fopen("output_signal.txt", "w");
    if (fout == NULL) {
        perror("Failed to open output file");
        return 1;
    }
    
    for (size_t i = 0; i < final_len; i++) {
        fprintf(fout, "%f\n", final_output[i]);
    }

    fclose(fout);
    printf("Output signal written to output_signal.txt\n");
    clReleaseMemObject(bufX);
    clReleaseMemObject(bufH);
    clReleaseMemObject(bufY);
    clfftDestroyPlan(&plan);
    clfftTeardown();
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(ctx);
    free(X); free(Y); free(H); free(final_output);

    return 0;
}