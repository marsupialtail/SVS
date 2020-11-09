#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include <math.h>
#include <stdio.h>
#include <cuda_profiler_api.h>

// B_FACTOR * FACTOR must equal BATCH_SIZE
#define FACTOR 16
#define B_FACTOR 8
#define BATCH_SIZE 128
#define K 1
#define OFF 3


__global__ void TonyConvKernelDraft(float* output, const int a, const int b, const int c, const int d) {

    int boundary = a * b * c * d;
    int tid = blockIdx.x * 1024 + threadIdx.x;
    if(tid < boundary)
    {
        output[tid] = 0.0;
    }



}

void TonyConvGradKernelLauncher(const float * input, const int input_size_[], const float * dy, const int dy_size_[], float * output,
    int filter_x_,int filter_y_,int stride) {

  int num_blocks = filter_x_ * filter_y_ * input_size_[3] * dy_size_[1] / 1024 + 1;

  cudaEvent_t start,stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaProfilerStart();
  cudaEventRecord(start);

 TonyConvKernelDraft<<<num_blocks,1024>>>(output,filter_x_,filter_y_,input_size_[3],dy_size_[1]);

  cudaEventRecord(stop);
  cudaProfilerStop();
  cudaEventSynchronize(stop);
  float time;
  cudaEventElapsedTime(&time,start,stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
//  std::cout << "kernel used " << time << std::endl;
}

#endif
