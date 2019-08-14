#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include <math.h>
#include <stdio.h>
#include <cuda_profiler_api.h>
#include <curand.h>
// B_FACTOR * FACTOR must equal BATCH_SIZE
#define FACTOR 16
#define B_FACTOR 8
#define BATCH_SIZE 128
#define K 1
#define OFF 3

__global__ void init_stuff(curandState *state) {
 int idx = blockIdx.x * blockDim.x + threadIdx.x;
 curand_init(1337, idx, 0, &state[idx]);
}
__global__ void make_rand(curandState *state, float
*randArray) {
 int idx = blockIdx.x * blockDim.x + threadIdx.x;
 randArray[idx] = curand_normal(&state[idx])/128;
}
void host_function() {

}

void TonyConvGradKernelLauncher(const float * input, const int input_size_[], const float * dy, const int dy_size_[], float * output,
    int filter_x_,int filter_y_,int stride) {

  int num_blocks = filter_x_ * filter_y_ * input_size_[3] * dy_size_[1] / 1024 + 1;

  cudaEvent_t start,stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaProfilerStart();


 curandState *d_state;
 cudaMalloc(&d_state, 1024 * num_blocks);
  cudaEventRecord(start);
 init_stuff<<<nblocks, nthreads>>>(d_state);
 make_rand<<<nblocks, nthreads>>>(d_state, output);
 cudaFree(d_state);

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
