#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include <math.h>
#include <stdio.h>
#include <cuda_profiler_api.h>

// B_FACTOR * FACTOR must equal BATCH_SIZE
#define TPC 32

__global__ void ScatterKernel(const float * __restrict__ values, const int * __restrict__ indices,
 const int nnz, const int c, const int hw, float * output)
{
    extern __shared__ float data[];

    for(int i = threadIdx.x; i < hw; i+= nnz)
    {
        data[i] = 0.0;
    }
    __syncthreads();

    int my_element = blockIdx.x / c;
    int my_channel = blockIdx.x % c;

    int my_start = my_element * c * nnz;
    int my_read_idx = my_start + threadIdx.x;
    float val = values[my_read_idx];
    int idx = indices[my_read_idx];
    data[idx] = val;

    int my_output_start = my_element * c * hw + my_channel * hw;
    for(int i = threadIdx.x; i < hw; i += nnz)
    {
        output[my_output_start + i] = data[i];
    }

}

void ScatterKernelLauncher(const float * values, const int * indices, int n, int nnz,
int c, int hw, float * output) {


  int shared_size = hw * 4;

/*  cudaEvent_t start,stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaProfilerStart();
  cudaEventRecord(start); */

  ScatterKernel<<<n * c,nnz,shared_size>>>(values,indices,nnz,c,hw,output);

/*  cudaEventRecord(stop);
  cudaProfilerStop();
  cudaEventSynchronize(stop);
  float time;
  cudaEventElapsedTime(&time,start,stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  std::cout << "kernel used " << time << std::endl;
*/
  }

#endif
