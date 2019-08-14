#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include <math.h>
#include <stdio.h>
#include <cuda_profiler_api.h>

// B_FACTOR * FACTOR must equal BATCH_SIZE

#define BATCH_SIZE 64
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define filter_x_ 5
#define filter_y_ 5
#define stride 1
#define is_2 12
#define is_3 12
#define OC 64
#define IC 64
#define DATA_DIVIDE is_3 * is_2 * BATCH_SIZE
#define DATA_DIVIDE2 DATA_DIVIDE + filter_x_ * filter_y_ * OC
#define DATA_SIZE DATA_DIVIDE2 + OC
#define BLOCK_SIZE 1024
#define TPG (BLOCK_SIZE / OC)
#define PAD 2

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__inline__ __device__ float warpReduceSum(float sum, int width)
{
    if(width > 16)
    sum += __shfl_xor_sync(0xffffffff,sum,16);
    if(width > 8)
    sum += __shfl_xor_sync(0xffffffff,sum,8);
    sum += __shfl_xor_sync(0xffffffff,sum,4);
    sum += __shfl_xor_sync(0xffffffff,sum,2);
    sum += __shfl_xor_sync(0xffffffff,sum,1);
    return sum;
}
__inline__ __device__ float warpReduceMax(float maxv, int width)
{
    if(width > 16)
    maxv = max(__shfl_xor_sync(0xffffffff,maxv,16),maxv);
    if(width > 8)
    maxv = max(__shfl_xor_sync(0xffffffff,maxv,8),maxv);
    maxv = max(__shfl_xor_sync(0xffffffff,maxv,4),maxv);
    maxv = max(__shfl_xor_sync(0xffffffff,maxv,2),maxv);
    maxv = max(__shfl_xor_sync(0xffffffff,maxv,1),maxv);
    return maxv;
}

__global__ void TonyConvKernelDraft(const float* __restrict__ input, const float * __restrict__ sums,
 const int * __restrict__ indices, float* output) {

  /*

  The plan is now this. Every blOCk will prOCess an input channel and store the NHW
  in shared memory. First draft, redundant work will be done figuring out the max idx


  */
  __shared__ float data[DATA_SIZE];
  //float * filter = data + DATA_DIVIDE;
  //int * indices = (int *)(filter + filter_x_ * filter_y_ * OC);
  int input_read_offset = blockIdx.x * DATA_DIVIDE;
  for(int i = threadIdx.x; i < DATA_DIVIDE; i+= BLOCK_SIZE)
    data[i] = input[input_read_offset+i];
  for(int i = threadIdx.x; i < DATA_SIZE - DATA_DIVIDE; i+= BLOCK_SIZE)
    data[DATA_DIVIDE + i] = 0.0;

  // do we need it?
  __syncthreads();

  int my_channel = threadIdx.x / TPG; // this hits 0-BATCH_SIZE
  int my_lane = threadIdx.x % TPG; // this could be smaller than 32
  int filter_write_offset = my_channel * filter_y_ * filter_x_;

  #pragma unroll 2
  for(int element = 0; element < BATCH_SIZE; element ++)
  {

    int read_offset = element * OC + my_channel;
    int idx = indices[read_offset];
    float sum = sums[read_offset];
    int max_row = idx / ys_3;
    int max_col = idx % ys_3;
    int shared_read_offset = element * is_2 * is_3;

    #pragma unroll
    for(int i = my_lane; i < filter_x_ * filter_y_; i+=TPG)
    {
        int row = i / filter_y_;
        int col = i % filter_y_;
        int input_row = row + max_row - PAD;
        int input_col = col + max_col - PAD;
        if(input_row > -1 && input_row < is_2 && input_col > -1 && input_col < is_3){
            int shared_read_idx = shared_read_offset + input_row * is_3 + input_col;
            int filter_write_idx = filter_write_offset + row * filter_y_ + col;
            data[DATA_DIVIDE + filter_write_idx] += data[shared_read_idx] * sum / BATCH_SIZE;
        }
    }

  }

  __syncthreads();
  int output_write_offset = blockIdx.x * filter_y_ * filter_x_;
  for(int i = threadIdx.x; i < OC * filter_x_ * filter_y_; i += BLOCK_SIZE)
  {
    output[output_write_offset + i] = data[DATA_DIVIDE + i];
  }

   
}

void TonyConvGradKernelLauncher(const float * input,  const float * sums,
const int * indices, float * output) {

//  assert(input_size_[0] % input_size_[3] == 0);
#if TEST
  cudaEvent_t start,stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
#endif
//  cudaProfilerStart();
std::cout << DATA_DIVIDE << std::endl;
  TonyConvKernelDraft<<<IC,BLOCK_SIZE>>>(input,sums,indices,output);

//  cudaProfilerStop();
//  std::cout << "finished" << std::endl;

#if TEST
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float time;
  cudaEventElapsedTime(&time,start,stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  std::cout << "kernel used " << time << std::endl;
#endif
}

#endif
