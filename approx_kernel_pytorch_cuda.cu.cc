#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

#include <math.h>
#include <stdio.h>
#include <cuda_profiler_api.h>

define FACTOR 16
#define B_FACTOR 8
#define BATCH_SIZE 128
#define K 1
#define OFF 3

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
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

__global__ void TonyConvKernelDraft(const float* __restrict__ input, const float* __restrict__ dy, float* output,
        const int filter_x_, const int filter_y_, const int stride, const int is_1, const int is_2, const int input_channels, const int ys_1,
        const int ys_2, const int ys_3) {

  /*

  Here is the plan. Each block group (B_FACTOR blocks in a group) will execute an output channel. Within the block group,
  each block will handle a subset of the elements in the batch. A thread group is defined as input_channel number of threads.
  They will handle one batch element. There are FACTOR number of thread groups per block. This is why FACTOR * B_FACTOR = BATCH_SIZE

  First, each block will calculate the relevant max and sum statistics. Communicated by warp shuffle instruction. A potential caveat is that

  Each thead group will read in a single patch of shape filter_x * filter_y * input_channel. Memory accsess will be coalesced across
  the third dimension.

  */

  extern __shared__ float data[] ;
  extern __shared__ int data_i[];

  int data_divide = filter_x_*filter_y_*input_channels;
  int my_channel = blockIdx.x / B_FACTOR;
  int output_block_offset = my_channel * input_channels * filter_x_ * filter_y_;

  for(int i = output_block_offset + threadIdx.x; i < output_block_offset + data_divide; i += blockDim.x)
      output[i] = 0;
  for (int i = threadIdx.x; i < data_divide; i += blockDim.x)
      data[i] = 0;

  __syncthreads();
  int row,col,idx,idx_1,idx_2;
  float sum = 0.0;
  float maxv =0.0;

  int thread_group = threadIdx.x / input_channels;
  int write_offset = threadIdx.x % input_channels;
  int element =(blockIdx.x % B_FACTOR) * FACTOR + thread_group ;
  int read_offset = (element * ys_1 + my_channel) * ys_2 * ys_3;
  int width = min(32,input_channels);

  float sign;

  float prefetched = dy[read_offset + threadIdx.x%width];
  for(int i = read_offset + threadIdx.x%width; i < read_offset + ys_2*ys_3; i+=width)
  {
    float number = prefetched;
    prefetched = dy[min(i+width,read_offset + ys_2 * ys_3)];
    if(fabs(number) > maxv)
    {
      maxv = fabs(dy[i]);
      //sign = (dy[i] > 0.0) ? 1.0:-1.0;
      idx = i - read_offset;
    }
    sum = sum + dy[i];
  }

  // note that this works for out_channels > 32. Why? Because in this case redundant work is done.
  // Each warp has a self-consistent view of the entire output processed for each thread_group.
  // so the output iamge is processed thread_group/warp_size times. This saves a lot of trouble.

  sum = warpReduceSum(sum,width);
  for(int i = 0;i< K;i++)
  {
      if (maxv == warpReduceMax(maxv,width))
      {
          data_i[data_divide + FACTOR*i + thread_group] = idx;
//          data[data_divide + FACTOR * K + FACTOR * i + thread_group] = maxv * sign;
	  //if(blockIdx.x==0 && element == 0){
	//	printf("(%.2f,%.2f) ",maxv,sum);
	  // }
          maxv = 0.0;
      }
  }

  __syncthreads();

    for(int i = 0;i<K;i++)
    {
       row = data_i[data_divide + FACTOR * i + thread_group]/ ys_3 * stride;
       col = data_i[data_divide + FACTOR* i + thread_group] % ys_3 * stride;
       maxv = data[data_divide + FACTOR * K + FACTOR * i + thread_group];

       //if(blockIdx.x==0 && element == 0){
	//printf("(%d,%d,%d,%.2f,%.2f,%.2f) ",row,col,i,scaling,maxv,sum);
       //}

       idx = ( element * is_1 * is_2 + row * is_2 + col ) * input_channels;

       #pragma unroll

       for(row = thread_group%OFF; row < thread_group%OFF + filter_x_; row++)
       {
        for(col = 0; col < filter_y_;col++)
        {
        idx_1 = idx + (is_2 * (row % OFF) + col) * input_channels + write_offset;
        idx_2 = (filter_y_ * (row % OFF) + col) * input_channels + write_offset;

        atomicAdd(&data[idx_2], input[idx_1]*sum);
         }

       }
      }

  __syncthreads();


  //shitty access pattern but actually really fast

  int low = threadIdx.x * input_channels * filter_x_ * filter_y_ / blockDim.x;
  int high = (threadIdx.x+1) * input_channels * filter_x_ * filter_y_ / blockDim.x;
  for(int j= low;j<high;j++){
    int output_index = output_block_offset + j;
    atomicAdd(&output[output_index],data[j]/BATCH_SIZE);
  }

}

torch::Tensor tony_conv_kernel(
    torch::Tensor input,
    torch::Tensor dy,
    torch::Tensor stride,
    torch::Tensor filter_x_,
    torch::Tensor filter_y_,
) {

    const int batch_size = input.size(0);
    const int is_1 = input.size(1);
    const int is_2 = input.size(2);
    const int input_channels = input.size(3);
    const int ys_1 = dy.size(1);
    const int ys_2 = dy.size(2);
    const int ys_3 = dy.size(3);

    const int filter_x = * (filter_x_.data<int>());
    const int filter_y = * (filter_y_.data<int>());

    auto output = torch::zeros({ys_1,filter_x,filter_y,input_channels});

    int data_size = (filter_x * filter_y * input_channels + FACTOR * 2) * sizeof(float);
    int blocks = ys_1 * B_FACTOR;
    int threads = input_channels * FACTOR;

    if(batch_size == 128){
        TonyConvKernelDraft<<<blocks,threads,data_size>>>(input.data_ptr(),dy.data_pr(),output.data_ptr(), filter_x,
        filter_y, stride, is_1,is_2, input_channels, ys_1, ys_2, ys_3);}

    return output;

}