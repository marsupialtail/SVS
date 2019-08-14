#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include <math.h>
#include <stdio.h>
#include <cuda_profiler_api.h>

// B_FACTOR * FACTOR must equal BATCH_SIZE
#define FACTOR 8
#define B_FACTOR 16
#define BATCH_SIZE 128
#define K 1
#define OFF 3

#define ys_2 12
#define ys_3 12
#define is_1 17
#define is_2 17
#define filter_x_ 5
#define filter_y_ 5
#define input_channels 64
#define ys_1 64 
#define stride 1
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

__global__ void TonyConvKernelDraft(const float* __restrict__ input, const float* __restrict__ dy, float* output) {

  /*

  Here is the plan. Each block group (B_FACTOR blocks in a group) will execute an output channel. Within the block group,
  each block will handle a subset of the elements in the batch. A thread group is defined as input_channel number of threads.
  They will handle one batch element. There are FACTOR number of thread groups per block. This is why FACTOR * B_FACTOR = BATCH_SIZE

  First, each block will calculate the relevant max and sum statistics. Communicated by warp shuffle instruction. A potential caveat is that

  Each thead group will read in a single patch of shape filter_x * filter_y * input_channel. Memory accsess will be coalesced across
  the third dimension.

  */


    #if PROFILE
    int t1, t2;
    #endif

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

  #if PROFILE
  t1 = clock();
  #endif

  int row,col,idx,idx_1,idx_2;
  float sum = 0.0;
  float maxv =0.0;

  int thread_group = threadIdx.x / input_channels;
  int write_offset = threadIdx.x % input_channels;
  int element =(blockIdx.x % B_FACTOR) * FACTOR + thread_group ;
  int read_offset = (element * ys_1 + my_channel) * ys_2 * ys_3;
  int width = min(32,input_channels);


  float prefetched = dy[read_offset + threadIdx.x%width];
  for(int i = read_offset + threadIdx.x%width; i < read_offset + ys_2*ys_3; i+=width)
  {
    float number = prefetched;
    prefetched = dy[min(i+width,read_offset + ys_2 * ys_3)];
    if(fabs(number) > maxv)
    {
      maxv = fabs(number);
      //sign = (dy[i] > 0.0) ? 1.0:-1.0;
      idx = i - read_offset;
    }
    sum = sum + number;
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
          data[data_divide + FACTOR  + thread_group] = sum;
	  //if(blockIdx.x==0 && element == 0){
	//	printf("(%.2f,%.2f) ",maxv,sum);
	  // }
          maxv = 0.0;
      }
  }

  __syncthreads();

 //   float scaling =0.0;

//    for(int i = 0;i<K;i++)
 //   {
//       scaling += data[data_divide + FACTOR * K + FACTOR * i + thread_group];
 //   }
//    scaling = sum / (scaling+0.01) ;

    for(int i = 0; i < FACTOR; i ++)
    {
	#pragma unroll
        for(int j = threadIdx.x; j < filter_y_ * filter_x_ * input_channels; j+= blockDim.x)
        {

            int input_idx = data_i[data_divide + i];
            float sum = data[data_divide + FACTOR + i];
            //int my_channel = j % input_channels;
            int my_row = j / (filter_y_ * input_channels);
            int my_col = (j % (filter_y_ * input_channels)) / input_channels;
            int input_row = input_idx / ys_3 + my_row;
            int input_col = input_idx % ys_3 + my_col;

            int read_pos = ((i+(blockIdx.x % B_FACTOR) * FACTOR)  * is_1 * is_2 + input_row * is_2 + input_col ) * input_channels + write_offset;
            int write_pos = (my_row * filter_y_ + my_col ) * input_channels + write_offset;

            data[write_pos] += input[read_pos] * sum;

        }
    }

  __syncthreads();

  #if PROFILE
  t2 = clock();
  if(blockIdx.x==0){
    printf("%d ",t2-t1);
  }

  t1 = clock();
  #endif

  //shitty access pattern but actually really fast

  int low = threadIdx.x * input_channels * filter_x_ * filter_y_ / blockDim.x;
  int high = (threadIdx.x+1) * input_channels * filter_x_ * filter_y_ / blockDim.x;
  for(int j= low;j<high;j++){
    int output_index = output_block_offset + j;
    atomicAdd(&output[output_index],data[j]/BATCH_SIZE);
    //atomicAdd(&output[output_index],data[j]);
    //printf("%.f ",output[output_index]);
  }

  #if PROFILE
  t2 = clock();
  if(blockIdx.x==0){
    printf("%d ",t2-t1);
  }
  #endif

}

void TonyConvGradKernelLauncher(const float * input, const int input_size_[], const float * dy, const int dy_size_[], float * output,
    int filter_x,int filter_y,int stride_) {

  int data_size = (filter_x_*filter_y_*input_size_[3]+FACTOR * K*2)*sizeof(float) ;

  assert(input_size_[0] % input_size_[3] == 0);
 
 #if TEST
  cudaEvent_t start,stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaProfilerStart();
  cudaEventRecord(start);
 #endif 
  TonyConvKernelDraft<<<dy_size_[1] * B_FACTOR,input_size_[3]*FACTOR, data_size>>>(input,dy,output);

 #if TEST
  cudaEventRecord(stop);
  cudaProfilerStop();
  cudaEventSynchronize(stop);
  float time;
  cudaEventElapsedTime(&time,start,stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  std::cout << "kernel used " << time << std::endl;
  #endif
}

#endif
