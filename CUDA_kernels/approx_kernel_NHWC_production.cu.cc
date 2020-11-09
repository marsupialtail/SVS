#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include <math.h>
#include <stdio.h>
#include <cuda_profiler_api.h>

#define FACTOR 16
#define B_FACTOR 8 // this is how many blocks per dy channel.
#define BATCH_SIZE 128

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


void checkGpuMem()
{

  float free_m,total_m,used_m;
  size_t free_t,total_t;
  cudaMemGetInfo(&free_t,&total_t);
  free_m =(uint)free_t/1048576.0 ;
  total_m=(uint)total_t/1048576.0;
  used_m=total_m-free_m;
  printf ( "  mem free %d .... %f MB mem total %d....%f MB mem used %f MB\n",free_t,free_m,total_t,total_m,used_m);

}

__global__ void TonyConvKernelDraft(const float* __restrict__ input, const float* __restrict__ dy, float* output,
        const int filter_x_, const int filter_y_, const int stride, const int is_1, const int is_2, const int input_channels, const int ys_1,
        const int ys_2, const int ys_3) {

  /*

  Here is the plan. Each block will execute a different channel. Inside each block, every threads will
  more or less execute the find max and sum for each batch element. This is to make the reduce operation easier.

  */

  // first read the input and the corresponding dy into shared memory
  // shared memory doesn't work for now. Too small. 
    int t1, t2;

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

 t1 = clock();

  int row,col,idx,idx_1,idx_2;
  float sum = 0.0;
  float max = 0.0;

  int thread_group = threadIdx.x / input_channels;
  int write_offset = threadIdx.x % input_channels;
  int element =(blockIdx.x % B_FACTOR) * FACTOR + thread_group ;
  int read_offset = (element * ys_1 + my_channel) * ys_2 * ys_3;
  for(int i = read_offset; i < read_offset + ys_2*ys_3; i++)
  {
    if(fabs(dy[i]) > max)
    {
      max = fabs(dy[i]);
      idx = i-read_offset;
    }
    sum = sum + dy[i];
  }
   row = idx / ys_3 * stride;
   col = idx % ys_3 * stride;

t2 = clock();
  t1 = clock();
   if(blockIdx.x == 0)
  {
    printf("some garbarge");
  }
   idx = ( element * is_1 * is_2 + row * is_2 + col ) * input_channels;

   #pragma unroll

   for(row = thread_group%3; row < thread_group%3 + filter_x_; row++)
   {
    for(col = 0; col < filter_y_;col++) 
    {
	idx_1 = idx + (is_2 * (row % 3) + col) * input_channels + write_offset;
	idx_2 = (filter_y_ * (row % 3) + col) * input_channels + write_offset;
	//            if(blockIdx.x==0 && element == 0){
	//                printf("(%d,%d, %d) ",threadIdx.x,idx_1,idx_2);
	//               }
	atomicAdd(&data[idx_2], input[idx_1]*sum);
	//data[idx_2] += input[idx_1] * sum;
     }

   }      

  __syncthreads();
  t2 = clock();
  t2 = t2 - t1;

  t1 = clock();


  //shitty access pattern but actually really fast

  int low = threadIdx.x * input_channels * filter_x_ * filter_y_ / blockDim.x;
  int high = (threadIdx.x+1) * input_channels * filter_x_ * filter_y_ / blockDim.x;
  for(int j= low;j<high;j++){
    int output_index = output_block_offset + j;
    atomicAdd(&output[output_index],data[j]/BATCH_SIZE);
    //printf("%.f ",output[output_index]);
  }
  t2 = clock();

}

void TonyConvGradKernelLauncher(const float * input, const int input_size_[], const float * dy, const int dy_size_[], float * output,
    int filter_x_,int filter_y_,int stride) {

  int data_size = (filter_x_*filter_y_*input_size_[3] + input_size_[0]/B_FACTOR * 3)*sizeof(float) ;

  assert(input_size_[0] % input_size_[3] == 0);

  cudaProfilerStart();

  TonyConvKernelDraft<<<dy_size_[1] * B_FACTOR,input_size_[3]*FACTOR, data_size>>>(input,dy,output,filter_x_,filter_y_,stride,
  input_size_[1],input_size_[2],input_size_[3],dy_size_[1],dy_size_[2],dy_size_[3]);

  cudaProfilerStop();
}

#endif
