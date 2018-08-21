#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include <math.h>
#include <stdio.h>

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

__global__ void TonyConvKernelDraft(const float* input, const float* dy, float* output,
        const int filter_x_, const int filter_y_, const int stride, const int is_0, const int is_1, const int is_2, const int is_3,
         const int ys_0, const int ys_1, const int ys_2, const int ys_3) {

  /*

  Here is the plan. Each block will execute a different channel. Inside each block, every threads will
  more or less execute the find max and sum for each batch element. This is to make the reduce operation easier.

  */

  // first read the input and the corresponding dy into shared memory
  // shared memory doesn't work for now. Too small. 
  extern __shared__ float data[] ;

  // both dy_size and input_size should be in NCHW format

  for (int i = threadIdx.x; i < filter_x_*filter_y_*is_1; i += blockDim.x)
      data[i] = 0;
  __syncthreads();

  int i = threadIdx.x * (ys_1 * ys_2 * ys_3) + blockIdx.x * (ys_2 * ys_3);
  int max_idx =0;
  float sum = 0.0;
  float max = 0.0;

  int t1, t2;
  t1 = clock();
  float val;

  for(int j=0;j<ys_2 * ys_3;j++){
    //printf("dy %.2f\n",dy[i+j]);
    if(fabs(dy[i+j]) > max)
    {
      max = fabs(dy[i+j]);
      max_idx = j;
    }

//    val = fabs(dy[i+j]);
//    max = (max > val) * max + (!(max > val)) * val;
//    max_idx = j * (max == val) + max_idx * (max > val);

    sum = sum + dy[i+j];

  }

  t2 = clock();
  if(blockIdx.x == 0){
    printf("%d ",t2-t1);
  }

  //printf("threadIdx %d %.2f", threadIdx.x, sum);

  int max_row = max_idx / ys_3 * stride;
  int max_col = max_idx % ys_3 * stride;

  // load stuff into registers that will be used over and over again. Try to think about memory IO here
  int input_channels = is_1;

  // data has dimensions input_channels by filter_x by filter_y
  int input_thread_offset = threadIdx.x * input_channels * is_2 * is_3;

  // copy the corresponding patch of input into shared memory.
  // note there is no mention of blockIdx.x since input doesn't know about output channels.

  t1 = clock();

  for(int c=threadIdx.x;c<input_channels+threadIdx.x;c++){
    int data_channel_offset = (c % input_channels) * filter_x_ * filter_y_;
    int input_channel_offset = (c % input_channels) * is_2 * is_3;

    for(int j=0;j<filter_x_;j++){
      int data_row_offset = j * filter_y_;
      int input_row_offset = (j+max_row) * is_3;

      for(int k=0;k<filter_y_;k++)
      {
        int data_idx = data_channel_offset + data_row_offset + k;
        int input_idx = input_thread_offset + input_channel_offset + input_row_offset + k+max_col;

        // introducing this race condition saves 0.3 seconds
        //atomicAdd(&data[data_idx], input[input_idx] * sum);
        data[data_idx] += input[input_idx] * sum;
       //printf("data %.f %.f %d %d %d %d ",data[data_idx], sum, input_idx, c, j, k);
      }
    }
  }
  // now do the parallel reduction, this can definitely be made better
  __syncthreads();
  t2 = clock();
  if(blockIdx.x==0){
    printf("%d ",t2-t1);
  }

  int block_offset = blockIdx.x * input_channels * filter_x_ * filter_y_;
  // this is really shitty memory access pattern but that's why we moved it to shared memory
  // the idea is now that we have a N * C_i * filter_x * filter_y thing in shared memory
  // we want to reduce it across the N dimension.

  int low = threadIdx.x * input_channels * filter_x_ * filter_y_ / blockDim.x;
  int high = (threadIdx.x+1) * input_channels * filter_x_ * filter_y_ / blockDim.x;
  for(int j= low;j<high;j++){
    int output_index = block_offset + j;
    output[output_index] = data[j]/blockDim.x;
    //printf("%.f ",output[output_index]);
  }


}

void TonyConvGradKernelLauncher(const float * input, const int input_size_[], const float* dy, const int dy_size_[], float * output,
    int filter_x_,int filter_y_,int stride) {

  //std::cout << "dy size " << dy_size_[0] << " " <<  dy_size_[1] << " " << dy_size_[2] << " " << dy_size_[3] << std::endl;
  //std::cout << input_size_[0] << " " <<  input_size_[1] << " " << input_size_[2] << " " << input_size_[3] << std::endl;

  int data_size = filter_x_*filter_y_*input_size_[1]*sizeof(float);
  // float * data;
  //checkGpuMem();
  // std::cout << "allocating memory " << cudaMalloc((void**)&data, data_size * 2) << " " << data_size << std::endl;

  //std::cout << input_size_[0]*filter_x_*filter_y_*input_size_[1] << std::endl;

  TonyConvKernelDraft<<<dy_size_[1],input_size_[0], data_size>>>(input,dy,output,filter_x_,filter_y_,stride,
  input_size_[0],input_size_[1],input_size_[2],input_size_[3], dy_size_[0],dy_size_[1],dy_size_[2],dy_size_[3]);

  //std::cout << "kernel finished successfully" << std::endl;
  //float *d_debug = (float*) malloc(sizeof(float)*filter_x_*filter_y_*input_size[1]*dy_size[1]);
  //std::cout << "transferred to debug host variable" << std::endl;

  //std::cout << 0 << std::endl;
  //std::cout << cudaFree(data) << std::endl;
//  for(int i = 0;i<filter_x_*filter_y_*input_size[1]*dy_size[1];i++)
//  {
//    std::cout << d_debug[i];
//  }

  //cudaMemcpy(h_output,d_output,sizeof(float)*filter_x_*filter_y_*input_size[1]*dy_size[1],cudaMemcpyDeviceToHost);

  //cudaFree(d_input);
  //cudaFree(d_dy);
  //cudaFree(d_output);
}

#endif
