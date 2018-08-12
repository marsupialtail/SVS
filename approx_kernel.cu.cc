#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include <math.h>
#include <stdio.h>

__constant__ int input_size[4];
__constant__ int dy_size[4];

__global__ void TonyConvKernelDraft(const float* input, const float* dy, float* data, float* output,
        const int filter_x_, const int filter_y_, const int stride) {

  /*

  Here is the plan. Each block will execute a different channel. Inside each block, every threads will
  more or less execute the find max and sum for each batch element. This is to make the reduce operation easier.

  */

  // first read the input and the corresponding dy into shared memory
  // shared memory doesn't work for now. Too small. 
  // extern __shared__ float data[];

  // both dy_size and input_size should be in NCHW format

  int i = threadIdx.x * (dy_size[1] * dy_size[2] * dy_size[3]) + blockIdx.x * (dy_size[2] * dy_size[3]);
  int max_idx;
  float sum = 0.0;
  float max = 0.0;

  for(int j=0;j<dy_size[2]*dy_size[3];j++){
    //printf("dy %.2f\n",dy[i+j]);
    if(fabs(dy[i+j]) > max)
    {
      max = fabs(dy[i+j]);
      max_idx = j;
    }
    sum = sum + dy[i+j];

  }

  //printf("threadIdx %d %.2f", threadIdx.x, sum);

  int max_row = max_idx / dy_size[3] * stride;
  int max_col = max_idx % dy_size[3] * stride;

  // load stuff into registers that will be used over and over again. Try to think about memory IO here
  int input_height = input_size[2];
  int input_width = input_size[3];
  int input_channels = input_size[1];

  int data_block_offset = blockIdx.x * input_size[0]*filter_x_*filter_y_*input_size[1];
  int data_thread_offset = threadIdx.x * input_channels * filter_x_ * filter_y_;
  int input_thread_offset = threadIdx.x * input_channels * input_height * input_width;

  // copy the corresponding patch of input into shared memory.
  // note there is no mention of blockIdx.x since input doesn't know about output channels.

  for(int c=0;c<input_channels;c++){
    int data_channel_offset = c * filter_x_ * filter_y_;
    int input_channel_offset = c * input_height * input_width;

    for(int j=0;j<filter_x_;j++){
      int data_row_offset = j * filter_y_;
      int input_row_offset = (j+max_row) * input_width;

      for(int k=0;k<filter_y_;k++)
      {
        int data_idx = data_block_offset + data_thread_offset + data_channel_offset + data_row_offset + k;
        int input_idx = input_thread_offset + input_channel_offset + input_row_offset + k+max_col;
        data[data_idx] = input[input_idx] * sum;
       //printf("data %.f %.f %d %d %d %d ",data[data_idx], sum, input_idx, c, j, k);
      }
    }
  }



  // now do the parallel reduction, this can definitely be made better
  __syncthreads();

  int block_offset = blockIdx.x * input_channels * filter_x_ * filter_y_;
  // this is really shitty memory access pattern but that's why we moved it to shared memory
  // the idea is now that we have a N * C_i * filter_x * filter_y thing in shared memory
  // we want to reduce it across the N dimension.

  for(int j= data_block_offset + threadIdx.x;j<input_channels*filter_x_*filter_y_;j+=blockDim.x){
    int output_index = block_offset + j;
    sum = 0.0;
    for(int k =j;k<blockDim.x*input_channels*filter_x_*filter_y_;k+=input_channels*filter_x_*filter_y_)
    {
      sum = sum + data[k];
    }
    sum = sum/blockDim.x;
    output[output_index] = sum;
    //printf("%.f ",output[output_index]);
  }


}

void TonyConvGradKernelLauncher(const float * input, const int input_size_[], const float* dy, const int dy_size_[], float * output,
    int filter_x_,int filter_y_,int stride) {

  //std::cout << "dy size " << dy_size_[0] << " " <<  dy_size_[1] << " " << dy_size_[2] << " " << dy_size_[3] << std::endl;
  //std::cout << input_size_[0] << " " <<  input_size_[1] << " " << input_size_[2] << " " << input_size_[3] << std::endl;
  cudaMemcpyToSymbol(input_size,input_size_,sizeof(int)*4);
  cudaMemcpyToSymbol(dy_size,dy_size_,sizeof(int)*4);
  int data_size = dy_size_[1] * input_size_[0]*filter_x_*filter_y_*input_size_[1]*sizeof(float);
  float * data;
  cudaMalloc((void**)&data, data_size);

  //std::cout << input_size_[0]*filter_x_*filter_y_*input_size_[1] << std::endl;

  TonyConvKernelDraft<<<dy_size_[1],input_size_[0]>>>(input,dy,data,output,filter_x_,filter_y_,stride);

  //std::cout << "kernel finished successfully" << std::endl;
  //float *d_debug = (float*) malloc(sizeof(float)*filter_x_*filter_y_*input_size[1]*dy_size[1]);
  //std::cout << "transferred to debug host variable" << std::endl;

  cudaFree(data);

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
