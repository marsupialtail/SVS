#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include <math.h>

__global__ void TonyConvKernelDraft(const float* input, const int* input_size, const float* dy, const int * dy_size, float* output,
        const int filter_x_, const int filter_y_, const int stride) {

  /*

  Here is the plan. Each block will execute a different channel. Inside each block, every threads will
  more or less execute the find max and sum for each batch element. This is to make the reduce operation easier.

  */

  // first read the input and the corresponding dy into shared memory
  extern __shared__ float data [];

  int i = threadIdx.x * (dy_size[1] * dy_size[2] * dy_size[3]) + blockIdx.x * (dy_size[2] * dy_size[3]);
  int max_idx;
  float sum = 0.0;
  float max = 0.0;

  for(int j=0;j<dy_size[2]*dy_size[3];j++){
    if(fabs(dy[i+j]) > max)
    {
      max = fabs(dy[j]);
      max_idx = j;
    }
    sum = sum + dy[j];

  }

  int max_row = max_idx / dy_size[2] * stride;
  int max_col = max_idx % dy_size[2] * stride;

  // load stuff into registers that will be used over and over again. Try to think about memory IO here
  int input_height = input_size[2];
  int input_width = input_size[3];
  int input_channels = input_size[1];

  int data_thread_offset = threadIdx.x * input_channels * filter_x_ * filter_y_;
  int input_thread_offset = threadIdx.x * input_channels * input_height * input_width;

  for(int c=0;c<input_channels;c++){
    int data_channel_offset = c * filter_x_ * filter_y_;
    int input_channel_offset = c * input_height * input_width;

    for(int j=0;j<filter_x_;j++){
      int data_row_offset = j * filter_y_;
      int input_row_offset = (j+max_row) * input_width;

      for(int k=0;k<filter_y_;k++)
      {
        int data_idx = data_thread_offset + data_channel_offset + j * data_row_offset + k;
        int input_idx = input_thread_offset + input_channel_offset + input_row_offset + k+max_col;
        data[data_idx] = input[input_idx] * sum;
      }
    }
  }

  // now do the parallel reduction, this can definitely be made better
  __syncthreads();

  int block_offset = blockIdx.x * input_channels * filter_x_ * filter_y_;
  // this is really shitty memory access pattern but that's why we moved it to shared memory
  for(int j=threadIdx.x;j<input_channels*filter_x_*filter_y_;j+=blockDim.x){
    int output_index = block_offset + j;
    sum = 0.0;
    for(int k =j;k<blockDim.x*input_channels*filter_x_*filter_y_;k+=input_channels*filter_x_*filter_y_)
    {
      sum = sum + data[k];
    }
    sum = sum/blockDim.x;
    output[output_index] = sum;
  }


}

void TonyConvGradKernelLauncher(const float * h_input, const int input_size[], const float* h_dy, const int dy_size[], float * h_output,
    int filter_x_,int filter_y_,int stride) {

  float *d_input=0, *d_dy=0, *d_output = 0;
  cudaMalloc((void**)&d_input,sizeof(float)*input_size[0] * input_size[1] * input_size[2] * input_size[3]);
  cudaMalloc((void**)&d_dy,sizeof(float)*dy_size[0] * dy_size[1] * dy_size[2] * dy_size[3]);
  cudaMalloc((void**)&d_output,sizeof(float)*filter_x_*filter_y_*input_size[1]*dy_size[1]);

  cudaMemcpy(d_input,h_input,sizeof(float)*input_size[0] * input_size[1] * input_size[2] * input_size[3],cudaMemcpyHostToDevice);
  cudaMemcpy(d_dy,h_dy,sizeof(float)*dy_size[0] * dy_size[1] * dy_size[2] * dy_size[3],cudaMemcpyHostToDevice);

  std::cout << "copied tensors to device" << std::endl;

  // figure out what's going on with passing arguments by pointer for the sizes

  TonyConvKernelDraft<<<dy_size[3],input_size[0],input_size[0]*filter_x_*filter_y_*input_size[1]>>>(d_input,
  input_size,d_dy,dy_size,d_output,filter_x_,filter_y_,stride);

  std::cout << "kernel finished successfully" << std::endl;
  float *d_debug = (float*) malloc(sizeof(float)*filter_x_*filter_y_*input_size[1]*dy_size[1]);
  cudaMemcpy(h_output,d_debug,sizeof(float)*filter_x_*filter_y_*input_size[1]*dy_size[1],cudaMemcpyDeviceToHost);
  std::cout << "transferred to debug host variable" << std::endl;

  for(int i = 0;i<filter_x_*filter_y_*input_size[1]*dy_size[1];i++)
  {
    std::cout << d_debug[i];
  }

  cudaMemcpy(h_output,d_output,sizeof(float)*filter_x_*filter_y_*input_size[1]*dy_size[1],cudaMemcpyDeviceToHost);

  cudaFree(d_input);
  cudaFree(d_dy);
  cudaFree(d_output);
}

#endif