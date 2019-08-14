#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include <math.h>
#include <stdio.h>
#include <cuda_profiler_api.h>

/*

    ASSUMING in_channel = out_channel, in practice, will split up work
    In this new kernel, each block is going to process one example in the batch
    each block will have shared memory allocated to input_height * input_width * input_channels to basically store the input image
    each thread will handle an output channel
    this kernel will be bottlenecked by shared memeory usage.

*/
__global__ void TonyConvKernelDraft(const float* input, const float* dy, float* output,
        const int filter_x_, const int filter_y_, const int stride, const int input_channels, const int is_2, const int is_3,
       const int ys_1, const int ys_2, const int ys_3) {

    extern __shared__ float data[];

    for (int i = threadIdx.x; i < filter_x_*filter_y_*input_channels; i += blockDim.x)
      data[i] = 0;

    __syncthreads();
    // no need to sync threads here

    float sum = 0.0;
    float max = 0.0;
    int t1, t2, max_idx;

    t1 = clock();
    int i = blockIdx.x * ys_1 * ys_2 * ys_3 + threadIdx.x * ys_2 * ys_3;
    for(int j = i; j < i + ys_2 * ys_3; j ++ )
    {
        if(fabs(dy[j]) > max)
        {
          max = fabs(dy[j]);
          max_idx = j;
        }

        sum += dy[j];
    }
    max_idx -= i;

    t2 = clock();
      if(blockIdx.x == 0){
        printf("%d ",t2-t1);
      }

    t1 = clock();
    int data_idx = threadIdx.x * is_2 * is_3;
    int input_idx = blockIdx.x * input_channels * is_2 * is_3 + threadIdx.x * is_2 * is_3;
    for(int j = input_idx; j < input_idx + is_2 * is_3; j++)
    {
        data[data_idx++] = input[j];
    }

    t2 = clock();
      if(blockIdx.x == 0){
        printf("%d ",t2-t1);
      }


    __syncthreads();


    int max_row = max_idx / ys_3 * stride;
    int max_col = max_idx % ys_3 * stride;

    // an option here is to use cuTT to transpose the shared memory first before operating.
    // however then we need 2x shared memory and we might have some problems.
    // now we rely on "lucky" warp coalescing when writing.

//    t1 = clock();
//
//    int output_offset = threadIdx.x * input_channels * filter_x_ * filter_y_;
//    for(int c = blockIdx.x; c < blockIdx.x + input_channels; c++ )
//    {
//        int output_channel_offset = output_offset + (c%input_channels) * filter_x_ * filter_y_;
//        int data_channel_offset = (c%input_channels) * is_2 * is_3;
//        for(int j = 0; j < filter_x_; j++)
//        {
//            int output_row_offset = output_channel_offset + j * filter_y_;
//            int data_row_offset = data_channel_offset + (j+max_row) * is_3;
//            for(int k = 0; k <  filter_y_; k ++)
//            {
//                int output_col_offset = output_row_offset + k;
//                int data_col_offset = data_row_offset + k + max_col;
//                //output[output_col_offset] += data[data_col_offset] / blockDim.x;
//
//                atomicAdd(&output[output_col_offset], data[data_col_offset] *sum/gridDim.x);
//            }
//        }
//    }
//
//    t2 = clock();
//      if(blockIdx.x == 0){
//        printf("%d ",t2-t1);
//      }


  //

}
void TonyConvGradKernelLauncher(const float * input, const int input_size_[], const float* dy, const int dy_size_[], float * output,
    int filter_x_,int filter_y_,int stride) {

  //std::cout << "dy size " << dy_size_[0] << " " <<  dy_size_[1] << " " << dy_size_[2] << " " << dy_size_[3] << std::endl;
  //std::cout << input_size_[0] << " " <<  input_size_[1] << " " << input_size_[2] << " " << input_size_[3] << std::endl;

  int data_size = input_size_[2]*input_size_[3]*input_size_[1]*sizeof(float);
  // float * data;
  //checkGpuMem();
  // std::cout << "allocating memory " << cudaMalloc((void**)&data, data_size * 2) << " " << data_size << std::endl;

  //std::cout << input_size_[0]*filter_x_*filter_y_*input_size_[1] << std::endl;

  cudaProfilerStart();

  TonyConvKernelDraft<<<input_size_[0],input_size_[1], data_size>>>(input,dy,output,filter_x_,filter_y_,stride,
  input_size_[1],input_size_[2],input_size_[3] ,dy_size_[1],dy_size_[2], dy_size_[3]);

  cudaProfilerStop();

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