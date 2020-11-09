#include <cudnn.h>
#include <cassert>
#include <cstdlib>
#include <iostream>

#define BATCH_SIZE 128
#define FILTER_X 3
#define FILTER_Y 3
#define WITH_RELU 0

#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      std::cerr << "Error on line " << __LINE__ << ": "      \
                << cudnnGetErrorString(status) << std::endl; \
      std::exit(EXIT_FAILURE);                               \
    }                                                        \
  }



void CudnnConvKernelLauncher(const float * input, const float * dy, float * kernel)
{

  cudnnHandle_t cudnn;
  cudnnCreate(&cudnn);

  cudnnTensorDescriptor_t input_descriptor;
  checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
  checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*batch_size=*/BATCH_SIZE,
                                        /*channels=*/IC,
                                        /*image_height=*/IMAGE_DIM,
                                        /*image_width=*/IMAGE_DIM));

  cudnnFilterDescriptor_t kernel_descriptor;
  checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
  checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        /*out_channels=*/OC,
                                        /*in_channels=*/IC,
                                        /*kernel_height=*/FILTER_X,
                                        /*kernel_width=*/FILTER_Y));

  cudnnTensorDescriptor_t output_descriptor;
  checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
  checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
                                        /*format=*/CUDNN_TENSOR_NCHW,
                                        /*dataType=*/CUDNN_DATA_FLOAT,
                                        /*batch_size=*/BATCH_SIZE,
                                        /*channels=*/OC,
                                        /*image_height=*/IMAGE_DIM,
                                        /*image_width=*/IMAGE_DIM));

  cudnnConvolutionDescriptor_t convolution_descriptor;
  checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
  checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
                                             /*pad_height=*/1,
                                             /*pad_width=*/1,
                                             /*vertical_stride=*/1,
                                             /*horizontal_stride=*/1,
                                             /*dilation_height=*/1,
                                             /*dilation_width=*/1,
                                             /*mode=*/CUDNN_CROSS_CORRELATION,
                                             /*computeType=*/CUDNN_DATA_FLOAT));


  cudnnConvolutionBwdFilterAlgo_t convolution_algorithm;
  checkCUDNN(
      cudnnGetConvolutionBackwardFilterAlgorithm(cudnn,
                                          input_descriptor,
                                          output_descriptor,
                                          convolution_descriptor,
                                          kernel_descriptor,
                                          CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
                                          1000000,
                                          &convolution_algorithm));
  std::cout << convolution_algorithm << std::endl;
  size_t workspace_bytes{0};
  checkCUDNN(cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn,
                                                     input_descriptor,
                                                     output_descriptor,
                                                     convolution_descriptor,
                                                     kernel_descriptor,
                                                     convolution_algorithm,
                                                     &workspace_bytes));
  std::cerr << "Workspace size: " << (workspace_bytes / 1048576.0) << "MB"
            << std::endl;
  //assert(workspace_bytes > 0);

  cudaEvent_t start,stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  void* d_workspace{nullptr};
  cudaMalloc(&d_workspace, workspace_bytes);


  const float alpha = 1.0f, beta = 0.0f;

  
  cudaEventRecord(start);

  checkCUDNN(cudnnConvolutionBackwardFilter(cudnn,
                                     &alpha,
                                     input_descriptor,
                                     input,
                                     output_descriptor,
                                     dy,
                                     convolution_descriptor,
                                     convolution_algorithm,
                                     d_workspace,
                                     workspace_bytes,
                                     &beta,
                                     kernel_descriptor,
                                     kernel));

  //checkCUDNN(cudnnLRNCrossChannelForward(cudnn,


  



  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float time;
  cudaEventElapsedTime(&time,start,stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  std::cout << "baseline used " << time << std::endl;

  cudaFree(d_workspace);

  cudnnDestroyTensorDescriptor(input_descriptor);
  cudnnDestroyTensorDescriptor(output_descriptor);
  cudnnDestroyFilterDescriptor(kernel_descriptor);
  cudnnDestroyConvolutionDescriptor(convolution_descriptor);

  cudnnDestroy(cudnn);
}
