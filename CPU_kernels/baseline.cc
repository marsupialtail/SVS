#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

#define FILTER_X 3
#define FILTER_Y 3

using namespace tensorflow;  // NOLINT(build/namespaces)

REGISTER_OP("CudnnConv")
    .Input("input: float")
    .Input("dy: float")
    .Output("kernel: float")
    .Doc(R"doc(
performs cudnn filter grad
)doc");

void CudnnConvKernelLauncher(const float * input, const float* dy, float * kernel);

class CudnnConvOp : public OpKernel {
 public:
  explicit CudnnConvOp(OpKernelConstruction* context) : OpKernel(context) {


  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    // THESE WILL ALREADY BE IN SPARSE FORMAT!!!
    const Tensor& input_v_tensor = context->input(0);
    auto input_v = input_v_tensor.flat<float>();
    TensorShape input_shape = input_v_tensor.shape();

    const int ic = input_shape.dim_size(1);

    const Tensor& dy_v_tensor = context->input(1);
    auto dy_v = dy_v_tensor.flat<float>();
    TensorShape dy_shape = dy_v_tensor.shape();

    const int oc =dy_shape.dim_size(1);

    Tensor * output_tensor = nullptr;
    TensorShape output_shape;
    output_shape.AddDim(FILTER_X);
    output_shape.AddDim(FILTER_Y);
    output_shape.AddDim(ic);
    output_shape.AddDim(oc);

    
    OP_REQUIRES_OK(context,context->allocate_output(0,output_shape,&output_tensor));
    auto kernel = output_tensor -> template flat<float>();

//    Tensor * output_v_tensor = nullptr;
//    Tensor * output_i_tensor = nullptr;
//    TensorShape output_shape;
//    output_shape.AddDim(batch_size);
//    output_shape.AddDim(NNZ_IN * IMAGE_DIM);
//    output_shape.AddDim(oc);
//
//    OP_REQUIRES_OK(context,context->allocate_output(0,output_shape,&output_v_tensor));
//    OP_REQUIRES_OK(context,context->allocate_output(0,output_shape,&output_i_tensor));


//    auto output_v = output_tensor->template flat<float>();
//    auto output_i = output_tensor->template flat<float>();


    //std::cout << input.data() << std::endl;
    //std::cout << input_dims[0] << input_dims[1] << input_dims[2] << std::endl;

    std::cout << input_shape << std::endl;
    std::cout << dy_shape << std::endl;
    std::cout << output_shape << std::endl;
    CudnnConvKernelLauncher(input_v.data(), dy_v.data(), kernel.data());
  }

};

REGISTER_KERNEL_BUILDER(Name("CudnnConv").Device(DEVICE_GPU), CudnnConvOp);
