#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

#define filter_x_ 5
#define filter_y_ 5

using namespace tensorflow;  // NOLINT(build/namespaces)

REGISTER_OP("TonyConvGrad")
    .Input("input: float")
    .Input("sums: float")
    .Input("indices: int32")
    .Output("output: float")
    .Doc(R"doc(
performs tony grad
)doc");

void TonyConvGradKernelLauncher(const float * h_input, const float * sums, const int * indices, float * h_output);

class TonyConvGradOp : public OpKernel {
 public:
  explicit TonyConvGradOp(OpKernelConstruction* context) : OpKernel(context) {


  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<float>();
    TensorShape input_shape = input_tensor.shape();

    // the input will be NHWC but dy will be NCHW. This needs to be compatible with the conv framework.

    const Tensor& sums_tensor = context->input(1);
    auto sums = sums_tensor.flat<float>();
    TensorShape dy_shape = sums_tensor.shape();


    const int y_channels = dy_shape.dim_size(1);



    const Tensor& indices_tensor = context->input(2);
    auto indices = indices_tensor.flat<int32>();

    // Think very carefully about the shape of this. Will need to be transposed. This is different from the native NCHW kernel.
    Tensor * output_tensor = nullptr;
    TensorShape output_shape;
    output_shape.AddDim(input_shape.dim_size(0));
    output_shape.AddDim(y_channels);
    output_shape.AddDim(filter_x_);
    output_shape.AddDim(filter_y_);

    OP_REQUIRES_OK(context,context->allocate_output(0,output_shape,&output_tensor));

    auto output = output_tensor->template flat<float>();

    //std::cout << input.data() << std::endl;
    //std::cout << input_dims[0] << input_dims[1] << input_dims[2] << std::endl;

    TonyConvGradKernelLauncher(input.data(), sums.data(), indices.data(), output.data());
  }

};

REGISTER_KERNEL_BUILDER(Name("TonyConvGrad").Device(DEVICE_GPU), TonyConvGradOp);

