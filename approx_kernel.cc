#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;  // NOLINT(build/namespaces)

REGISTER_OP("TonyConvGrad")
    .Input("input: float")
    .Input("dy: float")
    .Attr("stride: int = 1")
    .Attr("filter_x: int = 5")
    .Attr("filter_y: int = 5")
    .Output("output: float")
    .Doc(R"doc(
performs tony grad
)doc");

void TonyConvGradKernelLauncher(const float * h_input, const int input_size[], const float* h_dy, const int dy_size[], float * h_output,
    int filter_x_,int filter_y_,int stride);

class TonyConvGradOp : public OpKernel {
 public:
  explicit TonyConvGradOp(OpKernelConstruction* context) : OpKernel(context) {

    OP_REQUIRES_OK(context,
                   context->GetAttr("stride", &stride_));
    OP_REQUIRES_OK(context,
                    context->GetAttr("filter_x", &filter_x_));
    OP_REQUIRES_OK(context,
                    context->GetAttr("filter_y", &filter_y_));


  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.flat<float>();
    TensorShape input_shape = input_tensor.shape();

    // assumes NCHW. NHWC will be faster in CUDA due to memory coalescing but transposing the filter gradient
    // is easier than transposing the input.
    const int input_channels = input_shape.dim_size(1);
    const int batch_size = input_shape.dim_size(0);
    const int input_height = input_shape.dim_size(2);
    const int input_width = input_shape.dim_size(3);

    const int input_dims [4] = {batch_size,input_channels,input_height,input_width};

    const Tensor& dy_tensor = context->input(1);
    auto dy = dy_tensor.flat<float>();
    TensorShape dy_shape = dy_tensor.shape();

    const int y_channels = dy_shape.dim_size(1);

    const int output_dims[4] = {batch_size,y_channels,dy_shape.dim_size(2),dy_shape.dim_size(3)};


    // Think very carefully about the shape of this. Will need to be transposed.
    Tensor * output_tensor = nullptr;
    TensorShape output_shape;
    output_shape.AddDim(y_channels);
    output_shape.AddDim(input_channels);
    output_shape.AddDim(filter_x_);
    output_shape.AddDim(filter_y_);


    //std::cout << "result shape " <<  output_shape << std::endl;

    OP_REQUIRES_OK(context,context->allocate_output(0,output_shape,&output_tensor));

    auto output = output_tensor->template flat<float>();

    //std::cout << input.data() << std::endl;
    //std::cout << input_dims[0] << input_dims[1] << input_dims[2] << std::endl;


    const float * test = input.data();
    //std::cout << test[0] << std::endl;

    TonyConvGradKernelLauncher(input.data(), input_dims, dy.data(), output_dims, output.data(),
        filter_x_,filter_y_,stride_);
  }

  private:
    int filter_x_;
    int filter_y_;
    int stride_;
};

REGISTER_KERNEL_BUILDER(Name("TonyConvGrad").Device(DEVICE_GPU), TonyConvGradOp);
