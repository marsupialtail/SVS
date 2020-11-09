#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

#define XSTR(x) #x
#define STR(x) XSTR(x)
using namespace tensorflow;  // NOLINT(build/namespaces)

REGISTER_OP(STR(NAME))
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

    // the input will be NHWC but dy will be NCHW. This needs to be compatible with the conv framework.

    const int input_dims [4] = {input_shape.dim_size(0),input_shape.dim_size(1),input_shape.dim_size(2),input_shape.dim_size(3)};

    const Tensor& dy_tensor = context->input(1);
    auto dy = dy_tensor.flat<float>();
    TensorShape dy_shape = dy_tensor.shape();

    const int y_channels = dy_shape.dim_size(1);

    const int output_dims[4] = {dy_shape.dim_size(0),y_channels,dy_shape.dim_size(2),dy_shape.dim_size(3)};


    // Think very carefully about the shape of this. Will need to be transposed. This is different from the native NCHW kernel.
    Tensor * output_tensor = nullptr;
    TensorShape output_shape;
    output_shape.AddDim(y_channels);
    output_shape.AddDim(filter_x_);
    output_shape.AddDim(filter_y_);
    output_shape.AddDim(input_dims[3]);

    OP_REQUIRES_OK(context,context->allocate_output(0,output_shape,&output_tensor));

    auto output = output_tensor->template flat<float>();

    //std::cout << input.data() << std::endl;
    //std::cout << "NHWC launch params" << input_dims[0] << input_dims[1] << input_dims[2] << input_dims[3] << std::endl;

 //   std::cout << STR(NAME) << " "<< output_dims[0] << output_dims[1] << output_dims[2]<< output_dims[3] << std::endl;


    //std::cout << "NHWC launch params " << output_shape << std::endl;
    TonyConvGradKernelLauncher(input.data(), input_dims, dy.data(), output_dims, output.data(),
        filter_x_,filter_y_,stride_);
  }

  private:
    int filter_x_;
    int filter_y_;
    int stride_;
};

REGISTER_KERNEL_BUILDER(Name(STR(NAME)).Device(DEVICE_GPU), TonyConvGradOp);

