#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;  // NOLINT(build/namespaces)

REGISTER_OP("NchwScatter")
    .Input("values: float")
    .Input("indices: int32")
    .Input("input: float")
    .Output("output: float")
    .Doc(R"doc(
performs scatter
)doc");

void ScatterKernelLauncher(const float * values, const int * indices, int n, int nnz,
int c, int hw, float *output) ;

class NchwScatterOp : public OpKernel {
 public:
  explicit NchwScatterOp(OpKernelConstruction* context) : OpKernel(context) {


  }

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor

    const Tensor& input_tensor = context->input(2);
    const Tensor& input_v_tensor = context->input(0);
    auto values = input_v_tensor.flat<float>();
    const Tensor& input_i_tensor = context->input(1);
    auto indices = input_i_tensor.flat<int>();
    TensorShape input_shape = input_v_tensor.shape();
    TensorShape dense_input_shape = input_tensor.shape();

    int h = dense_input_shape.dim_size(2);
    int w = dense_input_shape.dim_size(3);
    int n = input_shape.dim_size(0);
    int c = input_shape.dim_size(1);
    int nnz = input_shape.dim_size(2);
    // Think very carefully about the shape of this. Will need to be transposed. This is different from the native NCHW kernel.

    //std::cout << h << w << std::endl; 
    Tensor * output_tensor = nullptr;
    TensorShape output_shape;
    output_shape.AddDim(n);
    output_shape.AddDim(c);
    output_shape.AddDim(h);
    output_shape.AddDim(w);

    OP_REQUIRES_OK(context,context->allocate_output(0,output_shape,&output_tensor));

    auto output = output_tensor->template flat<float>();

    //std::cout << input.data() << std::endl;
    //std::cout << input_dims[0] << input_dims[1] << input_dims[2] << std::endl;

    ScatterKernelLauncher(values.data(), indices.data(), n, nnz, c, h* w,output.data());
  }

};

REGISTER_KERNEL_BUILDER(Name("NchwScatter").Device(DEVICE_GPU), NchwScatterOp);
