#include <torch/extension.h>
#include <vector>

torch::Tensor tony_conv_kernel(

    torch::Tensor input,
    torch::Tensor dy,
    torch::Tensor stride_,
    torch::Tensor filter_x_,
    torch::Tensor filter_y_

) ;

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor tony_conv(

    torch::Tensor input,
    torch::Tensor dy,
    torch::Tensor stride,
    torch::Tensor filter_x,
    torch::Tensor filter_y

) {

    CHECK_INPUT(input);
    CHECK_INPUT(dy);
    CHECK_INPUT(stride);
    CHECK_INPUT(filter_x);
    CHECK_INPUT(filter_y);

    return tony_conv_kernel(input,dy,stride,filter_x,filter_y);

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("tony_conv", &tony_conv, "Tony conv (CUDA)");
}
