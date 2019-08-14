#include <torch/extension.h>
#include <vector>

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