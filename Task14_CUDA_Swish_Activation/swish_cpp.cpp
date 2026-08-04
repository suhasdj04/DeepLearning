#include <torch/extension.h>
#include <vector>

// Forward declarations of CUDA launcher functions
torch::Tensor swish_cuda_forward(const torch::Tensor& x);
torch::Tensor swish_cuda_backward(const torch::Tensor& dy, const torch::Tensor& x);

// C++ wrappers
torch::Tensor swish_forward(const torch::Tensor& x) {
    return swish_cuda_forward(x);
}

torch::Tensor swish_backward(const torch::Tensor& dy, const torch::Tensor& x) {
    return swish_cuda_backward(dy, x);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &swish_forward, "Swish forward (CUDA)");
    m.def("backward", &swish_backward, "Swish backward (CUDA)");
}
