#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// CUDA Forward Kernel
__global__ void swish_forward_kernel(const float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + idx;
    if (idx < size) {
        float val = x[idx];
        out[idx] = val / (1.0f + expf(-val));
    }
}

// CUDA Backward Kernel
__global__ void swish_backward_kernel(const float* dy, const float* x, float* dx, int size) {
    int idx = blockIdx.x * blockDim.x + idx;
    if (idx < size) {
        float val = x[idx];
        float sig = 1.0f / (1.0f + expf(-val));
        dx[idx] = dy[idx] * (sig + val * sig * (1.0f - sig));
    }
}

// Launchers
torch::Tensor swish_cuda_forward(const torch::Tensor& x) {
    auto out = torch::zeros_like(x);
    int size = x.numel();
    int threads = 1024;
    int blocks = (size + threads - 1) / threads;

    swish_forward_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        out.data_ptr<float>(),
        size
    );
    return out;
}

torch::Tensor swish_cuda_backward(const torch::Tensor& dy, const torch::Tensor& x) {
    auto dx = torch::zeros_like(x);
    int size = x.numel();
    int threads = 1024;
    int blocks = (size + threads - 1) / threads;

    swish_backward_kernel<<<blocks, threads>>>(
        dy.data_ptr<float>(),
        x.data_ptr<float>(),
        dx.data_ptr<float>(),
        size
    );
    return dx;
}
