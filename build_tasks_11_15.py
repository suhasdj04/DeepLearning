import json
import os

def create_notebook(folder_name, file_name, cells):
    os.makedirs(os.path.join("d:/Deep_Learning_Tasks", folder_name), exist_ok=True)
    filepath = os.path.join("d:/Deep_Learning_Tasks", folder_name, file_name)
    
    notebook_content = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            }
        },
        "cells": cells
    }
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(notebook_content, f, indent=1)
    print(f"Created: {filepath}")

def build_task_11():
    cells = []
    
    cells.append({
        "cell_type": "markdown",
        "id": "t11-md1",
        "metadata": {},
        "source": [
            "# Task 11: Vision Transformer (ViT) Patch Projection and Multi-Head Self-Attention from Scratch\n",
            "\n",
            "**Objective:** Adapt Transformer architectures to visual data by constructing spatial patch extraction modules, adding learnable class and position embeddings, and implementing multi-head self-attention using Einstein summation syntax via `einops`."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": "t11-code1",
        "metadata": {},
        "outputs": [],
        "source": [
            "import torch\n",
            "import torch.nn as nn\n",
            "import torch.nn.functional as F\n",
            "from einops import rearrange, repeat\n",
            "import matplotlib.pyplot as plt\n",
            "import numpy as np\n",
            "\n",
            "# 1. Linear Patch Projection Layer\n",
            "class PatchEmbedding(nn.Module):\n",
            "    def __init__(self, in_channels=3, patch_size=16, embed_dim=256):\n",
            "        super().__init__()\n",
            "        self.patch_size = patch_size\n",
            "        # Projection of flattened patches to embedding dimension\n",
            "        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)\n",
            "        \n",
            "    def forward(self, x):\n",
            "        # Input: (B, C, H, W)\n",
            "        x = self.projection(x) # Shape: (B, embed_dim, H/P, W/P)\n",
            "        x = rearrange(x, 'b e h w -> b (h w) e') # Shape: (B, N_patches, embed_dim)\n",
            "        return x\n",
            "\n",
            "# 2. Multi-Head Self-Attention (MHSA) using einops and einsum\n",
            "class MultiHeadSelfAttention(nn.Module):\n",
            "    def __init__(self, embed_dim=256, num_heads=8, dropout=0.1):\n",
            "        super().__init__()\n",
            "        self.num_heads = num_heads\n",
            "        self.head_dim = embed_dim // num_heads\n",
            "        self.scale = self.head_dim ** -0.5\n",
            "        \n",
            "        # QKV Projections\n",
            "        self.qkv_projection = nn.Linear(embed_dim, embed_dim * 3, bias=False)\n",
            "        self.out_projection = nn.Linear(embed_dim, embed_dim)\n",
            "        self.attn_dropout = nn.Dropout(dropout)\n",
            "        \n",
            "        # Save attention weights for visualization\n",
            "        self.attention_weights = None\n",
            "        \n",
            "    def forward(self, x):\n",
            "        B, N, E = x.shape\n",
            "        \n",
            "        # Calculate QKV matrices\n",
            "        qkv = self.qkv_projection(x) # (B, N, 3*E)\n",
            "        # Split queries, keys, and values and project into heads using einops\n",
            "        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=self.num_heads, d=self.head_dim)\n",
            "        \n",
            "        # Compute scaled attention weights using Einstein Summation\n",
            "        # Attention = Softmax(Q K^T / sqrt(d_k))\n",
            "        scores = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale\n",
            "        attn = F.softmax(scores, dim=-1)\n",
            "        attn = self.attn_dropout(attn)\n",
            "        \n",
            "        # Cache for visualization\n",
            "        self.attention_weights = attn.detach()\n",
            "        \n",
            "        # Compute weighted sum of values\n",
            "        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)\n",
            "        # Concatenate heads and project\n",
            "        out = rearrange(out, 'b h n d -> b n (h d)')\n",
            "        out = self.out_projection(out)\n",
            "        return out"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": "t11-code2",
        "metadata": {},
        "outputs": [],
        "source": [
            "# 3. Full Transformer Block and test pass with attention extraction\n",
            "class ViTBlock(nn.Module):\n",
            "    def __init__(self, embed_dim=256, num_heads=8):\n",
            "        super().__init__()\n",
            "        self.ln1 = nn.LayerNorm(embed_dim)\n",
            "        self.attn = MultiHeadSelfAttention(embed_dim, num_heads)\n",
            "        self.ln2 = nn.LayerNorm(embed_dim)\n",
            "        self.mlp = nn.Sequential(\n",
            "            nn.Linear(embed_dim, embed_dim * 4),\n",
            "            nn.GELU(),\n",
            "            nn.Linear(embed_dim * 4, embed_dim)\n",
            "        )\n",
            "        \n",
            "    def forward(self, x):\n",
            "        # Residual connection + pre-LN self-attention\n",
            "        x = x + self.attn(self.ln1(x))\n",
            "        # Residual connection + pre-LN feed forward network\n",
            "        x = x + self.mlp(self.ln2(x))\n",
            "        return x\n",
            "\n",
            "# Verify layout using dummy image\n",
            "img = torch.randn(2, 3, 224, 224) # batch size 2, 224x224 RGB image\n",
            "patch_size = 16\n",
            "embed_dim = 256\n",
            "\n",
            "pat_embed = PatchEmbedding(in_channels=3, patch_size=patch_size, embed_dim=embed_dim)\n",
            "vit_block = ViTBlock(embed_dim=embed_dim, num_heads=8)\n",
            "\n",
            "patches = pat_embed(img)\n",
            "\n",
            "# Append Class Token\n",
            "cls_token = nn.Parameter(torch.randn(2, 1, embed_dim))\n",
            "patches = torch.cat([cls_token, patches], dim=1)\n",
            "\n",
            "# Add Position Embeddings\n",
            "pos_embed = nn.Parameter(torch.randn(1, 1 + (224//patch_size)**2, embed_dim))\n",
            "x = patches + pos_embed\n",
            "\n",
            "# Feed to ViT block\n",
            "out = vit_block(x)\n",
            "\n",
            "print(\"Patches shape:         \", patches.shape)\n",
            "print(\"Class-token & pos shape:\", x.shape)\n",
            "print(\"Output shape:          \", out.shape)\n",
            "print(\"Self-Attention weight shape:\", vit_block.attn.attention_weights.shape)\n",
            "\n",
            "assert out.shape == x.shape, \"Verification failed! Shape mismatch!\"\n",
            "print(\"Success: Vision Transformer embedding projection and self-attention operations completed successfully!\")"
        ]
    })
    
    create_notebook("Task11_Vision_Transformer", "Vision_Transformer_MHSA.ipynb", cells)

def build_task_12():
    cells = []
    
    cells.append({
        "cell_type": "markdown",
        "id": "t12-md1",
        "metadata": {},
        "source": [
            "# Task 12: High-Dimensional Latent Space Anomaly Detection via Convolutional Autoencoders\n",
            "\n",
            "**Objective:** Identify structural anomalies and defects in fabric/manufacturing images by training a deep convolutional autoencoder on fault-free datasets and evaluating SSIM reconstruction error maps."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": "t12-code1",
        "metadata": {},
        "outputs": [],
        "source": [
            "import torch\n",
            "import torch.nn as nn\n",
            "import torch.nn.functional as F\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "from sklearn.metrics import roc_curve, auc\n",
            "\n",
            "# 1. SSIM Loss function from scratch (Simplified 2D implementation using local kernels)\n",
            "def calculate_ssim(x, y, window_size=11, eps=1e-8):\n",
            "    # x, y shape: (B, 1, H, W)\n",
            "    C1 = (0.01 * 2.0) ** 2\n",
            "    C2 = (0.03 * 2.0) ** 2\n",
            "    \n",
            "    # Local mean kernel\n",
            "    kernel = torch.ones(1, 1, window_size, window_size, device=x.device) / (window_size ** 2)\n",
            "    \n",
            "    mu1 = F.conv2d(x, kernel, padding=window_size//2)\n",
            "    mu2 = F.conv2d(y, kernel, padding=window_size//2)\n",
            "    \n",
            "    mu1_sq = mu1.pow(2)\n",
            "    mu2_sq = mu2.pow(2)\n",
            "    mu1_mu2 = mu1 * mu2\n",
            "    \n",
            "    sigma1_sq = F.conv2d(x * x, kernel, padding=window_size//2) - mu1_sq\n",
            "    sigma2_sq = F.conv2d(y * y, kernel, padding=window_size//2) - mu2_sq\n",
            "    sigma12 = F.conv2d(x * y, kernel, padding=window_size//2) - mu1_mu2\n",
            "    \n",
            "    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))\n",
            "    return torch.clamp(ssim_map, min=-1.0, max=1.0)\n",
            "\n",
            "# 2. Convolutional Autoencoder with deep bottleneck\n",
            "class ConvAutoencoder(nn.Module):\n",
            "    def __init__(self):\n",
            "        super().__init__()\n",
            "        # Encoder: compress image down to bottleneck size\n",
            "        self.encoder = nn.Sequential(\n",
            "            nn.Conv2d(1, 16, 3, stride=2, padding=1), # 128x128 -> 64x64\n",
            "            nn.ReLU(),\n",
            "            nn.Conv2d(16, 32, 3, stride=2, padding=1), # 64x64 -> 32x32\n",
            "            nn.ReLU(),\n",
            "            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 32x32 -> 16x16\n",
            "            nn.ReLU(),\n",
            "            nn.Conv2d(64, 8, 3, stride=1, padding=1)  # Deep Bottleneck (8 feature channels)\n",
            "        )\n",
            "        # Decoder: reconstruct back\n",
            "        self.decoder = nn.Sequential(\n",
            "            nn.ConvTranspose2d(8, 64, 3, stride=2, padding=1, output_padding=1), # 16x16 -> 32x32\n",
            "            nn.ReLU(),\n",
            "            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), # 32x32 -> 64x64\n",
            "            nn.ReLU(),\n",
            "            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1), # 64x64 -> 128x128\n",
            "            nn.ReLU(),\n",
            "            nn.Conv2d(16, 1, 3, padding=1),\n",
            "            nn.Sigmoid()\n",
            "        )\n",
            "        \n",
            "    def forward(self, x):\n",
            "        latent = self.encoder(x)\n",
            "        reconstructed = self.decoder(latent)\n",
            "        return reconstructed"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": "t12-code2",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Setup mock training and anomaly evaluation pipeline\n",
            "autoencoder = ConvAutoencoder()\n",
            "optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)\n",
            "\n",
            "# Generate mock defect-free training images (normal horizontal grid lines)\n",
            "normal_images = []\n",
            "for _ in range(20):\n",
            "    grid = np.zeros((1, 128, 128), dtype=np.float32)\n",
            "    grid[0, ::8, :] = 1.0 # Draw clean grid\n",
            "    normal_images.append(torch.tensor(grid))\n",
            "normal_loader = torch.stack(normal_images)\n",
            "\n",
            "# Simple training loop\n",
            "for epoch in range(10):\n",
            "    reconstructed = autoencoder(normal_loader)\n",
            "    # Composite Loss: MSE + SSIM\n",
            "    mse = F.mse_loss(reconstructed, normal_loader)\n",
            "    ssim = 1.0 - calculate_ssim(reconstructed, normal_loader).mean()\n",
            "    loss = mse + 0.5 * ssim\n",
            "    \n",
            "    optimizer.zero_grad()\n",
            "    loss.backward()\n",
            "    optimizer.step()\n",
            "    \n",
            "# Evaluate an anomaly (introducing random blobs/scratches on normal grids)\n",
            "normal_test = normal_loader[0:1]\n",
            "anomaly_test = normal_loader[0:1].clone()\n",
            "anomaly_test[0, 0, 40:60, 40:60] = 0.5 # Add defect\n",
            "\n",
            "with torch.no_grad():\n",
            "    rec_normal = autoencoder(normal_test)\n",
            "    rec_anomaly = autoencoder(anomaly_test)\n",
            "    \n",
            "    # Compute reconstruction error map\n",
            "    err_normal = (normal_test - rec_normal).pow(2)\n",
            "    err_anomaly = (anomaly_test - rec_anomaly).pow(2)\n",
            "\n",
            "print(\"Defect-free Reconstruction Loss (MSE):\", err_normal.mean().item())\n",
            "print(\"Anomalous Image Reconstruction Loss (MSE): \", err_anomaly.mean().item())\n",
            "assert err_anomaly.mean().item() > err_normal.mean().item(), \"Anomaly detection failed!\"\n",
            "print(\"Success: Anomaly autoencoder correctly flags structural defects!\")"
        ]
    })
    
    create_notebook("Task12_Latent_Anomaly_Detection", "Anomaly_Detection_Autoencoder.ipynb", cells)

def build_task_13():
    cells = []
    
    cells.append({
        "cell_type": "markdown",
        "id": "t13-md1",
        "metadata": {},
        "source": [
            "# Task 13: Temporal Convolutional Networks (TCN) with Dilated Causal Convolutions\n",
            "\n",
            "**Objective:** Model sequential patterns and time-series history without temporal look-ahead leaks by implementing 1D causal convolutions featuring exponentially increasing dilation factors ($d=2^l$)."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": "t13-code1",
        "metadata": {},
        "outputs": [],
        "source": [
            "import torch\n",
            "import torch.nn as nn\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "# Custom layer to crop/slice future outputs from convolution\n",
            "class Chomp1d(nn.Module):\n",
            "    def __init__(self, chomp_size):\n",
            "        super().__init__()\n",
            "        self.chomp_size = chomp_size\n",
            "        \n",
            "    def forward(self, x):\n",
            "        # x shape: (B, C, L)\n",
            "        # Slice out future indices introduced by symmetric padding\n",
            "        return x[:, :, :-self.chomp_size].contiguous()\n",
            "\n",
            "# 1. Dilated Causal Residual Block\n",
            "class TemporalBlock(nn.Module):\n",
            "    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):\n",
            "        super().__init__()\n",
            "        # First dilated causal Conv\n",
            "        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,\n",
            "                               stride=stride, padding=padding, dilation=dilation)\n",
            "        self.chomp1 = Chomp1d(padding)\n",
            "        self.relu1 = nn.ReLU()\n",
            "        self.dropout1 = nn.Dropout(dropout)\n",
            "        \n",
            "        # Second dilated causal Conv\n",
            "        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,\n",
            "                               stride=stride, padding=padding, dilation=dilation)\n",
            "        self.chomp2 = Chomp1d(padding)\n",
            "        self.relu2 = nn.ReLU()\n",
            "        self.dropout2 = nn.Dropout(dropout)\n",
            "        \n",
            "        self.net = nn.Sequential(\n",
            "            self.conv1, self.chomp1, self.relu1, self.dropout1,\n",
            "            self.conv2, self.chomp2, self.relu2, self.dropout2\n",
            "        )\n",
            "        \n",
            "        # Linear projection shortcut if input/output dimensions differ\n",
            "        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None\n",
            "        self.relu = nn.ReLU()\n",
            "        \n",
            "    def forward(self, x):\n",
            "        out = self.net(x)\n",
            "        res = x if self.downsample is None else self.downsample(x)\n",
            "        return self.relu(out + res)"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": "t13-code2",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Assemble multi-layered TCN and test with a 1D sequence\n",
            "class TemporalConvNet(nn.Module):\n",
            "    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):\n",
            "        super().__init__()\n",
            "        layers = []\n",
            "        num_levels = len(num_channels)\n",
            "        \n",
            "        for i in range(num_levels):\n",
            "            # Systematically double the dilation factor: 1, 2, 4, 8...\n",
            "            dilation_size = 2 ** i\n",
            "            in_channels = num_inputs if i == 0 else num_channels[i-1]\n",
            "            out_channels = num_channels[i]\n",
            "            \n",
            "            # To ensure output matches input length, padding must equal (kernel_size - 1) * dilation\n",
            "            padding_size = (kernel_size - 1) * dilation_size\n",
            "            \n",
            "            layers.append(\n",
            "                TemporalBlock(in_channels, out_channels, kernel_size, stride=1, \n",
            "                              dilation=dilation_size, padding=padding_size, dropout=dropout)\n",
            "            )\n",
            "            \n",
            "        self.network = nn.Sequential(*layers)\n",
            "        \n",
            "    def forward(self, x): return self.network(x)\n",
            "\n",
            "# Test sequence shape (B, C, L) where length = 100\n",
            "B, C, L = 4, 1, 100\n",
            "x = torch.randn(B, C, L)\n",
            "\n",
            "tcn = TemporalConvNet(num_inputs=1, num_channels=[8, 16, 32], kernel_size=3)\n",
            "out = tcn(x)\n",
            "\n",
            "print(\"Input Sequence Shape: \", x.shape)\n",
            "print(\"Output Sequence Shape:\", out.shape)\n",
            "assert out.shape == (B, 32, L), \"Verification failed: output length modified!\"\n",
            "print(\"Success: TCN causal convolutions successfully compiled!\")"
        ]
    })
    
    create_notebook("Task13_Temporal_Conv_Networks", "Temporal_Convolutional_Network.ipynb", cells)

def build_task_14():
    cells = []
    
    cells.append({
        "cell_type": "markdown",
        "id": "t14-md1",
        "metadata": {},
        "source": [
            "# Task 14: Custom CUDA-Accelerated Swish Activation Layer Integration\n",
            "\n",
            "**Objective:** Build high-performance GPU kernels to accelerate non-linear activation layers, compiling PyTorch C++ bindings dynamically.\n",
            "\n",
            "### Formula\n",
            "$$\\text{Swish}(x) = x \\cdot \\text{Sigmoid}(x)$$\n",
            "$$\\text{Swish}'(x) = \\text{Sigmoid}(x) + x \\cdot \\text{Sigmoid}(x) \\cdot (1 - \\text{Sigmoid}(x))$$\n",
            "\n",
            "This folder contains:\n",
            "1. C++ bindings (`swish_cpp.cpp`)\n",
            "2. CUDA source (`swish_kernel.cu`)\n",
            "3. Compiler setup (`setup.py`)\n",
            "\n",
            "The notebook contains custom PyTorch Autograd fallbacks and benchmarks."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": "t14-code1",
        "metadata": {},
        "outputs": [],
        "source": [
            "import torch\n",
            "import torch.nn as nn\n",
            "import time\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "# 1. Custom Autograd Function (Python Fallback version for numerical correctness and testing)\n",
            "class CustomSwishFunction(torch.autograd.Function):\n",
            "    @staticmethod\n",
            "    def forward(ctx, x):\n",
            "        ctx.save_for_backward(x)\n",
            "        return x * torch.sigmoid(x)\n",
            "        \n",
            "    @staticmethod\n",
            "    def backward(ctx, dy):\n",
            "        x, = ctx.saved_tensors\n",
            "        sig = torch.sigmoid(x)\n",
            "        dx = dy * (sig + x * sig * (1.0 - sig))\n",
            "        return dx\n",
            "\n",
            "class CustomSwish(nn.Module):\n",
            "    def forward(self, x):\n",
            "        return CustomSwishFunction.apply(x)\n",
            "\n",
            "# 2. Benchmark code against PyTorch Native\n",
            "x_eval = torch.randn(5000, 5000, device='cuda' if torch.cuda.is_available() else 'cpu')\n",
            "\n",
            "# Evaluation\n",
            "swish_custom = CustomSwish()\n",
            "swish_native = nn.SiLU() # PyTorch's native Swish implementation\n",
            "\n",
            "t0 = time.time()\n",
            "for _ in range(100):\n",
            "    y_custom = swish_custom(x_eval)\n",
            "t1 = time.time()\n",
            "\n",
            "t2 = time.time()\n",
            "for _ in range(100):\n",
            "    y_native = swish_native(x_eval)\n",
            "t3 = time.time()\n",
            "\n",
            "print(f\"Custom Swish (100 passes): {t1 - t0:.6f} seconds\")\n",
            "print(f\"Native SiLU  (100 passes): {t3 - t2:.6f} seconds\")"
        ]
    })
    
    create_notebook("Task14_CUDA_Swish_Activation", "CUDA_Swish_Activation.ipynb", cells)
    
    # Write C++ bindings and CUDA files
    cpp_content = r'''#include <torch/extension.h>
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
'''
    
    cuda_content = r'''#include <torch/extension.h>
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
'''
    
    setup_content = r'''from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='swish_cuda',
    ext_modules=[
        CUDAExtension('swish_cuda', [
            'swish_cpp.cpp',
            'swish_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
'''
    
    task_dir = "d:/Deep_Learning_Tasks/Task14_CUDA_Swish_Activation"
    os.makedirs(task_dir, exist_ok=True)
    with open(os.path.join(task_dir, "swish_cpp.cpp"), "w") as f:
        f.write(cpp_content)
    with open(os.path.join(task_dir, "swish_kernel.cu"), "w") as f:
        f.write(cuda_content)
    with open(os.path.join(task_dir, "setup.py"), "w") as f:
        f.write(setup_content)

def build_task_15():
    cells = []
    
    cells.append({
        "cell_type": "markdown",
        "id": "t15-md1",
        "metadata": {},
        "source": [
            "# Task 15: Distributed Data Parallel (DDP) Multi-GPU Node Sync Gateway with GLOO/NCCL\n",
            "\n",
            "**Objective:** Master large-scale deep learning model partitioning and gradient synchronization across hardware nodes by deploying DDP routines.\n",
            "\n",
            "This folder contains:\n",
            "1. Multi-GPU training script (`ddp_train.py`)\n",
            "2. Container environment (`Dockerfile`)\n",
            "3. Orchestration gateway (`docker-compose.yml`)\n",
            "\n",
            "The notebook contains structural walkthroughs and local execution templates."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": "t15-code1",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Local multi-process DDP simulation hook using PyTorch multiprocessing\n",
            "import torch\n",
            "import torch.distributed as dist\n",
            "import torch.multiprocessing as mp\n",
            "import torch.nn as nn\n",
            "\n",
            "def run_worker(rank, world_size):\n",
            "    # Set master details\n",
            "    import os\n",
            "    os.environ['MASTER_ADDR'] = 'localhost'\n",
            "    os.environ['MASTER_PORT'] = '12355'\n",
            "    \n",
            "    # Initialize process group using GLOO backend (fully supported on Windows/CPU!)\n",
            "    dist.init_process_group(\"gloo\", rank=rank, world_size=world_size)\n",
            "    \n",
            "    model = nn.Linear(10, 1).to(rank)\n",
            "    # Wrap model in PyTorch DDP\n",
            "    ddp_model = nn.parallel.DistributedDataParallel(model)\n",
            "    \n",
            "    # Generate dummy input w.r.t process rank\n",
            "    inputs = torch.randn(4, 10)\n",
            "    outputs = ddp_model(inputs)\n",
            "    loss = outputs.sum()\n",
            "    \n",
            "    loss.backward()\n",
            "    dist.destroy_process_group()\n",
            "    print(f\"Worker {rank} successfully executed training step!\")\n",
            "\n",
            "if __name__ == '__main__':\n",
            "    world_size = 2\n",
            "    # Spawn workers locally to simulate sync gateway\n",
            "    mp.spawn(run_worker, args=(world_size,), nprocs=world_size, join=True)"
        ]
    })
    
    create_notebook("Task15_Distributed_Data_Parallel", "DDP_Distributed_Sync.ipynb", cells)
    
    # Write Dockerfile, docker-compose.yml and training script
    ddp_train_script = r'''import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, TensorDataset

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    # NCCL backend for GPU sync on Linux containers, GLOO for CPU/Windows
    backend = 'nccl' if torch.cuda.is_available() else 'gloo'
    dist.init_process_group(backend, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)
    
    device = torch.device(f'cuda:{rank}' if torch.cuda.is_available() else 'cpu')
    
    # Model Setup
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Linear(50, 10)
    ).to(device)
    
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank] if torch.cuda.is_available() else None)
    
    # Data Setup
    x = torch.randn(1000, 100)
    y = torch.randint(0, 10, (1000,))
    dataset = TensorDataset(x, y)
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(5):
        sampler.set_epoch(epoch)
        for data, targets in dataloader:
            data, targets = data.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            
        if rank == 0:
            print(f"Epoch {epoch} complete | Loss: {loss.item():.4f}")
            
    cleanup()

if __name__ == '__main__':
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    train(rank, world_size)
'''
    
    dockerfile_content = r'''FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /workspace

RUN pip install --no-cache-dir matplotlib pandas scikit-learn

COPY ddp_train.py /workspace/ddp_train.py

ENTRYPOINT ["python", "ddp_train.py"]
'''
    
    docker_compose_content = r'''version: '3.8'

services:
  master:
    build: .
    environment:
      - WORLD_SIZE=2
      - RANK=0
      - MASTER_ADDR=master
      - MASTER_PORT=12355
    volumes:
      - .:/workspace
    network_mode: "host"

  worker:
    build: .
    environment:
      - WORLD_SIZE=2
      - RANK=1
      - MASTER_ADDR=master
      - MASTER_PORT=12355
    volumes:
      - .:/workspace
    network_mode: "host"
'''
    
    task_dir = "d:/Deep_Learning_Tasks/Task15_Distributed_Data_Parallel"
    os.makedirs(task_dir, exist_ok=True)
    with open(os.path.join(task_dir, "ddp_train.py"), "w") as f:
        f.write(ddp_train_script)
    with open(os.path.join(task_dir, "Dockerfile"), "w") as f:
        f.write(dockerfile_content)
    with open(os.path.join(task_dir, "docker-compose.yml"), "w") as f:
        f.write(docker_compose_content)

if __name__ == "__main__":
    build_task_11()
    build_task_12()
    build_task_13()
    build_task_14()
    build_task_15()
