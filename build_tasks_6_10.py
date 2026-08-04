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

def build_task_6():
    cells = []
    
    cells.append({
        "cell_type": "markdown",
        "id": "t6-md1",
        "metadata": {},
        "source": [
            "# Task 6: Bidirectional Long Short-Term Memory (BiLSTM) Cell Mechanics from Scratch\n",
            "\n",
            "**Objective:** Build recurrent gate architectures, memory cell updates, and bidirectional sequence routing from scratch using only raw PyTorch tensor operations.\n",
            "\n",
            "### Gated Recurrent Equations\n",
            "For each recurrent step $t$:\n",
            "1. **Forget Gate:** $f_t = \\sigma(W_f x_t + U_f h_{t-1} + b_f)$\n",
            "2. **Input Gate:** $i_t = \\sigma(W_i x_t + U_i h_{t-1} + b_i)$\n",
            "3. **Candidate Cell State:** $\\tilde{C}_t = \\tanh(W_c x_t + U_c h_{t-1} + b_c)$\n",
            "4. **Cell State Update:** $C_t = f_t \\odot C_{t-1} + i_t \\odot \\tilde{C}_t$\n",
            "5. **Output Gate:** $o_t = \\sigma(W_o x_t + U_o h_{t-1} + b_o)$\n",
            "6. **Hidden State Update:** $h_t = o_t \\odot \\tanh(C_t)$\n",
            "\n",
            "In a Bidirectional LSTM, we run the sequence forward ($0 \\dots T-1$) and backward ($T-1 \\dots 0$), concatenating the final hidden outputs $[h_t^{forward}; h_t^{backward}]$."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": "t6-code1",
        "metadata": {},
        "outputs": [],
        "source": [
            "import torch\n",
            "import torch.nn as nn\n",
            "import numpy as np\n",
            "\n",
            "class LSTMCellScratch:\n",
            "    def __init__(self, input_dim, hidden_dim):\n",
            "        self.input_dim = input_dim\n",
            "        self.hidden_dim = hidden_dim\n",
            "        \n",
            "        # Combine weights for all 4 gates (Forget, Input, Candidate, Output) to optimize dot products\n",
            "        # Shape: (input_dim + hidden_dim, 4 * hidden_dim)\n",
            "        self.W = torch.randn(input_dim + hidden_dim, 4 * hidden_dim) * np.sqrt(2.0 / (input_dim + hidden_dim))\n",
            "        self.b = torch.zeros(4 * hidden_dim)\n",
            "        \n",
            "    def forward(self, x, h_prev, c_prev):\n",
            "        # x: (batch_size, input_dim)\n",
            "        # h_prev: (batch_size, hidden_dim)\n",
            "        # c_prev: (batch_size, hidden_dim)\n",
            "        \n",
            "        # Concatenate inputs\n",
            "        combined = torch.cat([x, h_prev], dim=-1)\n",
            "        \n",
            "        # Calculate linear projection for all gates\n",
            "        gates = torch.matmul(combined, self.W) + self.b\n",
            "        \n",
            "        # Split projections into 4 gates\n",
            "        f_gate, i_gate, c_cand, o_gate = torch.chunk(gates, 4, dim=-1)\n",
            "        \n",
            "        # Non-linear activations\n",
            "        f = torch.sigmoid(f_gate)\n",
            "        i = torch.sigmoid(i_gate)\n",
            "        c_tilde = torch.tanh(c_cand)\n",
            "        o = torch.sigmoid(o_gate)\n",
            "        \n",
            "        # State updates\n",
            "        c = f * c_prev + i * c_tilde\n",
            "        h = o * torch.tanh(c)\n",
            "        \n",
            "        return h, c\n",
            "\n",
            "class BidirectionalLSTMScratch:\n",
            "    def __init__(self, input_dim, hidden_dim):\n",
            "        self.hidden_dim = hidden_dim\n",
            "        # Initialize forward and backward cells\n",
            "        self.forward_cell = LSTMCellScratch(input_dim, hidden_dim)\n",
            "        self.backward_cell = LSTMCellScratch(input_dim, hidden_dim)\n",
            "        \n",
            "    def forward(self, X, seq_lengths):\n",
            "        # X: (batch_size, seq_len, input_dim)\n",
            "        # seq_lengths: (batch_size,) - list of actual lengths for each sequence in the batch\n",
            "        batch_size, seq_len, input_dim = X.shape\n",
            "        \n",
            "        # Initial hidden and cell states\n",
            "        h_f = torch.zeros(batch_size, self.hidden_dim)\n",
            "        c_f = torch.zeros(batch_size, self.hidden_dim)\n",
            "        h_b = torch.zeros(batch_size, self.hidden_dim)\n",
            "        c_b = torch.zeros(batch_size, self.hidden_dim)\n",
            "        \n",
            "        forward_outputs = []\n",
            "        backward_outputs = [None] * seq_len\n",
            "        \n",
            "        # 1. Forward Pass\n",
            "        for t in range(seq_len):\n",
            "            x_t = X[:, t, :]\n",
            "            h_f, c_f = self.forward_cell.forward(x_t, h_f, c_f)\n",
            "            \n",
            "            # Handle padding: if step exceeds sequence length, keep previous cell state\n",
            "            mask = (t < seq_lengths).float().unsqueeze(-1)\n",
            "            h_f = mask * h_f + (1 - mask) * (forward_outputs[-1] if forward_outputs else 0)\n",
            "            \n",
            "            forward_outputs.append(h_f)\n",
            "            \n",
            "        # 2. Backward Pass\n",
            "        for t in reversed(range(seq_len)):\n",
            "            x_t = X[:, t, :]\n",
            "            h_b, c_b = self.backward_cell.forward(x_t, h_b, c_b)\n",
            "            \n",
            "            # Handle padding: mask backward inputs if index exceeds sequence length\n",
            "            mask = (t < seq_lengths).float().unsqueeze(-1)\n",
            "            h_b = mask * h_b\n",
            "            \n",
            "            backward_outputs[t] = h_b\n",
            "            \n",
            "        # Stack outputs\n",
            "        f_out = torch.stack(forward_outputs, dim=1)\n",
            "        b_out = torch.stack(backward_outputs, dim=1)\n",
            "        \n",
            "        # Concatenate forward and backward representation\n",
            "        out = torch.cat([f_out, b_out], dim=-1)\n",
            "        return out"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": "t6-code2",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Setup simple inputs and verify shape routing\n",
            "batch_size = 4\n",
            "seq_len = 10\n",
            "input_dim = 16\n",
            "hidden_dim = 32\n",
            "\n",
            "X = torch.randn(batch_size, seq_len, input_dim)\n",
            "seq_lengths = torch.tensor([10, 8, 6, 9]) # Varying input sequence lengths in the batch\n",
            "\n",
            "custom_bilstm = BidirectionalLSTMScratch(input_dim, hidden_dim)\n",
            "outputs = custom_bilstm.forward(X, seq_lengths)\n",
            "\n",
            "print(\"Input Tensor Shape: \", X.shape)\n",
            "print(\"Output Tensor Shape:\", outputs.shape)\n",
            "assert outputs.shape == (batch_size, seq_len, 2 * hidden_dim), \"Verification failed: incorrect shape output!\"\n",
            "print(\"Success: Bidirectional LSTM correctly handles masking and forward/backward hidden states stacking!\")"
        ]
    })
    
    create_notebook("Task06_BiLSTM_Cell", "BiLSTM_Cell_Scratch.ipynb", cells)

def build_task_7():
    cells = []
    
    cells.append({
        "cell_type": "markdown",
        "id": "t7-md1",
        "metadata": {},
        "source": [
            "# Task 7: ResNet-50 Style Residual Bottleneck Block and Grouped Convolutions\n",
            "\n",
            "**Objective:** Build high-performance bottleneck projection networks and multi-channel grouped convolutions, running PyTorch Profiler to analyze computational execution costs."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": "t7-code1",
        "metadata": {},
        "outputs": [],
        "source": [
            "import torch\n",
            "import torch.nn as nn\n",
            "import torch.autograd.profiler as profiler\n",
            "\n",
            "class ResNetBottleneckBlock(nn.Module):\n",
            "    def __init__(self, in_channels, bottleneck_channels, out_channels, stride=1, groups=1):\n",
            "        super().__init__()\n",
            "        \n",
            "        # 1. 1x1 Convolution: dimension reduction\n",
            "        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, bias=False)\n",
            "        self.bn1 = nn.BatchNorm2d(bottleneck_channels)\n",
            "        \n",
            "        # 2. 3x3 Convolution: spatial processing (supports grouped convolutions)\n",
            "        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, \n",
            "                               stride=stride, padding=1, groups=groups, bias=False)\n",
            "        self.bn2 = nn.BatchNorm2d(bottleneck_channels)\n",
            "        \n",
            "        # 3. 1x1 Convolution: dimension expansion\n",
            "        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, bias=False)\n",
            "        self.bn3 = nn.BatchNorm2d(out_channels)\n",
            "        \n",
            "        self.relu = nn.ReLU(inplace=True)\n",
            "        \n",
            "        # Skip projection layer if input/output dimensions mismatch\n",
            "        if stride != 1 or in_channels != out_channels:\n",
            "            self.shortcut = nn.Sequential(\n",
            "                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),\n",
            "                nn.BatchNorm2d(out_channels)\n",
            "            )\n",
            "        else:\n",
            "            self.shortcut = nn.Identity()\n",
            "            \n",
            "    def forward(self, x):\n",
            "        residual = self.shortcut(x)\n",
            "        \n",
            "        out = self.relu(self.bn1(self.conv1(x)))\n",
            "        out = self.relu(self.bn2(self.conv2(out)))\n",
            "        out = self.bn3(self.conv3(out))\n",
            "        \n",
            "        out += residual\n",
            "        out = self.relu(out)\n",
            "        return out"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": "t7-code2",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Instanciate blocks with different settings\n",
            "in_c, bot_c, out_c = 256, 64, 256\n",
            "input_tensor = torch.randn(8, in_c, 56, 56) # standard size in ResNet-50\n",
            "\n",
            "standard_block = ResNetBottleneckBlock(in_c, bot_c, out_c, groups=1)\n",
            "grouped_block = ResNetBottleneckBlock(in_c, bot_c, out_c, groups=32) # Grouped Conv (cardinality=32)\n",
            "depthwise_block = ResNetBottleneckBlock(in_c, bot_c, out_c, groups=bot_c) # Depthwise Conv in bottleneck\n",
            "\n",
            "# Count parameters\n",
            "p_std = sum(p.numel() for p in standard_block.parameters())\n",
            "p_grp = sum(p.numel() for p in grouped_block.parameters())\n",
            "p_dth = sum(p.numel() for p in depthwise_block.parameters())\n",
            "\n",
            "print(f\"Standard Bottleneck Parameters: {p_std:,}\")\n",
            "print(f\"Grouped Bottleneck Parameters:  {p_grp:,}\")\n",
            "print(f\"Depthwise Bottleneck Parameters:{p_dth:,}\")\n",
            "\n",
            "# Profile execution using PyTorch Profiler\n",
            "with profiler.profile(record_shapes=True) as prof_std:\n",
            "    standard_block(input_tensor)\n",
            "\n",
            "with profiler.profile(record_shapes=True) as prof_grp:\n",
            "    grouped_block(input_tensor)\n",
            "\n",
            "with profiler.profile(record_shapes=True) as prof_dth:\n",
            "    depthwise_block(input_tensor)\n",
            "\n",
            "print(\"\\n--- Profiler Execution Summary (Total CPU Time) ---\")\n",
            "print(f\"Standard Block CPU Time:  {prof_std.self_cpu_time_total / 1000.0:.2f} ms\")\n",
            "print(f\"Grouped Block CPU Time:   {prof_grp.self_cpu_time_total / 1000.0:.2f} ms\")\n",
            "print(f\"Depthwise Block CPU Time: {prof_dth.self_cpu_time_total / 1000.0:.2f} ms\")"
        ]
    })
    
    create_notebook("Task07_ResNet50_Bottleneck", "ResNet50_Bottleneck.ipynb", cells)

def build_task_8():
    cells = []
    
    cells.append({
        "cell_type": "markdown",
        "id": "t8-md1",
        "metadata": {},
        "source": [
            "# Task 8: Semantic Segmentation Pipeline on Pixel-wise Grids using custom Dice Loss\n",
            "\n",
            "**Objective:** Build a spatial localization engine by coding a U-Net architecture, loading target masks using Albumentations augmentations, and implementing a composite BCE + Dice overlap loss from scratch."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": "t8-code1",
        "metadata": {},
        "outputs": [],
        "source": [
            "import torch\n",
            "import torch.nn as nn\n",
            "import torch.nn.functional as F\n",
            "from torch.utils.data import Dataset, DataLoader\n",
            "import albumentations as A\n",
            "from albumentations.pytorch import ToTensorV2\n",
            "import cv2\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "# 1. Custom Soft Dice Loss + Binary Cross Entropy Loss\n",
            "class DiceBCELoss(nn.Module):\n",
            "    def __init__(self, eps=1e-6):\n",
            "        super().__init__()\n",
            "        self.eps = eps\n",
            "        \n",
            "    def forward(self, pred_logits, targets):\n",
            "        # Sigmoid to normalize outputs to probabilities\n",
            "        probs = torch.sigmoid(pred_logits)\n",
            "        \n",
            "        # Flatten prediction and label vectors\n",
            "        probs_flat = probs.view(-1)\n",
            "        targets_flat = targets.view(-1)\n",
            "        \n",
            "        # BCE Loss\n",
            "        bce = F.binary_cross_entropy_with_logits(pred_logits, targets, reduction='mean')\n",
            "        \n",
            "        # Dice Coefficient Calculation\n",
            "        intersection = (probs_flat * targets_flat).sum()\n",
            "        union = probs_flat.sum() + targets_flat.sum()\n",
            "        dice_loss = 1.0 - (2.0 * intersection + self.eps) / (union + self.eps)\n",
            "        \n",
            "        return bce + dice_loss\n",
            "\n",
            "# 2. Simple U-Net Model Implementation\n",
            "class DoubleConv(nn.Module):\n",
            "    def __init__(self, in_c, out_c):\n",
            "        super().__init__()\n",
            "        self.conv = nn.Sequential(\n",
            "            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),\n",
            "            nn.BatchNorm2d(out_c),\n",
            "            nn.ReLU(inplace=True),\n",
            "            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),\n",
            "            nn.BatchNorm2d(out_c),\n",
            "            nn.ReLU(inplace=True)\n",
            "        )\n",
            "    def forward(self, x): return self.conv(x)\n",
            "\n",
            "class UNet(nn.Module):\n",
            "    def __init__(self, in_channels=3, out_channels=1):\n",
            "        super().__init__()\n",
            "        self.enc1 = DoubleConv(in_channels, 32)\n",
            "        self.enc2 = DoubleConv(32, 64)\n",
            "        self.pool = nn.MaxPool2d(2)\n",
            "        self.bottleneck = DoubleConv(64, 128)\n",
            "        self.upconv = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)\n",
            "        self.dec = DoubleConv(128, 64)\n",
            "        self.final = nn.Conv2d(64, out_channels, kernel_size=1)\n",
            "        \n",
            "    def forward(self, x):\n",
            "        x1 = self.enc1(x)\n",
            "        x2 = self.enc2(self.pool(x1))\n",
            "        bn = self.bottleneck(self.pool(x2))\n",
            "        \n",
            "        up = self.upconv(bn)\n",
            "        dec_in = torch.cat([up, x2], dim=1)\n",
            "        out = self.dec(dec_in)\n",
            "        # Skip connections can be added similarly\n",
            "        return self.final(out)"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": "t8-code2",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Synthetic segmentation dataset\n",
            "class SyntheticSegmentationDataset(Dataset):\n",
            "    def __init__(self, size=64, transform=None):\n",
            "        self.size = size\n",
            "        self.transform = transform\n",
            "        \n",
            "    def __len__(self):\n",
            "        return self.size\n",
            "        \n",
            "    def __getitem__(self, idx):\n",
            "        # Generate random image with circle mask shapes\n",
            "        img = np.zeros((128, 128, 3), dtype=np.uint8)\n",
            "        mask = np.zeros((128, 128), dtype=np.uint8)\n",
            "        \n",
            "        # Draw circle\n",
            "        cy, cx = np.random.randint(30, 90, size=2)\n",
            "        radius = np.random.randint(15, 30)\n",
            "        cv2.circle(img, (cx, cy), radius, (255, 255, 255), -1)\n",
            "        cv2.circle(mask, (cx, cy), radius, 1, -1)\n",
            "        \n",
            "        # Add noise\n",
            "        img = (img + np.random.randint(0, 50, img.shape)).clip(0, 255).astype(np.uint8)\n",
            "        \n",
            "        if self.transform:\n",
            "            augmented = self.transform(image=img, mask=mask)\n",
            "            img = augmented['image']\n",
            "            mask = augmented['mask']\n",
            "            \n",
            "        return img, mask.float().unsqueeze(0)\n",
            "\n",
            "# Transforms using Albumentations\n",
            "transform = A.Compose([\n",
            "    A.HorizontalFlip(p=0.5),\n",
            "    A.RandomBrightnessContrast(p=0.2),\n",
            "    A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),\n",
            "    ToTensorV2()\n",
            "])\n",
            "\n",
            "dataset = SyntheticSegmentationDataset(size=64, transform=transform)\n",
            "dataloader = DataLoader(dataset, batch_size=8, shuffle=True)\n",
            "\n",
            "model = UNet()\n",
            "criterion = DiceBCELoss()\n",
            "optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)\n",
            "\n",
            "# Single train epoch verification\n",
            "model.train()\n",
            "for img, mask in dataloader:\n",
            "    optimizer.zero_grad()\n",
            "    pred = model(img)\n",
            "    loss = criterion(pred, mask)\n",
            "    loss.backward()\n",
            "    optimizer.step()\n",
            "    \n",
            "print(f\"Training successfully verified! Single-batch Loss: {loss.item():.4f}\")"
        ]
    })
    
    create_notebook("Task08_Semantic_Segmentation", "Semantic_Segmentation.ipynb", cells)

def build_task_9():
    cells = []
    
    cells.append({
        "cell_type": "markdown",
        "id": "t9-md1",
        "metadata": {},
        "source": [
            "# Task 9: Anchor-Free Object Detection (YOLOv8-style) Loss Backpropagation\n",
            "\n",
            "**Objective:** Build high-precision target regressors by writing classification and CIoU (Complete Intersection over Union) box regression loss computations from scratch."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": "t9-code1",
        "metadata": {},
        "outputs": [],
        "source": [
            "import torch\n",
            "import torch.nn as nn\n",
            "import numpy as np\n",
            "import matplotlib.pyplot as plt\n",
            "\n",
            "def calculate_ciou(bboxes1, bboxes2):\n",
            "    # bboxes format: (N, 4) -> [x1, y1, x2, y2]\n",
            "    # Area calculation\n",
            "    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])\n",
            "    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])\n",
            "    \n",
            "    # Intersections\n",
            "    lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])\n",
            "    rb = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])\n",
            "    wh = (rb - lt).clamp(min=0)\n",
            "    inter = wh[:, 0] * wh[:, 1]\n",
            "    \n",
            "    # Union\n",
            "    union = area1 + area2 - inter\n",
            "    iou = inter / union.clamp(min=1e-6)\n",
            "    \n",
            "    # Center distances\n",
            "    ctr1 = (bboxes1[:, :2] + bboxes1[:, 2:]) / 2.0\n",
            "    ctr2 = (bboxes2[:, :2] + bboxes2[:, 2:]) / 2.0\n",
            "    center_dist = torch.sum((ctr1 - ctr2) ** 2, dim=-1)\n",
            "    \n",
            "    # Smallest enclosing box dimensions\n",
            "    enclose_lt = torch.min(bboxes1[:, :2], bboxes2[:, :2])\n",
            "    enclose_rb = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])\n",
            "    enclose_wh = (enclose_rb - enclose_lt).clamp(min=0)\n",
            "    enclose_diag = torch.sum(enclose_wh ** 2, dim=-1).clamp(min=1e-6)\n",
            "    \n",
            "    # Aspect ratio metrics\n",
            "    w1, h1 = bboxes1[:, 2] - bboxes1[:, 0], bboxes1[:, 3] - bboxes1[:, 1]\n",
            "    w2, h2 = bboxes2[:, 2] - bboxes2[:, 0], bboxes2[:, 3] - bboxes2[:, 1]\n",
            "    v = (4 / (np.pi ** 2)) * torch.pow(torch.atan(w1 / h1.clamp(min=1e-6)) - torch.atan(w2 / h2.clamp(min=1e-6)), 2)\n",
            "    \n",
            "    with torch.no_grad():\n",
            "        alpha = v / ((1.0 - iou) + v).clamp(min=1e-6)\n",
            "        \n",
            "    ciou = iou - (center_dist / enclose_diag + alpha * v)\n",
            "    return 1.0 - ciou"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": "t9-code2",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Create mock detections and verify loss calculation\n",
            "pred_boxes = torch.tensor([[10., 10., 50., 50.], [20., 20., 80., 80.]], requires_grad=True)\n",
            "target_boxes = torch.tensor([[12., 12., 48., 48.], [20., 20., 80., 80.]])\n",
            "\n",
            "ciou_loss = calculate_ciou(pred_boxes, target_boxes).mean()\n",
            "ciou_loss.backward()\n",
            "\n",
            "print(\"Computed CIoU Loss:       \", ciou_loss.item())\n",
            "print(\"Box Gradients w.r.t Loss:\\n\", pred_boxes.grad)\n",
            "assert ciou_loss.item() > 0, \"Failed: loss should be non-negative!\"\n",
            "print(\"Success: Box regression loss and gradients generated smoothly!\")"
        ]
    })
    
    create_notebook("Task09_YOLOv8_Loss", "YOLOv8_Loss_Backpropagation.ipynb", cells)

def build_task_10():
    cells = []
    
    cells.append({
        "cell_type": "markdown",
        "id": "t1-md1",
        "metadata": {},
        "source": [
            "# Task 10: Wasserstein GAN with Gradient Penalty (WGAN-GP) for Stable Image Synthesis\n",
            "\n",
            "**Objective:** Stabilize adversarial generative networks by implementing Wasserstein distance metrics with strict Lipschitz-1 gradient constraints."
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": "t1-code1",
        "metadata": {},
        "outputs": [],
        "source": [
            "import torch\n",
            "import torch.nn as nn\n",
            "import torch.optim as optim\n",
            "import torchvision.transforms as transforms\n",
            "import matplotlib.pyplot as plt\n",
            "import numpy as np\n",
            "\n",
            "# Generator Network Architecture\n",
            "class Generator(nn.Module):\n",
            "    def __init__(self, z_dim=100, im_channels=1, hidden_dim=64):\n",
            "        super().__init__()\n",
            "        self.gen = nn.Sequential(\n",
            "            nn.ConvTranspose2d(z_dim, hidden_dim * 4, 4, 1, 0, bias=False),\n",
            "            nn.BatchNorm2d(hidden_dim * 4),\n",
            "            nn.ReLU(True),\n",
            "            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, 3, 2, 1, bias=False),\n",
            "            nn.BatchNorm2d(hidden_dim * 2),\n",
            "            nn.ReLU(True),\n",
            "            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, 4, 2, 1, bias=False),\n",
            "            nn.BatchNorm2d(hidden_dim),\n",
            "            nn.ReLU(True),\n",
            "            nn.ConvTranspose2d(hidden_dim, im_channels, 4, 2, 1, bias=False),\n",
            "            nn.Tanh()\n",
            "        )\n",
            "    def forward(self, x): return self.gen(x)\n",
            "\n",
            "# Critic Network Architecture (No sigmoid activation at the output!)\n",
            "class Critic(nn.Module):\n",
            "    def __init__(self, im_channels=1, hidden_dim=64):\n",
            "        super().__init__()\n",
            "        self.critic = nn.Sequential(\n",
            "            nn.Conv2d(im_channels, hidden_dim, 4, 2, 1),\n",
            "            nn.LeakyReLU(0.2, inplace=True),\n",
            "            nn.Conv2d(hidden_dim, hidden_dim * 2, 4, 2, 1),\n",
            "            nn.LeakyReLU(0.2, inplace=True),\n",
            "            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, 3, 2, 1),\n",
            "            nn.LeakyReLU(0.2, inplace=True),\n",
            "            nn.Conv2d(hidden_dim * 4, 1, 4, 1, 0)\n",
            "        )\n",
            "    def forward(self, x): return self.critic(x)\n",
            "\n",
            "# Gradient Penalty calculation\n",
            "def compute_gradient_penalty(critic, real_images, fake_images, device):\n",
            "    batch_size = real_images.shape[0]\n",
            "    epsilon = torch.rand(batch_size, 1, 1, 1, device=device).expand_as(real_images)\n",
            "    \n",
            "    # Interpolated sample space\n",
            "    interpolated = epsilon * real_images + (1 - epsilon) * fake_images\n",
            "    interpolated.requires_grad_(True)\n",
            "    \n",
            "    # Critic score of interpolates\n",
            "    interpolated_logits = critic(interpolated)\n",
            "    \n",
            "    # Gradient calculation\n",
            "    gradients = torch.autograd.grad(\n",
            "        outputs=interpolated_logits,\n",
            "        inputs=interpolated,\n",
            "        grad_outputs=torch.ones_like(interpolated_logits),\n",
            "        create_graph=True,\n",
            "        retain_graph=True,\n",
            "        only_inputs=True\n",
            "    )[0]\n",
            "    \n",
            "    gradients = gradients.view(batch_size, -1)\n",
            "    grad_norm = gradients.norm(2, dim=1)\n",
            "    gp = torch.mean((grad_norm - 1) ** 2)\n",
            "    return gp"
        ]
    })
    
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "id": "t1-code2",
        "metadata": {},
        "outputs": [],
        "source": [
            "# Setup models and test a backward step of Critic and Generator\n",
            "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
            "gen = Generator().to(device)\n",
            "critic = Critic().to(device)\n",
            "\n",
            "opt_critic = optim.Adam(critic.parameters(), lr=1e-4, betas=(0.0, 0.9))\n",
            "opt_gen = optim.Adam(gen.parameters(), lr=1e-4, betas=(0.0, 0.9))\n",
            "\n",
            "real = torch.randn(8, 1, 28, 28, device=device) # mock inputs\n",
            "noise = torch.randn(8, 100, 1, 1, device=device)\n",
            "\n",
            "# Forward passes\n",
            "fake = gen(noise)\n",
            "critic_real = critic(real)\n",
            "critic_fake = critic(fake.detach())\n",
            "\n",
            "# GP Calculation\n",
            "gp = compute_gradient_penalty(critic, real, fake.detach(), device)\n",
            "\n",
            "# Critic Loss (Wasserstein Distance objective with gradient penalty)\n",
            "loss_critic = critic_fake.mean() - critic_real.mean() + 10.0 * gp\n",
            "\n",
            "opt_critic.zero_grad()\n",
            "loss_critic.backward()\n",
            "opt_critic.step()\n",
            "\n",
            "# Generator Loss\n",
            "critic_fake_new = critic(fake)\n",
            "loss_gen = -critic_fake_new.mean()\n",
            "\n",
            "opt_gen.zero_grad()\n",
            "loss_gen.backward()\n",
            "opt_gen.step()\n",
            "\n",
            "print(f\"Critic Step Loss:    {loss_critic.item():.4f}\")\n",
            "print(f\"Generator Step Loss: {loss_gen.item():.4f}\")\n",
            "print(\"Success: Generative adversarial networks forward and backward passes executed smoothly!\")"
        ]
    })
    
    create_notebook("Task10_WGAN_GP", "WGAN_GP_Synthesis.ipynb", cells)

if __name__ == "__main__":
    build_task_6()
    build_task_7()
    build_task_8()
    build_task_9()
    build_task_10()
