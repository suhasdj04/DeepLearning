import os
import json

tasks = [
    ("Task01_Vectorized_MLP", "Vectorized_MLP.ipynb"),
    ("Task02_Weight_Init", "Weight_Initialization_Diagnostic.ipynb"),
    ("Task03_Vectorized_Conv", "Vectorized_Convolution_im2col.ipynb"),
    ("Task04_Custom_Optimizers", "Custom_Optimizers.ipynb"),
    ("Task05_BatchNorm_LayerNorm", "Custom_Normalization.ipynb"),
    ("Task06_BiLSTM_Cell", "BiLSTM_Cell_Scratch.ipynb"),
    ("Task07_ResNet50_Bottleneck", "ResNet50_Bottleneck.ipynb"),
    ("Task08_Semantic_Segmentation", "Semantic_Segmentation.ipynb"),
    ("Task09_YOLOv8_Loss", "YOLOv8_Loss_Backpropagation.ipynb"),
    ("Task10_WGAN_GP", "WGAN_GP_Synthesis.ipynb"),
    ("Task11_Vision_Transformer", "Vision_Transformer_MHSA.ipynb"),
    ("Task12_Latent_Anomaly_Detection", "Anomaly_Detection_Autoencoder.ipynb"),
    ("Task13_Temporal_Conv_Networks", "Temporal_Convolutional_Network.ipynb"),
    ("Task14_CUDA_Swish_Activation", "CUDA_Swish_Activation.ipynb"),
    ("Task15_Distributed_Data_Parallel", "DDP_Distributed_Sync.ipynb")
]

base_dir = "d:/Deep_Learning_Tasks"
all_ok = True

print("Checking generated notebooks...")
for idx, (folder, file) in enumerate(tasks, 1):
    path = os.path.join(base_dir, folder, file)
    if not os.path.exists(path):
        print(f"[-] Task {idx:02d} FAILED: {path} does not exist.")
        all_ok = False
        continue
        
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if "cells" not in data or "metadata" not in data:
            print(f"[-] Task {idx:02d} FAILED: {path} is not a valid Jupyter Notebook structure.")
            all_ok = False
        else:
            print(f"[+] Task {idx:02d} PASSED: {folder}/{file} ({len(data['cells'])} cells)")
    except json.JSONDecodeError as e:
        print(f"[-] Task {idx:02d} FAILED: {path} contains invalid JSON: {e}")
        all_ok = False

if all_ok:
    print("\nSUCCESS: All 15 tasks have valid notebooks!")
else:
    print("\nFAILURE: Some tasks are missing or invalid.")
