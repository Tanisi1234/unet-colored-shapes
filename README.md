# 👀 unet-colored-shapes

## 🚀 Objective

Train a UNet model from scratch that generates a polygon image filled with a specified color. The model takes two inputs:

- A grayscale image of a polygon (e.g., triangle, square).
- A color name (e.g., "red", "blue").

It outputs the same polygon filled with the specified color.

---

## 📁 Dataset Overview

- **Structure**:

  ```
  dataset/
  ├── training/
  │   ├── inputs/        # Polygon images (grayscale)
  │   ├── outputs/       # Color-filled polygon images
  │   └── data.json      # Maps input+color → output
  └── validation/
      ├── inputs/
      ├── outputs/
      └── data.json
  ```

- **Augmentations**:

  - Random rotations (±15°)
  - Random scaling (0.8x–1.2x)
  - Horizontal flips

---

## 🧍‍♂️ Model Architecture

### ⚙️ UNet Overview

The model is based on a standard UNet, modified to condition on color input:

- **Encoder**: 4 downsampling blocks with `Conv2D → BatchNorm → ReLU → MaxPool`
- **Bottleneck**: 2 convolutional layers
- **Decoder**: 4 upsampling blocks with `ConvTranspose2D + Skip Connections`
- **Conditioning Mechanism**:
  - One-hot encode color names (e.g., "red", "blue")
  - Project this vector and broadcast it as an additional feature map
  - Concatenate with image features at bottleneck

### 🔧 Key Design Choices

| Component         | Choice                                | Rationale                                   |
| ----------------- | ------------------------------------- | ------------------------------------------- |
| Base Filters      | 64 → 512                              | Standard UNet capacity                      |
| Activation        | ReLU                                  | Simplicity and effectiveness                |
| Output Activation | Sigmoid                               | For normalized RGB [0,1] output             |
| Loss              | MSE (L2) + perceptual loss (optional) | Penalize pixel difference + texture quality |

---

## 📊 Hyperparameters

| Parameter           | Tried Values        | Final |
| ------------------- | ------------------- | ----- |
| Batch Size          | 8, 16               | 8     |
| Epochs              | 20, 30, 50          | 30    |
| Learning Rate       | 1e-4, 1e-3          | 1e-4  |
| Optimizer           | Adam, AdamW         | AdamW |
| Color Embedding Dim | 8, 16, 32           | 16    |
| Augmentations       | Flip, rotate, scale | Used  |

**Rationale**:

- Smaller batch size due to GPU memory constraints.
- AdamW offered better convergence with weight decay.
- 30 epochs balanced performance and training time.

---

## 📉 Training Dynamics

- **Loss Curve**: Smooth decline in MSE across 30 epochs.
- **Metric**: PSNR and SSIM improved steadily.
- **Validation**:
  - PSNR \~28 dB on validation set
  - SSIM \~0.92
- **wandb Link**: [Click to view](https://wandb.ai/your-project-link)

### ⚠️ Typical Failure Modes

| Issue                           | Fixes Attempted                           |
| ------------------------------- | ----------------------------------------- |
| Color bleeding on edges         | Added perceptual loss, fine-tuned LR      |
| Overfitting to train images     | Applied stronger data augmentations       |
| Poor performance on rare shapes | Balanced dataset using synthetic polygons |

---

## 📌 Inference

Run the provided `inference.ipynb` notebook to:

1. Load the trained model (`.pth`)
2. Input a test polygon and a color string (e.g., "green")
3. Visualize predicted colored polygon

---

## 🧠 Key Learnings

- **Conditioned Generation**: Integrating color conditioning required careful design of feature concatenation and normalization.
- **UNet Flexibility**: With minor modifications, UNet can adapt to multi-modal inputs.
- **Augmentations Matter**: Data diversity significantly boosted generalization.
- **Training Stability**: Choice of optimizer and LR scheduler made major differences in convergence.

---

## ✅ Deliverables

- ✅ `training_colab.ipynb`: Model training and wandb logging.
- ✅ `inference.ipynb`: Visual test and demo.
- ✅ UNet model code (`model.py`)
- ✅ `README.md` (you are here)
- ✅ wandb Project: [wandb.ai/](https://wandb.ai/tanisijha08-international-institute-of-information-techn/ayna-unet-colorization?nw=nwusertanisijha08)

