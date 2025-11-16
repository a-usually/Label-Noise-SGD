# [AAAI2026 (Oral)] On the Learning Dynamics of Two-layer Linear Networks with Label Noise SGD

This repository contains the implementation for the paper *"On the Learning Dynamics of Two-layer Linear Networks with Label Noise SGD"*, accepted as an oral presentation at AAAI 2026.

## Abstract
> <small>One crucial factor behind the success of deep learning lies in the implicit bias induced by noise inherent in gradient-based training algorithms. Motivated by empirical observations that training with noisy labels improves model generalization, we delve into the underlying mechanisms behind stochastic gradient descent (SGD) with label noise. Focusing on a two-layer over-parameterized linear network, we analyze the learning dynamics of label noise SGD, unveiling a two-phase learning behavior. In Phase I, the magnitudes of model weights progressively diminish, and the model escapes the lazy regime; enters the rich regime. In Phase II, the alignment between model weights and the ground-truth interpolator increases, and the model eventually converges. Our analysis highlights the critical role of label noise in driving the transition from the lazy to the rich regime and minimally explains its empirical success. Furthermore, we extend these insights to Sharpness-Aware Minimization (SAM), showing that the principles governing label noise SGD also apply to broader optimization algorithms. Extensive experiments, conducted under both synthetic and real-world setups, strongly support our theory. </small>

## Code Structure

The code is organized into three main parts:

### 1. Synthetic Experiments (`Synthetic/`)

- **`sgd_label_noise.ipynb`**: Main synthetic experiments demonstrating the two-phase phenomenon in a regression task with 200 samples from standard normal distribution, network width 1000, input dimension 20, learning rate 0.001, and noise level 4.
  
  **Key Observation**: Clear **two-phase behavior** where in Phase I, first-layer weights progressively diminish, followed by Phase II where weights align with ground truth.

- **`oscillation.ipynb`**: To investigate the cause of Phase I, we conducted experiments using GD while **directly oscillating the second-layer network weights** under the same setting. We also observed that the first-layer network weights progressively diminish to very small values. This further demonstrates that label noise accelerates the oscillations in the second layer, thereby contributing to the progressive diminishing of the first-layer weights.

- **`oscillation_periodic.ipynb`** & **`sgd_label_noise_periodic.ipynb`**: To more intuitively understand the role of label noise, we trained the model using SGD while **alternating** label noise every 5,000 steps (adding label noise for 5,000 steps and then removing label noise for 5,000 steps). This experiment reveals two key phenomena:
  - The norms of first-layer neurons gradually decrease when adding label noise, and the norm reduction stops when removing label noise.

  - The second-layer neuron weights oscillate near zero under noise but stabilize when noise is removed.

These observations collectively demonstrate the crucial role of label noise in facilitating the transition from lazy to rich regimes.

### 2. Real-World Experiments (`Real-World/`)

- **`function_space_linearization_MSE_loss_v1.ipynb`**: In the Real-World experiments, we investigated the results of training ResNet architectures on the CIFAR-10 dataset both with and without label noise. We randomly selected 64 images from CIFAR-10 to compute the Neural Tangent Kernel (NTK).

### 3. SAM Experiments (`SAM/`)

- **`SAM.ipynb`**: The principles governing label noise SGD also apply to broader optimization algorithms. We tested the SAM (Sharpness-Aware Minimization) algorithm with label noise and observed similar phenomena.

## Requirements

- **Synthetic Experiments & SAM Experiments**: Can run on CPU and complete within minutes
- **Real-World Experiments**: 
Make sure CUDA >= 11.0 is installed if running Real-World experiments on GPU. Verify with:
```bash
nvidia-smi
```
If you use Anaconda3 or Miniconda, you can run the following instructions to download the required packages in Python:
```bash
conda create -n label-noise-sgd python=3.11
conda activate label-noise-sgd
pip install torch torchvision torchaudio numpy matplotlib pandas tqdm
```