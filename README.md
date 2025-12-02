# NeRF: Neural Radiance Fields for 3D Scene Reconstruction

<div align="center">
  <img src="./media/nerf_blender/lego.gif" width="400" />
</div>

## Overview

This project implements Neural Radiance Fields (NeRF), a deep learning approach for synthesizing novel views of complex 3D scenes from a sparse set of 2D images. The implementation uses PyTorch and demonstrates how implicit neural representations combined with volume rendering can achieve photorealistic view synthesis.

**Key Features:**
- Fully-connected MLP architecture with positional encoding for high-frequency detail capture
- Differentiable volume rendering pipeline with stratified ray sampling
- Training and evaluation on the Blender synthetic dataset
- Quantitative evaluation using PSNR, SSIM, and LPIPS metrics
- Modular, research-oriented codebase with Hydra configuration management

## Method

NeRF represents a 3D scene as a continuous 5D function that maps spatial coordinates (x, y, z) and viewing direction (θ, φ) to color (RGB) and volume density (σ). The core components include:

### Architecture
- **MLP Network**: Multi-layer perceptron with skip connections mapping encoded positions and directions to density and RGB values
- **Positional Encoding**: Sinusoidal encoding of input coordinates to enable learning of high-frequency scene details
- **Volume Rendering**: Numerical integration along camera rays to compute pixel colors

### Training Pipeline
1. Generate camera rays from known camera poses and intrinsics
2. Sample 3D points along each ray using stratified sampling
3. Query the MLP network to predict density and color at each point
4. Integrate along rays using classical volume rendering equations
5. Compute photometric loss against ground-truth images

## Results

Trained and evaluated on the **Blender Synthetic Dataset** (Lego scene):

| Metric | Value |
|--------|-------|
| PSNR ↑ | TBD |
| SSIM ↑ | TBD |
| LPIPS ↓ | TBD |

*Note: Run evaluation to populate these metrics*

The model successfully reconstructs 3D geometry and appearance, enabling photorealistic novel view synthesis as shown in the visualization above.

## Repository Structure

```
torch_nerf/
├── configs/              # Hydra configuration files
├── runners/              # Training, rendering, and evaluation scripts
└── src/
    ├── cameras/          # Camera models and ray generation
    ├── network/          # NeRF MLP implementation
    ├── renderer/         # Ray sampling and volume rendering
    ├── scene/            # Scene representation and dataset loaders
    ├── signal_encoder/   # Positional encoding implementations
    └── utils/            # Utility functions and helpers
```

## Setup

### Environment

Create a conda environment with Python 3.8:

```bash
conda create --name nerf python=3.8
conda activate nerf
```

### Dependencies

Install PyTorch with CUDA support:

```bash
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```

Install additional requirements:

```bash
pip install -r requirements.txt
pip install torchmetrics[image]
pip install tensorboard
```

Set the Python path:

```bash
export PYTHONPATH=.
```

### Dataset

Download the NeRF Blender synthetic dataset:

```bash
# Download and extract to data/nerf_synthetic/
wget http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_synthetic.zip
unzip nerf_synthetic.zip -d data/
```

The directory structure should look like:
```
data/nerf_synthetic/
└── lego/
    ├── train/
    ├── val/
    ├── test/
    └── transforms_*.json
```

## Usage

### Training

Train NeRF on the lego scene:

```bash
python torch_nerf/runners/train.py
```

Training outputs (checkpoints, logs, visualizations) are saved to `outputs/` with automatic experiment tracking via Hydra. Monitor training progress with TensorBoard:

```bash
tensorboard --logdir outputs/
```

**Configuration**: Modify training parameters in `configs/` or override via command line:

```bash
python torch_nerf/runners/train.py data.scene=lego train.batch_size=1024 train.lr=5e-4
```

### Rendering

Render novel views from a trained model:

```bash
# Render spiral path
python torch_nerf/runners/render.py +log_dir=outputs/<experiment_dir> +render_test_views=False

# Render test set views
python torch_nerf/runners/render.py +log_dir=outputs/<experiment_dir> +render_test_views=True
```

### Evaluation

Compute quantitative metrics on the test set:

```bash
python torch_nerf/runners/evaluate.py <rendered_test_dir> data/nerf_synthetic/lego/test
```

This outputs PSNR, SSIM, and LPIPS scores comparing rendered images to ground truth.

## Implementation Details

Key technical components implemented:

1. **MLP Architecture**: 8-layer fully-connected network with skip connection at layer 4
2. **Positional Encoding**: Frequency encoding with L=10 for positions, L=4 for directions
3. **Stratified Sampling**: 64 coarse samples + 128 fine samples per ray with importance sampling
4. **Volume Rendering**: Numerical quadrature using alpha compositing
5. **Hierarchical Sampling**: Two-stage coarse-to-fine sampling strategy

## Extensions & Future Work

- **Speed Optimization**: Integrate instant-NGP hash encoding for faster training
- **Unbounded Scenes**: Adapt for large-scale outdoor scenes (mip-NeRF 360)
- **Dynamic Scenes**: Extend to time-varying content (D-NeRF)
- **Custom Data**: Support COLMAP reconstructions for in-the-wild captures
- **Compression**: Explore baking to explicit representations for real-time rendering

## References

- Mildenhall et al., "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis," ECCV 2020
- [Original NeRF Paper](https://arxiv.org/abs/2003.08934)
- [NeRF Project Page](https://www.matthewtancik.com/nerf)

## Acknowledgments

This implementation was developed as part of exploring 3D machine learning and neural rendering techniques. The codebase structure follows modern research practices with modular components and reproducible experiment management.

## License

MIT License - feel free to use this code for research and educational purposes.
