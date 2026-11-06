# Environment setup

Tested on: Ubuntu 26.04 (WSL2), RTX 5060 Laptop (Blackwell, sm_120, 8 GB), driver 592.82.

```bash
# conda-forge only (Anaconda's default channel requires registration)
conda create -y -n sr3d -c conda-forge --override-channels python=3.11
conda install -y -n sr3d -c conda-forge --override-channels \
    "cuda-nvcc=12.8" "cuda-cudart-dev=12.8" "cuda-cccl=12.8" "gcc=13" "gxx=13" ninja libusb
conda activate sr3d

# Blackwell GPUs need a CUDA >= 12.8 PyTorch build
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install gsplat transformers accelerate open3d huggingface_hub opencv-python-headless pillow gdown
SAM2_BUILD_CUDA=0 pip install --no-build-isolation "git+https://github.com/facebookresearch/sam2.git"
```

Add `$CONDA_PREFIX/etc/conda/activate.d/cuda_build.sh` so gsplat's JIT build finds the
conda-forge CUDA headers (they live under `targets/x86_64-linux/`, not `include/`):

```bash
export CUDA_HOME="$CONDA_PREFIX"
export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include${CPATH:+:$CPATH}"
export LIBRARY_PATH="$CONDA_PREFIX/targets/x86_64-linux/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
export TORCH_CUDA_ARCH_LIST="12.0"
export MAX_JOBS=8
```

Verify with `python scripts/check_env.py`. The first run compiles gsplat (~75 s), later runs use the cache.

| Component | Version | Check result |
|---|---|---|
| PyTorch | 2.11.0+cu128 | OK |
| gsplat | 1.5.3 | OK (JIT-compiled for sm_120) |
| Depth Anything V2 Small | transformers 5.21 | OK |
| SAM 2.1 hiera-small | 1.0 | OK, 0.61 GB peak VRAM |
| Open3D | 0.20.0 | OK |

## Data

| Dataset | Source | Access | License |
|---|---|---|---|
| StereoMIS (porcine) | Zenodo record 8154924 | Open | Non-commercial |
| EndoNeRF samples | Google Drive via github.com/med-air/EndoNeRF | Open | Non-commercial, no redistribution, cite Wang et al. MICCAI 2022 |
| EndoVis 2018 | grand-challenge.org | Registration | Challenge rules |
| SCARED | Challenge organizers | Request | Challenge rules |
| SERV-CT (ex-vivo porcine, CT reference) | figshare DOI 10.5522/04/26352199 (`SERV-CT.zip`, MD5 911048d7…) | Open | CC BY-NC-SA 4.0 per its readme (figshare page says CC BY 4.0; the stricter terms are followed), cite Edwards et al. MedIA 2022 |

Raw and converted data are git-ignored and must never be pushed: several licenses forbid redistribution.
Google Drive throttles scripted downloads (gdown "Cannot retrieve the public link"); download EndoNeRF in a browser.
