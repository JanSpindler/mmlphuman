#!/bin/bash

pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu128
export TORCH_CUDA_ARCH_LIST="8.0"
pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git
pip install --no-build-isolation git+https://github.com/facebookresearch/pytorch3d.git
pip install imageio numba omegaconf open3d opencv-python scipy smplx scikit-image tensorboardx tensorboard \
    trimesh websockets torchmetrics websocket-client dearpygui plyfile torch_pca lpips pykdtree torchmetrics[image]
