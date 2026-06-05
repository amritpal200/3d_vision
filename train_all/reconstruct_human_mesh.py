

# Only fusion layers trained
# CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 python3 train_all/reconstruct_human_mesh.py --checkpoint /data/125-1/users/asingh/proves/train_only_fusion_imgE/aligned/fusion_only_new_layers/epoch_14.pth --dataroot /data/113-1/users/asingh/project/3d/MPV3D --datalist test_pairs --sample_index 0 --output_obj mesh_results/fusion_only_new_layers.obj --resolution 128 --gpu_id 0 --save_point_cloud --latent_dim 128 --mtm_z_dim 1024 --sdf_hidden_dim 812 --sdf_num_layers 10 --pe_L 12 --image_in_channels 3 --image_feature_dim 256 --image_scale 1.0 --fusion_scale 1.0

# Coarse DRM also trained

#!/usr/bin/env python3

"""Reconstruct a single human mesh from a trained MTM + DRM fusion checkpoint.

This is a convenience entry point that reuses the image-conditioned fusion
reconstruction pipeline in `reconstruct_mtm_drm_fusion.py`.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from train_all.reconstruct_mtm_drm_fusion import main


if __name__ == '__main__':
    main()
