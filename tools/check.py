import torch

ckpt = torch.load(
    "/data/113-1/users/asingh/project/3d/checkpoints/together_pretrained/aligned/MTM_DRM_pretrained_run/best_net_MTM_DRM.pth",
    map_location="cpu"
)

print(ckpt["drm_state"]["layers.0.weight"].shape)
print(ckpt["drm_state"]["layers.3.weight"].shape)
print(ckpt["drm_state"]["output_layer.weight"].shape)