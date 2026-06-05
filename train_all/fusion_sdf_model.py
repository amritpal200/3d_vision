import torch
import torch.nn as nn

from models_2 import DRMSDFModel


class SimpleImageEncoder(nn.Module):
    def __init__(self, in_channels=3, feature_dim=256):
        super().__init__()

        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool2d(1),
        )

        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, feature_dim),
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.proj(x)


class ImageConditionedFusionSDF(nn.Module):
    def __init__(
        self,
        latent_dim=128,
        mtm_z_dim=1024,
        sdf_hidden_dim=512,
        sdf_num_layers=8,
        pe_L=6,
        image_in_channels=3,
        image_feature_dim=256,
        image_scale=0.1,
        fusion_scale=1.0,
        residual_scale=1.0,
        drm_mode="residual",
        debug=False,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.mtm_z_dim = mtm_z_dim
        self.image_feature_dim = image_feature_dim
        self.image_scale = float(image_scale)
        self.fusion_scale = float(fusion_scale)
        self.residual_scale = float(residual_scale)
        self.drm_mode = drm_mode
        self.debug = bool(debug)

        # MTM z: 1024 -> 128
        self.z_proj = nn.Linear(mtm_z_dim, latent_dim)

        # Image encoder: person image only
        self.image_encoder = SimpleImageEncoder(
            in_channels=image_in_channels,
            feature_dim=image_feature_dim,
        )

        # Predict latent residual, not full latent from scratch
        self.fusion_mlp = nn.Sequential(
            nn.Linear(latent_dim + image_feature_dim, latent_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_dim, latent_dim),
        )

        self.coarse = DRMSDFModel(
            latent_dim=latent_dim,
            point_dim=3,
            hidden_dim=sdf_hidden_dim,
            num_layers=sdf_num_layers,
            pe_L=pe_L,
        )

        if drm_mode == "residual":
            self.residual = DRMSDFModel(
                latent_dim=latent_dim,
                point_dim=3,
                hidden_dim=sdf_hidden_dim,
                num_layers=sdf_num_layers,
                pe_L=pe_L,
            )
        else:
            self.residual = None

    def fuse_latent(self, latent_z, image_tensor):
        if latent_z.dim() == 3 and latent_z.size(1) == 1:
            latent_z = latent_z.squeeze(1)

        if latent_z.dim() != 2:
            raise ValueError(
                f"latent_z must have shape [B, D] or [B, 1, D], got {latent_z.shape}"
            )

        if latent_z.size(-1) != self.mtm_z_dim:
            raise ValueError(
                f"Expected latent_z dim {self.mtm_z_dim}, got {latent_z.size(-1)}. "
                "Check --mtm_z_dim and MTM latent_dim."
            )

        z_projected = self.z_proj(latent_z)

        image_feature = self.image_encoder(image_tensor)
        image_feature = self.image_scale * image_feature

        fusion_input = torch.cat([z_projected, image_feature], dim=-1)

        latent_delta = self.fusion_mlp(fusion_input)

        # safer than replacing z completely
        fused_z = z_projected + self.fusion_scale * latent_delta

        fused_z = fused_z.unsqueeze(1)

        return fused_z, z_projected, image_feature, latent_delta

    def forward(self, latent_z, image_tensor, points):
        fused_z, z_projected, image_feature, latent_delta = self.fuse_latent(
            latent_z,
            image_tensor,
        )

        coarse_sdf = self.coarse(fused_z, points)

        residual_sdf = None
        if self.drm_mode == "residual" and self.residual is not None:
            residual_sdf = self.residual(fused_z, points)
            final_sdf = coarse_sdf + self.residual_scale * residual_sdf
        else:
            final_sdf = coarse_sdf

        if (not self.training) and self.debug:
            print("z_projected mean:", z_projected.mean().item())
            print("z_projected std:", z_projected.std().item())
            print("image_feature mean:", image_feature.mean().item())
            print("image_feature std:", image_feature.std().item())
            print("latent_delta mean:", latent_delta.mean().item())
            print("latent_delta std:", latent_delta.std().item())
            print("fused_z mean:", fused_z.mean().item())
            print("fused_z std:", fused_z.std().item())
            print("coarse_sdf mean:", coarse_sdf.mean().item())
            if residual_sdf is not None:
                print("residual_sdf mean:", residual_sdf.mean().item())
            print("final_sdf mean:", final_sdf.mean().item())

        return {
            "final_sdf": final_sdf,
            "coarse_sdf": coarse_sdf,
            "residual_sdf": residual_sdf,
            "z_projected": z_projected,
            "fused_z": fused_z,
            "image_feature": image_feature,
            "latent_delta": latent_delta,
        }