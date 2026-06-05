import torch
import torch.nn as nn


class ConvImageEncoder(nn.Module):
    """Small convolutional image encoder that predicts one latent vector."""

    def __init__(
        self,
        latent_dim=128,
        in_channels=3,
        base_channels=32,
        num_blocks=5,
        head_hidden_dim=512,
        dropout=0.0,
        use_batchnorm=True,
    ):
        super().__init__()
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if base_channels <= 0:
            raise ValueError("base_channels must be positive")
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")

        layers = []
        current_channels = int(in_channels)
        next_channels = int(base_channels)

        for _ in range(int(num_blocks)):
            block = [
                nn.Conv2d(current_channels, next_channels, kernel_size=4, stride=2, padding=1),
            ]
            if use_batchnorm:
                block.append(nn.BatchNorm2d(next_channels))
            block.append(nn.LeakyReLU(0.2, inplace=True))
            if dropout > 0:
                block.append(nn.Dropout2d(float(dropout)))
            layers.append(nn.Sequential(*block))

            current_channels = next_channels
            next_channels = min(next_channels * 2, int(base_channels) * 8)

        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)

        if head_hidden_dim and head_hidden_dim > 0:
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(current_channels, int(head_hidden_dim)),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Linear(int(head_hidden_dim), int(latent_dim)),
            )
        else:
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(current_channels, int(latent_dim)),
            )

        self.latent_dim = int(latent_dim)
        self.in_channels = int(in_channels)

    def forward(self, image):
        if image.dim() != 4:
            raise ValueError("image must have shape [B, C, H, W]")
        features = self.features(image)
        pooled = self.pool(features)
        return self.head(pooled)


def build_image_encoder_from_args(args):
    return ConvImageEncoder(
        latent_dim=args.latent_dim,
        in_channels=args.image_channels,
        base_channels=args.encoder_base_channels,
        num_blocks=args.encoder_num_blocks,
        head_hidden_dim=args.encoder_head_hidden_dim,
        dropout=args.encoder_dropout,
        use_batchnorm=bool(args.encoder_use_batchnorm),
    )

