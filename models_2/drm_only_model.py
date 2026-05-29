import torch
import torch.nn as nn

def positional_encoding(x, L=6):
    out = [x]
    for i in range(L):
        out.append(torch.sin((2 ** i) * torch.pi * x))
        out.append(torch.cos((2 ** i) * torch.pi * x))
    return torch.cat(out, dim=-1)


class DRMSDFModel(nn.Module):
    """Positional-encoding SDF MLP with a single skip connection."""

    def __init__(self, latent_dim=128, point_dim=3, hidden_dim=512, num_layers=8, pe_L=6):
        super().__init__()
        self.latent_dim = latent_dim
        self.point_dim = point_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pe_L = pe_L

        encoded_point_dim = point_dim * (1 + 2 * pe_L)
        input_dim = latent_dim + encoded_point_dim

        self.skip_after = 3
        self.layers = nn.ModuleList()

        current_dim = input_dim
        for layer_index in range(max(num_layers - 1, 1)):
            if layer_index == self.skip_after:
                current_dim += input_dim
            self.layers.append(nn.Linear(current_dim, hidden_dim))
            current_dim = hidden_dim

        self.output_layer = nn.Linear(current_dim, 1)

    def forward(self, latent_z, points):
        if latent_z.dim() == 2:
            latent_z = latent_z.unsqueeze(1)
        if points.dim() == 2:
            points = points.unsqueeze(1)

        if latent_z.size(1) == 1 and points.size(1) > 1:
            latent_z = latent_z.expand(-1, points.size(1), -1)
        elif points.size(1) == 1 and latent_z.size(1) > 1:
            points = points.expand(-1, latent_z.size(1), -1)
        elif latent_z.size(1) != points.size(1):
            raise ValueError('latent_z and points must share the same sample dimension')

        encoded_points = positional_encoding(points, self.pe_L)
        sdf_input = torch.cat([latent_z, encoded_points], dim=-1)
        skip_input = sdf_input

        x = sdf_input.reshape(-1, sdf_input.size(-1))
        skip_flat = skip_input.reshape(-1, skip_input.size(-1))

        for layer_index, layer in enumerate(self.layers):
            if layer_index == self.skip_after:
                x = torch.cat([x, skip_flat], dim=-1)
            x = torch.relu(layer(x))

        sdf = self.output_layer(x)
        return sdf.view(points.size(0), points.size(1), -1)


class LatentCodebook(nn.Module):
    """Per-sample latent vectors optimized jointly with the SDF network."""

    def __init__(self, num_embeddings, latent_dim, init_std=0.02):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, latent_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=init_std)

    def forward(self, sample_indices):
        return self.embedding(sample_indices)


def build_checkpoint(model, latent_codebook, epoch, global_step, config, sample_names):
    return {
        'epoch': int(epoch),
        'global_step': int(global_step),
        'model_state': model.state_dict(),
        'latent_state': latent_codebook.state_dict(),
        'config': dict(config),
        'sample_names': list(sample_names),
    }
