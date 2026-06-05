import os
import random
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate
import torchvision.transforms as transforms

from data import create_dataset
from models_2 import DRMSDFModel

from image_encoder_model import build_image_encoder_from_args


DEFAULT_IMAGE_KEY = "person"


class ImageConditionedSDFDataset(Dataset):
    """Thin wrapper that validates the conditioning image key."""

    def __init__(self, base_dataset, image_key=DEFAULT_IMAGE_KEY):
        self.base_dataset = base_dataset
        self.image_key = image_key

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        try:
            sample = self.base_dataset[idx]
        except FileNotFoundError as exc:
            print(f"[skip sample] index={idx}: missing file: {exc}")
            return None
        except OSError as exc:
            print(f"[skip sample] index={idx}: failed to load file: {exc}")
            return None

        image = sample.get(self.image_key, None)
        if not isinstance(image, torch.Tensor):
            print(
                f'Dataset sample is missing tensor image key "{self.image_key}". '
                f"Available keys: {sorted(sample.keys())}"
            )
            return None
        sample["conditioning_image"] = image
        return sample


def safe_collate(batch):
    valid = [item for item in batch if item is not None]
    if not valid:
        return None
    return default_collate(valid)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(gpu_id):
    if torch.cuda.is_available() and gpu_id >= 0:
        return torch.device(f"cuda:{gpu_id}")
    return torch.device("cpu")


def build_dataset_opt(args, is_train=True, batch_size=None, serial_batches=False, num_workers=None):
    opt = SimpleNamespace()
    opt.dataroot = args.dataroot
    opt.datalist = args.datalist
    opt.datamode = args.datamode
    opt.model = args.dataset_model
    opt.batch_size = args.batch_size if batch_size is None else batch_size
    opt.img_width = args.img_width
    opt.img_height = args.img_height
    opt.isTrain = is_train
    opt.max_dataset_size = float("inf")
    opt.num_threads = args.num_workers if num_workers is None else num_workers
    opt.serial_batches = serial_batches
    opt.no_pin_memory = False
    opt.radius = args.radius
    opt.warproot = args.warproot
    return opt


def create_image_sdf_dataset(args):
    dataset_loader = create_dataset(build_dataset_opt(args, is_train=True))
    base_dataset = dataset_loader.dataset
    return ImageConditionedSDFDataset(base_dataset, image_key=args.image_key), base_dataset


def build_drm_from_args(args):
    return DRMSDFModel(
        latent_dim=args.latent_dim,
        point_dim=3,
        hidden_dim=args.sdf_hidden_dim,
        num_layers=args.sdf_num_layers,
        pe_L=args.pe_L,
    )


def build_models_from_args(args, device):
    encoder = build_image_encoder_from_args(args).to(device)
    drm = build_drm_from_args(args).to(device)
    return encoder, drm


def ensure_shape_sdf(points, sdf_gt):
    if points is None or sdf_gt is None:
        return None, None
    if points.dim() == 2:
        points = points.unsqueeze(0)
    if sdf_gt.dim() == 1:
        sdf_gt = sdf_gt.view(1, -1, 1)
    elif sdf_gt.dim() == 2:
        sdf_gt = sdf_gt.unsqueeze(-1)
    return points, sdf_gt


def predict_with_grad(model, latent_z, points):
    points_req = points.clone().detach().requires_grad_(True)
    sdf_pred = model(latent_z, points_req)
    grads = torch.autograd.grad(
        outputs=sdf_pred.sum(),
        inputs=points_req,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    return sdf_pred, grads


def optional_tensor_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    return value


def prepare_sdf_batch(batch, device):
    if batch is None:
        return None

    points = batch.get("sdf_points", None)
    sdf_gt = batch.get("sdf_gt", None)
    points, sdf_gt = ensure_shape_sdf(points, sdf_gt)

    if points is None or sdf_gt is None or points.size(1) == 0:
        return None

    images = batch.get("conditioning_image", None)
    if not isinstance(images, torch.Tensor):
        raise RuntimeError('Batch is missing tensor key "conditioning_image".')

    points = points.to(device)
    sdf_gt = sdf_gt.to(device)
    images = images.to(device)

    surface_points = optional_tensor_to_device(batch.get("surface_points", None), device)
    surface_normals = optional_tensor_to_device(batch.get("surface_normals", None), device)

    sdf_scale = batch.get("sdf_scale", None)
    if sdf_scale is None:
        sdf_scale = torch.tensor(1.0, dtype=torch.float32, device=device)
    else:
        sdf_scale = sdf_scale.to(device)
        if sdf_scale.dim() == 0:
            sdf_scale = sdf_scale.unsqueeze(0).expand(points.size(0))
        elif sdf_scale.dim() == 1 and sdf_scale.size(0) == 1:
            sdf_scale = sdf_scale.expand(points.size(0))

    return {
        "images": images,
        "points": points,
        "sdf_gt": sdf_gt,
        "surface_points": surface_points,
        "surface_normals": surface_normals,
        "sdf_scale": sdf_scale,
    }


def compute_drm_sdf_losses(drm, latent_z, batch, args):
    points = batch["points"]
    sdf_gt = batch["sdf_gt"]
    surface_points = batch["surface_points"]
    surface_normals = batch["surface_normals"]

    sdf_pred = drm(latent_z, points)
    sign_labels = torch.where(sdf_gt >= 0, torch.ones_like(sdf_gt), -torch.ones_like(sdf_gt))
    total_loss = None

    if args.lambda_coarse > 0:
        loss_coarse = F.mse_loss(sdf_pred, sdf_gt)
        total_loss = args.lambda_coarse * loss_coarse
    else:
        loss_coarse = torch.zeros(1, device=points.device, dtype=sdf_pred.dtype)

    if args.lambda_surface > 0 and isinstance(surface_points, torch.Tensor) and surface_points.numel() > 0:
        surface_pred = drm(latent_z, surface_points)
        loss_surface = surface_pred.abs().mean()
        total_loss = args.lambda_surface * loss_surface if total_loss is None else total_loss + args.lambda_surface * loss_surface
    else:
        loss_surface = torch.zeros(1, device=points.device, dtype=sdf_pred.dtype)

    if args.lambda_sign > 0:
        loss_sign = torch.relu(-sign_labels * sdf_pred).mean()
        total_loss = args.lambda_sign * loss_sign if total_loss is None else total_loss + args.lambda_sign * loss_sign
    else:
        loss_sign = torch.zeros(1, device=points.device, dtype=sdf_pred.dtype)

    if args.lambda_eikonal > 0:
        _, grads = predict_with_grad(drm, latent_z, points)
        grad_norm = torch.linalg.norm(grads, dim=-1)
        target = 1.0
        loss_eikonal = ((grad_norm - target) ** 2).mean()
        total_loss = args.lambda_eikonal * loss_eikonal if total_loss is None else total_loss + args.lambda_eikonal * loss_eikonal
    else:
        loss_eikonal = torch.zeros(1, device=points.device, dtype=sdf_pred.dtype)

    has_normals = (
        isinstance(surface_points, torch.Tensor)
        and isinstance(surface_normals, torch.Tensor)
        and surface_points.numel() > 0
        and surface_normals.numel() > 0
    )
    if args.lambda_normal > 0 and has_normals:
        _, surface_grads = predict_with_grad(drm, latent_z, surface_points)
        n_pred = F.normalize(surface_grads, p=2, dim=-1, eps=1e-8)
        n_gt = F.normalize(surface_normals, p=2, dim=-1, eps=1e-8)
        loss_normal = (1.0 - (n_pred * n_gt).sum(dim=-1)).mean()
        total_loss = args.lambda_normal * loss_normal if total_loss is None else total_loss + args.lambda_normal * loss_normal
    else:
        loss_normal = torch.zeros(1, device=points.device, dtype=sdf_pred.dtype)

    if total_loss is None:
        total_loss = sdf_pred.sum() * 0.0

    return total_loss, {
        "coarse": loss_coarse,
        "surface": loss_surface,
        "sign": loss_sign,
        "eikonal": loss_eikonal,
        "normal": loss_normal,
    }


def build_image_conditioned_checkpoint(
    encoder,
    drm,
    optimizer,
    epoch,
    global_step,
    best_loss,
    config,
    sample_names=None,
):
    checkpoint = {
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_loss": float(best_loss),
        "encoder_state": encoder.state_dict(),
        "drm_state": drm.state_dict(),
        "config": dict(config),
    }
    if optimizer is not None:
        checkpoint["optimizer_state"] = optimizer.state_dict()
    if sample_names is not None:
        checkpoint["sample_names"] = list(sample_names)
    return checkpoint


def load_training_checkpoint(path, encoder, drm, optimizer, device, load_optimizer=True):
    checkpoint = torch.load(path, map_location="cpu")
    encoder.load_state_dict(checkpoint["encoder_state"])
    drm.load_state_dict(checkpoint["drm_state"])
    encoder.to(device)
    drm.to(device)

    if load_optimizer and optimizer is not None and "optimizer_state" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)

    return checkpoint


def image_transform(image_width, image_height, image_channels=3):
    return transforms.Compose(
        [
            transforms.Resize((int(image_height), int(image_width)), interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )


def load_conditioning_image(image_path, image_width, image_height, image_channels=3):
    if image_channels == 1:
        image = Image.open(image_path).convert("L")
    else:
        image = Image.open(image_path).convert("RGB")
    return image_transform(image_width, image_height, image_channels=image_channels)(image)


def bounds_from_npz(npz_path, padding=0.10):
    if not npz_path:
        return None
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"Bounds npz not found: {npz_path}")

    data = np.load(npz_path)
    pts = None
    if "surface_points" in data and data["surface_points"].size > 0:
        pts = data["surface_points"]
    elif "points" in data and data["points"].size > 0:
        pts = data["points"]

    if pts is None or pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] == 0:
        raise ValueError(f"Could not infer 3D bounds from: {npz_path}")

    mins = pts.min(axis=0) - float(padding)
    maxs = pts.max(axis=0) + float(padding)
    return (
        (float(mins[0]), float(maxs[0])),
        (float(mins[1]), float(maxs[1])),
        (float(mins[2]), float(maxs[2])),
    )

