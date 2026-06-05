import os
import random
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import Dataset

from data import create_dataset
from models import networks
from models_2 import DRMSDFModel
from tools_2_image_encoder.common import safe_collate
from tools_2_image_encoder.image_encoder_model import build_image_encoder_from_args


class SafeMTMDataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

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

        for key in ("agnostic", "cloth", "person", "sdf_points", "sdf_gt"):
            value = sample.get(key, None)
            if not isinstance(value, torch.Tensor):
                print(f"[skip sample] index={idx}: missing tensor key {key}")
                return None
        sample["conditioning_image"] = sample["person"]
        return sample


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


def build_dataset_opt(args, is_train=True, serial_batches=False):
    opt = SimpleNamespace()
    opt.dataroot = args.dataroot
    opt.datalist = args.datalist
    opt.datamode = args.datamode
    opt.model = "MTM"
    opt.batch_size = args.batch_size
    opt.img_width = args.img_width
    opt.img_height = args.img_height
    opt.isTrain = is_train
    opt.max_dataset_size = float("inf")
    opt.num_threads = args.num_workers
    opt.serial_batches = serial_batches
    opt.no_pin_memory = False
    opt.radius = args.radius
    opt.warproot = ""
    return opt


def create_mtm_dataset(args, is_train=True, serial_batches=False):
    dataset_loader = create_dataset(build_dataset_opt(args, is_train=is_train, serial_batches=serial_batches))
    return SafeMTMDataset(dataset_loader.dataset), dataset_loader.dataset


def runtime_from_image_drm_config(config):
    return SimpleNamespace(
        latent_dim=int(config.get("latent_dim", 128)),
        sdf_hidden_dim=int(config.get("sdf_hidden_dim", 512)),
        sdf_num_layers=int(config.get("sdf_num_layers", 8)),
        pe_L=int(config.get("pe_L", 6)),
        image_channels=int(config.get("image_channels", 3)),
        encoder_base_channels=int(config.get("encoder_base_channels", 32)),
        encoder_num_blocks=int(config.get("encoder_num_blocks", 5)),
        encoder_head_hidden_dim=int(config.get("encoder_head_hidden_dim", 512)),
        encoder_dropout=float(config.get("encoder_dropout", 0.0)),
        encoder_use_batchnorm=int(config.get("encoder_use_batchnorm", 1)),
    )


def build_mtm(args, latent_dim, device):
    return networks.define_MTM(
        input_nc_A=args.mtm_input_nc_A,
        input_nc_B=args.mtm_input_nc_B,
        ngf=args.mtm_ngf,
        n_layers=args.mtm_n_layers_feat_extract,
        img_height=args.img_height,
        img_width=args.img_width,
        grid_size=args.mtm_grid_size,
        add_tps=bool(args.mtm_add_tps),
        add_depth=bool(args.mtm_add_depth),
        add_segmt=bool(args.mtm_add_segmt),
        latent_dim=latent_dim,
        norm=args.mtm_norm,
        use_dropout=bool(args.mtm_use_dropout),
        init_type=args.mtm_init_type,
        init_gain=args.mtm_init_gain,
        gpu_ids=[device.index] if device.type == "cuda" else [],
    )


def load_mtm_pretrained(mtm, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        state = checkpoint.get("mtm_state") or checkpoint.get("model_state") or checkpoint
    else:
        state = checkpoint
    if hasattr(state, "_metadata"):
        del state._metadata
    result = mtm.load_state_dict(state, strict=False)
    mtm.to(device)
    if hasattr(result, "missing_keys") and result.missing_keys:
        print("MTM missing keys:", result.missing_keys[:30])
    if hasattr(result, "unexpected_keys") and result.unexpected_keys:
        print("MTM unexpected keys:", result.unexpected_keys[:30])
    return checkpoint, result


def load_image_drm_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {})
    runtime = runtime_from_image_drm_config(config)

    encoder = build_image_encoder_from_args(runtime).to(device)
    drm = DRMSDFModel(
        latent_dim=runtime.latent_dim,
        point_dim=3,
        hidden_dim=runtime.sdf_hidden_dim,
        num_layers=runtime.sdf_num_layers,
        pe_L=runtime.pe_L,
    ).to(device)

    encoder.load_state_dict(checkpoint["encoder_state"])
    drm_state = checkpoint.get("drm_state", checkpoint.get("model_state"))
    if drm_state is None:
        raise RuntimeError(f"{checkpoint_path} is missing drm_state/model_state")
    drm.load_state_dict(drm_state)
    return checkpoint, runtime, encoder, drm




def copy_drm_state_to_new_latent_dim(target_drm, source_state, old_latent_dim, new_latent_dim):
    """Copy a DRM checkpoint into a DRM whose latent input is larger.

    The DRM input order is [latent, positional_encoded_point].  For concat
    fusion the new order is [image_latent, mtm_latent, positional_encoded_point].
    We preserve the old image-latent and point weights, while keeping the new
    MTM-latent columns at the target module's initialization.
    """
    if new_latent_dim < old_latent_dim:
        raise ValueError("new_latent_dim must be >= old_latent_dim")

    target_state = target_drm.state_dict()
    encoded_point_dim = target_drm.point_dim * (1 + 2 * target_drm.pe_L)
    old_input_dim = old_latent_dim + encoded_point_dim
    new_input_dim = new_latent_dim + encoded_point_dim
    copied_expanded = []

    def copy_input_columns(target, source, prefix_cols):
        result = target.clone()
        result[:, :prefix_cols] = source[:, :prefix_cols]
        src_input_start = prefix_cols
        tgt_input_start = prefix_cols
        result[:, tgt_input_start:tgt_input_start + old_latent_dim] = source[
            :, src_input_start:src_input_start + old_latent_dim
        ]
        result[:, tgt_input_start + new_latent_dim:tgt_input_start + new_input_dim] = source[
            :, src_input_start + old_latent_dim:src_input_start + old_input_dim
        ]
        return result

    for key, source_value in source_state.items():
        if key not in target_state:
            continue
        source_value = source_value.to(target_state[key].device)
        target_value = target_state[key]
        if source_value.shape == target_value.shape:
            target_state[key] = source_value
            continue
        if source_value.ndim == 2 and target_value.ndim == 2 and source_value.size(0) == target_value.size(0):
            if source_value.size(1) == old_input_dim and target_value.size(1) == new_input_dim:
                target_state[key] = copy_input_columns(target_value, source_value, prefix_cols=0)
                copied_expanded.append(key)
                continue
            hidden_dim = target_drm.hidden_dim
            if source_value.size(1) == hidden_dim + old_input_dim and target_value.size(1) == hidden_dim + new_input_dim:
                target_state[key] = copy_input_columns(target_value, source_value, prefix_cols=hidden_dim)
                copied_expanded.append(key)
                continue
        print(f"[DRM concat init] kept initialized target parameter for shape-mismatched key {key}: {tuple(source_value.shape)} -> {tuple(target_value.shape)}")

    target_drm.load_state_dict(target_state)
    if copied_expanded:
        print(f"Expanded DRM latent input {old_latent_dim} -> {new_latent_dim}; copied old weights for: {copied_expanded}")
    return target_drm


def build_concat_drm_from_loaded_drm(loaded_drm, runtime, device):
    concat_latent_dim = int(runtime.latent_dim) * 2
    concat_drm = DRMSDFModel(
        latent_dim=concat_latent_dim,
        point_dim=3,
        hidden_dim=runtime.sdf_hidden_dim,
        num_layers=runtime.sdf_num_layers,
        pe_L=runtime.pe_L,
    ).to(device)
    copy_drm_state_to_new_latent_dim(
        target_drm=concat_drm,
        source_state=loaded_drm.state_dict(),
        old_latent_dim=int(runtime.latent_dim),
        new_latent_dim=concat_latent_dim,
    )
    return concat_drm

def freeze_module(module):
    for param in module.parameters():
        param.requires_grad = False
    module.eval()


def train_only_mtm_z_projection(mtm):
    for param in mtm.parameters():
        param.requires_grad = False
    if not hasattr(mtm, "z_proj"):
        raise RuntimeError("MTM model does not expose z_proj.")
    for param in mtm.z_proj.parameters():
        param.requires_grad = True
    mtm.eval()
    mtm.z_proj.train()


def prepare_mtm_inputs(batch, device):
    if batch is None:
        return None
    agnostic = batch.get("agnostic", None)
    cloth = batch.get("cloth", None)
    if not isinstance(agnostic, torch.Tensor) or not isinstance(cloth, torch.Tensor):
        return None
    return agnostic.to(device), cloth.to(device)


def build_checkpoint(
    mtm,
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
        "mtm_state": mtm.state_dict(),
        "mtm_z_proj_state": mtm.z_proj.state_dict(),
        "encoder_state": encoder.state_dict(),
        "drm_state": drm.state_dict(),
        "config": dict(config),
    }
    if optimizer is not None:
        checkpoint["optimizer_state"] = optimizer.state_dict()
    if sample_names is not None:
        checkpoint["sample_names"] = list(sample_names)
    return checkpoint

