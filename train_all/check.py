import torch
import pprint
import sys

ckpt_path = sys.argv[1]

ckpt = torch.load(ckpt_path, map_location="cpu")

print("=" * 80)
print("TOP LEVEL KEYS")
print("=" * 80)

if isinstance(ckpt, dict):
    print(list(ckpt.keys()))
else:
    print(type(ckpt))

print()

# ------------------------------------------------------------------
# Saved config
# ------------------------------------------------------------------

if isinstance(ckpt, dict) and "config" in ckpt:
    print("=" * 80)
    print("CONFIG")
    print("=" * 80)

    pprint.pprint(ckpt["config"])

    print("\nLikely reconstruction values:\n")

    for key in [
        "latent_dim",
        "sdf_hidden_dim",
        "sdf_num_layers",
        "pe_L",
        "image_in_channels",
        "image_feature_dim",
    ]:
        if key in ckpt["config"]:
            print(f"{key}: {ckpt['config'][key]}")

print()

# ------------------------------------------------------------------
# State dict inspection
# ------------------------------------------------------------------

state = None

for key in [
    "image_fusion_state",
    "fusion_state",
    "model_state",
    "coarse_state",
]:
    if isinstance(ckpt, dict) and key in ckpt:
        state = ckpt[key]
        print(f"Using state dict: {key}")
        break

if state is not None:

    print("\n")
    print("=" * 80)
    print("FIRST 40 PARAMETERS")
    print("=" * 80)

    for i, (k, v) in enumerate(state.items()):
        print(f"{k:70s} {tuple(v.shape)}")

        if i >= 39:
            break

print()

# ------------------------------------------------------------------
# Sample names
# ------------------------------------------------------------------

if isinstance(ckpt, dict) and "sample_names" in ckpt:
    print("=" * 80)
    print("SAMPLE NAMES")
    print("=" * 80)
    print("num samples:", len(ckpt["sample_names"]))

print()

# ------------------------------------------------------------------
# Training metadata
# ------------------------------------------------------------------

if isinstance(ckpt, dict):

    if "epoch" in ckpt:
        print("epoch:", ckpt["epoch"])

    if "global_step" in ckpt:
        print("global_step:", ckpt["global_step"])