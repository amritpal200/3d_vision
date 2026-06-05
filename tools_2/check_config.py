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

print("\n")

if isinstance(ckpt, dict) and "config" in ckpt:
    print("=" * 80)
    print("CONFIG")
    print("=" * 80)
    pprint.pprint(ckpt["config"])
    print("\n")

if isinstance(ckpt, dict) and "epoch" in ckpt:
    print("epoch:", ckpt["epoch"])

if isinstance(ckpt, dict) and "global_step" in ckpt:
    print("global_step:", ckpt["global_step"])

print("\n")

# find model state
if isinstance(ckpt, dict):
    if "model_state" in ckpt:
        state = ckpt["model_state"]
        print("Using model_state")
    elif "coarse_state" in ckpt:
        state = ckpt["coarse_state"]
        print("Using coarse_state")
    else:
        state = ckpt
        print("Using checkpoint directly as state_dict")
else:
    state = ckpt

print("\n")
print("=" * 80)
print("FIRST 20 PARAMETERS")
print("=" * 80)

for i, (k, v) in enumerate(state.items()):
    print(f"{k:60s} {tuple(v.shape)}")
    if i > 20:
        break

print("\n")

print("=" * 80)
print("INFERRING ARCHITECTURE")
print("=" * 80)

for k, v in state.items():

    if "latent" in k.lower():
        print("latent-related:", k, tuple(v.shape))

    if "fc_in.weight" in k:
        print("fc_in.weight:", tuple(v.shape))

    if "fc_out.weight" in k:
        print("fc_out.weight:", tuple(v.shape))