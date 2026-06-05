import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset


class SDFDataset(Dataset):

    def __init__(self, root_dir, points_per_sample=None, seed=2026):
        self.files = sorted(glob.glob(os.path.join(root_dir, "*.npz")))
        self.sample_names = [os.path.splitext(os.path.basename(path))[0] for path in self.files]
        self.points_per_sample = points_per_sample
        self.seed = int(seed)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        data = np.load(self.files[idx])

        points = np.asarray(data["points"], dtype=np.float32)
        sdf = np.asarray(data["sdf"], dtype=np.float32)

        if sdf.ndim == 1:
            sdf = sdf[:, None]

        if self.points_per_sample is not None and points.shape[0] > self.points_per_sample:
            rng = np.random.default_rng(self.seed + idx)
            chosen = rng.choice(points.shape[0], size=self.points_per_sample, replace=False)
            points = points[chosen]
            sdf = sdf[chosen]

        return {
            "sdf_points": torch.from_numpy(points).float(),
            "sdf_gt": torch.from_numpy(sdf).float(),
            "sample_index": idx,
            "sample_name": self.sample_names[idx],
        }