from pathlib import Path
import random
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset

def list_pairs(root_dirs):
    roots = [Path(r) for r in root_dirs]
    images = []
    for r in roots:
        for p in sorted(r.glob("*.img.nii.gz")):
            lbl = p.with_name(p.name.replace(".img.", ".label."))
            if lbl.exists():
                images.append((p, lbl))
    return images

def load_nii(path):
    img = nib.load(str(path))
    arr = img.get_fdata(dtype=np.float32)
    return arr  # already normalized 0–1 by your pipeline


def _pad_to_shape(vol, target_shape):
    dz, dy, dx = target_shape
    Z, Y, X = vol.shape
    pad = [(0, max(0, dz - Z)), (0, max(0, dy - Y)), (0, max(0, dx - X))]
    if pad[0][1] or pad[1][1] or pad[2][1]:
        vol = np.pad(vol, pad, mode="constant", constant_values=0)
    return vol

def random_crop_3d(img, msk, size, pos_ratio=0.5, tries=16):
    dz, dy, dx = size
    Z, Y, X = img.shape
    needs_pos = (random.random() < pos_ratio) and (msk.sum() > 0)

    for _ in range(tries):
        z0 = random.randint(0, max(0, Z - dz))
        y0 = random.randint(0, max(0, Y - dy))
        x0 = random.randint(0, max(0, X - dx))
        crop_i = img[z0:z0+dz, y0:y0+dy, x0:x0+dx]
        crop_m = msk[z0:z0+dz, y0:y0+dy, x0:x0+dx]
        if (not needs_pos) or crop_m.any():
            # pad to exact patch size if near borders or the volume is smaller
            return _pad_to_shape(crop_i, size), _pad_to_shape(crop_m, size)

    # fallback random + pad
    z0 = random.randint(0, max(0, Z - dz))
    y0 = random.randint(0, max(0, Y - dy))
    x0 = random.randint(0, max(0, X - dx))
    crop_i = img[z0:z0+dz, y0:y0+dy, x0:x0+dx]
    crop_m = msk[z0:z0+dz, y0:y0+dy, x0:x0+dx]
    return _pad_to_shape(crop_i, size), _pad_to_shape(crop_m, size)

class CoronarySegDataset(Dataset):
    def __init__(self, pairs, patch_size=(160,160,160), pos_ratio=0.6, train=True):
        self.pairs = pairs
        self.patch = patch_size
        self.pos_ratio = pos_ratio
        self.train = train

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, msk_path = self.pairs[idx]
        img = load_nii(img_path)
        msk = nib.load(str(msk_path)).get_fdata(dtype=np.float32)
        msk = (msk > 0.5).astype(np.float32)

        if self.train:
            vol, gt = random_crop_3d(img, msk, self.patch, self.pos_ratio)
        else:
            # center crop/pad to patch size for validation quick pass
            Z, Y, X = img.shape
            dz, dy, dx = self.patch
            z0 = max(0, (Z - dz)//2); y0 = max(0, (Y - dy)//2); x0 = max(0, (X - dx)//2)
            z1 = min(Z, z0 + dz); y1 = min(Y, y0 + dy); x1 = min(X, x0 + dx)
            vol = img[z0:z1, y0:y1, x0:x1]
            gt  = msk[z0:z1, y0:y1, x0:x1]
            # pad if needed
            pad = [(0, max(0, dz - vol.shape[0])), (0, max(0, dy - vol.shape[1])), (0, max(0, dx - vol.shape[2]))]
            vol = np.pad(vol, pad, mode='constant')
            gt  = np.pad(gt,  pad, mode='constant')

        vol = torch.from_numpy(vol[None, ...])  # (1, D, H, W)
        gt  = torch.from_numpy(gt[None, ...])   # (1, D, H, W)
        return vol.float(), gt.float()

def split_pairs(pairs, val_frac=0.15, test_frac=0.15, seed=42):
    rng = random.Random(seed)
    shuffled = pairs[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_test = int(n * test_frac)
    n_val  = int(n * val_frac)
    test = shuffled[:n_test]
    val  = shuffled[n_test:n_test+n_val]
    train = shuffled[n_test+n_val:]
    return train, val, test
