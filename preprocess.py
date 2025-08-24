import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from chestray_labels import LABELS, NUM_CLASSES

# ---------- path resolution ----------
def _ensure_rgb(img: Image.Image) -> Image.Image:
    return img.convert('RGB') if img.mode != 'RGB' else img

def _resolve_path(base_root: str, rel_or_abs: str) -> str:
    if os.path.isabs(rel_or_abs):
        return rel_or_abs
    head = rel_or_abs.split('/', 1)[0]
    if head in ('CheXpert-v1.0-small', 'ChestRay-v1.0-small'):
        parent = os.path.abspath(os.path.join(base_root, os.pardir))
        return os.path.join(parent, rel_or_abs)
    return os.path.join(base_root, rel_or_abs)

# ---------- dataset ----------
class ChestRayDataset(Dataset):
    def __init__(self, data_root, csv_name='train.csv', split='train',
                 policy='ignore', img_size=512, frontal_only=True, verbose=False):
        self.data_root = data_root
        self.csv_path = os.path.join(data_root, csv_name)
        self.split = split
        self.policy = policy
        self.img_size = img_size
        self.frontal_only = frontal_only

        if not os.path.isfile(self.csv_path):
            raise FileNotFoundError(f'CSV not found: {self.csv_path}')

        df = pd.read_csv(self.csv_path)

        if self.frontal_only and 'Frontal/Lateral' in df.columns:
            df = df[df['Frontal/Lateral'] == 'Frontal'].reset_index(drop=True)

        rel_paths = df['Path'].astype(str).tolist()
        abs_paths = [_resolve_path(self.data_root, p) for p in rel_paths]
        exists = [os.path.isfile(p) for p in abs_paths]
        self.df = df.loc[np.where(exists)[0]].reset_index(drop=True)
        self.abs_paths = [p for p, ok in zip(abs_paths, exists) if ok]

        for col in LABELS:
            if col not in self.df.columns:
                self.df[col] = np.nan

        targets = self.df[LABELS].astype('float32').values
        mask = np.ones_like(targets, dtype='float32')

        if policy == 'zeros':
            targets = np.nan_to_num(targets, nan=0.0)
            targets[targets == -1.0] = 0.0
        elif policy == 'ones':
            targets = np.nan_to_num(targets, nan=0.0)
            targets[targets == -1.0] = 1.0
        elif policy == 'ignore':
            ign = np.isnan(targets) | (targets == -1.0)
            mask[ign] = 0.0
            targets = np.nan_to_num(targets, nan=0.0)
            targets[targets == -1.0] = 0.0
        else:
            raise ValueError("policy must be one of {'ignore','zeros','ones'}")

        self.targets = targets.astype('float32')
        self.mask = mask.astype('float32')

        norm = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        if split == 'train':
            self.tfm = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.RandomRotation(5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                norm
            ])
        else:
            self.tfm = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                norm
            ])

        if verbose:
            print(f'[{split}] rows={len(df)} -> kept={len(self.abs_paths)}')

    def __len__(self): return len(self.abs_paths)

    def __getitem__(self, idx):
        path = self.abs_paths[idx]
        img = _ensure_rgb(Image.open(path))
        img = self.tfm(img)
        y = torch.from_numpy(self.targets[idx])
        m = torch.from_numpy(self.mask[idx])
        return img, y, m, path

# ---------- loaders ----------
def build_loaders(data_root, img_size=320, batch_size=32, num_workers=4,
                  policy='ignore', frontal_only=True):
    train_ds = ChestRayDataset(data_root, 'train.csv', 'train', policy, img_size, frontal_only)
    val_ds   = ChestRayDataset(data_root, 'valid.csv', 'valid', policy, img_size, frontal_only)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, train_ds, val_ds

# ---------- class weights ----------
def compute_pos_weight(loader, device):
    pos = torch.zeros(NUM_CLASSES, dtype=torch.float64)
    neg = torch.zeros(NUM_CLASSES, dtype=torch.float64)
    for _, y, m, _ in tqdm(loader, desc="Computing pos_weight"):
        y = y.double(); m = m.double()
        pos += ((y == 1.0) * m).sum(dim=0)
        neg += ((y == 0.0) * m).sum(dim=0)
    pos = torch.clamp(pos, min=1.0); neg = torch.clamp(neg, min=1.0)
    return (neg / pos).to(torch.float32).to(device)
