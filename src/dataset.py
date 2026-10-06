import os
import random
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(img_size=256, is_train=True):
    """
    Returns image augmentation pipeline using Albumentations.
    """
    if is_train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.Affine(scale=(0.9, 1.1), rotate=(-15, 15), translate_percent=(-0.0625, 0.0625), p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])


class LungDataset(Dataset):
    """
    Standard Chest X-Ray lung segmentation dataset loader.
    Reads images from disk on each batch.
    """
    def __init__(self, csv_file, img_size=256, is_train=True):
        self.df = pd.read_csv(csv_file)
        self.transforms = get_transforms(img_size=img_size, is_train=is_train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row['image_path']
        mask_path = row['mask_path']

        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Failed to read image at: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Failed to read mask at: {mask_path}")
        mask = (mask > 127).astype(np.float32)

        augmented = self.transforms(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        return image, mask, row['filename']


# In-memory resized image cache for high-throughput GPU training
GLOBAL_CACHE = {}


def get_cached_item(image_path, mask_path, img_size=256):
    key = (image_path, mask_path, img_size)
    if key not in GLOBAL_CACHE:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Failed to read image at: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)

        msk = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if msk is None:
            raise FileNotFoundError(f"Failed to read mask at: {mask_path}")
        msk = cv2.resize(msk, (img_size, img_size), interpolation=cv2.INTER_NEAREST)
        msk = (msk > 127).astype(np.float32)

        GLOBAL_CACHE[key] = (img, msk)
    return GLOBAL_CACHE[key]


class FastLungDataset(Dataset):
    """
    High-throughput dataset using in-memory pre-resized 256x256 cache.
    Bypasses disk decompression bottlenecks on local machines.
    """
    def __init__(self, csv_file, img_size=256, is_train=True):
        self.df = pd.read_csv(csv_file)
        self.img_size = img_size
        self.transforms = get_transforms(img_size=img_size, is_train=is_train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_np, msk_np = get_cached_item(row['image_path'], row['mask_path'], self.img_size)

        augmented = self.transforms(image=img_np, mask=msk_np)
        image = augmented['image']
        mask = augmented['mask']
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)

        return image, mask, row['filename']


def get_loader(csv_file, img_size=256, batch_size=8, is_train=True, num_workers=2):
    """
    Instantiates standard DataLoader for training, validation, or testing.
    """
    ds = LungDataset(csv_file, img_size=img_size, is_train=is_train)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )


def get_fast_loader(csv_file, img_size=256, batch_size=8, is_train=True):
    """
    Instantiates cached high-throughput DataLoader (single-worker, pinned memory).
    """
    ds = FastLungDataset(csv_file, img_size=img_size, is_train=is_train)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )


def prepare_splits(data_dir='Lung Segmentation', output_dir='splits', test_ratio=0.20, seed=42):
    """
    Discovers image-mask pairs and constructs stratified nested dataset splits (10%, 25%, 50%, 100%).
    """
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)

    cxr_dir = os.path.join(data_dir, 'CXR_png')
    mask_dir = os.path.join(data_dir, 'masks')

    if not os.path.exists(cxr_dir) or not os.path.exists(mask_dir):
        print(f"[Warning] Directory {cxr_dir} or {mask_dir} not found. Skipping split generation.")
        return

    mask_files = set(os.listdir(mask_dir))
    pairs = []

    for f in os.listdir(cxr_dir):
        if not f.endswith('.png'):
            continue
        dataset_type = 'unknown'
        mask_name = None
        if f.startswith('MCUCXR') and f in mask_files:
            dataset_type = 'Montgomery'
            mask_name = f
        elif f.startswith('CHNCXR'):
            target_mask = f.replace('.png', '_mask.png')
            if target_mask in mask_files:
                dataset_type = 'Shenzhen'
                mask_name = target_mask

        if mask_name is not None:
            pairs.append({
                'filename': f,
                'dataset': dataset_type,
                'image_path': os.path.join(cxr_dir, f),
                'mask_path': os.path.join(mask_dir, mask_name)
            })

    mcu = [p for p in pairs if p['dataset'] == 'Montgomery']
    chn = [p for p in pairs if p['dataset'] == 'Shenzhen']
    random.shuffle(mcu)
    random.shuffle(chn)

    n_test_mcu = int(len(mcu) * test_ratio)
    n_test_chn = int(len(chn) * test_ratio)

    test_pool = mcu[:n_test_mcu] + chn[:n_test_chn]
    train_val_pool = mcu[n_test_mcu:] + chn[n_test_chn:]
    random.shuffle(test_pool)
    random.shuffle(train_val_pool)

    n_val = int(len(train_val_pool) * 0.15)
    val_pool = train_val_pool[:n_val]
    train_full = train_val_pool[n_val:]

    pd.DataFrame(test_pool).to_csv(os.path.join(output_dir, 'test_fixed_20pct.csv'), index=False)
    pd.DataFrame(val_pool).to_csv(os.path.join(output_dir, 'val_fixed.csv'), index=False)

    shuffled_train = list(train_full)
    random.shuffle(shuffled_train)

    ratios = {
        'train_10pct.csv': 0.10,
        'train_25pct.csv': 0.25,
        'train_50pct.csv': 0.50,
        'train_100pct.csv': 1.00
    }
    for filename, ratio in ratios.items():
        n = max(2, int(len(shuffled_train) * ratio))
        pd.DataFrame(shuffled_train[:n]).to_csv(os.path.join(output_dir, filename), index=False)

    print(f"Total valid pairs: {len(pairs)}")
    print(f"Test: {len(test_pool)} | Val: {len(val_pool)} | Train 100%: {len(train_full)}")
    for f, r in ratios.items():
        print(f"  -> {f}: {max(2, int(len(shuffled_train) * r))} samples")
